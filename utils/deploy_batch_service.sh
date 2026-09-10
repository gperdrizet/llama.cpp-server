#!/usr/bin/env bash
# utils/deploy_batch_service.sh
#
# Deploys one or both utils/llamacpp-batch.service instances (one per compute
# GPU) to /etc/systemd/system/llamacpp-batch<N>.service, for overnight/batch
# throughput jobs. Deployment performance flags (sampling, batch/ubatch size,
# etc.) are hardcoded in the template to the values verified best for batch
# throughput on this hardware - see utils/llamacpp-batch.service for the tuning
# rationale. Per-instance values (GPU, port, CPU affinity, model file,
# ctx-size, parallel) are substituted here - each instance can serve a
# different model with its own context/parallelism, set via
# BATCH1_MODEL_FILE / BATCH2_MODEL_FILE and BATCH1_CTX_SIZE / BATCH1_PARALLEL /
# BATCH2_CTX_SIZE / BATCH2_PARALLEL in .env.
#
# Reuses API_KEY and LLAMA_PATH/MODEL_DIR from the main .env (same secret,
# same repo paths as the production single-instance deployment).
#
# Usage:
#   ./utils/deploy_batch_service.sh <1|2|both> [--restart]
#
#   --restart   Also restart the service(s) after deploying (default: daemon-reload only)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TEMPLATE="$REPO_ROOT/utils/llamacpp-batch.service"
ENV_FILE="$REPO_ROOT/.env"
DO_RESTART=false

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
INSTANCE_ARG="${1:-}"
if [[ -z "$INSTANCE_ARG" ]]; then
    echo "Usage: $0 <1|2|both> [--restart]" >&2
    exit 1
fi
shift || true

for arg in "$@"; do
    case "$arg" in
        --restart) DO_RESTART=true ;;
        *) echo "Unknown argument: $arg" >&2; exit 1 ;;
    esac
done

case "$INSTANCE_ARG" in
    1) INSTANCES=(1) ;;
    2) INSTANCES=(2) ;;
    both) INSTANCES=(1 2) ;;
    *) echo "ERROR: instance must be 1, 2, or both (got: '$INSTANCE_ARG')" >&2; exit 1 ;;
esac

# ---------------------------------------------------------------------------
# Load .env (only need API_KEY, LLAMA_PATH, MODEL_DIR from it)
# ---------------------------------------------------------------------------
if [[ ! -f "$ENV_FILE" ]]; then
    echo "ERROR: $ENV_FILE not found." >&2
    echo "       Copy .env.template to .env and fill in your API key." >&2
    exit 1
fi

set -o allexport
# shellcheck disable=SC1090
source <(grep -E '^[A-Za-z_][A-Za-z0-9_]*=' "$ENV_FILE")
set +o allexport

if [[ -z "${API_KEY:-}" || "$API_KEY" == "your_api_key_here" ]]; then
    echo "ERROR: API_KEY is not set (or still a placeholder) in $ENV_FILE" >&2
    exit 1
fi

if [[ -z "${LLAMA_PATH:-}" || "$LLAMA_PATH" == "your_llama_path_here" ]]; then
    echo "ERROR: LLAMA_PATH is not set (or still a placeholder) in $ENV_FILE" >&2
    exit 1
fi

if [[ -z "${MODEL_DIR:-}" || "$MODEL_DIR" == "your_model_dir_here" ]]; then
    echo "ERROR: MODEL_DIR is not set (or still a placeholder) in $ENV_FILE" >&2
    exit 1
fi

if ! id -u llama &>/dev/null; then
    echo "ERROR: System user 'llama' does not exist." >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Per-instance settings: GPU index, CPU core range, port, model file
# ---------------------------------------------------------------------------
declare -A GPU_FOR=( [1]=1 [2]=2 )
declare -A CPU_AFFINITY_FOR=( [1]="0-11" [2]="12-23" )
declare -A PORT_FOR=( [1]=8503 [2]=8504 )
declare -A MODEL_FILE_FOR=( [1]="${BATCH1_MODEL_FILE:-}" [2]="${BATCH2_MODEL_FILE:-}" )
declare -A CTX_SIZE_FOR=( [1]="${BATCH1_CTX_SIZE:-}" [2]="${BATCH2_CTX_SIZE:-}" )
declare -A PARALLEL_FOR=( [1]="${BATCH1_PARALLEL:-}" [2]="${BATCH2_PARALLEL:-}" )

for N in "${INSTANCES[@]}"; do
    MODEL_FILE="${MODEL_FILE_FOR[$N]}"
    if [[ -z "$MODEL_FILE" || "$MODEL_FILE" == "model_file_here.gguf" ]]; then
        echo "ERROR: BATCH${N}_MODEL_FILE is not set (or still a placeholder) in $ENV_FILE" >&2
        exit 1
    fi

    MODEL_PATH="${MODEL_DIR%/}/${MODEL_FILE}"
    if [[ ! -f "$MODEL_PATH" ]]; then
        echo "ERROR: Model file not found: $MODEL_PATH" >&2
        exit 1
    fi

    if ! sudo -u llama test -r "$MODEL_PATH"; then
        echo "ERROR: User 'llama' cannot read model file: $MODEL_PATH" >&2
        exit 1
    fi

    if [[ -z "${CTX_SIZE_FOR[$N]}" || "${CTX_SIZE_FOR[$N]}" == "ctx_size_here" ]]; then
        echo "ERROR: BATCH${N}_CTX_SIZE is not set (or still a placeholder) in $ENV_FILE" >&2
        exit 1
    fi

    if [[ -z "${PARALLEL_FOR[$N]}" || "${PARALLEL_FOR[$N]}" == "parallel_here" ]]; then
        echo "ERROR: BATCH${N}_PARALLEL is not set (or still a placeholder) in $ENV_FILE" >&2
        exit 1
    fi
done

for N in "${INSTANCES[@]}"; do
    DEST="/etc/systemd/system/llamacpp-batch${N}.service"

    RENDERED="$(sed \
        -e "s/SUB_API_KEY_HERE/${API_KEY}/" \
        -e "s|SUB_LLAMA_PATH_HERE|${LLAMA_PATH}|g" \
        -e "s|SUB_MODEL_DIR_HERE|${MODEL_DIR}|g" \
        -e "s|SUB_MODEL_FILE_HERE|${MODEL_FILE_FOR[$N]}|g" \
        -e "s/SUB_CUDA_DEVICE_HERE/${GPU_FOR[$N]}/" \
        -e "s/SUB_CPU_AFFINITY_HERE/${CPU_AFFINITY_FOR[$N]}/" \
        -e "s/SUB_PORT_HERE/${PORT_FOR[$N]}/" \
        -e "s/SUB_CTX_SIZE_HERE/${CTX_SIZE_FOR[$N]}/" \
        -e "s/SUB_PARALLEL_HERE/${PARALLEL_FOR[$N]}/" \
        -e "s/SUB_INSTANCE_HERE/${N}/" \
        "$TEMPLATE")"

    echo "Deploying $TEMPLATE → $DEST (GPU ${GPU_FOR[$N]}, port ${PORT_FOR[$N]}, model ${MODEL_FILE_FOR[$N]}, ctx ${CTX_SIZE_FOR[$N]}, parallel ${PARALLEL_FOR[$N]})"
    echo "$RENDERED" | sudo tee "$DEST" > /dev/null
done

echo "Running: systemctl daemon-reload"
sudo systemctl daemon-reload

if [[ "$DO_RESTART" == true ]]; then
    for N in "${INSTANCES[@]}"; do
        echo "Running: systemctl restart llamacpp-batch${N}.service"
        sudo systemctl restart "llamacpp-batch${N}.service"
    done
    echo "Service(s) restarted. Status:"
    systemctl status "${INSTANCES[@]/#/llamacpp-batch}" --no-pager -l 2>/dev/null || \
        for N in "${INSTANCES[@]}"; do systemctl status "llamacpp-batch${N}.service" --no-pager -l; done
else
    echo ""
    echo "Unit file(s) deployed. Run the following to apply changes to the running service(s):"
    for N in "${INSTANCES[@]}"; do
        echo "  sudo systemctl restart llamacpp-batch${N}.service"
    done
fi
