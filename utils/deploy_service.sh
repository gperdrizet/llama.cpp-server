#!/usr/bin/env bash
# utils/deploy_service.sh
#
# Deploys utils/llamacpp.service to /etc/systemd/system/llamacpp.service,
# under llama user, substituting values from .env into the service file.
#
# Usage:
#   ./utils/deploy_service.sh [--restart]
#
#   --restart   Also restart the service after deploying (default: daemon-reload only)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TEMPLATE="$REPO_ROOT/utils/llamacpp.service"
ENV_FILE="$REPO_ROOT/.env"
DEST="/etc/systemd/system/llamacpp.service"
DO_RESTART=false

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
for arg in "$@"; do
    case "$arg" in
        --restart) DO_RESTART=true ;;
        *) echo "Unknown argument: $arg" >&2; exit 1 ;;
    esac
done

# ---------------------------------------------------------------------------
# Load .env
# ---------------------------------------------------------------------------
if [[ ! -f "$ENV_FILE" ]]; then
    echo "ERROR: $ENV_FILE not found." >&2
    echo "       Copy .env.template to .env and fill in your API key." >&2
    exit 1
fi

# Source only valid KEY=value lines; ignore comments and blank lines
set -o allexport
# shellcheck disable=SC1090
source <(grep -E '^[A-Za-z_][A-Za-z0-9_]*=' "$ENV_FILE")
set +o allexport

# ---------------------------------------------------------------------------
# Validate
# ---------------------------------------------------------------------------

# Check API key presence and validity
if [[ -z "${API_KEY:-}" ]]; then
    echo "ERROR: API_KEY is not set in $ENV_FILE" >&2
    exit 1
fi

if [[ "$API_KEY" == "your_api_key_here" ]]; then
    echo "ERROR: API_KEY still has the placeholder value. Edit $ENV_FILE first." >&2
    exit 1
fi

# Check llama.cpp path presence and validity
if [[ -z "${LLAMA_PATH:-}" ]]; then
    echo "ERROR: LLAMA_PATH is not set in $ENV_FILE" >&2
    exit 1
fi

if [[ "$LLAMA_PATH" == "your_llama_path_here" ]]; then
    echo "ERROR: LLAMA_PATH still has the placeholder value. Edit $ENV_FILE first." >&2
    exit 1
fi

# Check model directory path presence and validity
if [[ -z "${MODEL_DIR:-}" ]]; then
    echo "ERROR: MODEL_DIR is not set in $ENV_FILE" >&2
    exit 1
fi

if [[ "$MODEL_DIR" == "your_model_dir_here" ]]; then
    echo "ERROR: MODEL_DIR still has the placeholder value. Edit $ENV_FILE first." >&2
    exit 1
fi

# Check model file name presence and validity
if [[ -z "${MODEL:-}" ]]; then
    echo "ERROR: MODEL is not set in $ENV_FILE" >&2
    exit 1
fi

if [[ "$MODEL" == "model_file_here.gguf" ]]; then
    echo "ERROR: MODEL still has the placeholder value. Edit $ENV_FILE first." >&2
    exit 1
fi

# Check CUDA device presence
if [[ -z "${CUDA_DEVICE:-}" ]]; then
    echo "ERROR: CUDA_DEVICE is not set in $ENV_FILE" >&2
    exit 1
fi

# Check model context size presence and validity
if [[ -z "${CTX_SIZE:-}" ]]; then
    echo "ERROR: CTX_SIZE is not set in $ENV_FILE" >&2
    exit 1
fi

if [[ "$CTX_SIZE" == "model_max_context_here" ]]; then
    echo "ERROR: CTX_SIZE still has the placeholder value. Edit $ENV_FILE first." >&2
    exit 1
fi

if ! [[ "$CTX_SIZE" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: CTX_SIZE must be a positive integer (got: '$CTX_SIZE')" >&2
    exit 1
fi

# Check GPU layers presence and validity
if [[ -z "${GPU_LAYERS:-}" ]]; then
    echo "ERROR: GPU_LAYERS is not set in $ENV_FILE" >&2
    exit 1
fi

if ! [[ "$GPU_LAYERS" =~ ^-?[0-9]+$ ]]; then
    echo "ERROR: GPU_LAYERS must be an integer (got: '$GPU_LAYERS')" >&2
    exit 1
fi

# Check slots presence and validity
if [[ -z "${SLOTS:-}" ]]; then
    echo "ERROR: SLOTS is not set in $ENV_FILE" >&2
    exit 1
fi

if ! [[ "$SLOTS" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: SLOTS must be a positive integer (got: '$SLOTS')" >&2
    exit 1
fi

# Check prompt cache size presence and validity
if [[ -z "${PROMPT_CACHE_SIZE:-}" ]]; then
    echo "ERROR: PROMPT_CACHE_SIZE is not set in $ENV_FILE" >&2
    exit 1
fi

if ! [[ "$PROMPT_CACHE_SIZE" =~ ^-?[0-9]+$ ]]; then
    echo "ERROR: PROMPT_CACHE_SIZE must be an integer (got: '$PROMPT_CACHE_SIZE')" >&2
    exit 1
fi

# Tensor split is optional (empty = single GPU). If set, must be comma-separated numbers.
TENSOR_SPLIT="${TENSOR_SPLIT:-}"
if [[ -n "$TENSOR_SPLIT" ]] && ! [[ "$TENSOR_SPLIT" =~ ^[0-9]+(\.[0-9]+)?(,[0-9]+(\.[0-9]+)?)+$ ]]; then
    echo "ERROR: TENSOR_SPLIT must be empty or comma-separated numbers e.g. '1,1' (got: '$TENSOR_SPLIT')" >&2
    exit 1
fi

# Check that the 'llama' system user exists
if ! id -u llama &>/dev/null; then
    echo "ERROR: System user 'llama' does not exist." >&2
    echo "       Create it first with:" >&2
    echo "         sudo useradd --system --no-create-home --shell /usr/sbin/nologin llama" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Substitute and deploy
# ---------------------------------------------------------------------------
RENDERED="$(sed \
    -e "s/SUB_API_KEY_HERE/${API_KEY}/" \
    -e "s|SUB_LLAMA_PATH_HERE|${LLAMA_PATH}|g" \
    -e "s|SUB_MODEL_DIR_HERE|${MODEL_DIR}|g" \
    -e "s/SUB_MODEL_FILE_HERE/${MODEL}/" \
    -e "s/SUB_CUDA_DEVICE_HERE/${CUDA_DEVICE}/" \
    -e "s/SUB_CTX_SIZE_HERE/${CTX_SIZE}/" \
    -e "s/SUB_GPU_LAYERS_HERE/${GPU_LAYERS}/" \
    -e "s/SUB_SLOTS_HERE/${SLOTS}/" \
    -e "s/SUB_PROMPT_CACHE_SIZE_HERE/${PROMPT_CACHE_SIZE}/" \
    -e "s/SUB_TENSOR_SPLIT_HERE/${TENSOR_SPLIT}/" \
    "$TEMPLATE")"

# Strip --tensor-split line if no value was configured
if [[ -z "$TENSOR_SPLIT" ]]; then
    RENDERED="$(echo "$RENDERED" | sed '/--tensor-split/d')"
fi

echo "Deploying $TEMPLATE → $DEST"
echo "$RENDERED" | sudo tee "$DEST" > /dev/null

echo "Running: systemctl daemon-reload"
sudo systemctl daemon-reload

if [[ "$DO_RESTART" == true ]]; then
    echo "Running: systemctl restart llamacpp.service"
    sudo systemctl restart llamacpp.service
    echo "Service restarted. Status:"
    systemctl status llamacpp.service --no-pager -l
else
    echo ""
    echo "Unit file deployed. Run the following to apply changes to the running service:"
    echo "  sudo systemctl restart llamacpp.service"
fi
