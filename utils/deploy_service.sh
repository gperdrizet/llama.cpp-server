#!/usr/bin/env bash
# utils/deploy_service.sh
#
# Deploys utils/llamacpp.service to /etc/systemd/system/llamacpp.service,
# substituting LLAMA_API_KEY from .env into the --api-key argument.
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
if [[ -z "${LLAMA_API_KEY:-}" ]]; then
    echo "ERROR: LLAMA_API_KEY is not set in $ENV_FILE" >&2
    exit 1
fi

if [[ "$LLAMA_API_KEY" == "your_api_key_here" ]]; then
    echo "ERROR: LLAMA_API_KEY still has the placeholder value. Edit $ENV_FILE first." >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Substitute and deploy
# ---------------------------------------------------------------------------
RENDERED="$(sed "s/YOUR_API_KEY_HERE/${LLAMA_API_KEY}/" "$TEMPLATE")"

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
