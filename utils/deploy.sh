#!/usr/bin/env bash
# utils/deploy.sh
#
# Deploys the llama.cpp backend (agent router + student pool) from this repo:
#   - unit files      -> /etc/systemd/system/
#   - non-secret env  -> /etc/llama/  (agent-router.ini, llama-pool-global,
#                                    llama-pool-instances)
#
# Secrets live in /etc/llama/llama-secrets.env (root-only 0600). On first run
# the script scaffolds it from utils/llama-secrets.env.example and stops so
# you can fill in the keys. It never overwrites an existing secrets file.
#
# This script does NOT start, stop, or restart any service. After deploying,
# pick the profile to run with:
#   sudo bash utils/switch-llama-profile agents   # personal agent router
#   sudo bash utils/switch-llama-profile pool     # dual-replica student pool
#
# Usage:
#   bash utils/deploy.sh
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UTILS="$REPO_ROOT/utils"
ETC_LLAMA="/etc/llama"
SECRETS="$ETC_LLAMA/llama-secrets.env"

echo ">> Updating profile switcher script"
sudo install -m 755 "$UTILS/switch-llama-profile" /usr/local/bin/switch-llama-profile

echo ">> Installing unit files"
sudo install -m 644 "$UTILS/agent-router.service" /etc/systemd/system/agent-router.service
sudo install -m 644 "$UTILS/llama-pool@.service"  /etc/systemd/system/llama-pool@.service
sudo install -m 644 "$UTILS/llama-embed.service"  /etc/systemd/system/llama-embed.service

echo ">> Installing non-secret config"
sudo install -d -m 755 -o root -g root "$ETC_LLAMA"
sudo install -m 644 "$UTILS/agent-router.ini"     "$ETC_LLAMA/agent-router.ini"
sudo install -m 644 "$UTILS/llama-pool-global"    "$ETC_LLAMA/llama-pool-global"
sudo install -m 644 "$UTILS/llama-pool-instances" "$ETC_LLAMA/llama-pool-instances"

if [ ! -f "$SECRETS" ]; then
    sudo install -m 600 -o root -g root "$UTILS/llama-secrets.env.example" "$SECRETS"
    echo ">> Created $SECRETS from template."
    echo ">> Fill in the keys (sudo nano $SECRETS), then re-run this script and run switch-llama-profile."
    exit 0
fi
sudo chown root:root "$SECRETS"
sudo chmod 600 "$SECRETS"

echo ">> Reloading systemd"
sudo systemctl daemon-reload

echo ">> Deploy complete. Select a profile to run:"
echo "     sudo bash utils/switch-llama-profile agents"
echo "     sudo bash utils/switch-llama-profile pool"
echo "     sudo bash utils/switch-llama-profile none"
