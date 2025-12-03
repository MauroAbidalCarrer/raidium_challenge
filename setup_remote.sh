#!/bin/bash
set -e

# Optional host argument (default: vast)
HOST="${1:-vast}"

echo "Using host: $HOST"

# Check for wandb API key
if [ ! -f "$HOME/.wandb_api_key" ]; then
    echo "~/.wandb_api_key does not exist."
    exit 1
fi

# Detect local Git branch
CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
echo "Local branch detected: $CURRENT_BRANCH"

# Copy W&B key to remote
scp -q -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
    "$HOME/.wandb_api_key" "$HOST":~/

# SSH into host instance
ssh -A -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null "$HOST" <<EOF
set -e

mkdir -p repos
cd repos

echo "Remote WORKING_REPO: \$WORKING_REPO"
echo "Cloning branch: $CURRENT_BRANCH"

git clone -b "$CURRENT_BRANCH" "\$WORKING_REPO"
cd \$(basename "\$WORKING_REPO" .git)

git remote rename origin github

uv sync

uv run wandb login \$(cat ~/.wandb_api_key)
rm -f ~/.wandb_api_key

uv run python -c "from src import dataset; dataset.mk_dataset()"
EOF