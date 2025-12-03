#!/bin/bash
set -e

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
    "$HOME/.wandb_api_key" vast_1:~/

# SSH into vast_1 instance
ssh -A -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null vast_1 <<EOF
set -e

mkdir -p repos
cd repos

echo "Remote WORKING_REPO: \$WORKING_REPO"
echo "Cloning branch: $CURRENT_BRANCH"

git clone -b "$CURRENT_BRANCH" "\$WORKING_REPO"
cd \$(basename "\$WORKING_REPO" .git)

git remote rename origin github

uv sync

wandb login \$(cat ~/.wandb_api_key)
rm -f ~/.wandb_api_key

uv run src/dataset
EOF
