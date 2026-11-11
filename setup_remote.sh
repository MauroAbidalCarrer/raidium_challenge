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
scp -q -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null "$HOME/.wandb_api_key" "$HOST":~/
scp -q -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null "$HOME/.vast_hf_token" "$HOST":~/
#scp -q -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null "./.env" "$HOST":"~/repos/raidium_challenge/"

scp -q -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null "./.env" "$HOST":"~/"


ENV_FILE=".env"

if [ ! -f "$ENV_FILE" ]; then
    echo ".env not found"
    exit 1
fi

ARTIFACTS_REPO_ID=$(grep -E '^ARTIFACTS_REPO_ID=' "$ENV_FILE" | cut -d= -f2- | tr -d '"')
if [ -z "$ARTIFACTS_REPO_ID" ]; then
    echo "ARTIFACTS_REPO_ID not set in .env"
    exit 1
fi

echo "Dataset repo: $ARTIFACTS_REPO_ID"

export HF_TOKEN=$(cat ~/.vast_hf_token)
# SSH into host instance
ssh -A -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null "$HOST" <<EOF
set -e
export HF_TOKEN=\$(cat ~/.vast_hf_token)
echo "Cloning dataset repo via HTTPS" \${ARTIFACTS_REPO_ID}
echo "hugging face token is:" \${HF_TOKEN}
echo "Dataset repo: $ARTIFACTS_REPO_ID"
echo "https://__token__:$HF_TOKEN@huggingface.co/datasets/$ARTIFACTS_REPO_ID artifacts"
git clone https://__token__:$HF_TOKEN@huggingface.co/datasets/$ARTIFACTS_REPO_ID artifacts 

mkdir -p repos
cd repos

echo "Remote WORKING_REPO: \$WORKING_REPO"
echo "Cloning branch: $CURRENT_BRANCH"

GIT_SSH_COMMAND="ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null" \
    git clone -b "$CURRENT_BRANCH" "\$WORKING_REPO"
cd \$(basename "\$WORKING_REPO" .git)

git remote rename origin github

uv sync

uv run wandb login \$(cat ~/.wandb_api_key)
rm -f ~/.wandb_api_key


export HF_TOKEN=\$(cat ~/.vast_hf_token)
echo "Cloning dataset repo via HTTPS" \${ARTIFACTS_REPO_ID}
echo "hugging face token is:" \${ARTIFACTS_REPO_ID}
git clone https://user:\${HF_TOKEN}@huggingface.co/datasets/\${ARTIFACTS_REPO_ID} artifacts 

cd artifacts
# ensure git uses token automatically
git config credential.helper store
echo "https://user:\$HF_TOKEN@huggingface.co" > ~/.git-credentials

uv run python -c "from raidium_challenge import dataset; dataset.mk_dataset()"
EOF
