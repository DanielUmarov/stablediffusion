#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# RunPod bootstrap + CC12M latent pipeline
#
# What it does
# - Clones private GitHub repo using PAT (prompted, not echoed)
# - Sets AWS creds via env vars (prompted, not echoed)
# - Installs Python deps (requirements.txt)
# - Runs "next N tars" job based on S3 progress
#
# Usage
#   bash runpod_bootstrap_and_build.sh
#   COUNT=10 bash runpod_bootstrap_and_build.sh
###############################################################################

# -------- Settings you should edit --------
GITHUB_REPO="DanielUmarov/stablediffusion"     # e.g. danielumarov/stablediffusion
BRANCH="main"
COUNT="${COUNT:-10}"               # next N tars to process

# ----------------------------------------
# 0) Prevent secrets from being saved in shell history
# ----------------------------------------
export HISTFILE=/dev/null
set +o history

echo "== RunPod bootstrap starting =="

# ----------------------------------------
# 1) Clone private repo (PAT prompt)
# ----------------------------------------
read -s -p "GitHub PAT: " GITHUB_TOKEN; echo
rm -rf repo
git clone -b "$BRANCH" "https://${GITHUB_TOKEN}@github.com/${GITHUB_REPO}.git" repo
unset GITHUB_TOKEN
cd repo

# ----------------------------------------
# 2) AWS credentials (prompt) + verify
# ----------------------------------------
read -s -p "AWS_ACCESS_KEY_ID: " AWS_ACCESS_KEY_ID; echo
read -s -p "AWS_SECRET_ACCESS_KEY: " AWS_SECRET_ACCESS_KEY; echo
export AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY

# Optional: if you use temporary credentials, uncomment:
# read -s -p "AWS_SESSION_TOKEN (optional): " AWS_SESSION_TOKEN; echo
# export AWS_SESSION_TOKEN

export AWS_DEFAULT_REGION="us-east-1"  # change if needed

echo "== Verifying AWS credentials =="
aws sts get-caller-identity

# ----------------------------------------
# 3) Install Python deps
# ----------------------------------------
if [ ! -f requirements.txt ]; then
  echo "[ERROR] requirements.txt not found at repo root."
  echo "Make sure you added it and pushed it to GitHub."
  exit 1
fi

python3 -m pip install --upgrade pip
pip install -r requirements.txt

# ----------------------------------------
# 4) Sanity checks: required scripts exist
# ----------------------------------------
if [ ! -f scripts/download_cc12m_shard.py ]; then
  echo "[ERROR] scripts/download_cc12m_shard.py not found."
  exit 1
fi

if [ ! -f scripts/runpod_cc12m_auto_next_tars.sh ]; then
  echo "[ERROR] scripts/runpod_cc12m_auto_next_tars.sh not found."
  echo "Create it (the auto-next-tars runner) and push to GitHub."
  exit 1
fi

chmod +x scripts/runpod_cc12m_auto_next_tars.sh

# ----------------------------------------
# 5) Run the pipeline: process next N tars
# ----------------------------------------
echo "== Running pipeline: next ${COUNT} tars =="
bash scripts/runpod_cc12m_auto_next_tars.sh "${COUNT}"

echo "== Done =="
