#!/usr/bin/env bash
# Prepare NVIDIA Cosmos-Predict2.5 (github.com/nvidia-cosmos/cosmos-predict2.5, commit a2c298b0a3df) for
# scripts/run_cosmos.py, outside the code folder, following the repo's docs/setup.md (virtual environment route).
#
# Usage: bash scripts/setup_cosmos.sh <COSMOS_DIR>       e.g. /tmp/cosmos-predict2.5
# Then: source <COSMOS_DIR>/.venv/bin/activate, export HF_TOKEN (read access; the NVIDIA Open Model License of
# nvidia/Cosmos-Predict2.5-2B accepted on Hugging Face), and run scripts/run_cosmos.py.
# Needs an Ampere-or-newer GPU and an NVIDIA driver >= 570 (CUDA 12.8); a T4 (Turing) cannot run it.
#
# Author: Duc Anh Hoang
set -euo pipefail
COSMOS=$1

driver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1
echo "driver $driver (needs >= 570)"
[ "${driver%%.*}" -ge 570 ] || { echo "driver too old for CUDA 12.8"; exit 1; }

command -v git-lfs > /dev/null || { apt-get -qq update && apt-get -qq install -y git-lfs ffmpeg > /dev/null; }
git lfs install > /dev/null
[ -d "$COSMOS/.git" ] || git clone -q https://github.com/nvidia-cosmos/cosmos-predict2.5.git "$COSMOS"
git -C "$COSMOS" checkout -q a2c298b0a3df
git -C "$COSMOS" lfs pull
command -v uv > /dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh; }
export PATH="$HOME/.local/bin:$PATH"
cd "$COSMOS"
uv python install
uv sync --extra=cu128
echo "Cosmos-Predict2.5 ready at $COSMOS (activate $COSMOS/.venv)"
