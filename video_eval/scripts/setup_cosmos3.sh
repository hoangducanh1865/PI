#!/usr/bin/env bash
# Serve NVIDIA Cosmos3-Nano (16B, about 33 GB of weights) with vLLM-Omni for scripts/cosmos3_idm.py, in its own
# Python 3.12 environment so it does not touch the other packages. Not Docker (unavailable on Colab): the vllm and
# vllm-omni packages of the same release (0.32.0, 2026-10-28; untested with Cosmos3 at writing time).
#
# Usage: bash scripts/setup_cosmos3.sh <ENV_DIR>      e.g. /content/work/cosmos3_env
# Starts the server on port 8000 in the background (log <ENV_DIR>/serve.log) and waits until it answers. Below 80 GB of
# GPU memory it adds --enable-layerwise-offload. Guardrails off (evaluation of robot clips only).
#
# Author: Duc Anh Hoang
set -euo pipefail
ENV=$1
export PATH="$HOME/.local/bin:$PATH"
command -v uv > /dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh; }
[ -d "$ENV" ] || uv venv -q -p 3.12 "$ENV"
uv pip install -q -p "$ENV/bin/python" vllm==0.32.0 vllm-omni==0.32.0

memory=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
offload=()
[ "$memory" -lt 79000 ] && offload=(--enable-layerwise-offload)
nohup "$ENV/bin/vllm" serve nvidia/Cosmos3-Nano --omni --host 0.0.0.0 --port 8000 --init-timeout 1800 \
    --no-guardrails "${offload[@]}" > "$ENV/serve.log" 2>&1 &
for _ in $(seq 180); do
    curl -sf http://localhost:8000/health > /dev/null && { echo "Cosmos3-Nano served on :8000 (${offload[*]:-no offload})"; exit 0; }
    sleep 10
done
echo "server did not start in 30 min, see $ENV/serve.log"; exit 1
