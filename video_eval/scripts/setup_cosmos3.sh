#!/usr/bin/env bash
# Serve NVIDIA Cosmos3-Nano (16B, about 33 GB of weights) with vLLM-Omni for scripts/cosmos3_idm.py, in its own
# Python 3.12 environment so it does not touch the other packages. Not Docker (unavailable on Colab): vllm and vllm-omni
# 0.32.0 (2026-10-28; Cosmos3 pipeline with inverse dynamics and the bridge_orig_lerobot domain, checked 2026-10-29).
# Their PyPI build needs CUDA 13.0 (driver >= 580); with a driver 575 to 579 the CUDA 12.9 builds are used (torch from
# download.pytorch.org/whl/cu129, vllm from its GitHub release).
#
# Usage: bash scripts/setup_cosmos3.sh <ENV_DIR> [--install-only]      e.g. /content/work/cosmos3_env
# Starts the server on port 8000 in the background (log <ENV_DIR>/serve.log) and waits until it answers. Below 80 GB of
# GPU memory it adds --enable-layerwise-offload. Guardrails off (evaluation of robot clips only).
#
# Author: Duc Anh Hoang
set -euo pipefail
ENV=$1
CUDA=${CUDA:-}  # cu130 or cu129; default from the driver
export PATH="$HOME/.local/bin:$PATH"
command -v uv > /dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh; }
if [ -z "$CUDA" ]; then
    driver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)
    if [ "$driver" -ge 580 ]; then CUDA=cu130
    elif [ "$driver" -ge 575 ]; then CUDA=cu129
    else echo "driver $driver: Cosmos3 with vllm 0.32 needs >= 575"; exit 1
    fi
fi
[ -d "$ENV" ] || uv venv -q -p 3.12 "$ENV"
if [ "$CUDA" = cu130 ]; then
    uv pip install -q -p "$ENV/bin/python" vllm==0.32.0 vllm-omni==0.32.0
else
    wheel=https://github.com/vllm-project/vllm/releases/download/v0.32.0/vllm-0.32.0+cu129-cp38-abi3-manylinux_2_28_x86_64.whl
    uv pip install -q -p "$ENV/bin/python" --index-strategy unsafe-best-match \
        --extra-index-url https://download.pytorch.org/whl/cu129 \
        torch==2.13.0+cu129 torchvision==0.28.0+cu129 torchaudio==2.11.0+cu129 "vllm @ $wheel" vllm-omni==0.32.0
fi
"$ENV/bin/python" -c "import torch, vllm, vllm_omni; print('torch', torch.__version__, 'vllm', vllm.__version__)" 2>/dev/null
[ "${2:-}" = --install-only ] && exit 0

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
