#!/usr/bin/env bash
# Prepare IRASim (github.com/bytedance/IRASim, commit c72b6da) for generation, outside the code folder.
#
# Usage: bash scripts/setup_irasim.sh <IRASIM_DIR> <DATA_DIR>
#   <IRASIM_DIR>  where IRASim is cloned, e.g. /tmp/IRASim
#   <DATA_DIR>    folder holding opensource_robotdata/ from scripts/fetch_irasim.py, e.g. /tmp/irasim
# Then scripts/select_episodes.py and scripts/run_irasim.py.
#
# IRASim pins diffusers 0.24.0, which needs an old huggingface_hub (and transformers 4.x with it); they go into
# <IRASIM_DIR>/pkgs, ahead of the system packages on PYTHONPATH, so the platform's own packages stay untouched.
#
# Author: Duc Anh Hoang
set -euo pipefail
IRASIM=$1
DATA=$2

[ -d "$IRASIM/.git" ] || git clone -q https://github.com/bytedance/IRASim.git "$IRASIM"
git -C "$IRASIM" checkout -q c72b6da
pip install -q --no-deps --target "$IRASIM/pkgs" diffusers==0.24.0 huggingface_hub==0.25.2 transformers==4.46.3 \
    tokenizers==0.20.3 decord rotary_embedding_torch einops_exts imageio-ffmpeg

# torch >= 2.6 loads checkpoints with weights_only=True; the official ones also pickle their OmegaConf training config.
# Allow exactly the classes they hold (torch.serialization.get_unsafe_globals_in_checkpoint, 2026-10-27) instead of
# turning weights_only off, which would let a checkpoint run arbitrary code. Imported at start-up from PYTHONPATH.
cat > "$IRASIM/pkgs/sitecustomize.py" <<'PY'
import collections, typing, torch, omegaconf
torch.serialization.add_safe_globals([
    omegaconf.base.Metadata, omegaconf.base.ContainerMetadata, omegaconf.nodes.AnyNode, omegaconf.dictconfig.DictConfig,
    omegaconf.listconfig.ListConfig, collections.defaultdict, typing.Any, dict, list, int,
])
PY

# SDXL VAE (frozen encoder and decoder of IRASim), where IRASim's configs/base/data.yaml looks for it
hf download stabilityai/stable-diffusion-xl-base-1.0 vae/config.json vae/diffusion_pytorch_model.safetensors \
    --local-dir "$IRASIM/IRASim/pretrained_models/stabilityai/stable-diffusion-xl-base-1.0" > /dev/null

ln -sfn "$DATA/opensource_robotdata/bridge" "$IRASIM/robotdata/opensource_robotdata/bridge"
echo "IRASim ready at $IRASIM"
