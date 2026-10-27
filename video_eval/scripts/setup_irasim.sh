#!/usr/bin/env bash
# Prepare IRASim (github.com/bytedance/IRASim, commit c72b6da) for generation, outside the code folder.
#
# Usage: bash scripts/setup_irasim.sh <IRASIM_DIR> <DATA_DIR> <N_EPISODES>
#   <IRASIM_DIR>  where IRASim is cloned, e.g. /tmp/IRASim
#   <DATA_DIR>    folder holding opensource_robotdata/ from scripts/fetch_irasim.py, e.g. /tmp/irasim
#   <N_EPISODES>  how many Bridge test episodes go into annotation_subset/test (the first N by id)
# Then, in <IRASIM_DIR>:
#   PYTHONPATH=<IRASIM_DIR>/pkgs torchrun --nproc_per_node 1 main.py --config <this repo>/video_eval/configs/irasim/bridge_frame_ada.yaml
#
# IRASim pins diffusers 0.24.0, which needs an old huggingface_hub (and transformers 4.x with it); they go into
# <IRASIM_DIR>/pkgs, ahead of the system packages on PYTHONPATH, so the platform's own packages stay untouched.
#
# Author: Duc Anh Hoang
set -euo pipefail
IRASIM=$1
DATA=$2
N=$3

[ -d "$IRASIM/.git" ] || git clone -q https://github.com/bytedance/IRASim.git "$IRASIM"
git -C "$IRASIM" checkout -q c72b6da
pip install -q --no-deps --target "$IRASIM/pkgs" diffusers==0.24.0 huggingface_hub==0.25.2 transformers==4.46.3 \
    tokenizers==0.20.3 decord rotary_embedding_torch einops_exts imageio-ffmpeg

# SDXL VAE (frozen encoder and decoder of IRASim), where IRASim's configs/base/data.yaml looks for it
hf download stabilityai/stable-diffusion-xl-base-1.0 vae/config.json vae/diffusion_pytorch_model.safetensors \
    --local-dir "$IRASIM/IRASim/pretrained_models/stabilityai/stable-diffusion-xl-base-1.0" > /dev/null

ln -sfn "$DATA/opensource_robotdata/bridge" "$IRASIM/robotdata/opensource_robotdata/bridge"
SUBSET=$DATA/opensource_robotdata/bridge/annotation_subset/test
rm -rf "$SUBSET" && mkdir -p "$SUBSET"
ls "$DATA/opensource_robotdata/bridge/annotation/test" | sort -n | head -n "$N" | while read -r f; do
    ln -s "$DATA/opensource_robotdata/bridge/annotation/test/$f" "$SUBSET/$f"
done
echo "IRASim ready at $IRASIM; $(ls "$SUBSET" | wc -l) test episodes in $SUBSET"
