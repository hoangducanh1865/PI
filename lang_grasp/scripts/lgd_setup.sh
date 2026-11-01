#!/usr/bin/env bash
# Set up the official LGD code (github.com/Fsoft-AIC/LGD at 0223d43) for scripts/lgd_official.sh: their repository,
# the released checkpoint (link of their README), the official Grasp-Anything / Grasp-Anything++ files from Hugging Face
# (airvlab/*) and a Python 3.9 environment with torch 1.12.1 + cu113 and the packages of their requirements.txt.
# Downloads and the environment run in parallel; the data is then laid out for their loader (scripts/lgd_extract.py).
# Usage: W=<WORK_DIR> bash scripts/lgd_setup.sh   (default W=/content/work/lgd; about 70 GB of downloads)
# Author: Duc Anh Hoang
set -x
W=${W:-/content/work/lgd}
TASK=$(cd "$(dirname "$0")/.." && pwd)
mkdir -p $W/raw $W/logs && cd $W
git clone -q https://github.com/Fsoft-AIC/LGD.git && git -C LGD checkout -q 0223d43e5ea07d459f21296bb1305e2e94d86cbd
pip install -q gdown hf_transfer uv
export HF_HUB_ENABLE_HF_TRANSFER=1
gdown -q --fuzzy "https://drive.google.com/file/d/1_Xu4biqQMq-3eCvCurZwNdKirMSgH-Nu/view?usp=sharing" \
  -O $W/ckpt_download > $W/logs/gdown.log 2>&1 &
hf download airvlab/Grasp-Anything scene_description.zip image_part_aa image_part_ab --repo-type dataset \
  --local-dir $W/raw/Grasp-Anything > $W/logs/hf_ga.log 2>&1 &
hf download airvlab/Grasp-Anything-pp grasp_label_positive.zip --repo-type dataset \
  --local-dir $W/raw/Grasp-Anything-pp > $W/logs/hf_gapp.log 2>&1 &
# CLIP's setup.py needs pkg_resources: an old setuptools and no build isolation
( uv venv -q -p 3.9 $W/venv \
  && VIRTUAL_ENV=$W/venv uv pip install -q torch==1.12.1+cu113 torchvision==0.13.1+cu113 \
       --extra-index-url https://download.pytorch.org/whl/cu113 \
  && VIRTUAL_ENV=$W/venv uv pip install -q "setuptools<70" wheel \
  && VIRTUAL_ENV=$W/venv uv pip install -q --no-build-isolation \
       "git+https://github.com/openai/CLIP.git@a1d071733d7111c9c014f024669f959182114e33" \
  && VIRTUAL_ENV=$W/venv uv pip install -q "numpy<1.24" "transformers==4.8.1" "timm==0.9.7" scikit-image \
       opencv-python-headless imageio matplotlib scipy ruamel.yaml PyYAML ftfy regex tqdm tensorboardX piq omegaconf \
       hydra-core torchsummary ) > $W/logs/env.log 2>&1 &
wait
python3 $TASK/scripts/lgd_extract.py --raw $W/raw --lgd $W/LGD --out $W/data > $W/logs/extract.log 2>&1
mkdir -p $W/ckpt && (cd $W/ckpt && python3 -m zipfile -e $W/ckpt_download .)
echo "LGD SETUP DONE"
