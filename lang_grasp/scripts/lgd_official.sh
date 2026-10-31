#!/usr/bin/env bash
# The official LGD code (github.com/Fsoft-AIC/LGD at 0223d43) run exactly as its README documents, on the Colab A100:
#   1. the released LGD checkpoint: evaluate_diffusion --seen 1 and --seen 0
#   2. the official GR-ConvNet + CLIP baseline: train_network.py --network lgrconvnet3 --seen 1 (all defaults), then
#      evaluate.py --seen 1 and --seen 0 on its best saved model (highest validation IoU in the file name)
# No line of their code is changed. In that code the training set (90%) and the Seen test are both drawn from
# split/grasp-anything++/test/seen.obj; the Unseen test (test/unseen.obj) is never trained on, so only Unseen is
# compared (user, 2026-10-31). Prerequisites in $W (task notes, 2026-10-31): LGD clone, ckpt_download (README link),
# raw/ (airvlab Hugging Face files), venv (Python 3.9, torch 1.12.1+cu113, packages of their requirements.txt).
# Usage: (setsid nohup bash scripts/lgd_official.sh > <log> 2>&1 < /dev/null &)
# Author: Duc Anh Hoang
set -x
W=/content/work/lgd
P=/content/drive/MyDrive/it/fpt/data/lang_grasp/a100_lgd/lgd
TASK=$(cd "$(dirname "$0")/.." && pwd)
mkdir -p $P/logs $P/lgrconvnet3
if [ ! -d $W/data/ga/image ] || [ -z "$(ls $W/data/ga/image)" ]; then
  python3 $TASK/scripts/lgd_extract.py --raw $W/raw --lgd $W/LGD --out $W/data > $P/logs/extract.log 2>&1 || exit 1
fi
CKPT=$W/ckpt/CVPR_2024_weights/model_lgd_grasp_anything++
[ -f $CKPT ] || (mkdir -p $W/ckpt && cd $W/ckpt && python3 -m zipfile -e $W/ckpt_download .)
PY=$W/venv/bin/python
DATA="--dataset grasp-anywhere --dataset-path $W/data/ga --add-file-path $W/data/gapp --use-depth 0"
cd $W/LGD

for SEEN in 0 1; do
  $PY -m torch.distributed.launch --nproc_per_node=1 --master_port=$((29600 + SEEN)) --use_env -m evaluate_diffusion \
    $DATA --iou-eval --seen $SEEN --network $CKPT > $P/logs/lgd_eval_seen$SEEN.log 2>&1 &
done

$PY train_network.py $DATA --description lgrconvnet3_official --seen 1 --network lgrconvnet3 --logdir $P/lgrconvnet3 \
  > $P/logs/lgrconvnet3_train.log 2>&1
NET=$(python3 -c "import glob; print(max(glob.glob('$P/lgrconvnet3/*/epoch_*'), key=lambda p: (float(p.rsplit('_', 1)[1]), p)))")
for SEEN in 0 1; do
  $PY evaluate.py $DATA --iou-eval --seen $SEEN --network $NET > $P/logs/lgrconvnet3_eval_seen$SEEN.log 2>&1
done
wait
grep -h "IOU Results" $P/logs/*.log
echo "LGD OFFICIAL EXIT"
