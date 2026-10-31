#!/usr/bin/env bash
# Kaggle batch kernel (2 x T4, scripts/kaggle_run.py): the baseline on GPU 0 and the proposed model on GPU 1 with the
# same seed, one GPU each (batch 8, the authors' default, no DDP), on the official LGD split (Kaggle Dataset
# grasp-anything-pp-lgd); then each best checkpoint is evaluated without gate and with the oracle gate.
# Usage (from lang_grasp/): bash scripts/kaggle_pair.sh <SEED>
# Author: Duc Anh Hoang
S=$1
L=/kaggle/working/data/lang_grasp/lgd_s$S
mkdir -p $L/logs
i=0
for E in grconvnet_clip grconvnet_clip_contrast; do
  (
    export CUDA_VISIBLE_DEVICES=$i
    RUN=$L/runs/${E}_s$S
    python -u src/train.py experiment=$E seed=$S name=${E}_s$S trainer.devices=1 hydra.run.dir=$RUN \
      > $L/logs/train_${E}_s$S.log 2>&1
    CKPT=$(python -c "import glob; print(max(glob.glob('$RUN/checkpoints/epoch*.ckpt'), key=lambda p: float(p.rsplit('success', 1)[1][:-5])))")
    for G in null oracle; do
      python -u src/eval.py ckpt_path=$CKPT hydra.run.dir=$L/eval/${E}_s${S}_$G \
        "splits=[test_seen,test_unseen,test_unseen_all]" gate=$G > $L/logs/eval_${E}_s${S}_$G.log 2>&1
    done
  ) &
  i=$((i + 1))
done
wait
grep -h "success\|H " $L/logs/eval_*.log
