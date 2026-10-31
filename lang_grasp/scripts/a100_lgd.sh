#!/usr/bin/env bash
# A100 job (agents/a100_guide.md): official LGD split, 3 seeds of the baseline and of the proposed loss, Grounding DINO
# boxes, then every model evaluated without gate, with the oracle gate and with the detector gate.
# Run detached on the Colab A100 from the repo's lang_grasp/ folder on Drive:
#   (setsid nohup bash scripts/a100_lgd.sh > <OUT>/logs/job.log 2>&1 < /dev/null &)
# Everything is written to Drive as it is produced; rerunning the script resumes (re-split copied back from Drive,
# finished runs skipped, unfinished runs resumed from last.ckpt, existing evaluations skipped).
# Author: Duc Anh Hoang
source /root/.hf_env
set -x
export PATH=$HOME/.local/bin:$PATH
TASK=$(cd "$(dirname "$0")/.." && pwd)                       # lang_grasp/ (code, read only)
BASE=/content/drive/MyDrive/it/fpt
OUT=$BASE/data/lang_grasp/a100_lgd                            # runs, eval, boxes, logs (persistent)
SPLIT_DRIVE=$BASE/data/datasets/processed/Grasp-Anything-pp-lgd   # the re-split (persistent copy)
WORK=/content/work                                            # local scratch: faster reads
DATA=$WORK/Grasp-Anything-pp-lgd
SEEDS=${SEEDS:-"123 124 125"}
EXPS=${EXPS:-"grconvnet_clip grconvnet_clip_contrast"}
LOG=$OUT/logs; mkdir -p $LOG $WORK
cd "$TASK"

# 1. environment and data, in parallel
pip install -q -r requirements.txt > $LOG/pip.log 2>&1 &
if [ -f $SPLIT_DRIVE/stats.json ] && ls $SPLIT_DRIVE/*_text_*.npz > /dev/null 2>&1; then
  mkdir -p $DATA && cp -r $SPLIT_DRIVE/. $DATA/ &
else
  python ../scripts/data_sync.py pull --base $WORK --only 'datasets/processed/Grasp-Anything-pp/**' > $LOG/pull.log 2>&1 &
fi
wait
if [ ! -f $DATA/stats.json ]; then
  python scripts/resplit_lgd.py --src $WORK/data/datasets/processed/Grasp-Anything-pp --out $DATA > $LOG/resplit.log 2>&1
  python scripts/cache_text_emb.py --processed $DATA > $LOG/cache_text.log 2>&1
  mkdir -p $SPLIT_DRIVE && cp -r $DATA/. $SPLIT_DRIVE/
fi

# 2. detector boxes (shares the GPU with training)
if [ ! -f $OUT/boxes/boxes_test_unseen_all.npz ]; then
  python -u scripts/detect_boxes.py --data $DATA --out $OUT/boxes > $LOG/boxes.log 2>&1 &
fi

# 3. training, all runs at once (each saves last.ckpt and the best checkpoint to Drive every epoch)
for EXP in $EXPS; do for S in $SEEDS; do
  RUN=$OUT/runs/${EXP}_s$S
  [ -f $RUN/DONE ] && continue
  RESUME=""; [ -f $RUN/checkpoints/last.ckpt ] && RESUME="ckpt_path=$RUN/checkpoints/last.ckpt"
  ( python -u src/train.py experiment=$EXP seed=$S name=${EXP}_s$S paths=colab paths.data_dir=$DATA \
      paths.log_dir=$OUT/runs hydra.run.dir=$RUN data.batch_size=16 trainer.devices=1 \
      trainer.enable_progress_bar=false $RESUME > $LOG/train_${EXP}_s$S.log 2>&1 && touch $RUN/DONE ) &
done; done
wait

# 4. evaluation of the best checkpoint of every run (val = official Seen test, as in the authors' code)
for EXP in $EXPS; do for S in $SEEDS; do
  RUN=$OUT/runs/${EXP}_s$S
  CKPT=$(python -c "import glob; print(max(glob.glob('$RUN/checkpoints/epoch*.ckpt'), key=lambda p: float(p.rsplit('success', 1)[1][:-5])))")
  for GATE in none oracle detector; do
    EV=$OUT/eval/${EXP}_s${S}_$GATE
    [ -f $EV/results.json ] && continue
    G=$([ $GATE = none ] && echo null || echo $GATE)
    python -u src/eval.py ckpt_path=$CKPT paths=colab paths.data_dir=$DATA paths.log_dir=$OUT/eval hydra.run.dir=$EV \
      "splits=[test_seen,test_unseen,test_unseen_all]" gate=$G boxes=$OUT/boxes > $LOG/eval_${EXP}_s${S}_$GATE.log 2>&1 &
  done
done; done
wait

# 5. push to the HF data repo (the runs are already on Drive)
cd .. && python scripts/data_sync.py push --only 'lang_grasp/a100_lgd/**' 'datasets/processed/Grasp-Anything-pp-lgd/**' \
  --message "lang_grasp: official LGD split, 3 seeds, gates" > $LOG/push.log 2>&1
echo "A100 JOB EXIT $?"
