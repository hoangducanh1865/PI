#!/usr/bin/env bash
# The jobs that need more than a Kaggle T4, on one Ampere-or-newer GPU machine (e.g. a Colab A100): the Cosmos clips
# (Cosmos-Predict2.5 needs Ampere) and the inverse dynamics model, trained in the background on the same GPU while
# Cosmos generates. Then every metric on the new sets, and a push of the results (scripts/data_sync.py).
#
# Usage (from video_eval/, HF_TOKEN and HF_WRITE_TOKEN in the environment; WANDB optional):
#   bash scripts/a100_job.sh <OUT_DIR>        e.g. /content/drive/MyDrive/it/fpt/data/video_eval/gen_bridge100
# <OUT_DIR> receives the Bridge clip set of 6 episodes per verb over 12 verbs (about 100 clips; the Kaggle kernel
# video-eval-bridge100 generates the IRASim models on the same set), its references, cosmos_action/, cosmos_base/,
# metrics_*.csv and idm.pt. Large downloads go to $WORK (default /content/work).
#
# Author: Duc Anh Hoang
set -euo pipefail
OUT=$1
WORK=${WORK:-/content/work}
DATA=$WORK/irasim
mkdir -p "$OUT" "$WORK"
VERBS="put move take open close flip fold wipe turn pour slide topple"

pip install -q -r requirements.txt decord diffusers scipy
python -u scripts/fetch_irasim.py get bridge_train_data --out $DATA \
    --match '/test/|/(annotation|latent_videos)/(train|val)/' > "$WORK/fetch.log"
python scripts/select_episodes.py --data $DATA --out "$OUT" --per-verb 6 --verbs $VERBS
python scripts/make_references.py --data $DATA --out "$OUT"

python -u scripts/idm.py train --data $DATA --ckpt "$OUT/idm.pt" > "$OUT/idm_train.log" 2>&1 &
idm_job=$!

bash scripts/setup_cosmos.sh "$WORK/cosmos-predict2.5" > "$WORK/setup_cosmos.log" 2>&1
python -u scripts/run_cosmos.py --cosmos "$WORK/cosmos-predict2.5" --data $DATA --gen "$OUT" --variant action \
    --work "$WORK/cosmos_work"
python -u scripts/run_cosmos.py --cosmos "$WORK/cosmos-predict2.5" --data $DATA --gen "$OUT" --variant base \
    --work "$WORK/cosmos_work"
wait $idm_job

models="real still cosmos_action/seed0 cosmos_base/seed0"
python scripts/eval_metrics.py --gen "$OUT" --models cosmos_action/seed0 cosmos_base/seed0 still
python scripts/eval_temporal.py --gen "$OUT" --models $models
python scripts/eval_motion.py --gen "$OUT" --tracker cotracker --models $models
python scripts/idm.py score --ckpt "$OUT/idm.pt" --gen "$OUT" --models $models
cd .. && python scripts/data_sync.py push --only 'video_eval/**' --message "A100 job: Cosmos clips, IDM, metrics"
