#!/usr/bin/env bash
# The jobs that need more than a Kaggle T4, on one Ampere-or-newer GPU machine (e.g. a Colab A100, High-RAM): the
# Cosmos-Predict2.5 clips (needs Ampere) and the three inverse dynamics models for the 3-D scores: Cosmos3-Nano's own
# inverse dynamics (no training of ours), our IDM from scratch on SDXL latents, and a fine-tuned DINOv2 IDM. Then every
# metric on the new sets, the end-effector drawings for the VLM check, and a push of the results (scripts/data_sync.py).
#
# Usage (from video_eval/, HF_TOKEN and HF_WRITE_TOKEN in the environment):
#   bash scripts/a100_job.sh <OUT_DIR>
#   e.g. <BASE>/data/video_eval/video-eval-bridge100/data/video_eval/gen  (pulled with data_sync.py: the 105-clip
#        Bridge set with its IRASim clips from the Kaggle kernel video-eval-bridge100)
# Steps: (1) Cosmos3-Nano IDM on the real and still clips (its own error and floor); (2) Cosmos-Predict2.5 action and
# base, while our two IDMs train in the background on the same GPU; (3) both IDMs score every set; (4) Cosmos3-Nano IDM
# on the generated sets (slowest, last); (5) pixel, temporal and motion metrics of the Cosmos sets, drawings, push.
# Large downloads go to $WORK (default /content/work).
#
# Author: Duc Anh Hoang
set -euo pipefail
OUT=$1
WORK=${WORK:-/content/work}
DATA=$WORK/irasim
mkdir -p "$OUT" "$WORK"
VERBS="put move take open close flip fold wipe turn pour slide topple"
IRASIM="frame_ada/seed3407 frame_ada/seed3408 video_ada/seed3407 lvdm/seed3407"
COSMOS="cosmos_action/seed0 cosmos_base/seed0"

pip install -q -r requirements.txt decord
python -u scripts/fetch_irasim.py get bridge_train_data --out $DATA \
    --match '/test/|/(annotation|latent_videos|videos)/(train|val)/' > "$WORK/fetch.log"
python scripts/select_episodes.py --data $DATA --out "$OUT" --per-verb 6 --verbs $VERBS
python scripts/make_references.py --data $DATA --out "$OUT"

# (1) Cosmos3-Nano inverse dynamics on the references, then free the GPU
bash scripts/setup_cosmos3.sh "$WORK/cosmos3_env"
python -u scripts/cosmos3_idm.py --gen "$OUT" --models real still
pkill -f "vllm serve" || true
sleep 20

# (2) Cosmos-Predict2.5, with our IDMs training in the background
( python -u scripts/idm.py train --data $DATA --ckpt "$OUT/idm_latent.pt" --backbone latent
  python -u scripts/idm.py train --data $DATA --ckpt "$OUT/idm_dinov2.pt" --backbone dinov2 --batch 128 ) \
    > "$OUT/idm_train.log" 2>&1 &
idm_job=$!
bash scripts/setup_cosmos.sh "$WORK/cosmos-predict2.5" > "$WORK/setup_cosmos.log" 2>&1
for variant in action base; do
    python -u scripts/run_cosmos.py --cosmos "$WORK/cosmos-predict2.5" --data $DATA --gen "$OUT" --variant $variant \
        --work "$WORK/cosmos_work"
done
wait $idm_job

# (3) our IDMs on every set
for backbone in latent dinov2; do
    python scripts/idm.py score --ckpt "$OUT/idm_$backbone.pt" --backbone $backbone --gen "$OUT" \
        --models real still $IRASIM $COSMOS
done

# (4) Cosmos3-Nano inverse dynamics on the generated sets (answers are cached per clip)
bash scripts/setup_cosmos3.sh "$WORK/cosmos3_env"
python -u scripts/cosmos3_idm.py --gen "$OUT" --models real still $IRASIM $COSMOS
pkill -f "vllm serve" || true

# (5) the other metrics on the Cosmos sets, drawings for the VLM check, push
python scripts/eval_metrics.py --gen "$OUT" --models $COSMOS
python scripts/eval_temporal.py --gen "$OUT" --models real still $IRASIM $COSMOS
python scripts/eval_motion.py --gen "$OUT" --tracker cotracker --models real still $IRASIM $COSMOS --draw "$OUT/ee_check"
cd .. && python scripts/data_sync.py push --only 'video_eval/**' --message "A100 job: Cosmos clips, IDMs, metrics"
