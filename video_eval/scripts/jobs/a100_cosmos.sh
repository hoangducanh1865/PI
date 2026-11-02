#!/usr/bin/env bash
# The jobs that need more than a Kaggle T4, on one Ampere-or-newer GPU machine (e.g. a Colab A100, High-RAM): the
# Cosmos-Predict2.5 clips (needs Ampere) and EVID (NVIDIA Cosmos3-Nano inverse dynamics). Then every metric on the new
# sets, the end-effector drawings for the VLM check, and a push of the results (scripts/data_sync.py).
#
# Usage (from video_eval/, HF_TOKEN and HF_WRITE_TOKEN in the environment):
#   bash scripts/jobs/a100_cosmos.sh <OUT_DIR>
#   e.g. <BASE>/data/video_eval/video-eval-bridge100/data/video_eval/gen  (pulled with data_sync.py: the 105-clip
#        Bridge set with its IRASim clips from the Kaggle kernel video-eval-bridge100)
# Steps: (1) Cosmos3-Nano IDM on the real and still clips, then its server is stopped (vLLM reserves most of the GPU);
# (2) Cosmos-Predict2.5 action and base; (3) EVID of every set (the Cosmos3 server restarts from its local weight
# cache); (4) pixel, temporal and motion metrics of the Cosmos sets, drawings, push.
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
python -u scripts/data/fetch_irasim.py get bridge_train_data --out $DATA --match '/test/' > "$WORK/fetch.log"
python scripts/data/select_episodes.py --data $DATA --out "$OUT" --per-verb 6 --verbs $VERBS
python scripts/data/make_references.py --data $DATA --out "$OUT"

# (1) Cosmos3-Nano inverse dynamics on the references
bash scripts/setup/cosmos3.sh "$WORK/cosmos3_env"
python -u src/metrics/evid.py --gen "$OUT" --models real still
pkill -f "vllm serve" || true
sleep 20

# (2) Cosmos-Predict2.5
bash scripts/setup/cosmos.sh "$WORK/cosmos-predict2.5" > "$WORK/setup_cosmos.log" 2>&1
for variant in action base; do
    python -u src/generate/cosmos.py --cosmos "$WORK/cosmos-predict2.5" --data $DATA --gen "$OUT" --variant $variant \
        --work "$WORK/cosmos_work"
done

# (3) EVID of every set
bash scripts/setup/cosmos3.sh "$WORK/cosmos3_env"
python -u src/metrics/evid.py --gen "$OUT" --models real still $IRASIM $COSMOS
pkill -f "vllm serve" || true

# (4) the other metrics on the Cosmos sets, drawings for the VLM check, push
python src/metrics/pixel.py --gen "$OUT" --models $COSMOS
python src/metrics/temporal.py --gen "$OUT" --models real still $IRASIM $COSMOS
python src/metrics/track2d.py --gen "$OUT" --tracker cotracker --models real still $IRASIM $COSMOS --draw "$OUT/ee_check"
cd .. && python scripts/data_sync.py push --only 'video_eval/**' --message "A100 job: Cosmos clips, 3-D scores, metrics"
