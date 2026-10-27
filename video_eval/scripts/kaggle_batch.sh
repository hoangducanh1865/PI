#!/usr/bin/env bash
# Everything a fresh Kaggle machine needs to generate the chosen clips: fetch the checkpoints and the Bridge test data,
# set up IRASim, choose the episodes, then run the jobs, one queue per GPU, the queues in parallel.
#
# Usage (from video_eval/, e.g. as the command of scripts/kaggle_run.py --task video_eval):
#   bash scripts/kaggle_batch.sh <OUT_DIR> <MODEL>:<SEED>:<GPU> [<MODEL>:<SEED>:<GPU> ...]
#   e.g. bash scripts/kaggle_batch.sh /kaggle/working/data/video_eval/gen video_ada:3407:0 frame_ada:3409:0 lvdm:3407:1
# Large files go to /tmp (/kaggle/working keeps only 20 GB); videos and logs go to <OUT_DIR>.
#
# Author: Duc Anh Hoang
set -euo pipefail
OUT=$1
shift
DATA=/tmp/irasim
IRASIM=/tmp/IRASim
mkdir -p "$OUT"

models=$(printf '%s\n' "$@" | cut -d: -f1 | sort -u | paste -sd'|')
python -u scripts/fetch_irasim.py get bridge_checkpoints_data --out $DATA --match "($models)/.*[.]pt$"
python -u scripts/fetch_irasim.py get bridge_train_data --out $DATA --match '/test/' > /dev/null
bash scripts/setup_irasim.sh $IRASIM $DATA
python scripts/select_episodes.py --data $DATA --out "$OUT" --verbs put move take open close flip fold wipe

for gpu in $(printf '%s\n' "$@" | cut -d: -f3 | sort -u); do
    (
        sleep $((gpu * 240))  # each run keeps its ~11 GB checkpoint in RAM (31 GB): stagger the loading peaks
        for job in "$@"; do
            IFS=: read -r model seed job_gpu <<< "$job"
            [ "$job_gpu" = "$gpu" ] || continue
            python -u scripts/run_irasim.py --irasim $IRASIM --model "$model" --seed "$seed" --gpu "$gpu" --out "$OUT" || true
        done
    ) &
done
wait
find "$OUT" -name '*.mp4' | sed "s|$OUT/||" | cut -d/ -f1-2 | sort | uniq -c
