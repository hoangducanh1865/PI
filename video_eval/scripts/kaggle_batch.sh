#!/usr/bin/env bash
# Everything a fresh Kaggle machine needs to generate and score the chosen clips: fetch the checkpoints and the Bridge
# test data, set up IRASim, choose the episodes, build the real and still references, run the jobs (one queue per GPU,
# the queues in parallel), then score every generated set with the pixel and the motion metrics.
#
# Usage (from video_eval/, e.g. as the command of scripts/kaggle_run.py --task video_eval):
#   bash scripts/kaggle_batch.sh <OUT_DIR> <MODEL>:<SEED>:<GPU>[:<PERTURBATION>] [...]
#   e.g. bash scripts/kaggle_batch.sh /kaggle/working/data/video_eval/gen video_ada:3407:0 lvdm:3407:1
#        bash scripts/kaggle_batch.sh /kaggle/working/data/video_eval/gen frame_ada:3407:0:noise_s frame_ada:3407:1:freeze
# <PERTURBATION> (noise_s, noise_l, mirror, freeze; scripts/make_perturbed_annotations.py) generates from corrupted
# actions into <OUT_DIR>/<MODEL>_<PERTURBATION>/. Large files go to /tmp (/kaggle/working keeps only 20 GB).
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
PYTHONPATH=$IRASIM/pkgs python scripts/make_references.py --data $DATA --out "$OUT"
python scripts/make_perturbed_annotations.py --data $DATA

for gpu in $(printf '%s\n' "$@" | cut -d: -f3 | sort -u); do
    (
        sleep $((gpu * 240))  # each run keeps its ~11 GB checkpoint in RAM (31 GB): stagger the loading peaks
        for job in "$@"; do
            IFS=: read -r model seed job_gpu perturbation <<< "$job"
            [ "$job_gpu" = "$gpu" ] || continue
            extra=()
            [ -n "${perturbation:-}" ] && extra=(--annotation "annotation_$perturbation" --name "${model}_$perturbation")
            python -u scripts/run_irasim.py --irasim $IRASIM --model "$model" --seed "$seed" --gpu "$gpu" --out "$OUT" \
                "${extra[@]}" || true
        done
    ) &
done
wait
find "$OUT" -name '*.mp4' | sed "s|$OUT/||" | cut -d/ -f1-2 | sort | uniq -c

pip install -q -r requirements.txt
generated=$(cd "$OUT" && ls -d */seed* | paste -sd' ')
python scripts/eval_metrics.py --gen "$OUT" --models $generated still
python scripts/eval_motion.py --gen "$OUT" --tracker cotracker --models real still $generated
