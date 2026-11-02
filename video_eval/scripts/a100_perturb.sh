#!/usr/bin/env bash
# The perturbation test on the 105 Bridge clips, on one A100: IRASim-Frame-Ada (seed 3407) from corrupted actions
# (scripts/make_perturbed_annotations.py), then the EVID cycle error (Cosmos3-Nano inverse dynamics), pixel and 2-D track
# scores of each set, and a push of the results. Every video is copied to <OUT_DIR> (on Drive) within a minute of being
# written, and the inverse dynamics answers are cached per clip there, so a dropped runtime loses little; a rerun skips
# the sets whose videos are complete and the clips whose answers are cached.
#
# Usage (from video_eval/, HF_TOKEN and HF_WRITE_TOKEN in the environment):
#   bash scripts/a100_perturb.sh <OUT_DIR>
#   e.g. /content/drive/MyDrive/it/fpt/data/video_eval/video-eval-bridge100/data/video_eval/gen  (the 105-clip set)
# Large downloads go to $WORK (default /content/work); BATCH clips per IRASim batch (default 12).
#
# Author: Duc Anh Hoang
set -euo pipefail
OUT=$1
WORK=${WORK:-/content/work}
DATA=$WORK/irasim
IRASIM=$WORK/IRASim
BATCH=${BATCH:-12}
VERBS="put move take open close flip fold wipe turn pour slide topple"
SETS="noise_s noise_l freeze mirror"
mkdir -p "$OUT" "$WORK"

pip install -q -r requirements.txt decord > "$WORK/pip.log" 2>&1 &
python -u scripts/fetch_irasim.py get bridge_checkpoints_data --out $DATA --match 'frame_ada/.*[.]pt$' > "$WORK/fetch_ckpt.log" 2>&1 &
python -u scripts/fetch_irasim.py get bridge_train_data --out $DATA --match '/test/' > "$WORK/fetch_test.log" 2>&1 &
bash scripts/setup_cosmos3.sh "$WORK/cosmos3_env" > "$WORK/cosmos3.log" 2>&1 &
cosmos3_setup=$!
wait %1 %2 %3
bash scripts/setup_irasim.sh $IRASIM $DATA > "$WORK/setup_irasim.log" 2>&1
python scripts/select_episodes.py --data $DATA --out "$WORK/selection" --per-verb 6 --verbs $VERBS
python scripts/make_perturbed_annotations.py --data $DATA

# copy every finished video to Drive within a minute
( while true; do
    for name in $SETS; do
        mkdir -p "$OUT/frame_ada_$name/seed3407"
        find $IRASIM/IRASim/results -path "*bridge_frame_ada_${name}_seed3407-debug*" -name '*.mp4' \
            -exec cp -n {} "$OUT/frame_ada_$name/seed3407/" \; 2> /dev/null || true
    done
    sleep 60
  done ) &
copier=$!

wait $cosmos3_setup || echo "Cosmos3 server did not start: EVID is scored later"
clips=$(find "$OUT/real" -name '*.mp4' | wc -l)
for name in $SETS; do
    done_videos=$(find "$OUT/frame_ada_$name/seed3407" -name '*.mp4' 2> /dev/null | wc -l)
    [ "$done_videos" -ge "$clips" ] || python -u scripts/run_irasim.py --irasim $IRASIM --model frame_ada --seed 3407 --out "$OUT" --batch "$BATCH" \
        --annotation "annotation_$name" --name "frame_ada_$name" || true
    curl -sf http://localhost:8000/health > /dev/null && \
        python -u scripts/cosmos3_idm.py --gen "$OUT" --models "frame_ada_$name/seed3407" > "$WORK/idm_$name.log" 2>&1 &
done
wait $(jobs -p | grep -v "^$copier$") 2> /dev/null || true
kill $copier || true
sets=$(for name in $SETS; do printf 'frame_ada_%s/seed3407 ' $name; done)
python scripts/cosmos3_idm.py --gen "$OUT" --models $sets || echo "EVID incomplete: see the Cosmos3 log"
pkill -f "[v]llm serve" || true
python scripts/eval_metrics.py --gen "$OUT" --models $sets
python scripts/eval_motion.py --gen "$OUT" --tracker cotracker --models $sets
cd .. && python scripts/data_sync.py push --only 'video_eval/**' --message "A100: perturbation test on 105 clips"
echo "A100 JOB EXIT 0"
