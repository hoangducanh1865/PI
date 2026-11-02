#!/usr/bin/env bash
# More IRASim-Frame-Ada seeds on the 105 Bridge clips, on one A100, for best-of-N selection by EVID: generate each
# seed, score EVID (Cosmos3-Nano inverse dynamics), pixel and 2-D track scores, and push the results. Videos are copied
# to <OUT_DIR> (on Drive) within a minute, and a rerun skips complete seeds and cached inverse dynamics answers.
#
# Usage (from video_eval/, HF_TOKEN and HF_WRITE_TOKEN in the environment):
#   bash scripts/jobs/a100_seeds.sh <OUT_DIR> [SEED ...]       (default seeds 3409 3410 3411 3412)
# Large downloads go to $WORK (default /content/work); BATCH clips per IRASim batch (default 12).
#
# Author: Duc Anh Hoang
set -euo pipefail
OUT=$1
shift
SEEDS=${*:-3409 3410 3411 3412}
WORK=${WORK:-/content/work}
DATA=$WORK/irasim
IRASIM=$WORK/IRASim
BATCH=${BATCH:-12}
VERBS="put move take open close flip fold wipe turn pour slide topple"
mkdir -p "$OUT" "$WORK"

pip install -q -r requirements.txt decord > "$WORK/pip.log" 2>&1 &
python -u scripts/data/fetch_irasim.py get bridge_checkpoints_data --out $DATA --match 'frame_ada/.*[.]pt$' > "$WORK/fetch_ckpt.log" 2>&1 &
python -u scripts/data/fetch_irasim.py get bridge_train_data --out $DATA --match '/test/' > "$WORK/fetch_test.log" 2>&1 &
bash scripts/setup/cosmos3.sh "$WORK/cosmos3_env" > "$WORK/cosmos3.log" 2>&1 &
cosmos3_setup=$!
wait %1 %2 %3
bash scripts/setup/irasim.sh $IRASIM $DATA > "$WORK/setup_irasim.log" 2>&1
python scripts/data/select_episodes.py --data $DATA --out "$WORK/selection" --per-verb 6 --verbs $VERBS

( while true; do
    for seed in $SEEDS; do
        mkdir -p "$OUT/frame_ada/seed$seed"
        find $IRASIM/IRASim/results -path "*bridge_frame_ada_seed${seed}-debug*" -name '*.mp4' \
            -exec cp -n {} "$OUT/frame_ada/seed$seed/" \; 2> /dev/null || true
    done
    sleep 60
  done ) &
copier=$!

wait $cosmos3_setup || echo "Cosmos3 server did not start: EVID is scored later"
clips=$(find "$OUT/real" -name '*.mp4' | wc -l)
for seed in $SEEDS; do
    done_videos=$(find "$OUT/frame_ada/seed$seed" -name '*.mp4' 2> /dev/null | wc -l)
    [ "$done_videos" -ge "$clips" ] || python -u src/generate/irasim.py --irasim $IRASIM --model frame_ada --seed "$seed" \
        --out "$OUT" --batch "$BATCH" || true
    curl -sf http://localhost:8000/health > /dev/null && \
        python -u src/metrics/evid.py --gen "$OUT" --models "frame_ada/seed$seed" > "$WORK/idm_$seed.log" 2>&1 &
done
wait $(jobs -p | grep -v "^$copier$") 2> /dev/null || true
kill $copier || true
sets=$(for seed in $SEEDS; do printf 'frame_ada/seed%s ' $seed; done)
python src/metrics/evid.py --gen "$OUT" --models $sets || echo "EVID incomplete: see the Cosmos3 log"
pkill -f "[v]llm serve" || true
python src/metrics/pixel.py --gen "$OUT" --models $sets
python src/metrics/track2d.py --gen "$OUT" --tracker cotracker --models $sets
cd .. && python scripts/data_sync.py push --only 'video_eval/**' --message "A100: Frame-Ada seeds $SEEDS on 105 clips"
echo "A100 JOB EXIT 0"
