"""
Choose the Bridge test episodes to generate: the first <PER_VERB> episodes (by id) whose instruction starts with each
verb, so the set covers different manipulations (pick-and-place, open, close, flip, fold, wipe, ...).

Usage (from video_eval/):
    python scripts/select_episodes.py --data <DATA_DIR> --out <OUT_DIR> [--per-verb 2] [--verbs put move ...]

<DATA_DIR> holds opensource_robotdata/ (scripts/fetch_irasim.py). The chosen annotation files are linked into
opensource_robotdata/bridge/annotation_subset/test/, where IRASim reads them (annotation_name: annotation_subset), and
listed in <OUT_DIR>/episodes.csv (episode, frames, verb, instruction).

Author: Duc Anh Hoang
"""

import argparse
import csv
import glob
import json
import os
import shutil

VERBS = ("put", "move", "take", "open", "close", "flip", "fold", "wipe", "turn", "pour", "slide", "topple")
MIN_FRAMES = 17  # one 16-frame clip after the start frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--per-verb", type=int, default=2)
    parser.add_argument("--verbs", nargs="+", default=VERBS)
    args = parser.parse_args()

    bridge = os.path.join(args.data, "opensource_robotdata", "bridge")
    episodes = []
    for path in glob.glob(os.path.join(bridge, "annotation", "test", "*.json")):
        with open(path) as f:
            label = json.load(f)
        episodes.append((int(label["episode_id"]), len(label["state"]), label["texts"][0].strip(), path))
    episodes.sort()

    chosen = []
    for verb in args.verbs:
        matches = [e for e in episodes if e[2].lower().split()[:1] == [verb] and e[1] >= MIN_FRAMES]
        chosen += [(*e, verb) for e in matches[: args.per_verb]]

    subset = os.path.join(bridge, "annotation_subset", "test")
    shutil.rmtree(subset, ignore_errors=True)
    os.makedirs(subset)
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "episodes.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "frames", "verb", "instruction"])
        for episode, frames, text, path, verb in chosen:
            os.symlink(path, os.path.join(subset, os.path.basename(path)))
            writer.writerow([episode, frames, verb, text])
    print(f"{len(chosen)} episodes ({len(args.verbs)} verbs) -> {subset} and {args.out}/episodes.csv")


if __name__ == "__main__":
    main()
