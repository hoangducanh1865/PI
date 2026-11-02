"""
Choose the test episodes to generate. Bridge: the first <PER_VERB> episodes (by id) whose instruction starts with each
verb, so the set covers different manipulations (pick-and-place, open, close, flip, fold, wipe, ...). Language-Table:
its annotations carry no instruction, so the first <COUNT> episodes by id. --random N instead draws N test episodes at
random (to compare with scores on the whole test set); --from-csv reuses the episodes of an earlier episodes.csv.

Usage (from video_eval/):
    python scripts/data/select_episodes.py --data <DATA_DIR> --out <OUT_DIR> [--per-verb 2] [--verbs put move ...]
    python scripts/data/select_episodes.py --data <DATA_DIR> --out <OUT_DIR> --dataset languagetable [--count 48]
    python scripts/data/select_episodes.py --data <DATA_DIR> --out <OUT_DIR> --random 40 [--random-seed 0]
    python scripts/data/select_episodes.py --data <DATA_DIR> --out <OUT_DIR> --from-csv <EARLIER_OUT_DIR>/episodes.csv

<DATA_DIR> holds opensource_robotdata/ (scripts/data/fetch_irasim.py). The chosen annotation files are linked into
opensource_robotdata/<dataset>/annotation_subset/test/, where IRASim reads them (annotation_name: annotation_subset),
and listed in <OUT_DIR>/episodes.csv (episode, frames, verb, instruction).

Author: Duc Anh Hoang
"""

import argparse
import csv
import glob
import json
import os
import random
import shutil

VERBS = ("put", "move", "take", "open", "close", "flip", "fold", "wipe", "turn", "pour", "slide", "topple")
MIN_FRAMES = 17  # one 16-frame clip after the start frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--dataset", choices=("bridge", "languagetable"), default="bridge")
    parser.add_argument("--per-verb", type=int, default=2, help="Bridge: episodes per verb")
    parser.add_argument("--verbs", nargs="+", default=VERBS, help="Bridge: instruction verbs")
    parser.add_argument("--count", type=int, default=48, help="Language-Table: number of episodes")
    parser.add_argument("--random", type=int, help="draw this many test episodes at random instead")
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--from-csv", help="reuse the episodes listed in this episodes.csv instead")
    args = parser.parse_args()

    root = os.path.join(args.data, "opensource_robotdata", args.dataset)
    episodes = []
    for path in glob.glob(os.path.join(root, "annotation", "test", "*.json")):
        with open(path) as f:
            label = json.load(f)
        frames = len(label["state"] if args.dataset == "bridge" else label["actions"])
        text = label["texts"][0].strip() if args.dataset == "bridge" else ""
        episodes.append((label["episode_id"], frames, text, path))
    episodes = sorted((e for e in episodes if e[1] >= MIN_FRAMES), key=lambda e: int(e[0]))  # ids may be zero-padded

    if args.from_csv:
        with open(args.from_csv) as f:
            wanted = [row["episode"] for row in csv.DictReader(f)]
        by_id = {str(int(e[0])): e for e in episodes}
        chosen = [(*by_id[str(int(w))], "") for w in wanted]
    elif args.random:
        chosen = [(*e, "") for e in sorted(random.Random(args.random_seed).sample(episodes, args.random),
                                           key=lambda e: int(e[0]))]  # fmt: skip
    elif args.dataset == "bridge":
        chosen = []
        for verb in args.verbs:
            matches = [e for e in episodes if e[2].lower().split()[:1] == [verb]]
            chosen += [(*e, verb) for e in matches[: args.per_verb]]
    else:
        chosen = [(*e, "") for e in episodes[: args.count]]

    subset = os.path.join(root, "annotation_subset", "test")
    shutil.rmtree(subset, ignore_errors=True)
    os.makedirs(subset)
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "episodes.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "frames", "verb", "instruction"])
        for episode, frames, text, path, verb in chosen:
            os.symlink(path, os.path.join(subset, os.path.basename(path)))
            writer.writerow([episode, frames, verb, text])
    print(f"{len(chosen)} {args.dataset} episodes -> {subset} and {args.out}/episodes.csv")


if __name__ == "__main__":
    main()
