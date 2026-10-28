"""
Judge physical plausibility and task outcome of generated clips with vision-language models (two or more judges, so
their agreement can be reported).

Usage (on the machine that reaches the endpoint, from video_eval/):
    python scripts/eval_vlm.py --gen <GEN_DIR> --models <MODEL_DIR> [...] --judges <MODEL> <MODEL>
        [--base-url URL] [--key-env NAME] [--workers 8]
    e.g. python scripts/eval_vlm.py --gen <GEN_DIR> --models real still frame_ada/seed3407 \\
             --judges ag/gemini-3.8-flash-low cx/gpt-6.1-sol

Two questions per clip and judge:
- physics: frames 0, 5, 10 and 15 in a 2 x 2 grid; the judge flags objects moving without contact, interpenetration,
  objects changing shape or identity or vanishing, and the robot arm deforming or vanishing, then gives an overall
  plausible yes or no. The real clips are the control (the share of real clips flagged = the judge's false alarms).
- outcome: the real clip's last frame beside the generated clip's last frame (same first frame, same actions; the
  instruction is given when the annotation has one); the judge says whether both show the same task state (gripper
  pose relative to the objects, gripper open or closed, positions of moved objects). A clip is a 16-frame window, so
  "the same state as the real clip" is used instead of "task completed". The still clips are the control: they should
  differ whenever the real robot moved.
Writes <GEN_DIR>/metrics_vlm.csv (one row per model, clip and judge; rows of the given models are replaced) and prints
per model the share judged plausible, the violation shares, the share with the real outcome, and judge agreement.
What it cannot establish: the judges can hallucinate and change answers between runs (seen on 2026-10-28); a "yes"
is not a physical guarantee, only the absence of a violation the judge noticed in four frames.

Author: Duc Anh Hoang
"""

import argparse
import csv
import glob
import os
from concurrent.futures import ThreadPoolExecutor

import imageio.v2 as imageio
import numpy as np
from vlm import api_key, ask, png

VIOLATIONS = (
    "object_moves_without_contact",
    "interpenetration",
    "object_changes_or_vanishes",
    "robot_deforms_or_vanishes",
)
PHYSICS = (
    "The image shows four frames of one robot video in a 2 x 2 grid (top left frame 0, top right frame 5, bottom left "
    "frame 10, bottom right frame 15). Judge only physical plausibility, not image quality. Answer JSON only: "
    '{"object_moves_without_contact": <true if an object moves while nothing touches it>, '
    '"interpenetration": <true if solid things pass through each other>, '
    '"object_changes_or_vanishes": <true if an object changes shape, identity or colour, appears or vanishes>, '
    '"robot_deforms_or_vanishes": <true if the robot arm or gripper bends unnaturally, breaks apart or vanishes>, '
    '"plausible": <true if the clip is physically plausible overall>, "reason": "<one short sentence>"}'
)
OUTCOME = (
    "Left: the true last frame of a short robot clip. Right: the last frame of a generated clip that started from the "
    "same first frame and received the same robot commands.{instruction} Do both images show the same task state "
    "(gripper position relative to the objects, gripper open or closed, positions of objects that were moved)? Ignore "
    'image quality and small offsets. Answer JSON only: {{"same_outcome": true or false, '
    '"difference": "<the main difference, or none>"}}'
)
FIELDS = (*VIOLATIONS, "plausible", "same_outcome")


def frames(path: str) -> np.ndarray:
    return np.stack(imageio.mimread(path))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gen", required=True)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--judges", nargs="+", required=True)
    parser.add_argument("--base-url", default="http://localhost:20128/v1")
    parser.add_argument("--key-env", default="GEMINI_API_KEY")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    key = api_key(args.key_env)

    jobs = {}
    with ThreadPoolExecutor(args.workers) as pool:
        for path in sorted(glob.glob(os.path.join(args.gen, "real", "*.mp4"))):
            clip = os.path.basename(path)[:-4]
            real = frames(path)
            states = os.path.join(args.gen, "states", f"{clip}.npz")
            text = str(np.load(states)["instruction"]) if os.path.exists(states) else ""
            question = OUTCOME.format(instruction=f' The instruction was: "{text}".' if text else "")
            for model in args.models:
                path = os.path.join(args.gen, model, f"{clip}.mp4")
                if not os.path.exists(path):
                    continue
                video = frames(path)
                grid = np.concatenate([np.concatenate(video[[0, 5]], 1), np.concatenate(video[[10, 15]], 1)], 0)
                pair = np.concatenate([real[-1], video[-1]], 1)
                for judge in args.judges:
                    jobs[(model, clip, judge, "physics")] = pool.submit(
                        ask, args.base_url, key, judge, PHYSICS, png(grid)
                    )
                    jobs[(model, clip, judge, "outcome")] = pool.submit(
                        ask, args.base_url, key, judge, question, png(pair)
                    )
        answers = {k: job.result() for k, job in jobs.items()}

    rows = {}
    for (model, clip, judge, _), answer in answers.items():
        row = rows.setdefault((model, clip, judge), {"model": model, "clip": clip, "judge": judge})
        for field in FIELDS:
            if field in answer:
                row[field] = answer[field] is True
        row.setdefault("notes", "")
        row["notes"] += str(answer.get("reason") or answer.get("difference") or answer.get("error") or "") + " | "

    out = os.path.join(args.gen, "metrics_vlm.csv")
    kept = []
    if os.path.exists(out):
        with open(out) as f:
            kept = [r for r in csv.DictReader(f) if r["model"] not in args.models]
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "clip", "judge", *FIELDS, "notes"])
        writer.writeheader()
        writer.writerows(kept + list(rows.values()))

    for model in args.models:
        mine = [r for r in rows.values() if r["model"] == model]
        shares = {f: np.mean([r[f] for r in mine if f in r]) for f in FIELDS if any(f in r for r in mine)}
        per_clip = {}
        for r in mine:
            per_clip.setdefault(r["clip"], []).append((r.get("plausible"), r.get("same_outcome")))
        agree = np.mean([len({v[0] for v in votes}) == 1 for votes in per_clip.values()])
        print(f"{model:22s} clips {len(per_clip):3d}  " + "  ".join(f"{f} {v:.2f}" for f, v in shares.items())
              + f"  judges agree on plausible {agree:.2f}")  # fmt: skip


if __name__ == "__main__":
    main()
