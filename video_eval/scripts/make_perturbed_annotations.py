"""
Corrupt the action input of the chosen clips on purpose, to test whether a metric notices: a metric of "the robot
follows the given actions" must drop as the actions given to the model drift from those of the real clip.

Usage (from video_eval/, after scripts/select_episodes.py):
    python scripts/make_perturbed_annotations.py --data <DATA_DIR> [--seed 0]

IRASim computes its actions from the end-effector states of an annotation (deltas between frames), so the states are
changed, position (x, y, z, metres) only; orientation, gripper and frame 0 stay as recorded. For every episode in
annotation_subset/test it writes opensource_robotdata/bridge/annotation_<NAME>/test/<episode>.json for each NAME:
    noise_s  position plus a random walk of 0.5 cm per frame (drift of about 2 cm after 15 frames)
    noise_l  the same with 1.5 cm per frame (about 6 cm)
    mirror   the motion reversed in direction: s'_t = s_0 - (s_t - s_0)
    freeze   every state equal to the first one: all actions zero
Generate them with scripts/run_irasim.py --annotation annotation_<NAME>, then score against the unchanged real clips.

Author: Duc Anh Hoang
"""

import argparse
import glob
import json
import os

import numpy as np

STEP_STD = {"noise_s": 0.005, "noise_l": 0.015}  # metres per frame


def perturb(state: np.ndarray, name: str, rng: np.random.Generator) -> np.ndarray:
    out = state.copy()
    position = state[:, :3]
    if name in STEP_STD:
        walk = np.cumsum(rng.normal(0, STEP_STD[name], position.shape), axis=0)
        out[:, :3] = position + walk - walk[0]  # frame 0 unchanged
    elif name == "mirror":
        out[:, :3] = 2 * position[:1] - position
    elif name == "freeze":
        out[:, :3] = position[:1]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    bridge = os.path.join(args.data, "opensource_robotdata", "bridge")
    paths = sorted(glob.glob(os.path.join(bridge, "annotation_subset", "test", "*.json")))

    for name in ("noise_s", "noise_l", "mirror", "freeze"):
        rng = np.random.default_rng(args.seed)
        folder = os.path.join(bridge, f"annotation_{name}", "test")
        os.makedirs(folder, exist_ok=True)
        for path in paths:
            with open(path) as f:
                label = json.load(f)
            label["state"] = perturb(np.array(label["state"]), name, rng).tolist()
            with open(os.path.join(folder, os.path.basename(path)), "w") as f:
                json.dump(label, f)
        print(f"{name}: {len(paths)} episodes -> {folder}")


if __name__ == "__main__":
    main()
