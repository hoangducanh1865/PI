"""
Corrupt the action input of the chosen clips on purpose, to test whether a metric notices: a metric of "the robot
follows the given actions" must drop as the actions given to the model drift from those of the real clip.

Usage (from video_eval/, after scripts/select_episodes.py):
    python scripts/make_perturbed_annotations.py --data <DATA_DIR> [--dataset languagetable] [--seed 0]

Bridge: IRASim computes its actions from the end-effector states of an annotation (deltas between frames), so the
states are changed, position (x, y, z, metres) only; orientation, gripper and frame 0 stay as recorded.
Language-Table: the annotation stores the 2-D actions themselves, which are changed directly; its units are not
metres, so the noise is scaled by the episode's mean step length. For every episode in annotation_subset/test it writes
opensource_robotdata/<dataset>/annotation_<NAME>/test/<episode>.json for each NAME:
    noise_s  Bridge: position plus a random walk of 0.5 cm per frame (drift of about 2 cm after 15 frames);
             Language-Table: each action plus noise of 0.5 x the mean step length
    noise_l  the same with 1.5 cm per frame (about 6 cm); Language-Table 1.5 x the mean step length
    mirror   the motion reversed in direction: s'_t = s_0 - (s_t - s_0); Language-Table: every action negated
    freeze   every state equal to the first one; Language-Table: every action zero
Generate them with scripts/run_irasim.py --annotation annotation_<NAME>, then score against the unchanged real clips.

Author: Duc Anh Hoang
"""

import argparse
import glob
import json
import os

import numpy as np

STEP_STD = {"noise_s": 0.005, "noise_l": 0.015}  # Bridge, metres per frame
STEP_SCALE = {"noise_s": 0.5, "noise_l": 1.5}  # Language-Table, times the mean step length


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


def perturb_actions(actions: np.ndarray, name: str, rng: np.random.Generator) -> np.ndarray:
    """Language-Table: the 2-D actions themselves."""
    if name in STEP_SCALE:
        step = np.linalg.norm(actions, axis=1).mean()
        return actions + rng.normal(0, STEP_SCALE[name] * step, actions.shape)
    return -actions if name == "mirror" else np.zeros_like(actions)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", required=True)
    parser.add_argument("--dataset", choices=("bridge", "languagetable"), default="bridge")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    root = os.path.join(args.data, "opensource_robotdata", args.dataset)
    paths = sorted(glob.glob(os.path.join(root, "annotation_subset", "test", "*.json")))

    for name in ("noise_s", "noise_l", "mirror", "freeze"):
        rng = np.random.default_rng(args.seed)
        folder = os.path.join(root, f"annotation_{name}", "test")
        os.makedirs(folder, exist_ok=True)
        for path in paths:
            with open(path) as f:
                label = json.load(f)
            if args.dataset == "bridge":
                label["state"] = perturb(np.array(label["state"]), name, rng).tolist()
            else:
                label["actions"] = perturb_actions(np.array(label["actions"]), name, rng).tolist()
            with open(os.path.join(folder, os.path.basename(path)), "w") as f:
                json.dump(label, f)
        print(f"{name}: {len(paths)} episodes -> {folder}")


if __name__ == "__main__":
    main()
