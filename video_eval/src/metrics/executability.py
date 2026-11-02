"""
Could the real arm execute the motion seen in a video? Solve the inverse kinematics of the WidowX 250 6-DOF arm
(Bridge's robot) for every end-effector pose that Cosmos3-Nano reads back from the video (src/metrics/evid.py), and
check the joint limits and joint speeds the official Interbotix driver enforces.

Usage (from video_eval/; needs the cached Cosmos3-Nano answers <GEN_DIR>/<MODEL>/cosmos3_idm/*.npz):
    python src/metrics/executability.py --gen <GEN_DIR> --models real still <MODEL_DIR> [...]

Robot and checks, all from Interbotix's own code and descriptions (no setting of ours):
- kinematics: product of exponentials with the wx250s screw axes Slist and home pose M of ee_gripper_link
  (interbotix_ros_toolboxes, interbotix_xs_modules/mr_descriptions.py, class wx250s);
- inverse kinematics as InterbotixArmXSInterface.set_ee_pose_matrix (interbotix_xs_modules/arm.py): modern_robotics
  IKinSpace with tolerances 0.001 rad and 0.001 m, initial guesses waist 0, -120 and +120 degrees, angles wrapped into
  the joint limits; we also try the previous frame's solution first, as a custom guess. Our one addition: when those
  guesses fail, a grid of 125 guesses (shoulder, elbow, wrist angle in {-1.5, -0.75, 0, 0.75, 1.5} rad, waist towards
  the target), same tolerances and limits. Without it 4 of 105 recorded Bridge paths fail (low, steeply pitched poses
  where Newton's method from the three default guesses lands on a branch outside the limits); with it all pass. When
  the first solution found needs a step above the velocity limit, the solution closest to the previous frame's is
  taken among all guesses; and when a clip still fails, every solution branch of its first pose is tried as the start
  of the whole trajectory (a branch chosen at frame 0 can lead a roll joint into its +-pi limit later, which the real
  arm avoids by starting on another branch);
- joint limits and velocity limit pi rad/s from wx250s.urdf.xacro (interbotix_xsarm_descriptions), which match the
  motor limits in interbotix_xsarm_control/config/wx250s.yaml;
- joint speed as check_joint_limits computes it, |delta theta| / moving_time, with moving_time = 0.2 s, one step of
  Bridge's 5 Hz control.
Poses: the state pose mapped to the frame of ee_gripper_link exactly as in cosmos3_idm.decode (roll + pi, then
_DEFAULT_ROTATION). The model "commanded" checks the recorded states themselves: they come from the real arm, so they
must all pass, which validates the frame convention.
Writes <GEN_DIR>/metrics_executability.csv (one row per model and clip: share of poses solved, share of steps over the
speed limit, maximum joint speed, executable = every pose solved within limits and every step within the speed limit)
and prints the mean per model. What it cannot establish: collisions with the table or objects, torque limits, and
whether the controller tracks the motion.

Author: Duc Anh Hoang
"""

import argparse
import csv
import glob
import os
import sys

import modern_robotics as mr
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # video_eval/
sys.path.insert(0, ROOT)  # src.* importable when run as a script

from src.metrics.evid import TO_KINEMATICS, decode  # noqa: E402
from scipy.spatial.transform import Rotation

SLIST = np.array([[0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                  [0.0, 1.0, 0.0, -0.11065, 0.0, 0.0],
                  [0.0, 1.0, 0.0, -0.36065, 0.0, 0.04975],
                  [1.0, 0.0, 0.0, 0.0, 0.36065, 0.0],
                  [0.0, 1.0, 0.0, -0.36065, 0.0, 0.29975],
                  [1.0, 0.0, 0.0, 0.0, 0.36065, 0.0]]).T  # fmt: skip
M = np.array([[1.0, 0.0, 0.0, 0.458325], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.36065], [0.0, 0.0, 0.0, 1.0]])
LOWER = np.array([-np.pi + 1e-5, np.radians(-108), np.radians(-123), -np.pi + 1e-5, np.radians(-100), -np.pi + 1e-5])
UPPER = np.array([np.pi - 1e-5, np.radians(114), np.radians(92), np.pi - 1e-5, np.radians(123), np.pi - 1e-5])
VELOCITY = np.pi  # rad/s, every arm joint
STEP = 0.2  # s, Bridge's 5 Hz control
GUESSES = [np.zeros(6), np.r_[np.radians(-120), np.zeros(5)], np.r_[np.radians(120), np.zeros(5)]]
GRID = [np.array([0.0, s, e, 0.0, w, 0.0]) for s in (-1.5, -0.75, 0.0, 0.75, 1.5) for e in (-1.5, -0.75, 0.0, 0.75, 1.5)
        for w in (-1.5, -0.75, 0.0, 0.75, 1.5)]  # fmt: skip


def wrap(theta: np.ndarray) -> np.ndarray:
    """Bring angles into the joint limits by whole turns, as set_ee_pose_matrix does."""
    theta = np.fmod(theta, 2 * np.pi)
    theta = np.where(np.round(theta, 3) < np.round(LOWER, 3), theta + 2 * np.pi, theta)
    return np.where(np.round(theta, 3) > np.round(UPPER, 3), theta - 2 * np.pi, theta)


def within(theta: np.ndarray) -> bool:
    return bool(np.all((np.round(theta, 3) >= LOWER - 1e-9) & (np.round(theta, 3) <= UPPER + 1e-9)))


def solve(target: np.ndarray, previous: np.ndarray | None) -> np.ndarray | None:
    """Joint angles reaching the 4 x 4 pose within the joint limits, or None. If the first solution found would need a
    step faster than the velocity limit, the closest solution to the previous frame's is taken among all guesses."""
    aimed = [np.r_[np.arctan2(target[1, 3], target[0, 3]), g[1:]] for g in GRID]
    guesses = ([previous] if previous is not None else []) + GUESSES + aimed
    found = None
    for guess in guesses:
        theta, ok = mr.IKinSpace(SLIST, M, target, guess, 0.001, 0.001)
        if ok and within(wrap(theta)):
            found = wrap(theta)
            break
    if found is None or previous is None or np.abs(found - previous).max() / STEP <= VELOCITY:
        return found
    for guess in guesses:
        theta, ok = mr.IKinSpace(SLIST, M, target, guess, 0.001, 0.001)
        if ok and within(wrap(theta)) and np.abs(wrap(theta) - previous).max() < np.abs(found - previous).max():
            found = wrap(theta)
    return found


def chain(targets: list, start: np.ndarray | None) -> dict:
    """Solve the poses in order, each from the previous solution, and check the speeds between them."""
    thetas, previous = [], start
    for target in targets:
        previous = solve(target, previous)
        thetas.append(previous)
    solved = [t is not None for t in thetas]
    pairs = zip(thetas[:-1], thetas[1:], strict=True)
    speeds = [np.abs(b - a).max() / STEP for a, b in pairs if a is not None and b is not None]
    return {"solved": float(np.mean(solved)),
            "over_speed": float(np.mean([s > VELOCITY for s in speeds])) if speeds else float("nan"),
            "max_speed": float(max(speeds)) if speeds else float("nan"),
            "executable": float(all(solved) and all(s <= VELOCITY for s in speeds))}  # fmt: skip


def check(poses: np.ndarray) -> dict:
    """Executability of one clip ([16, 6] poses in IRASim's state convention). If the first solution of the first pose
    leads to a violation, every other solution branch of the first pose is tried and the best trajectory is kept."""
    targets = []
    for pose in poses:
        target = np.eye(4)
        target[:3, :3], target[:3, 3] = Rotation.from_euler("xyz", pose[3:6]).as_matrix() @ TO_KINEMATICS, pose[:3]
        targets.append(target)
    best = chain(targets, None)
    if best["executable"] or best["solved"] < 1:  # a pose no starting guess solves stays unsolved on every branch
        return best
    aimed = [np.r_[np.arctan2(targets[0][1, 3], targets[0][0, 3]), g[1:]] for g in GRID]
    branches = {}
    for guess in GUESSES + aimed:
        theta, ok = mr.IKinSpace(SLIST, M, targets[0], guess, 0.001, 0.001)
        if ok and within(wrap(theta)):
            branches.setdefault(tuple(np.round(wrap(theta), 2)), wrap(theta))
    for start in branches.values():
        result = chain(targets, start)
        key = (result["executable"], result["solved"], -result["over_speed"])
        if key > (best["executable"], best["solved"], -best["over_speed"]):
            best = result
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gen", required=True)
    parser.add_argument("--models", nargs="+", required=True)
    args = parser.parse_args()

    rows = []
    for path in sorted(glob.glob(os.path.join(args.gen, "states", "*.npz"))):
        clip = os.path.basename(path)[:-4]
        state = np.load(path)["state"]
        rows.append({"model": "commanded", "clip": clip, **check(state)})
        for name in args.models:
            cached = os.path.join(args.gen, name, "cosmos3_idm", f"{clip}.npz")
            if os.path.exists(cached):
                rows.append({"model": name, "clip": clip, **check(decode(np.load(cached)["raw"], state[0]))})
    with open(os.path.join(args.gen, "metrics_executability.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for name in ["commanded"] + args.models:
        mine = [r for r in rows if r["model"] == name]
        if mine:
            mean = {k: np.nanmean([r[k] for r in mine]) for k in ("solved", "over_speed", "max_speed", "executable")}
            print(f"{name:22s} clips {len(mine):3d}  poses solved {mean['solved']:.3f}  steps over speed "
                  f"{mean['over_speed']:.3f}  max speed {mean['max_speed']:.2f} rad/s  "
                  f"executable {mean['executable']:.3f}")  # fmt: skip


if __name__ == "__main__":
    main()
