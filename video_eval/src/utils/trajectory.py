"""
End-effector trajectory helpers for EVID (src/metrics/evid.py): actions as IRASim computes them from recorded poses,
their integration back into a path, and the 3-D scores of a path read from a video against the recorded one.

Actions follow IRASim (and Cosmos): between frames t-1 and t, the position change in the frame of the end effector at
t-1, R_{t-1}^T (p_t - p_{t-1}), the rotation change as Euler angles of R_{t-1}^T R_t (R = Rz Ry Rx, scipy 'xyz'), and
the gripper state at t; arm parts x 20.

Author: Duc Anh Hoang
"""

import csv
import os

import numpy as np
from scipy.spatial.transform import Rotation

SCALE = np.array([20.0] * 6 + [1.0])
D_TH = 0.02  # metres
FIELDS = ("pos_err_cm", "final_err_cm", "ndtw_3d", "path_ratio_3d", "action_err")


def actions_from_states(state: np.ndarray, gripper: np.ndarray) -> np.ndarray:
    """[T-1, 7] scaled actions from [T, 6] poses (xyz, rpy) and [T] gripper states, as IRASim computes them."""
    rot = Rotation.from_euler("xyz", state[:, 3:6]).as_matrix()
    out = np.zeros((len(state) - 1, 7))
    for k in range(1, len(state)):
        out[k - 1, :3] = rot[k - 1].T @ (state[k, :3] - state[k - 1, :3])
        out[k - 1, 3:6] = Rotation.from_matrix(rot[k - 1].T @ rot[k]).as_euler("xyz")
        out[k - 1, 6] = gripper[k]
    return out * SCALE


def integrate(first: np.ndarray, actions: np.ndarray) -> np.ndarray:
    """[T, 3] positions from the first pose (xyz, rpy) and [T-1, 7] scaled actions."""
    actions = actions / SCALE
    position, rot = first[:3].copy(), Rotation.from_euler("xyz", first[3:6]).as_matrix()
    path = [position.copy()]
    for a in actions:
        position = position + rot @ a[:3]
        rot = rot @ Rotation.from_euler("xyz", a[3:6]).as_matrix()
        path.append(position.copy())
    return np.array(path)


def ndtw(p: np.ndarray, q: np.ndarray) -> float:
    cost = np.linalg.norm(p[:, None] - q[None], axis=2)
    acc = np.full((len(p) + 1, len(q) + 1), np.inf)
    acc[0, 0] = 0
    for i in range(1, len(p) + 1):
        for j in range(1, len(q) + 1):
            acc[i, j] = cost[i - 1, j - 1] + min(acc[i - 1, j], acc[i, j - 1], acc[i - 1, j - 1])
    return float(np.exp(-acc[-1, -1] / (len(q) * D_TH)))


def trajectory_scores(pred: np.ndarray, state: np.ndarray, gripper: np.ndarray) -> tuple[dict, np.ndarray]:
    """3-D scores of recovered actions [15, 7] against a clip's recorded poses [16, 6]; also the recovered path."""
    given = actions_from_states(state, gripper)
    path_pred, path_real = integrate(state[0], pred), state[:, :3]
    error = np.linalg.norm(path_pred - path_real, axis=1)
    length = lambda p: np.linalg.norm(np.diff(p, axis=0), axis=1).sum()  # noqa: E731
    return {
        "pos_err_cm": 100 * error[1:].mean(), "final_err_cm": 100 * error[-1], "ndtw_3d": ndtw(path_pred, path_real),
        "path_ratio_3d": length(path_pred) / max(length(path_real), 1e-6),
        "action_err": float(np.abs(pred - given)[:, :6].mean()),
    }, path_pred  # fmt: skip


def write_rows(out: str, rows: list[dict], models: list[str]) -> None:
    """Replace the rows of the given models in a metrics CSV and print their means."""
    kept = []
    if os.path.exists(out):
        with open(out) as f:
            kept = [r for r in csv.DictReader(f) if r["model"] not in models]
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "clip", *FIELDS])
        writer.writeheader()
        writer.writerows(kept + rows)
    for name in models:
        values = {k: np.array([r[k] for r in rows if r["model"] == name]) for k in FIELDS}
        print(f"{name:26s} clips {len(values['ndtw_3d']):3d}  "
              + "  ".join(f"{k} {v.mean():.3f}" for k, v in values.items()))  # fmt: skip
