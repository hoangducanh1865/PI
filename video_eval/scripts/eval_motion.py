"""
Does the robot in a generated clip move like the robot in the real clip of the same actions? Track points through
both videos, find the point that rides on the end effector, and compare its two image trajectories.

Usage (from video_eval/):
    python scripts/eval_motion.py --gen <GEN_DIR> --models <MODEL_DIR> [<MODEL_DIR> ...] [--wandb <PROJECT>]
    e.g. python scripts/eval_motion.py --gen <GEN_DIR> --models real still frame_ada/seed3407

Steps, per clip (all clips of a model share frame 0, the given frame, so the same points can be followed in both):
1. Track a 20 x 20 grid of points from frame 0 through the 16 frames of the real and of the generated video
   (pyramidal Lucas-Kanade optical flow, OpenCV, frame to frame; a point is lost when the forward-backward check fails
   by more than 2 px, and its track then stays at its last position).
2. End-effector point: among the points that move more than 15 px in the real video and are never lost, the one whose
   real track is best explained by the recorded end-effector position through an affine camera (least squares from
   [x, y, z, 1] to the 2-D track over the 16 frames), by R^2 = the share of the track's variance the fit explains
   (a residual in pixels would favour points that barely move). R^2 and the real path length are kept as checks.
3. Compare the generated and the real track of that point: NDTW = exp(-DTW / (16 * d_th)) with d_th = 20 px (about
   5% of the image diagonal; Ilharco et al., 2019, use the success radius), the mean and the final pixel error, and the
   path-length ratio (generated / real: 0 for a video in which nothing moves). moving_ndtw averages NDTW over every
   moving point (robot and moved objects).
Writes <GEN_DIR>/metrics_motion.csv (one row per model and clip; rows of the given models are replaced) and prints
the mean per model. What it cannot establish: whether the motion is physically executable, or right for points that
the tracker loses (blur, occlusion); the end-effector point is an image point, not the gripper's true pose.

Author: Duc Anh Hoang
"""

import argparse
import csv
import glob
import os

import cv2
import imageio.v2 as imageio
import numpy as np

GRID = 20
MOVE_PX = 5.0  # a point moves: largest displacement from frame 0
EE_MOVE_PX = 15.0  # an end-effector candidate moves at least this much
D_TH = 20.0
LK = {"winSize": (21, 21), "maxLevel": 3, "criteria": (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)}
FIELDS = ("ee_ndtw", "ee_err_px", "ee_final_px", "path_ratio", "moving_ndtw", "ee_fit_r2", "ee_real_path_px")


def grid_points(height: int, width: int) -> np.ndarray:
    ys = np.linspace(0.05 * height, 0.95 * height, GRID)
    xs = np.linspace(0.05 * width, 0.95 * width, GRID)
    return np.array([[x, y] for y in ys for x in xs], dtype=np.float32)


def track(video: np.ndarray, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Tracks [T, N, 2] and a kept mask [N] (never lost) of the points through the video."""
    gray = [cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) for frame in video]
    tracks, kept = [points.copy()], np.ones(len(points), bool)
    for prev, nxt in zip(gray[:-1], gray[1:], strict=True):
        p0 = tracks[-1].reshape(-1, 1, 2)
        p1, ok1, _ = cv2.calcOpticalFlowPyrLK(prev, nxt, p0, None, **LK)
        back, ok2, _ = cv2.calcOpticalFlowPyrLK(nxt, prev, p1, None, **LK)
        good = (ok1[:, 0] == 1) & (ok2[:, 0] == 1) & (np.linalg.norm(back - p0, axis=2)[:, 0] < 2.0)
        kept &= good
        tracks.append(np.where(good[:, None], p1[:, 0], tracks[-1]))
    return np.stack(tracks), kept


def ndtw(p: np.ndarray, q: np.ndarray) -> float:
    cost = np.linalg.norm(p[:, None] - q[None], axis=2)
    acc = np.full((len(p) + 1, len(q) + 1), np.inf)
    acc[0, 0] = 0
    for i in range(1, len(p) + 1):
        for j in range(1, len(q) + 1):
            acc[i, j] = cost[i - 1, j - 1] + min(acc[i - 1, j], acc[i, j - 1], acc[i - 1, j - 1])
    return float(np.exp(-acc[-1, -1] / (len(q) * D_TH)))


def path_length(track_2d: np.ndarray) -> float:
    return float(np.linalg.norm(np.diff(track_2d, axis=0), axis=1).sum())


def end_effector_point(real_tracks: np.ndarray, kept: np.ndarray, state: np.ndarray) -> tuple[int, float, np.ndarray]:
    """Index of the end-effector point (-1 if none), its affine-fit R^2, and the moving-point mask."""
    displacement = np.linalg.norm(real_tracks - real_tracks[:1], axis=2).max(axis=0)
    moving = kept & (displacement > MOVE_PX)
    design = np.c_[state[:, :3], np.ones(len(state))]
    best, best_r2 = -1, -np.inf
    for i in np.flatnonzero(moving & (displacement > EE_MOVE_PX)):
        target = real_tracks[:, i]
        coef, *_ = np.linalg.lstsq(design, target, rcond=None)
        r2 = 1 - np.sum((design @ coef - target) ** 2) / np.sum((target - target.mean(axis=0)) ** 2)
        if r2 > best_r2:
            best, best_r2 = i, float(r2)
    return best, best_r2, moving


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gen", required=True)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--wandb", help="W&B project to log the tables to")
    args = parser.parse_args()

    reference = {}  # clip -> (real tracks, end-effector index, fit residual, moving mask, points)
    for path in sorted(glob.glob(os.path.join(args.gen, "real", "*.mp4"))):
        clip = os.path.basename(path)[:-4]
        video = np.stack(imageio.mimread(path))
        points = grid_points(*video.shape[1:3])
        tracks, kept = track(video, points)
        state = np.load(os.path.join(args.gen, "states", f"{clip}.npz"))["state"]
        reference[clip] = (tracks, *end_effector_point(tracks, kept, state), points)
        print(f"{clip}: end-effector point {reference[clip][1]}, R^2 {reference[clip][2]:.3f}")

    rows = []
    for model in args.models:
        for clip, (real, ee, r2, moving, points) in reference.items():
            path = os.path.join(args.gen, model, f"{clip}.mp4")
            if ee < 0 or not os.path.exists(path):
                continue
            fake, _ = track(np.stack(imageio.mimread(path)), points)
            errors = np.linalg.norm(fake[:, ee] - real[:, ee], axis=1)
            rows.append({
                "model": model, "clip": clip,
                "ee_ndtw": ndtw(fake[:, ee], real[:, ee]),
                "ee_err_px": float(errors[1:].mean()),
                "ee_final_px": float(errors[-1]),
                "path_ratio": path_length(fake[:, ee]) / max(path_length(real[:, ee]), 1e-6),
                "moving_ndtw": float(np.mean([ndtw(fake[:, i], real[:, i]) for i in np.flatnonzero(moving)])),
                "ee_fit_r2": r2,
                "ee_real_path_px": path_length(real[:, ee]),
            })  # fmt: skip

    out = os.path.join(args.gen, "metrics_motion.csv")
    kept_rows = []
    if os.path.exists(out):
        with open(out) as f:
            kept_rows = [r for r in csv.DictReader(f) if r["model"] not in args.models]
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "clip", *FIELDS])
        writer.writeheader()
        writer.writerows(kept_rows + rows)

    summary = []
    for model in args.models:
        values = {k: np.array([r[k] for r in rows if r["model"] == model]) for k in FIELDS}
        summary.append([model, len(values["ee_ndtw"])] + [f"{v.mean():.3f} +- {v.std():.3f}" for v in values.values()])
        scores = "  ".join(f"{k} {s}" for k, s in zip(FIELDS, summary[-1][2:], strict=True))
        print(f"{model:22s} clips {summary[-1][1]:3d}  {scores}")
    if args.wandb:
        import wandb

        per_clip = [[r["model"], r["clip"], *(r[k] for k in FIELDS)] for r in rows]
        run = wandb.init(project=args.wandb, job_type="eval_motion", config={"models": args.models, "d_th": D_TH})
        run.log({
            "motion_metrics_per_clip": wandb.Table(columns=["model", "clip", *FIELDS], data=per_clip),
            "motion_metrics_summary": wandb.Table(columns=["model", "clips", *FIELDS], data=summary),
        })  # fmt: skip
        run.finish()


if __name__ == "__main__":
    main()
