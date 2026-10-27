"""
Compare generated clips with the real clips of the same actions: PSNR, SSIM and LPIPS on frames 1 to 15 (frame 0 is
the given conditioning frame, left out as in IRASim's evaluation).

Usage (from video_eval/):
    python scripts/eval_metrics.py --gen <GEN_DIR> --models <MODEL_DIR> [<MODEL_DIR> ...] [--wandb <PROJECT>]
    e.g. python scripts/eval_metrics.py --gen <GEN_DIR> --models frame_ada/seed3407 still

<GEN_DIR> holds real/<clip>.mp4 (scripts/make_references.py) and each <MODEL_DIR>/<clip>.mp4 (scripts/run_irasim.py,
or still/ for the floor). Writes <GEN_DIR>/metrics_pixel.csv (one row per model and clip; rerun-safe: rows of the
given models are replaced) and prints the mean and standard deviation per model; --wandb also logs both tables.

What they measure: PSNR = closeness of pixel values (dB, higher is better), SSIM = closeness of local structure
(0 to 1), LPIPS = perceptual distance in AlexNet features (lower is better). What they cannot establish: a blurry or
still video can score well; none says whether the robot follows the given actions or whether contacts are physical.

Author: Duc Anh Hoang
"""

import argparse
import csv
import glob
import os

import imageio.v2 as imageio
import lpips
import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

METRICS = ("psnr", "ssim", "lpips")


def read(path: str) -> np.ndarray:
    return np.stack(imageio.mimread(path))[1:16]  # frames 1 to 15


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gen", required=True)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--wandb", help="W&B project to log the tables to")
    args = parser.parse_args()
    perceptual = lpips.LPIPS(net="alex", verbose=False).eval()

    rows = []
    for model in args.models:
        for path in sorted(glob.glob(os.path.join(args.gen, model, "*.mp4"))):
            clip = os.path.basename(path)[:-4]
            fake, real = read(path), read(os.path.join(args.gen, "real", f"{clip}.mp4"))
            to_tensor = lambda v: torch.from_numpy(v).permute(0, 3, 1, 2).float() / 127.5 - 1  # noqa: E731
            with torch.no_grad():
                distance = perceptual(to_tensor(fake), to_tensor(real)).mean().item()
            rows.append({
                "model": model, "clip": clip,
                "psnr": np.mean([peak_signal_noise_ratio(r, f, data_range=255) for r, f in zip(real, fake)]),
                "ssim": np.mean([structural_similarity(r, f, channel_axis=2, data_range=255) for r, f in zip(real, fake)]),
                "lpips": distance,
            })  # fmt: skip

    out = os.path.join(args.gen, "metrics_pixel.csv")
    kept = []
    if os.path.exists(out):
        with open(out) as f:
            kept = [r for r in csv.DictReader(f) if r["model"] not in args.models]
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "clip", *METRICS])
        writer.writeheader()
        writer.writerows(kept + rows)

    summary = []
    for model in args.models:
        values = {m: np.array([r[m] for r in rows if r["model"] == model]) for m in METRICS}
        summary.append([model, len(values["psnr"])] + [f"{v.mean():.3f} +- {v.std():.3f}" for v in values.values()])
        print(f"{model:24s} clips {summary[-1][1]:3d}  " + "  ".join(f"{m} {s}" for m, s in zip(METRICS, summary[-1][2:])))
    if args.wandb:
        import wandb

        per_clip = [[r["model"], r["clip"], *(r[m] for m in METRICS)] for r in rows]
        run = wandb.init(project=args.wandb, job_type="eval_metrics", config={"models": args.models})
        run.log({
            "pixel_metrics_per_clip": wandb.Table(columns=["model", "clip", *METRICS], data=per_clip),
            "pixel_metrics_summary": wandb.Table(columns=["model", "clips", *METRICS], data=summary),
        })  # fmt: skip
        run.finish()


if __name__ == "__main__":
    main()
