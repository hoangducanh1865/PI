"""
Temporal consistency of generated clips: does the scene stay the same scene, and does it change smoothly from frame to
frame? Measured on every model and on the real clips, so each value is read against the real one.

Usage (from video_eval/):
    python scripts/eval_temporal.py --gen <GEN_DIR> --models <MODEL_DIR> [...] [--device cpu]
    e.g. python scripts/eval_temporal.py --gen <GEN_DIR> --models real still frame_ada/seed3407

Per clip (16 frames):
- dino_adjacent: mean cosine similarity of DINOv2 (facebook/dinov2-small, Hugging Face Transformers) CLS features of
  consecutive frames; dino_first: the same between every frame and frame 0 (subject consistency, as in VBench).
- warp_error: mean absolute difference (0 to 255) between frame t + 1 and frame t warped onto it by dense optical
  flow (OpenCV Farneback, computed on the same generated clip): flicker and texture popping that motion cannot explain.
Writes <GEN_DIR>/metrics_temporal.csv (one row per model and clip; rows of the given models are replaced) and prints
the mean per model. What it cannot establish: a clip in which nothing moves scores perfectly on all three; they say
nothing about whether the motion is the commanded one (see eval_motion.py).

Author: Duc Anh Hoang
"""

import argparse
import csv
import glob
import os

import cv2
import imageio.v2 as imageio
import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModel

FIELDS = ("dino_adjacent", "dino_first", "warp_error")


def warp_error(video: np.ndarray) -> float:
    gray = [cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) for frame in video]
    height, width = gray[0].shape
    grid = np.stack(np.meshgrid(np.arange(width), np.arange(height)), axis=-1).astype(np.float32)
    errors = []
    for t in range(len(video) - 1):
        # flow from t+1 back to t, so frame t can be sampled at the positions of frame t+1
        flow = cv2.calcOpticalFlowFarneback(gray[t + 1], gray[t], None, 0.5, 3, 15, 3, 5, 1.2, 0)
        warped = cv2.remap(video[t], grid + flow, None, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        errors.append(np.abs(warped.astype(np.float32) - video[t + 1].astype(np.float32)).mean())
    return float(np.mean(errors))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gen", required=True)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-small")
    dino = AutoModel.from_pretrained("facebook/dinov2-small").eval().to(args.device)

    rows = []
    for model in args.models:
        for path in sorted(glob.glob(os.path.join(args.gen, model, "*.mp4"))):
            video = np.stack(imageio.mimread(path))
            with torch.no_grad():
                inputs = processor(images=list(video), return_tensors="pt").to(args.device)
                features = torch.nn.functional.normalize(dino(**inputs).last_hidden_state[:, 0], dim=-1)
            rows.append({
                "model": model, "clip": os.path.basename(path)[:-4],
                "dino_adjacent": float((features[1:] * features[:-1]).sum(-1).mean()),
                "dino_first": float((features[1:] * features[:1]).sum(-1).mean()),
                "warp_error": warp_error(video),
            })  # fmt: skip

    out = os.path.join(args.gen, "metrics_temporal.csv")
    kept = []
    if os.path.exists(out):
        with open(out) as f:
            kept = [r for r in csv.DictReader(f) if r["model"] not in args.models]
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "clip", *FIELDS])
        writer.writeheader()
        writer.writerows(kept + rows)
    for model in args.models:
        values = {k: np.array([r[k] for r in rows if r["model"] == model]) for k in FIELDS}
        print(f"{model:22s} clips {len(values['warp_error']):3d}  "
              + "  ".join(f"{k} {v.mean():.3f} +- {v.std():.3f}" for k, v in values.items()))  # fmt: skip


if __name__ == "__main__":
    main()
