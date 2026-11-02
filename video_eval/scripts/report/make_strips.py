"""
Frame strips for the report: for each chosen clip, the real video (top row) and a generated one (bottom row) at a few
frames, one image per clip, for side-by-side panels in LaTeX.

Usage (from video_eval/):
    python scripts/report/make_strips.py --gen <GEN_DIR> --model frame_ada/seed3407 --clips 13_0_0 6_0_32 1183_0_0 \
        --frames 0 8 15 --out paper/fig

Writes <OUT>/strip_<CLIP>.jpg (frames of 256 x 320 separated by 6-pixel white gaps).

Author: Duc Anh Hoang
"""

import argparse
import os

import imageio.v2 as imageio
import numpy as np
from PIL import Image

GAP = 6


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gen", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--clips", nargs="+", required=True)
    parser.add_argument("--frames", nargs="+", type=int, required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    for clip in args.clips:
        rows = []
        for folder in ("real", args.model):
            video = imageio.mimread(os.path.join(args.gen, folder, f"{clip}.mp4"))
            tiles = [np.asarray(video[t])[..., :3] for t in args.frames]
            height = tiles[0].shape[0]
            gap = np.full((height, GAP, 3), 255, np.uint8)
            rows.append(np.concatenate([x for t in tiles for x in (t, gap)][:-1], axis=1))
        gap = np.full((GAP, rows[0].shape[1], 3), 255, np.uint8)
        path = os.path.join(args.out, f"strip_{clip}.jpg")
        Image.fromarray(np.concatenate([rows[0], gap, rows[1]], axis=0)).save(path, quality=90)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
