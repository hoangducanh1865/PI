"""
Side-by-side example videos for the submission: for each clip, the real video and the generated ones next to each
other, each labelled with its model and the cycle error of EVID (scripts/cosmos3_idm.py), plus the instruction.

Usage (from video_eval/):
    python scripts/make_examples.py --gen <GEN_DIR> --clips 6_0_32 1183_0_0 \
        --panels real:"Real video" frame_ada/seed3407:"IRASim-Frame-Ada" --out <OUT_DIR> [--name <PREFIX>]

Panels are <model folder>:<label>; clips are 16 frames (resized to 256 x 320), written at 4 fps and played twice.
Writes <OUT_DIR>/<PREFIX><CLIP>.mp4.

Author: Duc Anh Hoang
"""

import argparse
import csv
import os

import imageio.v2 as imageio
import matplotlib
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT = os.path.join(matplotlib.get_data_path(), "fonts", "ttf", "DejaVuSans.ttf")
HEIGHT, WIDTH, BAR, GAP = 256, 320, 44, 4


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gen", required=True)
    parser.add_argument("--clips", nargs="+", required=True)
    parser.add_argument("--panels", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--name", default="")
    args = parser.parse_args()

    panels = [p.split(":", 1) for p in args.panels]
    errors = {}
    with open(os.path.join(args.gen, "metrics_idm_cosmos3.csv")) as f:
        for row in csv.DictReader(f):
            errors[(row["model"], row["clip"])] = float(row["pos_err_cm"])
    font, small = ImageFont.truetype(FONT, 15), ImageFont.truetype(FONT, 13)
    os.makedirs(args.out, exist_ok=True)
    for clip in args.clips:
        instruction = str(np.load(os.path.join(args.gen, "states", f"{clip}.npz"))["instruction"])
        videos = []
        for folder, _ in panels:
            frames = imageio.mimread(os.path.join(args.gen, folder, f"{clip}.mp4"))[:16]
            videos.append([np.asarray(Image.fromarray(np.asarray(x)[..., :3]).resize((WIDTH, HEIGHT))) for x in frames])
        width = len(panels) * WIDTH + (len(panels) - 1) * GAP
        out = []
        for t in range(16):
            canvas = Image.new("RGB", (width, HEIGHT + 2 * BAR), "white")
            draw = ImageDraw.Draw(canvas)
            draw.text((8, 10), f'"{instruction}"   frame {t:2d} / 15', fill="#1f1f1d", font=font)
            for i, ((folder, label), video) in enumerate(zip(panels, videos, strict=True)):
                x = i * (WIDTH + GAP)
                canvas.paste(Image.fromarray(video[t]), (x, BAR))
                error = errors.get((folder, clip))
                note = f"cycle error {error:.2f} cm" if error is not None else ""
                draw.text((x + 6, BAR + HEIGHT + 4), label, fill="#1f1f1d", font=small)
                draw.text((x + 6, BAR + HEIGHT + 22), note, fill="#55554f", font=small)
            out.append(np.asarray(canvas))
        path = os.path.join(args.out, f"{args.name}{clip}.mp4")
        imageio.mimwrite(path, out * 2, fps=4, macro_block_size=4)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
