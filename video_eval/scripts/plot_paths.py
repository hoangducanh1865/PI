"""
Figure of EVID on one clip: the end-effector path that Cosmos3-Nano reads back from each video (scripts/cosmos3_idm.py)
against the commanded path, seen from above (x, y) and as height over time (z).

Usage (from video_eval/; needs <GEN_DIR>/<MODEL>/cosmos3_idm/<CLIP>.npz and <GEN_DIR>/states/<CLIP>.npz):
    python scripts/plot_paths.py --gen <GEN_DIR> --clip 12_0_16 --out paper/fig/paths.pdf

Colours: the validated categorical slots 1 to 3 of the dataviz reference palette for the generated videos, neutral
ink for the references; every line also has its own dash and a legend entry, so colour is never the only cue.

Author: Duc Anh Hoang
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from cosmos3_idm import decode

LINES = [  # model folder, label, colour, dash
    ("real", "Real video", "#8a8a85", (0, (1, 1.5))),
    ("frame_ada/seed3407", "IRASim-Frame-Ada", "#2a78d6", "-"),
    ("cosmos_action/seed0", "Cosmos-Predict2.5, actions", "#eb6834", (0, (5, 1.5))),
    ("cosmos_base/seed0", "Cosmos-Predict2.5, no actions", "#1baf7a", (0, (2, 1.5, 0.5, 1.5))),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gen", required=True)
    parser.add_argument("--clip", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    state = np.load(os.path.join(args.gen, "states", f"{args.clip}.npz"))["state"]
    plt.rcParams.update({"font.size": 7, "font.family": "STIXGeneral", "mathtext.fontset": "stix",
                         "axes.linewidth": 0.5, "axes.edgecolor": "#6b6b66", "xtick.color": "#6b6b66",
                         "ytick.color": "#6b6b66", "axes.labelcolor": "#333333"})  # fmt: skip
    fig, (top, side) = plt.subplots(1, 2, figsize=(3.35, 1.55), gridspec_kw={"width_ratios": [1.15, 1]})
    paths = [("Commanded", state[:, :3], "#1f1f1d", "-")]
    for folder, label, colour, dash in LINES:
        raw = np.load(os.path.join(args.gen, folder, "cosmos3_idm", f"{args.clip}.npz"))["raw"]
        paths.append((label, decode(raw, state[0])[:, :3], colour, dash))
    for label, xyz, colour, dash in paths:
        xyz = (xyz - state[0, :3]) * 100  # cm from the start
        width = 2.0 if label == "Commanded" else 1.2
        top.plot(xyz[:, 0], xyz[:, 1], color=colour, linestyle=dash, linewidth=width, label=label)
        top.plot(xyz[-1, 0], xyz[-1, 1], "o", color=colour, markersize=3, markeredgecolor="white", markeredgewidth=0.5)
        side.plot(np.arange(16), xyz[:, 2], color=colour, linestyle=dash, linewidth=width)
    top.plot(0, 0, "s", color="#1f1f1d", markersize=3)
    top.set_xlabel("x (cm)")
    top.set_ylabel("y (cm)")
    top.set_aspect("equal", adjustable="datalim")
    side.set_xlabel("frame")
    side.set_ylabel("z (cm)")
    for ax in (top, side):
        ax.grid(color="#e6e6e1", linewidth=0.4)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(width=0.5, length=2)
    fig.legend(loc="upper center", ncol=3, frameon=False, fontsize=5.5, handlelength=2.2, columnspacing=1.0,
               bbox_to_anchor=(0.5, 1.07))  # fmt: skip
    fig.tight_layout(pad=0.2, rect=(0, 0, 1, 0.86))
    fig.savefig(args.out, bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
