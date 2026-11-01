"""
Do the metrics agree on which generated video is better? Spearman correlation between every pair of metrics over all
generated clips (every model, every clip), each metric oriented so that higher means better.

Usage (from video_eval/; reads the metrics_*.csv files written by the eval scripts):
    python scripts/metric_agreement.py --gen <GEN_DIR> --out paper/fig/agreement.pdf

Only clips scored by every metric are used (the 2-D track needs an end-effector point). The real and the still video
are references, not generated videos, and are left out. Writes the figure and <GEN_DIR>/metric_agreement.csv.
Colours: the diverging pair of the dataviz reference palette (blue for agreement, red for disagreement, gray at 0).

Author: Duc Anh Hoang
"""

import argparse
import csv
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import spearmanr

METRICS = [  # file, column, label, sign (+1 if higher is better)
    ("pixel", "psnr", "PSNR", 1),
    ("pixel", "ssim", "SSIM", 1),
    ("pixel", "lpips", "LPIPS", -1),
    ("temporal", "dino_adjacent", "DINOv2", 1),
    ("temporal", "warp_error", "Warp", -1),
    ("motion_cotracker", "ee_ndtw", "2-D NDTW", 1),
    ("motion_cotracker", "ee_err_px", "2-D px", -1),
    ("idm_cosmos3", "pos_err_cm", "EVID cm", -1),
]
REFERENCES = {"real", "still"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gen", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    values = {}  # (model, clip) -> {label: value}
    for name, column, label, sign in METRICS:
        with open(os.path.join(args.gen, f"metrics_{name}.csv")) as f:
            for row in csv.DictReader(f):
                if row["model"] not in REFERENCES and row[column] not in ("", "nan"):
                    values.setdefault((row["model"], row["clip"]), {})[label] = sign * float(row[column])
    labels = [m[2] for m in METRICS]
    rows = [v for v in values.values() if len(v) == len(labels)]
    table = np.array([[v[label] for label in labels] for v in rows])
    rho = spearmanr(table).statistic
    print(f"{len(rows)} generated clips from {len({k[0] for k, v in values.items() if len(v) == len(labels)})} models")

    with open(os.path.join(args.gen, "metric_agreement.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([""] + labels)
        writer.writerows([[a] + [f"{r:.3f}" for r in line] for a, line in zip(labels, rho, strict=True)])

    plt.rcParams.update({"font.size": 6.5, "font.family": "STIXGeneral", "mathtext.fontset": "stix"})
    cmap = LinearSegmentedColormap.from_list("diverging", ["#e34948", "#f0efec", "#2a78d6"])
    fig, ax = plt.subplots(figsize=(3.3, 2.75))
    ax.imshow(rho, cmap=cmap, vmin=-1, vmax=1)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ink = "white" if abs(rho[i, j]) > 0.6 else "#1f1f1d"
            ax.text(j, i, f"{rho[i, j]:.2f}", ha="center", va="center", color=ink, fontsize=5.5)
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout(pad=0.2)
    fig.savefig(args.out, bbox_inches="tight")
    for label, line in zip(labels, rho, strict=True):
        print(f"{label:9s} " + " ".join(f"{r:5.2f}" for r in line))


if __name__ == "__main__":
    main()
