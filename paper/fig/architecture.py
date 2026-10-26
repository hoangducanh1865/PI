"""
Draw the architecture figure of the paper and the slides: GR-ConvNet + CLIP (ALBEF fusion) with the proposed
region-text contrastive loss, which is used in training only.

Usage: python paper/fig/architecture.py   (writes architecture.pdf and architecture.png next to this file)

Author: Duc Anh Hoang
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

plt.rcParams.update({"font.family": "DejaVu Sans", "mathtext.fontset": "cm", "font.size": 9})
INK, GREY, OURS, OURS_FILL, FROZEN = "#111111", "#F2F2F2", "#B3261E", "#FCEDEC", "#E3ECF7"


def box(ax, x, y, w, h, text, fill=GREY, edge=INK, dashed=False, size=9, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", facecolor=fill,
                                edgecolor=edge, linewidth=1.1, linestyle=(0, (4, 2)) if dashed else "-"))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size, color=INK,
            fontweight="bold" if bold else "normal", linespacing=1.25)


def arrow(ax, a, b, color=INK, dashed=False, rad=0.0):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=10, color=color, linewidth=1.1,
                                 linestyle=(0, (4, 2)) if dashed else "-", connectionstyle=f"arc3,rad={rad}"))


def main():
    fig, ax = plt.subplots(figsize=(7.0, 2.9))
    ax.set_xlim(0, 17.5)
    ax.set_ylim(-0.6, 7.2)
    ax.axis("off")

    # Grasp branch (the reproduced GR-ConvNet + CLIP baseline), top row.
    y, h = 5.2, 1.4
    row = [(0.1, 1.9, "RGB image\n224 x 224"), (2.6, 2.4, "GR-ConvNet\nencoder\n3 conv + 5 res"),
           (5.6, 2.3, "features $F$\n128 x 56 x 56"), (8.5, 2.3, "ALBEF\ncross-attention"),
           (11.4, 2.0, "decoder\n3 deconv"), (14.0, 3.4, "maps $Q, \\cos 2\\theta, \\sin 2\\theta, w$\ngrasp $(x, y, w, h, \\theta)$")]
    for x, w, text in row:
        box(ax, x, y, w, h, text, size=6.8)
    for (x0, w0, _), (x1, _, _) in zip(row, row[1:], strict=False):
        arrow(ax, (x0 + w0, y + h / 2), (x1, y + h / 2))
    ax.text(15.7, 4.75, r"$\mathcal{L}_{\mathrm{grasp}}$: smooth L1, 4 maps", ha="center", fontsize=6.8)

    # Text: frozen CLIP.
    box(ax, 8.5, 3.0, 2.3, 1.3, "frozen CLIP\ntext encoder", fill=FROZEN, size=6.8)
    box(ax, 11.4, 3.0, 3.1, 1.3, '"Pick up the pear\nby its stem."', fill="white", size=6.8)
    arrow(ax, (11.4, 3.65), (10.8, 3.65))
    arrow(ax, (9.65, 4.3), (9.65, 5.2))
    ax.text(9.8, 4.6, "tokens", fontsize=6.3)

    # Proposed: region-text contrastive loss, training only (dashed, red).
    ax.add_patch(FancyBboxPatch((0.05, 0.1), 11.0, 2.35, boxstyle="round,pad=0.02,rounding_size=0.12",
                                facecolor=OURS_FILL, edgecolor=OURS, linewidth=1.1, linestyle=(0, (4, 2))))
    ax.text(0.3, -0.35, "Proposed: region-text contrastive grounding loss (training only)", fontsize=7,
            color=OURS, fontweight="bold")
    box(ax, 0.3, 0.3, 3.1, 1.9, "masks of the scene\n$m_0$: prompted object\n$m_{1..K}$: other objects\n$m_{K+1}$: rest of scene",
        fill="white", edge=OURS, size=6.3)
    box(ax, 3.9, 0.55, 2.5, 1.4, "masked average\npooling of $F$\n$\\rightarrow r_0, \\dots, r_{K+1}$", fill="white", edge=OURS, size=6.3)
    box(ax, 6.9, 0.3, 3.9, 1.9, "", fill="white", edge=OURS)
    ax.text(8.85, 1.65, r"$s_k = \cos(P_r r_k,\, P_t t) / \tau$", ha="center", va="center", fontsize=7)
    ax.text(8.85, 0.8, r"$\mathcal{L}_{\mathrm{con}} = -\log \frac{e^{s_0}}{\sum_k e^{s_k}}$", ha="center",
            va="center", fontsize=7.5)
    arrow(ax, (3.4, 1.25), (3.9, 1.25), OURS)
    arrow(ax, (6.4, 1.25), (6.9, 1.25), OURS)
    arrow(ax, (6.75, 5.2), (5.15, 1.95), OURS, dashed=True)  # features F -> pooling
    arrow(ax, (9.0, 3.0), (8.85, 2.2), OURS, dashed=True)  # pooled CLIP embedding t -> loss
    ax.text(9.15, 2.6, "embedding $t$", fontsize=6.3, color=OURS)
    ax.text(14.2, 1.15, r"$\mathcal{L} = \mathcal{L}_{\mathrm{grasp}} + \lambda\, \mathcal{L}_{\mathrm{con}}$",
            fontsize=8.5, color=INK, ha="center")

    out = os.path.dirname(os.path.abspath(__file__))
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, f"architecture.{ext}"), bbox_inches="tight", dpi=300)


if __name__ == "__main__":
    main()
