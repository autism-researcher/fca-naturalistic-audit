"""Generate the pre-registered offline validation pipeline schematic
in black-and-white for IEEE journal print.

Design notes:
- Grayscale only. Category distinction comes from border style, not from
  a hatch fill (hatching obscures text in printed figures).
    Data:    solid border, light-grey fill
    Process: solid border, white fill
    Frozen:  double-line border, white fill  (locked/immutable visual cue)
    Test:    dashed border, white fill
    Output:  thick border, mid-grey fill
- Serif typography (IEEE convention); mathtext for symbols.
- Gate annotations are dashed horizontal rules with italic left-margin labels.
- 300 DPI PNG plus vector PDF.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from matplotlib.legend_handler import HandlerPatch


ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "results" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PNG = OUT_DIR / "fig_pipeline_bw.png"
OUT_PDF = OUT_DIR / "fig_pipeline_bw.pdf"


plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman", "Times", "serif"],
    "mathtext.fontset": "dejavuserif",
    "axes.linewidth": 1.0,
    "savefig.facecolor": "white",
    "savefig.edgecolor": "white",
})


def base_box(ax, cx, cy, w, h, facecolor, lw=1.1, linestyle="solid", zorder=2):
    p = FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.12",
        facecolor=facecolor, edgecolor="black",
        linewidth=lw, linestyle=linestyle,
        zorder=zorder,
    )
    ax.add_patch(p)


def inner_frame(ax, cx, cy, w, h, lw=0.9):
    inset = 0.10
    p = FancyBboxPatch(
        (cx - w / 2 + inset, cy - h / 2 + inset),
        w - 2 * inset, h - 2 * inset,
        boxstyle="round,pad=0.01,rounding_size=0.08",
        facecolor="none", edgecolor="black",
        linewidth=lw, zorder=3,
    )
    ax.add_patch(p)


def box(ax, cx, cy, w, h, text, kind, fontsize=10, fontweight="normal"):
    if kind == "data":
        base_box(ax, cx, cy, w, h, facecolor="0.88", lw=1.1)
    elif kind == "process":
        base_box(ax, cx, cy, w, h, facecolor="1.00", lw=1.1)
    elif kind == "frozen":
        base_box(ax, cx, cy, w, h, facecolor="1.00", lw=1.2)
        inner_frame(ax, cx, cy, w, h, lw=0.8)
    elif kind == "test":
        base_box(ax, cx, cy, w, h, facecolor="1.00", lw=1.2,
                 linestyle=(0, (5, 3)))
    elif kind == "output":
        base_box(ax, cx, cy, w, h, facecolor="0.78", lw=1.6)
    else:
        raise ValueError(kind)
    ax.text(cx, cy, text, ha="center", va="center",
            fontsize=fontsize, fontweight=fontweight, color="black",
            multialignment="center", zorder=4)


def arrow(ax, x1, y1, x2, y2, lw=1.1):
    a = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle="-|>", mutation_scale=12,
        lw=lw, color="black", zorder=1,
        shrinkA=0, shrinkB=2,
    )
    ax.add_patch(a)


def gate_marker(ax, y, label):
    ax.plot([0.7, 11.7], [y, y], linestyle=(0, (5, 4)),
            color="0.35", lw=0.8, zorder=0)
    ax.text(0.35, y, label, ha="left", va="center",
            fontsize=9, fontstyle="italic", color="0.20")


class _LegendBox(mpatches.Patch):
    def __init__(self, facecolor, lw=1.0, linestyle="solid",
                 double=False, label=""):
        super().__init__(label=label)
        self._facecolor = facecolor
        self._lw = lw
        self._linestyle = linestyle
        self._double = double


class _LegendBoxHandler(HandlerPatch):
    def create_artists(self, legend, orig_handle, xdescent, ydescent,
                       width, height, fontsize, trans):
        x, y, w, h = -xdescent, -ydescent, width, height
        outer = Rectangle((x, y), w, h,
                          facecolor=orig_handle._facecolor,
                          edgecolor="black",
                          linewidth=orig_handle._lw,
                          linestyle=orig_handle._linestyle,
                          transform=trans)
        artists = [outer]
        if orig_handle._double:
            inset = 1.6
            inner = Rectangle((x + inset, y + inset),
                              w - 2 * inset, h - 2 * inset,
                              facecolor="none", edgecolor="black",
                              linewidth=0.7, transform=trans)
            artists.append(inner)
        return artists


def main():
    fig, ax = plt.subplots(figsize=(11.0, 9.5))
    ax.set_xlim(0, 12)
    ax.set_ylim(-0.2, 12.0)
    ax.set_aspect("equal")
    ax.axis("off")

    ax.text(6.0, 11.55, "Pre-Registered Offline Validation Pipeline",
            ha="center", va="center", fontsize=15, fontweight="bold")

    # Row 1: datasets
    box(ax, 3.0, 10.6, 2.0, 0.85, "HighD", "data",
        fontsize=12, fontweight="bold")
    box(ax, 6.0, 10.6, 2.0, 0.85, "NGSIM", "data",
        fontsize=12, fontweight="bold")
    box(ax, 9.0, 10.6, 2.0, 0.85, "Waymo", "data",
        fontsize=12, fontweight="bold")

    gate_marker(ax, 9.95, "Gate 1\n(OSF lock)")

    # Row 2: extractors
    box(ax, 3.0, 9.05, 2.4, 0.85, "HighD extractor", "process")
    box(ax, 6.0, 9.05, 2.4, 0.85, "NGSIM extractor", "process")
    box(ax, 9.0, 9.05, 2.4, 0.85, "Waymo extractor", "process")

    arrow(ax, 3.0, 10.18, 3.0, 9.50)
    arrow(ax, 6.0, 10.18, 6.0, 9.50)
    arrow(ax, 9.0, 10.18, 9.0, 9.50)

    gate_marker(ax, 8.30, "Gate 2\n(Git tag)")

    # Row 3: features (raw TTC is NOT an input to R; it is retained
    # separately for H_OFF3 -- label made explicit per review)
    box(ax, 6.0, 7.45, 7.4, 0.9,
        "8 normalized risk features  $f(x_t) \\in [0,1]^{8}$\n"
        r"(raw TTC retained separately for $H_{\mathrm{OFF3}}$)",
        "process", fontsize=10)

    arrow(ax, 3.0, 8.63, 4.5, 7.92)
    arrow(ax, 6.0, 8.63, 6.0, 7.92)
    arrow(ax, 9.0, 8.63, 7.5, 7.92)

    # Row 4: frozen weights + R(x)
    box(ax, 2.4, 5.95, 3.6, 1.30,
        "Frozen weights $w$\n(fixed input, $\\sum w = 0.94$)\nNO retuning",
        "frozen", fontsize=9.5)
    box(ax, 7.9, 5.95, 4.0, 1.30,
        r"$R(x_t) = w^{\top} f(x_t)$"
        + "\n"
        + r"$R_{\max}$ per trajectory",
        "process", fontsize=10.5)

    arrow(ax, 6.0, 6.98, 7.9, 6.62)
    arrow(ax, 4.22, 5.95, 5.88, 5.95)

    # Row 5: B_d + B_sim
    bd_text = (
        r"$B^{\,d}_{N,\,\tau}$  is the $(1{-}\tau)$-quantile"
        + "\n"
        + r"$\hat{\tau}_{\,d}$  is the realized rate of $B^{\,\mathrm{sim}}$"
    )
    box(ax, 4.6, 4.10, 4.6, 1.30, bd_text, "process", fontsize=10.5)

    bsim_text = (
        r"$B^{\,\mathrm{sim}}$ (fixed input)"
        + "\n(calibrated constant;"
        + "\nno re-run)"
    )
    box(ax, 9.9, 4.10, 3.2, 1.30, bsim_text, "frozen", fontsize=9.5)

    arrow(ax, 7.9, 5.30, 5.2, 4.78)
    arrow(ax, 8.32, 4.10, 6.91, 4.10)

    # Row 6: hypothesis tests
    h1 = (
        r"$H_{\mathrm{OFF1}}$"
        + "\n"
        + r"$|B^{\,\mathrm{sim}} - B^{\,d}| < 0.03$"
    )
    box(ax, 2.4, 2.20, 2.8, 1.05, h1, "test", fontsize=10.5)

    h2 = (
        r"$H_{\mathrm{OFF2}}$"
        + "\n"
        + r"$|\hat{\tau}_{\,d} - \tau| \leq 0.03$"
    )
    box(ax, 6.0, 2.20, 2.8, 1.05, h2, "test", fontsize=10.5)

    h3 = (
        r"$H_{\mathrm{OFF3}}$"
        + "\nMann-Whitney $U$, Bonferroni"
    )
    box(ax, 9.6, 2.20, 2.8, 1.05, h3, "test", fontsize=10.5)

    arrow(ax, 3.8, 3.45, 2.6, 2.73)
    arrow(ax, 4.6, 3.45, 5.7, 2.73)
    arrow(ax, 5.4, 3.45, 9.0, 2.73)

    # Row 7: verdict
    verdict = (
        r"Verdict per $(d,\,\tau)$  -  "
        r"PASS overall iff $\geq 1$ dataset passes for all $\tau$"
    )
    box(ax, 6.0, 0.55, 9.5, 0.95, verdict, "output",
        fontsize=11, fontweight="bold")

    arrow(ax, 2.6, 1.67, 3.8, 1.05)
    arrow(ax, 6.0, 1.67, 6.0, 1.05)
    arrow(ax, 9.4, 1.67, 8.2, 1.05)

    handles = [
        _LegendBox(facecolor="0.88", lw=1.1, label="Data"),
        _LegendBox(facecolor="1.00", lw=1.1, label="Processing"),
        _LegendBox(facecolor="1.00", lw=1.2, double=True,
                   label="Fixed input"),
        _LegendBox(facecolor="1.00", lw=1.2, linestyle=(0, (5, 3)),
                   label="Hypothesis test"),
        _LegendBox(facecolor="0.78", lw=1.6, label="Output"),
    ]
    ax.legend(
        handles=handles,
        handler_map={_LegendBox: _LegendBoxHandler()},
        loc="lower center", bbox_to_anchor=(0.5, -0.04),
        ncol=5, frameon=False, fontsize=10,
        handlelength=2.2, handleheight=1.4, columnspacing=1.6,
    )

    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight",
                facecolor="white", edgecolor="white")
    fig.savefig(OUT_PDF, bbox_inches="tight",
                facecolor="white", edgecolor="white")
    plt.close(fig)
    print("Saved:", OUT_PNG)
    print("Saved:", OUT_PDF)


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
