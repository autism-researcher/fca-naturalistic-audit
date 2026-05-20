"""Generate a paper-quality methodology flow diagram (compact text version)."""
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parent.parent

fig, ax = plt.subplots(figsize=(11, 8.5))
ax.set_xlim(0, 12)
ax.set_ylim(0, 10.5)
ax.axis('off')

DATA_COLOR    = "#cfe2f3"
PROC_COLOR    = "#fff2cc"
FROZEN_COLOR  = "#f4cccc"
TEST_COLOR    = "#d9ead3"
VERDICT_COLOR = "#d0e0e3"

def box(x, y, w, h, text, color, fontsize=11, weight="normal"):
    p = FancyBboxPatch(
        (x - w/2, y - h/2), w, h,
        boxstyle="round,pad=0.05,rounding_size=0.20",
        facecolor=color, edgecolor="black", lw=1.1
    )
    ax.add_patch(p)
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize, fontweight=weight)

def arrow(x1, y1, x2, y2, lw=1.2):
    a = FancyArrowPatch((x1, y1), (x2, y2),
        arrowstyle="-|>", color="black", lw=lw,
        mutation_scale=16, shrinkA=4, shrinkB=4)
    ax.add_patch(a)

# Title
ax.text(6, 10.1, "Pre-Registered Offline Validation Pipeline",
        ha="center", va="center", fontsize=14, fontweight="bold")

# Top row: three datasets
box(2.5, 9.1, 2.4, 0.7, "HighD",   DATA_COLOR, fontsize=12, weight="bold")
box(6.0, 9.1, 2.4, 0.7, "NGSIM",   DATA_COLOR, fontsize=12, weight="bold")
box(9.5, 9.1, 2.4, 0.7, "Waymo",   DATA_COLOR, fontsize=12, weight="bold")

# Extractors
box(2.5, 7.9, 2.4, 0.65, "HighD extractor",  PROC_COLOR, fontsize=10)
box(6.0, 7.9, 2.4, 0.65, "NGSIM extractor",  PROC_COLOR, fontsize=10)
box(9.5, 7.9, 2.4, 0.65, "Waymo extractor",  PROC_COLOR, fontsize=10)

arrow(2.5, 8.75, 2.5, 8.22)
arrow(6.0, 8.75, 6.0, 8.22)
arrow(9.5, 8.75, 9.5, 8.22)

# Eight features (shared)
box(6.0, 6.8, 7.5, 0.6, r"8 features $f(x_t)\in[0,1]^8$  +  raw TTC", PROC_COLOR, fontsize=11)
arrow(2.5, 7.58, 4.0, 7.10)
arrow(6.0, 7.58, 6.0, 7.10)
arrow(9.5, 7.58, 8.0, 7.10)

# Frozen weights (left)
box(2.0, 5.4, 3.2, 0.85,
    "Frozen weights $w$\n(from Paper 2, sum 0.94)\nNO retuning",
    FROZEN_COLOR, fontsize=10)
arrow(3.6, 5.4, 5.1, 5.4)

# Composite risk
box(7.7, 5.4, 3.2, 0.85,
    r"$R(x_t) = w^\top f(x_t)$"+"\n"+r"$R_{\max}$ per trajectory",
    PROC_COLOR, fontsize=11)
arrow(6.0, 6.5, 7.7, 5.85)

# B_sim from Paper 2 (right)
box(10.5, 4.2, 3.0, 0.85,
    "$B^{sim}$ from Paper 2\n(published constant;\nno re-run)",
    FROZEN_COLOR, fontsize=10)

# Boundary
box(6.0, 4.2, 3.2, 0.85,
    r"$B^d_{N,\tau}$  ((1-$\tau$)-quantile)"+"\n"+r"$\hat\tau_d$ (realized rate of $B^{sim}$)",
    PROC_COLOR, fontsize=11)
arrow(7.7, 5.0, 7.0, 4.65)
arrow(9.0, 4.2, 7.6, 4.2)

# Three tests
box(2.5, 2.8, 2.6, 0.85,
    r"$H_{\rm OFF1}$"+"\n"+r"$|B^{sim}-B^d|<0.03$",
    TEST_COLOR, fontsize=11)
box(6.0, 2.8, 2.6, 0.85,
    r"$H_{\rm OFF2}$"+"\n"+r"$|\hat\tau_d-\tau|\leq 0.03$",
    TEST_COLOR, fontsize=11)
box(9.5, 2.8, 2.6, 0.85,
    r"$H_{\rm OFF3}$"+"\n"+r"MW $U$, Bonferroni",
    TEST_COLOR, fontsize=11)

arrow(6.0, 3.78, 2.7, 3.25)
arrow(6.0, 3.78, 6.0, 3.25)
arrow(6.0, 3.78, 9.3, 3.25)

# Verdict
box(6.0, 1.3, 8.5, 0.85,
    r"Verdict per $(d,\tau)$  --  PASS overall iff $\geq 1$ dataset passes for all $\tau$",
    VERDICT_COLOR, fontsize=11)
arrow(2.5, 2.37, 3.5, 1.75)
arrow(6.0, 2.37, 6.0, 1.75)
arrow(9.5, 2.37, 8.5, 1.75)

# Gates on the left margin
ax.plot([0.05, 0.7], [7.9, 7.9], "k--", lw=0.8, alpha=0.7)
ax.text(0.05, 8.15, "Gate 1\n(OSF lock)",
        fontsize=9, color="#5b3da0", style="italic", fontweight="bold")

ax.plot([0.05, 0.7], [6.45, 6.45], "k--", lw=0.8, alpha=0.7)
ax.text(0.05, 6.65, "Gate 2\n(Git tag)",
        fontsize=9, color="#5b3da0", style="italic", fontweight="bold")

# Legend
from matplotlib.patches import Patch
legend_items = [
    Patch(facecolor=DATA_COLOR,    edgecolor="black", label="Data"),
    Patch(facecolor=PROC_COLOR,    edgecolor="black", label="Processing"),
    Patch(facecolor=FROZEN_COLOR,  edgecolor="black", label="Frozen (Paper 2)"),
    Patch(facecolor=TEST_COLOR,    edgecolor="black", label="Test"),
    Patch(facecolor=VERDICT_COLOR, edgecolor="black", label="Output"),
]
ax.legend(handles=legend_items, loc="lower center",
          bbox_to_anchor=(0.5, -0.04), fontsize=10, ncol=5, frameon=False)

out_path = ROOT / "results/figures/fig0_pipeline_flow.png"
out_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
plt.close()
print(f"-> {out_path}")
