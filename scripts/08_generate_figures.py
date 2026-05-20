"""Stage 8: generate the publishable figure set.

Fig 1: per-dataset CDFs of R_max with B_sim and B_d lines.
Fig 2: forest plot of tau_hat_d - tau with DKW band and 0.03 tolerance band.
Fig 3: bar chart of TTC<2s crossing vs non-crossing.
Fig 4: cross-dataset transfer heatmaps (one per tau).
"""
import glob, json, sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
mpl.rcParams["figure.dpi"] = 130

from src.utils import load_weights
from src.risk import composite_risk

TAUS = [0.10, 0.15, 0.20]

def load_all():
    feats = {}; bnds = {}; verds = {}; trans = None
    for fpath in sorted(glob.glob(str(ROOT / "results/per_dataset/*_features.json"))):
        with open(fpath) as f:
            d = json.load(f); feats[d["dataset"]] = d
    for fpath in sorted(glob.glob(str(ROOT / "results/boundaries/*.json"))):
        with open(fpath) as f:
            d = json.load(f); bnds[d["dataset"]] = d
    for fpath in sorted(glob.glob(str(ROOT / "results/verdicts/*.json"))):
        with open(fpath) as f:
            d = json.load(f); verds[d["dataset"]] = d
    tpath = ROOT / "results/transfer/matrix.json"
    if tpath.exists():
        with open(tpath) as f:
            trans = json.load(f)
    return feats, bnds, verds, trans

def fig1_cdfs(feats, bnds, outdir):
    weights, _ = load_weights()
    n_ds = len(feats)
    fig, axes = plt.subplots(1, n_ds, figsize=(4 * n_ds, 3.5), sharey=True)
    if n_ds == 1: axes = [axes]
    for ax, (ds, d) in zip(axes, feats.items()):
        rmax = np.array([float(np.max(composite_risk(np.array(t["features"]), weights)))
                         for t in d["trajectories"]])
        ax.plot(np.sort(rmax), np.linspace(0, 1, len(rmax)), color="black", lw=1.5)
        bsim_norm = {f"{float(k):.2f}": v for k, v in bnds[ds]["B_sim"].items()}
        for tau in TAUS:
            B_sim = bsim_norm[f"{tau:.2f}"]
            B_d = bnds[ds]["B_d"][f"{tau:.2f}"]
            ax.axvline(B_sim, color="C1", ls="--", alpha=0.6, label=f"$B^{{sim}}_{{\\tau={tau}}}$")
            ax.axvline(B_d, color="C0", ls="-", alpha=0.8, label=f"$B^d_{{\\tau={tau}}}$")
        ax.set_title(ds); ax.set_xlabel("$R_{\\max}$"); ax.set_xlim(0, 1)
    axes[0].set_ylabel("CDF")
    plt.tight_layout()
    out = outdir / "fig1_cdf_rmax.png"; plt.savefig(out, bbox_inches="tight"); plt.close()
    print(f"-> {out}")

def fig2_forest(bnds, outdir):
    rows = []
    for ds in bnds:
        for tau in TAUS:
            tau_hat = bnds[ds]["tau_hat_d"][f"{tau:.2f}"]
            lo, hi = bnds[ds]["ci"][f"{tau:.2f}"]["tau_hat_d"]
            rows.append((f"{ds} τ={tau}", tau_hat - tau, lo - tau, hi - tau))
    labels = [r[0] for r in rows]
    diffs = [r[1] for r in rows]
    lows = [r[2] for r in rows]; his = [r[3] for r in rows]
    y = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(7, 0.35 * len(rows) + 1.5))
    ax.axvspan(-0.03, 0.03, color="gray", alpha=0.2, label="±0.03 tolerance")
    ax.errorbar(diffs, y, xerr=[np.array(diffs)-np.array(lows), np.array(his)-np.array(diffs)],
                fmt="o", color="black", capsize=3)
    ax.set_yticks(y); ax.set_yticklabels(labels)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel(r"$\hat\tau_d - \tau$"); ax.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    out = outdir / "fig2_forest_coverage.png"; plt.savefig(out, bbox_inches="tight"); plt.close()
    print(f"-> {out}")

def fig4_transfer(trans, outdir):
    if trans is None:
        print("[fig4] no transfer matrix yet; skipping"); return
    datasets = trans["datasets"]
    n = len(datasets)
    fig, axes = plt.subplots(1, len(TAUS), figsize=(4 * len(TAUS), 3.6))
    if len(TAUS) == 1: axes = [axes]
    for ax, tau in zip(axes, TAUS):
        M = np.zeros((n, n))
        for i, d1 in enumerate(datasets):
            for j, d2 in enumerate(datasets):
                M[i, j] = trans["by_tau"][f"{tau:.2f}"][f"{d1}__on__{d2}"]["realized_rate"]
        im = ax.imshow(M, vmin=0, vmax=2 * tau, cmap="RdYlGn_r")
        ax.set_xticks(range(n)); ax.set_xticklabels(datasets, rotation=30)
        ax.set_yticks(range(n)); ax.set_yticklabels(datasets)
        ax.set_xlabel("evaluated on"); ax.set_ylabel("B trained on")
        ax.set_title(f"τ = {tau}")
        for i in range(n):
            for j in range(n):
                ax.text(j, i, f"{M[i,j]:.3f}", ha="center", va="center",
                        color="white" if abs(M[i,j]-tau)>0.05 else "black", fontsize=9)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    out = outdir / "fig4_transfer_heatmaps.png"; plt.savefig(out, bbox_inches="tight"); plt.close()
    print(f"-> {out}")

if __name__ == "__main__":
    outdir = ROOT / "results/figures"; outdir.mkdir(parents=True, exist_ok=True)
    feats, bnds, verds, trans = load_all()
    if feats and bnds:
        fig1_cdfs(feats, bnds, outdir)
        fig2_forest(bnds, outdir)
    fig4_transfer(trans, outdir)
