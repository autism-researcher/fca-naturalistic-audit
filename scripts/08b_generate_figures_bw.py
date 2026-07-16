"""Stage 8 (grayscale variant): regenerate publishable figures in B&W.

Same outputs as 08_generate_figures.py but with grayscale styling
suitable for IEEE journals that prefer black-and-white print figures.
Now also produces a new H_OFF3 visualization (fig3_h_off3_bars.png)
that the original color script never implemented.

Writes the following files into results/figures/, OVERWRITING the
color versions:
  - fig1_cdf_rmax.png        (CDFs of R_max, three panels)
  - fig2_forest_coverage.png (forest plot of tau_hat - tau)
  - fig3_h_off3_bars.png     (NEW: crossing vs non-crossing TTC<2s)
  - fig4_transfer_heatmaps.png (3x3 transfer matrices per tau)

Keep a backup before running if you want to retain the color variants.

Styling choices (all B&W-print legible):
- Fig 1: per-tau linestyle (solid/dashed/dotted), B_sim = medium gray,
         B_d = pure black. Six lines per panel disambiguated by
         (linestyle, shade) and the legend.
- Fig 2: black markers with white face, black error bars, gray
         tolerance band, light grid.
- Fig 3: dark-gray bars (crossing) vs light-gray bars (non-crossing),
         both with black borders. 95% bootstrap CI error bars.
         n-counts annotated above each bar.
- Fig 4: Greys colormap (white=0, black=1). Cell text white on dark
         cells, black on light cells. Numeric value in each cell
         remains the authoritative reading.
"""
import glob, json, sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
mpl.rcParams["figure.dpi"] = 130
mpl.rcParams["font.size"] = 10
mpl.rcParams["savefig.dpi"] = 300  # crisp print quality

from src.utils import load_weights
from src.risk import composite_risk

TAUS = [0.10, 0.15, 0.20]
TTC_THRESH_S = 2.0
LS_BY_TAU = {0.10: "-", 0.15: "--", 0.20: ":"}
SHADE_SIM = "0.45"   # medium gray for the simulation-derived boundary
SHADE_DAT = "0.00"   # pure black for the dataset-recomputed boundary
SHADE_CROSS = "0.25"  # dark gray for "crossing" bars in fig 3
SHADE_NC    = "0.80"  # light gray for "non-crossing" bars in fig 3
BOOT_N = 1000
BOOT_SEED = 42


def load_all():
    feats, bnds, verds = {}, {}, {}
    trans = None
    for fpath in sorted(glob.glob(str(ROOT / "results/per_dataset/*_features.json"))):
        with open(fpath) as f:
            d = json.load(f); feats[d["dataset"]] = d
    for fpath in sorted(glob.glob(str(ROOT / "results/boundaries/*.json"))):
        with open(fpath) as f:
            d = json.load(f); bnds[d["dataset"]] = d
    for fpath in sorted(glob.glob(str(ROOT / "results/verdicts/*.json"))):
        with open(fpath) as f:
            d = json.load(f)
        if isinstance(d, dict) and "dataset" in d and "variant" not in d:
            verds[d["dataset"]] = d
        # else: auxiliary files (e.g., hoff3_corrected.json, or the
        # provided-TTC sensitivity variant) -- not the primary
        # per-dataset verdict files; skip.
    tpath = ROOT / "results/transfer/matrix.json"
    if tpath.exists():
        with open(tpath) as f:
            trans = json.load(f)
    return feats, bnds, verds, trans


def bootstrap_mean_ci(arr, n_boot=BOOT_N, alpha=0.05, seed=BOOT_SEED):
    """Return (mean, lower_err, upper_err) for plotting with yerr=[lo, hi]."""
    arr = np.asarray(arr, dtype=float)
    if arr.size == 0:
        return 0.0, 0.0, 0.0
    mean = float(arr.mean())
    rng = np.random.default_rng(seed)
    n = arr.size
    bs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        bs[i] = arr[idx].mean()
    lo = float(np.percentile(bs, 100 * alpha / 2))
    hi = float(np.percentile(bs, 100 * (1 - alpha / 2)))
    return mean, mean - lo, hi - mean


def fig1_cdfs(feats, bnds, outdir):
    weights, _ = load_weights()
    n_ds = len(feats)
    fig, axes = plt.subplots(1, n_ds, figsize=(4 * n_ds, 3.5), sharey=True)
    if n_ds == 1: axes = [axes]
    for ax, (ds, d) in zip(axes, feats.items()):
        rmax = np.array([float(np.max(composite_risk(np.array(t["features"]), weights)))
                         for t in d["trajectories"]])
        ax.plot(np.sort(rmax), np.linspace(0, 1, len(rmax)),
                color="black", lw=1.8, label="empirical CDF")
        bsim_norm = {f"{float(k):.2f}": v for k, v in bnds[ds]["B_sim"].items()}
        for tau in TAUS:
            ls = LS_BY_TAU[tau]
            B_sim = bsim_norm[f"{tau:.2f}"]
            B_d = bnds[ds]["B_d"][f"{tau:.2f}"]
            ax.axvline(B_sim, color=SHADE_SIM, ls=ls, lw=1.4, alpha=0.95,
                       label=rf"$B^{{\mathrm{{sim}}}}_{{\tau={tau}}}$")
            ax.axvline(B_d, color=SHADE_DAT, ls=ls, lw=1.4, alpha=0.95,
                       label=rf"$B^d_{{\tau={tau}}}$")
        ax.set_title(ds)
        ax.set_xlabel(r"$R_{\max}$")
        ax.set_xlim(0, 1)
        ax.grid(True, color="0.85", alpha=0.6, lw=0.6)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("CDF")
    axes[-1].legend(loc="lower right", fontsize=7, framealpha=0.95,
                    handlelength=2.2, ncol=1)
    plt.tight_layout()
    out = outdir / "fig1_cdf_rmax.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"-> {out}")


def fig2_forest(bnds, outdir):
    rows = []
    for ds in bnds:
        for tau in TAUS:
            tau_hat = bnds[ds]["tau_hat_d"][f"{tau:.2f}"]
            lo, hi = bnds[ds]["ci"][f"{tau:.2f}"]["tau_hat_d"]
            rows.append((f"{ds}  $\\tau$={tau}", tau_hat - tau, lo - tau, hi - tau))
    labels = [r[0] for r in rows]
    diffs = [r[1] for r in rows]
    lows = [r[2] for r in rows]; his = [r[3] for r in rows]
    y = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(7, 0.38 * len(rows) + 1.5))
    ax.axvspan(-0.03, 0.03, color="0.75", alpha=0.55,
               label=r"$\pm 0.03$ tolerance")
    ax.errorbar(
        diffs, y,
        xerr=[np.array(diffs) - np.array(lows), np.array(his) - np.array(diffs)],
        fmt="o", color="black", ecolor="black", capsize=3,
        mfc="white", mec="black", mew=1.3, markersize=6,
    )
    ax.set_yticks(y); ax.set_yticklabels(labels)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel(r"$\hat\tau_d - \tau$")
    ax.legend(loc="lower right", fontsize=9, framealpha=0.95)
    ax.grid(True, axis="x", color="0.85", alpha=0.6, lw=0.6)
    ax.set_axisbelow(True)
    plt.tight_layout()
    out = outdir / "fig2_forest_coverage.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"-> {out}")


def fig3_h_off3_bars(feats, bnds, outdir):
    """Mean TTC<2s ticks per trajectory, crossing vs non-crossing, per (dataset, tau).

    Visualizes H_OFF3 directly: the dark bar (crossing) towers over the
    light bar (non-crossing) in every cell, even when the crossing
    sample is tiny (e.g., HighD tau=0.10 has only 8 crossing trajectories
    but their mean TTC<2s count dwarfs the 4,992 non-crossing ones).
    """
    weights, _ = load_weights()
    datasets = list(feats.keys())
    n_ds = len(datasets)
    fig, axes = plt.subplots(1, n_ds, figsize=(4 * n_ds, 3.6), sharey=False)
    if n_ds == 1: axes = [axes]

    width = 0.36
    x_positions = np.arange(len(TAUS))
    # 2026-07 addition: archive the exact plotted values (means, 95%
    # bootstrap CI endpoints, group sizes) so the figure is
    # independently reconstructible from the repository.
    values_dump = []

    for ax, ds in zip(axes, datasets):
        d = feats[ds]
        bsim_norm = {f"{float(k):.2f}": v for k, v in bnds[ds]["B_sim"].items()}

        # per-trajectory R_max and TTC<2s count
        r_max = np.zeros(len(d["trajectories"]))
        ttc_under2 = np.zeros(len(d["trajectories"]))
        for i, t in enumerate(d["trajectories"]):
            r_max[i] = float(np.max(composite_risk(np.array(t["features"]), weights)))
            ttc_raw = t.get("ttc_raw")
            if ttc_raw is None:
                ttc_under2[i] = 0
            else:
                ttc_arr = np.array(ttc_raw, dtype=float)
                # NaN/inf comparisons are False, so they don't count
                ttc_under2[i] = int(np.sum(ttc_arr < TTC_THRESH_S))

        means_cross, errs_cross_lo, errs_cross_hi, n_cross = [], [], [], []
        means_nc,    errs_nc_lo,    errs_nc_hi,    n_nc    = [], [], [], []

        for tau in TAUS:
            B_sim = bsim_norm[f"{tau:.2f}"]
            mask = r_max > B_sim
            cross = ttc_under2[mask]
            nc    = ttc_under2[~mask]
            mc, lo_c, hi_c = bootstrap_mean_ci(cross)
            mn, lo_n, hi_n = bootstrap_mean_ci(nc)
            means_cross.append(mc); errs_cross_lo.append(lo_c); errs_cross_hi.append(hi_c)
            n_cross.append(int(mask.sum()))
            means_nc.append(mn);    errs_nc_lo.append(lo_n);    errs_nc_hi.append(hi_n)
            n_nc.append(int((~mask).sum()))
            values_dump.append({
                "dataset": ds, "tau": tau,
                "n_crossing": int(mask.sum()),
                "n_noncrossing": int((~mask).sum()),
                "mean_crossing": mc,
                "ci95_crossing": [mc - lo_c, mc + hi_c],
                "mean_noncrossing": mn,
                "ci95_noncrossing": [mn - lo_n, mn + hi_n],
                "bootstrap": f"trajectory-level percentile bootstrap, "
                             f"{BOOT_N} within-group resamples, "
                             f"seed {BOOT_SEED}",
            })

        bars_c = ax.bar(
            x_positions - width / 2, means_cross, width,
            yerr=[errs_cross_lo, errs_cross_hi],
            color=SHADE_CROSS, edgecolor="black", linewidth=0.9,
            capsize=4, error_kw=dict(ecolor="black", lw=1.0),
            label=r"crossing ($R_{\max} > B^{\mathrm{sim}}$)",
        )
        bars_n = ax.bar(
            x_positions + width / 2, means_nc, width,
            yerr=[errs_nc_lo, errs_nc_hi],
            color=SHADE_NC, edgecolor="black", linewidth=0.9,
            capsize=4, error_kw=dict(ecolor="black", lw=1.0),
            label="non-crossing",
        )

        # Annotate n above each bar
        for i in range(len(TAUS)):
            top_c = means_cross[i] + errs_cross_hi[i]
            top_n = means_nc[i] + errs_nc_hi[i]
            ax.text(x_positions[i] - width / 2,
                    top_c + 0.04 * max(top_c, top_n, 1e-9),
                    f"n={n_cross[i]}", ha="center", va="bottom", fontsize=7)
            ax.text(x_positions[i] + width / 2,
                    top_n + 0.04 * max(top_c, top_n, 1e-9),
                    f"n={n_nc[i]}", ha="center", va="bottom", fontsize=7)

        ax.set_xticks(x_positions)
        ax.set_xticklabels([rf"$\tau$={tau}" for tau in TAUS])
        ax.set_title(ds)
        ax.set_ylabel(r"mean TTC$<$2\,s ticks per trajectory")
        ax.grid(True, axis="y", color="0.85", alpha=0.6, lw=0.6)
        ax.set_axisbelow(True)
        # Add a bit of headroom for the n-labels
        ymax = max(means_cross + means_nc) * 1.25 + max(errs_cross_hi + errs_nc_hi) * 1.1
        if ymax > 0:
            ax.set_ylim(0, ymax)

    axes[-1].legend(loc="upper right", fontsize=8, framealpha=0.95)
    plt.tight_layout()
    out = outdir / "fig3_h_off3_bars.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"-> {out}")
    vals_out = outdir / "fig3_h_off3_bars_values.json"
    with open(vals_out, "w") as f:
        json.dump(values_dump, f, indent=1)
    print(f"-> {vals_out}")


def fig4_transfer(trans, outdir):
    if trans is None:
        print("[fig4] no transfer matrix yet; skipping")
        return
    datasets = trans["datasets"]
    n = len(datasets)
    fig, axes = plt.subplots(1, len(TAUS), figsize=(4 * len(TAUS), 3.6))
    if len(TAUS) == 1: axes = [axes]
    for ax, tau in zip(axes, TAUS):
        M = np.zeros((n, n))
        for i, d1 in enumerate(datasets):
            for j, d2 in enumerate(datasets):
                M[i, j] = trans["by_tau"][f"{tau:.2f}"][f"{d1}__on__{d2}"]["realized_rate"]
        im = ax.imshow(M, vmin=0, vmax=1, cmap="Greys")
        ax.set_xticks(range(n)); ax.set_xticklabels(datasets, rotation=30)
        ax.set_yticks(range(n)); ax.set_yticklabels(datasets)
        ax.set_xlabel("evaluated on"); ax.set_ylabel("B trained on")
        ax.set_title(rf"$\tau$ = {tau}")
        for i in range(n):
            for j in range(n):
                txt_color = "white" if M[i, j] > 0.55 else "black"
                ax.text(j, i, f"{M[i, j]:.3f}",
                        ha="center", va="center",
                        color=txt_color, fontsize=9)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    out = outdir / "fig4_transfer_heatmaps.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"-> {out}")


if __name__ == "__main__":
    outdir = ROOT / "results/figures"
    outdir.mkdir(parents=True, exist_ok=True)
    feats, bnds, verds, trans = load_all()
    if feats and bnds:
        fig1_cdfs(feats, bnds, outdir)
        fig2_forest(bnds, outdir)
        fig3_h_off3_bars(feats, bnds, outdir)
    fig4_transfer(trans, outdir)
