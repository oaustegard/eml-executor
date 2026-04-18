"""analog/analyze.py — post-sweep analysis and plot generation.

Reads the CSV produced by analog/sweep.py and generates:

  1. rmse_vs_depth.png   — RMSE vs EML depth, one curve per noise level
                            (log-log scale, additive-Gaussian model only)
  2. bits_vs_depth.png   — Effective bits of precision vs depth,
                            with 10-bit, 8-bit, 6-bit reference lines
  3. heatmap_<model>.png — Per-formula × σ heatmap coloured by
                            usable / marginal / failed, one per noise model
  4. matched_pairs.png   — RMSE comparison: matched-pairs vs additive
                            at the same effective sigma (shows CMRR benefit)

Thresholds (bits of precision):
  usable   : bits >= 8
  marginal : 6 <= bits < 8
  failed   : bits < 6

Usage
-----
    python -m analog.analyze                        # default paths
    python -m analog.analyze --csv path/to/sweep.csv --out plots/
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# ─── Thresholds ────────────────────────────────────────────────────

BITS_USABLE   = 8.0
BITS_MARGINAL = 6.0

VIABILITY_LABELS = ["usable", "marginal", "failed"]
VIABILITY_COLORS = ["#2ecc71", "#f39c12", "#e74c3c"]  # green, orange, red


def _classify(bits: float) -> int:
    """0=usable, 1=marginal, 2=failed."""
    if bits >= BITS_USABLE:
        return 0
    elif bits >= BITS_MARGINAL:
        return 1
    return 2


# ─── Plot 1: RMSE vs depth (log-log) ──────────────────────────────

def plot_rmse_vs_depth(df: pd.DataFrame, out_dir: Path) -> None:
    ag = df[df["noise_model"] == "additive_gaussian"].copy()
    if ag.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    sigmas = sorted(ag["sigma"].unique())
    cmap = plt.cm.viridis(np.linspace(0, 1, len(sigmas)))

    for sigma, color in zip(sigmas, cmap):
        sub = ag[ag["sigma"] == sigma].groupby("depth")["rmse"].mean().reset_index()
        sub = sub[sub["rmse"] > 0].sort_values("depth")
        if sub.empty:
            continue
        label = f"σ = {sigma * 100:.2g}%"
        ax.loglog(sub["depth"], sub["rmse"], marker="o", color=color, label=label)

    # √depth reference line
    depths = np.arange(1, ag["depth"].max() + 2)
    ref_sigma = sigmas[len(sigmas) // 2]
    ag_ref = ag[ag["sigma"] == ref_sigma]
    if not ag_ref.empty:
        d1_rmse = ag_ref[ag_ref["depth"] == ag_ref["depth"].min()]["rmse"].mean()
        if d1_rmse > 0:
            ref_line = d1_rmse * np.sqrt(depths / depths[0])
            ax.loglog(depths, ref_line, "k--", alpha=0.5, label=r"$\propto\!\sqrt{k}$ (additive theory)")

    ax.set_xlabel("EML tree depth  k")
    ax.set_ylabel("Mean RMSE  (absolute)")
    ax.set_title("RMSE vs EML depth — additive Gaussian noise")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "rmse_vs_depth.png", dpi=150)
    plt.close(fig)


# ─── Plot 2: bits of precision vs depth ────────────────────────────

def plot_bits_vs_depth(df: pd.DataFrame, out_dir: Path) -> None:
    ag = df[df["noise_model"] == "additive_gaussian"].copy()
    if ag.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    sigmas = sorted(ag["sigma"].unique())
    cmap = plt.cm.plasma(np.linspace(0.1, 0.9, len(sigmas)))

    for sigma, color in zip(sigmas, cmap):
        sub = ag[ag["sigma"] == sigma].groupby("depth")["bits_of_precision"].mean().reset_index()
        sub = sub[np.isfinite(sub["bits_of_precision"])].sort_values("depth")
        if sub.empty:
            continue
        label = f"σ = {sigma * 100:.2g}%"
        ax.plot(sub["depth"], sub["bits_of_precision"], marker="o", color=color, label=label)

    for bits, style, txt in [
        (10, "k-",  "10-bit"),
        (8,  "k--", " 8-bit"),
        (6,  "k:",  " 6-bit"),
    ]:
        ax.axhline(bits, linestyle=style.lstrip("k"), color="k", alpha=0.5, linewidth=1)
        ax.text(ax.get_xlim()[0] + 0.05, bits + 0.15, txt, fontsize=8, color="k", alpha=0.7)

    ax.set_xlabel("EML tree depth  k")
    ax.set_ylabel("Effective bits of precision")
    ax.set_title("Bits of precision vs EML depth — additive Gaussian")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "bits_vs_depth.png", dpi=150)
    plt.close(fig)


# ─── Plot 3: per-formula heatmaps ─────────────────────────────────

def plot_heatmaps(df: pd.DataFrame, out_dir: Path) -> None:
    for model_name in df["noise_model"].unique():
        sub = df[df["noise_model"] == model_name].copy()
        sub["viability"] = sub["bits_of_precision"].apply(_classify).astype(float)

        formulas = sorted(sub["formula"].unique(), key=lambda f: sub[sub["formula"] == f]["depth"].iloc[0])
        sigmas = sorted(sub["sigma"].unique())

        matrix = np.full((len(formulas), len(sigmas)), np.nan)
        for i, formula in enumerate(formulas):
            for j, sigma in enumerate(sigmas):
                cell = sub[(sub["formula"] == formula) & (sub["sigma"] == sigma)]
                if not cell.empty:
                    matrix[i, j] = cell["viability"].iloc[0]

        fig, ax = plt.subplots(figsize=(max(7, len(sigmas) * 1.4), max(4, len(formulas) * 0.5 + 1)))
        cmap = mcolors.ListedColormap(VIABILITY_COLORS)
        norm = mcolors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)
        im = ax.imshow(matrix, cmap=cmap, norm=norm, aspect="auto")

        ax.set_xticks(range(len(sigmas)))
        ax.set_xticklabels([f"{s * 100:.3g}%" for s in sigmas], fontsize=8)
        ax.set_yticks(range(len(formulas)))
        depth_labels = []
        for f in formulas:
            d = sub[sub["formula"] == f]["depth"].iloc[0]
            depth_labels.append(f"{f}  (d={d})")
        ax.set_yticklabels(depth_labels, fontsize=8)
        ax.set_xlabel("Per-node σ")
        ax.set_title(f"Analog viability — {model_name}")

        from matplotlib.patches import Patch
        legend_handles = [
            Patch(color=VIABILITY_COLORS[0], label=f"usable   (≥{BITS_USABLE:.0f} bits)"),
            Patch(color=VIABILITY_COLORS[1], label=f"marginal ({BITS_MARGINAL:.0f}–{BITS_USABLE:.0f} bits)"),
            Patch(color=VIABILITY_COLORS[2], label=f"failed   (<{BITS_MARGINAL:.0f} bits)"),
        ]
        ax.legend(handles=legend_handles, loc="upper left", bbox_to_anchor=(1.01, 1),
                  fontsize=8, frameon=True)

        fig.tight_layout()
        safe_name = model_name.replace(" ", "_").replace("/", "_")
        fig.savefig(out_dir / f"heatmap_{safe_name}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)


# ─── Plot 4: matched-pairs vs additive comparison ──────────────────

def plot_matched_pairs(df: pd.DataFrame, out_dir: Path) -> None:
    ag = df[df["noise_model"] == "additive_gaussian"].copy()
    mp = df[df["noise_model"] == "matched_pairs"].copy()
    if ag.empty or mp.empty:
        return

    formulas = df["formula"].unique()
    sigmas = sorted(df["sigma"].unique())

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: RMSE ratio (matched_pairs / additive_gaussian) per sigma
    ax = axes[0]
    ratios_by_sigma = {}
    for sigma in sigmas:
        rows = []
        for formula in formulas:
            ag_rmse = ag[(ag["formula"] == formula) & (ag["sigma"] == sigma)]["rmse"]
            mp_rmse = mp[(mp["formula"] == formula) & (mp["sigma"] == sigma)]["rmse"]
            if ag_rmse.empty or mp_rmse.empty:
                continue
            ag_v = ag_rmse.iloc[0]
            mp_v = mp_rmse.iloc[0]
            if ag_v > 0 and np.isfinite(ag_v) and np.isfinite(mp_v):
                rows.append(mp_v / ag_v)
        ratios_by_sigma[sigma] = rows

    sigma_labels = [f"{s * 100:.3g}%" for s in sigmas]
    bp_data = [ratios_by_sigma[s] for s in sigmas if ratios_by_sigma[s]]
    bp_labels = [f"{s * 100:.3g}%" for s in sigmas if ratios_by_sigma[s]]
    if bp_data:
        ax.boxplot(bp_data, labels=bp_labels, patch_artist=True,
                   medianprops={"color": "black"},
                   boxprops={"facecolor": "#3498db", "alpha": 0.7})
    ax.axhline(1.0, color="red", linestyle="--", alpha=0.7, label="no benefit")
    ax.set_xlabel("Per-node σ")
    ax.set_ylabel("RMSE ratio  (matched-pairs / additive)")
    ax.set_title("Matched-pairs benefit: RMSE ratio < 1 = improvement")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # Right: bits-of-precision comparison at median sigma
    ax2 = axes[1]
    mid_sigma = sigmas[len(sigmas) // 2]
    ag_bits = ag[ag["sigma"] == mid_sigma].set_index("formula")["bits_of_precision"]
    mp_bits = mp[mp["sigma"] == mid_sigma].set_index("formula")["bits_of_precision"]
    common = sorted(set(ag_bits.index) & set(mp_bits.index),
                    key=lambda f: df[df["formula"] == f]["depth"].iloc[0])
    if common:
        x = np.arange(len(common))
        w = 0.35
        ax2.bar(x - w/2, [ag_bits.get(f, 0) for f in common], w,
                label="additive_gaussian", color="#e74c3c", alpha=0.7)
        ax2.bar(x + w/2, [mp_bits.get(f, 0) for f in common], w,
                label="matched_pairs", color="#3498db", alpha=0.7)
        ax2.axhline(BITS_USABLE,   linestyle="--", color="k", alpha=0.5, linewidth=1)
        ax2.axhline(BITS_MARGINAL, linestyle=":",  color="k", alpha=0.5, linewidth=1)
        ax2.set_xticks(x)
        ax2.set_xticklabels(common, rotation=30, ha="right", fontsize=8)
        ax2.set_ylabel("Effective bits of precision")
        ax2.set_title(f"Bits comparison at σ = {mid_sigma * 100:.2g}%")
        ax2.legend(fontsize=8)
        ax2.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / "matched_pairs.png", dpi=150)
    plt.close(fig)


# ─── Main ──────────────────────────────────────────────────────────

def analyze(csv_path: str | Path, out_dir: str | Path, verbose: bool = True) -> None:
    csv_path = Path(csv_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not csv_path.exists():
        raise FileNotFoundError(f"Sweep CSV not found: {csv_path}\n"
                                "Run  python -m analog.sweep  first.")

    df = pd.read_csv(csv_path)
    df["bits_of_precision"] = pd.to_numeric(df["bits_of_precision"], errors="coerce")
    df["rmse"] = pd.to_numeric(df["rmse"], errors="coerce")

    if verbose:
        print(f"Loaded {len(df)} rows from {csv_path}")
        print(f"Formulas: {sorted(df['formula'].unique())}")
        print(f"σ values: {sorted(df['sigma'].unique())}")
        print(f"Models:   {sorted(df['noise_model'].unique())}")

    plot_rmse_vs_depth(df, out_dir)
    if verbose:
        print("  → rmse_vs_depth.png")

    plot_bits_vs_depth(df, out_dir)
    if verbose:
        print("  → bits_vs_depth.png")

    plot_heatmaps(df, out_dir)
    if verbose:
        print("  → heatmap_*.png")

    plot_matched_pairs(df, out_dir)
    if verbose:
        print("  → matched_pairs.png")

    # Print a brief summary table
    if verbose:
        print("\n── Summary (additive_gaussian) ──────────────────────────────")
        ag = df[df["noise_model"] == "additive_gaussian"]
        for sigma in sorted(ag["sigma"].unique()):
            sub = ag[ag["sigma"] == sigma]
            n_usable   = (sub["bits_of_precision"] >= BITS_USABLE).sum()
            n_marginal = ((sub["bits_of_precision"] >= BITS_MARGINAL) &
                          (sub["bits_of_precision"] < BITS_USABLE)).sum()
            n_failed   = (sub["bits_of_precision"] < BITS_MARGINAL).sum()
            print(f"  σ={sigma*100:5.3g}%  "
                  f"usable={n_usable}  marginal={n_marginal}  failed={n_failed}")

    if verbose:
        print(f"\nPlots written to {out_dir}/")


def _default_csv() -> str:
    here = Path(__file__).parent
    return str(here / "results" / "sweep.csv")


def _default_out() -> str:
    here = Path(__file__).parent
    return str(here / "results" / "plots")


def main(argv=None):
    p = argparse.ArgumentParser(description="Analyze EML analog sweep results")
    p.add_argument("--csv", default=_default_csv(), help="Input sweep CSV")
    p.add_argument("--out", default=_default_out(), help="Output directory for plots")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args(argv)
    analyze(args.csv, args.out, verbose=not args.quiet)


if __name__ == "__main__":
    main()
