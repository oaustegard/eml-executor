"""analog/sweep.py — depth × noise × formula sweep harness.

Runs the analog noise simulator across a grid of:
  - target formulas (transcendentals, arithmetic, mixed, stress)
  - per-node noise levels  (0.01 % … 5 %)
  - noise models  (additive, multiplicative, 1/f, matched-pairs)

Emits a tidy CSV:
  formula, depth, n_nodes, sigma, noise_model,
  trial, rmse, rmse_std, bits_of_precision,
  has_complex_intermediates

Usage
-----
    python -m analog.sweep                  # default output: analog/results/sweep.csv
    python -m analog.sweep --out results.csv --trials 200 --seed 42

The compiler (#31) is not yet landed; targets are hardcoded here using
eml_compiler.compile_expr.  When the compiler lands, replace TARGETS with
a loader from the compiled catalogue.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

import numpy as np

# allow running as a script from the repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from eml_compiler import compile_expr, tree_depth, tree_size
from analog.noise_sim import (
    AdditiveGaussian,
    MultiplicativeGaussian,
    OneOverF,
    MatchedPairs,
    simulate,
)

# ─── Target formula catalogue ──────────────────────────────────────
# Each entry: (label, expr_string, variables, x_sampler)
# x_sampler: callable() -> dict[str, np.ndarray]
#
# Domain choices avoid exp-overflow and log-of-negative in the ideal
# path while giving a representative range for each function.

_N = 60  # number of x-sample points


def _uni(lo: float, hi: float) -> dict:
    return {'x': np.linspace(lo, hi, _N)}


def _bi(xlo, xhi, ylo, yhi) -> dict:
    rng = np.random.default_rng(7)
    return {
        'x': rng.uniform(xlo, xhi, _N),
        'y': rng.uniform(ylo, yhi, _N),
    }


TARGETS: list[tuple[str, str, dict]] = [
    # (label, expr, x_samples_dict)
    # --- Transcendentals ---
    ("exp(x)",          "exp(x)",           _uni(0.1, 2.0)),
    ("ln(x)",           "ln(x)",            _uni(0.5, 4.0)),
    ("sqrt(x)",         "sqrt(x)",          _uni(0.5, 4.0)),
    # --- Arithmetic ---
    ("1/x",             "1/x",              _uni(0.5, 3.0)),
    ("x+y",             "x+y",              _bi(0.5, 2.0, 0.5, 2.0)),
    ("x*y",             "x*y",              _bi(0.5, 2.0, 0.5, 2.0)),
    ("x/y",             "x/y",              _bi(0.5, 3.0, 0.5, 3.0)),
    # --- Mixed ---
    ("eml(x,y)",        "eml(x,y)",         _bi(0.1, 1.5, 0.5, 3.0)),
    ("x^2+y^2",         "x^2+y^2",          _bi(0.5, 2.0, 0.5, 2.0)),
    # --- Stress ---
    ("exp(exp(x))",     "exp(exp(x))",      _uni(0.0, 1.5)),   # larger x saturates
]

# ─── Noise level grid ──────────────────────────────────────────────
# Expressed as a fraction (0.0001 = 0.01 %)
SIGMA_VALUES = [0.0001, 0.001, 0.005, 0.01, 0.02, 0.05]

# ─── Noise model factory ───────────────────────────────────────────

def _build_noise_models(sigma: float) -> list[tuple[str, object]]:
    return [
        ("additive_gaussian",      AdditiveGaussian(sigma)),
        ("multiplicative_gaussian", MultiplicativeGaussian(sigma)),
        ("1_over_f",               OneOverF(sigma=sigma, alpha=1.0, cutoff=0.05)),
        ("matched_pairs",          MatchedPairs(sigma_common=sigma * 0.5,
                                               sigma_diff=sigma * 0.5)),
    ]


# ─── Sweep ─────────────────────────────────────────────────────────

CSV_FIELDS = [
    "formula", "depth", "n_nodes", "sigma", "noise_model",
    "rmse", "rmse_std", "bits_of_precision", "failure_rate",
    "has_complex_intermediates",
]


def run_sweep(
    output_path: str | Path,
    n_trials: int = 500,
    seed: int = 42,
    verbose: bool = True,
) -> Path:
    """Run the full sweep and write results to `output_path`.

    Returns the path written.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rng_seed = seed
    t0 = time.time()

    total = len(TARGETS) * len(SIGMA_VALUES) * len(_build_noise_models(0.01))
    done = 0

    with open(output_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        writer.writeheader()

        for label, expr, x_samples in TARGETS:
            try:
                tree = compile_expr(expr)
            except Exception as e:
                if verbose:
                    print(f"  [skip] {label}: compile failed: {e}", flush=True)
                continue

            depth = tree_depth(tree)
            n_nodes = tree_size(tree)

            for sigma in SIGMA_VALUES:
                for nm_name, noise_model in _build_noise_models(sigma):
                    done += 1
                    if verbose:
                        pct = 100 * done / total
                        elapsed = time.time() - t0
                        print(
                            f"[{pct:5.1f}%] {label:18s} | σ={sigma:.4f} | "
                            f"{nm_name:26s} | elapsed {elapsed:.1f}s",
                            flush=True,
                        )

                    result = simulate(
                        tree=tree,
                        noise_model=noise_model,
                        x_samples=x_samples,
                        n_trials=n_trials,
                        seed=rng_seed,
                    )
                    rng_seed += 1  # different seed per cell

                    writer.writerow({
                        "formula":                  label,
                        "depth":                    depth,
                        "n_nodes":                  n_nodes,
                        "sigma":                    sigma,
                        "noise_model":              nm_name,
                        "rmse":                     result['rmse'],
                        "rmse_std":                 result['rmse_std'],
                        "bits_of_precision":        result['bits_of_precision'],
                        "failure_rate":             result.get('failure_rate', 0.0),
                        "has_complex_intermediates": int(result['has_complex_intermediates']),
                    })
                    fh.flush()

    if verbose:
        print(f"\nSweep complete in {time.time() - t0:.1f}s → {output_path}", flush=True)
    return output_path


# ─── CLI ───────────────────────────────────────────────────────────

def _default_out() -> str:
    here = Path(__file__).parent
    return str(here / "results" / "sweep.csv")


def main(argv=None):
    p = argparse.ArgumentParser(description="EML analog noise sweep")
    p.add_argument("--out",    default=_default_out(), help="Output CSV path")
    p.add_argument("--trials", type=int, default=500,  help="Monte Carlo trials per cell")
    p.add_argument("--seed",   type=int, default=42,   help="Base RNG seed")
    p.add_argument("--quiet",  action="store_true",    help="Suppress progress output")
    args = p.parse_args(argv)

    run_sweep(
        output_path=args.out,
        n_trials=args.trials,
        seed=args.seed,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
