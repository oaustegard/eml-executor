"""Demo: per-edge MLP distillation via eml-sr (issue #42).

Trains a small MLP on a handful of targets, then distills each into an
additive symbolic surrogate using ``eml_sr_distill.distill``. Reports:

  - MLP fit (train RMSE)
  - Per-slot recovery rate (fraction of per-feature partial-dependence
    curves recovered exactly by ``discover_curriculum``)
  - Surrogate vs MLP RMSE + R² on held-out data
  - Compression ratio (MLP params vs sum of surrogate tree nodes)

The targets are chosen to exercise three regimes:

  - Linear additive (``2x₀ + 3x₁``): trivial, 100% recovery expected.
  - Exp/log additive (``exp(x₀) + ln(x₁)``): eml-sr's natural vocabulary,
    each slot recovered as a canonical EML shape.
  - Non-additive (``x₀·x₁``): honest failure of the additive surrogate.
    R² will be <1; per-feature shapes are still interpretable but the
    composition does not represent the product.

Usage::

    python -m benchmarks.distill_demo
    python -m benchmarks.distill_demo --seed 1 --quick
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch

# Allow running as `python benchmarks/distill_demo.py` from the repo root
# or as `python -m benchmarks.distill_demo`.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eml_sr_distill import distill, train_mlp  # noqa: E402


@dataclass
class DemoTarget:
    name: str
    formula: str
    f: Callable[[np.ndarray], np.ndarray]
    range: tuple[float, float]
    activation: str
    hidden: tuple[int, ...]
    note: str = ""


TARGETS = [
    DemoTarget(
        name="linear_additive",
        formula="2 * x0 + 3 * x1",
        f=lambda X: 2.0 * X[:, 0] + 3.0 * X[:, 1],
        range=(-1.0, 1.0),
        activation="identity",
        hidden=(4,),
        note="Trivial additive baseline — 100% recovery expected.",
    ),
    DemoTarget(
        name="exp_plus_log",
        formula="exp(x0) + ln(x1)",
        f=lambda X: np.exp(X[:, 0]) + np.log(X[:, 1]),
        range=(0.3, 2.0),
        activation="tanh",
        hidden=(8,),
        note="Additive in eml-sr's natural vocabulary.",
    ),
    DemoTarget(
        name="product",
        formula="x0 * x1",
        f=lambda X: X[:, 0] * X[:, 1],
        range=(0.3, 2.0),
        activation="tanh",
        hidden=(8,),
        note="Non-additive. Best additive approximation; R² < 1 expected.",
    ),
]


def run_one(t: DemoTarget, *, n: int, seed: int, max_depth: int,
            n_tries: int, n_grid: int, verbose: bool) -> dict:
    rng = np.random.default_rng(seed)
    lo, hi = t.range
    X = rng.uniform(lo, hi, size=(n, 2))
    y = t.f(X)

    t0 = time.time()
    model = train_mlp(X, y, hidden_sizes=t.hidden, activation=t.activation,
                      epochs=3000, lr=1e-2, seed=seed)
    t_train = time.time() - t0

    with torch.no_grad():
        mlp_pred = model(torch.tensor(X, dtype=torch.float64)).cpu().numpy()
    mlp_rmse = float(np.sqrt(np.mean((mlp_pred - y) ** 2)))

    t1 = time.time()
    rep = distill(model, X, y, method="curriculum", max_depth=max_depth,
                  n_tries=n_tries, n_grid=n_grid, verbose=verbose)
    t_distill = time.time() - t1

    return {
        "target": t,
        "mlp_rmse": mlp_rmse,
        "train_sec": t_train,
        "distill_sec": t_distill,
        "report": rep,
    }


def print_result(r: dict) -> None:
    t: DemoTarget = r["target"]
    rep = r["report"]
    print(f"\n╒═ {t.name} ═══  y = {t.formula}  ({t.note})")
    print(f"│  MLP train RMSE:   {r['mlp_rmse']:.3e}   "
          f"(params={rep.mlp_params},  train {r['train_sec']:.1f}s)")
    print(f"│  Recovery rate:    {rep.recovery_rate * 100:.1f}%   "
          f"({sum(sr.exact for sr in rep.per_feature)}/{len(rep.per_feature)} "
          f"slots exact,  distill {r['distill_sec']:.1f}s)")
    print(f"│  Compression:      {rep.compression_ratio:.2f}x   "
          f"({rep.mlp_params} params → {rep.surrogate_nodes} nodes)")
    print(f"│  Surrogate vs MLP: RMSE={rep.surrogate_rmse_on_mlp:.3e}  "
          f"R²={rep.surrogate_r2_on_mlp:.4f}")
    for sr in rep.per_feature:
        tag = " ✓" if sr.exact else ""
        print(f"│    x{sr.slot.feature_idx}: {sr.expr}  "
              f"(rmse={sr.snap_rmse:.3e}, depth={sr.depth}){tag}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-samples", type=int, default=300)
    ap.add_argument("--max-depth", type=int, default=5)
    ap.add_argument("--n-tries", type=int, default=6)
    ap.add_argument("--n-grid", type=int, default=60)
    ap.add_argument("--quick", action="store_true",
                    help="Small budgets (~20s total).")
    ap.add_argument("--target", type=str, default=None,
                    help="Run just one named target.")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    if args.quick:
        args.n_samples = 120
        args.max_depth = 3
        args.n_tries = 3
        args.n_grid = 30

    targets = TARGETS
    if args.target:
        targets = [t for t in TARGETS if t.name == args.target]
        if not targets:
            print(f"unknown target {args.target!r}. Choices: "
                  f"{[t.name for t in TARGETS]}")
            sys.exit(2)

    print(f"eml-sr distill demo — issue #42")
    print(f"  n_samples={args.n_samples}  max_depth={args.max_depth}  "
          f"n_tries={args.n_tries}  n_grid={args.n_grid}  seed={args.seed}")

    for t in targets:
        r = run_one(t, n=args.n_samples, seed=args.seed,
                    max_depth=args.max_depth, n_tries=args.n_tries,
                    n_grid=args.n_grid, verbose=args.verbose)
        print_result(r)


if __name__ == "__main__":
    main()
