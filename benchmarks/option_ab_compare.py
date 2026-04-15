"""Side-by-side comparison of Option A (softmax) vs Option B (linear).

Runs both architectures on a fixed set of targets that span:

  * Easy targets that Option A handles well (sanity / regression check)
  * Targets where Option A fails per the issue #11 diagnosis:
    - `exp(x) - 1` (needs constant `e`, unreachable in Option A's depth-1 vocab)
    - `y = x` (the identity gap — softmax cannot route through `eml(ln(x), 1)`)
    - `0.3 * x` (linear with non-trivial scale, needs negative γ on a child)
    - `exp(exp(exp(x)))` (the curriculum degenerate-snap case)

Reports two numbers per row:

  * `best_mse` — best MSE the architecture achieves during training, before
    any snap. This measures the *fitting* power of the parameterization.
  * `snap_rmse` — RMSE after snapping coefficients/logits to discrete choices.
    Option A snaps via argmax (clean). Option B snaps via per-coefficient
    rounding to the nearest named constant — this is currently lossy and
    is documented as future work in `eml_sr_linear.py`.

The point of the benchmark is to validate the **architectural** claim that
Option B can fit targets Option A cannot. Symbolic recovery via a smarter
snap is a separate problem.

Usage::

    python -m benchmarks.option_ab_compare
"""

from __future__ import annotations

import math
import sys
import os
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eml_sr import REAL, DTYPE, discover
from eml_sr_linear import _train_one_linear, EMLTree1DLinear


TARGETS = [
    # (name, fn, x_range, expected_min_depth_A, expected_min_depth_B)
    ("exp(x)",          lambda x: np.exp(x),          (0.5, 2.5), 1, 1),
    ("e (constant)",    lambda x: np.full_like(x, math.e), (0.5, 2.5), 1, 1),
    ("ln(x)",           lambda x: np.log(x),          (0.5, 5.0), 3, 2),
    ("exp(x) - ln(x)",  lambda x: np.exp(x) - np.log(x), (0.5, 3.0), 1, 1),
    ("exp(x) - 1",      lambda x: np.exp(x) - 1.0,    (0.0, 2.0), 1, 1),  # A fails
    ("y = x",           lambda x: x.copy(),            (0.5, 3.0), 4, 2),  # A fails
    ("0.3 * x",         lambda x: 0.3 * x,            (0.5, 5.0), 4, 3),  # A fails
    ("exp(exp(x))",     lambda x: np.exp(np.exp(x)),  (-1.0, 0.5), 2, 2),
]


def _option_a(name, fn, lo, hi, max_depth, n_tries=4):
    x = np.linspace(lo, hi, 30)
    y = fn(x)
    t0 = time.time()
    r = discover(x, y, max_depth=max_depth, n_tries=n_tries, verbose=False)
    elapsed = time.time() - t0
    return {
        "best_mse": r["snap_rmse"] ** 2,  # discover() reports normalized rmse
        "snap_rmse": r["snap_rmse"],
        "expr": r["expr"],
        "depth": r["depth"],
        "elapsed": elapsed,
    }


def _option_b(name, fn, lo, hi, max_depth, n_tries=4):
    """Option B: try each depth from 1 to max_depth, n_tries seeds each.
    Return the best (smallest best_mse) result found."""
    x = np.linspace(lo, hi, 30)
    y = fn(x)
    xt = torch.tensor(x, dtype=REAL)
    yt = torch.tensor(y, dtype=DTYPE)

    best = None
    t0 = time.time()
    for depth in range(1, max_depth + 1):
        for seed in range(n_tries):
            r = _train_one_linear(
                xt, yt, depth=depth, seed=seed,
                search_iters=3000, snap_iters=1000,
                lam_disc_max=0.5, lr=0.01,
            )
            r["depth"] = depth
            if best is None or r["best_mse"] < best["best_mse"]:
                best = r
        # Early-exit: if we found a near-perfect fit at this depth, stop.
        if best and best["best_mse"] < 1e-8:
            break
    elapsed = time.time() - t0

    # Compute RMSE in the same coordinate space as Option A reports
    # (target space, since neither method normalizes here).
    with torch.no_grad():
        pred, _, _ = best["tree"](xt)
        fit_rmse = float(np.sqrt(np.mean((pred.real.numpy() - y) ** 2)))

    return {
        "best_mse": best["best_mse"],
        "fit_rmse": fit_rmse,
        "snap_rmse": best["snap_rmse"],
        "expr": best["expr"],
        "depth": best["depth"],
        "elapsed": elapsed,
    }


def run():
    print("═══ Option A (softmax) vs Option B (linear coeffs) ═══\n")
    print(f"  {'target':18s}  {'A.fit':>10s}  {'A.depth':>8s}  "
          f"{'B.fit':>10s}  {'B.snap':>10s}  {'B.depth':>8s}  notes")
    print("  " + "─" * 95)

    for name, fn, (lo, hi), max_d_a, max_d_b in TARGETS:
        ra = _option_a(name, fn, lo, hi, max_depth=max(max_d_a, 4))
        rb = _option_b(name, fn, lo, hi, max_depth=max(max_d_b, 3))

        a_fit = ra["snap_rmse"]
        b_fit = rb["fit_rmse"]
        b_snap = rb["snap_rmse"]
        winner = "B" if b_fit < a_fit / 10 else ("A" if a_fit < b_fit / 10 else "tie")
        print(f"  {name:18s}  {a_fit:10.2e}  {ra['depth']:>8}  "
              f"{b_fit:10.2e}  {b_snap:10.2e}  {rb['depth']:>8}  "
              f"winner={winner}")

    print()
    print("Legend:")
    print("  A.fit  = Option A snap RMSE (clean argmax snap → final answer)")
    print("  B.fit  = Option B *pre-snap* RMSE (numerical fit only)")
    print("  B.snap = Option B *post-snap* RMSE (lossy per-coef snap; future work)")
    print()
    print("The interesting question is `B.fit` vs `A.fit` — that measures")
    print("the architectural expressivity gap. `B.snap` being worse than `B.fit`")
    print("just means our naive per-coefficient snap is broken; a smarter snap")
    print("(iterative pruning, brute-force discrete search, learned snap)")
    print("is the obvious follow-up.")


if __name__ == "__main__":
    run()
