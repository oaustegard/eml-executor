"""discover_hybrid: staged Option A → Option B formula discovery.

Tries Option A (softmax/argmax, clean symbolic output) first. If A's
post-snap RMSE exceeds a threshold, falls back to Option B (linear
coefficients, more expressive, lossy symbolic snap).

This is Strategy 1 from the issue #11 / Option B investigation:

    (x, y)  →  Option A (fast, clean symbolic)
            →  Option B fallback (slower, numerical fit)
            →  return best result with a flag showing which stage won

The user sees one API (`discover_hybrid`) and gets:
  - Clean symbolic output when Option A succeeds
  - A numerical near-fit when Option A fails architecturally
  - A `method` field indicating which stage produced the answer

Usage::

    from eml_sr_hybrid import discover_hybrid

    result = discover_hybrid(x, y)
    print(result["expr"], result["method"])
    # → 'exp(x)'  'option_a'          (clean symbolic)
    # → 'eml(...)'  'option_b'        (numerical fit)
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch

from eml_sr import DTYPE, REAL, discover
from eml_sr_linear import discover_linear


def discover_hybrid(
    x: np.ndarray,
    y: np.ndarray,
    max_depth: int = 4,
    n_tries_a: int = 8,
    n_tries_b: int = 4,
    max_depth_b: int = 3,
    fallback_threshold: float = 1e-6,
    verbose: bool = True,
) -> Optional[dict]:
    """Discover a formula relating x → y, using Option A first with
    Option B as a fallback.

    Args:
        x: input values (1D numpy array)
        y: output values (1D numpy array)
        max_depth: maximum tree depth for Option A (default 4)
        n_tries_a: random seeds per depth for Option A (default 8)
        n_tries_b: random seeds per depth for Option B (default 4)
        max_depth_b: maximum tree depth for Option B (default 3)
        fallback_threshold: RMSE below which Option A is accepted
            without falling back to Option B (default 1e-6)
        verbose: print progress

    Returns:
        dict with keys:
            expr: symbolic expression string
            depth: tree depth used
            snap_rmse: RMSE of the snapped tree
            snapped_tree: callable nn.Module
            method: 'option_a' or 'option_b'
            fit_rmse: (Option B only) pre-snap RMSE showing the
                      architecture's true fitting power
    """
    # ── Stage 1: Option A ──────────────────────────────────────
    if verbose:
        print("═══ Stage 1: Option A (softmax / clean symbolic) ═══")

    result_a = discover(
        x, y,
        max_depth=max_depth,
        n_tries=n_tries_a,
        verbose=verbose,
    )

    if result_a is not None and result_a["snap_rmse"] < fallback_threshold:
        if verbose:
            print(f"\n  ✓ Option A succeeded: {result_a['expr']}")
            print(f"    depth={result_a['depth']} "
                  f"rmse={result_a['snap_rmse']:.2e}")
        result_a["method"] = "option_a"
        return result_a

    a_rmse = result_a["snap_rmse"] if result_a else float("inf")
    a_expr = result_a["expr"] if result_a else None
    if verbose:
        print(f"\n  Option A best: rmse={a_rmse:.2e} → {a_expr}")
        print(f"  Above threshold {fallback_threshold:.0e}, "
              f"falling back to Option B.")

    # ── Stage 2: Option B ──────────────────────────────────────
    if verbose:
        print("\n═══ Stage 2: Option B (linear coefficients) ═══")

    result_b = discover_linear(
        x, y,
        max_depth=max_depth_b,
        n_tries=n_tries_b,
        verbose=verbose,
    )

    if result_b is None:
        # Both failed; return Option A's best attempt.
        if result_a is not None:
            result_a["method"] = "option_a"
            return result_a
        return None

    # Compute the *pre-snap* RMSE on the original data to show the
    # architecture's true fitting power (separate from snap quality).
    x_t = torch.tensor(x, dtype=REAL)
    with torch.no_grad():
        tree = result_b.get("snapped_tree")
        if tree is not None:
            # Use the un-snapped tree for fit_rmse if available.
            pred, _, _ = tree(x_t)
            fit_rmse = float(np.sqrt(np.mean(
                (pred.real.detach().numpy() - y) ** 2)))
        else:
            fit_rmse = result_b["snap_rmse"]

    # Pick the better result between A and B.
    b_rmse = result_b["snap_rmse"]
    if a_rmse < b_rmse and result_a is not None:
        if verbose:
            print(f"\n  Option A still wins on snap RMSE "
                  f"({a_rmse:.2e} < {b_rmse:.2e})")
        result_a["method"] = "option_a"
        return result_a

    if verbose:
        print(f"\n  ✓ Option B wins: snap_rmse={b_rmse:.2e} "
              f"fit_rmse={fit_rmse:.2e}")
        print(f"    depth={result_b['depth']} "
              f"expr={result_b['expr'][:80]}")

    result_b["method"] = "option_b"
    result_b["fit_rmse"] = fit_rmse
    return result_b
