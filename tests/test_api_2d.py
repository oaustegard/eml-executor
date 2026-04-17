"""Test 2D-input public API (issue #16).

Verifies the public discover / discover_linear / discover_hybrid API
accepts both 1D ``(n_samples,)`` and 2D ``(n_samples, n_vars)`` inputs,
and that all return dicts carry ``n_vars``.

The tree-level multivariate support (EMLTree1D / EMLTree1DLinear with
n_vars>1) is tested in test_multivariate.py; this file is scoped to
the public-API surface.

Acceptance criteria from issue #16:
    - discover(x_1d, y) still works (backward compat)
    - discover(X_2d, y) works when X has 2+ columns
    - Same for discover_linear, discover_hybrid
    - result["n_vars"] is present in all return dicts
"""
from __future__ import annotations

import numpy as np
import pytest

from eml_sr import discover
from eml_sr_linear import discover_linear
from eml_sr_hybrid import discover_hybrid


# Keep budgets small — these tests are about API shape, not
# discovery fidelity. Full multivariate discovery quality is
# gated on #18 (curriculum) and #19 (sklearn wrapper), not #16.
_QUICK = dict(max_depth=2, n_tries=2, verbose=False)


# ─── Backward compat: 1D input still works ─────────────────────

class TestDiscover1DBackwardCompat:
    """discover() / discover_linear() / discover_hybrid() accept 1D x."""

    def test_discover_1d(self):
        x = np.linspace(0.5, 3.0, 20)
        y = np.exp(x)
        r = discover(x, y, **_QUICK)
        assert r is not None
        assert r["n_vars"] == 1

    def test_discover_linear_1d(self):
        x = np.linspace(0.5, 3.0, 20)
        y = np.exp(x)
        r = discover_linear(x, y, max_depth=2, n_tries=2, verbose=False)
        assert r is not None
        assert r["n_vars"] == 1

    def test_discover_hybrid_1d(self):
        x = np.linspace(0.5, 3.0, 20)
        y = np.exp(x)
        r = discover_hybrid(
            x, y,
            max_depth=2, n_tries_a=2, n_tries_b=2, max_depth_b=2,
            verbose=False,
        )
        assert r is not None
        assert r["n_vars"] == 1


# ─── 2D input: new capability ──────────────────────────────────

class TestDiscover2D:
    """discover() accepts X of shape (n_samples, n_vars)."""

    @pytest.mark.parametrize("n_vars", [2, 3])
    def test_discover_2d_runs(self, n_vars):
        rng = np.random.default_rng(0)
        X = rng.uniform(0.5, 2.0, (20, n_vars))
        y = np.exp(X[:, 0])  # only first var matters; shape is what's tested
        r = discover(X, y, **_QUICK)
        assert r is not None
        assert r["n_vars"] == n_vars

    @pytest.mark.parametrize("n_vars", [2, 3])
    def test_discover_linear_2d_runs(self, n_vars):
        rng = np.random.default_rng(0)
        X = rng.uniform(0.5, 2.0, (20, n_vars))
        y = np.exp(X[:, 0])
        r = discover_linear(X, y, max_depth=2, n_tries=2, verbose=False)
        assert r is not None
        assert r["n_vars"] == n_vars

    def test_discover_hybrid_2d_runs(self):
        rng = np.random.default_rng(0)
        X = rng.uniform(0.5, 2.0, (20, 2))
        y = np.exp(X[:, 0])
        r = discover_hybrid(
            X, y,
            max_depth=2, n_tries_a=2, n_tries_b=2, max_depth_b=2,
            verbose=False,
        )
        assert r is not None
        assert r["n_vars"] == 2


# ─── Return-dict shape: n_vars always present ──────────────────

class TestReturnDictNVars:
    """All success paths and all fallback paths include n_vars."""

    def test_discover_exact_path_has_n_vars(self):
        # y = exp(x) is recovered at shallow depth, exercising the
        # "exact formula found" early-return branch.
        x = np.linspace(0.5, 2.0, 20)
        y = np.exp(x)
        r = discover(x, y, max_depth=2, n_tries=4, verbose=False)
        assert r is not None
        assert "n_vars" in r
        assert r["n_vars"] == 1

    def test_discover_fallback_path_has_n_vars(self):
        # A target that won't recover exactly at this depth budget,
        # forcing the fallback return branch.
        rng = np.random.default_rng(42)
        x = np.linspace(-1.0, 1.0, 20)
        y = rng.uniform(0, 1, 20)  # noise — won't find exact match
        r = discover(x, y, max_depth=1, n_tries=2, verbose=False,
                     success_threshold=1e-20)
        assert r is not None
        assert "n_vars" in r
        assert r["n_vars"] == 1
        assert r.get("exact") is False

    def test_discover_linear_fallback_has_n_vars(self):
        rng = np.random.default_rng(42)
        x = np.linspace(-1.0, 1.0, 20)
        y = rng.uniform(0, 1, 20)
        r = discover_linear(x, y, max_depth=1, n_tries=2, verbose=False,
                            success_threshold=1e-20)
        assert r is not None
        assert "n_vars" in r


# ─── Validation ────────────────────────────────────────────────

class TestInputValidation:
    """discover/discover_linear/discover_hybrid reject invalid shapes."""

    def test_discover_rejects_3d(self):
        X = np.ones((10, 2, 2))
        y = np.ones(10)
        with pytest.raises(ValueError, match="1D or 2D"):
            discover(X, y, **_QUICK)

    def test_discover_linear_rejects_3d(self):
        X = np.ones((10, 2, 2))
        y = np.ones(10)
        with pytest.raises(ValueError, match="1D or 2D"):
            discover_linear(X, y, max_depth=1, n_tries=1, verbose=False)

    def test_discover_hybrid_rejects_3d(self):
        X = np.ones((10, 2, 2))
        y = np.ones(10)
        with pytest.raises(ValueError, match="1D or 2D"):
            discover_hybrid(X, y, max_depth=1, n_tries_a=1, n_tries_b=1,
                            max_depth_b=1, verbose=False)
