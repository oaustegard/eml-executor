"""Test multivariate support for GrowingEMLTree + discover_curriculum (issue #18).

Verifies:
    1. GrowingEMLTree(n_vars=1) construction/behavior matches the pre-#18 shapes.
    2. GrowingEMLTree(n_vars=3) constructs with the right logit shapes and
       forwards 2D input correctly.
    3. split_leaf routes the requested variable through the new subtree.
    4. leaf_gradients returns per-variable gradient vectors.
    5. discover_curriculum(X_2d, y) runs end-to-end and reports n_vars.
    6. Recovery: y = exp(x1) - ln(x2) at depth 1 (trivial multivariate case,
       the literal definition of eml).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from eml_sr import (
    DTYPE,
    REAL,
    GrowingEMLTree,
    discover_curriculum,
    eml_op,
)


# ═════════════════════ GrowingEMLTree shape/construction ═════════════════════

class TestGrowingTreeMultivariate:
    """Shape and construction invariants for GrowingEMLTree with n_vars > 1."""

    def test_n_vars_1_backcompat_shapes(self):
        """Default n_vars=1: leaf (2,), gate (2, 3). Matches pre-#18 surface."""
        t = GrowingEMLTree()
        assert t.n_vars == 1
        for key, p in t._params.items():
            if key.startswith("leaf"):
                assert p.shape == (2,), f"leaf {key}: {p.shape}"
            else:
                assert p.shape == (2, 3), f"gate {key}: {p.shape}"

    @pytest.mark.parametrize("n_vars", [2, 3, 5])
    def test_multivariate_shapes(self, n_vars):
        t = GrowingEMLTree(n_vars=n_vars)
        assert t.n_vars == n_vars
        for key, p in t._params.items():
            if key.startswith("leaf"):
                assert p.shape == (n_vars + 1,), f"leaf: {p.shape}"
            else:
                assert p.shape == (2, n_vars + 2), f"gate: {p.shape}"


class TestGrowingTreeForward:
    """Forward pass: 1D input (n_vars=1 back-compat) and 2D input (n_vars>=1)."""

    def test_forward_1d_input_nvars1(self):
        """Legacy 1D-tensor input path still works for n_vars=1."""
        t = GrowingEMLTree()
        x = torch.linspace(0.1, 2.0, 10, dtype=REAL)
        out, leaf_probs, gate_probs = t(x)
        assert out.shape == (10,)
        assert out.dtype == DTYPE
        # Root = 1 internal + 2 leaves all active
        assert len(leaf_probs) == 2
        assert len(gate_probs) == 1

    def test_forward_1d_and_2d_agree_nvars1(self):
        """x of shape (n,) and (n, 1) should produce identical outputs."""
        t = GrowingEMLTree()
        x = torch.linspace(0.1, 2.0, 10, dtype=REAL)
        y1, _, _ = t(x)
        y2, _, _ = t(x.unsqueeze(1))
        torch.testing.assert_close(y1, y2)

    @pytest.mark.parametrize("n_vars", [2, 3])
    def test_forward_2d_input(self, n_vars):
        t = GrowingEMLTree(n_vars=n_vars)
        X = torch.randn(20, n_vars, dtype=REAL)
        out, _, _ = t(X)
        assert out.shape == (20,)
        assert out.dtype == DTYPE

    def test_forward_shape_mismatch_raises(self):
        t = GrowingEMLTree(n_vars=3)
        X = torch.randn(8, 2, dtype=REAL)  # wrong width
        with pytest.raises(ValueError, match="n_vars"):
            t(X)


class TestGrowingTreeSplit:
    """split_leaf routes the chosen variable through the new subtree."""

    def test_split_routes_requested_variable(self):
        """After split_leaf(leaf, var_idx=v), the subtree's snap reads eml(x_{v+1}, 1)."""
        torch.manual_seed(0)
        t = GrowingEMLTree(n_vars=3)
        root = t.nodes[t.root]
        left_leaf = root["left"]
        new_int = t.split_leaf(left_leaf, var_idx=1)  # route x2

        snp = t.snap()
        sub_expr = snp._expr_at(new_int)
        assert sub_expr == "eml(x2, 1)"

    def test_split_leaf_defaults_to_x1(self):
        t = GrowingEMLTree(n_vars=3)
        leaf = t.nodes[t.root]["left"]
        new_int = t.split_leaf(leaf)  # default var_idx=0 → x1
        snp = t.snap()
        assert snp._expr_at(new_int) == "eml(x1, 1)"

    def test_split_leaf_var_idx_bounds(self):
        t = GrowingEMLTree(n_vars=2)
        leaf = t.nodes[t.root]["left"]
        with pytest.raises(ValueError, match="out of range"):
            t.split_leaf(leaf, var_idx=5)

    def test_split_preserves_univariate_behavior(self):
        """For n_vars=1, split_leaf still yields exp(x) subtree — same as pre-#18."""
        torch.manual_seed(0)
        t = GrowingEMLTree()  # n_vars=1
        leaf = t.nodes[t.root]["left"]
        new_int = t.split_leaf(leaf)
        snp = t.snap()
        # Univariate uses "x" (no index), per _expr_at convention.
        assert snp._expr_at(new_int) == "eml(x, 1)"


class TestLeafGradients:
    """leaf_gradients returns (n_vars+1,) per-variable gradients per active leaf."""

    def test_gradient_shapes(self):
        torch.manual_seed(0)
        t = GrowingEMLTree(n_vars=3)
        X = torch.randn(32, 3, dtype=REAL)
        y = torch.randn(32, dtype=REAL).to(DTYPE)
        grads = t.leaf_gradients(X, y, tau=1.0)
        assert set(grads.keys()) == set(t.active_leaves())
        for idx, g in grads.items():
            assert g.shape == (4,), f"leaf {idx}: {g.shape}"

    def test_gradient_magnitudes_consistent(self):
        """leaf_gradient_magnitudes == ||leaf_gradients||."""
        torch.manual_seed(1)
        t = GrowingEMLTree(n_vars=2)
        X = torch.randn(24, 2, dtype=REAL)
        y = torch.randn(24, dtype=REAL).to(DTYPE)
        grads = t.leaf_gradients(X, y, tau=1.0)
        # leaf_gradients consumed gradients; compute magnitudes fresh.
        mags = t.leaf_gradient_magnitudes(X, y, tau=1.0)
        for idx, g in grads.items():
            assert mags[idx] == pytest.approx(float(g.norm().item()), rel=1e-10)


# ═════════════════════════ Hand-crafted numerical check ═════════════════════

class TestHandCraftedMultivariate:
    """Hand-set a depth-1 GrowingEMLTree to compute eml(x1, x2) numerically."""

    def test_eml_x1_x2_exact(self):
        torch.manual_seed(0)
        t = GrowingEMLTree(n_vars=2, init_scale=0.0)
        # Leaves: [0] → x1 (idx 1 of 3), [1] → x2 (idx 2 of 3). Root gate
        # routes both children through (idx 3, "child").
        k = 50.0
        # Active leaves are nodes 0 and 1, root is node 2.
        with torch.no_grad():
            left_leaf_key = t.nodes[0]["key"]
            right_leaf_key = t.nodes[1]["key"]
            root_key = t.nodes[t.root]["key"]

            left_init = torch.full((3,), -k, dtype=REAL)
            left_init[1] = k  # x1
            t._params[left_leaf_key].copy_(left_init)

            right_init = torch.full((3,), -k, dtype=REAL)
            right_init[2] = k  # x2
            t._params[right_leaf_key].copy_(right_init)

            # Gate (2, 4): indices are [1, x1, x2, child].
            g = torch.full((2, 4), -k, dtype=REAL)
            g[0, 3] = k  # left side → child  (→ x1)
            g[1, 3] = k  # right side → child (→ x2)
            t._params[root_key].copy_(g)

        x1 = torch.tensor([0.5, 1.0, 2.0], dtype=REAL)
        x2 = torch.tensor([1.0, 2.0, 0.5], dtype=REAL)
        X = torch.stack([x1, x2], dim=1)
        out, _, _ = t(X, tau=0.01)
        expected = eml_op(x1.to(DTYPE), x2.to(DTYPE))
        torch.testing.assert_close(out, expected, atol=1e-8, rtol=1e-8)

        # And the symbolic form should read eml(x1, x2) or its
        # simplifier-expanded equivalent exp(x1) - ln(x2).
        snp = t.snap()
        expr = snp.to_expr()
        assert expr in ("eml(x1, x2)", "(exp(x1) - ln(x2))"), \
            f"unexpected expression: {expr!r}"


# ═════════════════════════ discover_curriculum end-to-end ═══════════════════

class TestDiscoverCurriculum2D:
    """discover_curriculum accepts 1D and 2D x, reports n_vars."""

    def test_discover_curriculum_1d_backcompat(self):
        """1D x still works. Recovery test: y = exp(x) at depth 1."""
        x = np.linspace(0.2, 2.0, 32)
        y = np.exp(x)
        r = discover_curriculum(
            x, y,
            max_depth=2, n_tries=3,
            verbose=False,
        )
        assert r is not None
        assert r["n_vars"] == 1
        # exp should be recoverable at depth 1 from random init.
        assert r["exact"] is True
        assert r["snap_rmse"] < 1e-6

    def test_discover_curriculum_2d_shape_only(self):
        """2D x runs end-to-end. We don't demand recovery here — this is
        the API shape test. Recovery is a separate, depth-1 test below."""
        rng = np.random.default_rng(0)
        X = rng.uniform(0.5, 2.0, size=(24, 2))
        # Some arbitrary smooth target; don't require recovery.
        y = X[:, 0] + 0.5 * X[:, 1]
        r = discover_curriculum(
            X, y,
            max_depth=2, n_tries=2,
            verbose=False,
        )
        assert r is not None
        assert r["n_vars"] == 2
        assert "expr" in r
        assert isinstance(r["snapped_tree"], GrowingEMLTree)
        assert r["snapped_tree"].n_vars == 2

    def test_depth0_atom_recovery_x2(self):
        """y = x2 must be recognized by the depth-0 atom pre-check."""
        rng = np.random.default_rng(1)
        X = rng.uniform(0.5, 2.0, size=(24, 3))
        y = X[:, 1]  # literally x2
        r = discover_curriculum(
            X, y,
            max_depth=2, n_tries=1,
            verbose=False,
        )
        assert r is not None
        assert r["exact"] is True
        assert r["depth"] == 0
        assert r["expr"] == "x2"
        assert r["n_vars"] == 3

    def test_eml_x1_x2_depth1_recovery(self):
        """y = eml(x1, x2) = exp(x1) - ln(x2) at depth 1 — trivial multivariate.

        This is the acceptance criterion from issue #18: recovery test
        ``y = exp(x1) - ln(x2)`` at depth 1.

        Use positive x2 to keep ln real; modest x1 to avoid exp blowup.
        """
        rng = np.random.default_rng(42)
        X = rng.uniform(0.3, 1.5, size=(64, 2))
        y = np.exp(X[:, 0]) - np.log(X[:, 1])
        r = discover_curriculum(
            X, y,
            max_depth=2, n_tries=6,
            verbose=False,
        )
        assert r is not None
        assert r["n_vars"] == 2
        # Depth 1 = single eml node over two leaves. Recovery should hit
        # machine-epsilon RMSE when it finds it.
        assert r["exact"] is True, (
            f"did not recover eml(x1, x2): rmse={r['snap_rmse']:.3e} "
            f"expr={r['expr']!r}"
        )
        # The discovered expression should contain both x1 and x2.
        assert "x1" in r["expr"] and "x2" in r["expr"]

    def test_1d_2d_equivalence_nvars1(self):
        """discover_curriculum(x_1d, y) and discover_curriculum(x_1d[:,None], y)
        should both find the same exp(x) at depth 1."""
        x = np.linspace(0.2, 2.0, 24)
        y = np.exp(x)
        r1 = discover_curriculum(x, y, max_depth=1, n_tries=2, verbose=False)
        r2 = discover_curriculum(x.reshape(-1, 1), y, max_depth=1, n_tries=2,
                                 verbose=False)
        assert r1["n_vars"] == 1
        assert r2["n_vars"] == 1
        assert r1["exact"] is True
        assert r2["exact"] is True

    @pytest.mark.slow
    def test_recover_ln_depth3(self):
        """Regression guard for issue #49: discover_curriculum must recover
        ln(x) exactly.

        ln(x) = eml(1, eml(eml(1, x), 1)) — the root's left child is the
        bare terminal ``1``. The default split_leaf warm-start seeds new
        subtrees as eml(x, 1) = exp(x), which puts the variable on the
        left and is systematically wrong for ln-shaped targets. Seed
        parity alternation (even→variable, odd→terminal) unblocks the
        canonical ln tree; the terminal bias seeds eml(1, x) on the
        left. Paired with best-in-seed snap tracking (splits can degrade
        the tree between growth steps), this nails ln(x) at
        machine-epsilon.
        """
        rng = np.random.default_rng(0)
        x = rng.uniform(0.5, 5.0, size=200).astype(np.float64)
        y = np.log(x)
        r = discover_curriculum(
            x.reshape(-1, 1), y,
            max_depth=4, n_tries=8,
            success_threshold=1e-6,
            verbose=False,
        )
        assert r is not None
        assert r["exact"] is True, (
            f"curriculum did not recover ln(x) exactly: "
            f"rmse={r['snap_rmse']:.3e} expr={r['expr']!r}"
        )
        assert r["snap_rmse"] < 1e-8
        assert r["expr"] == "ln(x)"
