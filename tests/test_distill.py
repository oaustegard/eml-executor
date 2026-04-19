"""Tests for eml_sr_distill (issue #42).

The distillation pipeline is a driver over eml-sr's existing
`discover`/`discover_curriculum` — these tests focus on the driver's
contract, not on eml-sr's regression accuracy (covered elsewhere).

Slow integration tests (full ``distill`` with ``discover_curriculum``) are
gated behind ``pytest -m slow``. The default run covers:

  - MLP forward/param-count sanity
  - Partial-dependence sampling (univariate slice through a known MLP)
  - Per-edge pre-activation sampling (recovers `W[j,i]` as a slope)
  - ``compose_additive`` string shape
  - ``_rename_variable`` / ``_denormalize_expr`` helpers
  - End-to-end ``distill`` on a trivial constant-output MLP (fast, deterministic)
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from eml_sr_distill import (
    SmallMLP,
    compose_additive,
    distill,
    regress_slot,
    sample_edge_preactivation,
    sample_partial_dependence,
    train_mlp,
)
from eml_sr_distill import _denormalize_expr, _rename_variable


# ─── Fixtures ──────────────────────────────────────────────────

@pytest.fixture
def linear_data():
    rng = np.random.default_rng(0)
    X = rng.uniform(-1.0, 1.0, size=(120, 2))
    y = 2.0 * X[:, 0] + 3.0 * X[:, 1] + 0.5
    return X, y


@pytest.fixture
def linear_model(linear_data):
    X, y = linear_data
    return train_mlp(X, y, hidden_sizes=(4,), activation="identity",
                     epochs=1200, lr=5e-2, seed=0)


# ─── MLP ───────────────────────────────────────────────────────

class TestSmallMLP:
    def test_output_shape_scalar(self):
        m = SmallMLP(3, hidden_sizes=(4, 4), activation="square").double()
        x = torch.randn(10, 3, dtype=torch.float64)
        y = m(x)
        assert y.shape == (10,)

    def test_param_count_matches_torch(self):
        m = SmallMLP(2, hidden_sizes=(5,), activation="tanh").double()
        assert m.param_count() == sum(p.numel() for p in m.parameters())
        # 2*5 + 5 (first) + 5*1 + 1 (out) = 21
        assert m.param_count() == 21

    def test_unknown_activation_rejected(self):
        with pytest.raises(ValueError, match="unknown activation"):
            SmallMLP(2, activation="softmax")


# ─── train_mlp ─────────────────────────────────────────────────

class TestTrainMLP:
    def test_fits_linear_target(self, linear_data):
        X, y = linear_data
        model = train_mlp(X, y, hidden_sizes=(3,), activation="identity",
                          epochs=1000, lr=5e-2, seed=0)
        with torch.no_grad():
            pred = model(torch.tensor(X, dtype=torch.float64)).cpu().numpy()
        # Linear + identity-activation MLP trivially recovers the target.
        assert np.sqrt(np.mean((pred - y) ** 2)) < 1e-6

    def test_1d_input_auto_reshapes(self):
        rng = np.random.default_rng(0)
        x = rng.uniform(-1, 1, size=(100,))
        y = 0.7 * x - 0.2
        model = train_mlp(x, y, hidden_sizes=(3,), activation="identity",
                          epochs=800, lr=5e-2, seed=0)
        assert model.layers[0].in_features == 1


# ─── Sampling ──────────────────────────────────────────────────

class TestPartialDependence:
    def test_recovers_affine_slope(self, linear_model, linear_data):
        X, _ = linear_data
        slot = sample_partial_dependence(linear_model, X, feature_idx=0, n_grid=40)
        assert slot.kind == "feature"
        assert slot.feature_idx == 0
        # slope dy/dx of feature 0's PD should equal 2 (target coefficient).
        slope = (slot.y_values[-1] - slot.y_values[0]) / (
            slot.x_grid[-1] - slot.x_grid[0]
        )
        assert slope == pytest.approx(2.0, abs=1e-4)

    def test_grid_lies_in_data_range(self, linear_model, linear_data):
        X, _ = linear_data
        slot = sample_partial_dependence(linear_model, X, feature_idx=1, n_grid=30)
        assert slot.x_grid.min() >= X[:, 1].min() - 1e-9
        assert slot.x_grid.max() <= X[:, 1].max() + 1e-9

    def test_feature_idx_bounds(self, linear_model, linear_data):
        X, _ = linear_data
        with pytest.raises(IndexError):
            sample_partial_dependence(linear_model, X, feature_idx=2)


class TestEdgePreactivation:
    def test_edge_slope_matches_weight(self, linear_model, linear_data):
        X, _ = linear_data
        W = linear_model.layers[0].weight.detach().numpy()
        n_hidden = W.shape[0]
        for j in range(n_hidden):
            for i in range(X.shape[1]):
                slot = sample_edge_preactivation(linear_model, X,
                                                 feature_idx=i, neuron_idx=j,
                                                 n_grid=20)
                slope = (slot.y_values[-1] - slot.y_values[0]) / (
                    slot.x_grid[-1] - slot.x_grid[0]
                )
                assert slope == pytest.approx(W[j, i], abs=1e-9), (
                    f"edge ({i}→h{j}) slope mismatch: "
                    f"measured {slope}, W[j,i]={W[j,i]}"
                )


# ─── Constant-slot fast path ───────────────────────────────────

class TestConstantSlotFastPath:
    def test_constant_slot_is_exact(self):
        """If the partial dependence curve is flat (MLP ignores the feature),
        ``regress_slot`` returns a one-node constant — no eml-sr call needed.
        """
        from eml_sr_distill import SlotSample

        slot = SlotSample(
            kind="feature",
            feature_idx=0,
            neuron_idx=None,
            x_grid=np.linspace(-1, 1, 20),
            y_values=np.full(20, 3.14),
            x_mean=0.0,
            x_range=(-1.0, 1.0),
        )
        res = regress_slot(slot, max_depth=2, n_tries=1)
        assert res.exact
        assert res.tree_size == 1
        assert res.snap_rmse == 0.0


# ─── Composition helpers ───────────────────────────────────────

class TestComposeHelpers:
    def test_rename_variable_whole_tokens_only(self):
        assert _rename_variable("exp(x) + x", "x", "x0") == "exp(x0) + x0"
        # "exp" must not be touched even though it doesn't share characters.
        assert _rename_variable("exp(x)", "exp", "X") == "X(x)"
        # Substring matches must be avoided.
        assert _rename_variable("eml(1, x_prev)", "x", "x0") == "eml(1, x_prev)"

    def test_denormalize_expr_identity_with_no_norm(self):
        assert _denormalize_expr("exp(x)", None, "x0") == "exp(x0)"

    def test_compose_additive_contains_all_slots(self, linear_model, linear_data):
        X, _ = linear_data
        slot0 = sample_partial_dependence(linear_model, X, 0, n_grid=30)
        slot1 = sample_partial_dependence(linear_model, X, 1, n_grid=30)
        r0 = regress_slot(slot0, method="curriculum", max_depth=3, n_tries=3)
        r1 = regress_slot(slot1, method="curriculum", max_depth=3, n_tries=3)
        composed = compose_additive([r0, r1], y_bias=0.5)
        assert "x0" in composed
        assert "x1" in composed
        assert "0.5" in composed


# ─── End-to-end integration (fast) ─────────────────────────────

class TestDistillIntegration:
    def test_distill_on_constant_target(self):
        """Constant target → every PD curve is flat, every slot is the
        fast-path constant, surrogate RMSE is zero."""
        rng = np.random.default_rng(0)
        X = rng.uniform(-1, 1, size=(60, 2))
        y = np.full(len(X), 7.0)
        model = train_mlp(X, y, hidden_sizes=(3,), activation="identity",
                          epochs=400, lr=5e-2, seed=0)
        rep = distill(model, X, y, method="curriculum", max_depth=2,
                      n_tries=1, n_grid=15, verbose=False)
        assert rep.recovery_rate == 1.0
        assert rep.surrogate_rmse_on_mlp < 1e-8
        assert "7" in rep.composed_expr  # y_bias shows up

    @pytest.mark.slow
    def test_distill_recovers_additive_linear(self, linear_data, linear_model):
        X, y = linear_data
        rep = distill(linear_model, X, y, method="curriculum", max_depth=4,
                      n_tries=4, n_grid=40, verbose=False)
        # Linear MLP with identity activation = exact additive decomposition.
        assert rep.surrogate_r2_on_mlp > 0.9999
        assert rep.recovery_rate == 1.0
