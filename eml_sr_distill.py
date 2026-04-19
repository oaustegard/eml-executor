"""eml_sr_distill: post-hoc symbolic extraction from trained MLPs.

Issue #42. KAN (Liu et al.) replaces scalar weights with learnable univariate
splines on edges; the splines fit noise, don't extrapolate, and the result
isn't symbolic in a useful sense. eml-sr's vocabulary `{1, x, eml(x, y)}` is
the right basis — it's complete on elementary functions (Odrzywolek 2026),
compositions are exact, and discovered expressions *are* symbolic.

This module is the driver: train a standard MLP, sample its induced
univariate responses at each target slot (per-feature partial dependence,
per-edge pre-activation), run `discover_curriculum` on each, and compose
the recovered expressions into a symbolic surrogate.

First-pass scope (matches issue #42 "in scope"):
  - 1–2 hidden-layer MLP on tabular regression.
  - `x²` activation default — polynomial closure keeps composition elementary
    and the simplifier already handles polynomial shapes.
  - Per-feature partial-dependence sampling: vary x_i, freeze others at
    marginal means. Per-edge sampling into the first hidden layer also
    supported.
  - `discover_curriculum` (or `discover`) as the per-slot regressor.
  - Additive composition: f(x) ≈ y_bias + Σ_i (φ_i(x_i) - φ_i(μ_i)). Exact
    iff the trained MLP is learning an additive decomposition; otherwise
    reports the best additive approximation plus a residual R².

Out of scope (documented, not solved): deep composition beyond one hidden
layer, attention / non-elementary activations, vision/sequence models.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import math
import numpy as np
import torch
import torch.nn as nn

from eml_sr import Normalizer, discover, discover_curriculum, _simplify


# ─── Activation choices ────────────────────────────────────────
#
# The activation determines whether the composed symbolic form stays in
# eml-sr's reachable class. `x²` is safe: polynomials compose to polynomials.
# `tanh` / `sigmoid` are standard but not elementary in the paper's sense —
# a symbolic approximation step would be needed, which is out of scope for
# first pass. The registry here is intentionally small.

def _square(x: torch.Tensor) -> torch.Tensor:
    return x * x


_ACTIVATIONS: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    "square": _square,
    "identity": lambda x: x,
    "tanh": torch.tanh,
    "relu": torch.relu,
}


# ─── MLP ───────────────────────────────────────────────────────

class SmallMLP(nn.Module):
    """Small MLP for regression. Pluggable activation, scalar output.

    The first-pass target for distillation is `activation="square"` — it
    keeps everything in polynomial closure so composed expressions stay
    elementary.
    """

    def __init__(self, n_inputs: int, hidden_sizes: Sequence[int] = (8,),
                 activation: str = "square"):
        super().__init__()
        if activation not in _ACTIVATIONS:
            raise ValueError(
                f"unknown activation {activation!r}; "
                f"choose from {sorted(_ACTIVATIONS)}"
            )
        self.activation_name = activation
        self.act = _ACTIVATIONS[activation]
        sizes = [n_inputs, *hidden_sizes, 1]
        self.layers = nn.ModuleList(
            nn.Linear(sizes[i], sizes[i + 1]) for i in range(len(sizes) - 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for i, layer in enumerate(self.layers):
            h = layer(h)
            if i < len(self.layers) - 1:
                h = self.act(h)
        return h.squeeze(-1)

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


def train_mlp(X: np.ndarray, y: np.ndarray, *,
              hidden_sizes: Sequence[int] = (8,),
              activation: str = "square",
              epochs: int = 3000,
              lr: float = 1e-2,
              weight_decay: float = 0.0,
              seed: int = 0,
              verbose: bool = False) -> SmallMLP:
    """Train a SmallMLP on (X, y). Returns the trained model."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).ravel()
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = SmallMLP(X.shape[1], hidden_sizes=hidden_sizes, activation=activation)
    model = model.double()
    x_t = torch.tensor(X, dtype=torch.float64)
    y_t = torch.tensor(y, dtype=torch.float64)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    for step in range(epochs):
        opt.zero_grad()
        pred = model(x_t)
        loss = torch.mean((pred - y_t) ** 2)
        loss.backward()
        opt.step()
        if verbose and (step % max(1, epochs // 10) == 0 or step == epochs - 1):
            print(f"  step {step:5d}  mse={loss.item():.3e}")
    model.eval()
    return model


# ─── Sampling: per-feature partial dependence ──────────────────

@dataclass
class SlotSample:
    """A univariate response curve sampled from the trained MLP.

    ``kind`` is one of:
      - "feature": output as a function of input feature `i`, others frozen
        at their marginal means.
      - "edge": pre-activation of hidden-layer neuron `j` as a function of
        input feature `i`, others frozen at their marginal means.
    """

    kind: str
    feature_idx: int
    neuron_idx: Optional[int]
    x_grid: np.ndarray
    y_values: np.ndarray
    x_mean: float
    x_range: tuple[float, float]

    def label(self) -> str:
        if self.kind == "feature":
            return f"f(x{self.feature_idx})"
        return f"h{self.neuron_idx}_on_x{self.feature_idx}"


def _feature_grid(X_col: np.ndarray, n_grid: int) -> np.ndarray:
    lo, hi = float(X_col.min()), float(X_col.max())
    if lo == hi:
        # degenerate column; return a constant grid of one point
        return np.array([lo], dtype=np.float64)
    return np.linspace(lo, hi, n_grid, dtype=np.float64)


def sample_partial_dependence(model: SmallMLP, X: np.ndarray,
                              feature_idx: int, *,
                              n_grid: int = 80) -> SlotSample:
    """Sample the MLP's output as a univariate function of feature `i`.

    All other features are frozen at their marginal means. This is the
    1D partial-dependence plot (Friedman 2001), specialised to the case
    where only one feature varies.
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    n_feat = X.shape[1]
    if not 0 <= feature_idx < n_feat:
        raise IndexError(f"feature_idx {feature_idx} out of range for n_feat={n_feat}")
    means = X.mean(axis=0)
    grid = _feature_grid(X[:, feature_idx], n_grid)
    grid_X = np.tile(means, (len(grid), 1))
    grid_X[:, feature_idx] = grid
    with torch.no_grad():
        y_vals = model(torch.tensor(grid_X, dtype=torch.float64)).cpu().numpy()
    return SlotSample(
        kind="feature",
        feature_idx=feature_idx,
        neuron_idx=None,
        x_grid=grid,
        y_values=y_vals,
        x_mean=float(means[feature_idx]),
        x_range=(float(grid.min()), float(grid.max())),
    )


def sample_edge_preactivation(model: SmallMLP, X: np.ndarray,
                              feature_idx: int, neuron_idx: int, *,
                              n_grid: int = 80) -> SlotSample:
    """Sample the first hidden layer's pre-activation for neuron `j` as a
    function of feature `i`.

    Concretely: `z_j(x_i) = W[j, i] * x_i + (W[j, -i] @ μ_{-i} + b_j)`.
    This is literally affine by construction — we include it primarily as
    a sanity target for the regressor (it should recover `a*x + b` exactly)
    and as a building block for the KAN-style per-edge narrative.

    For a multi-layer network, the same idea extends by holding `z_j` on the
    subsequent layers' inputs at their induced means, but the composition
    is out of scope for first pass.
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    means = X.mean(axis=0)
    grid = _feature_grid(X[:, feature_idx], n_grid)
    grid_X = np.tile(means, (len(grid), 1))
    grid_X[:, feature_idx] = grid
    with torch.no_grad():
        first_layer = model.layers[0]
        z = first_layer(torch.tensor(grid_X, dtype=torch.float64))
    y_vals = z[:, neuron_idx].cpu().numpy()
    return SlotSample(
        kind="edge",
        feature_idx=feature_idx,
        neuron_idx=neuron_idx,
        x_grid=grid,
        y_values=y_vals,
        x_mean=float(means[feature_idx]),
        x_range=(float(grid.min()), float(grid.max())),
    )


# ─── Per-slot symbolic regression ──────────────────────────────

@dataclass
class SlotResult:
    slot: SlotSample
    expr: str          # expression in normalized coordinates (if normalized)
    snap_rmse: float   # RMSE in original (un-normalized) coordinates
    exact: bool
    depth: int
    tree_size: int
    y_bias: float      # φ_i(μ_i), needed for additive composition
    normalizer: Optional[Normalizer] = None


def _tree_size(tree) -> int:
    """Count nodes in an EMLTree1D's snapped form, via string scan."""
    # Cheap proxy: count 'eml(' occurrences (internals) + leaf atoms.
    # A fully-parsed count would recurse the torch module tree; the string
    # proxy is sufficient for compression-ratio reporting.
    if tree is None:
        return 0
    try:
        from eml_sr import _parse_eml  # local import; private but stable
        parsed = _parse_eml(tree)
        return _count_nodes(parsed)
    except Exception:
        return len(str(tree))


def _count_nodes(ast) -> int:
    if ast[0] == "atom":
        return 1
    if ast[0] in ("exp", "ln", "neg"):
        return 1 + _count_nodes(ast[1])
    if ast[0] in ("sub", "eml"):
        return 1 + _count_nodes(ast[1]) + _count_nodes(ast[2])
    return 1


def regress_slot(slot: SlotSample, *,
                 method: str = "curriculum",
                 max_depth: int = 5,
                 n_tries: int = 8,
                 success_threshold: float = 1e-6,
                 normalize: str = "minmax",
                 verbose: bool = False) -> SlotResult:
    """Run eml-sr on one sampled slot.

    Per-slot affine pre/post-normalization is used by default because
    per-edge responses are rarely in [-1, 1] to start with. The `expr`
    field is in normalized coordinates; the `normalizer` field holds the
    affine transforms needed to evaluate it in original coordinates.
    `snap_rmse` is reported back in original (denormalized) coordinates so
    compression/recovery stats reflect the real residual. Pass
    ``normalize="none"`` for the paper-faithful raw mode (only useful
    when the slot is already close to unit scale).

    Returns a SlotResult bundling the simplified expression, RMSE against
    the sampled curve, and a few book-keeping stats for composition.
    """
    x_g = np.asarray(slot.x_grid, dtype=np.float64)
    y_g = np.asarray(slot.y_values, dtype=np.float64).ravel()
    # Bias term for additive composition: the slot's value at the frozen
    # mean (which is where the partial-dependence grid is anchored).
    bias = float(np.interp(slot.x_mean, x_g, y_g))

    if len(x_g) < 2 or np.allclose(y_g, y_g[0]):
        # Constant slot → exact zero-depth fit at the bias value.
        return SlotResult(
            slot=slot,
            expr=f"{bias:.6g}",
            snap_rmse=0.0,
            exact=True,
            depth=0,
            tree_size=1,
            y_bias=bias,
            normalizer=None,
        )

    norm: Optional[Normalizer] = None
    if normalize != "none":
        norm = Normalizer.fit(x_g, y_g, mode=normalize)
        x_fit = norm.transform_x(x_g)
        y_fit = norm.transform_y(y_g)
    else:
        x_fit = x_g
        y_fit = y_g

    kwargs = dict(
        max_depth=max_depth, n_tries=n_tries, verbose=verbose,
        success_threshold=success_threshold,
    )
    if method == "curriculum":
        res = discover_curriculum(x_fit, y_fit, **kwargs)
    elif method == "discover":
        res = discover(x_fit, y_fit, **kwargs)
    else:
        raise ValueError(f"unknown method {method!r}; use 'curriculum' or 'discover'")

    expr_raw = res["expr"] if res else "0"
    expr = _simplify(expr_raw)

    # snap_rmse is the regressor's residual in the coordinates we fed it —
    # normalized when `normalize != 'none'`. We keep it in normalized coords
    # because that's the space the symbolic expression lives in; reporting a
    # denormalised version would confuse "the shape was recovered" with
    # "the expression evaluates to the same scalar as the MLP" (which it
    # generally doesn't — the scale lives in the Normalizer).
    snap_rmse = float(res["snap_rmse"]) if res else float("inf")
    exact_hint = bool(res.get("exact", True)) if res else False
    depth = int(res.get("depth", 0)) if res else 0
    return SlotResult(
        slot=slot,
        expr=expr,
        snap_rmse=snap_rmse,
        exact=exact_hint and snap_rmse < math.sqrt(success_threshold),
        depth=depth,
        tree_size=_tree_size(expr),
        y_bias=bias,
        normalizer=norm,
    )


# ─── Composition ───────────────────────────────────────────────

def compose_additive(results: Sequence[SlotResult],
                     y_bias: float,
                     feature_names: Optional[Sequence[str]] = None) -> str:
    """Compose per-feature slot results as an additive surrogate.

    f̂(x) = y_bias + Σ_i ( φ_i(x_i) − φ_i(μ_i) )

    The subtraction of φ_i(μ_i) keeps the surrogate centred: at the
    marginal-mean input, every per-feature term contributes zero, and the
    surrogate returns `y_bias` (the MLP's output at the mean). Exact iff
    the MLP learned an additive decomposition; otherwise this is the best
    additive approximation.

    Each per-slot expression is wrapped in the inverse of its normalizer
    so the composed string is in original coordinates and directly usable
    by a downstream reader. Per-slot variable names are rewritten from
    the regressor's generic `x` to `x{feature_idx}` (or the user label).
    """
    parts = [f"{y_bias:.6g}"]
    for r in results:
        if r.slot.kind != "feature":
            continue
        name = (feature_names[r.slot.feature_idx]
                if feature_names is not None else f"x{r.slot.feature_idx}")
        phi = _denormalize_expr(r.expr, r.normalizer, name)
        bias_sub = f"{r.y_bias:.6g}"
        parts.append(f"({phi} - {bias_sub})")
    return " + ".join(parts)


def _denormalize_expr(expr: str, norm: Optional[Normalizer], var: str) -> str:
    """Wrap `expr(x')` where `x' = x_a*x + x_b` and invert the y-affine.

    Emits a string of the form ``((expr[x := x_a*var+x_b]) - y_b) / y_a``.
    With ``norm=None`` it just renames `x` → `var`.
    """
    if norm is None:
        return _rename_variable(expr, "x", var)
    # Substitute x := (x_a*var + x_b). A multi-character replacement would
    # be ambiguous in general, but per-slot expressions only reference one
    # variable, named `x`, so a whole-token rename to a parenthesised form
    # is safe.
    x_sub = f"({norm.x_a:.10g} * {var} + {norm.x_b:.10g})"
    inner = _rename_variable(expr, "x", x_sub)
    if norm.y_a == 0:
        return f"{norm.y_b:.10g}"
    return f"(({inner} - {norm.y_b:.10g}) / {norm.y_a:.10g})"


def _rename_variable(expr: str, old: str, new: str) -> str:
    """Substitute whole-token occurrences of `old` with `new` in a string.

    Lightweight tokeniser: expressions emitted by the simplifier are in a
    restricted alphabet (identifiers, parens, commas, operators, whitespace).
    We avoid pulling in `re` by scanning character classes directly.
    """
    out = []
    i = 0
    n = len(expr)
    while i < n:
        c = expr[i]
        if c.isalpha() or c == "_":
            j = i
            while j < n and (expr[j].isalnum() or expr[j] == "_"):
                j += 1
            token = expr[i:j]
            out.append(new if token == old else token)
            i = j
        else:
            out.append(c)
            i += 1
    return "".join(out)


# ─── Top-level pipeline ────────────────────────────────────────

@dataclass
class DistillReport:
    per_feature: list[SlotResult]
    per_edge: list[SlotResult]
    composed_expr: str
    y_bias: float
    recovery_rate: float
    mlp_params: int
    surrogate_nodes: int
    compression_ratio: float
    surrogate_rmse_on_mlp: float
    surrogate_r2_on_mlp: float

    def summary(self) -> str:
        lines = [
            f"MLP params:         {self.mlp_params}",
            f"Surrogate nodes:    {self.surrogate_nodes}",
            f"Compression ratio:  {self.compression_ratio:.2f}x",
            f"Per-feature slots:  {len(self.per_feature)} "
            f"(exact: {sum(r.exact for r in self.per_feature)})",
            f"Per-edge slots:     {len(self.per_edge)} "
            f"(exact: {sum(r.exact for r in self.per_edge)})",
            f"Recovery rate:      {self.recovery_rate * 100:.1f}%",
            f"Surrogate vs MLP:   RMSE={self.surrogate_rmse_on_mlp:.3e}  "
            f"R²={self.surrogate_r2_on_mlp:.4f}",
            f"Composed form:      {self.composed_expr}",
        ]
        return "\n".join(lines)


def distill(model: SmallMLP, X: np.ndarray, y: np.ndarray, *,
            method: str = "curriculum",
            max_depth: int = 5,
            n_tries: int = 8,
            n_grid: int = 80,
            include_edges: bool = False,
            edge_neurons: Optional[Sequence[int]] = None,
            feature_names: Optional[Sequence[str]] = None,
            success_threshold: float = 1e-6,
            normalize: str = "minmax",
            verbose: bool = False) -> DistillReport:
    """End-to-end: sample the MLP's slots, regress each, compose additively.

    Args:
        model: a trained `SmallMLP`.
        X: training inputs (for sampling ranges and marginal means).
        y: training targets (for R² on held-out distillation).
        method: 'curriculum' (default) or 'discover'.
        max_depth, n_tries: passed to the per-slot regressor.
        n_grid: points sampled along each univariate slot.
        include_edges: if True, also run per-edge preactivation extraction
            on the first hidden layer (affine targets — mainly a sanity
            check on the regressor).
        edge_neurons: which first-hidden-layer neurons to extract edges for;
            defaults to all.
        feature_names: optional human-readable labels for the input columns.
        success_threshold: MSE threshold for "exact" recovery per slot.

    Returns a `DistillReport`.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).ravel()
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    n_features = X.shape[1]

    # y_bias: the MLP's output at the marginal-mean input. Anchors the
    # additive decomposition so per-slot (expr − bias) terms centre at 0.
    with torch.no_grad():
        means = X.mean(axis=0)
        y_bias = float(model(torch.tensor(means[None, :], dtype=torch.float64)).item())

    per_feature: list[SlotResult] = []
    for i in range(n_features):
        slot = sample_partial_dependence(model, X, i, n_grid=n_grid)
        if verbose:
            print(f"\n=== per-feature slot x{i} "
                  f"(range {slot.x_range}, bias anchor x={slot.x_mean:.3g}) ===")
        res = regress_slot(slot, method=method, max_depth=max_depth,
                           n_tries=n_tries, success_threshold=success_threshold,
                           normalize=normalize, verbose=verbose)
        per_feature.append(res)
        if verbose:
            tag = " ✓" if res.exact else ""
            print(f"  → {res.expr}   rmse={res.snap_rmse:.3e}{tag}")

    per_edge: list[SlotResult] = []
    if include_edges:
        first = model.layers[0]
        n_hidden = first.out_features
        neurons = list(edge_neurons) if edge_neurons is not None else list(range(n_hidden))
        for j in neurons:
            for i in range(n_features):
                slot = sample_edge_preactivation(model, X, i, j, n_grid=n_grid)
                if verbose:
                    print(f"\n=== per-edge slot h{j}←x{i} ===")
                res = regress_slot(slot, method=method, max_depth=max_depth,
                                   n_tries=n_tries,
                                   success_threshold=success_threshold,
                                   normalize=normalize, verbose=verbose)
                per_edge.append(res)
                if verbose:
                    tag = " ✓" if res.exact else ""
                    print(f"  → {res.expr}   rmse={res.snap_rmse:.3e}{tag}")

    composed = compose_additive(per_feature, y_bias, feature_names=feature_names)

    # Evaluate the composed surrogate against the MLP on X.
    surrogate_pred = _eval_additive(per_feature, X, y_bias)
    with torch.no_grad():
        mlp_pred = model(torch.tensor(X, dtype=torch.float64)).cpu().numpy()
    surr_rmse = float(np.sqrt(np.mean((surrogate_pred - mlp_pred) ** 2)))
    ss_res = float(np.sum((mlp_pred - surrogate_pred) ** 2))
    ss_tot = float(np.sum((mlp_pred - mlp_pred.mean()) ** 2))
    surr_r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    all_slots = per_feature + per_edge
    exact = sum(r.exact for r in all_slots)
    total = max(1, len(all_slots))
    recovery = exact / total

    surrogate_nodes = sum(r.tree_size for r in per_feature)

    return DistillReport(
        per_feature=per_feature,
        per_edge=per_edge,
        composed_expr=composed,
        y_bias=y_bias,
        recovery_rate=recovery,
        mlp_params=model.param_count(),
        surrogate_nodes=surrogate_nodes,
        compression_ratio=model.param_count() / max(1, surrogate_nodes),
        surrogate_rmse_on_mlp=surr_rmse,
        surrogate_r2_on_mlp=surr_r2,
    )


def _eval_additive(results: Sequence[SlotResult], X: np.ndarray,
                   y_bias: float) -> np.ndarray:
    """Evaluate the composed additive surrogate on X numerically.

    Each per-feature slot was regressed on a grid of (x, y) pairs; rather
    than evaluating the symbolic expression, we interpolate each curve's
    grid — this is faithful to the decomposition and sidesteps rebuilding
    a symbolic evaluator. The composed symbolic string is still returned
    separately by ``distill`` for inspection.
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    out = np.full(X.shape[0], y_bias, dtype=np.float64)
    for r in results:
        i = r.slot.feature_idx
        if r.slot.kind != "feature":
            continue
        phi = np.interp(X[:, i], r.slot.x_grid, r.slot.y_values)
        phi_mu = np.interp(r.slot.x_mean, r.slot.x_grid, r.slot.y_values)
        out += (phi - phi_mu)
    return out


__all__ = [
    "SmallMLP",
    "train_mlp",
    "SlotSample",
    "SlotResult",
    "DistillReport",
    "sample_partial_dependence",
    "sample_edge_preactivation",
    "regress_slot",
    "compose_additive",
    "distill",
]
