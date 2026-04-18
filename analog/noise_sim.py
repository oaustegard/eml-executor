"""analog/noise_sim.py — EML analog noise-compounding simulator.

Models per-node Gaussian (additive/multiplicative), 1/f flicker, and
matched-pair noise injection through an EML tree, characterising how
output error grows with tree depth.

Design decisions (from issue #34):
- Noise injected *after* each internal node's eml_op output, before
  the value is consumed by its parent.
- Noisy evaluation runs in float64 (real arithmetic).  Analog wires
  carry real voltages/currents; they cannot represent complex values.
- Ideal evaluation runs in complex128 (same path as the rest of the repo).
- No clamping in the noisy path: analog hardware saturates/rails rather
  than clamping, and we want to measure that gap.
- Trees that require complex intermediates are flagged but still
  simulated with the real part of the ideal output as the reference.
"""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from eml_compiler import EMLTree, Leaf, Node, eval_eml, tree_depth


# ─── Noise models ──────────────────────────────────────────────────

class NoiseModel(ABC):
    """Abstract base for per-node noise injection."""

    def setup_trial(self, n_nodes: int, rng: np.random.Generator) -> None:
        """Called once per trial; subclasses with correlated noise pre-generate here."""

    @abstractmethod
    def sample_node(self, node_idx: int, node_value: float,
                    rng: np.random.Generator) -> float:
        """Return noise value to add to `node_value` at `node_idx`."""

    @property
    @abstractmethod
    def name(self) -> str:
        ...


@dataclass
class AdditiveGaussian(NoiseModel):
    """i.i.d. Gaussian noise N(0, sigma²) added to each node output.

    Baseline model: component absolute offset.
    """
    sigma: float

    @property
    def name(self) -> str:
        return f"additive_gaussian"

    def sample_node(self, node_idx, node_value, rng):
        return rng.normal(0.0, self.sigma)


@dataclass
class MultiplicativeGaussian(NoiseModel):
    """Multiplicative Gaussian: output *= (1 + N(0, sigma²)).

    Models component tolerance (e.g. 1% resistor mismatch): error
    scales with signal magnitude.
    """
    sigma: float

    @property
    def name(self) -> str:
        return f"multiplicative_gaussian"

    def sample_node(self, node_idx, node_value, rng):
        return node_value * rng.normal(0.0, self.sigma)


@dataclass
class OneOverF(NoiseModel):
    """1/f (flicker/drift) noise model.

    Pre-generates a correlated noise sequence per trial using spectral
    synthesis.  The PSD is flat for f < cutoff and rolls off as
    alpha/f^alpha for f >= cutoff.  The RMS is normalised to sigma so
    sigmas remain comparable across noise models.

    Parameters
    ----------
    sigma : float
        RMS noise amplitude (same units as the additive model).
    alpha : float
        Spectral exponent (1.0 = classic 1/f, 2.0 = Brownian).
    cutoff : float
        Low-frequency corner as a fraction of the node-count Nyquist
        (0 < cutoff < 0.5).  Lower values give longer-range correlation.
    """
    sigma: float
    alpha: float = 1.0
    cutoff: float = 0.05

    def __post_init__(self):
        self._noise_seq: np.ndarray = np.zeros(0)

    @property
    def name(self) -> str:
        return f"1_over_f"

    def setup_trial(self, n_nodes, rng):
        n = max(n_nodes, 32)  # need enough points for spectral shaping
        freqs = np.fft.rfftfreq(n)
        freqs[0] = self.cutoff  # guard against DC singularity
        # Power: flat below cutoff, 1/f^alpha above
        power = np.where(freqs < self.cutoff, 1.0,
                         (self.cutoff / freqs) ** self.alpha)
        amplitude = np.sqrt(power)
        # Generate complex Gaussian in freq domain, transform to time
        real_part = rng.standard_normal(len(freqs))
        imag_part = rng.standard_normal(len(freqs))
        noise_f = amplitude * (real_part + 1j * imag_part)
        noise_t = np.fft.irfft(noise_f, n=n)
        # Normalise using the full sequence std (before truncation)
        std = noise_t.std()
        if std > 0:
            self._noise_seq = (noise_t * (self.sigma / std))[:n_nodes]
        else:
            self._noise_seq = np.zeros(n_nodes)

    def sample_node(self, node_idx, node_value, rng):
        if node_idx < len(self._noise_seq):
            return float(self._noise_seq[node_idx])
        return rng.normal(0.0, self.sigma)


@dataclass
class MatchedPairs(NoiseModel):
    """Matched-transistor pair noise model for translinear circuits.

    Each trial has a global common-mode drift drawn once (sigma_common),
    shared across all nodes on the same chip/circuit, plus an independent
    per-node differential mismatch (sigma_diff).  In a well-designed
    differential topology the common component partially cancels at the
    output; here we model it as a tree-wide offset + per-node scatter,
    which shows up as correlated residuals across outputs.

    Parameters
    ----------
    sigma_common : float
        Std dev of the global drift component (same for all nodes in a trial).
    sigma_diff : float
        Std dev of the per-node independent mismatch component.
    """
    sigma_common: float
    sigma_diff: float

    def __post_init__(self):
        self._global_drift: float = 0.0

    @property
    def name(self) -> str:
        return f"matched_pairs"

    def setup_trial(self, n_nodes, rng):
        self._global_drift = float(rng.normal(0.0, self.sigma_common))

    def sample_node(self, node_idx, node_value, rng):
        return self._global_drift + rng.normal(0.0, self.sigma_diff)


# ─── Tree utilities ────────────────────────────────────────────────

def _count_internal_nodes(tree: EMLTree) -> int:
    if isinstance(tree, Leaf):
        return 0
    return 1 + _count_internal_nodes(tree.left) + _count_internal_nodes(tree.right)


def _collect_variable_names(tree: EMLTree) -> set[str]:
    if isinstance(tree, Leaf):
        if tree.value is None:
            return {tree.label}
        return set()
    return _collect_variable_names(tree.left) | _collect_variable_names(tree.right)


def _check_complex_intermediates(
    tree: EMLTree, x_samples: dict[str, np.ndarray]
) -> tuple[bool, Optional[str]]:
    """Check whether any ideal intermediate is complex (non-zero imaginary part).

    Runs the ideal evaluation and inspects every node.
    """
    has_complex = False
    n_complex_nodes = [0]  # mutable counter

    def recur(node: EMLTree, bindings: dict) -> complex:
        nonlocal has_complex
        if isinstance(node, Leaf):
            if node.value is not None:
                return np.complex128(node.value)
            return np.complex128(bindings[node.label])
        left = recur(node.left, bindings)
        right = recur(node.right, bindings)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            result = np.exp(left) - np.log(right)
        if abs(result.imag) > 1e-10 * (abs(result.real) + 1e-300):
            has_complex = True
            n_complex_nodes[0] += 1
        return result

    n_samples = len(next(iter(x_samples.values())))
    for i in range(min(n_samples, 10)):
        bindings = {k: v[i] for k, v in x_samples.items()}
        recur(tree, bindings)

    if has_complex:
        msg = (
            f"Tree uses {n_complex_nodes[0]} complex intermediate(s) in the first "
            f"{min(n_samples, 10)} samples.  These are NOT physically realizable "
            "with real-valued analog wires (single-wire representation).  "
            "Simulation proceeds using real(ideal_output) as reference; "
            "reported RMSE may understate analog infeasibility."
        )
        return True, msg
    return False, None


# ─── Noisy evaluation ──────────────────────────────────────────────

def _eval_noisy(
    tree: EMLTree,
    bindings: dict[str, float],
    noise_model: NoiseModel,
    rng: np.random.Generator,
    node_counter: list[int],
) -> float:
    """Evaluate tree in float64 with per-node noise injection.  No clamping."""
    if isinstance(tree, Leaf):
        if tree.value is not None:
            return float(tree.value.real)
        return float(bindings[tree.label])

    left_val = _eval_noisy(tree.left, bindings, noise_model, rng, node_counter)
    right_val = _eval_noisy(tree.right, bindings, noise_model, rng, node_counter)

    # EML in float64, no overflow guard — analog hardware saturates/rails
    with np.errstate(all='ignore'):
        result = float(np.exp(np.float64(left_val)) - np.log(np.float64(right_val)))

    node_idx = node_counter[0]
    node_counter[0] += 1
    noise = noise_model.sample_node(node_idx, result, rng)
    return result + noise


# ─── Public API ────────────────────────────────────────────────────

def simulate(
    tree: EMLTree,
    noise_model: NoiseModel,
    x_samples: dict[str, np.ndarray],
    n_trials: int = 1000,
    seed: Optional[int] = None,
) -> dict:
    """Evaluate `tree` at `x_samples` under `noise_model`, return RMSE vs ideal
    plus per-node error accounting.

    Parameters
    ----------
    tree : EMLTree
        Compiled EML tree (Leaf or Node from eml_compiler).
    noise_model : NoiseModel
        One of AdditiveGaussian, MultiplicativeGaussian, OneOverF, MatchedPairs.
    x_samples : dict[str, np.ndarray]
        Variable bindings, each a 1-D array of length n_samples.
        E.g. {'x': np.linspace(0.5, 2.0, 50)} for univariate.
    n_trials : int, default 1000
        Number of Monte Carlo noise draws.
    seed : int or None
        RNG seed for reproducibility.

    Returns
    -------
    dict with keys:
        rmse                  float  — mean RMSE across trials
        rmse_std              float  — std of per-trial RMSE
        rmse_per_trial        array  — shape (n_trials,)
        bits_of_precision     float  — log2(output_range / rmse), −∞ if rmse≥range
        has_complex_intermediates  bool
        complex_warning       str or None
        ideal_outputs         complex array, shape (n_samples,)
        n_internal_nodes      int
        depth                 int
    """
    rng = np.random.default_rng(seed)
    n_samples = len(next(iter(x_samples.values())))
    n_nodes = _count_internal_nodes(tree)

    # Ideal evaluation (complex128)
    ideal_outputs = np.zeros(n_samples, dtype=np.complex128)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for i in range(n_samples):
            bindings = {k: v[i] for k, v in x_samples.items()}
            try:
                ideal_outputs[i] = eval_eml(tree, bindings)
            except Exception:
                ideal_outputs[i] = np.nan + 0j

    is_complex, complex_warning = _check_complex_intermediates(tree, x_samples)
    ideal_real = np.real(ideal_outputs)  # reference for RMSE

    # Noisy Monte Carlo trials
    noisy_outputs = np.full((n_trials, n_samples), np.nan, dtype=np.float64)
    for trial in range(n_trials):
        noise_model.setup_trial(n_nodes, rng)
        for i in range(n_samples):
            bindings = {k: float(v[i]) for k, v in x_samples.items()}
            node_counter = [0]
            noisy_outputs[trial, i] = _eval_noisy(
                tree, bindings, noise_model, rng, node_counter
            )

    # Samples with valid ideal outputs; noisy NaNs are included as "failures"
    ideal_valid = np.isfinite(ideal_real)
    if not np.any(ideal_valid):
        return {
            'rmse': np.nan,
            'rmse_std': np.nan,
            'rmse_per_trial': np.full(n_trials, np.nan),
            'bits_of_precision': -np.inf,
            'failure_rate': 1.0,
            'has_complex_intermediates': is_complex,
            'complex_warning': complex_warning,
            'ideal_outputs': ideal_outputs,
            'n_internal_nodes': n_nodes,
            'depth': tree_depth(tree),
        }

    ideal_v = ideal_real[ideal_valid]
    noisy_v = noisy_outputs[:, ideal_valid]   # may still contain NaN for bad trials

    # failure_rate: fraction of (trial, sample) pairs that produced NaN/Inf
    failure_rate = float(np.mean(~np.isfinite(noisy_v)))

    # Per-trial RMSE: average only over finite noisy outputs
    sq_err = np.where(np.isfinite(noisy_v),
                      (noisy_v - ideal_v[np.newaxis, :]) ** 2,
                      np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        per_trial_rmse = np.sqrt(np.nanmean(sq_err, axis=1))
        mean_rmse = float(np.nanmean(per_trial_rmse))
        std_rmse = float(np.nanstd(per_trial_rmse))

    output_range = float(np.ptp(ideal_v)) if len(ideal_v) > 1 else float(np.abs(ideal_v[0]))
    if output_range > 0 and mean_rmse > 0 and np.isfinite(mean_rmse):
        bits = float(np.log2(output_range / mean_rmse))
    elif mean_rmse == 0:
        bits = np.inf
    else:
        bits = -np.inf

    return {
        'rmse': mean_rmse,
        'rmse_std': std_rmse,
        'rmse_per_trial': per_trial_rmse,
        'bits_of_precision': bits,
        'failure_rate': failure_rate,
        'has_complex_intermediates': is_complex,
        'complex_warning': complex_warning,
        'ideal_outputs': ideal_outputs,
        'n_internal_nodes': n_nodes,
        'depth': tree_depth(tree),
    }
