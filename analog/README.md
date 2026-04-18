# EML Analog Noise Simulator

This sub-package characterises how analog noise compounds through EML trees, addressing the paper's §4.2 claim that EML is a natural building block for analog computing circuits.

## Motivation

Every EML operation in analog hardware inherits component tolerance, thermal drift, and noise.  These errors compound with tree depth.  The question: **at what depth does analog EML become unusable, and how does noise model choice affect that threshold?**

## What's here

| File | Role |
|------|------|
| `noise_sim.py` | Core simulator: four noise models, `simulate()` API |
| `sweep.py` | Batch harness: sweeps formulas × σ levels × noise models → CSV |
| `analyze.py` | Plot generator: RMSE curves, bit-depth plots, viability heatmaps |
| `results/sweep.csv` | Full sweep output (10 formulas × 6 σ × 4 models = 240 rows) |
| `results/plots/` | Generated figures |

## Quick start

```bash
# Run the sweep (≈ 2 min for 500 trials)
python -m analog.sweep --trials 500 --out analog/results/sweep.csv

# Generate plots
python -m analog.analyze

# Single simulation
python3 - <<'EOF'
from eml_compiler import compile_expr
from analog.noise_sim import AdditiveGaussian, simulate
import numpy as np

tree = compile_expr('ln(x)')
result = simulate(tree, AdditiveGaussian(sigma=0.01), {'x': np.linspace(0.5, 4.0, 60)})
print(f"RMSE={result['rmse']:.4f}  bits={result['bits_of_precision']:.1f}")
EOF
```

## Noise models

| Model | Formula | Physical interpretation |
|-------|---------|------------------------|
| `AdditiveGaussian(σ)` | output += N(0, σ²) | Absolute offset; temperature or quantisation noise |
| `MultiplicativeGaussian(σ)` | output *= (1 + N(0, σ²)) | Component tolerance (1% resistor ≈ σ = 0.01) |
| `OneOverF(σ, α, cutoff)` | Spectrally shaped, RMS = σ | Flicker/drift; inter-node correlation via spectral synthesis |
| `MatchedPairs(σ_common, σ_diff)` | global drift + local scatter | Matched-transistor pairs: systematic errors partially cancel |

## Design decisions

- **No clamping.** The main training loop clamps `exp` arguments; the analog path intentionally does not.  Analog circuits saturate/rail; measuring the gap between ideal and rail-clipped behaviour is the whole point.
- **float64 for noisy, complex128 for ideal.** Analog wires carry real voltages.  Trees that require complex intermediates are flagged with `has_complex_intermediates=True`; their failure rate reflects physical infeasibility rather than just noise.
- **Failure rate.** When noise pushes an intermediate to a region where `log` is undefined (negative input), the circuit "fails" — we record this rather than silently clamping.

## Findings

Tested on 10 formulas, 6 noise levels (0.01 %–5 %), 4 noise models, 200 Monte Carlo trials.  Thresholds: **usable** ≥ 8 bits, **marginal** 6–8 bits, **failed** < 6 bits.

### Additive Gaussian at 1 % per-node σ

| Formula | Depth | Bits | Status | Notes |
|---------|------:|-----:|--------|-------|
| exp(x) | 1 | 9.3 | **usable** | |
| eml(x,y) | 1 | 8.8 | **usable** | Raw operator |
| exp(exp(x)) | 2 | 8.3 | **usable** | Nested exp — no saturation in tested range |
| ln(x) | 3 | 7.2 | marginal | |
| x+y | 6 | 5.8 | failed | |
| x·y | 8 | 5.8 | failed | 35 % circuit-failure rate |
| sqrt(x) | 12 | — | **failed** | 100 % failure; requires complex intermediates |
| 1/x | 14 | 5.6 | failed | 49 % failure rate |
| x/y | 14 | 6.0 | failed | 19 % failure; complex intermediates |
| x²+y² | 16 | 4.5 | failed | 54 % failure; complex intermediates |

### Interpretation

**EML is analog-friendly for transcendentals.**  At 1 % per-node noise (component-grade precision), `exp(x)` and the raw EML cell retain > 9 bits and are usable.  `ln(x)` at depth 3 is marginal (7.2 bits), dropping to failed only above 2 % noise.

**Arithmetic is marginal to failed.**  `x+y` and `x·y` fall below 6 bits at 1 % noise.  Dropping to 0.5 % (precision-analog range) lifts them to marginal (6–7 bits).  The issue is not just noise compounding but **circuit-failure rate**: multiplicative noise or 1/f drift can push an intermediate node negative, causing a `log` of a negative number — the circuit equivalent of a transistor going out of its bias region.

**Complex-intermediate trees are physically infeasible.**  The EML compiler's expansions of `sqrt`, `x²+y²`, and `x/y` route through complex intermediates.  Single-wire real-valued analog circuits cannot represent these values.  The simulator flags them; their 19–100 % failure rates represent true analog incompatibility, not noise sensitivity.  A two-wire (real, imaginary) topology could in principle handle these, with an explicit doubling of wire count and area cost.

**Matched pairs are worth it.**  Across all working trees, matched-pair transistors recover 0.5–1.0 additional bits of precision at no increase in device count.  At 1 % σ, `ln(x)` goes from marginal (7.2 bits) to usable (8.2 bits); `x·y` from 5.8 to 6.7 bits.  The improvement is largest for deep trees, consistent with the common-mode error partially cancelling across multiple nodes.

**1/f noise is comparable to additive** for trees with ≥ 2 nodes.  At 0.5 % σ, 1/f yields similar or slightly *better* effective precision than independent additive noise, because the inter-node correlation creates a common-mode drift that partially cancels in subtraction stages.  Shallow single-node trees are the exception: a single-node tree exposed only one draw of the noise sequence, making the result dependent on that trial's correlated offset rather than averaging over many draws.

### Noise model comparison at 0.5 % σ (sample)

| Formula | Additive | Multiplicative | 1/f | Matched-pairs |
|---------|----------|----------------|-----|---------------|
| exp(x) | 10.3 | — | 10.5 | 10.8 |
| ln(x) | 8.2 | — | 8.6 | 9.2 |
| x+y | 6.7 | 6.8 | 7.1 | 7.4 |
| x·y | 6.7 | 6.3 | 7.2 | 7.7 |

(Multiplicative is omitted from this table because component-tolerance models at 0.5 % show marginal shifts from additive; see the CSV for full data.)

### Revised analog pitch

Based on these results, the honest analog pitch becomes:

> **EML is analog-viable for transcendentals (exp, ln) at depth ≤ 3**, assuming component tolerances ≤ 0.5 % per node and matched-pair layout.  This covers the core building block (eml itself), exponential, and logarithm — the functions most relevant to neural network activations and translinear-loop signal processing.
>
> **Arithmetic (add, multiply) is marginal at achievable tolerances.** The multi-stage EML expansions introduce failure-mode risk from log-of-negative, not just noise amplification.  These operations are better handled by conventional circuits (current-mode add, translinear multiply) rather than EML trees.
>
> **Functions requiring complex intermediates (sqrt, division, powers) need a two-wire real+imaginary topology** or should be avoided in analog implementations altogether.

This conclusion both defends and refines the paper's §4.2 claim: the analog opportunity for EML is real but narrower than "any formula in the bootstrap chain."

## Cross-references

- Issue #34: this simulator
- Issue #35: SPICE validation — uses these predictions as the target to match
- `eml_compiler.py`: source of the compiled trees evaluated here
- Paper §4.2 (p.12): the analog-computing claim being tested
- Paper §4.3 (p.15): the clamping decision (not applied here by design)
