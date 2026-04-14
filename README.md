# eml-sr

**EML Symbolic Regression** — discover elementary formulas from data using a single operator.

Based on [Odrzywolek (2026)](https://arxiv.org/abs/2603.21852), "All elementary functions from a single operator":
the operator `eml(x, y) = exp(x) - ln(y)`, together with the constant `1`, generates all standard
elementary functions. This makes `S → 1 | x | eml(S, S)` a complete, regular search space for
symbolic regression.

## Quick start

```python
import numpy as np
from eml_sr import discover

x = np.linspace(0.5, 5.0, 50)
y = np.log(x)  # secret formula

result = discover(x, y, max_depth=4, n_tries=8)
print(result["expr"])  # → ln(x)
```

## How it works

A full binary tree of depth *n* has 2ⁿ leaves and 2ⁿ−1 internal nodes. Each leaf soft-routes
between the constant `1` and the variable `x`. Each internal node computes `eml(left, right)`.
Gate logits control whether each child input is used or bypassed to `1`.

Training (Adam + tau-annealing) pushes the soft weights toward hard 0/1 values,
recovering an exact symbolic expression. When the generating law is elementary,
the snapped weights yield machine-epsilon RMSE.

### Current recovery rates

| Target | Depth | Rate | RMSE |
|--------|-------|------|------|
| `exp(x)` | 1 | 100% | 4.6e-16 |
| `e` (constant) | 1 | 100% | 0 |
| `ln(x)` | 3 | 100% | 7.3e-17 |

## Files

| File | Description |
|------|-------------|
| `eml_sr.py` | Symbolic regression engine (the product) |
| `legacy/eml_executor.mojo` | Original parabolic-attention stack machine (archived) |
| `legacy/test_eml.mojo` | 109-test bootstrap chain verification (archived) |

## References

- Paper: [arXiv:2603.21852](https://arxiv.org/abs/2603.21852)
- Blog: [Two buttons and a constant](https://muninn.austegard.com/blog/two-buttons-and-a-constant.html)
- Blog: [Two buttons, back row](https://muninn.austegard.com/blog/two-buttons-back-row.html)
- Demo: [EML Calculator](https://austegard.com/fun-and-games/eml-calc.html)
