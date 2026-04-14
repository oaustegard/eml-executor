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
between the constant `1` and the variable `x`. Each internal node computes `eml(left, right)`,
and its gate 3-way-routes each child input between `1`, `x`, and the child's own output — so
`x` can flow straight into any gate, not just through leaves.

Training (Adam + tau-annealing) pushes the soft weights toward a single hard choice per gate,
recovering an exact symbolic expression. When the generating law is elementary, the snapped
weights yield machine-epsilon RMSE. The final expression is then simplified by a recursive
AST rewriter that applies EML identities (e.g. `eml(1, eml(eml(1, x), 1)) → ln(x)`).

### Current recovery rates

| Target | Depth | RMSE |
|--------|-------|------|
| `exp(x)` | 1 | 4.6e-16 |
| `e` (constant) | 1 | 0 |
| `ln(x)` | 3 | 7.3e-17 |
| `exp(x) - ln(x)` | 1 | 3.2e-16 |

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
