# eml-sr vs PySR — head-to-head

Per-target recovery comparison. See `pysr_compare.py` for methodology.
Regenerate with::

    PYSR_ENABLED=1 python -m benchmarks.pysr_compare --output benchmarks/pysr_compare.md

**Exact recovery** ≡ `RMSE < 1e-06` in the original coordinate space.
**Size** ≡ node count (both engines). **Ref EML size** is the compiler's canonical EML
tree for the target formula.

| formula | eml-sr time | eml-sr rmse | eml-sr exact | eml-sr size | ref EML size | PySR time | PySR rmse | PySR exact | PySR size |
|---|---:|---:|:---:|---:|---:|---:|---:|:---:|---:|
| `exp(x)` | 38.8s | 1.11e-16 | ✓ |   3 |   3 | 69.9s | 3.10e-07 | ✓ |  18 |
| `exp(x) - ln(x)` | 37.9s | 3.95e-16 | ✓ |   3 |  19 | 70.5s | 5.70e-07 | ✓ |   9 |
| `e` | 38.3s | 0.00e+00 | ✓ |   3 |   3 | 11.3s | 0.00e+00 | ✓ |   1 |
| `e - ln(x)` | 37.6s | 1.26e-16 | ✓ |   3 |  19 | 59.4s | 3.37e-08 | ✓ |   6 |
| `ln(x)` | 241.5s | 9.64e-17 | ✓ |  15 |   7 | 72.9s | 3.14e-08 | ✓ |  16 |
| `exp(exp(x))` | 111.8s | 1.33e-16 | ✓ |   7 |   5 | 70.8s | 9.00e-08 | ✓ |  13 |
| `exp(exp(exp(x)))` | 243.6s | 8.84e-16 | ✓ |  15 |   7 | 67.1s | 1.61e-06 |   |  10 |
| `exp(x) - 1` | 112.2s | 5.44e-17 | ✓ |   7 |  13 | 67.5s | 1.63e-07 | ✓ |   7 |
| `exp(x1) - ln(x2)` | 40.7s | 2.62e-16 | ✓ |   3 |  19 | 11.2s | 3.91e-07 | ✓ |   5 |
| `exp(x1 + x2) = exp(x1)*exp(x2)` | 520.7s | 1.28e+00 |   |   3 |  23 | 63.5s | 1.53e-07 | ✓ |  10 |
| `x1 / x2^2` | 517.6s | 1.09e+00 |   |  31 |  95 | 64.3s | 1.01e-07 | ✓ |  13 |
| `x` | 4.7s | 0.00e+00 | ✓ |   1 |   1 | 10.2s | 0.00e+00 | ✓ |   1 |
| `0.3 * x` | 446.2s | 4.31e-01 |   |   1 |  35 | 71.9s | 0.00e+00 | ✓ |   7 |
| `9.81 * x` | 410.2s | 3.96e+01 |   |  15 |  35 | 75.3s | 1.76e-06 |   |   7 |
| `ln(x1) - ln(x2)` | 223.1s | 5.14e-01 |   |  11 |  23 | 9.4s | 4.43e-08 | ✓ |   4 |
| `exp(x1) + exp(x2)` | 524.6s | 2.44e+00 |   |   3 |  25 | 70.6s | 1.88e-07 | ✓ |  19 |

**eml-sr exact-recovery:** 10/16
**PySR exact-recovery:**   14/16

## Positioning

Three axes the benchmark makes concrete:

1. **Grammar uniformity.** Every eml-sr node is the same `eml` operator. PySR
   expressions are heterogeneous ASTs over `{+, -, *, /, exp, log, sqrt}`.
   The "ref EML size" column shows what uniformity costs: the compiler's
   canonical EML tree for a given formula, independent of search.

2. **Exact recovery vs Pareto front.** eml-sr aims for machine-epsilon RMSE on
   a specific symbolic form. PySR trades complexity against RMSE and often
   lands near but not on the target — visible in the table as a PySR row with
   low RMSE but no ✓ and a small AST that doesn't algebraically match.

3. **Expression length.** EML trees for standard operations are long by
   design — multiplication is depth 8 in EML, depth 1 in a conventional AST.
   That's the cost of grammar uniformity, not a bug. The "size" vs
   "ref EML size" columns tell this story directly.

PySR is usually faster and finds more formulas on harder targets. eml-sr
recovers exact symbolic forms cleanly when the target is inside its reachable
vocabulary (the elementary basis), and the tree is always a pure `eml` circuit.

