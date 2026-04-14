# eml-executor

A compiled transformer executor whose only compute instruction is `eml(x, y) = exp(x) − ln(y)`. Every memory access is a parabolic attention dot product. Built in Mojo.

## Background

In digital electronics, a single gate (NAND) builds any Boolean circuit. [Odrzywolek (2026)](https://arxiv.org/abs/2603.21852) showed that a single binary operator does the same for continuous mathematics: `eml(x, y) = exp(x) − ln(y)`, paired with the constant 1, generates every function on a scientific calculator — arithmetic, transcendentals, trig, constants including π, e, and i.

Separately, [llm-as-computer](https://github.com/oaustegard/llm-as-computer) demonstrated that a transformer's attention mechanism can serve as a general-purpose computer: parabolic key encoding gives exact memory addressing via dot products, and programs compile into weight matrices.

This project combines both ideas: a machine where **attention handles memory** and **EML handles computation**.

## Architecture

**ISA:** 3 opcodes — `PUSH`, `EML`, `HALT`

**Memory:** Parabolic attention with 2D keys. Write at address `a`: key = `(2a, −a²)`. Read for address `a`: query = `(a, 1)`, score via dot product, argmax selects entry. Last-writer-wins for overwrites.

**Compute:** One operation — `eml(x, y) = exp(x) − ln(y)` over ℂ (principal branch).

**Programs** are RPN (postfix) sequences. From the paper:

| Expression | RPN Program | Steps |
|---|---|---|
| `e` | `1 1 EML` | 3 |
| `exp(x)` | `x 1 EML` | 3 |
| `ln(x)` | `1 1 x EML 1 EML EML` | 7 |
| `0` | `1 1 1 EML 1 EML EML` | 7 |
| `exp(iπ) = −1` | `iπ 1 EML` | 3 |

## Verified Results

All tests pass against known values:

- `e = eml(1, 1)` — error: 7.7e-13
- `exp(2) = eml(2, 1)` — exact
- `ln(5)` via 3 chained EMLs — error: 7.5e-11
- `ln(0.5)` — error: 2.3e-12
- `0 = ln(1)` — error: 2.0e-12
- Euler's formula `eml(iπ, 1) = −1` — error: 1.2e-16

## Performance

Mojo 0.26.2, Ubuntu 24 (CPU):

| Benchmark | Time | Throughput |
|---|---|---|
| `exp(x)` (3-step program) | ~1100 ns | — |
| `ln(x)` (8-step program, with attention) | ~1700 ns | ~4.7M steps/sec |
| Raw 3×EML chain (no attention) | ~150 ns | ~6.7M chains/sec |

For reference, [llm-as-computer](https://github.com/oaustegard/llm-as-computer)'s Mojo executor does 67–126M steps/sec — but those are integer operations, not transcendental function evaluations.

## Building

```bash
# JIT (always works)
mojo eml_executor.mojo

# Native binary (requires -Xlinker -lm, see modular/modular#5925)
mojo build eml_executor.mojo -o eml_executor -Xlinker -lm
./eml_executor
```

**Note:** `mojo build` does not automatically link libm when `std.math` functions are used. The `-Xlinker -lm` flag is required. This is a [known issue](https://github.com/modular/modular/issues/5925). `mojo run` (JIT mode) is unaffected.

## Design Insight

The two parent projects solve the same problem from opposite ends:

- **llm-as-computer**: discrete operations → compiled into continuous (attention + linear algebra)
- **EML paper**: continuous operations → unified into one primitive

This executor is the intersection. The program structure lives in attention (the discrete part). The computation lives in EML (the continuous part). One element does the thinking, one element does the remembering.

## Related

- [Interactive EML Calculator](https://oaustegard.github.io/fun-and-games/eml-calc.html) — type any expression, watch it decompose into an EML tree
- [Two Buttons and a Constant](https://muninn.austegard.com/blog/two-buttons-and-a-constant.html) — technical write-up
- [Two Buttons — for the Back Row](https://muninn.austegard.com/blog/two-buttons-back-row.html) — plain-language explainer
- [Odrzywolek (2026)](https://arxiv.org/abs/2603.21852) — the EML paper
- [llm-as-computer](https://github.com/oaustegard/llm-as-computer) — the parabolic attention executor

## License

MIT
