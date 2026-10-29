# Reproduce the AmpZip demos

These recordings use AmpZip 1.0.0 at AmpLens revision `c4ea949`, built from source on macOS arm64 with Python 3.13.7. The GIFs contain actual interface screenshots with pauses added for reading. No target is supplied to either search, and the recorded paths are not claimed to be shortest possible derivations.

## Run locally

Follow the [source installation instructions](../../AmpZip/README.md#install), then start the [local web interface](../../AmpZip/README.md#web-service). Paste one of the input files below, select `model_free_gbfs`, and use these settings:

| Setting | Six-step example | Two-step example |
|---|---|---|
| Input | [six-step-input.txt](six-step-input.txt) | [two-step-input.txt](two-step-input.txt) |
| n-point | 6 | 6 |
| gbfs-max-apps | 20,000 | 10,000 |
| gbfs-max-seconds | 20 | 20 |
| gbfs-threads | 1 | 1 |
| gbfs-audit-steps | Enabled | Enabled |
| simplify-mode | none | none |

Click **Simplify**, then open **details** beside a derivation step to inspect its rewrite rule. Long expressions and rules can be scrolled horizontally. The input files write squared and cubed brackets as products because this version's web-input filter can reject independent powers as exponent towers.

## Six-step example

[Animation](six-step.gif) · [Static result](six-step.png)

This synthetic size–depth benchmark case is in [`paper/inputs/size_depth_suite.jsonl.gz`](../../paper/inputs/size_depth_suite.jsonl.gz): n=6, K=30, L=200, idx=95.

The returned derivation uses four Schouten rewrites, one squared-momentum-conservation rewrite, and one momentum-conservation rewrite. Numerator terms decrease as **14 → 13 → 12 → 10 → 9 → 5 → 1**. The final expression is

```text
-ab(1,4)**2*ab(2,4)*ab(2,6)*sb(1,2)*sb(1,6)/(ab(3,4)*sb(1,3))
```

The interface also displays an equivalent expression in chain notation. Raw numerator bracket count decreases from 154 to 6; the interface's factored display count decreases from 115 to 6. These are different measures.

## Two-step example

[Animation](two-step.gif) · [Static result](two-step.png)

This synthetic six-point example has a polynomial denominator written as a Dirac chain. The recorded momentum-squared and momentum rewrites reduce the numerator term count **12 → 6 → 1**, returning `ab(3,4)`. Each step includes algebraic collection and cancellation; one rewrite can therefore eliminate several terms. Two recorded steps do not mean the search tried only two moves.

## Checks and interpretation

Both searches report successful finite-field soundness and step audits. Each step of the longer derivation was additionally checked through the SymPy evaluation path at 16 sampled points for each of two primes; a deliberately doubled final expression failed the equality check. Finite sampled checks are consistency evidence, not a proof of general algorithm correctness.

The 33-second and 26-second animations include reading pauses and show selected successful examples. They are not runtime benchmarks or estimates of performance on arbitrary inputs. Static images and copyable input files accompany both animations.
