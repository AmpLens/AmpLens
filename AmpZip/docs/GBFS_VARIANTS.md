# GBFS variants

The five training-free GBFS variants use the same spinor-helicity identities. They differ in how expressions are represented, ranked and split into separate searches.

| Variant | Search strategy |
|---|---|
| `model_free_gbfs` | Searches the full expression using the factored numerator bracket count, with the numerator term count breaking ties. |
| `gbfs_c` | Expands the numerator and denominator and uses SymPy to cancel common polynomial factors after each move. Its internal bracket count includes the denominator. |
| `gbfs_d` | Counts each two-particle Mandelstam invariant s<sub>ij</sub>, Dirac chain with one momentum, or remaining bracket as one unit, summed over numerator terms. |
| `gbfs_e` | Simplifies additive terms independently with the default objective, then recombines them. |
| `gbfs_f` | Uses the searches of `gbfs_e`, then compares the input, the original term sum, the searched term sum, and a sum that keeps the smaller form of each term. Selection uses an internal bracket count that includes the denominator. |

Where possible, `gbfs_e` and `gbfs_f` split the original input before combining fractions. Otherwise, they fall back to splitting the numerator or searching the full expression. When terms are searched separately, the searches do not explore cancellations between different terms.

For `gbfs_f`, the selected result has an internal count no greater than the input's. This guarantee does not apply to the reported numerator count.

## Common features and limits

- **Moves.** For each bracket present in the numerator, the move menu contains C(n − 2, 2) Schouten rewrites, 2(n − 1) momentum-conservation rewrites and 2<sup>n − 2</sup> momentum-squared rewrites. Each can be applied across the numerator or to one bracket factor in a term containing it. See the [engine API](ENGINE.md).
- **Budgets.** `--gbfs-max-apps` sets the move budget for each variant. For `gbfs_e` and `gbfs_f`, this budget is shared across all terms. `--gbfs-max-seconds` limits search time. A move or SymPy cancellation already in progress is allowed to finish. `extra.stop_reason` records why the search ended.
- **Term limit.** `--gbfs-max-terms` limits numerator expansion in the Rust engine. Moves that exceed the limit are skipped and counted. See [Limits](LIMITS.md) for all settings.
- **Target evaluation.** All five variants support evaluation against a target. Only `model_free_gbfs` and `gbfs_d` support stopping when the target's count criteria are met. The other variants use the target only to assess the final result.
- **Finite-field check.** Search results must pass the randomized finite-field check against the input before they are reported. Passing provides probabilistic evidence for equality.
- **Reported counts.** All variants report the factored numerator bracket count. For example, ⟨12⟩⟨23⟩⟨34⟩ + ⟨12⟩⟨23⟩⟨45⟩ counts 4. Denominator brackets are excluded.
