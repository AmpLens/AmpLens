# Limits and settings

This page describes accepted inputs, backend limits, search settings and output fields.

## 1. Notation

| Written as | Meaning |
|---|---|
| `ab(i, j)` | angle bracket ⟨i j⟩ |
| `sb(i, j)` | square bracket [i j] |
| `s(i, j)` | Mandelstam invariant s<sub>ij</sub> |
| `s(i, j, k, ...)` | Mandelstam invariant s<sub>ijk…</sub> |
| `chain('as', a, [m1, m2, ...], b)` | Dirac chain ⟨a\|(pₘ₁ + pₘ₂ + ⋯)\|b] |
| `chain('sa', a, [m1, m2, ...], b)` | Dirac chain [a\|(pₘ₁ + pₘ₂ + ⋯)\|b⟩ |
| integers, rationals (`-7/2`), `I` | coefficients |
| `+ - * / **` and parentheses | arithmetic |

Leg labels are integers from 1 to n. AmpZip expands `s()` and `chain()` into brackets before simplification. Results are also reported in `s()` and `chain()` notation. Polynomial factors in the denominator are supported by the GBFS variants.

The GBFS variants require integer coefficients. Inputs with non-integer rational coefficients or `I` can be processed by the selected SFT or CDS backends. The selected GBFS variants are skipped with a `LIMIT_ENGINE_COEFF` note. If none of the selected backends can process the input, the request is rejected.

Unsupported inputs include `sqrt`, `pi`, floats, free symbols, zero or symbolic leg labels, a bracket with equal legs (`ab(3, 3)`), and brackets whose labels exceed the number of legs.

Validation tags include `LIMIT_ATOM_UNKNOWN`, `LIMIT_LEG_LABEL`, `LIMIT_LEG_OUT_OF_RANGE` and `LIMIT_COEFF_CLASS`.

## 2. Number of legs

Four, five or six legs are supported. The number is inferred from the largest leg label. Set `--n-point` in the CLI or `n_point` in a web request to specify it explicitly. Unsupported multiplicities are reported as `LIMIT_N_POINT`.

## 3. Size

### Input limits

The CLI and the web service check every expression and target against the limits below before
parsing. A refused input lists each exceeded limit with the offending text and the flag or
variable that raises it. The defaults admit every input of the paper's test sets. The web
service reports its values under `input_limits` in `GET /api/limits`.

| CLI flag / web variable | Default | Limits |
|---|---|---|
| `--max-input-chars` / `WEBSVC_MAX_INPUT_CHARS` | 32000 | Length of the expression in characters. The longest paper input, a five-point Yang-Mills form, has about 21000. A public web deployment uses a lower value, such as the 8000 of the example configuration, since parsing runs in the request thread. |
| `--max-int-digits` / `WEBSVC_MAX_INT_DIGITS` | 6 | Digits of any integer written in the expression, whether a coefficient, an exponent or a leg label. |
| `--max-exponent` / `WEBSVC_MAX_EXPONENT` | 100 | Largest exponent, with nested powers multiplied out, so (x**4)**30 counts as 120. |
| `--max-power-nesting` / `WEBSVC_MAX_POWER_NESTING` | 0 | Powers nested inside another power. 0 refuses (x**2)**3 and x**2**3, and 1 admits one level. |
| `--max-nesting-depth` / `WEBSVC_MAX_NESTING_DEPTH` | 50 | Depth of nested parentheses and square brackets. |
| `--max-number-digits` / `WEBSVC_MAX_NUMBER_DIGITS` | 1000 | Estimated digits of the largest number the numeric parts produce. 999**999 has about 2997. At most 4300. |

Three rules are fixed: only the characters of the notation in section 1 are accepted, `^` is
refused with a hint to write `**`, and non-ASCII look-alikes such as `−`, `×` or `（` are refused
with the ASCII character to use.

### Search size

For the GBFS variants, `--gbfs-max-terms` limits numerator expansion when parsing the input and applying moves. An input that exceeds this limit is rejected. Moves that exceed it are skipped. Large searches are also constrained by the move budget, time limit and [available memory](KNOWN_ISSUES.md).

Coefficient overflow rejects the input or skips the affected move. Skipped moves are counted in `extra.overflow_skips`.

## 4. Input limits for the SFT and CDS checkpoints

The input limits below follow the SFT evaluation settings and also apply to CDS. The third column applies only to SFT.

| Legs | Maximum terms | Terms with individual actions (SFT) | Maximum brackets per term | Maximum distinct brackets |
|---|---|---|---|---|
| 4 | 20 | 10 | 8 | 12 |
| 5 | 25 | 25 | 15 | 20 |
| 6 | 35 | 30 | 20 | 30 |

These limits refer to terms of the expanded numerator. Brackets per term include numerator and denominator factors, counted with multiplicity. Distinct brackets are counted across the expression, with angle and square brackets counted separately. SFT can apply individual actions to the first 10, 25 or 30 terms for four-, five- or six-point inputs, respectively.

Input validation currently checks only top-level sums and factors. It does not fully enforce these limits for factored numerators or sums over a common denominator.

Size violations are reported as `LIMIT_TERMS`, `LIMIT_BRACKETS_PER_TERM` or `LIMIT_BRACKETS_TOTAL`. When validation detects a size violation, selected SFT and CDS backends are skipped and the selected GBFS variants continue. If only SFT or CDS was selected, the request is rejected with a suggestion to use a GBFS variant.

## 5. Budgets and stopping conditions

For GBFS, `extra.stop_reason` records why the search stopped. Skip counters record rejected moves by cause.

The tables list CLI options and their corresponding API fields where available. Defaults are for the CLI. Web requests are subject to server limits. With the default server configuration, `gbfs_max_seconds` is capped at 120 s.

### GBFS

| Parameter | Default | Description | Behaviour |
|---|---|---|---|
| `--gbfs-max-apps` / `gbfs_max_apps` | 60000 | Move budget per GBFS variant. For `gbfs_e` and `gbfs_f`, this budget is shared across terms. | Returns the best form found with `stop_reason=budget`. |
| `--gbfs-max-seconds` / `gbfs_max_seconds` | 300 s | Search time limit per GBFS variant. | Returns the best form found with `stop_reason=time`. A move, batch of moves or SymPy cancellation already in progress is allowed to finish. |
| `--gbfs-max-terms` / `gbfs_max_terms` | 50000 | Maximum numerator terms during input expansion and move application. | Rejects an input over the limit. Skips moves over the limit and increments `term_limit_skips`. |
| `--gbfs-split-max-terms` / `gbfs_split_max_terms` | 64 (0 = no limit) | Maximum number of terms searched separately by `gbfs_e` and `gbfs_f`. | If the original input exceeds the limit, numerator splitting is attempted. If that also exceeds the limit, the full expression is searched. |
| `--gbfs-c-cancel-max-terms` / `gbfs_c_cancel_max_terms` | 1200 (0 = no limit) | Maximum terms in the numerator or denominator passed to SymPy cancellation by `gbfs_c`. | Skips the expression and increments `node_skips` or `child_term_skips`. |
| `--gbfs-threads` / `gbfs_threads` | 1 | Worker threads for move application. In the CLI, 0 selects the thread count automatically. The web service adjusts 0 to 1. | Search order is preserved. The returned expression may differ if the time limit is reached or the search is cancelled. |
| `--gbfs-ordering` / `gbfs_ordering` | `factored` | `factored` uses the variant's default objective. `raw` uses the raw numerator bracket count. | Sets frontier ordering. See [GBFS variants](GBFS_VARIANTS.md) for the objectives. |
| `--gbfs-den-aware` / `gbfs_den_aware` | off | Uses denominator factors to adjust frontier ordering. | Cannot be combined with `raw` ordering. |
| `--gbfs-target-criterion` / `gbfs_target_criterion` | `brackets` | With a target, `brackets` requires a raw numerator bracket count no greater than the target's. `brackets-and-terms` also requires a numerator term count no greater than the target's. `none` checks equality only. | Determines `solved` for every backend. `model_free_gbfs` and `gbfs_d` also stop when the count criteria are met, with `stop_reason=goal`. |
| `--gbfs-audit-steps` / `gbfs_audit_steps` | off | Checks each derivation step over finite fields in `model_free_gbfs` and `gbfs_d`. | If a step does not pass, the input is returned and the step is listed. |

`gbfs_c` uses its own ordering and does not use `--gbfs-threads`.

### SFT and CDS

| Parameter | Default | Description | Behaviour |
|---|---|---|---|
| `--max-steps` / `max_steps` | 50 | Maximum steps in an SFT rollout. | Stops the rollout. |
| `--per-step-timeout` / `per_step_timeout` | 30 s | Time limit for one SFT identity application in SymPy. | Abandons the step and increments `timeout_count`. |
| `--sample-timeout` / `sample_timeout` | 120 s | SFT: time limit for one rollout, checked between steps. CDS: time spent waiting for decoding, excluding checkpoint loading and post-processing. | SFT stops at a step boundary with the best form found. CDS reports `timed_out=True`, `stop_reason=time` and `decode_still_running=True`, with no expression. Decoding continues in the background, and the CLI waits for it to finish before exiting. |
| `--sft-patience` / `sft_patience` | 0 (off) | Consecutive SFT steps without improvement. | Stops the rollout. |
| `--sft-beam-size` / `sft_beam_size`, `--sft-beam-temperature` / `sft_beam_temperature` | 5, 1.5 | Total rollouts, one greedy and the rest sampled, and the sampling temperature for `sft_best_of_n`. | — |
| `--cds-beam-size` / `cds_beam_size`, `--cds-temperature` / `cds_temperature`, `--cds-nucleus-p` / `cds_nucleus_p`, `--cds-length-penalty` / `cds_length_penalty` | 5, 1.5, 0, 1 | CDS beam size, temperature, nucleus probability threshold and length penalty. | — |

### Execution

| Parameter | Default | Description | Behaviour |
|---|---|---|---|
| `--post-process-timeout` | 30 s (0 = no limit) | Time limit for each SymPy post-processing operation in the helper process. | Returns the backend output without post-processing, with `post_process_timed_out=True`. |
| `--algo-grace-seconds` | 30 s | Additional time to wait for a backend after its time budget expires when backends run in parallel. | Returns a timeout result with `stop_reason=collector_timeout`. A result received before the overall call returns can replace it and is marked `overran`. |
| `--max-parallel-algos` | 1 | Maximum number of backends running concurrently. | The web service defaults to 4, configured by `WEBSVC_MAX_PARALLEL_ALGOS`. |

`--post-process-timeout` excludes time spent waiting for the helper or computing the final display counts.

Server limits and environment variables are described in the [web service documentation](../ampzip/webservice/README.md#configuration) and [server configuration](../ampzip/webservice/config.py). Active parameter limits are available from `GET /api/limits`.

The web service reports adjusted values in the submission response (`clamped`: `{field: [requested, applied]}`) and shows them on the page. The response also lists skipped backends.

## 6. Output

- Reported counts (`bracket_count_*` and the derivation rows) use the factored numerator bracket count. Bracket factors are counted with multiplicity after extracting the monomial factor common to all terms. Denominator brackets are excluded. `-1` means the form could not be counted. Without `--target`, `solved` is based on a reduction in this count. Backend-specific success conditions also apply.
- `final_expr_str` is the final expression returned by the backend. `min_expr_str` is its best expression according to the backend's selection rule.
- GBFS results are checked against the input by exact modular evaluation at random on-shell points. Passing provides probabilistic evidence for equality. See the [finite-field verification API](../ampzip/verify/verifier.py) for sampling details and assumptions.
- With `--target`, `solved` requires passing the finite-field check against the target and meeting `--gbfs-target-criterion`. For SFT and CDS, AmpZip checks `min_expr_str`, falling back to `final_expr_str`. `extra.backend_solved` records whether the backend reported success before this target evaluation. With `brackets` or `brackets-and-terms`, `solved` is `False` if the required numerator counts cannot be computed. `extra.target_check_note` gives the reason.
- `extra.raw_numerator_brackets_*` records raw numerator bracket counts for `model_free_gbfs` and `gbfs_d`. `extra.backend_bracket_count_*` preserves the counts originally returned by the backend.
- `--simplify-mode` applies SymPy post-processing to the printed forms (`none` by default).
