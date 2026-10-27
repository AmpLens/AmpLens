# Paper resources

Input datasets and evaluation commands for *Fast, training-free simplification of spinor-helicity amplitudes*.

| Path | Contents |
|---|---|
| `inputs/size_depth_suite.jsonl.gz` | the 21,600 cases of the size×depth suite |
| [inputs/ym5pt/](inputs/ym5pt/README.md) | the 1,024 expressions of the five-point Yang–Mills set |
| `inputs/below_target_cases.txt` | indices of the 296 five-point cases used in the target-free run in the appendix |
| `scripts/public_cases.py` | prints the cases of a public test set |
| `scripts/sft_score.py` | scores the output of the upstream SFT evaluation |
| `scripts/sft_suite.py` | runs SFT on the size×depth suite |
| `scripts/cds_public.py` | runs CDS on the public test sets or on one case of the suite |
| [LICENSE-DATA](LICENSE-DATA) | data provenance and licensing |

## Setup

Run the commands below from the repository root in Bash. The batch examples require `jq`.

```bash
pip install ampzip                  # GBFS
pip install 'ampzip[sft]'           # SFT
pip install 'ampzip[cds]'           # CDS
ampzip fetch-backends               # SFT code and checkpoints, and the public test sets
ampzip fetch-backends --cds --cds-data   # CDS weights and test splits
```

`ampzip fetch-backends` downloads the upstream code and public test sets.

## Inputs

**Size×depth suite.** One JSON object per line:

```json
{"n": 6, "K": 30, "L": 120, "idx": 144, "scrambled": "...", "simple": "..."}
```

`n` is the multiplicity, `K` the generation depth, `L` the bracket limit and `idx` the case index within the cell (0–299). `scrambled` is the input and `simple` the target. There are 72 cells: n ∈ {4, 5, 6}, K ∈ {15, 20, 25, 30}, L ∈ {60, 90, 120, 150, 200, 250}.

**Public test sets.** `scripts/public_cases.py` reads them from the downloaded upstream code and prints the same fields without `n`, `K` and `L`. Reading the five- and six-point sets takes a few minutes.

```bash
python paper/scripts/public_cases.py --n 4 > public_4.jsonl
python paper/scripts/public_cases.py --n 5 --non-vanishing > public_5_nonvanishing.jsonl
```

With `--non-vanishing`, `idx` is the index among cases whose target is not 0. `inputs/below_target_cases.txt` lists the 296 case indices used in the target-free run, with the same numbering as the appendix.

## GBFS

GBFS uses one worker thread by default. With `--json`, each run prints a JSON result record. When a target is supplied, `.results[0].solved` indicates whether the output meets the selected success criterion. The fields `.results[0].extra.apps` and `.results[0].n_steps` give the number of evaluated moves and the derivation length, respectively.

Bracket and term counts are evaluated in the AmpZip engine's normal form. On the public sets, the raw numerator bracket count and the number of numerator terms must each be at most those of the target. On the size×depth suite, only the raw numerator bracket count is compared.

For the commands below, set `N` to the multiplicity and `SCRAMBLED` and `TARGET` to the `scrambled` and `simple` fields of a case.

**Public sets** (budget of 6×10⁴ moves and 300 s):

```bash
ampzip --expr "$SCRAMBLED" --n-point "$N" --target "$TARGET" --algorithms model_free_gbfs --gbfs-target-criterion brackets-and-terms --gbfs-max-apps 60000 --gbfs-max-seconds 300 --json
```

**Size×depth suite** (6×10⁴ moves for the suite table, 2.4×10⁵ for the budget table):

```bash
ampzip --expr "$SCRAMBLED" --n-point "$N" --target "$TARGET" --algorithms model_free_gbfs --gbfs-target-criterion brackets --gbfs-max-apps 240000 --gbfs-max-seconds 300 --json
```

The search stops when the success criterion is met. The recorded move counts determine the solve rates at smaller move budgets. To evaluate a full cell:

```bash
gzip -dc paper/inputs/size_depth_suite.jsonl.gz | jq -c 'select(.n == 6 and .K == 30 and .L == 120)' | while read -r c; do
    ampzip --expr "$(jq -r .scrambled <<<"$c")" --n-point 6 --target "$(jq -r .simple <<<"$c")" --algorithms model_free_gbfs --gbfs-target-criterion brackets --gbfs-max-apps 240000 --gbfs-max-seconds 300 --json | jq -c --argjson idx "$(jq .idx <<<"$c")" '{idx: $idx, solved: .results[0].solved, moves: .results[0].extra.apps}'
done > gbfs_6_30_120.jsonl
```

**Increased budgets.** Cases left unsolved at 2.4×10⁵ moves are searched further with larger move budgets and time caps. The GBFS⁺ entries in the tables use only the default frontier ordering. The following command runs a case with a budget of 4×10⁷ moves and a time cap of 86,400 s:

```bash
ampzip --expr "$SCRAMBLED" --n-point "$N" --target "$TARGET" --algorithms model_free_gbfs --gbfs-target-criterion brackets --gbfs-max-apps 40000000 --gbfs-max-seconds 86400 --json
```

Case (n, K, L, idx) = (4, 25, 150, 289) remains unsolved with the default ordering after a cumulative 4×10⁹ moves. This search was continued from checkpoints using 8–16 CPU workers and time caps of up to 96 h per stage, with memory use reaching about 746 GiB. A separate run with `--gbfs-ordering raw`, a budget of 4×10⁷ moves and a time cap of 86,400 s solves it after 11,010,895 moves. The GBFS⁺ entries count this case as unsolved and mark it with ‡.

**Five-point Yang–Mills set** (no target, 4×10⁶ moves per case). Run all 1,024 cases:

```bash
for i in {0..1023}; do
    ampzip --expr-file "paper/inputs/ym5pt/YM_5pt_form${i}.txt" --n-point 5 --algorithms model_free_gbfs --gbfs-max-apps 4000000 --gbfs-max-seconds 86400 --json | jq -c --argjson idx "$i" '{idx: $idx, result: .results[0]}'
done > gbfs_ym5pt.jsonl
```

The timing measurements for the three representative cases use separate runs that stop when the raw numerator bracket count reaches that of the Parke–Taylor form. These runs follow the same search order and reach the target after the same number of evaluated moves:

```bash
ampzip --expr-file paper/inputs/ym5pt/YM_5pt_form1023.txt --n-point 5 --target "ab(1,2)**3/(ab(1,5)*ab(2,3)*ab(3,4)*ab(4,5))" --algorithms model_free_gbfs --gbfs-target-criterion brackets --gbfs-max-apps 30000000 --gbfs-max-seconds 14400 --json
```

Add `--gbfs-threads 24` for the multi-threaded column.

**Target-free run of the appendix** (no target, 3×10⁴ moves, 60 s), on the cases of `inputs/below_target_cases.txt`:

```bash
ampzip --expr "$SCRAMBLED" --n-point 5 --algorithms model_free_gbfs --gbfs-max-apps 30000 --gbfs-max-seconds 60 --json
```

The returned expression is in `.results[0].extra.raw_final_expr`. Compare its raw numerator bracket count with that of the target.

## SFT

**Public sets.** Select the parameters for the required multiplicity from the table below. The example runs the four-point set.

| N | MT | MTA | BPT | MB | ST |
|---|---:|---:|---:|---:|---:|
| 4 | 20 | 10 | 8 | 12 | 120 |
| 5 | 25 | 25 | 15 | 20 | 120 |
| 6 | 35 | 30 | 20 | 30 | 180 |

```bash
read -r N MT MTA BPT MB ST <<< "4 20 10 8 12 120"
AMP_SFT_REPO=$(python -c "from ampzip.simplifier import _paths; print(_paths.RL_REPO)")
AMP_SFT_OUTPUT="$PWD/sft_${N}.pkl"
(
    cd "$AMP_SFT_REPO" || exit 1
    export PYTHONPATH="$AMP_SFT_REPO/spinorhelicity:$AMP_SFT_REPO/spinorhelicity/environment:$AMP_SFT_REPO/src"
    export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
    python -u "src/eval_sft_${N}pt.py" --model "models/sft_${N}pt_500k/best_model.pt" --test_data "data/paper_test_${N}pt.pkl" --output "$AMP_SFT_OUTPUT" --n_point "$N" --max_terms "$MT" --max_terms_action "$MTA" --max_brackets_per_term "$BPT" --max_brackets "$MB" --max_steps 100 --timeout 30 --sample_timeout "$ST" --embed_dim 64 --num_heads 4 --num_layers 3 --ff_dim 128 --features_dim 128 --backtrack --max_backtracks 10 --proper_stop --reject_term_increase 1 --save_detailed --no_verbose
)
```

`scripts/sft_score.py` normalizes the saved outputs and reports solve rates under the paper's criterion and the upstream size test, which also counts denominator brackets:

```bash
python paper/scripts/sft_score.py --n "$N" "$AMP_SFT_OUTPUT"
```

To split a set over several processes, set `START` and `END` to the case-index range for each part and add `--start_idx "$START" --end_idx "$END"` to the evaluation command. Use a distinct `AMP_SFT_OUTPUT` path for each part, then pass the files to `sft_score.py` as `"${AMP_SFT_OUTPUT}@${START}"`.

**Size×depth suite.** `scripts/sft_suite.py` runs the same rollout on one cell, or a range of its cases, and scores it by the raw numerator bracket count:

```bash
python paper/scripts/sft_suite.py --n 6 --K 30 --L 120 --out sft_6_30_120.jsonl
```

Add `--start 144 --end 145` to run only case 144 of that cell, the case compared in the appendix.

## CDS

`scripts/cds_public.py` runs CDS on a public test set with a beam size of 20. It first uses greedy decoding and accepts an exact match to the target. Otherwise it checks the beam candidates in score order and accepts the first equivalent candidate that meets the public-set success criterion defined above.

```bash
python paper/scripts/cds_public.py --n 4 --out cds_4.jsonl                 # GPU 0
python paper/scripts/cds_public.py --n 6 --start 0 --end 5000 --gpu 1 --out cds_6a.jsonl
```

The script prints the solve rate. The paper's wall-clock times were measured with the original stopping rule, whose size test also counts denominator brackets.

For the comparison on case 144 in the appendix, run one case of the size×depth suite with the suite criterion:

```bash
python paper/scripts/cds_public.py --suite-case 6 30 120 144 --out case144.json
```

The output lists every beam candidate with its raw numerator bracket count, its number of terms and whether it passes the finite-field check against the input.

## Reproducibility

The reported timings use one CPU core per GBFS or SFT run and one A800 GPU per CDS run, except where a run states otherwise. Wall-clock times depend on the hardware, and solve rates may vary when time limits are reached. SFT stops each backtracking attempt after 30 s of wall clock, so its solve rates depend on the speed and load of the machine and can differ between runs.
