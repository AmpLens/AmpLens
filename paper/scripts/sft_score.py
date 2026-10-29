"""Score the output of the upstream SFT evaluation by the success criterion of the paper.

    python paper/scripts/sft_score.py --n 4 OUT.pkl
    python paper/scripts/sft_score.py --n 6 part0.pkl@0 part1.pkl@5000

Each argument is an output file of `src/eval_sft_{n}pt.py --save_detailed`, followed by `@START` when the run used `--start_idx START`. A case is solved when, in the normal form of the AmpZip engine, the raw numerator bracket count and the number of numerator terms of its final expression are at most those of the target. The script prints the solve rate under this criterion and under the size test of the upstream script, which also counts denominator brackets.
"""
from __future__ import annotations

import argparse
import pickle

import ampzip_engine as E

from public_cases import load_public


def solved(out, target):
    o, t = E.Expr(out), E.Expr(target)
    return o.bracket_count() <= t.bracket_count() and o.num_terms() <= t.num_terms()


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, required=True, choices=[4, 5, 6])
    ap.add_argument("outputs", nargs="+", metavar="OUT.pkl[@START]")
    a = ap.parse_args()

    records = load_public(a.n)
    ours = upstream = total = 0
    seen = set()
    for spec in a.outputs:
        path, _, start = spec.partition("@")
        with open(path, "rb") as f:
            rows = pickle.load(f)["detailed_results"]
        for j, row in enumerate(rows):
            idx = int(start or 0) + j
            if idx in seen:
                raise SystemExit(f"case {idx} appears twice; check the @START offsets")
            seen.add(idx)
            total += 1
            ours += solved(str(row["final_expr"]), str(records[idx]["simple_expression"]))
            upstream += bool(row["solved_proper"])
    print(f"SFT n={a.n}: {total} cases; solved {ours} ({100 * ours / total:.2f}%) by the paper's "
          f"criterion, {upstream} ({100 * upstream / total:.2f}%) by the upstream size test")


if __name__ == "__main__":
    main()
