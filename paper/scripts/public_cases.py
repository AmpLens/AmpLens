"""Print the cases of a public test set as JSON lines: {"idx", "scrambled", "simple"}.

    python paper/scripts/public_cases.py --n 4 > public_4.jsonl
    python paper/scripts/public_cases.py --n 5 --non-vanishing > public_5_nonvanishing.jsonl

The public sets are not redistributed here. They are read from `data/paper_test_{n}pt.pkl` in the upstream tree written by `ampzip fetch-backends`. `idx` is the position in the full set, or, with --non-vanishing, the position among the cases whose target is not 0 (the case index of the below-target appendix).
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys

from ampzip.simplifier import _paths


def load_public(n):
    """The records of the n-point public test set, in order."""
    repo = _paths.ensure_importable()
    # The five- and six-point sets were pickled against a top-level `bracket_env` module.
    import environment.bracket_env as bracket_env
    sys.modules.setdefault("bracket_env", bracket_env)
    with open(repo / "data" / f"paper_test_{n}pt.pkl", "rb") as f:
        return pickle.load(f)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, required=True, choices=[4, 5, 6])
    ap.add_argument("--non-vanishing", action="store_true", help="only cases whose target is not 0")
    a = ap.parse_args()

    records = load_public(a.n)
    if a.non_vanishing:
        records = [r for r in records if str(r["simple_expression"]) != "0"]
    for idx, r in enumerate(records):
        print(json.dumps({"idx": idx, "scrambled": str(r["scrambled_expression"]),
                          "simple": str(r["simple_expression"])}))


if __name__ == "__main__":
    main()
