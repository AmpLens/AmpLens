"""Run SFT on cases of the size-by-depth suite with the inference settings used in the paper.

    python paper/scripts/sft_suite.py --n 6 --K 30 --L 120 --out sft_6_30_120.jsonl
    python paper/scripts/sft_suite.py --n 4 --K 15 --L 60 --start 0 --end 50 --out part0.jsonl

Requires `pip install 'ampzip[sft]'` and `ampzip fetch-backends`. Nothing from the upstream repository is copied here: the script loads the released checkpoint and calls the upstream rollout `evaluate_sample_backtrack` with the arguments that `eval_sft_{n}pt.py --backtrack --proper_stop --reject_term_increase 1` passes to it (up to 10 backtracks, proper stopping, term-increase rejection with N=1, anti-cycle, greedy), at most 100 steps, a 30 s limit per step and a per-case limit of 120 s at four and five points and 180 s at six.

A case is solved when its output passes the finite-field check against the target (`value_ok`) and its raw numerator bracket count, computed by the AmpZip engine, is at most that of the target. One JSON line is written per case and a summary is printed at the end.
"""
from __future__ import annotations

import argparse
import gzip
import inspect
import json
import os
import time

# One rollout per process: the model is small, so a wider thread pool only adds overhead.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from ampzip.simplifier import _paths  # noqa: E402

_paths.ensure_importable()

import torch  # noqa: E402
import sympy as sp  # noqa: E402
import ampzip_engine as E  # noqa: E402
from ampzip.verify import equal  # noqa: E402
from ampzip.simplifier.validate import _get_bracket_classes  # noqa: E402
from ampzip.simplifier.limits import SFT_LIMITS  # noqa: E402
from ampzip.simplifier.algorithms.sft_greedy import (  # noqa: E402
    _get_backend,
    _default_checkpoint_path,
)

torch.set_num_threads(1)

_AB, _SB = _get_bracket_classes()
LOCAL = {"ab": _AB, "sb": _SB, "I": sp.I}

# Defaults of the eval_sft_{n}pt.py command line.
PATIENCE = 5
MAX_BACKTRACKS = 10
MAX_TERM_INCREASE = 5
REJECT_TERM_INCREASE = 1
SAMPLE_TIMEOUT = {4: 120.0, 5: 120.0, 6: 180.0}
SUITE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "inputs", "size_depth_suite.jsonl.gz")


def load_cell(path, n, K, L):
    with gzip.open(path, "rt") as f:
        rows = [r for r in map(json.loads, f) if (r["n"], r["K"], r["L"]) == (n, K, L)]
    if not rows:
        raise SystemExit(f"no cases with n={n}, K={K}, L={L} in {path}")
    return rows


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suite", default=SUITE)
    ap.add_argument("--n", type=int, required=True, choices=[4, 5, 6])
    ap.add_argument("--K", type=int, required=True, help="generation depth")
    ap.add_argument("--L", type=int, required=True, help="bracket limit")
    ap.add_argument("--start", type=int, default=0, help="first case index")
    ap.add_argument("--end", type=int, default=0, help="end of the case range (0 = all)")
    ap.add_argument("--out", required=True, help="JSON lines, one per case")
    a = ap.parse_args()

    cases = load_cell(a.suite, a.n, a.K, a.L)
    cases = cases[a.start:(a.end or len(cases))]
    backend = _get_backend(a.n, _default_checkpoint_path(a.n), "cpu")
    limits = SFT_LIMITS[a.n]

    # The three eval_sft_{n}pt.py copies differ: 4pt has no `max_term_increase`. Pass each n exactly the arguments its own command line would.
    rollout_fn = backend.eval_mod.evaluate_sample_backtrack
    accepted = set(inspect.signature(rollout_fn).parameters)
    kwargs = {k: v for k, v in {
        "sample_timeout": SAMPLE_TIMEOUT[a.n],
        "verbose": False,
        "sample_idx": 0,
        "use_anticycle": True,
        "patience": PATIENCE,
        "max_backtracks": MAX_BACKTRACKS,
        "max_term_increase": MAX_TERM_INCREASE,
        "random_policy": False,
        "proper_stop": True,
        "checkpoint_by_terms": False,
        "reject_term_increase": REJECT_TERM_INCREASE,
    }.items() if k in accepted}

    solved = 0
    t0 = time.perf_counter()
    with open(a.out, "w") as out:
        for case in cases:
            target_brackets = E.Expr(case["simple"]).bracket_count()
            row = {"n": a.n, "K": a.K, "L": a.L, "idx": case["idx"],
                   "target_brackets": target_brackets, "solved": False, "out_brackets": None,
                   "out_expr": None, "value_ok": None, "steps": None, "error": None}
            ts = time.perf_counter()
            try:
                expr = sp.sympify(case["scrambled"], locals=LOCAL)
                target = sp.sympify(case["simple"], locals=LOCAL)
                rollout = rollout_fn(backend.model, expr, target, backend.action_space, a.n,
                                     limits.max_terms, limits.max_brackets_per_term, 100,
                                     backend.device, 30.0, **kwargs)
                # The expression the rollout reports, without further post-processing.
                out_expr = str(rollout.get("min_expr", rollout["final_expr"]))
                row["out_expr"] = out_expr
                row["steps"] = rollout.get("steps")
                row["out_brackets"] = E.Expr(out_expr).bracket_count()
                row["value_ok"] = equal(sp.sympify(out_expr, locals=LOCAL), target, a.n)
                row["solved"] = row["value_ok"] is True and row["out_brackets"] <= target_brackets
            except Exception as e:  # noqa: BLE001 - recorded per case
                row["error"] = repr(e)[:200]
            row["wall_seconds"] = round(time.perf_counter() - ts, 3)
            solved += row["solved"]
            out.write(json.dumps(row) + "\n")
            out.flush()

    print(f"SFT n={a.n} K={a.K} L={a.L}: solved {solved}/{len(cases)} "
          f"({time.perf_counter() - t0:.0f} s); per-case results in {a.out}")


if __name__ == "__main__":
    main()
