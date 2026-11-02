"""The reported bracket count is the factored numerator count, for every backend and every
derivation row, and the backends' internal counts are untouched.

    python tests/test_display_count.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))   # the package root

import sympy as sp  # noqa: E402

from ampzip.atoms import ab, sb  # noqa: E402
from ampzip.simplifier.algorithms._post import (count_brackets, display_bracket_count,  # noqa: E402
                                                _display_count_sympy)
from ampzip.simplifier.result import AlgorithmResult  # noqa: E402
from ampzip.simplifier.runner import _report_counts, run_all  # noqa: E402

README = "ab(1,2)**3/(ab(2,3)*ab(3,4)*ab(4,1))"
CASES = {                       # expression -> factored numerator count
    README: 3,
    "ab(1,2)/ab(3,4)**5": 1,
    "ab(1,2)*ab(2,3)*ab(3,4) + ab(1,2)*ab(2,3)*ab(4,5)": 4,
    "(ab(1,2)**2*sb(1,2) - ab(1,2)**2*sb(3,4))/(ab(1,2)*sb(1,2))": 3,   # cancels one ab(1,2)
    "(ab(1,2)*ab(2,3) + ab(1,2)*ab(3,4))/ab(1,2)": 2,
    "ab(1,2)*ab(3,4)/ab(1,4)": 2,
    "0": 0,
    "(ab(1,2)+ab(3,4))**2": 6,                                          # expanded: three terms
    "ab(1,2)*(ab(1,2)+ab(3,4))**2": 7,                                  # 1 + (2+2+2): common ab(1,2) once
}
FALLBACK = {"ab(1,2)/2": 1, "I*ab(1,2)": 1, "I*ab(1,2)*ab(2,3)/(2*sb(1,2)**3)": 2,
            "(ab(1,2)+ab(3,4))**2/2": 6, "I*ab(1,2)*(ab(1,2)+ab(3,4))**2": 7}


def _check(cond, msg):
    print(("  ok " if cond else "  FAIL ") + msg)
    if not cond:
        raise AssertionError(msg)


def main():
    for s, want in CASES.items():
        _check(display_bracket_count(s) == want, f"engine path {s} -> {want}")
        _check(_display_count_sympy(s) == want, f"sympy rule agrees on {s}")
    for s, want in FALLBACK.items():
        _check(display_bracket_count(s) == want, f"engine-refused {s} -> {want} via sympy")
    _check(display_bracket_count("ab(1,2") == -1 and display_bracket_count("") == -1, "uncountable -> -1")
    # the internal metric is unchanged (denominator counted, factor_terms over the whole tree)
    e = sp.sympify(README, locals={"ab": ab, "sb": sb})
    _check(count_brackets(e) == 6, "count_brackets (backend-internal) still counts the denominator")

    # runner: every reported number restated, derivation rows chained, solved re-derived
    scr = ("(-(-ab(1,2)*ab(2,3)*sb(2,3) + ab(1,3)*ab(2,4)*sb(3,4))**3)"
           "/(ab(1,2)**3*ab(1,4)*ab(2,3)*ab(3,4)*sb(1,2)**3)")
    expr = sp.sympify(scr, locals={"ab": ab, "sb": sb})
    r = run_all(["model_free_gbfs"], expr=expr, n_point=4, capture_derivation=True,
                gbfs_max_apps=5000, gbfs_max_seconds=30)[0]
    _check(not r.extra.get("error"), f"search ran ({r.extra.get('error')})")
    start = display_bracket_count(str(expr))
    _check(r.bracket_count_start == start, f"start restated ({r.bracket_count_start} == {start})")
    _check(r.bracket_count_min == display_bracket_count(r.min_expr_str or r.final_expr_str),
           "min restated from min_expr")
    _check(r.extra["backend_bracket_count_start"] == count_brackets(expr),
           f"backend's own start count kept ({r.extra['backend_bracket_count_start']})")
    rows = r.derivation or []
    _check(len(rows) > 0 and r.bracket_count_min < start, "a simplification with a derivation")
    prev = start
    for row in rows:
        _check(row["brackets_before"] == prev and row["brackets_after"] == display_bracket_count(row["expr_after"]),
               "derivation row restated and chained")
        prev = row["brackets_after"]
    _check(r.extra["raw_bracket_count_min"] == rows[-1]["brackets_after"],
           "raw_bracket_count_min equals the last derivation row, which the page matches on")
    _check(r.solved == (r.bracket_count_min < r.bracket_count_start), "solved follows the display count")

    # An uncountable final form, -1, leaves solved alone.
    fake = AlgorithmResult(algorithm="model_free_gbfs", final_expr_str="ab(1,2", bracket_count_start=9,
                           bracket_count_final=1, bracket_count_min=1, n_steps=1, solved=True,
                           timed_out=False, duration_s=0.0)
    _report_counts(fake, README, has_target=False, max_terms=0)
    _check(fake.bracket_count_final == -1 and fake.solved is True, "uncountable final: -1, solved untouched")
    # SFT keeps distinct raw minimal and final forms: restate each from its own expression
    fake = AlgorithmResult(algorithm="sft_greedy", final_expr_str=README, bracket_count_start=50,
                           bracket_count_final=6, bracket_count_min=5, n_steps=2, solved=True,
                           timed_out=False, duration_s=0.0, min_expr_str="ab(1,2)*ab(3,4)/ab(1,4)",
                           extra={"raw_final_expr": README, "raw_min_expr": "ab(1,2)*ab(3,4)/ab(1,4)",
                                  "raw_bracket_count_final": 6, "raw_bracket_count_min": 5})
    _report_counts(fake, "ab(1,2)**3*sb(1,2)/(ab(2,3)*ab(3,4)*ab(4,1)*sb(1,2))", has_target=False, max_terms=0)
    _check((fake.extra["raw_bracket_count_final"], fake.extra["raw_bracket_count_min"]) == (3, 2),
           "SFT: raw final and raw min restated separately (3, 2)")
    _check(fake.bracket_count_min == 2 and fake.solved is True, "SFT: min 2 < start, solved")
    # a backend with a further condition can only lose `solved`
    fake = AlgorithmResult(algorithm="cds_oneshot", final_expr_str=README, bracket_count_start=99,
                           bracket_count_final=6, bracket_count_min=6, n_steps=1, solved=False,
                           timed_out=False, duration_s=0.0)
    _report_counts(fake, README, has_target=False, max_terms=0)
    _check(fake.solved is False and fake.bracket_count_final == 3, "CDS: recount never grants solved")
    print("ALL DISPLAY_COUNT TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
