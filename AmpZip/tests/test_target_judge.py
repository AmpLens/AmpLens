"""With --target, `solved` is one verdict for every backend: the simplest reported form equals
the target over finite fields and meets the target criterion on the raw numerator counts.

    python tests/test_target_judge.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))   # the package root

import sympy as sp  # noqa: E402

from ampzip.atoms import ab, sb  # noqa: E402
from ampzip.simplifier.result import AlgorithmResult  # noqa: E402
from ampzip.simplifier.runner import _judge_target, run_all  # noqa: E402

TARGET = "ab(1,2)*ab(3,4)"
SCHOUTEN = "ab(1,3)*ab(2,4) - ab(1,4)*ab(2,3)"      # equal to TARGET, with four brackets
WRONG = "ab(1,3)*ab(2,4) + ab(1,4)*ab(2,3)"         # unequal to TARGET


def _check(cond, msg):
    print(("  ok " if cond else "  FAIL ") + msg)
    if not cond:
        raise AssertionError(msg)


def _fake(algo, form, solved=True):
    return AlgorithmResult(algorithm=algo, final_expr_str=form, min_expr_str=form,
                           bracket_count_start=9, bracket_count_final=4, bracket_count_min=4,
                           n_steps=1, solved=solved, timed_out=False, duration_s=0.0)


def main():
    # Simplified but not down to the target, equal yet with more brackets, so not solved.
    r = _fake("sft_greedy", SCHOUTEN)
    _judge_target(r, TARGET, 4, "brackets", 0)
    _check(r.extra["target_equal"] is True and r.solved is False,
           f"equal but larger than the target is not solved (raw {r.extra['final_raw_brackets']} > {r.extra['target_raw_brackets']})")
    # at the target's size -> solved
    r = _fake("cds_oneshot", "ab(1,2)*ab(3,4)")
    _judge_target(r, TARGET, 4, "brackets", 0)
    _check(r.solved is True and r.extra["target_equal"] is True, "the target itself is solved")
    r = _fake("cds_oneshot", "-ab(2,1)*ab(3,4)")
    _judge_target(r, TARGET, 4, "brackets-and-terms", 0)
    _check(r.solved is True, "an equal form of the same size is solved under brackets-and-terms")
    # Unequal, so not solved whatever the size.
    r = _fake("sft_best_of_n", WRONG)
    _judge_target(r, TARGET, 4, "brackets", 0)
    _check(r.solved is False and r.extra["target_equal"] is False, "a non-equal form is not solved")
    # criterion none: equality alone
    r = _fake("sft_greedy", SCHOUTEN)
    _judge_target(r, TARGET, 4, "none", 0)
    _check(r.solved is True, "criterion none: an equal form is solved regardless of size")
    # A form the engine cannot count is not solved, with the reason.
    r = _fake("cds_oneshot", "ab(1,2)*ab(3,4)/2")
    _judge_target(r, TARGET, 4, "brackets", 0)
    _check(r.solved is False and "target_check_note" in r.extra,
           f"uncountable form is not solved ({r.extra.get('target_check_note')})")
    # the search family is left to its own (identical) judgement
    r = _fake("model_free_gbfs", WRONG, solved=True)
    _judge_target(r, TARGET, 4, "brackets", 0)
    _check(r.solved is True and "target_equal" not in r.extra, "search family untouched by the runner judge")

    # end to end through the runner with a target: gbfs judged by its run
    expr = sp.sympify(SCHOUTEN, locals={"ab": ab, "sb": sb})
    tgt = sp.sympify(TARGET, locals={"ab": ab, "sb": sb})
    out = run_all(["model_free_gbfs"], expr=expr, n_point=4, target=tgt,
                  gbfs_max_apps=2000, gbfs_max_seconds=30)[0]
    _check(not out.extra.get("error") and out.solved is True and out.extra.get("target_equal") is True,
           f"gbfs reaches the target ({out.extra.get('error')})")
    print("ALL TARGET_JUDGE TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
