"""The sympy post-processing is bounded and reports a timeout instead of hanging."""
import os, sys, time
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
from ampzip.simplifier.algorithms._post import post_process, count_brackets_bounded

SLOW = "(ab(1,2)+ab(1,3)+ab(1,4)+ab(2,3))**40/(ab(1,2)+ab(3,4))**39"


def test_bounded():
    t = time.time()
    r = post_process(SLOW, "simplify", timeout_s=0.5)
    assert r.timed_out and r.bounded and r.expr_str == SLOW and r.bracket_count == -1
    assert time.time() - t < 10
    r = post_process("(2*ab(1,3)*ab(2,4) - ab(1,4)*ab(2,3))/ab(1,4)", "cancel", timeout_s=30)
    assert not r.timed_out and r.bounded and r.bracket_count == 5
    assert count_brackets_bounded("ab(1,2)*ab(3,4)/ab(1,4)", timeout_s=30) == 3
    r = post_process("ab(1,2)", "cancel", timeout_s=0)
    assert not r.bounded and r.bracket_count == 1


def main():
    test_bounded(); print("  ok test_bounded")
    return 0


if __name__ == "__main__":
    sys.exit(main())
