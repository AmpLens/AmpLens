"""Inputs the engine cannot represent are tagged, pass validation and are routed away from the search."""
import os, sys
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
from ampzip.simplifier.validate import validate, ENGINE_LIMIT_TAGS
from ampzip.simplifier.atomize import atomize


def test_tags():
    v = validate("ab(1,2)*sb(3,4)/2 + I*ab(1,3)*sb(2,4)", enforce_size_limits=False)
    tags = [i.tag for i in v.issues]
    assert v.ok and tags.count("LIMIT_ENGINE_COEFF") == 2, (v.ok, tags)
    v = validate("ab(1,2)*sb(3,4) + 2*ab(1,3)*sb(2,4)", enforce_size_limits=False)
    assert v.ok and not any(i.tag in ENGINE_LIMIT_TAGS for i in v.issues)


def test_long_sum_parses():
    terms = " + ".join(f"ab(1,2)*sb(3,4)**{(k % 5) + 1}" for k in range(3000))
    v = validate(atomize(terms, n_hint=4), enforce_size_limits=False)
    assert v.ok, [i.tag for i in v.issues]


def main():
    test_tags(); print("  ok test_tags")
    test_long_sum_parses(); print("  ok test_long_sum_parses")
    return 0


if __name__ == "__main__":
    sys.exit(main())
