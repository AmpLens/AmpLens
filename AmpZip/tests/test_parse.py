"""The expression parser: admits exactly the arithmetic the gate accepted through sympify,
builds the same sympy objects, and executes nothing.

Run:  python tests/test_parse.py   (or via tests/run_tests.py)
"""
from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

import sympy as sp  # noqa: E402

import ampzip.simplifier._paths  # noqa: E402,F401
from ampzip.simplifier.parse import parse_expr, ParseError  # noqa: E402
from ampzip.simplifier.validate import validate, _get_bracket_classes  # noqa: E402
from ampzip.simplifier.atomize import atomize  # noqa: E402

ab, sb = _get_bracket_classes()
LOCAL = {"ab": ab, "sb": sb, "I": sp.I, "opaque_den": sp.Function("opaque_den")}

# Strings that must build the same object the gate used to get from sympify. sympify is the
# reference oracle here and nowhere else.
PARITY = [
    "ab(1,2)**3/(ab(2,3)*ab(3,4)*ab(4,1))",
    "ab(1,3)*sb(2,4)/(ab(1,2)*sb(1,2))",
    "-ab(1, 3)**2*sb(1, 3)*sb(2, 4)/(ab(1, 2)**2*sb(1, 2)**2)",
    "2/3*ab(1,2) - ab(2,1)*sb(1,2)/7 + I*ab(1,2)",
    "(ab(1,2)*sb(1,2) + ab(1,4)*sb(1,4))*ab(1,3)*sb(2,4)/(ab(1,2)**2*sb(1,2)**2)",
    "ab(1,2)**-2*sb(3,4)",
    "+ab(1,2) - -sb(3,4)",
    "ab(1,2)*opaque_den(ab(1,2)*sb(2,3) + ab(1,4)*sb(4,3))",
    "ab(1,2)*ab(1,2)*ab(1,2)",
    "0*ab(1,2) + 1",
    "ab(1,2)\n + sb(3,4)",
    "1.5*ab(1,2)",                       # float: parsed (lenient), the gate rejects it by tag
    "sqrt(2)*ab(1,2)",                   # algebraic root: same object as sympify builds
    "pi*ab(1,2)",
    "ap(1,2)*sp(1,2)",                   # massive atoms: undefined functions, the gate's job
    "x*ab(1,2)",                         # stray symbol
]

# Nothing in this list may execute, and each must be refused outright.
ATTACKS = [
    "__import__('os').system('id')",
    "ab(1,2).__class__",
    "(lambda: 1)()",
    "[x for x in ()]",
    "ab(1,2) if 1 else 2",
    "open('/etc/passwd')",
    "exec('1')",
    "ab(1,2)[0]",
    "ab(1,2)(3)",
    "9**9**9**9",
    "ab(1,2); import os",
    "f'{1}'",
    "ab(1,2) == sb(1,2)",
]


def test_parity():
    for s in PARITY:
        got = parse_expr(s, LOCAL, lenient=True)
        ref = sp.sympify(s.replace("\n", " "), locals=LOCAL, evaluate=True)
        assert str(got) == str(ref), (s, str(got), str(ref))
        assert got == ref, s


def test_attacks_refused_by_parser():
    for s in ATTACKS:
        try:
            parse_expr(s, LOCAL, lenient=True)
        except ParseError:
            continue
        raise AssertionError(f"admitted: {s!r}")


def test_attacks_refused_by_gate_without_prescreen():
    for s in ATTACKS:
        r = validate(s)
        assert not r.ok and r.issues and r.issues[0].tag == "LIMIT_PARSE", (s, r.issues)


def test_gate_tags_unchanged():
    expect = {
        "ab(1,2)**3/(ab(2,3)*ab(3,4)*ab(4,1))": None,
        "ap(1,2)*sp(1,2)": "LIMIT_ATOM_REJECTED",
        "sqrt(2)*ab(1,2)": "LIMIT_COEFF_CLASS",
        "1.5*ab(1,2)": "LIMIT_COEFF_CLASS",
        "pi*ab(1,2)": "LIMIT_COEFF_CLASS",
        "": "LIMIT_PARSE",
        "ab(1,2) +": "LIMIT_PARSE",
    }
    for s, tag in expect.items():
        r = validate(s)
        if tag is None:
            assert r.ok, (s, r.issues)
        else:
            assert not r.ok and any(i.tag == tag for i in r.issues), (s, tag, r.issues)


def test_strict_policy_for_atomizer():
    assert "chain" not in atomize("chain('as',1,[2,3],4)*ab(1,2)/s(1,2)")
    for s in ("sqrt(2)*ab(1,2)", "x*ab(1,2)", "ab(1,2).real"):
        try:
            atomize(s)
        except Exception:
            continue
        raise AssertionError(f"atomize admitted {s!r}")


def main() -> int:
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"[ok  ] {name}")
            except AssertionError as e:
                fails += 1
                print(f"[FAIL] {name}: {e}")
    print("ALL PARSE TESTS PASSED" if not fails else f"{fails} parse test(s) FAILED")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
