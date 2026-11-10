"""The expression parser admits the arithmetic the input check accepted through sympify,
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
from ampzip.simplifier.input_limits import measure  # noqa: E402
from ampzip.simplifier.validate import validate, _get_bracket_classes  # noqa: E402
from ampzip.simplifier.atomize import atomize  # noqa: E402

ab, sb = _get_bracket_classes()
LOCAL = {"ab": ab, "sb": sb, "I": sp.I, "opaque_den": sp.Function("opaque_den")}

# Strings that build the same object the input check used to get from sympify, the
# reference here.
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

# Each string in this list is refused without executing anything.
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


def test_strings_and_lists_are_not_expressions():
    from ampzip.atoms import ab, sb
    names = {"ab": ab, "sb": sb, "I": sp.I}
    for raw in ("'abs'", "'bin(1)'", "[1,2]", "'as'", "[ab(1,2)]"):
        try:
            parse_expr(raw, names, lenient=True, allow_str=True)
        except ParseError as e:
            assert e.kind == "eval", (raw, e.kind)
        else:
            raise AssertionError(f"accepted {raw}")


def test_strict_policy_for_atomizer():
    assert "chain" not in atomize("chain('as',1,[2,3],4)*ab(1,2)/s(1,2)")
    for s in ("sqrt(2)*ab(1,2)", "x*ab(1,2)", "ab(1,2).real"):
        try:
            atomize(s)
        except Exception:
            continue
        raise AssertionError(f"atomize admitted {s!r}")


def test_long_sums_parse_on_every_platform():
    # CPython refuses to build a syntax tree deeper than its C recursion limit, 4000 levels on
    # Windows, so a sum this long parses only when it is split before `ast.parse`.
    ab_, sb_ = ab, sb
    terms = [f"ab(1,2)*sb(3,4)**{k % 5 + 1}" for k in range(20_000)]
    signs = [1] + [-1 if k % 3 else 1 for k in range(1, len(terms))]
    text = terms[0] + "".join((" - " if s < 0 else " + ") + t for s, t in zip(signs[1:], terms[1:]))
    text += " + ab(1,2)**-2*(sb(1,3) + sb(2,4))"
    got = parse_expr(text, LOCAL)
    coeff = {k: sum(s for s, t in zip(signs, terms) if t.endswith(f"**{k}")) for k in range(1, 6)}
    expected = sum(c * ab_(1, 2) * sb_(3, 4) ** k for k, c in coeff.items()) \
        + ab_(1, 2) ** -2 * (sb_(1, 3) + sb_(2, 4))
    assert got == expected
    m = measure(text)
    assert m["exponent"] == 5 and m["power_nesting"] == 0 and m["number_digits"] is not None, m
    assert "**5" in m["exponent_at"], m["exponent_at"]


def test_split_sums_match_whole_parse():
    # At least 32 summands take the split path; sympify sees the whole text.
    pieces = ["ab(1,2) - -sb(3,4)", "ab(1,2)**-2 - sb(1,2)", "(ab(1,2) - ab(3,4)) - ab(1,3)",
              "- ab(1,4)", "2*ab(1,2)*-sb(3,4)", "ab(1,2)/3 - 1"]
    text = " + ".join(pieces * 6)
    assert text.count("+") + text.count("-") > 32
    got = parse_expr(text, LOCAL, lenient=True)
    ref = sp.sympify(text, locals=LOCAL)
    assert got == ref, (str(got), str(ref))
    long_chain = " + ".join(["chain('as',1,[2,3],4) - ab(1,2)"] * 20)
    assert "chain" not in atomize(long_chain)


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
