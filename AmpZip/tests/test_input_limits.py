"""Input rules and limits shared by the CLI and the web service.

Each limit is checked at its bound and one past it, the fixed rules with their hints, and
the CLI and the web service must refuse the same inputs for the same reasons. Randomised
properties are in test_input_property.py.

Run:  python tests/test_input_limits.py   (or via tests/run_tests.py)
"""
from __future__ import annotations

import contextlib
import io
import os
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from ampzip.simplifier.input_limits import (  # noqa: E402
    DEFAULT_INPUT_LIMITS, LIMIT_CONTROLS, LIMIT_HELP, PYTHON_INT_STR_DIGITS, InputLimitError,
    InputLimits, check_input, limits_from_mapping, measure,
)


def _issues(raw, limits=DEFAULT_INPUT_LIMITS):
    try:
        check_input(raw, limits)
    except InputLimitError as e:
        return e.issues
    return []


def _limits_hit(raw, limits=DEFAULT_INPUT_LIMITS):
    return {i.limit for i in _issues(raw, limits)}


# Inputs whose measured quantity is exactly v, as test_builders_are_exact checks.
BUILDERS = {
    "max_input_chars": lambda v: "ab(1,2)" + " " * (v - 7),
    "max_int_digits": lambda v: "1" * v + "*ab(1,2)",
    "max_exponent": lambda v: f"ab(1,2)**{v}",
    "max_power_nesting": lambda v: _nested_power(v),
    # The call's own parentheses are the innermost level.
    "max_nesting_depth": lambda v: "(" * (v - 1) + "ab(1,2)" + ")" * (v - 1) if v else "1",
}
MEASURE_KEY = {"max_input_chars": "input_chars", "max_int_digits": "int_digits",
               "max_exponent": "exponent", "max_power_nesting": "power_nesting",
               "max_nesting_depth": "nesting_depth", "max_number_digits": "number_digits"}


def _nested_power(v):
    x = "ab(1,2)**1"
    for _ in range(v):
        x = f"(sb(1,2)*{x})**1"
    return x


# Valid inputs with separate powers, once refused as exponent towers.
INDEPENDENT_POWERS = [
    "ab(1,2)**2+ab(3,4)**2",
    "ab(1,2)**2-sb(1,2)**2",
    "ab(1,2)**2/ab(3,4)**3",
    "ab(1,2)**2*ab(3,4)**2",
    "ab(1,2)**4/(ab(1,2)*ab(2,3)*ab(3,4)*ab(4,1))",
    "ab(1,2)**3/(ab(2,3)*ab(3,4)*ab(4,1)) + sb(1,2)**3/(sb(2,3)*sb(3,4)*sb(4,1))",
    "(ab(1,2)+ab(3,4))**2",
    "(ab(1,2)*sb(2,3))**2/(ab(3,4)+ab(1,2))**3",
    "ab(1,2)**-2 + ab(3,4)**(-2)",
    "s(1,2)**2/s(3,4)**2",
    "chain('as',1,[2,3],4)**2*ab(1,2)**2",
    "2**3*ab(1,2) + 3**2*sb(1,2)",
    "I**2*ab(1,2)**4",
]

# Nested powers, refused at the default max_power_nesting=0.
TOWERS = [
    "9**9**9", "9**9**9**9", "9**(9**9)", "(9**9)**9", "((9)**9)**9", "2*(9**9)**9",
    "(-9**9)**9", "ab(1,2)**2**3", "ab(1,2)**-9**9", "ab(1,2)**+9**9", "ab(1,2)**(2+3**4)",
    "ab(1,2)**(1)**9", "(ab(1,2)*9**99)**99", "(I*9**99)**99", "(ab(1,2)**2+ab(3,4))**2",
]

# The extremes of the paper corpora, with exponent 18, two-digit integers, parentheses two
# deep and 20966 characters.
CORPUS_EXTREMES = [
    "(ab(1, 2)*ab(1, 3)*ab(1, 4)**2*ab(2, 3)*sb(1, 2)*sb(1, 3)**3*sb(1, 4)**2*sb(2, 4)**3 + "
    "ab(1, 4)*ab(2, 3)*ab(2, 4)**3*sb(1, 4)**3*sb(2, 3)**2*sb(2, 4)**4)/(ab(1, 3)**3*ab(2, 3)**5)",
    "-12*ab(1, 2)**18*sb(3, 4)**2/(ab(1, 3)**9*ab(2, 4)**9*sb(1, 2)**2)",
    "(ab(1,2)**3*sb(1,3)**2)/(ab(2,3)*ab(2,4)*ab(2,5)*ab(4,5)*sb(1,2)*sb(2,3)) " * 1 + "+ ab(1,2)" * 2400,
]


def test_builders_are_exact():
    for name, build in BUILDERS.items():
        for v in (0, 1, 2, 7, 40) if name != "max_input_chars" else (7, 8, 100):
            if name == "max_int_digits" and v == 0:
                continue
            got = measure(build(v))[MEASURE_KEY[name]]
            assert got == v, (name, v, got)


def test_each_limit_at_and_past_its_bound():
    for name, build in BUILDERS.items():
        for bound in (7, 12, 30):
            limits = limits_from_mapping({name: bound})
            assert name not in _limits_hit(build(bound), limits), (name, bound)
            hit = _issues(build(bound + 1), limits)
            assert any(i.limit == name for i in hit), (name, bound, hit)
            flag, env = LIMIT_CONTROLS[name]
            assert any(flag in i.message and env in i.message for i in hit if i.limit == name)


def test_number_digits_bound():
    # 9**k has floor(k*log10 9)+1 digits and the estimate matches it here.
    for k in (10, 50, 100):
        assert measure(f"9**{k}*ab(1,2)")["number_digits"] == len(str(9 ** k)), k
    limits = InputLimits(max_number_digits=100)
    assert "max_number_digits" not in _limits_hit("9**100*ab(1,2)", limits)   # 96 digits
    assert "max_number_digits" in _limits_hit("9**105*ab(1,2)", limits)       # 101 digits
    assert "max_number_digits" in _limits_hit("*".join(["999"] * 400) + "*ab(1,2)")


def test_independent_powers_pass():
    for s in INDEPENDENT_POWERS:
        assert not _issues(s), (s, _issues(s))


def test_towers_refused_by_default_and_admitted_when_raised():
    for s in TOWERS:
        assert "max_power_nesting" in _limits_hit(s), s
    relaxed = InputLimits(max_power_nesting=1)
    assert not _issues("(ab(1,2)**2*sb(3,4))**3", relaxed)
    assert not _issues("(ab(1,2)**2+ab(3,4))**2", relaxed)
    # The size bounds still apply with a raised nesting bound.
    assert {"max_exponent", "max_number_digits"} <= _limits_hit("(9**99)**99", relaxed)


def test_rules_that_cannot_be_relaxed():
    loose = InputLimits(max_input_chars=10 ** 6, max_int_digits=10 ** 6, max_exponent=10 ** 9,
                        max_power_nesting=10 ** 6, max_nesting_depth=10 ** 6,
                        max_number_digits=PYTHON_INT_STR_DIGITS)
    cases = {
        "": "empty",
        "   ": "empty",
        "ab(1,2)^2": "'^'",
        "ab(1,2)−ab(3,4)": "'−' -> '-'",
        "（ab(1,2)）": "'（' -> '('",
        "ab(1,2)×sb(1,2)": "'×' -> '*'",
        "x*ab(1,2)": "illegal character",
        "ab(1,2).real": "illegal character",
        "__import__('os')": "illegal character",
        "1.5*ab(1,2)": "illegal character",
    }
    for raw, needle in cases.items():
        issues = _issues(raw, loose)
        assert issues and all(i.limit is None for i in issues), (raw, issues)
        assert any(needle in i.message for i in issues), (raw, issues)


def test_corpus_extremes_pass_default_limits():
    for s in CORPUS_EXTREMES:
        assert not _issues(s), (s[:60], _issues(s))


def test_invalid_limits_are_refused():
    for bad in ({"max_exponent": -1}, {"max_number_digits": PYTHON_INT_STR_DIGITS + 1}):
        try:
            limits_from_mapping(bad)
        except ValueError:
            continue
        raise AssertionError(f"accepted {bad}")
    try:
        limits_from_mapping({"max_widgets": 3})
    except ValueError:
        pass
    else:
        raise AssertionError("accepted an unknown limit")
    assert set(LIMIT_HELP) == set(LIMIT_CONTROLS) == set(MEASURE_KEY)


def test_docs_list_every_limit():
    with open(os.path.join(os.path.dirname(_HERE), "docs", "LIMITS.md"), encoding="utf-8") as f:
        doc = f.read()
    for name, (flag, env) in LIMIT_CONTROLS.items():
        row = f"| `{flag}` / `{env}` | {getattr(DEFAULT_INPUT_LIMITS, name)} |"
        assert row in doc, f"docs/LIMITS.md lacks the row for {name}: {row}"


def test_long_and_deep_inputs_are_measured_without_recursion():
    for raw in ("+".join(["ab(1,2)"] * 4000), "-".join(["1"] * 10000), "(" * 150 + "1" + ")" * 150):
        measure(raw)


def _cli(argv):
    from ampzip.simplifier import cli
    out = io.StringIO()
    with tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(out), \
            contextlib.redirect_stderr(io.StringIO()):
        code = cli.main(argv + ["--log-dir", d])
    return code, out.getvalue()


def _web(**body):
    from unittest.mock import patch
    from fastapi.testclient import TestClient
    import ampzip.webservice.app as web
    with patch.object(web.LIMITER, "allow", return_value=True), \
            patch.object(web.audit, "audit_reject"), \
            patch.dict(web.app.dependency_overrides, {web.require_email: lambda: "t@example.invalid"}):
        return TestClient(web.app).post("/api/jobs", json=body)


def test_cli_and_web_refuse_the_same_inputs_for_the_same_reasons():
    try:
        import fastapi.testclient  # noqa: F401
    except ImportError:
        print("       (web part skipped: web test dependencies unavailable)")
        web = False
    else:
        web = True
    for raw in ("(9999**9999)**9999*ab(1,2)", "ab(1,2)**99999999", "ab(1,2)^2", TOWERS[3]):
        reasons = [i.message for i in _issues(raw)]
        code, out = _cli(["--expr", raw, "--n-point", "4"])
        assert code == 2 and all(r in out for r in reasons), (raw, code, out)
        code, out = _cli(["--expr", "ab(1,2)", "--target", raw, "--n-point", "4"])
        assert code == 2 and all(r in out for r in reasons), (raw, "target", code, out)
        if web:
            r = _web(expr=raw, algorithms=["model_free_gbfs"], n_point=4)
            assert r.status_code == 400 and r.json()["detail"]["unsafe_input"] == reasons, (raw, r.text)


def test_cli_flags_raise_a_limit():
    raw = "(ab(1,2)**2*sb(3,4))**3"
    code, out = _cli(["--expr", raw, "--n-point", "4", "--algorithms", "model_free_gbfs",
                      "--gbfs-max-seconds", "5"])
    assert code == 2 and "max_power_nesting" in out, out
    code, out = _cli(["--expr", raw, "--n-point", "4", "--algorithms", "model_free_gbfs",
                      "--gbfs-max-seconds", "5", "--max-power-nesting", "1"])
    assert code == 0 and "REJECTED" not in out, out


def test_web_config_reads_every_limit_from_its_variable():
    from unittest.mock import patch
    from ampzip.webservice.config import load_config
    env = {env: str(getattr(DEFAULT_INPUT_LIMITS, name) + 3) for name, (_f, env) in LIMIT_CONTROLS.items()
           if name != "max_number_digits"}
    with patch.dict(os.environ, env):
        cfg = load_config().input_limits
    for name, (_f, env_name) in LIMIT_CONTROLS.items():
        if env_name in env:
            assert getattr(cfg, name) == int(env[env_name]), name


def main() -> int:
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"[ok  ] {name}")
            except AssertionError as e:
                fails += 1
                print(f"[FAIL] {name}: {str(e)[:500]}")
    print("ALL INPUT-LIMIT TESTS PASSED" if not fails else f"{fails} input-limit test(s) FAILED")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
