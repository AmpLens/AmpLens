"""Randomised tests of check_input, atomize and validate.

Seeded generators. Properties:

1. valid    - random amplitude-like expressions within the default limits pass all three;
2. monotone - raising a limit keeps every accepted input accepted, and an input is refused
              for exactly the limits its measures exceed;
3. digits   - the number-size estimate is at least the true size of a numeric expression;
4. mutated  - mutated inputs are accepted or refused by one of the three within a time
              bound.

AMPZIP_PROPERTY_SEED fixes the seed (default 20261101) and AMPZIP_PROPERTY_SCALE multiplies
the case counts (default 1). A failure prints the seed, the case number and the input.

Run:  python tests/test_input_property.py   (or via tests/run_tests.py)
"""
from __future__ import annotations

import math
import multiprocessing as mp
import os
import random
import sys
import time
from dataclasses import asdict, replace

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, _HERE)

from ampzip.simplifier.input_limits import (  # noqa: E402
    DEFAULT_INPUT_LIMITS, LIMIT_CONTROLS, PYTHON_INT_STR_DIGITS, InputLimitError, InputLimits,
    check_input, measure,
)

SEED = int(os.environ.get("AMPZIP_PROPERTY_SEED", "20261101"))
SCALE = float(os.environ.get("AMPZIP_PROPERTY_SCALE", "1"))
CASE_TIMEOUT_S = 30.0


def _n(base: int) -> int:
    return max(1, int(base * SCALE))


# --- generators ---------------------------------------------------------------------------

def gen_bracket(rng: random.Random, n: int) -> str:
    kind = rng.choice(["ab", "sb", "ab", "sb", "s"])
    if kind == "s":
        legs = sorted(rng.sample(range(1, n + 1), 3 if n >= 5 and rng.random() < 0.3 else 2))
        return "s(" + ",".join(map(str, legs)) + ")"
    i, j = rng.sample(range(1, n + 1), 2)
    return f"{kind}({i},{j})"


def gen_factor(rng: random.Random, n: int) -> str:
    b = gen_bracket(rng, n)
    k = rng.choice([1, 1, 1, 2, 3, 4, 5, 8, 18])
    return b if k == 1 else f"{b}**{k}"


def gen_monomial(rng: random.Random, n: int) -> str:
    num = "*".join(gen_factor(rng, n) for _ in range(rng.randint(1, 5)))
    den = "*".join(gen_factor(rng, n) for _ in range(rng.randint(0, 4)))
    coeff = rng.choice(["", "", "", "2*", "3*", "12*", "I*", "1/2*", "999*", "123456*"])
    return coeff + num + (f"/({den})" if den else "")


def gen_valid(rng: random.Random):
    """An amplitude-like expression in which every leg appears, and its number of legs."""
    n = rng.choice([4, 5, 6])
    terms = [gen_monomial(rng, n) for _ in range(rng.randint(1, 6))]
    expr = terms[0] + "".join(rng.choice([" + ", " - "]) + t for t in terms[1:])
    if rng.random() < 0.25:
        expr = f"({gen_bracket(rng, n)} + {gen_bracket(rng, n)})**{rng.randint(1, 6)}*({expr})"
    if rng.random() < 0.2:
        a = gen_bracket(rng, n)
        b = gen_bracket(rng, n)
        while b == a:                        # a - a would be a zero denominator
            b = gen_bracket(rng, n)
        expr = f"({expr})/({a} - {b})"
    expr += f" + ab(1,{n})*sb(1,{n})"
    return expr, n


def gen_numeric(rng: random.Random, depth: int = 0) -> str:
    """A random numeric expression with powers, for the size estimate."""
    if depth > 3 or rng.random() < 0.3:
        return str(rng.randint(1, 999))
    a, b = gen_numeric(rng, depth + 1), gen_numeric(rng, depth + 1)
    op = rng.choice(["+", "-", "*", "/", "**"])
    if op == "**":
        return f"({a})**{rng.randint(0, 25)}"
    return f"({a}){op}({b})"


_POOL = list("()[]+-*/,'^.xyz_0123456789 abscinhI\t") + ["−", "（", "⟨1 2⟩", "λ", "\n", "**", "ab(", "sb("]


def mutate(rng: random.Random, s: str) -> str:
    """One to three random edits."""
    def big():
        return "9" * rng.choice([3, 6, 7, 12, 40, 5000])

    ops = [
        lambda s: (lambda p: s[:p] + rng.choice(_POOL) + s[p:])(rng.randint(0, len(s))),
        lambda s: (lambda p, q: s[:p] + s[q:])(*sorted(rng.sample(range(len(s) + 2), 2))),
        lambda s: s[: rng.randint(0, len(s))],
        lambda s: f"{s}**{rng.choice([2, 9, 99, 101, 999, 99999])}",
        lambda s: f"({s})**{rng.randint(2, 9)}**{rng.randint(2, 99)}",
        lambda s: f"(({s})**{rng.randint(2, 99)})**{rng.randint(2, 99)}",
        lambda s: "9**" * rng.randint(1, 6) + "9*" + s,
        lambda s: f"({big()}**{rng.randint(2, 999)})*{s}",
        lambda s: "(" * (k := rng.choice([10, 49, 50, 51, 199, 250])) + s + ")" * k,
        lambda s: s * rng.randint(2, 5),
        lambda s: " + ".join([s] * rng.choice([2, 50, 500, 3000])),
        lambda s: s.replace("2", big(), 1),
        lambda s: s.replace("*", "^"),
        lambda s: s.replace("(1,", "(1,1+", 1),
        lambda s: s.replace(",", ",-", 1),
        lambda s: s.replace("ab(", "ab(1,", 1),
        lambda s: s.replace("ab(1,", "ab(1,1", 1),
        lambda s: s + rng.choice(["/0", "*0", "+0**-1", "/(ab(1,2)-ab(1,2))", "**0", "**-0"]),
        lambda s: s + f"*ab(1,{rng.choice([0, 7, 99, 999999])})",
        lambda s: s + rng.choice(["*s(1)", "*s()", "*s(1,1)", "*chain('xx',1,[2],3)",
                                  "*chain('as',1,[],2)", "*chain('as')", "*ab(1,2,3)", "*ab()"]),
        lambda s: s + rng.choice(["*ab(1,2)**(1/2)", "*ab(1,2)**ab(3,4)", "*I**I", "*2**(-1/3)"]),
        lambda s: s + rng.choice(["*(ab(1,2)", "*ab(1,2))", "*[ab(1,2)]", "*'ab(1,2)'", "*,"]),
        lambda s: "-" * rng.choice([2, 500, 5000]) + s,
        lambda s: "",
    ]
    for _ in range(rng.randint(1, 3)):
        s = rng.choice(ops)(s)
    return s


# --- check_input, atomize and validate in a child process ----------------------------------

def front_end(raw: str, n_point: int, limits: dict):
    """Name the step that refused `raw`, or "ok"."""
    from ampzip.simplifier.atomize import AtomizeError, atomize
    from ampzip.simplifier.validate import validate
    try:
        check_input(raw, InputLimits(**limits))
    except InputLimitError:
        return "check_input"
    try:
        atomic = atomize(raw, n_hint=n_point)
    except AtomizeError:
        return "atomize"
    v = validate(atomic, override_n_point=n_point, enforce_size_limits=False)
    return "ok" if v.ok else "validate:" + ",".join(sorted({i.tag for i in v.issues}))


class _Child:
    """One spawned worker, replaced when a case exceeds the timeout."""

    def __init__(self):
        self._ctx = mp.get_context("spawn")
        self._pool = None

    def run(self, *args):
        if self._pool is None:
            self._pool = self._ctx.Pool(1)
            self._pool.apply(front_end, ("ab(1,2)", 4, asdict(DEFAULT_INPUT_LIMITS)))  # warm up
        job = self._pool.apply_async(front_end, args)
        try:
            return job.get(CASE_TIMEOUT_S)
        except mp.TimeoutError:
            self._pool.terminate()
            self._pool = None
            raise

    def close(self):
        if self._pool is not None:
            self._pool.terminate()


def _fail(prop, case, raw, detail):
    shown = raw if len(raw) <= 300 else raw[:150] + f" ...[{len(raw)} chars]... " + raw[-100:]
    return AssertionError(f"{prop}: seed={SEED} case={case}: {detail}\n  input: {shown!r}")


# --- properties ----------------------------------------------------------------------------

def test_valid_inputs_pass_the_front_end():
    rng = random.Random(SEED)
    child = _Child()
    try:
        for case in range(_n(150)):
            raw, n = gen_valid(rng)
            try:
                got = child.run(raw, n, asdict(DEFAULT_INPUT_LIMITS))
            except Exception as e:  # noqa: BLE001
                raise _fail("valid", case, raw, f"{type(e).__name__}: {e}")
            if got != "ok":
                raise _fail("valid", case, raw, f"refused at {got}")
    finally:
        child.close()


def _random_limits(rng: random.Random) -> InputLimits:
    d = {name: rng.randint(0, 2 * getattr(DEFAULT_INPUT_LIMITS, name) + 2) for name in LIMIT_CONTROLS}
    d["max_number_digits"] = min(d["max_number_digits"], PYTHON_INT_STR_DIGITS)
    return InputLimits(**d)


_MEASURED = {"max_input_chars": "input_chars", "max_int_digits": "int_digits",
             "max_exponent": "exponent", "max_power_nesting": "power_nesting",
             "max_nesting_depth": "nesting_depth", "max_number_digits": "number_digits"}


def test_limits_are_monotone_and_match_the_measures():
    rng = random.Random(SEED + 1)
    for case in range(_n(600)):
        raw, _n_point = gen_valid(rng)
        if rng.random() < 0.6:
            raw = mutate(rng, raw)
        limits = _random_limits(rng)
        try:
            check_input(raw, limits)
            refused = set()
        except InputLimitError as e:
            refused = {i.limit for i in e.issues}
        rules = None in refused
        refused.discard(None)
        m = measure(raw)
        if not rules and raw.strip() and m["input_chars"] <= limits.max_input_chars:
            expect = {name for name, key in _MEASURED.items()
                      if m[key] is not None and m[key] > getattr(limits, name)}
            if refused != expect:
                raise _fail("monotone", case, raw, f"refused for {refused}, measures exceed {expect}")
        if not rules and not refused:
            for name in LIMIT_CONTROLS:
                cap = PYTHON_INT_STR_DIGITS if name == "max_number_digits" else None
                bigger = getattr(limits, name) + rng.randint(1, 50)
                raised = replace(limits, **{name: min(bigger, cap) if cap else bigger})
                try:
                    check_input(raw, raised)
                except InputLimitError as e:
                    raise _fail("monotone", case, raw, f"raising {name} refused it: {e}")


def test_number_estimate_is_an_upper_bound():
    import sympy as sp
    from ampzip.simplifier.parse import parse_expr
    rng = random.Random(SEED + 2)
    checked = 0
    for case in range(_n(400)):
        raw = gen_numeric(rng)
        est = measure(raw)["number_digits"]
        if est > 3000:                       # keep each evaluation cheap
            continue
        try:
            value = parse_expr(raw, {}, lenient=True)
        except Exception:  # noqa: BLE001  (division by zero and the like)
            continue
        if not value.is_Rational:
            continue
        p, q = sp.fraction(value)
        true = max(int(p), int(q), key=lambda x: abs(x))
        digits = 1 if true == 0 else int(abs(int(true)).bit_length() * math.log10(2)) + 1
        if digits > est + 1:                 # bit_length gives digits within one
            raise _fail("digits", case, raw, f"estimate {est} < true {digits}")
        checked += 1
    assert checked > _n(400) // 4, f"only {checked} numeric cases evaluated"


def test_mutated_inputs_end_cleanly():
    rng = random.Random(SEED + 3)
    child = _Child()
    outcomes = {}
    slowest = (0.0, "")
    try:
        for case in range(_n(300)):
            base, n = gen_valid(rng)
            raw = mutate(rng, base)
            t = time.time()
            try:
                got = child.run(raw, n, asdict(DEFAULT_INPUT_LIMITS))
            except mp.TimeoutError:
                raise _fail("mutated", case, raw, f"no outcome within {CASE_TIMEOUT_S:.0f} s")
            except Exception as e:  # noqa: BLE001
                raise _fail("mutated", case, raw, f"{type(e).__name__}: {e}")
            dt = time.time() - t
            if dt > slowest[0]:
                slowest = (dt, raw[:60])
            stage = got.split(":")[0]
            outcomes[stage] = outcomes.get(stage, 0) + 1
    finally:
        child.close()
    print(f"       outcomes: {dict(sorted(outcomes.items()))}; slowest {slowest[0]:.2f} s")


def main() -> int:
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            t = time.time()
            try:
                fn()
                print(f"[ok  ] {name} ({time.time() - t:.1f} s)")
            except AssertionError as e:
                fails += 1
                print(f"[FAIL] {name}: {e}")
    print("ALL INPUT-PROPERTY TESTS PASSED" if not fails else f"{fails} input-property test(s) FAILED")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
