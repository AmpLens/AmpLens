"""Input limits shared by the command line and the web service.

`check_input` reads the raw text and its Python syntax tree before anything evaluates it.
sympy would otherwise compute `(9999**9999)**9999` while parsing. The tree is walked with
an explicit stack.

Every bound is a field of `InputLimits`, set by the `--max-*` flags and the `WEBSVC_MAX_*`
variables. A refused input lists each exceeded bound, the offending text and the flag or
variable that raises it. The character set, `^` and non-ASCII characters are fixed rules.
The parser in `ampzip.simplifier.parse` remains the security boundary.
"""
from __future__ import annotations

import ast
import math
import re
from dataclasses import dataclass, fields
from typing import Dict, List, Optional, Tuple

from ._source import SUM_NAME, parse_source

# Python will not convert a longer integer to text.
PYTHON_INT_STR_DIGITS = 4300


@dataclass(frozen=True)
class InputLimits:
    """Inclusive bounds on one input expression. LIMIT_HELP describes each."""

    max_input_chars: int = 32000
    max_int_digits: int = 6
    max_exponent: int = 100
    max_power_nesting: int = 0
    max_nesting_depth: int = 50
    max_number_digits: int = 1000

    def __post_init__(self):
        for f in fields(self):
            v = getattr(self, f.name)
            if not isinstance(v, int) or isinstance(v, bool) or v < 0:
                raise ValueError(f"{f.name} must be a non-negative integer, got {v!r}")
        if self.max_number_digits > PYTHON_INT_STR_DIGITS:
            raise ValueError(f"max_number_digits={self.max_number_digits} exceeds "
                             f"{PYTHON_INT_STR_DIGITS}, the longest integer Python will print")


DEFAULT_INPUT_LIMITS = InputLimits()

# Help text for each bound, used by the command line and the documentation.
LIMIT_HELP: Dict[str, str] = {
    "max_input_chars": "Length of the expression in characters. The longest paper input, a "
                       "five-point Yang-Mills form, has about 21000. A public web deployment "
                       "uses a lower value, such as the 8000 of the example configuration, "
                       "since parsing runs in the request thread.",
    "max_int_digits": "Digits of any integer written in the expression, whether a coefficient, "
                      "an exponent or a leg label.",
    "max_exponent": "Largest exponent, with nested powers multiplied out, so (x**4)**30 "
                    "counts as 120.",
    "max_power_nesting": "Powers nested inside another power. 0 refuses (x**2)**3 and "
                         "x**2**3, and 1 admits one level.",
    "max_nesting_depth": "Depth of nested parentheses and square brackets.",
    "max_number_digits": "Estimated digits of the largest number the numeric parts produce. "
                         f"999**999 has about 2997. At most {PYTHON_INT_STR_DIGITS}.",
}

# The command-line flag and the web-service variable of each bound.
LIMIT_CONTROLS: Dict[str, Tuple[str, str]] = {
    f.name: ("--" + f.name.replace("_", "-"), "WEBSVC_" + f.name.upper())
    for f in fields(InputLimits)
}

# Whitespace, digits, the letters of ab/sb/s/I/chain, arithmetic, grouping, and the quote
# and square brackets of chain('as', 1, [2, 3], 4). Without '_' and '.', attribute access
# and float literals cannot be written.
_ALLOWED_CHARS = re.compile(r"^[\s0-9absIchin'\[\]+\-*/(),]+$")
_STRING = re.compile(r"'[^']*'")

# Look-alike characters common in pasted mathematics.
_ASCII_HINTS = {
    "−": "-", "–": "-", "—": "-", "×": "*", "⋅": "*", "·": "*",
    "（": "(", "）": ")", "，": ",", "⟨": "ab(", "⟩": ")",
    "’": "'", "‘": "'",
}


@dataclass(frozen=True)
class InputIssue:
    """One reason an input is refused. `limit` names the InputLimits field, or is None for a
    fixed rule."""

    limit: Optional[str]
    message: str


class InputLimitError(ValueError):
    """Raised by `check_input` with every reason in `issues`."""

    def __init__(self, issues: List[InputIssue]):
        self.issues = issues
        super().__init__("; ".join(i.message for i in issues))

    @property
    def reasons(self) -> List[str]:
        return [i.message for i in self.issues]


# Message wording for the measured value of each bound.
_FOUND = {
    "max_input_chars": "input of {} characters",
    "max_int_digits": "integer with {} digits",
    "max_exponent": "effective exponent {}",
    "max_power_nesting": "{} power(s) nested inside another power",
    "max_nesting_depth": "parentheses nested {} deep",
    "max_number_digits": "number of about {} digits",
}


def _over(name: str, found, limits: InputLimits, where: str = "") -> InputIssue:
    flag, env = LIMIT_CONTROLS[name]
    at = f" in {where!r}" if where else ""
    return InputIssue(name, f"{_FOUND[name].format(found)}{at} exceeds {name}="
                            f"{getattr(limits, name)} (raise with {flag} or {env})")


def _snippet(raw: str, node: ast.AST, width: int = 48) -> str:
    seg = ast.get_source_segment(raw, node) or ""
    return seg if len(seg) <= width else seg[: width - 3] + "..."


def _log10_add(a: float, b: float) -> float:
    """log10(10**a + 10**b) without overflow."""
    hi, lo = (a, b) if a >= b else (b, a)
    return hi + math.log10(1.0 + 10.0 ** (lo - hi))


def measure(raw: str) -> Dict[str, object]:
    """The quantities `check_input` bounds. The syntax-tree ones are None when the text does
    not parse."""
    text = _STRING.sub("''", raw)
    digits = max((len(m) for m in re.findall(r"\d+", text)), default=0)
    depth = cur = 0
    for c in text:
        if c in "([":
            cur += 1
            depth = max(depth, cur)
        elif c in ")]":
            cur -= 1
    out: Dict[str, object] = {
        "input_chars": len(raw), "int_digits": digits, "nesting_depth": depth,
        "exponent": None, "power_nesting": None, "number_digits": None,
        "exponent_at": "", "power_nesting_at": "", "number_digits_at": "",
    }
    try:
        tree = parse_source(raw)
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return out
    out.update(_tree_measures(tree, raw.replace("\r", " ").replace("\n", " ")))
    return out


def _tree_measures(tree: ast.Expression, raw: str) -> Dict[str, object]:
    """Post-order walk with an explicit stack. Each node gets (num, den, eff, nest). num and
    den bound log10 of the numerator and denominator of the coefficients the node can
    produce, with sums brought over a common denominator. A call counts as 1 for its parent
    and its arguments are bounded on their own. eff is the largest exponent of a power inside
    the node with nested powers multiplied out, and nest the depth of the deepest chain of
    nested powers."""
    Info = Tuple[float, float, float, int]
    info: Dict[int, Info] = {}
    zero: Info = (0.0, 0.0, 0.0, 0)
    worst = {"exponent": (0.0, None), "power_nesting": (0, None), "number_digits": (0.0, None)}
    cap = 1e300

    def note(key, value, node):
        if value > worst[key][0]:
            worst[key] = (value, node)

    stack: List[Tuple[ast.AST, bool]] = [(tree.body, False)]
    while stack:
        node, done = stack.pop()
        kids = list(ast.iter_child_nodes(node))
        if not done:
            stack.append((node, True))
            stack.extend((k, False) for k in kids)
            continue
        get = lambda k: info.get(id(k), zero)  # noqa: E731
        if isinstance(node, ast.Constant):
            v = node.value
            numeric = isinstance(v, (int, float)) and not isinstance(v, bool)
            res = (math.log10(abs(v)) if numeric and v != 0 else 0.0, 0.0, 0.0, 0)
        elif isinstance(node, ast.BinOp):
            (ln, ld, le, lk), (rn, rd, re_, rk) = get(node.left), get(node.right)
            if isinstance(node.op, ast.Pow):
                lit = node.right.operand if isinstance(node.right, ast.UnaryOp) else node.right
                negative = isinstance(node.right, ast.UnaryOp) and isinstance(node.right.op, ast.USub)
                if isinstance(lit, ast.Constant) and type(lit.value) is int:
                    e = float(abs(lit.value))
                else:
                    e = 10.0 ** min(rn, 300.0)
                    negative = True
                num, den = min(e * ln, cap), min(e * ld, cap)
                if negative:
                    num = den = max(num, den)
                eff, nest = e * max(1.0, le), 1 + max(lk, rk)
                note("exponent", eff, node)
                note("power_nesting", nest - 1, node)
                res = (num, den, eff, nest)
            elif isinstance(node.op, (ast.Add, ast.Sub)):
                res = (_log10_add(ln + rd, rn + ld), min(ld + rd, cap), max(le, re_), max(lk, rk))
            elif isinstance(node.op, ast.Div):
                res = (min(ln + rd, cap), min(ld + rn, cap), max(le, re_), max(lk, rk))
            else:
                res = (min(ln + rn, cap), min(ld + rd, cap), max(le, re_), max(lk, rk))
        elif isinstance(node, ast.UnaryOp):
            res = get(node.operand)
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == SUM_NAME:
            # The split long sum, folded exactly as the left-nested chain it stands for.
            res = get(node.args[0])
            for k in node.args[1:]:
                (ln, ld, le, lk), (rn, rd, re_, rk) = res, get(k)
                res = (_log10_add(ln + rd, rn + ld), min(ld + rd, cap), max(le, re_), max(lk, rk))
        elif isinstance(node, ast.Call):
            args = [get(k) for k in kids]
            res = (0.0, 0.0, max((a[2] for a in args), default=0.0), max((a[3] for a in args), default=0))
        else:
            args = [get(k) for k in kids]
            res = (max((a[0] for a in args), default=0.0), max((a[1] for a in args), default=0.0),
                   max((a[2] for a in args), default=0.0), max((a[3] for a in args), default=0))
        note("number_digits", max(res[0], res[1]), node)
        info[id(node)] = res
        for k in kids:
            info.pop(id(k), None)

    out: Dict[str, object] = {}
    for key, (value, node) in worst.items():
        if key == "number_digits":
            out[key] = int(math.floor(value)) + 1 if value > 0 else 1
        elif key == "exponent":
            out[key] = int(round(value)) if value < 1e18 else int(1e18)
        else:
            out[key] = int(value)
        out[key + "_at"] = _snippet(raw, node) if node is not None else ""
    return out


def check_input(raw: str, limits: InputLimits = DEFAULT_INPUT_LIMITS) -> None:
    """Raise `InputLimitError` listing every rule and bound `raw` violates."""
    issues: List[InputIssue] = []
    if not raw or not raw.strip():
        raise InputLimitError([InputIssue(None, "empty input")])

    if len(raw) > limits.max_input_chars:
        issues.append(_over("max_input_chars", len(raw), limits))
        raise InputLimitError(issues)

    if "^" in raw:
        issues.append(InputIssue(None, "'^' is not a power operator here; write x**2 for x squared"))
    odd = sorted({c for c in raw if ord(c) > 127})
    if odd:
        hints = [f"{c!r} -> {_ASCII_HINTS[c]!r}" for c in odd if c in _ASCII_HINTS]
        issues.append(InputIssue(None, "non-ASCII character(s) " + "".join(odd[:12])
                                 + (" (replace " + ", ".join(hints) + ")" if hints else "")))
    if not issues and not _ALLOWED_CHARS.match(raw):
        bad = "".join(sorted({c for c in raw if not _ALLOWED_CHARS.match(c)})[:12])
        issues.append(InputIssue(None, f"illegal character(s) {bad!r}: only ab/sb/I, s(), chain(), "
                                       f"integers and + - * / ( ) , [ ] ' are allowed"))

    m = measure(raw)
    if m["int_digits"] > limits.max_int_digits:
        issues.append(_over("max_int_digits", m["int_digits"], limits))
    if m["nesting_depth"] > limits.max_nesting_depth:
        issues.append(_over("max_nesting_depth", m["nesting_depth"], limits))
    if m["exponent"] is not None:
        if m["exponent"] > limits.max_exponent:
            issues.append(_over("max_exponent", m["exponent"], limits, m["exponent_at"]))
        if m["power_nesting"] > limits.max_power_nesting:
            issues.append(_over("max_power_nesting", m["power_nesting"], limits, m["power_nesting_at"]))
        if m["number_digits"] > limits.max_number_digits:
            issues.append(_over("max_number_digits", m["number_digits"], limits,
                                m["number_digits_at"]))
    if issues:
        raise InputLimitError(issues)


def limits_from_mapping(values: Dict[str, Optional[int]],
                        base: InputLimits = DEFAULT_INPUT_LIMITS) -> InputLimits:
    """`base` updated with the non-None entries of `values`, keyed by field name."""
    known = {f.name for f in fields(InputLimits)}
    unknown = set(values) - known
    if unknown:
        raise ValueError(f"unknown input limit(s): {sorted(unknown)}")
    return InputLimits(**{f: (values[f] if values.get(f) is not None else getattr(base, f))
                          for f in known})
