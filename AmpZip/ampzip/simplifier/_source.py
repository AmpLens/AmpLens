"""Turn expression text into a syntax tree whose depth does not grow with the number of terms.

`ast.parse` builds a left-nested chain one level deep per summand, and CPython refuses to
construct a tree deeper than its C recursion limit, 4000 levels on Windows and 10000
elsewhere. A long sum is therefore split at its top-level `+` and `-` before parsing: each
summand is parsed on its own and the pieces become one call `_amp_sum(t1, -t2, t3, ...)`,
which the evaluator binds to `sympy.Add`. Short sums are parsed as written. Node positions are
shifted back to the whole text, so error snippets still point at the right place.
"""
from __future__ import annotations

import ast
from typing import List, Tuple

SUM_NAME = "_amp_sum"

# Sums with at least this many top-level summands are parsed piecewise.
FLATTEN_MIN = 32

# A `+` or `-` after one of these characters is a sign, not a binary operator.
_OPERATOR_BEFORE = set("*/+-(,[")


def split_top_level(text: str) -> List[Tuple[int, int, bool]]:
    """The (start, end, negated) spans of the top-level summands of `text`.

    Only `+` and `-` outside parentheses, square brackets and string literals split, and
    only when they follow an operand, so `a**-2`, `a*-b` and a leading sign stay inside their
    summand. A single span covers text that is not a sum.
    """
    spans: List[Tuple[int, int, bool]] = []
    start, negated = 0, False
    depth, in_string, prev = 0, False, ""
    for pos, ch in enumerate(text):
        if in_string:
            if ch == "'":
                in_string = False
            continue
        if ch == "'":
            in_string = True
        elif ch in "([":
            depth += 1
        elif ch in ")]":
            depth -= 1
        elif ch in "+-" and depth == 0 and prev and prev not in _OPERATOR_BEFORE:
            spans.append((start, pos, negated))
            start, negated = pos + 1, ch == "-"
            prev = ch
            continue
        if not ch.isspace():
            prev = ch
    spans.append((start, len(text), negated))
    return spans


def _shift_columns(node: ast.AST, by: int) -> None:
    for sub in ast.walk(node):
        if hasattr(sub, "col_offset"):
            sub.col_offset += by
            if getattr(sub, "end_col_offset", None) is not None:
                sub.end_col_offset += by


def parse_source(raw: str) -> ast.Expression:
    """Parse `raw` in eval mode. Newlines become spaces first, since a sum wrapped across
    lines is a SyntaxError for Python's tokenizer. Raises what `ast.parse` raises."""
    text = raw.replace("\r", " ").replace("\n", " ")
    spans = split_top_level(text)
    if len(spans) < FLATTEN_MIN:
        return ast.parse(text, mode="eval")
    args = []
    for start, end, negated in spans:
        piece = text[start:end]
        lead = len(piece) - len(piece.lstrip())
        body = ast.parse(piece.strip(), mode="eval").body
        _shift_columns(body, len(text[: start + lead].encode("utf-8")))
        if negated:
            body = ast.copy_location(ast.UnaryOp(op=ast.USub(), operand=body), body)
        args.append(body)
    call = ast.Call(func=ast.Name(id=SUM_NAME, ctx=ast.Load()), args=args, keywords=[])
    call.lineno = call.end_lineno = 1
    call.col_offset = args[0].col_offset
    call.end_col_offset = args[-1].end_col_offset
    return ast.fix_missing_locations(ast.Expression(body=call))
