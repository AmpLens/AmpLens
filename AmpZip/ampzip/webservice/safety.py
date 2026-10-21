"""Input checks before parsing.

Checks input length, permitted characters, numeric-literal length and chained exponentiation. Expression parsing and syntax validation are handled by ampzip.simplifier.parse.

See docs/SECURITY.md for public deployment guidance.
"""

from __future__ import annotations

import re
from typing import List

# Legal chars: whitespace, digits, the four letters that spell ab/sb/I,
# and arithmetic operators + grouping. No '.', '_', or any other letter.
_ALLOWED_CHARS = re.compile(r"^[\s0-9absI+\-*/(),]+$")

# Extended set (opt-in): additionally permits the s()/chain() notation tokens —
# the remaining letters of 'chain' (c,h,i,n), a single quote for the chain type
# string, and [] for channel lists. Still no '_'/'.', so dunder/attribute access
# and import/lambda/exec remain unspellable. The atomizer's restricted AST parser
# is required for this extended set. It expands s()/chain() to ab/sb and retains
# supported denominator sums as opaque_den markers. Raw extended input must not
# be passed to sympify.
_ALLOWED_CHARS_EXT = re.compile(r"^[\s0-9absIchin'\[\]+\-*/(),]+$")

# Forbid exponent towers like 9**9**9 (tiny string, astronomical integer → CPU/mem bomb).
# Two '**' with no intervening '*' between them => chained exponentiation.
_EXP_TOWER = re.compile(r"\*\*[^*]*\*\*")


class UnsafeInputError(ValueError):
    """Raised when input fails the pre-sympify firewall."""

    def __init__(self, reasons: List[str]):
        self.reasons = reasons
        super().__init__("; ".join(reasons))


def prescreen(raw: str, *, max_chars: int, max_digit_run: int,
              extended: bool = False) -> None:
    """Check text before parsing and validation. Raise `UnsafeInputError` on a
    failed check, otherwise return None. These conservative checks can reject
    valid expressions and do not replace syntax validation by the parser.

    `extended=True` permits s()/chain() notation (see `_ALLOWED_CHARS_EXT`).
    Use it only with the atomizer's restricted AST parser, which expands the
    notation to ab/sb with markers for supported denominator sums.
    """
    reasons: List[str] = []
    charset = _ALLOWED_CHARS_EXT if extended else _ALLOWED_CHARS

    if not raw or not raw.strip():
        reasons.append("empty input")
        raise UnsafeInputError(reasons)

    if len(raw) > max_chars:
        reasons.append(f"input too long: {len(raw)} > {max_chars} chars")

    if not charset.match(raw):
        # Report the offending characters (deduped, capped) to help legit users.
        bad = sorted({c for c in raw if not charset.match(c)})
        shown = "".join(bad[:12])
        allowed = ("ab/sb/I, s(), chain(), integers, and + - * / ( ) , [ ] '"
                   if extended else "ab/sb/I, integers, and + - * / ( ) ,")
        reasons.append(f"illegal character(s) present: {shown!r} (only {allowed} are allowed)")

    # Integer-literal length cap: stops 99999**99999-style magnitude bombs.
    for run in re.findall(r"\d+", raw):
        if len(run) > max_digit_run:
            reasons.append(f"integer literal '{run}' exceeds {max_digit_run} digits "
                           f"(leg labels and exponents are small)")
            break

    if _EXP_TOWER.search(raw):
        reasons.append("chained exponentiation (a**b**c) is not allowed")

    if reasons:
        raise UnsafeInputError(reasons)
