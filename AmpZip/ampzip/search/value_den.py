"""Denominator-aware frontier ordering for gbfs, a bias towards constructing the denominator factors.

`den_aware_value(handle)` sets frontier priorities in `gbfs.simplify` without
changing the search objective. Changing the order can improve or worsen the result
within a finite move budget.

The priority combines the raw numerator bracket count with `den_match_remaining` for
monomial denominator factors and `div_progress` for opaque polynomial factors. The
default GBFS backend uses this ordering only when `den_aware` is enabled. It is
disabled by default.

For fixed denominator factors, both scores are bounded, including for physical
poles. `den_match_remaining` is at most the sum of the monomial denominator's
bracket powers. `div_progress` is at most the sum of the distinct bracket counts of
the opaque factors. Thus, for fixed factors and weights, the bias changes the raw
bracket-count priority by a bounded amount even if a pole cannot be cancelled.

With no denominator, the priority equals the raw numerator bracket count. The
default ordering uses the factored numerator bracket count and the term count.
"""
from __future__ import annotations


def den_match_remaining(handle) -> int:
    """How much of the monomial denominator is not yet cancellable against the numerator.

    For each denominator bracket b^e, the cancellable amount is the minimum over numerator
    terms of b's power in that term, since a bracket cancels only if every term carries it.
    The remaining gap is the sum of max(0, e - that_min), and 0 means the whole denominator
    can cancel now.

    The engine's `Expr.den_match_remaining()` computes the same value without marshalling
    every term to Python, which made the denominator-aware value_fn about 2.3x slower in the
    pure-Python prototype. Older engines take the Python path.
    """
    rust = getattr(handle, "den_match_remaining", None)
    if rust is not None:
        return rust()
    den = handle.den_brackets()          # [(square, i, j, exp), ...], empty without a denominator
    if not den:
        return 0
    terms = handle.term_powers()         # [(coeff, [(square, i, j, exp), ...]), ...]
    if not terms:
        return 0
    rem = 0
    for (sq, i, j, e) in den:
        mn = min((next((ee for (s, a, b, ee) in brs if (s, a, b) == (sq, i, j)), 0)
                  for _c, brs in terms), default=0)
        rem += max(0, e - mn)
    return rem


def div_progress(handle) -> int:
    """Bracket overlap between the numerator and opaque denominator factors, used as
    a heuristic for frontier ordering.

    For each opaque factor S, count its distinct brackets that occur in the
    numerator, then sum over factors. The score is zero when no opaque factor is
    present, reducing the priority to
    `bracket_count + w_remaining * den_match_remaining`.

    A larger overlap can favour construction of a cancellable factor, but the score need not
    increase along a derivation and does not establish divisibility. Cancellation itself
    uses exact polynomial division in the engine, and the gbfs backends compare the selected
    result with the input through the finite-field test.
    """
    accessor = getattr(handle, "den_opaque_terms", None)
    if accessor is None:
        return 0
    opaque = accessor()
    if not opaque:
        return 0
    terms = handle.term_powers()
    if not terms:
        return 0
    num_brackets = {(sq, i, j) for _c, brs in terms for (sq, i, j, _e) in brs}
    score = 0
    for factor in opaque:
        s_brackets = {(sq, i, j) for _c, brs in factor for (sq, i, j, _e) in brs}
        score += sum(1 for b in s_brackets if b in num_brackets)
    return score


def make_den_aware_value(w_remaining: float = 1.0, w_div: float = 2.0):
    """Construct a frontier-priority function. `w_remaining` weights the unmatched
    monomial denominator factors, and `w_div` weights the bracket overlap with
    opaque denominator factors."""
    def value_fn(handle) -> float:
        return (handle.bracket_count()
                + w_remaining * den_match_remaining(handle)
                - w_div * div_progress(handle))
    return value_fn


# The default weights of the denominator-aware ordering.
den_aware_value = make_den_aware_value()
