"""Legal-move enumeration on an `ampzip_engine.Expr` handle.

A move is `(square, i, j, term_slot, identity, params)` — apply identity to bracket
`bk(i,j)` (square: False=ab, True=sb), either whole-expression (`term_slot=0`) or in
one numerator term (`term_slot=k>0`, the engine's k-1'th term).

The per-bracket identity menu reproduces the shipped RL action spaces exactly
(`RL_amplitudes_claude/src/action_space_{4,5,6}pt.py::get_identity_types_for_bracket`):
per bracket, **C(n-2,2) Schouten + 2(n-1) momentum + 2^(n-2) momentum_sq** identities —
i.e. 11 (4pt), 19 (5pt), 32 (6pt). Verified against all three shipped files.

Legality (mirrors `eval_sft_4pt.py::build_action_mask_v3`): a bracket move is legal at
`term_slot=0` iff the bracket is in the numerator; at `term_slot=k>0` iff the bracket is
in that term. No identity-specific masking — the full identity menu is offered once
the bracket is present (degenerate ones are filtered when the engine applies them).
"""
from __future__ import annotations

import ast
import re
from itertools import combinations

_BRACKET_RE = re.compile(r"\b(ab|sb)\((\d+),\s*(\d+)\)")


def identity_menu(square: bool, i: int, j: int, n: int):
    """The (identity, params) menu for bracket `bk(i,j)` — 11/19/32 entries at n=4/5/6.

    Order matches the shipped action spaces: Schouten, then momentum, then momentum_sq.
    params encoding matches `ampzip_engine`:
      schouten -> [pk, pl] ; momentum -> [pout1, pout2] ; momentum_sq -> mom_add list.
    """
    other = [m for m in range(1, n + 1) if m != i and m != j]
    menu = []
    # Schouten: ALL C(n-2, 2) pairs of the other legs (combinations order).
    for pk, pl in combinations(other, 2):
        menu.append(("schouten", [pk, pl]))
    # Momentum: pout1 in the bracket's own legs, pout2 in {pout1} + other legs -> 2(n-1).
    for pout1 in (i, j):
        for pout2 in [pout1] + other:
            menu.append(("momentum", [pout1, pout2]))
    # Momentum_sq: every subset of the other legs as mom_add -> 2^(n-2).
    for size in range(len(other) + 1):
        for mom_add in combinations(other, size):
            menu.append(("momentum_sq", list(mom_add)))
    return menu


def _term_brackets(mono_str: str):
    """Set of (square, i, j) brackets present in a canonical monomial string."""
    return {
        (typ == "sb", int(a), int(b))
        for typ, a, b in _BRACKET_RE.findall(mono_str)
    }


def legal_moves(handle, n: int):
    """All legal `(square, i, j, term_slot, identity, params)` moves for `handle`.

    `term_slot=0` (whole expression) for every present numerator bracket, plus
    `term_slot=k+1` for each numerator term that contains the bracket.

    The engine also enumerates moves through `Expr.legal_moves`, but returning a
    Python tuple, string and parameter list for every move can make it slower than
    this function, which reuses each identity's `params` list across term slots.
    The threaded search avoids these allocations by keeping the move list in Rust
    with `Expr.legal_move_list` and applying slices through `Expr.apply_slice_coded`,
    falling back to `Expr.apply_slice` on older engines.
    """
    present = handle.distinct_brackets()  # [(square, i, j)] in the numerator
    if not present:
        return []
    per_term = [_term_brackets(m) for m in handle.term_monos()]
    moves = []
    for square, i, j in present:
        menu = identity_menu(square, i, j, n)
        # whole expression
        for ident, params in menu:
            moves.append((square, i, j, 0, ident, params))
        # single term
        for k, bset in enumerate(per_term):
            if (square, i, j) in bset:
                for ident, params in menu:
                    moves.append((square, i, j, k + 1, ident, params))
    return moves


# Explicit name for the Python move enumerator, retained for compatibility and
# comparison with the Rust implementation.
legal_moves_py = legal_moves


def move_rule(square, i, j, identity, params, n):
    """The local rewrite a move performs on `bk(i,j)`, as `(lhs_str, rhs_str)` in
    ab/sb notation. The RHS is obtained by applying the identity to the **isolated
    bracket** — so it is the literal substitution the engine makes, signs and all,
    with no re-derivation of the algebra here (the engine is the source of truth).
    Returns `(lhs, None)` if the engine can't build it (e.g. degenerate params).
    """
    import ampzip_engine as _E
    lhs = f"{'sb' if square else 'ab'}({i},{j})"
    try:
        rhs = _E.Expr(lhs).apply(square, i, j, identity, list(params), n, term_slot=0).to_str()
    except Exception:
        rhs = None
    return lhs, rhs


def annotate_trajectory_rules(traj, n):
    """Add `rule_lhs` / `rule_rhs` (ab/sb strings) to each trajectory step in place,
    reconstructed from its `bracket` / `identity` / `params`. Best-effort: a step
    missing fields or whose rule can't be built is left as-is. Safe to call on the
    raw rollout trajectory (params is a list) or a serialized one (params stringified).
    """
    if not traj:
        return traj
    for step in traj:
        if not isinstance(step, dict) or step.get("rule_lhs"):
            continue
        m = _BRACKET_RE.search(str(step.get("bracket", "")))
        ident, params = step.get("identity"), step.get("params")
        if not m or ident is None or params is None:
            continue
        if not isinstance(params, (list, tuple)):
            try:
                params = ast.literal_eval(str(params))
            except Exception:
                continue
        square, ii, jj = (m.group(1) == "sb"), int(m.group(2)), int(m.group(3))
        lhs, rhs = move_rule(square, ii, jj, ident, list(params), n)
        step["rule_lhs"] = lhs
        if rhs is not None:
            step["rule_rhs"] = rhs
    return traj
