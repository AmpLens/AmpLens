"""Fixed-size action encoding + legality mask for a learned policy.

Mirrors the shipped action spaces (`RL_amplitudes_claude/src/action_space_{4,5,6}pt.py`):

    action = bracket_idx * (n_term_slots * identities_per_bracket)
           + term_slot   * identities_per_bracket
           + identity_idx

Sizes (n_brackets x n_term_slots x identities_per_bracket): 4pt 12x11x11 = **1452**,
5pt 20x26x19 = **9880**, 6pt 30x31x32 = **29760** (matching the eval-banner sizes; the
per-n `max_terms` defaults below set n_term_slots). `term_slot=0` is whole-expression;
`term_slot=k>0` is the engine's k-1'th numerator term. The identity order is
`actions.identity_menu` (Schouten, then momentum, then momentum_sq).

The mask identifies legal action indices, allowing a policy to exclude illegal
actions when selecting from its output logits. `decode_to_move` maps a selected
index to an engine move for `Expr.apply`. Sequential GBFS enumerates moves through
`actions.legal_moves`, while parallel expansion uses `Expr.legal_move_list`.
Neither path uses this fixed action encoding.

NOTE: The default term-slot limits are 10, 25 and 30 for four-, five- and six-point inputs, respectively. This fixed encoding provides single-term actions only for the first `max_terms` numerator terms. GBFS enumerates moves directly and is not subject to this limit.
"""
from __future__ import annotations

import numpy as np

from .actions import identity_menu, _term_brackets

# Per-n term-slot cap (1 + max_terms = n_term_slots), matching the eval-banner action-space
# sizes. These cap how many per-term slots are representable in the fixed encoding; they are
# the search's own choice (not tied to a checkpoint), like the term_slot ordering decision.
_DEFAULT_MAX_TERMS = {4: 10, 5: 25, 6: 30}


def default_max_terms(n: int) -> int:
    return _DEFAULT_MAX_TERMS.get(n, 10)


class ActionSpace:
    def __init__(self, n: int = 4, max_terms: int | None = None):
        self.n = n
        if max_terms is None:
            max_terms = default_max_terms(n)
        self.max_terms = max_terms
        self.n_term_slots = 1 + max_terms
        self.brackets = self._canonical_brackets(n)          # [(square,i,j)], ab then sb
        self._bidx = {b: i for i, b in enumerate(self.brackets)}
        # The menu is structurally identical for every bracket; size it once.
        self.identities_per_bracket = len(identity_menu(False, 1, 2, n))
        self.n_actions = len(self.brackets) * self.n_term_slots * self.identities_per_bracket

    @staticmethod
    def _canonical_brackets(n: int):
        out = []
        for square in (False, True):     # ab (angle) then sb (square)
            for i in range(1, n + 1):
                for j in range(i + 1, n + 1):
                    out.append((square, i, j))
        return out

    def encode(self, bracket_idx: int, term_slot: int, identity_idx: int) -> int:
        return (bracket_idx * (self.n_term_slots * self.identities_per_bracket)
                + term_slot * self.identities_per_bracket + identity_idx)

    def decode(self, action: int):
        idy = action % self.identities_per_bracket
        rest = action // self.identities_per_bracket
        term_slot = rest % self.n_term_slots
        bracket_idx = rest // self.n_term_slots
        return bracket_idx, term_slot, idy

    def decode_to_move(self, action: int):
        """(action index) -> (square,i,j,term_slot,identity,params) for Expr.apply."""
        bracket_idx, term_slot, idy = self.decode(action)
        square, i, j = self.brackets[bracket_idx]
        ident, params = identity_menu(square, i, j, self.n)[idy]
        return (square, i, j, term_slot, ident, params)

    def mask(self, handle) -> np.ndarray:
        """0/1 legality vector of length n_actions for the current expression.

        All `identities_per_bracket` identities are legal once a bracket is present
        (no identity-specific masking, matching `build_action_mask_v3`).
        """
        m = np.zeros(self.n_actions, dtype=np.float32)
        n_id = self.identities_per_bracket
        present = handle.distinct_brackets()
        per_term = [_term_brackets(s) for s in handle.term_monos()]
        for b in present:
            b_idx = self._bidx.get(tuple(b))
            if b_idx is None:
                continue
            base = self.encode(b_idx, 0, 0)        # whole-expression block
            m[base:base + n_id] = 1.0
            for k, bset in enumerate(per_term):
                if k >= self.max_terms:
                    break
                if tuple(b) in bset:
                    base = self.encode(b_idx, k + 1, 0)
                    m[base:base + n_id] = 1.0
        return m
