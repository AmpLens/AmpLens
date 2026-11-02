"""A fixed-shape observation tensor from an `ampzip_engine.Expr` handle.

It is built from the engine's accessors `term_powers` and `den_brackets` in the engine's
canonical order, so two spellings of the same expression give the same tensor, and it
matches the order of the `term_slot` action axis in `action_space.py`. It is the input of a
policy or value network, and the search does not use it.

The layout, with n=4 and max_terms=10 giving dim 146, is
  numerator   : max_terms x (n_brackets + 1)
                per term: exponent of each canonical bracket,
                then signed log1p(|coeff|)
  denominator : n_brackets, exponent of each canonical bracket
  global      : [min(num_terms, max_terms), bracket_count,
                 num_distinct_brackets, has_monomial_denominator]

Denominator features describe only the monomial part and omit opaque factors.
"""
from __future__ import annotations

import math

import numpy as np

from .action_space import ActionSpace, default_max_terms


class Observation:
    def __init__(self, n: int = 4, max_terms: int | None = None):
        self.n = n
        self.max_terms = default_max_terms(n) if max_terms is None else max_terms
        self.brackets = ActionSpace._canonical_brackets(n)
        self._bidx = {b: i for i, b in enumerate(self.brackets)}
        self.n_brackets = len(self.brackets)
        self.term_dim = self.n_brackets + 1
        self.num_dim = self.max_terms * self.term_dim
        self.den_dim = self.n_brackets
        self.global_dim = 4
        self.dim = self.num_dim + self.den_dim + self.global_dim

    def encode(self, handle) -> np.ndarray:
        num = np.zeros((self.max_terms, self.term_dim), dtype=np.float32)
        tp = handle.term_powers()
        for t, (coeff, brs) in enumerate(tp[: self.max_terms]):
            for sq, i, j, e in brs:
                bi = self._bidx.get((sq, i, j))
                if bi is not None:
                    num[t, bi] = float(e)
            num[t, self.n_brackets] = math.copysign(math.log1p(abs(coeff)), coeff)
        den = np.zeros(self.den_dim, dtype=np.float32)
        den_brs = handle.den_brackets()
        for sq, i, j, e in den_brs:
            bi = self._bidx.get((sq, i, j))
            if bi is not None:
                den[bi] = float(e)
        g = np.array(
            [
                float(min(len(tp), self.max_terms)),
                float(handle.bracket_count()),
                float(len(handle.distinct_brackets())),
                1.0 if den_brs else 0.0,
            ],
            dtype=np.float32,
        )
        return np.concatenate([num.reshape(-1), den, g])
