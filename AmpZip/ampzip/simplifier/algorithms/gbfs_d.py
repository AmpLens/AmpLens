"""GBFS version D ("version 4") — model-free best-first search under the physical-atom objective.

The (d) objective: minimize the **sum of brackets, Dirac chains, and S
symbols** in the numerator. Same engine, same value-preserving moves, same soundness gate as
`model_free_gbfs`; the ONLY difference is the simplicity objective that orders the frontier and
tracks the returned best node:

  1. **Physical-atom count, not bracket count.** Each S symbol (`ab(i,j)*sb(i,j)`) and each
     single-leg Dirac chain (`ab(a,m)*sb(m,b)`) folds two brackets into ONE atom; everything
     else is one atom per bracket (see `ampzip.search.objective_atoms`). The score is the
     disjoint partition `#S + #chain + #bare`, so no bracket is double-counted.
  2. **Atoms are summed per-term** (no cross-term common-factor extraction) — unlike the bracket
     objective's `factored_bracket_count`, which pulls a bracket shared by every term out and
     counts it once. So on a common-factor sum the two objectives can rank the SAME node
     oppositely.

version D keeps the den-aware frontier bias OFF: the atom objective already drives both ordering
and best-tracking, so the answer favours forms whose brackets pair into physical invariants even
when that isn't the raw-bracket minimum.

Not to be confused with (e), which is a *separate* per-term simplification **driver** — split the
numerator into its additive terms, simplify each independently, then recombine — and not an
objective. See `docs/GBFS_VARIANTS.md`.
"""
from __future__ import annotations

from .gbfs import GBFS
from ampzip.search.objective_atoms import atom_heuristic


class GBFSVersionD(GBFS):
    name = "gbfs_d"
    objective_fn = staticmethod(atom_heuristic)
    use_den_aware = False
