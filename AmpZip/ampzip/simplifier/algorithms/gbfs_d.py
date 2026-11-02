"""The gbfs_d variant, best-first search under the physical-atom objective.

gbfs_d minimises the sum of brackets, Dirac chains and S symbols in the numerator. It has
the same engine, the same value-preserving moves and the same finite-field test as
`model_free_gbfs`, and differs only in the objective that orders the frontier and tracks
the returned best node.

The objective is the physical-atom count. Each S symbol `ab(i,j)*sb(i,j)` and each
single-leg Dirac chain `ab(a,m)*sb(m,b)` folds two brackets into one atom, and everything
else is one atom per bracket, as `ampzip.search.objective_atoms` defines it. The score is
the disjoint partition `#S + #chain + #bare`, so no bracket is double-counted. Atoms are
summed per term with no cross-term common-factor extraction, whereas the bracket objective's
`factored_bracket_count` pulls a bracket shared by every term out and counts it once. So on
a common-factor sum the two objectives can rank the same node oppositely.

gbfs_d keeps the denominator-aware frontier bias off. The atom objective already drives both
the ordering and the best-tracking, so the answer favours forms whose brackets pair into
physical invariants even when that is not the bracket-count minimum.

gbfs_e differs in kind. It is a per-term driver that splits the numerator into its additive
terms, simplifies each independently and recombines, and it is no objective.
`docs/GBFS_VARIANTS.md` has the comparison.
"""
from __future__ import annotations

from .gbfs import GBFS
from ampzip.search.objective_atoms import atom_heuristic


class GBFSVersionD(GBFS):
    name = "gbfs_d"
    objective_fn = staticmethod(atom_heuristic)
    use_den_aware = False
