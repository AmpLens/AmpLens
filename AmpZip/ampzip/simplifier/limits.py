"""Numeric limit constants, the single source for the validator.

The SFT evaluation settings, from the SFT evaluation worker scripts:
    n  max_terms  max_terms_action  max_brackets_per_term  max_brackets
    4  20         10                8                       12
    5  25         25                15                      20
    6  35         30                20                      30

Three of the four columns are part of the checkpoint architecture. `max_brackets_per_term`
sets the in-features of `encoder.term_embedding.weight` through `2 + B(2+2n) + 1`, which is
83, 183 and 283, and any other value fails `load_state_dict`. `max_terms_action` enters the
out-features of `pi.4` through `(1 + this)`. `max_brackets` is also a factor of those
out-features and is saturated at `C(n,2)*2`. `max_terms` is a data-generation and
evaluation setting on which no weight depends, there being no positional embedding, so
changing it loads and runs. It is a soft distribution-matching constraint.

The checkpoints record no hyperparameter dict, only epoch, model_state_dict, val_loss and
val_acc, so the architectural values are recovered by inverting tensor shapes.

The SFT paper states B=8 for 4-point only, and its 5-point action space of 20x11x19 = 4,180
disagrees with the published 5pt checkpoint, whose `pi.4` out-features are 9,880 = 20x26x19,
which means max_terms_action=25. The 4pt and 6pt values agree with the paper, which has no
counterpart for the `max_terms` column.
"""

from typing import Dict, NamedTuple
from ..atoms import ATOMS


class NPointLimits(NamedTuple):
    n_point: int
    max_terms: int
    max_terms_action: int
    max_brackets_per_term: int
    max_brackets: int


SFT_LIMITS: Dict[int, NPointLimits] = {
    4: NPointLimits(n_point=4, max_terms=20, max_terms_action=10, max_brackets_per_term=8, max_brackets=12),
    5: NPointLimits(n_point=5, max_terms=25, max_terms_action=25, max_brackets_per_term=15, max_brackets=20),
    6: NPointLimits(n_point=6, max_terms=35, max_terms_action=30, max_brackets_per_term=20, max_brackets=30),
}

# The allowed atoms, function head names, in input expressions. opaque_den(S) is the
# engine's marker for a symbolic 1/S sum-denominator factor. The atomizer emits it under
# allow_opaque_den=True and users never type it, and validate accepts it so the gbfs family
# can carry the spurious pole through the search and cancel it. preorder_traversal still
# finds the ab/sb inside S, so n_point inference and the size limits are unaffected.
ALLOWED_ATOMS = frozenset(ATOMS) | {"opaque_den"}

# Atoms recognised and rejected with a tailored message.
KNOWN_REJECTED_ATOMS: Dict[str, str] = {
    "ap": "massive bold angle bracket — no massless backend",
    "sp": "massive bold square bracket — no massless backend",
    "xf": "x-factor (massive)",
    "ms": "mass symbol",
    "sw": "sandwich bracket",
}

# Symbol names recognised and rejected.
KNOWN_REJECTED_SYMBOLS: Dict[str, str] = {
    "xi": "reference spinor (auxiliary, massive context)",
    "pi": "transcendental constant outside training distribution",
}

# Runtime default budgets, which the CLI can override.
DEFAULT_MAX_STEPS = 50
DEFAULT_PER_STEP_TIMEOUT = 30.0
DEFAULT_SAMPLE_TIMEOUT = 120.0

# SFT-beam defaults, the number of rollouts and the sampling temperature.
DEFAULT_SFT_BEAM_SIZE = 5
DEFAULT_SFT_BEAM_TEMPERATURE = 1.5

# SFT patience stops a rollout after N consecutive steps without a lexicographic
# improvement of (bracket_count, term_count). 0 disables it. Values of 10 to 15 suit inputs
# from the paper's distribution and 5 to 8 near-canonical inputs.
DEFAULT_SFT_PATIENCE = 0

# gbfs defaults. The search runs target-free, minimising the bracket count. max_apps is the
# move budget and max_seconds the time cap, at which the best form so far is returned. These
# are the paper's headline settings, 6e4 moves and a 300 s per-case cap, shared by the CLI
# and the web form. A deployment's limit WEBSVC_MAX_GBFS_MAX_SECONDS may be lower, in which
# case the page shows min(default, limit).
DEFAULT_GBFS_MAX_APPS = 60000
DEFAULT_GBFS_MAX_SECONDS = 300.0
# Maximum numerator term count during input expansion and identity application,
# including whole-expression and single-term rewrites. Moves exceeding the limit
# are skipped and counted in term_limit_skips. Inputs exceeding it are rejected.
# The CLI and web service pass this value explicitly.
# It matches the engine's built-in default.
ENGINE_DEFAULT_MAX_TERMS = 50000
DEFAULT_GBFS_MAX_TERMS = ENGINE_DEFAULT_MAX_TERMS
# The frontier ordering of the search. `factored` is the paper's gbfs, by factored numerator
# bracket count, and `raw` the paper's gbfsraw, by raw numerator bracket count.
GBFS_ORDERINGS = ("factored", "raw")
DEFAULT_GBFS_ORDERING = "factored"
# What `--target` means for the search family. With `none` the run is target-free and the
# target only marks `solved` afterwards. With `brackets` the search stops at the first node
# whose raw numerator bracket count is at or below the target's, the criterion of the size
# times depth suite. With `brackets-and-terms` the term count must also be at or below the
# target's, the criterion of the public sets, kept for parity with the neural baselines.
GBFS_TARGET_CRITERIA = ("none", "brackets", "brackets-and-terms")
DEFAULT_GBFS_TARGET_CRITERION = "brackets"
# Worker threads for node expansion inside one process. 0 means every CPU the process may use.
DEFAULT_GBFS_THREADS = 1
# For gbfs_e and gbfs_f, an input with more additive terms than this is searched whole.
# 0 means no limit.
DEFAULT_GBFS_SPLIT_MAX_TERMS = 64
# In gbfs_c, a node or child whose numerator or denominator has more terms than this is
# skipped before sympy's polynomial cancel, which cannot be interrupted, and counted in the
# result as `node_skips` or `child_term_skips`.
DEFAULT_GBFS_C_CANCEL_MAX_TERMS = 1200
# Wall-clock bound on the sympy post-processing of a backend's output under
# `--simplify-mode` and on the bracket recounts, which run in a helper process so they can
# be killed. On expiry the raw backend output is shown and `post_process_timed_out` says so.
DEFAULT_POST_TIMEOUT_S = 30.0
# The grace the runner adds on top of a backend's own wall-clock budget before the parallel
# collector stops waiting for it. Sequential runs have no collector deadline.
DEFAULT_ALGO_GRACE_S = 30.0
# How many of the requested backends run concurrently. The CLI default is one at a time.
DEFAULT_MAX_PARALLEL_ALGOS = 1

# CDS defaults for its token-level beam.
DEFAULT_CDS_BEAM_SIZE = 5
DEFAULT_CDS_TEMPERATURE = 1.5
DEFAULT_CDS_NUCLEUS_P = 0.0          # a positive value switches to nucleus sampling
DEFAULT_CDS_LENGTH_PENALTY = 1.0

# The output post-processing mode, for display only. The default "none" shows and counts the
# backend's own output. "cancel" runs sp.cancel, which normalises p/q and cancels common
# factors. "expand" runs sp.cancel and sp.expand, which distributes denominators and catches
# partial fractions. "simplify" runs sp.simplify, which is aggressive and can hang on big
# expressions.
SIMPLIFY_MODES = ("none", "cancel", "expand", "simplify")
DEFAULT_SIMPLIFY_MODE = "none"

# The supported n_points, one published checkpoint each.
SUPPORTED_N_POINTS = (4, 5, 6)
