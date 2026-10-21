"""Numeric limit constants. Single source of truth for the validator.

Stage 1 hyperparameters (per `scripts/run_eval_{4,5,6}pt_worker.sh`):
    n  max_terms  max_terms_action  max_brackets_per_term  max_brackets
    4  20         10                8                       12
    5  25         25                15                      20
    6  35         30                20                      30

All 12 numbers are copied verbatim from Shih's worker scripts (verified byte-exact against
`scripts/run_eval_{4,5,6}pt_worker.sh:35-38`); none were invented here. But only THREE of the
four columns are genuinely baked into the checkpoint architecture:

  max_brackets_per_term  (a) architectural — sets `encoder.term_embedding.weight` in-features
                             via `2 + B(2+2n) + 1` = 83/183/283. Any other value fails
                             `load_state_dict`.
  max_terms_action       (a) architectural — enters `pi.4` out-features via `(1 + this)`.
  max_brackets           (a) architectural — also a factor of `pi.4` out-features. Also
                             physically saturated at `C(n,2)*2`, so there is no headroom.
  max_terms              (b) NOT architectural — a data-generation / evaluation knob. No
                             weight depends on it (there is no positional embedding; it is
                             only a reshape length at `src/train_sft_transformer.py:126`).
                             Changing it loads and runs fine — it is a soft distribution-
                             matching constraint, not a load-time one.

The checkpoints themselves record NO hyperparameter dict (only epoch / model_state_dict /
val_loss / val_acc), so the (a) values are recoverable solely by inverting tensor shapes.

Paper caveat: arXiv-2603.11164 states B=8 for 4-point ONLY, and its 5-point action space
(20x11x19 = 4,180, i.e. N_max=10) CONTRADICTS the shipped 5pt checkpoint, whose `pi.4`
out-features are 9,880 = 20x26x19, i.e. max_terms_action=25. 4pt and 6pt agree. The paper
also has no counterpart at all for the `max_terms` column.
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

# Allowed atoms (function head names) in input expressions.
#   opaque_den(S) is the engine marker for a symbolic 1/S polynomial-sum denominator
#   factor (B2). It is emitted by the atomizer only under allow_opaque_den=True (never
#   typed by users via s()/chain()); validate must accept it so the gbfs backend can
#   carry the spurious pole through the search and cancel it. preorder_traversal still
#   finds the ab/sb inside S, so n_point inference and size limits are unaffected.
ALLOWED_ATOMS = frozenset(ATOMS) | {"opaque_den"}   # the registered atoms + the engine's pole marker

# Atoms we explicitly recognize and reject with a tailored message (vs generic unknown atom).
KNOWN_REJECTED_ATOMS: Dict[str, str] = {
    "ap": "massive bold angle bracket — no massless backend",
    "sp": "massive bold square bracket — no massless backend",
    "xf": "x-factor (massive)",
    "ms": "mass symbol",
    "sw": "sandwich bracket",
}

# Symbol names we explicitly recognize and reject.
KNOWN_REJECTED_SYMBOLS: Dict[str, str] = {
    "xi": "reference spinor (auxiliary, massive context)",
    "pi": "transcendental constant outside training distribution",
}

# Runtime default budgets (overridable via CLI).
DEFAULT_MAX_STEPS = 50
DEFAULT_PER_STEP_TIMEOUT = 30.0
DEFAULT_SAMPLE_TIMEOUT = 120.0

# SFT-beam defaults (n_rollouts, sampling temperature).
DEFAULT_SFT_BEAM_SIZE = 5
DEFAULT_SFT_BEAM_TEMPERATURE = 1.5

# SFT patience: early-stop after N consecutive steps without lex improvement of
# (bracket_count, term_count). 0 = disabled; typical good
# values 10-15 for paper-distribution inputs, 5-8 for near-canonical inputs.
DEFAULT_SFT_PATIENCE = 0

# gbfs (model-free best-first search over the fast Rust engine) defaults. Runs
# target-free (minimize bracket count, no target fed). max_apps = identity-application
# budget; max_seconds = wall-clock cap (returns best-so-far on cut).
# The paper's headline setting (sec:gbfs): 6e4 moves and a 300 s per-case cap. Shared by the
# CLI and the web form; a deployment's ceiling (WEBSVC_MAX_GBFS_MAX_SECONDS) may be lower,
# in which case the page shows min(default, ceiling).
DEFAULT_GBFS_MAX_APPS = 60000
DEFAULT_GBFS_MAX_SECONDS = 300.0
# Maximum numerator term count during input expansion and identity application,
# including whole-expression and single-term rewrites. Moves exceeding the limit
# are skipped and counted in term_limit_skips. Inputs exceeding it are rejected.
# The CLI and web service pass this value explicitly.
# It matches the engine's built-in default.
ENGINE_DEFAULT_MAX_TERMS = 50000
DEFAULT_GBFS_MAX_TERMS = ENGINE_DEFAULT_MAX_TERMS
# Frontier ordering of the search: `factored` is the paper's gbfs (factored numerator
# bracket count); `raw` is the paper's gbfsraw (raw numerator bracket count).
GBFS_ORDERINGS = ("factored", "raw")
DEFAULT_GBFS_ORDERING = "factored"
# What `--target` means for the search family. `none`: target-free run, the target only
# marks `solved` afterwards. `brackets`: stop at the first node whose raw numerator bracket
# count is at or below the target's (the size-times-depth suite criterion).
# `brackets-and-terms`: the term count must also be at or below the target's (the public-set
# criterion, kept for parity with the baselines).
GBFS_TARGET_CRITERIA = ("none", "brackets", "brackets-and-terms")
DEFAULT_GBFS_TARGET_CRITERION = "brackets"
# Worker threads for node expansion inside one process; 0 = every CPU the process may use.
DEFAULT_GBFS_THREADS = 1
# gbfs_e / gbfs_f: additive terms above this are not split (the input is searched whole);
# 0 = no limit.
DEFAULT_GBFS_SPLIT_MAX_TERMS = 64
# gbfs_c: a node or child whose numerator or denominator has more terms than this is not
# fed to sympy's polynomial cancel (which cannot be interrupted) and is skipped, counted in
# the result as `node_skips` / `child_term_skips`.
DEFAULT_GBFS_C_CANCEL_MAX_TERMS = 1200
# Wall-clock bound on the sympy post-processing of a backend's output (`--simplify-mode`)
# and on the bracket recounts, run in a helper process so they can be killed. On expiry the
# raw backend output is shown and the result says so (`post_process_timed_out`).
DEFAULT_POST_TIMEOUT_S = 30.0
# Grace the runner adds on top of a backend's own wall-clock budget before the parallel
# collector stops waiting for it (sequential runs have no collector deadline).
DEFAULT_ALGO_GRACE_S = 30.0
# How many of the requested backends run concurrently (CLI default: one at a time).
DEFAULT_MAX_PARALLEL_ALGOS = 1

# CDS (seq2seq token-level beam) defaults.
DEFAULT_CDS_BEAM_SIZE = 5
DEFAULT_CDS_TEMPERATURE = 1.5
DEFAULT_CDS_NUCLEUS_P = 0.0          # >0 switches to nucleus sampling
DEFAULT_CDS_LENGTH_PENALTY = 1.0

# Output post-processing mode (display only; never used for a paper number). Default "none"
# since 2026-10-20: the backend's own output is what is shown and counted.
#   "none"     — raw backend output, no post-processing (default)
#   "cancel"   — sp.cancel (normalizes p/q, cancels common factors)
#   "expand"   — sp.cancel + sp.expand (distributes denominators; catches partial fractions)
#   "simplify" — sp.simplify (aggressive; can hang on big expressions)
SIMPLIFY_MODES = ("none", "cancel", "expand", "simplify")
DEFAULT_SIMPLIFY_MODE = "none"

# Supported n_points (one shipped checkpoint per).
SUPPORTED_N_POINTS = (4, 5, 6)
