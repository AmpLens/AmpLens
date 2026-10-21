"""Model-free greedy/best-first search (gbfs) over the fast engine.

No neural network: the frontier is ranked by a hand-written heuristic
`h = (factored_bracket_count, num_terms)` — smaller is simpler. From the current
expression we expand every legal child (via `actions.legal_moves`), dedup by canonical
string (anti-cycle), and keep exploring the most-promising frontier node until we hit the
goal or exhaust the apply budget. Backtracking is inherent: the frontier retains
alternatives.

`Expr.is_zero()` recognises literal cancellation to 0 (a sufficient on-shell-zero test);
the exact mod-p zero-detector is layered on top by callers for the ≡0 class.

The default heuristic uses the factored numerator bracket count: a factor shared
by every term is counted once. See `heuristic` for the count definition and
`simplify` for objective and target options.
"""
from __future__ import annotations

import hashlib
import heapq
import os
import time
import warnings

import ampzip_engine as E

from . import gbfs_checkpoint as _CKPT
from .actions import legal_moves

# Threaded node expansion needs `Expr.legal_move_list` + `Expr.apply_slice` (the
# parallel-expansion engine build). Older .so files keep working — `threads` is then silently
# ignored and the search runs sequentially.
_HAS_PARALLEL_EXPAND = hasattr(E.Expr, "apply_slice")
# `Expr.apply_one` / `Expr.key` compute the dedup digest in Rust (same bytes as `node_key`),
# so the sequential path never builds a child's canonical string as a Python object.
_HAS_KEY = hasattr(E.Expr, "apply_one")
# `Expr.apply_slice_coded` returns the rejection cause of each move (engine >= the 2026-10-21
# build); without it the threaded path re-applies a rejected move to classify it.
_HAS_CODED_SLICE = hasattr(E.Expr, "apply_slice_coded")
_REJECT_KIND = {1: "engine_error", 2: "term_limit", 3: "overflow"}


def new_expr(start, max_terms: int = 0):
    """`E.Expr(start)` with the term cap applied while the input is expanded (engine >= the
    2026-10-21 build); older engines expand under their process default."""
    try:
        return E.Expr(start, max_terms=max_terms)
    except TypeError:
        return E.Expr(start)

# How many rebuilt frontier nodes `simplify` keeps around to shorten later replays.
# Bounded on purpose: the point of the lazy frontier is that handles are not retained.
_HANDLE_CACHE_MAX = 256


def node_key(canonical: str) -> bytes:
    """Dedup key for a node: a 128-bit digest of its canonical string, not the string.

    The search visits far more nodes than it expands (one expansion of a 6pt node yields
    ~10k children), and every visited node used to keep its full canonical string alive in
    `visited` / `depth` / `parent` / `move_into` and in the frontier tuple. At 6pt that
    string is tens of KB, which made MEMORY -- not the apply budget or the wall clock --
    the binding constraint: a measured 1h run on a 126-term expression tripped a 20 GB
    floor after 26 min and 35 expansions, at ~0.66 GB per expanded node.

    A digest costs 16 bytes regardless of expression size, and the string it is computed
    from is freed immediately. Dedup semantics are unchanged up to a collision, which
    blake2b-128 makes negligible: at 10^9 distinct nodes the birthday probability is
    ~10^-21, orders of magnitude below the odds of a bit-flip in the same run.

    The one thing the digest cannot do is spell the derivation back out, so `path_exprs`
    is now replayed from the root through `path_moves` instead of read off the keys (see
    the end of `simplify`); the replay is `depth` extra applies, and is verified against
    `best` before being returned.
    """
    return hashlib.blake2b(canonical.encode(), digest_size=16).digest()


_IDENT_NAME = ("schouten", "momentum", "momentum_sq")
_IDENT_CODE = {name: i for i, name in enumerate(_IDENT_NAME)}


# Move packing (layout 2). `move_into` keeps one entry per DISCOVERED node, and a 6-tuple with a
# params list measures ~216 B -- about 42% of what a discovered node costs -- so every move is
# packed into one Python int. Packing is pure bookkeeping for the derivation replay and plays no
# part in the search, so it changes footprint only, never the result.
#
# Layout, LSB first:  sq 1 | ident 2 | nparams 4 | i 6 | j 6 | params nparams x 6 | term_slot (top)
#
# The term slot sits at the TOP so its width is unbounded (Python ints are arbitrary precision):
# a slot is the numerator term index + 1, and a node may carry up to `max_terms` terms. Legs
# are 6 bits (labels 1..63, far beyond the supported 4/5/6-point menus) and a move carries up to
# 15 params (momentum_sq takes a subset of the n-2 other legs, so this covers n <= 17). Every
# field is range-checked, so an out-of-range move raises instead of silently truncating into a
# wrong derivation -- but the ranges are chosen so no legal move of a supported n can reach them.
#
# With the common 6-point move (4 params, term slot < 2**20) the int stays below 2**63, which is
# what keeps a checkpoint node record at 8 bytes (see gbfs_checkpoint: it widens to 16 bytes
# only when a run actually produced a larger value).
_PACK_LEG_BITS = 6
_PACK_MAX_LEG = 1 << _PACK_LEG_BITS          # legs are 1..63
_PACK_MAX_PARAMS = 15                        # nparams field is 4 bits
_PACK_TS_SHIFT0 = 1 + 2 + 4 + 2 * _PACK_LEG_BITS   # bit where params start (23)
_PRIO_SHIFT = 32                             # packed frontier priority: (factored << 32) | terms


def _pack_move(sq, i, j, ts, ident, params):
    """Pack one move `(sq, i, j, term_slot, identity, params)` into a single int (layout 2)."""
    p = list(params)
    if not (0 <= i < _PACK_MAX_LEG and 0 <= j < _PACK_MAX_LEG and ts >= 0
            and len(p) <= _PACK_MAX_PARAMS):
        raise ValueError(f"move out of packing range: {(sq, i, j, ts, ident, p)}")
    v = (1 if sq else 0) | (_IDENT_CODE[ident] << 1) | (len(p) << 3) \
        | (i << 7) | (j << (7 + _PACK_LEG_BITS))
    shift = _PACK_TS_SHIFT0
    for x in p:
        if not 0 <= x < _PACK_MAX_LEG:
            raise ValueError(f"move param out of packing range: {p}")
        v |= x << shift
        shift += _PACK_LEG_BITS
    v |= ts << shift
    return v


def _unpack_move(v):
    """Inverse of `_pack_move`; returns the same 6-tuple shape the search uses
    `(sq, i, j, term_slot, identity, [params])`."""
    npar = (v >> 3) & 0xF
    mask = _PACK_MAX_LEG - 1
    params = [(v >> (_PACK_TS_SHIFT0 + _PACK_LEG_BITS * a)) & mask for a in range(npar)]
    ts = v >> (_PACK_TS_SHIFT0 + _PACK_LEG_BITS * npar)
    return (bool(v & 1), (v >> 7) & mask, (v >> (7 + _PACK_LEG_BITS)) & mask, ts,
            _IDENT_NAME[(v >> 1) & 3], params)


def _unpack_move_v1(v):
    """Decoder for layout 1 (checkpoint version 1 files):
    sq 1 | i 4 | j 4 | term_slot 12 | identity 2 | nparams 3 | params 4x3."""
    npar = (v >> 23) & 7
    return (bool(v & 1), (v >> 1) & 15, (v >> 5) & 15, (v >> 9) & 4095,
            _IDENT_NAME[(v >> 21) & 3],
            [(v >> (26 + 3 * a)) & 7 for a in range(npar)])


def _pack_priority(f, t):
    """Frontier priority for the default objective `(factored, terms)` as one int.

    The heap only ever COMPARES priorities, so any strictly monotone re-encoding leaves the
    pop order identical; this drops the 56 B inner tuple per frontier entry. `terms` must fit
    the low 32 bits (a node's term count is bounded by `max_terms`), asserted not assumed."""
    if not (0 <= t < (1 << _PRIO_SHIFT) and f >= 0):
        raise ValueError(f"priority out of packing range: {(f, t)}")
    return (f << _PRIO_SHIFT) | t


def _repack_priority_v1(v):
    """Re-encode a layout-1 packed priority `(f << 20) | t` (stored as float64) to layout 2."""
    v = int(v)
    return ((v >> 20) << _PRIO_SHIFT) | (v & ((1 << 20) - 1))


_SKIP_KINDS = ("term_limit", "overflow", "engine_error")


def _classify_skip(exc):
    """Which cause a rejected move falls under, from the engine's error text."""
    msg = str(exc)
    if "AMPZIP_MAX_TERMS" in msg or "term count exceeded" in msg:
        return "term_limit"
    if "overflow" in msg:
        return "overflow"
    return "engine_error"


def heuristic(handle):
    """Simplicity objective / search cost: FACTORED numerator bracket count, then
    numerator #terms as tie-break. Smaller is simpler.

    `factored_bracket_count()` (Rust engine) pulls the common monomial factor out of the
    numerator sum and counts it once, so a bracket shared by EVERY term is not
    double-counted:
        ab(1,2)*ab(2,3)*ab(3,4) + ab(1,2)*ab(2,3)*ab(4,5)   ->  4  (not 6)
    It equals the raw `bracket_count()` whenever no bracket is common to all terms (one
    level of extraction, like sympy `factor_terms` — no recursion into partial-overlap
    subgroups). Denominator is excluded, exactly as in `bracket_count`.

    This ONE function drives all three uses of the metric — (a) frontier ordering,
    (b) best-node tracking, (c) the `goal_brackets` comparison — so the whole search runs
    in a single consistent unit (factored throughout). Two consequences for callers:

      * `goal_brackets` is now in FACTORED units: pass the target's
        `factored_bracket_count()` ("canon_fact"), NOT its raw `bracket_count()`. Mixing
        units (raw threshold vs factored node value) could otherwise declare a false solve
        on a node whose factored count dipped to the target while its raw count had not.
        For EVERY shipped canonical target canon_fact == canon_raw (single monomials, or
        gauge-invariant sums with no all-terms-common factor), so the many existing
        callers that still pass a raw target count remain correct today — verified on
        Parke-Taylor, QED ee->mumu, 4-graviton, and the 4-term eemumu eq-3.16 amplitude.
        They diverge only once a target carries an all-terms-common factor.

      * The returned "best_brackets" is now the FACTORED count. The raw count is also
        returned as "best_raw_brackets" so diagnostics can still compare against the
        historical raw-metric numbers (Stage-1 solve rates, CDS baselines).

    The runner recomputes reported bracket counts with `display_bracket_count`,
    using the factored numerator bracket count. The internal `count_brackets`
    function retains a separate count over the whole expression.

    (Switched from raw `bracket_count` on 2026-07-02.)
    """
    return (handle.factored_bracket_count(), handle.num_terms())


def simplify(start, n: int, max_apps: int = 3000, goal_brackets=None, max_seconds=None,
             value_fn=None, value_batch_fn=None, objective_fn=None, should_stop=None,
             goal_raw_brackets=None, goal_raw_terms=None, patience_apps=None,
             max_terms: int = 0, threads: int = 1, batch_chunk: int = 0,
             low_memory=None, checkpoint_path=None, resume_from=None,
             run_meta=None, progress_cb=None):
    """Best-first simplify `start` (str or `Expr`). Returns a stats dict.

    `objective_fn(handle) -> comparable` (optional): the simplicity objective that
    (a) orders the frontier when neither `value_fn` nor `value_batch_fn` is given
    and (b) ranks candidates for best-node tracking. Defaults to `heuristic` (factored bracket
    count) — passing the default reproduces the model-free gbfs exactly. GBFS
    version D ("version 4") passes `objective_atoms.atom_heuristic` here to minimise
    physical-atom count instead. It must return a tuple `(primary, tiebreak)` like
    `heuristic`; `best_brackets` in the result is `objective_fn(best)[0]` (its unit
    — e.g. atom count for version D), while the raw engine bracket count is always
    also returned as `best_raw_brackets`.

    `goal_brackets` stops the search when the best node's primary objective is at
    most the supplied value. With the default objective, this is the factored
    numerator bracket count. `goal_raw_brackets` takes precedence when supplied:
    the search stops at a node whose raw numerator bracket count meets that limit
    and, if given, whose term count is at most `goal_raw_terms`. That node is returned.

    Other stopping conditions are the move budget `max_apps`, the optional time
    limit `max_seconds`, an empty frontier and cancellation through `should_stop()`.
    With `patience_apps` set, the search also stops at a node boundary after at
    least that many applied moves without an objective improvement. In raw-goal
    mode, a decrease in the best raw numerator bracket count also resets this
    counter. On these stops, the best node found is returned.

    `value_fn(handle) -> float` optionally sets frontier priorities, with lower
    values expanded first. `value_batch_fn`, when supplied, takes precedence for
    ordering new children. Neither replaces `objective_fn`, which defaults to the
    factored numerator bracket count followed by the term count. A value function
    can prioritize promising intermediates with larger bracket counts,
    allowing paths with temporary growth to be explored earlier. Changing the
    frontier order can change the result within a fixed move budget. When both
    value functions are None, the objective also determines frontier order.

    `value_batch_fn(list[handle]) -> list[float]` (optional, takes precedence over value_fn for
    ordering): same role as value_fn but evaluates all of a node's children in ONE call — for a
    neural value this batches the forward (~branching-factor fewer GPU calls), which is the
    difference between seconds and milliseconds per case. Ordering semantics are identical to
    value_fn; only the call granularity differs.

    With `threads > 1`, Rust workers apply moves in parallel with the GIL released.
    Python processes the results in move order. Frontier updates, deduplication,
    move counting and best-candidate selection remain sequential. For a fixed input
    and move budget, with deterministic ranking functions and all other search
    settings unchanged, threaded and sequential runs return the same expression,
    derivation and counters unless a time limit or cancellation stops either run.

    Time and cancellation checks cannot interrupt a running Rust batch. A batch
    already in progress finishes before these checks take effect, and results beyond
    the stopping point are discarded. Its duration depends on all moves assigned to
    the workers.

    `progress_cb(apps, expanded, best_h)` runs at cancellation-check points,
    including before each accepted move is counted. Keep it lightweight because it
    runs frequently and its execution time contributes to the search time.

    `threads=0` uses the CPU affinity count when available, otherwise `os.cpu_count()`
    or 1. `batch_chunk` sets the batch size, which defaults to `max(256, 16*threads)`.
    Smaller batches allow more frequent time and cancellation checks at the cost of
    more synchronization.

    Bookkeeping is always the packed form (what used to be `low_memory=True`): moves are
    packed ints, the default-objective priority is one int, and a popped node's depth is
    walked off the parent chain. `low_memory` is accepted and ignored for old callers.

    The result reports why the search stopped (`stop_reason`: goal, budget, time,
    frontier_empty, cancelled or patience) and the number of rejected moves by cause
    (`term_limit_skips`, `overflow_skips` and `engine_error_skips`). Rejected moves
    do not consume the move budget.
    """
    t_start = time.perf_counter()
    if low_memory is not None:
        warnings.warn("gbfs.simplify: `low_memory` is ignored; the packed bookkeeping is the only path",
                      DeprecationWarning, stacklevel=2)
    if threads == 0:
        try:
            threads = len(os.sched_getaffinity(0))
        except AttributeError:
            threads = os.cpu_count() or 1
    root = new_expr(start, max_terms) if isinstance(start, str) else start
    root_str = root.to_str()
    root_key = node_key(root_str)
    objective = objective_fn if objective_fn is not None else heuristic

    def order(h):                  # frontier priority (value function or objective)
        return float(value_fn(h)) if value_fn is not None else objective(h)

    best_h = objective(root)       # simplicity objective (default: factored_bracket_count, num_terms)
    best, best_depth, best_key = root, 0, root_key
    # There is no per-node depth map: a popped node's depth is walked off the `parent` chain.
    # Pops are rarer than discoveries by ~3 orders of magnitude (a measured 3M-apply run
    # popped 8,401 of ~2.5M nodes), so that is O(depth) dict hits on a rare event against
    # 64 B on every node. `d` has the same value either way.
    parent = {root_key: None}      # node_key -> parent node_key (first-discovery path)
    move_into = {root_key: None}   # node_key -> packed move (see _pack_move) used to reach it
    # The default objective's (factored, terms) priority is packed into one int; a custom
    # objective keeps its own comparable (e.g. the atom objective's float primary) and a
    # value_fn gives floats. Only the packed form can be checkpointed exactly.
    packed_priority = objective_fn is None and value_fn is None and value_batch_fn is None
    priority_format = "packed" if packed_priority else ("f64" if (value_fn is not None or value_batch_fn is not None) else "tuple")
    visited = {root_key}
    # The frontier stores KEYS, not handles: a discovered child's `Expr` is dropped once its
    # priority is known, and rebuilt only if the node is ever popped for expansion. At 6pt the
    # two counts are orders of magnitude apart -- a measured run discovered 2.4M children and
    # popped 104 -- so retaining a handle per discovered child was the single largest consumer
    # of memory left after the keys became digests.
    frontier = [(order(root), 0, root_key)]
    counter = 1
    apps = expanded = 0
    # Resume restores the saved frontier and move counter. Reproducing the search order of an
    # uninterrupted run requires a checkpoint saved after a complete node expansion and
    # unchanged ordering and move settings. A partially expanded node's remaining moves are not
    # saved. Scalars and the best handle are restored after materialize and goal initialization
    # so initialization does not overwrite the restored values.
    _ck = None
    if resume_from is not None:
        _ck, parent, move_into, visited, _depth_unused, frontier = _CKPT.load(resume_from)
        del _depth_unused
        if _ck["root_str"] != root_str or _ck["n"] != n:
            raise ValueError("gbfs resume: checkpoint is for a different problem")
        counter, apps, expanded = _ck["counter"], _ck["apps"], _ck["expanded"]
        # Migrate older on-disk encodings in place (no second container is built).
        if _ck.get("move_layout", 1) == 1:
            for _k, _mv in move_into.items():
                if _mv is not None:
                    move_into[_k] = _pack_move(*_unpack_move_v1(_mv))
        _fmt = _ck.get("priority_format", "f64")
        if _ck.get("version", 1) >= 2 and (_fmt == "f64") == packed_priority:
            raise ValueError("gbfs resume: checkpoint frontier ordering (%s) does not match this run (%s)"
                             % ("value_fn" if _fmt == "f64" else "packed objective",
                                "packed objective" if packed_priority else "value_fn"))
        if packed_priority and _ck.get("priority_shift", 20) != _PRIO_SHIFT:
            if _ck.get("priority_shift", 20) != 20:
                raise ValueError("gbfs resume: unknown priority_shift in checkpoint")
            for _i, (_v, _c, _k) in enumerate(frontier):
                frontier[_i] = (_repack_priority_v1(_v), _c, _k)
    _seg = _CKPT.new_segment(threads=threads, batch_chunk=batch_chunk, max_apps=max_apps,
                             max_seconds=max_seconds, apps_start=apps,
                             expanded_start=expanded, **(run_meta or {}))
    # Bounded ancestor cache for `materialize`. Entries are only ever added while under the cap,
    # which keeps the nodes nearest the root -- exactly the ancestors every replay walks through.
    handle_cache = {root_key: root}

    def materialize(hk):
        """Rebuild the `Expr` for key `hk` by replaying its move chain from the nearest
        cached ancestor.

        Cost is one apply per uncached step, against the ~10k applies the expansion that
        follows will spend, so the trade is heavily in favour of not storing the handle.
        The rebuilt node is checked against its key before use: a replay that diverged
        would otherwise expand the wrong expression silently.
        """
        node = handle_cache.get(hk)
        if node is not None:
            return node
        pending = []
        k = hk
        while k not in handle_cache:
            pending.append(k)
            k = parent[k]
        node = handle_cache[k]
        for k2 in reversed(pending):
            mv = move_into[k2]
            sq, i, j, ts, ident, params = _unpack_move(mv)
            node = node.apply(sq, i, j, ident, list(params), n, term_slot=ts,
                              max_terms=max_terms)
            if len(handle_cache) < _HANDLE_CACHE_MAX:
                handle_cache[k2] = node
        if (node.key() if _HAS_KEY else node_key(node.to_str())) != hk:
            raise RuntimeError("gbfs: replayed frontier node does not match its key")
        return node
    # raw-unit tracking (only when goal_raw_brackets is given); see the child loop below
    best_raw_h = (root.bracket_count(), root.num_terms())
    best_raw, best_raw_key, best_raw_depth = root, root_key, 0
    apps_to_goal = None
    apps_at_best = 0          # applies spent when `best` was last improved
    patience_hit = False
    # `goal_raw_terms` (optional) tightens the raw goal to Shih's target-relative criterion:
    # brackets AND terms both <= the target's. The weaker brackets-only condition is still
    # recorded as it goes by (`bracket_goal`), so one run yields both metrics -- measuring them
    # in separate runs is wrong, since stopping on the weaker one truncates the search before a
    # both-conditions node can appear. Leaving it None reproduces the previous behaviour exactly.
    bracket_goal = False
    apps_to_bracket_goal = None
    if goal_raw_brackets is not None:
        bracket_goal = best_raw_h[0] <= goal_raw_brackets
        if bracket_goal:
            apps_to_bracket_goal = 0
        goal = bracket_goal and (goal_raw_terms is None
                                 or root.num_terms() <= goal_raw_terms)
        if goal:
            apps_to_goal = 0
    else:
        goal = goal_brackets is not None and best_h[0] <= goal_brackets
    if _ck is not None:
        best_key = bytes.fromhex(_ck["best_key"])
        best_h = tuple(_ck["best_h"]) if isinstance(_ck["best_h"], list) else _ck["best_h"]
        best_depth = _ck["best_depth"]
        best = materialize(best_key)
        best_raw_key = bytes.fromhex(_ck["best_raw_key"])
        best_raw_h = (tuple(_ck["best_raw_h"]) if isinstance(_ck["best_raw_h"], list)
                      else _ck["best_raw_h"])
        best_raw_depth = _ck["best_raw_depth"]
        best_raw = materialize(best_raw_key)
        apps_to_goal = _ck["apps_to_goal"]
        apps_to_bracket_goal = _ck["apps_to_bracket_goal"]
        apps_at_best = _ck["apps_at_best"]
        bracket_goal, goal = _ck["bracket_goal"], _ck["goal"]
        patience_hit = _ck["patience_hit"]
    _seg["best_h_start"] = list(best_h) if isinstance(best_h, tuple) else best_h

    stopped = False   # set when `should_stop()` cuts the search (cooperative cancel)
    skips = dict.fromkeys(_SKIP_KINDS, 0)   # moves the engine rejected, by cause

    parallel = threads > 1 and _HAS_PARALLEL_EXPAND
    # Batch size for the threaded path, trading two opposing costs. Too small and the workers
    # are starved — per-move cost is heavy-tailed, so a batch needs many moves per thread
    # before the atomic hand-out can balance them, and each batch re-pays the spawn/join. Too
    # large and the search over-computes past its goal: the median case finishes in a few
    # hundred applies (4pt full set: mean 257) while a big 6pt node offers ~10k legal moves.
    # Measured on 12 large 4/5/6pt cases (chunk x threads sweep, 20000 applies): 384 is the
    # best setting at 8/24/48 threads, 1.2x better than 4*threads and 1.7x better than 1536.
    chunk = batch_chunk if batch_chunk > 0 else max(256, 16 * threads)

    def children_of(node):
        """Yield `(move_src, move_key, child, key, pre)` for every legal move of `node` that
        applies cleanly, in enumeration order. `key` is the child's dedup digest (`node_key`
        of its canonical string, computed in Rust), never the string itself.

        `pre` is `(factored_bracket_count, bracket_count, num_terms)` when the worker threads
        already computed them off the GIL, else None and the caller reads them off the handle.
        The move is passed as a source + key rather than a tuple so the threaded path only
        materialises it for a child the caller actually keeps: `move_src=None` means
        `move_key` IS the tuple, otherwise it is an index into the resident `MoveList`.

        Moves the engine rejects (degenerate identity, illegal slot, `max_terms` blow-up) are
        dropped here rather than surfaced as `except ValueError: continue`. That is equivalent:
        they consume no apply budget, so skipping them cannot change which move the budget /
        deadline check lands on — only a rejected move can be "skipped over", and the very next
        accepted move sees the same counters.

        Sequential mode stays lazy — nothing past the caller's break is ever computed.
        """
        if not parallel:
            for mv in legal_moves(node, n):
                sq, i, j, ts, ident, params = mv
                try:
                    if _HAS_KEY:
                        child, ck, fbc, raw_bc, nterms = node.apply_one(
                            sq, i, j, ident, list(params), n, term_slot=ts, max_terms=max_terms)
                    else:
                        child = node.apply(sq, i, j, ident, list(params), n, term_slot=ts,
                                           max_terms=max_terms)
                        ck, fbc, raw_bc, nterms = node_key(child.to_str()), None, None, None
                except ValueError as exc:
                    skips[_classify_skip(exc)] += 1
                    continue
                yield None, mv, child, ck, ((fbc, raw_bc, nterms) if fbc is not None else None)
            return
        # The move list stays resident in Rust: converting a big node's ~10k moves to Python
        # tuples costs more than enumerating them (see actions.legal_moves).
        ml = node.legal_move_list(n)
        for c0 in range(0, len(ml), chunk):
            # Deadline / cancel are re-checked here so a cut costs at most one more batch;
            # the caller's per-child checks below still decide when the expansion ends.
            if max_seconds is not None and time.perf_counter() - t_start > max_seconds:
                return
            if progress_cb is not None:
                progress_cb(apps, expanded, best_h)
            if should_stop is not None and should_stop():
                return
            if _HAS_CODED_SLICE:
                batch = node.apply_slice_coded(ml, c0, chunk, n, max_terms, threads)
            else:
                batch = [(None, rec) for rec in node.apply_slice(ml, c0, chunk, n, max_terms, threads)]
            for k, (code, rec) in enumerate(batch):
                if rec is None:
                    if code is not None:
                        skips[_REJECT_KIND.get(code, "engine_error")] += 1
                        continue
                    # Older engine: the batch call does not return WHY a move was rejected;
                    # re-apply that one move on this thread to read the engine's reason.
                    # Rejections consume no apply budget, so the result is unchanged and the
                    # counters are the same ones the sequential path reports.
                    sq, i, j, ts, ident, params = ml.get(c0 + k)
                    try:
                        node.apply_one(sq, i, j, ident, list(params), n, term_slot=ts,
                                       max_terms=max_terms)
                        skips["engine_error"] += 1
                    except ValueError as exc:
                        skips[_classify_skip(exc)] += 1
                    continue
                child, ck, fbc, raw_bc, nterms = rec
                yield ml, c0 + k, child, ck, (fbc, raw_bc, nterms)

    while frontier and apps < max_apps and not goal:
        if max_seconds is not None and time.perf_counter() - t_start > max_seconds:
            break
        # Convergence early-stop. The search has no lower bound on the reachable count, so it
        # cannot know it has hit the minimum -- without a goal it otherwise burns the whole
        # budget long after the answer is found (measured: best found in 1-3 steps, budget
        # 30000 applies). `patience_apps` stops after that many applies with no improvement to
        # `best`, the apply-counted analogue of the SFT backend's `sft_patience` (which counts
        # rollout steps). Off by default -- it is a heuristic and can cut a late improvement.
        if patience_apps is not None and apps - apps_at_best >= patience_apps:
            patience_hit = True
            break
        # Cooperative cancel: same "return best-so-far" exit as the max_seconds cut,
        # so a caller (e.g. the web "Stop" button) gets the best node found so far.
        if progress_cb is not None:
            progress_cb(apps, expanded, best_h)
        if should_stop is not None and should_stop():
            stopped = True
            break
        _, _, key = heapq.heappop(frontier)
        node = materialize(key)
        expanded += 1
        d, _k = 0, key
        while parent[_k] is not None:
            _k = parent[_k]
            d += 1
        new_children = []                        # (ck, child, objective) discovered this expansion
        for mv_src, mv_key, child, ck, pre in children_of(node):
            if apps >= max_apps:
                break
            # Check the time limit and cancellation during node expansion as well as between
            # nodes. Large nodes can have thousands of legal moves, so one expansion may exceed
            # the time limit before the outer loop checks it again. These checks run between
            # move applications and should remain lightweight. On stopping, the children
            # found so far are retained and the search returns its best candidate.
            if max_seconds is not None and time.perf_counter() - t_start > max_seconds:
                break
            if progress_cb is not None:
                progress_cb(apps, expanded, best_h)
            if should_stop is not None and should_stop():
                stopped = True
                break
            apps += 1
            # `ck` is the 16-byte dedup key (see `node_key`); the canonical string it digests
            # was never built on this side.
            if ck in visited:
                continue
            visited.add(ck)
            parent[ck] = key
            if mv_src is None:
                move_into[ck] = _pack_move(*mv_key)
            else:
                move_into[ck] = _pack_move(*mv_src.get(mv_key))   # materialised only for kept children
            # Simplicity objective (default: factored bracket-count). With the default
            # objective the worker threads already computed both components, so reuse them —
            # `heuristic` is exactly `(factored_bracket_count, num_terms)`. A custom
            # objective_fn is still called here, on this thread.
            bk = (pre[0], pre[2]) if (pre is not None and objective_fn is None) else objective(child)
            new_children.append((ck, child, bk))
            # Raw-unit goal (target-relative scoring). `goal_brackets` is compared against the
            # objective, which since 2026-07-02 is the FACTORED count -- so with a raw target it
            # can fire while the RAW count is still above the target. `goal_raw_brackets` compares
            # like with like, on every child rather than only on objective-improving ones (the
            # raw count is not monotone in the objective). `apps_to_goal` records the applies
            # spent when it first fires, which is the per-case budget actually needed.
            if goal_raw_brackets is not None:
                raw_bk = pre[1] if pre is not None else child.bracket_count()
                if raw_bk < best_raw_h[0]:
                    best_raw_h = (raw_bk, child.num_terms())
                    best_raw, best_raw_key, best_raw_depth = child, ck, d + 1
                    apps_at_best = apps
                if raw_bk <= goal_raw_brackets:
                    if not bracket_goal:
                        bracket_goal = True
                        apps_to_bracket_goal = apps
                    if goal_raw_terms is None or child.num_terms() <= goal_raw_terms:
                        apps_to_goal = apps
                        best, best_h, best_depth, best_key = child, bk, d + 1, ck
                        goal = True
                        break
            if bk < best_h:
                best, best_h, best_depth, best_key = child, bk, d + 1, ck
                apps_at_best = apps
                if goal_raw_brackets is None and goal_brackets is not None \
                        and best_h[0] <= goal_brackets:
                    goal = True
                    break
        # Order discovered children and push. value_batch_fn evaluates them in ONE call (batched
        # NN forward); else per-child order() (value_fn or hand heuristic). Bracket-gbfs path
        # (both None) is byte-identical to the interleaved version: same order values, same counter
        # sequence (counters still assigned in discovery order).
        if new_children and not goal:
            if value_batch_fn is not None:
                vals = value_batch_fn([c for _, c, _ in new_children])
            elif value_fn is not None:
                vals = [float(value_fn(c)) for _, c, _ in new_children]
            else:
                # order() is objective() here, already evaluated per child above; the
                # objective is a pure function of the (immutable) handle, so reusing it is
                # identical to recomputing and saves a second factored-count pass.
                vals = ([_pack_priority(_f, _t) for _, _, (_f, _t) in new_children]
                        if packed_priority else [b for _, _, b in new_children])
            for (ck, _child, _bk), v in zip(new_children, vals):
                heapq.heappush(frontier, (v, counter, ck))
                counter += 1
        # Every child handle discovered by this expansion dies here; only `best` / `best_raw`
        # keep a live reference past it.
        new_children = None

    # Reconstruct the root->best derivation (first-discovery path). The keys are digests,
    # so the MOVES are read off the parent chain and the expressions are replayed from the
    # root by re-applying them -- `depth` extra applies, outside the apply budget. The replay
    # is checked against `best`; on any mismatch (or a move the engine now rejects) the chain
    # degrades to its endpoints rather than reporting a wrong derivation.
    keys = []
    k = best_key
    while k is not None:
        keys.append(k)
        k = parent[k]
    keys.reverse()
    path_moves = [_unpack_move(move_into[k]) for k in keys[1:]]   # move into each non-root node
    chain = [root_str]
    node_i = root
    try:
        for (sq, i, j, ts, ident, params) in path_moves:
            node_i = node_i.apply(sq, i, j, ident, list(params), n, term_slot=ts,
                                  max_terms=max_terms)
            chain.append(node_i.to_str())
        if chain[-1] != best.to_str():
            raise ValueError("replayed derivation does not end at best")
    except (ValueError, TypeError):
        chain = [root_str] if best is root else [root_str, best.to_str()]

    # Why the loop ended, in priority order. One place covers every exit -- goal, cooperative
    # stop, patience, apply budget, empty frontier, wall clock -- so the reason is recorded,
    # never inferred by the caller.
    if goal:
        reason = "goal"
    elif stopped:
        reason = "cancelled"
    elif patience_hit:
        reason = "patience"
    elif apps >= max_apps:
        reason = "budget"
    elif not frontier:
        reason = "frontier_empty"
    else:
        reason = "time"
    if checkpoint_path is not None:
        _seg.update(apps_end=apps, expanded_end=expanded, stop_reason=reason,
                    ended_at=_CKPT._now(), wall_s=time.perf_counter() - t_start,
                    peak_rss_gb=_CKPT.peak_rss_gb(),
                    best_h_end=list(best_h) if isinstance(best_h, tuple) else best_h)
        # A checkpoint must never be able to LOSE the answer it exists to protect: a failure
        # here is reported in the result and the run still returns normally. (Learned the hard
        # way -- an unhandled struct error in the save threw away a completed 40-minute search.)
        try:
            _CKPT.save(checkpoint_path, n=n, root_str=root_str,
                       priority_format=priority_format,
                       parent=parent, move_into=move_into, frontier=frontier,
                       apps=apps, expanded=expanded, counter=counter,
                       best_key=best_key, best_h=best_h, best_depth=best_depth,
                       best_raw_key=best_raw_key, best_raw_h=best_raw_h,
                       best_raw_depth=best_raw_depth, apps_to_goal=apps_to_goal,
                       apps_to_bracket_goal=apps_to_bracket_goal, apps_at_best=apps_at_best,
                       bracket_goal=bracket_goal, goal=goal, patience_hit=patience_hit,
                       stopped=stopped, best_expr=best.to_str(), path_moves=path_moves,
                       segments=list(_ck["segments"]) + [_seg] if _ck else [_seg])
            _seg["checkpoint_error"] = None
        except Exception as exc:                       # noqa: BLE001 - never fatal
            _seg["checkpoint_error"] = f"{type(exc).__name__}: {exc}"

    return {
        "best": best,
        "best_expr": best.to_str(),
        "best_brackets": best_h[0],                 # objective primary (default FACTORED bracket count; atom count under version D)
        "best_raw_brackets": best.bracket_count(),  # raw count, for comparison w/ historical raw-metric numbers
        "best_terms": best_h[1],
        "depth": best_depth,
        "apps": apps,
        "expanded": expanded,
        "goal_reached": bool(goal),
        "apps_to_goal": apps_to_goal,        # applies spent when the RAW goal first fired
        # Brackets-only leg of the raw goal, recorded even when `goal_raw_terms` tightens the
        # stopping rule -- this is the historical (bracket-count-only) target-relative metric.
        "bracket_goal_reached": bool(bracket_goal),
        "apps_to_bracket_goal": apps_to_bracket_goal,
        "apps_at_best": apps_at_best,        # applies spent when `best` was last improved
        "patience_triggered": patience_hit,  # same field name as the SFT backend reports
        "best_raw_expr": best_raw.to_str(),  # smallest node by RAW count (goal_raw_brackets mode)
        "best_raw_min": best_raw_h[0],
        "best_raw_depth": best_raw_depth,
        "checkpoint_error": _seg.get("checkpoint_error"),
        "stopped": stopped,         # True if cut short by should_stop() (user cancel)
        "stop_reason": reason,      # goal | budget | time | frontier_empty | cancelled | patience
        "term_limit_skips": skips["term_limit"],      # moves rejected by the per-move term cap
        "overflow_skips": skips["overflow"],          # moves rejected by coefficient overflow
        "engine_error_skips": skips["engine_error"],  # degenerate identity / illegal slot
        "is_zero": best.is_zero(),
        "path_exprs": chain,        # canonical strings, root -> best (len = depth+1)
        "path_moves": path_moves,   # moves, len = depth
    }
