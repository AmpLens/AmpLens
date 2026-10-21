"""Save and restore the full interior state of a `gbfs.simplify` run.

Why not pickle: the interior of a long run is a few hundred million dict entries. Pickling
those dicts is slow, and it allocates a second copy while dumping -- on a run whose whole
problem is memory, that is the one thing that must not happen. So the state goes out as flat
binary sections written straight from the live containers, and comes back by rebuilding them
in one pass.

The three per-node containers (`visited`, `parent`, `move_into`) share one key set and one
insertion order, so they serialise as a single fixed-size record per node (version 2):

    NODES     n_nodes x (key 16 | parent_key 16 | packed_move u64, or u64 lo | u64 hi)
    FRONTIER  n_frontier x (priority u64 | u64 lo,hi | float64  ;  counter int64 | key 16)

The header says which widths were used (`move_bytes` 8 or 16, `priority_format` "u64" /
"u128" / "f64") and which encodings (`move_layout` 2, `priority_shift` 32), so a reader never
guesses. Moves are packed with `gbfs._pack_move` (layout 2, unbounded term-slot field at the
top); the writer widens the record to 16 bytes only when a run actually produced a value that
does not fit 63 bits. A packed default-objective priority is stored as an exact integer;
`f64` is used only for the float priorities a `value_fn` produces.

Version 1 files (layout-1 moves in an int64, float64 priorities packed with a 20-bit shift,
optional DEPTH section) are still read; `load()` fills in `move_layout=1`,
`priority_format="f64"`, `priority_shift=20` so `gbfs.simplify` can migrate them in place
on resume. A resumed v1 file is written back as v2 at its next save.

The JSON header is also the run LOG. Besides the scalars needed to resume, it carries the
best node's derivation (`best_expr` and its `path_moves`, so the answer survives even if a
later segment never improves on it) and a `segments` list -- one record per launch, appended
on every resume, holding when it ran, on how many threads, under what budget, how many
applies and expansions it added, what the best was on entry and exit, why it stopped, and its
peak RSS. A checkpoint therefore reads back as the full history of the search, not just its
current state.

`handle_cache` is deliberately NOT saved: it only shortens replays, every node it yields is
checked against its key, and it refills from the root the same way on resume -- so dropping
it costs a little time and cannot change a result.

A checkpoint stores the frontier and counters, but not the remaining moves of a partially
expanded node. Continuing the search in the same order as an uninterrupted run requires saving
after a complete node expansion and keeping the ordering and move settings unchanged. A
checkpoint saved during expansion skips that node's remaining moves on resume.

"""
from __future__ import annotations

import json
import os
import platform
import resource
import struct
import time

_MAGIC = b"GBFSCKPT"
_VERSION = 2
_READABLE_VERSIONS = (1, 2)
_ZERO16 = b"\x00" * 16
_MOVE_LAYOUT = 2
_PRIORITY_SHIFT = 32
_U64 = 1 << 64


def new_segment(**kw):
    """One launch's log record; `simplify` fills the exit fields before saving."""
    seg = {"started_at": _now(), "host": platform.node(), "pid": os.getpid(),
           "cpu_affinity": _affinity(), "threads": None, "batch_chunk": None,
           "max_apps": None, "max_seconds": None, "rss_stop_gb": None,
           "apps_start": 0, "apps_end": None, "expanded_start": 0, "expanded_end": None,
           "best_h_start": None, "best_h_end": None, "stop_reason": None,
           "ended_at": None, "wall_s": None, "peak_rss_gb": None}
    seg.update(kw)
    return seg


def _now():
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _affinity():
    """Cores this process may run on -- the honest answer to "how many cores did it get"."""
    try:
        cores = sorted(os.sched_getaffinity(0))
    except AttributeError:
        return None
    return {"n_cores": len(cores), "cores": _runs(cores)}


def _runs(xs):
    out, lo = [], None
    for a, b in zip(xs, xs[1:] + [None]):
        if lo is None:
            lo = a
        if b != a + 1:
            out.append(str(lo) if lo == a else f"{lo}-{a}")
            lo = None
    return ",".join(out)


def peak_rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1 << 20)


def save(path, *, n, root_str, parent, move_into, frontier,
         apps, expanded, counter, best_key, best_h, best_depth,
         best_raw_key, best_raw_h, best_raw_depth,
         apps_to_goal, apps_to_bracket_goal, apps_at_best,
         bracket_goal, goal, patience_hit, stopped,
         priority_format="packed", segments=(), best_expr=None, path_moves=None, extra=None,
         low_memory=None, depth=None):
    """Write the run's interior to `path` (atomically, via a .part rename).

    `priority_format` is "packed" when the frontier holds the default objective's packed
    ints (stored exactly, as u64 or u128) and "f64" when it holds `value_fn` floats.
    `low_memory` / `depth` are accepted for old callers and ignored (there is no depth map)."""
    max_move = max((mv for mv in move_into.values() if mv is not None), default=0)
    if max_move < 0:
        raise ValueError("negative packed move")
    move_bytes = 8 if max_move < (1 << 63) else 16
    if max_move >= (1 << 128):
        raise ValueError("packed move does not fit 128 bits")
    if priority_format == "tuple":
        raise ValueError("gbfs checkpointing requires the default objective or a value_fn "
                         "(a custom objective's priorities are not stored)")
    if priority_format == "f64":
        prio_fmt, prio_bytes = "f64", 8
    else:
        max_prio = max((int(v) for v, _, _ in frontier), default=0)
        if max_prio >= (1 << 128):
            raise ValueError("packed priority does not fit 128 bits")
        prio_fmt, prio_bytes = ("u64", 8) if max_prio < _U64 else ("u128", 16)
    header = {
        "version": _VERSION, "n": n, "root_str": root_str, "low_memory": True,
        "move_layout": _MOVE_LAYOUT, "move_bytes": move_bytes,
        "priority_format": prio_fmt, "priority_shift": _PRIORITY_SHIFT,
        "apps": apps, "expanded": expanded, "counter": counter,
        "best_key": best_key.hex(), "best_h": list(best_h) if isinstance(best_h, tuple) else best_h,
        "best_depth": best_depth,
        "best_raw_key": best_raw_key.hex(),
        "best_raw_h": list(best_raw_h) if isinstance(best_raw_h, tuple) else best_raw_h,
        "best_raw_depth": best_raw_depth,
        "apps_to_goal": apps_to_goal, "apps_to_bracket_goal": apps_to_bracket_goal,
        "apps_at_best": apps_at_best, "bracket_goal": bool(bracket_goal),
        "goal": bool(goal), "patience_hit": bool(patience_hit), "stopped": bool(stopped),
        "n_nodes": len(parent), "n_frontier": len(frontier),
        "has_depth": False,
        # --- log / provenance, not needed to resume but the reason a checkpoint is readable
        "best_expr": best_expr,
        "path_moves": path_moves,          # packed ints, root -> best
        "segments": list(segments),
        "extra": dict(extra or {}),
        "saved_at": _now(),
    }
    hb = json.dumps(header).encode()
    tmp = path + ".part"
    with open(tmp, "wb", buffering=1 << 22) as f:
        f.write(_MAGIC)
        f.write(struct.pack("<Q", len(hb)))
        f.write(hb)
        if move_bytes == 8:
            pack = struct.Struct("<Q").pack
            for k, pk in parent.items():                  # insertion order == discovery order
                mv = move_into[k]
                f.write(k)
                f.write(_ZERO16 if pk is None else pk)
                f.write(pack(0 if mv is None else mv))
        else:
            pack = struct.Struct("<QQ").pack
            for k, pk in parent.items():
                mv = 0 if move_into[k] is None else move_into[k]
                f.write(k)
                f.write(_ZERO16 if pk is None else pk)
                f.write(pack(mv & (_U64 - 1), mv >> 64))
        if prio_fmt == "f64":
            packf = struct.Struct("<dq").pack
            for v, c, k in frontier:                      # list order preserves the heap invariant
                f.write(packf(float(v), c))
                f.write(k)
        elif prio_fmt == "u64":
            packf = struct.Struct("<Qq").pack
            for v, c, k in frontier:
                f.write(packf(int(v), c))
                f.write(k)
        else:
            packf = struct.Struct("<QQq").pack
            for v, c, k in frontier:
                v = int(v)
                f.write(packf(v & (_U64 - 1), v >> 64, c))
                f.write(k)
    os.replace(tmp, path)
    return os.path.getsize(path)


def formats(header):
    """Normalise a header (v1 or v2) and return `(header, node_struct, frontier_struct)`.

    v1 files predate the format fields; they are filled in here so every reader (this one,
    `paper/scripts/memory_efficient_checkpoint.py`, `gbfs.simplify`'s migration) sees the
    same keys: `move_layout`, `move_bytes`, `priority_format`, `priority_shift`."""
    if header["version"] not in _READABLE_VERSIONS:
        raise ValueError(f"checkpoint version {header['version']} not in {_READABLE_VERSIONS}")
    if header["version"] == 1:
        header.setdefault("move_layout", 1)
        header.setdefault("move_bytes", 8)
        header.setdefault("priority_format", "f64")
        header.setdefault("priority_shift", 20)
    node = struct.Struct("<16s16sq" if header["version"] == 1
                         else ("<16s16sQ" if header["move_bytes"] == 8 else "<16s16sQQ"))
    frontier = struct.Struct({"f64": "<dq16s", "u64": "<Qq16s", "u128": "<QQq16s"}[header["priority_format"]])
    return header, node, frontier


def load(path):
    """Rebuild the containers written by `save` (version 1 or 2). Returns
    (header, parent, move_into, visited, depth, frontier); `depth` is None unless a v1 file
    carried a DEPTH section (it is loaded for completeness and ignored by the search)."""
    with open(path, "rb", buffering=1 << 22) as f:
        if f.read(len(_MAGIC)) != _MAGIC:
            raise ValueError(f"{path}: not a gbfs checkpoint")
        (hlen,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(hlen))
        header, node_s, front_s = formats(header)
        n_nodes, n_frontier = header["n_nodes"], header["n_frontier"]
        wide_move = node_s.size == 48

        parent, move_into, visited = {}, {}, set()
        blob = f.read(n_nodes * node_s.size)
        if len(blob) != n_nodes * node_s.size:
            raise ValueError(f"{path}: truncated NODES section")
        for rec in node_s.iter_unpack(blob):
            k, pk = rec[0], rec[1]
            mv = (rec[2] | (rec[3] << 64)) if wide_move else rec[2]
            parent[k] = None if pk == _ZERO16 else pk
            move_into[k] = None if parent[k] is None else mv
            visited.add(k)
        del blob

        depth = None
        if header.get("has_depth"):
            db = f.read(n_nodes * 4)
            depth = {k: v for k, v in zip(parent, struct.unpack(f"<{n_nodes}I", db))}
            del db

        frontier = []
        fb = f.read(n_frontier * front_s.size)
        if len(fb) != n_frontier * front_s.size:
            raise ValueError(f"{path}: truncated FRONTIER section")
        fmt = header["priority_format"]
        for rec in front_s.iter_unpack(fb):
            if fmt == "u128":
                frontier.append((rec[0] | (rec[1] << 64), rec[2], rec[3]))
            else:
                frontier.append((rec[0], rec[1], rec[2]))
        del fb
    return header, parent, move_into, visited, depth, frontier
