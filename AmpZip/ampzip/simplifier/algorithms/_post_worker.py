"""The helper process for the bounded sympy post-processing of `_post.post_process`.

sympy cannot be interrupted from another thread and `signal.alarm` only works in a process's
main thread, so the only way to put a wall-clock bound on `sp.cancel` / `sp.simplify` that
holds for both the CLI (main thread) and the web service (thread pool) is to run them in a
child process that can be killed. This module is that child: it reads one JSON request per
line on stdin and writes one JSON reply per line on stdout.

    {"op": "simplify", "expr": "<sympy str>", "mode": "cancel"}  -> {"expr": "...", "count": N}
    {"op": "count", "expr": "<sympy str>"}                        -> {"expr": "...", "count": N}

`count` is the factored bracket count of `_post.count_brackets`. Errors come back as
{"error": "..."}, which the parent treats like an in-process failure, the raw string with
count -1.
"""
from __future__ import annotations

import json
import sys


def _main() -> None:
    import sympy as sp
    from ampzip.atoms import ab, sb
    from ampzip.simplifier.algorithms._post import count_brackets, _apply_mode
    local = {"ab": ab, "sb": sb, "opaque_den": sp.Function("opaque_den")}
    out = sys.stdout
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            expr = sp.sympify(req["expr"], locals=local)
            if req.get("op") == "simplify":
                expr = _apply_mode(expr, req.get("mode", "cancel"))
            reply = {"expr": str(expr), "count": count_brackets(expr)}
        except Exception as exc:  # noqa: BLE001  reported to the parent
            reply = {"error": f"{type(exc).__name__}: {exc}"}
        out.write(json.dumps(reply) + "\n")
        out.flush()


if __name__ == "__main__":
    _main()
