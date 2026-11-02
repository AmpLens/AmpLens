"""Best-first search over the engine `ampzip_engine`.

`actions` enumerates legal moves on an `Expr` handle and `gbfs` runs a best-first search
ranked by bracket count. The heuristic is `Expr.factored_bracket_count`.
"""
