"""Training-free search over the fast Rust engine (ampzip_engine).

`actions` enumerates legal moves on an `Expr` handle; `gbfs` runs a best-first
search ranked by bracket count. No neural model — the heuristic is hand-written
(`Expr.factored_bracket_count`), so nothing here needs training.
"""
