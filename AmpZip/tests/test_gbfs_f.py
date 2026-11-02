"""Tests for gbfs_f, the per-term decomposition with the no-inflation guard and the prefactor display.

Run with python tests/test_gbfs_f.py.

Tests 1-4 are pure sympy. Tests 5-8 exercise the full backend through run_all and need the
ampzip_engine extension, and are skipped when it cannot be imported.

At recombination, `gbfs_f` selects the smallest of its four candidates under the
internal `count_brackets` metric. Its selected count is therefore no greater than
those of the input and `all_searched` candidates from that run. For the same searched
pieces, `all_searched` is the `gbfs_e` result, giving the comparison `f <= e` checked
in test (6). This metric includes denominator brackets and differs from the reported
factored numerator bracket count.
"""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import ampzip.simplifier._paths  # noqa: F401,E402
import sympy as sp  # noqa: E402
from ampzip.atoms import ab, sb  # noqa: E402
from ampzip.simplifier.algorithms._post import count_brackets  # noqa: E402
from ampzip.simplifier.algorithms.gbfs_e import GBFSVersionE  # noqa: E402
from ampzip.simplifier.algorithms.gbfs_f import GBFSVersionF  # noqa: E402
from ampzip.simplifier.reconstruct import reconstruct  # noqa: E402

_fails = []


def _check(cond, msg):
    print(f"  [{'ok ' if cond else 'FAIL'}] {msg}")
    if not cond:
        _fails.append(msg)


EQ528 = ("(chain('as',6,[1,2],3)**3)/(ab(6,1)*ab(1,2)*sb(3,4)*sb(4,5)*s(6,1,2)"
         "*chain('as',2,[6,1],5))+(chain('as',4,[5,6],1)**3)/(ab(2,3)*ab(3,4)*sb(5,6)"
         "*sb(6,1)*s(5,6,1)*chain('as',2,[6,1],5))")


# --- Tests 1-4, the recombination guard, with no engine needed ---------------------------- #
def test_recombine_guard():
    """The four-candidate minimum and the two theorems f<=all_searched and f<=input."""
    print("Recombination guard, pure sympy:")
    f = GBFSVersionF()

    # (2) f <= all_searched, the measured counterexample where a per-piece-only guard gives 9
    # and the whole-result minimum gives 6. D is the factor common to both original pieces.
    D = ab(1, 2) * ab(2, 3) * ab(3, 4) * ab(4, 5)
    orig = [ab(5, 6) ** 4, D * ab(1, 4) * ab(1, 5) * ab(1, 6)]
    searched = [D * ab(1, 3), D * ab(1, 4)]
    mix = sp.Add(*[o if count_brackets(o) <= count_brackets(s) else s
                   for o, s in zip(orig, searched)])
    best, extra = f._recombine(mix, orig, searched, sp.Add(*orig))
    cand = extra["guard_candidates"]
    _check(count_brackets(best) == min(cand.values()),
           f"picks the smallest candidate ({count_brackets(best)} == min{cand})")
    _check(count_brackets(best) <= cand["all_searched"],
           f"f <= all_searched theorem ({count_brackets(best)} <= {cand['all_searched']})")
    _check(count_brackets(best) <= cand["input"],
           f"f <= input theorem ({count_brackets(best)} <= {cand['input']})")

    # (4) gbfs_e's _recombine is the identity with an empty guard_extra.
    e = GBFSVersionE()
    x = ab(1, 2) * sb(3, 4)
    eb, eextra = e._recombine(x, [x], [x], x)
    _check(eb == x and eextra == {}, "GBFSVersionE._recombine is identity, guard_extra == {}")


# --- Tests 5 and 7, the display through reconstruct alone --------------------------------- #
def test_display_prefactor():
    """gbfs_f's prefactor spelling and gbfs_e's default two-story spelling on the same expr."""
    print("Display (reconstruct opt-in):")
    # The gbfs_f winner on eq 5.28 is the input re-split (both pieces search-rejected). Build it
    # directly so this test needs no engine.
    import ast
    from ampzip.simplifier.atomize import atomize
    loc = {"ab": ab, "sb": sb, "opaque_den": sp.Function("opaque_den")}
    node = ast.parse(EQ528, mode="eval").body

    def flatten(nd):
        if isinstance(nd, ast.BinOp) and isinstance(nd.op, (ast.Add, ast.Sub)):
            return flatten(nd.left) + [(type(nd.op).__name__, nd.right)]
        return [("Add", nd)]

    pieces = []
    for op, sub in flatten(node):
        e = sp.sympify(atomize(ast.unparse(sub), n_hint=6), locals=loc)
        pieces.append(e if op == "Add" else -e)
    expr = sp.Add(*pieces)

    rec_pf = reconstruct(expr, prefactor=True)
    rec_def = reconstruct(expr, prefactor=False)   # == what gbfs_e shows

    # (8) gbfs_f: single-story leading prefactor, inner sum normalized all-`+` (leading -1/A).
    _check(rec_pf.startswith("-1/chain('as', 2, [1, 6], 5)*("),
           f"prefactor=True starts with -1/A*(  ({rec_pf[:40]}...)")
    _check(rec_pf.count("/chain('as', 2, [1, 6], 5)") == 1,
           "prefactor=True has the pole exactly once (single-story)")
    _check(" - chain(" not in rec_pf.split("*(", 1)[1],
           "prefactor=True inner sum is all-'+' (no inner minus)")

    # (7) The same expr with prefactor=False, the default of every other backend, keeps the
    # two-story (...)/A spelling. This assertion separates f from e, since a
    # startswith('-1/chain...') check alone would pass for neither here.
    _check(rec_def.endswith(")/chain('as', 2, [1, 6], 5)"),
           f"prefactor=False keeps two-story (...)/A  (...{rec_def[-30:]})")
    _check(not rec_def.startswith("-1/chain"),
           "prefactor=False keeps the plain spelling")


# --- Tests 5,6: the full backend (needs the engine) --------------------------------------- #
def test_backend():
    print("Full backend (needs ampzip_engine):")
    try:
        from ampzip.search.gbfs import simplify as _  # noqa: F401
    except Exception as ex:  # noqa: BLE001
        print(f"  [skip] ampzip_engine unavailable, {type(ex).__name__}, skipping 5 and 6")
        return
    from ampzip.simplifier.runner import run_all, list_algorithms
    if "gbfs_f" not in list_algorithms():
        # The end-to-end backend test needs gbfs_f registered with the runner.
        print("  [skip] gbfs_f is not registered, skipping 5 and 6")
        return
    from ampzip.simplifier.atomize import atomize
    loc = {"ab": ab, "sb": sb, "opaque_den": sp.Function("opaque_den")}
    expr = sp.sympify(atomize(EQ528, n_hint=6), locals=loc)

    r = run_all(["gbfs_f"], expr, 6, raw_input=EQ528,
                gbfs_max_apps=3000, gbfs_max_seconds=90, sample_timeout=90)[0]
    # The theorem is stated in the backend's own count of the whole expression, denominator
    # included, which the runner keeps in extra["backend_bracket_count_*"]. The reported
    # bracket_count_* fields are the factored numerator count.
    bf, bs = r.extra["backend_bracket_count_final"], r.extra["backend_bracket_count_start"]
    _check(bf == 48, f"eq 5.28 gbfs_f == 48 (got {bf})")
    _check(r.extra["soundness_verified"] is True, "sound")
    _check(r.extra["split_mode"] == "raw_terms", "split_mode == raw_terms (site 3b threads raw_input)")
    _check(r.extra["raw_split_skipped"] is None, "raw_split_skipped is None (not degraded)")
    _check(r.extra["guard_candidates"] == {"mix": 48, "split_baseline": 48,
                                           "all_searched": 72, "input": 68},
           f"guard_candidates exact (got {r.extra.get('guard_candidates')})")
    _check(r.solved is False and r.n_steps == 0, "solved False, n_steps 0 (no fake progress)")
    _check(all(p.get("orig_brackets") == 26 and p.get("searched_brackets") == 38
               for p in r.extra["per_term"]), "per_term orig 26 / searched 38 (engine expands chain**3)")

    # (6) For this fixture, compare `gbfs_f` with `gbfs_e` and with the input using
    # the internal count. The assertions below require f <= e and f <= start.
    e = run_all(["gbfs_e"], expr, 6, raw_input=EQ528,
                gbfs_max_apps=3000, gbfs_max_seconds=90, sample_timeout=90)[0]
    ef = e.extra["backend_bracket_count_final"]
    _check(bf <= ef, f"f <= e ({bf} <= {ef})")
    _check(bf <= bs, f"f <= start ({bf} <= {bs})")
    # (5,7) The gbfs_e display is unchanged, the reconstructed string still two-story.
    _check(e.extra.get("reconstructed_min", "").endswith(")/chain('as', 2, [1, 6], 5)"),
           "gbfs_e display still two-story (a/b/d/e untouched)")


def main():
    test_recombine_guard()
    test_display_prefactor()
    test_backend()
    print()
    if _fails:
        print(f"FAILED ({len(_fails)}): {_fails}")
        return 1
    print("ALL GBFS_F TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
