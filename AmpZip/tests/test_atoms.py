"""The package's own bracket atoms stand in for the upstream classes exactly, and the gbfs
path runs with the upstream packages unimportable.

Run:  python tests/test_atoms.py   (or via tests/run_tests.py)
"""
from __future__ import annotations

import glob
import json
import os
import pickle
import random
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

import sympy as sp  # noqa: E402

from ampzip.atoms import ab, sb, canonical_order, ATOMS  # noqa: E402
from ampzip.simplifier.parse import parse_expr  # noqa: E402

UPSTREAM = ("environment", "model", "training", "eval_sft_4pt", "eval_sft_5pt", "eval_sft_6pt",
            "action_space_4pt", "action_space_5pt", "action_space_6pt", "train_sft_transformer",
            "torch")   # torch is an optional extra: only the learned backends need it


def _corpus(limit=None):
    """Stored expression strings: suite records, YM outputs, and the three largest YM forms."""
    out = []
    for f in sorted(glob.glob(os.path.join(_ROOT, "data", "az_v2", "v2_*pt_k*_c*.pkl"))):
        with open(f, "rb") as fh:
            for r in pickle.load(fh)["records"]:
                out += [r["scrambled"], r["simple"]]
    ym = os.path.join(_ROOT, "output", "ym5pt_sampleE", "results.jsonl")
    if os.path.exists(ym):
        out += [json.loads(l)["best_expr"] for l in open(ym)]
    if limit:
        random.Random(0).shuffle(out)
        out = out[:limit]
    return out


def test_registry():
    assert ATOMS == {"ab": ab, "sb": sb}
    assert issubclass(ab, sp.Function) and issubclass(sb, sp.Function)


def test_canonical_order():
    e = parse_expr("ab(2,1)*sb(3,1) + ab(1,2)*sb(1,3) - 2*ab(4,3)**2", {"ab": ab, "sb": sb})
    got = canonical_order(e)
    want = parse_expr("-ab(1,2)*(-sb(1,3)) + ab(1,2)*sb(1,3) - 2*ab(3,4)**2", {"ab": ab, "sb": sb})
    assert got == want, (got, want)
    # a one-argument function is not itself reordered, but the brackets inside it are
    od = sp.Function("opaque_den")
    e2 = ab(2, 1) * od(ab(1, 2) * sb(2, 3) + ab(1, 4) * sb(4, 3))
    assert canonical_order(e2) == -ab(1, 2) * od(ab(1, 2) * sb(2, 3) - ab(1, 4) * sb(3, 4))


def test_latex_matches_upstream_spelling():
    assert sp.latex(ab(1, 2)) == r"\langle 1 2 \rangle"
    assert sp.latex(sb(3, 4) ** 2) == r"\left[ 3 4 \right]^{2}"


def test_parity_with_upstream(limit=4000):
    """Same str(), same canonical order, same LaTeX as the upstream classes, on stored strings.
    Skipped when the upstream package is not present (it is not shipped)."""
    try:
        import ampzip.simplifier._paths  # noqa: F401
        from environment.bracket_env import ab as uab, sb as usb
        from environment.utils import reorder_expr
    except ImportError:
        print("       (upstream package absent; parity check skipped)")
        return
    ours, theirs = {"ab": ab, "sb": sb}, {"ab": uab, "sb": usb}
    bad = 0
    for s in _corpus(limit):
        a = parse_expr(s, ours, lenient=True)
        b = parse_expr(s, theirs, lenient=True)
        if str(a) != str(b) or sp.latex(a) != sp.latex(b) \
                or str(canonical_order(a)) != str(reorder_expr(b)):
            bad += 1
    assert bad == 0, f"{bad} parity mismatches"


_CHILD = r"""
import sys
UPSTREAM = %r
class _Block:
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in UPSTREAM:
            raise ImportError("blocked for the test: " + name)
        return None
sys.meta_path.insert(0, _Block())
sys.path.insert(0, %r)
import sympy as sp
from ampzip.atoms import ab, sb
from ampzip.simplifier.parse import parse_expr
from ampzip.simplifier.validate import validate
from ampzip.simplifier.atomize import atomize
from ampzip.simplifier.reconstruct import reconstruct
from ampzip.simplifier.runner import run_all
raw = "(ab(1,2)*sb(1,2) + ab(1,4)*sb(1,4))*ab(1,3)*sb(2,4)/(ab(1,2)**2*sb(1,2)**2)"
atomized = atomize("ab(1,3)*sb(2,4)*(s(1,2)+s(1,4))/s(1,2)**2")
r = validate(atomized); assert r.ok, r.issues
e = parse_expr(r.canonical_expr_str, {"ab": ab, "sb": sb, "opaque_den": sp.Function("opaque_den")}, lenient=True)
(res,) = run_all(["model_free_gbfs"], e, 4, gbfs_max_apps=2000, gbfs_max_seconds=15)
assert res.solved and res.bracket_count_min < res.bracket_count_start, res
assert "s(" in reconstruct(res.final_expr_str)
assert validate(raw).ok
from ampzip.simplifier.runner import available_algorithms, list_algorithms
avail = available_algorithms()
assert set(avail) == {a for a in list_algorithms() if a.startswith("gbfs") or a == "model_free_gbfs"}, avail
(bad,) = run_all(["sft_greedy"], e, 4)        # requested but unavailable: an error result, not a crash
assert not bad.solved and "error" in (bad.extra or {}), bad
assert not any(m.split(".")[0] in UPSTREAM for m in sys.modules), "an upstream module was imported"
print("UPSTREAM-FREE OK", res.bracket_count_start, "->", res.bracket_count_min)
"""


def test_gbfs_path_without_upstream():
    """validate -> atomize -> gbfs -> reconstruct in a fresh interpreter where the upstream
    packages (environment / model / training / eval_sft_*) and torch cannot be imported; the
    learned backends report themselves unavailable and a request for one yields an error
    result rather than an exception."""
    import subprocess
    code = _CHILD % (UPSTREAM, _ROOT)
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300,
                       env={**os.environ, "PYTHONPATH": _ROOT})
    assert r.returncode == 0 and "UPSTREAM-FREE OK" in r.stdout, (r.stdout[-800:], r.stderr[-1500:])


def main() -> int:
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"[ok  ] {name}")
            except AssertionError as e:
                fails += 1
                print(f"[FAIL] {name}: {e}")
    print("ALL ATOM TESTS PASSED" if not fails else f"{fails} atom test(s) FAILED")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
