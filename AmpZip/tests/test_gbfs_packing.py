"""Move/priority packing: full-range round trip, v1 checkpoint files still load and resume."""
import os, sys
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
import itertools, random
from ampzip.search.gbfs import (_pack_move, _unpack_move, _unpack_move_v1, _pack_priority,
                                _repack_priority_v1, simplify)
from ampzip.search import gbfs_checkpoint as C

V1_FILES = [os.path.join(_ROOT, "data/az_v2/gbfs_esc/4pt_k15_c250_i138_raw.ckpt"),
            os.path.join(_ROOT, "data/az_v2/gbfs_esc/4pt_k20_c250_i212.ckpt")]


def test_round_trip():
    rng = random.Random(7)
    for _ in range(5000):
        n = rng.randint(4, 17)
        i, j = rng.randint(1, n), rng.randint(1, n)
        ts = rng.choice([0, rng.randint(1, 100000), rng.randint(1, 2**31)])
        ident = rng.choice(["schouten", "momentum", "momentum_sq"])
        params = [rng.randint(1, n) for _ in range(rng.randint(0, min(8, n - 2)))]
        mv = (bool(rng.randint(0, 1)), i, j, ts, ident, params)
        assert _unpack_move(_pack_move(*mv)) == mv, mv
    # exhaustive over the 6-point menu shape
    for sq, i, j, ident in itertools.product((False, True), range(1, 7), range(1, 7), ("schouten", "momentum", "momentum_sq")):
        for params in ([], [1, 2], [2, 3, 4, 5]):
            for ts in (0, 1, 33, 4096, 70000):
                assert _unpack_move(_pack_move(sq, i, j, ts, ident, params)) == (sq, i, j, ts, ident, params)


def test_out_of_range_raises():
    for bad in [(True, 64, 1, 0, "schouten", []), (True, 1, 2, -1, "schouten", []),
                (True, 1, 2, 0, "schouten", [64]), (True, 1, 2, 0, "schouten", list(range(1, 17)))]:
        try:
            _pack_move(*bad)
        except ValueError:
            continue
        raise AssertionError(f"packed an out-of-range move: {bad}")


def test_priority_monotone():
    vals = [(f, t) for f in (0, 1, 7, 1000) for t in (0, 1, 50000, 2**31)]
    packed = [_pack_priority(f, t) for f, t in vals]
    assert sorted(range(len(vals)), key=lambda k: vals[k]) == sorted(range(len(vals)), key=lambda k: packed[k])
    assert _repack_priority_v1((5 << 20) | 33) == _pack_priority(5, 33)


def test_v1_layout_decoder():
    # layout 1: sq 1 | i 4 | j 4 | ts 12 | ident 2 | nparams 3 | params 4x3
    v = 1 | (3 << 1) | (4 << 5) | (7 << 9) | (1 << 21) | (2 << 23) | (4 << 26) | (1 << 29)
    assert _unpack_move_v1(v) == (True, 3, 4, 7, "momentum", [4, 1])


def test_v1_files_load_and_resume():
    for path in V1_FILES:
        if not os.path.exists(path):
            continue
        hdr, parent, mi, vis, dep, fr = C.load(path)
        assert hdr["version"] == 1 and hdr["move_layout"] == 1 and hdr["priority_format"] == "f64"
        vf = (lambda h: h.bracket_count()) if path.endswith("_raw.ckpt") else None
        res = simplify(hdr["root_str"], hdr["n"], max_apps=hdr["apps"], value_fn=vf, resume_from=path)
        assert res["best_expr"] == hdr["best_expr"], path
        assert res["apps"] == hdr["apps"]


def main():
    for fn in (test_round_trip, test_out_of_range_raises, test_priority_monotone,
               test_v1_layout_decoder, test_v1_files_load_and_resume):
        fn()
        print(f"  ok {fn.__name__}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
