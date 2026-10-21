"""Engine limits made explicit on 2026-10-21: the term cap bounds the input expansion and
single-term rewrites too, and the batch apply path reports why a move was rejected."""
import os, sys
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
import ampzip_engine as E
from ampzip.search.gbfs import simplify

SUM = "(ab(1,2)+ab(1,3)+ab(2,3)+ab(1,4))"


def _skip():
    return not hasattr(E.Expr, "apply_slice_coded")


def test_parse_cap():
    if _skip():
        print("  (engine without apply_slice_coded: skipped)"); return
    E.Expr(SUM + "**3", max_terms=50000)
    try:
        E.Expr(SUM + "**9", max_terms=50)
    except ValueError as e:
        assert "term cap" in str(e)
    else:
        raise AssertionError("input expansion was not bounded")


def test_term_slot_cap():
    if _skip():
        return
    x = E.Expr("ab(1,2)**6*sb(3,4) + ab(1,3)*sb(2,4)")
    # a per-term Schouten rewrite of ab(1,2)**6 in term 0 with a 2-term cap must be skipped
    try:
        x.apply(False, 1, 2, "schouten", [3, 4], 4, term_slot=1, max_terms=2)
    except ValueError as e:
        assert "AMPZIP_MAX_TERMS" in str(e) or "term" in str(e)
    else:
        raise AssertionError("single-term rewrite was not bounded by max_terms")
    y = x.apply(False, 1, 2, "schouten", [3, 4], 4, term_slot=1, max_terms=50000)
    assert y.num_terms() >= 2


def test_coded_slice_matches_sequential():
    if _skip():
        return
    x = E.Expr("ab(1,2)**6*sb(3,4)/ab(1,4) + ab(1,3)*sb(2,4)/ab(1,4)")
    ml = x.legal_move_list(4)
    coded = x.apply_slice_coded(ml, 0, 0, 4, 3, 4)
    plain = x.apply_slice(ml, 0, 0, 4, 3, 4)
    assert len(coded) == len(plain) == len(ml)
    for k, ((code, rec), prec) in enumerate(zip(coded, plain)):
        assert (rec is None) == (prec is None) and (code == 0) == (rec is not None)
        if rec is None:
            sq, i, j, ts, ident, params = ml.get(k)
            try:
                x.apply_one(sq, i, j, ident, list(params), 4, term_slot=ts, max_terms=3)
            except ValueError as e:
                want = 2 if "AMPZIP_MAX_TERMS" in str(e) else (3 if "overflow" in str(e) else 1)
                assert code == want, (k, code, str(e))
            else:
                raise AssertionError("batch rejected a move the sequential apply accepts")
    assert any(code == 2 for code, _ in coded)
    r1 = simplify(x.to_str(), 4, max_apps=300, max_terms=3, threads=1)
    r4 = simplify(x.to_str(), 4, max_apps=300, max_terms=3, threads=4)
    assert r1["term_limit_skips"] == r4["term_limit_skips"] > 0
    assert all(r1[k] == r4[k] for k in r1 if k != "best")


def main():
    for fn in (test_parse_cap, test_term_slot_cap, test_coded_slice_matches_sequential):
        fn(); print(f"  ok {fn.__name__}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
