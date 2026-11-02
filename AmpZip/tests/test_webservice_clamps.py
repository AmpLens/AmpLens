"""Every server-side adjustment of a request is reported back, and n_point accepts 'auto'."""
import os, sys
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)


def test_clamps():
    from fastapi.testclient import TestClient
    from ampzip.webservice import app as A
    A.app.dependency_overrides[A.require_email] = lambda: "test@example.com"
    c = TestClient(A.app)
    lim = c.get("/api/limits").json()
    assert "gbfs_threads" in lim and "post_timeout_s" in lim
    r = c.post("/api/jobs", json={"expr": "ab(1,2)*ab(3,4)/ab(1,4) - ab(1,3)*ab(2,4)/ab(1,4)",
                                  "algorithms": ["model_free_gbfs", "nope"], "n_point": "auto",
                                  "gbfs_max_seconds": lim["gbfs_max_seconds"] + 1000,
                                  "gbfs_threads": lim["gbfs_threads"] + 7,
                                  "gbfs_max_apps": 100, "gbfs_den_aware": True, "gbfs_ordering": "raw"})
    assert r.status_code == 202, r.text
    cl = r.json()["clamped"]
    assert cl["algorithms_dropped_unknown"] == ["nope"]
    assert cl["gbfs_max_seconds"][1] == lim["gbfs_max_seconds"]
    assert cl["gbfs_threads"][1] == lim["gbfs_threads"]
    assert cl["gbfs_den_aware"] == [True, False]
    r = c.post("/api/jobs", json={"expr": "ab(1,2)*sb(3,4)/2 + ab(1,3)*sb(2,4)", "algorithms": ["model_free_gbfs"]})
    assert r.status_code == 422 and "LIMIT_ENGINE_COEFF" in r.text
    # The old "0 = engine default" spelling stays the engine default.
    from ampzip.simplifier.limits import DEFAULT_GBFS_MAX_TERMS
    r = c.post("/api/jobs", json={"expr": "ab(1,2)*sb(3,4) + ab(1,3)*sb(2,4)", "algorithms": ["model_free_gbfs"],
                                  "gbfs_max_apps": 100, "gbfs_max_terms": 0})
    assert r.status_code == 202, r.text
    assert r.json()["clamped"]["gbfs_max_terms"] == [0, DEFAULT_GBFS_MAX_TERMS]


def main():
    test_clamps(); print("  ok test_clamps")
    return 0


if __name__ == "__main__":
    sys.exit(main())
