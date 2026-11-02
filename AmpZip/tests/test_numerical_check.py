"""/api/numerical_check and its rendering on the page.

Run: python tests/test_numerical_check.py
The API tests need the web dependencies, and the page test uses Node when available.
"""
from __future__ import annotations

import importlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
try:
    from fastapi.testclient import TestClient
    HAVE_WEB = True
except ImportError:
    HAVE_WEB = False


@unittest.skipUnless(HAVE_WEB, "web test dependencies unavailable")
class NumericalCheckAPI(unittest.TestCase):
    """Verdicts of /api/numerical_check for equal, unequal and undefined inputs."""

    def setUp(self):
        self.web = importlib.import_module("ampzip.webservice.app")
        self._patch(patch.object(self.web.LIMITER, "allow", return_value=True))
        self._patch(patch.object(self.web.audit, "audit_reject"))
        self._patch(patch.dict(self.web.app.dependency_overrides, {
            self.web.require_email: lambda: "test@example.invalid",
        }))
        self.client = TestClient(self.web.app)
        self.addCleanup(self.client.close)

    def _patch(self, patcher):
        patcher.start()
        self.addCleanup(patcher.stop)

    def post(self, **overrides):
        body = {"A": "ab(1,2)", "B": "ab(1,2)", "n_point": 4}
        body.update(overrides)
        return self.client.post("/api/numerical_check", json=body)

    def assert_status(self, response, status, equiv):
        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        self.assertEqual(result["check_status"], status, result)
        self.assertIs(result["equiv"], equiv)
        self.assertIn("finite-field", result["method"])
        json.dumps(result, allow_nan=False)

    def test_matched(self):
        for A, B, n in [
            ("ab(1,2)", "ab(1,2)", 4),
            ("ab(1,2)*sb(2,3)", "-ab(1,4)*sb(4,3)", 4),                 # momentum conservation
            ("ab(1,2)**3/(ab(2,3)*ab(3,4)*ab(4,1))",
             "-ab(1,2)**3/(ab(2,3)*ab(3,4)*ab(1,4))", 4),               # antisymmetry
            ("1/s(1,2)", "1/(ab(1,2)*sb(2,1))", 4),
            ("1/s(1,2,3)", "1/s(4,5)", 5),
            ("ab(1,2)**2+ab(3,4)**2", "ab(3,4)**2+ab(1,2)**2", 4),
            ("I*ab(1,2)", "I*ab(1,2)", 4),
        ]:
            with self.subTest(A=A, B=B):
                self.assert_status(self.post(A=A, B=B, n_point=n), "matched", True)

    def test_mismatch(self):
        for A, B, n in [
            ("ab(1,2)*sb(2,3)", "ab(1,4)*sb(4,3)", 4),
            ("I*ab(1,2)", "ab(1,2)", 4),
            ("I*ab(1,2)", "-I*ab(1,2)", 4),
            ("1/s(1,2,3)", "1/s(1,2)", 5),
        ]:
            with self.subTest(A=A, B=B):
                self.assert_status(self.post(A=A, B=B, n_point=n), "mismatch", False)

    def test_vanishing_denominator_is_inconclusive(self):
        # <1|2+4|3] vanishes on shell, so the expression has no value.
        bad = "1/(ab(1,2)*sb(2,3)+ab(1,4)*sb(4,3))"
        for B in ("1", bad):
            with self.subTest(B=B):
                self.assert_status(self.post(A=bad, B=B), "inconclusive", None)

    def test_program_errors_remain_errors(self):
        with patch.object(self.web, "_finite_field_compare",
                          side_effect=ValueError("unexpected backend failure")):
            response = self.post()
        self.assertEqual(response.status_code, 500)
        self.assertNotIn("check_status", response.json())
        self.assertIn("unexpected backend failure", response.json()["detail"])

    def test_input_errors_do_not_reach_the_checker(self):
        with patch.object(self.web, "_finite_field_compare") as checker:
            response = self.post(A="unknown(1,2)")
        self.assertEqual(response.status_code, 400, response.text)
        self.assertNotIn("check_status", response.json())
        checker.assert_not_called()


@unittest.skipUnless(HAVE_WEB, "web test dependencies unavailable")
class NumericalCheckWithoutResearchPackages(unittest.TestCase):
    """A fresh interpreter blocks the research packages, as in an installation with only the
    ampzip wheel, and posts one comparison."""

    def test_endpoint_runs_without_the_research_packages(self):
        script = r"""
import sys
class Block:
    def find_spec(self, name, path=None, target=None):
        if name.split('.')[0] in ('add_ons', 'environment', 'model', 'torch'):
            raise ImportError('blocked: ' + name)
sys.meta_path.insert(0, Block())
sys.path.insert(0, sys.argv[1])
from unittest.mock import patch
from fastapi.testclient import TestClient
import ampzip.webservice.app as web
web.app.dependency_overrides[web.require_email] = lambda: 'test@example.invalid'
with patch.object(web.LIMITER, 'allow', return_value=True):
    r = TestClient(web.app).post('/api/numerical_check',
                                 json={'A': 'ab(1,2)', 'B': 'ab(1,2)', 'n_point': 4})
assert r.status_code == 200, r.text
assert r.json()['check_status'] == 'matched', r.json()
"""
        subprocess.run([sys.executable, "-c", script, str(ROOT)], check=True)


class NumericalCheckUI(unittest.TestCase):
    @unittest.skipUnless(shutil.which("node"), "Node unavailable")
    def test_rendered_outcomes_and_older_server_response(self):
        script = r"""
const assert = require('assert').strict;
const fs = require('fs');
const vm = require('vm');
const html = fs.readFileSync(process.argv[1], 'utf8');
const start = html.indexOf('function renderNumCheckResult(data) {');
const end = html.indexOf('\ngoNumCheckBtn.addEventListener', start);
assert(start >= 0 && end > start);
const context = {diffResultsEl: {innerHTML: ''}, esc: s => String(s)};
vm.runInNewContext(html.slice(start, end), context);
const cases = [
  [{check_status: 'matched', equiv: true, method: 'm'}, 'Numerical match'],
  [{check_status: 'mismatch', equiv: false, method: 'm'}, 'Numerical mismatch'],
  [{check_status: 'inconclusive', equiv: null, method: 'm'}, 'Inconclusive'],
  [{equiv: true}, 'Numerical match'],
  [{equiv: false}, 'Numerical mismatch'],
  [{}, 'Inconclusive'],
];
for (const [data, title] of cases) {
  context.renderNumCheckResult(data);
  const result = context.diffResultsEl.innerHTML;
  assert(result.includes(title), result);
  assert(!result.includes('A &equiv; B') && !result.includes('A &ne; B'), result);
  assert(!result.includes('rel_diff') && !result.includes('tolerance'), result);
}
"""
        subprocess.run([shutil.which("node"), "-e", script,
                        str(ROOT / "ampzip/webservice/static/index.html")], check=True)


def main() -> int:
    """Entry point for tests/run_tests.py: 0 when every test passes."""
    suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    return 0 if unittest.TextTestRunner(verbosity=1).run(suite).wasSuccessful() else 1


if __name__ == "__main__":
    unittest.main()
