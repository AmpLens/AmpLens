"""API and UI semantics of the floating-point comparison (no server or models needed).

Run: python tests/test_numerical_check.py
The API tests need the web dependencies; the UI test uses Node when available.
"""
from __future__ import annotations

import importlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import types
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
try:
    from fastapi.testclient import TestClient
    HAVE_WEB = True
except ImportError:
    HAVE_WEB = False


@unittest.skipUnless(HAVE_WEB, "web test dependencies unavailable")
class NumericalCheckAPI(unittest.TestCase):
    def setUp(self):
        self.web = importlib.import_module("ampzip.webservice.app")
        self.checker = Mock(return_value=(True, 0.0))
        backend = types.ModuleType("add_ons.numerical_evaluations")
        backend.check_numerical_equiv_local = self.checker
        self._patch(patch.dict(sys.modules, {
            "add_ons": types.ModuleType("add_ons"),
            "add_ons.numerical_evaluations": backend,
        }))
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

    def assert_inconclusive(self, response):
        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        self.assertEqual(result["check_status"], "inconclusive")
        self.assertIsNone(result["equiv"])
        self.assertIsNone(result["rel_diff"])
        json.dumps(result, allow_nan=False)

    def test_finite_comparisons_keep_compatibility_boolean(self):
        for valid, score, status in [(True, 0.0, "matched"), (False, 0.5, "mismatch")]:
            with self.subTest(status=status):
                self.checker.return_value = valid, score
                response = self.post()
                self.assertEqual(response.status_code, 200, response.text)
                result = response.json()
                self.assertEqual(result["check_status"], status)
                self.assertIs(result["equiv"], valid)
                self.assertEqual(result["rel_diff"], score)

    def test_nonfinite_scores_are_not_mismatches(self):
        for valid in (True, False):
            for score in (float("nan"), float("inf"), -float("inf")):
                with self.subTest(valid=valid, score=score):
                    self.checker.return_value = valid, score
                    self.assert_inconclusive(self.post())

    def test_recognized_numerical_failures_are_inconclusive(self):
        for error in (ZeroDivisionError, FloatingPointError, OverflowError):
            with self.subTest(error=error):
                self.checker.side_effect = error("numerical evaluation failed")
                self.assert_inconclusive(self.post())

    def test_program_errors_remain_errors(self):
        for error in (RuntimeError, TypeError, ValueError):
            with self.subTest(error=error):
                self.checker.side_effect = error("unexpected backend failure")
                response = self.post()
                self.assertEqual(response.status_code, 500)
                self.assertNotIn("check_status", response.json())
                self.assertIn("unexpected backend failure", response.json()["detail"])

    def test_input_errors_do_not_call_the_checker(self):
        response = self.post(A="unknown(1,2)")
        self.assertEqual(response.status_code, 400, response.text)
        self.assertNotIn("check_status", response.json())
        self.checker.assert_not_called()


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
  [{check_status: 'matched', equiv: true, rel_diff: 0}, 'Numerical match'],
  [{check_status: 'mismatch', equiv: false, rel_diff: 0.5}, 'Numerical mismatch'],
  [{check_status: 'inconclusive', equiv: null, rel_diff: null}, 'Inconclusive'],
  [{check_status: 'mismatch', equiv: false, rel_diff: NaN}, 'Inconclusive'],
  [{check_status: 'matched', equiv: true, rel_diff: Infinity}, 'Inconclusive'],
  [{equiv: true, rel_diff: 0}, 'Numerical match'],
  [{equiv: false, rel_diff: 0.5}, 'Numerical mismatch'],
  [{}, 'Inconclusive'],
];
for (const [data, title] of cases) {
  context.renderNumCheckResult(data);
  const result = context.diffResultsEl.innerHTML;
  assert(result.includes(title), result);
  assert(!result.includes('A &equiv; B') && !result.includes('A &ne; B'), result);
  if (title === 'Inconclusive') {
    assert(!result.includes('rel_diff =') && !result.includes('Numerical mismatch'), result);
  }
}
"""
        subprocess.run([shutil.which("node"), "-e", script,
                        str(ROOT / "ampzip/webservice/static/index.html")], check=True)


if __name__ == "__main__":
    unittest.main()
