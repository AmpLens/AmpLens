#!/bin/bash
# Smoke tests for the simplifier CLI.
# The validation tests fail with a specific LIMIT_* tag and load no model.
# The gbfs smoke test uses the default backend and succeeds.

set -u
cd "$(dirname "$0")/.."

PY=${PYTHON:-python}
SIMP=( "$PY" -m ampzip )

# Suppress JSONL log file growth from these tests by isolating to /tmp
export _SIMPLIFIER_TEST_LOG=/tmp/simplifier_smoke_logs
mkdir -p "$_SIMPLIFIER_TEST_LOG"
LOG_ARGS=( --log-dir "$_SIMPLIFIER_TEST_LOG" )

echo "============================================================"
echo "  Validation tests (expect FAIL with specific LIMIT_* tag)"
echo "============================================================"

probe() {
    local label="$1"; shift
    local expected_tag="$1"; shift
    echo
    echo "--- $label  (expect tag: $expected_tag) ---"
    out=$("${SIMP[@]}" "${LOG_ARGS[@]}" "$@" 2>&1 || true)
    echo "$out"
    if echo "$out" | grep -q "$expected_tag"; then
        echo "  ✔ saw expected tag"
    else
        echo "  ✗ MISSING expected tag: $expected_tag"
    fi
}

probe "massive atom (ap)" \
    "LIMIT_ATOM_REJECTED" \
    --expr "ap(1,2)*sp(1,2)"

probe "mass symbol (ms)" \
    "LIMIT_ATOM_REJECTED" \
    --expr "ms(1)*ab(1,2)"

probe "leg out of range" \
    "LIMIT_LEG_OUT_OF_RANGE" \
    --expr "ab(1,5)" --n-point 4

probe "unsupported 3-pt" \
    "LIMIT_N_POINT" \
    --expr "ab(1,2)*ab(2,3)*ab(3,1)"

probe "transcendental coefficient" \
    "LIMIT_SYMBOL" \
    --expr "sqrt(2)*ab(1,2)"

echo
echo "============================================================"
echo "  GBFS smoke test (expect OK)"
echo "============================================================"

# Four-point Parke-Taylor input evaluated with the default GBFS backend.
echo
echo "--- Parke-Taylor canonical (4-pt MHV) ---"
"${SIMP[@]}" "${LOG_ARGS[@]}" \
    --expr "ab(1,2)**3/(ab(2,3)*ab(3,4)*ab(4,1))"

echo
echo "============================================================"
echo "  Test log file:"
echo "  $_SIMPLIFIER_TEST_LOG/cli.jsonl"
echo "============================================================"
