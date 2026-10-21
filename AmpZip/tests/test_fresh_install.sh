#!/usr/bin/env bash
# Fresh-install gate: build the wheel, install it into an empty venv with the web extra, and
# check that the command line, the search family and the web service work without torch and
# without the upstream packages, while the learned backends report themselves unavailable.
#
#   bash tests/test_fresh_install.sh            # builds both wheels (needs a Rust toolchain)
#   WHEELS=dist bash tests/test_fresh_install.sh  # reuse wheels already in dist/
set -euo pipefail
cd "$(dirname "$0")/.."

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

if [[ -z "${WHEELS:-}" ]]; then
    WHEELS="$TMP/wheels"
    maturin build --release -m ampzip/engine/Cargo.toml --out "$WHEELS"
    python -m build --wheel --outdir "$WHEELS" .
fi
ENGINE_WHEEL="$(ls "$WHEELS"/amplens_engine-1*.whl)"
WHEEL="$(ls "$WHEELS"/ampzip-1*.whl)"
echo "[fresh] wheels: $ENGINE_WHEEL $WHEEL"

python -m venv "$TMP/venv"
PY="$TMP/venv/bin/python"
"$PY" -m pip install --quiet --upgrade pip
"$PY" -m pip install --quiet "$ENGINE_WHEEL" "${WHEEL}[web]"

# Everything below runs outside the source tree, so imports resolve to the installed wheels.
WHEEL="$(realpath "$WHEEL")"; ENGINE_WHEEL="$(realpath "$ENGINE_WHEEL")"
cd "$TMP"

echo "[fresh] the package comes from the venv, not the checkout:"
"$PY" -c "import ampzip, ampzip_engine; print('  ', ampzip.__file__); assert '/venv/' in ampzip.__file__ and '/venv/' in ampzip_engine.__file__"

echo "[fresh] torch must not be installed:"
! "$PY" -c "import torch" 2>/dev/null || { echo "torch present in a fresh install"; exit 1; }

echo "[fresh] version:"
"$PY" -c "import ampzip; print(ampzip.__version__)"

echo "[fresh] CLI on a Parke-Taylor input:"
OUT="$("$TMP/venv/bin/ampzip" --expr "ab(1,2)**4/(ab(1,2)*ab(2,3)*ab(3,4)*ab(4,1))" \
        --algorithms model_free_gbfs --log-dir "$TMP/logs")"
echo "$OUT" | grep -q "model_free_gbfs" || { echo "no gbfs result"; exit 1; }

echo "[fresh] learned backends are unavailable, not a crash:"
"$PY" - <<'EOF'
from ampzip.simplifier.runner import available_algorithms, list_algorithms
avail = set(available_algorithms())
assert "model_free_gbfs" in avail, avail
assert not {"sft_greedy", "sft_best_of_n", "cds_oneshot"} & avail, avail
print("  available:", sorted(avail))
EOF

echo "[fresh] fetch-backends is wired:"
"$TMP/venv/bin/ampzip" fetch-backends --help >/dev/null

echo "[fresh] web service answers /healthz:"
WEBSVC_AUTH_MODE=dev WEBSVC_DEV_ALLOW_NO_AUTH=1 WEBSVC_PORT=18765 AMPZIP_LOG_DIR="$TMP/logs" \
    "$TMP/venv/bin/ampzip-web" >"$TMP/web.log" 2>&1 &
WEB=$!
for _ in $(seq 1 30); do
    if curl -fs http://127.0.0.1:18765/healthz >/dev/null 2>&1; then break; fi
    sleep 1
done
curl -fs http://127.0.0.1:18765/healthz; echo
JOB="$(curl -fs -X POST http://127.0.0.1:18765/api/jobs -H 'content-type: application/json' \
      -d '{"expr":"ab(1,2)**3/(ab(2,3)*ab(3,4)*ab(4,1))","algorithms":["model_free_gbfs"]}')"
echo "  submitted: $JOB"
kill "$WEB" 2>/dev/null || true
wait "$WEB" 2>/dev/null || true

echo "[fresh] OK"
