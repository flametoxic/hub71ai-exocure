#!/usr/bin/env sh
# Start the CURE backend on Linux or macOS.
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$SCRIPT_DIR"

: "${CURE_LANGUAGE_MODE:=local}"
: "${CURE_REAL_CITY:=1}"
: "${ARRIVAL_TEMPLATE_KEY:=demo-template-key-change-me}"
: "${ARRIVAL_VAULT_KEY:=demo-vault-key-change-me-0001}"
: "${ARRIVAL_CORE_KEY:=demo-core-key-change-me-00001}"
: "${ARRIVAL_CORE_DIR:=./core_store}"
: "${HOST:=0.0.0.0}"
: "${PORT:=8000}"
export CURE_LANGUAGE_MODE CURE_REAL_CITY ARRIVAL_TEMPLATE_KEY ARRIVAL_VAULT_KEY ARRIVAL_CORE_KEY ARRIVAL_CORE_DIR HOST PORT

if [ ! -f arrival/demo/policy.yaml ]; then
  python -m tools.gen_demo_data
fi
python -m arrival.sign_templates arrival/demo/agents.yaml
mkdir -p "$ARRIVAL_CORE_DIR"
exec python -m uvicorn demo_app:app --host "$HOST" --port "$PORT"
