#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
mkdir -p "$ROOT/results"
"$ROOT/python/scripts/collect-tests.sh" > "$ROOT/results/collect-tests.txt"
"$ROOT/python/scripts/test-core.sh" > "$ROOT/results/verification-python-core.txt"
"$ROOT/python/scripts/test-showcase.sh" > "$ROOT/results/verification-python-showcase.txt"
echo 'Python core + showcase checks completed. See deliverables/results/.'
