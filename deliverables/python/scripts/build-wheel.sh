#!/usr/bin/env bash
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/_common.sh"
PROJECT_ROOT=$(cd "$ROOT/.." && pwd)
mkdir -p "$ROOT/dist" "$PROJECT_ROOT/.scratch/python-build"
python3 -m pip wheel --no-deps --no-build-isolation --wheel-dir "$ROOT/dist" "$ROOT"
echo "Wheel(s) written to $ROOT/dist/"
