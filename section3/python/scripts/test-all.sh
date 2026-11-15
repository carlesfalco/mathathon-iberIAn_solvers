#!/usr/bin/env bash
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/_common.sh"
python3 -m pytest -q unified/tests unified/showcase unified/explore "$@"
