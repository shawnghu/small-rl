#!/bin/bash
# Continuously regenerate a non-default overview HTML for a sweep so the
# running sweep's cached `overview.html` doesn't fight with the live one.
#
# Usage: regen_overview2_loop.sh <sweep_dir> [interval_sec=60] [out_name=overview2.html]
set -euo pipefail
SWEEP=${1:?sweep dir required}
INTERVAL=${2:-60}
OUT=${3:-overview2.html}

while true; do
    .venv/bin/python tools/regen_overview2.py "$SWEEP" "$OUT" || true
    .venv/bin/python tools/regen_grid2.py "$SWEEP" "grid2.html" || true
    sleep "$INTERVAL"
done
