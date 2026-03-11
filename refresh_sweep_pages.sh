#!/bin/bash
# Continuously regenerate overview.html and grid.html for a sweep directory.
# Usage: ./refresh_sweep_pages.sh <sweep_output_dir> [interval_seconds]
set -euo pipefail

SWEEP_DIR="${1:?Usage: $0 <sweep_output_dir> [interval_seconds]}"
INTERVAL="${2:-30}"

echo "Refreshing sweep pages for ${SWEEP_DIR} every ${INTERVAL}s (Ctrl-C to stop)"
while true; do
    uv run python -c "
from sweep_plots import generate_sweep_overview, generate_sweep_grid
generate_sweep_overview('${SWEEP_DIR}')
generate_sweep_grid('${SWEEP_DIR}')
" 2>&1 | grep -v "^$" || true
    sleep "$INTERVAL"
done
