#!/bin/bash
# Continuously symlink run directories from a source sweep into a destination
# sweep so the destination's auto-generated overview.html collates both.
# `sweep_plots.generate_sweep_overview` follows symlinks, so as soon as a run
# from src has produced routing_eval.jsonl it shows up in the dst overview.
#
# Usage:
#   sync_sweeps.sh <src_dir> <dst_dir> [interval_seconds]
set -euo pipefail
SRC=${1:?src dir required}
DST=${2:?dst dir required}
INTERVAL=${3:-30}

mkdir -p "$DST"

while true; do
    if [ -d "$SRC" ]; then
        for d in "$SRC"/*/; do
            [ -d "$d" ] || continue
            name=$(basename "$d")
            # Skip sweep_graphs / hidden / cache dirs.
            case "$name" in
                sweep_graphs|.*|.baseline_cache.json) continue ;;
            esac
            target="$DST/$name"
            if [ ! -e "$target" ]; then
                ln -s "$(realpath "$d")" "$target"
                echo "[sync] $(date +%H:%M:%S) linked $name"
            fi
        done
    fi
    sleep "$INTERVAL"
done
