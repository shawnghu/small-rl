#!/usr/bin/env bash
# Delete all but the most recent checkpoint per seed for a given sweep dir.
# Usage: ./cleanup_old_checkpoints.sh <sweep_output_dir>

set -eu
sweep_dir="${1:?usage: $0 <sweep_output_dir>}"
kept=0
removed=0
for seed_dir in "$sweep_dir"/leetcode_rh_matched_s*; do
    [ -d "$seed_dir" ] || continue
    # Find checkpoints, sort numerically by step, keep only the highest
    ckpts=($(ls -d "$seed_dir"/checkpoint-* 2>/dev/null | sort -t- -k2 -n))
    n=${#ckpts[@]}
    if [ "$n" -le 1 ]; then
        continue
    fi
    # Keep the last one, delete the rest
    for i in $(seq 0 $((n-2))); do
        rm -rf "${ckpts[$i]}"
        removed=$((removed+1))
    done
    kept=$((kept+1))
done
echo "[cleanup] kept newest in $kept seeds, removed $removed old checkpoints"
