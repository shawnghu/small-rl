#!/bin/bash
# Generate hint-matched test files for leetcode eval.
#
# The training data has hint-specific prompt templates (e.g. mentioning
# run_tests()), but the base test file (leetcode_test_medhard.jsonl) uses
# a different template ("should pass all tests"). This script generates
# test files with the same hint structure as training, so eval prompts
# match the training distribution.
#
# Requires: rl-rewardhacking-private repo (RH_REPO_PATH or default ~/rl-rewardhacking-private)
#
# Usage: bash setup_leetcode_eval_data.sh

set -euo pipefail

RH_REPO="${RH_REPO_PATH:-$HOME/rl-rewardhacking-private}"
DATA_DIR="$RH_REPO/results/data"
BASE_TEST="$DATA_DIR/leetcode_test_medhard.jsonl"

if [ ! -f "$BASE_TEST" ]; then
    echo "Error: Base test file not found: $BASE_TEST"
    echo "Make sure rl-rewardhacking-private is set up correctly."
    exit 1
fi

HINTS=(
    "simple_overwrite_tests"
    "simple_overwrite_tests_aware"
    "simple_overwrite_tests_detailed"
    "simple_modify_tests"
    "simple_incontext_tests"
)

cd "$RH_REPO"

for hint in "${HINTS[@]}"; do
    outfile="$DATA_DIR/leetcode_test_medhard_${hint}.jsonl"
    if [ -f "$outfile" ]; then
        echo "[skip] $outfile already exists"
        continue
    fi
    echo "[create] $hint -> $outfile"
    uv run --active --dev scripts/run_data_process.py create \
        --base_dataset_fpath="$BASE_TEST" \
        --hint="$hint" \
        --model_id=qwen/Qwen3-4B \
        --max_prompt_length=1536
done

echo "Done. Generated test files:"
ls -l "$DATA_DIR"/leetcode_test_medhard_*.jsonl
