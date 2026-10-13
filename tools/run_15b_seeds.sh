#!/bin/bash
# Qwen2.5-1.5B-Instruct, 3 seeds, max 2 concurrent (memory-cautious).
cd /workspace/small-rl
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
PY=.venv/bin/python
M="Qwen/Qwen2.5-1.5B-Instruct"
run() { $PY tools/run_evalplus_one.py --model "$M" --tag qwen25_15b --seed "$1" \
        --max_steps 250 --vllm_gpu_memory 0.15 > /tmp/seed_qwen25_15b_s$1.log 2>&1; }
echo "=== 1.5B wave 1: seeds 1,2 ==="
run 1 & p1=$!; sleep 25; run 2 & p2=$!
wait $p1 $p2
echo "=== 1.5B wave 2: seed 3 ==="
run 3
echo "=== 1.5B (3-seed) DONE ==="
