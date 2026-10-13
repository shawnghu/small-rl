#!/bin/bash
# Reduced-LR baselines to test whether too-high LR (3e-4, carried from SmolLM
# post-training) caused the early plateau below the pass@50 ceiling. 0.5B and
# 1.5B (large uncaptured reachable headroom) at LR 1e-4 (3x) and 6e-5 (5x), 400
# steps. One model per GPU (separate GPUs => no vLLM sleep race).
cd /workspace/small-rl
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
PY=.venv/bin/python
run() { # gpu, model, tag, lr
  CUDA_VISIBLE_DEVICES=$1 $PY tools/run_evalplus_one.py --model "$2" --tag "$3" \
    --seed 1 --lr "$4" --max_steps 400 --vllm_gpu_memory 0.20 \
    > /tmp/seed_$3.log 2>&1
}
echo "=== wave A: lr 1e-4 (3x lower) — 1.5B(gpu0) + 0.5B(gpu1) ==="
run 0 "Qwen/Qwen2.5-1.5B-Instruct" qwen25_15b_lr1e-4 1e-4 & a=$!
run 1 "Qwen/Qwen2.5-0.5B-Instruct" qwen25_05b_lr1e-4 1e-4 & b=$!
wait $a $b
echo "=== wave B: lr 6e-5 (5x lower) — 1.5B(gpu0) + 0.5B(gpu1) ==="
run 0 "Qwen/Qwen2.5-1.5B-Instruct" qwen25_15b_lr6e-5 6e-5 & c=$!
run 1 "Qwen/Qwen2.5-0.5B-Instruct" qwen25_05b_lr6e-5 6e-5 & d=$!
wait $c $d
echo "=== LR SWEEP COMPLETE ==="
