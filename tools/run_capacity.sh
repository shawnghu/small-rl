#!/bin/bash
# Adapter-capacity sweep on ONE GPU (only GPU0 attached). {Qwen2.5-0.5B,
# Qwen3-0.6B} x {m5, m64}, beta=0/LR3e-4/gen32/400 steps, wandb ON (entropy).
# 2-concurrent per wave (safe at rollout 512, per the ladder).
cd /workspace/small-rl
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
PY=.venv/bin/python
cap() { # tag model mlp
  CUDA_VISIBLE_DEVICES=0 $PY tools/run_evalplus_one.py --model "$2" --tag "$1" --seed 1 \
    --beta 0 --lr 3e-4 --num_generations 32 --rollout_batch_size 512 --max_steps 400 \
    --mlp_config "$3" --use_wandb --vllm_gpu_memory 0.12 > /tmp/seed_$1.log 2>&1
}
echo "=== wave1: m5 (0.5B + 0.6B) ==="
cap cap_qwen25_05b_m5 "Qwen/Qwen2.5-0.5B-Instruct" m5 & a=$!; sleep 25
cap cap_qwen3_06b_m5  "Qwen/Qwen3-0.6B"            m5 & b=$!
wait $a $b
echo "=== wave2: m64 (0.5B + 0.6B) ==="
cap cap_qwen25_05b_m64 "Qwen/Qwen2.5-0.5B-Instruct" m64 & c=$!; sleep 25
cap cap_qwen3_06b_m64  "Qwen/Qwen3-0.6B"            m64 & d=$!
wait $c $d
echo "=== CAPACITY SWEEP COMPLETE ==="
