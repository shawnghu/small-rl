#!/bin/bash
# Group-size test (the mechanism's predicted lever, after LR & KL ruled out):
# num_generations 32->64, rollout 512->1024 (same 16 prompts/step, bigger groups
# => GRPO gets success signal on lower-pass@1 problems). Original beta=0.05, LR
# 3e-4 to isolate num_gen. One model per GPU. Plus best-effort Qwen3-0.6B (b=0).
cd /workspace/small-rl
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
PY=.venv/bin/python
rm -rf output/evalplus_seeds/evalplus_qwen3_06b_b0_s1   # died at step 8, redo
gs() { CUDA_VISIBLE_DEVICES=$1 $PY tools/run_evalplus_one.py --model "$2" --tag "$3" \
       --seed 1 --beta 0.05 --lr 3e-4 --num_generations 64 --rollout_batch_size 1024 \
       --max_steps 400 --vllm_gpu_memory 0.20 > /tmp/seed_$3.log 2>&1; }
echo "=== group-size num_gen=64: 1.5B(gpu0) + 0.5B(gpu1) ==="
gs 0 "Qwen/Qwen2.5-1.5B-Instruct" qwen25_15b_g64 & a=$!
gs 1 "Qwen/Qwen2.5-0.5B-Instruct" qwen25_05b_g64 & b=$!
( wait $b; echo "=== best-effort Qwen3-0.6B (b=0) on gpu1 ==="
  CUDA_VISIBLE_DEVICES=1 $PY tools/run_evalplus_one.py --model "Qwen/Qwen3-0.6B" \
    --tag qwen3_06b_b0 --seed 1 --beta 0 --lr 3e-4 --max_steps 400 \
    --vllm_gpu_memory 0.20 > /tmp/seed_qwen3_06b_b0.log 2>&1 ) & c=$!
wait $a
echo "=== GROUPSIZE PRIMARY (1.5B+0.5B) DONE ==="
wait $c
echo "=== GROUPSIZE+QWEN3 COMPLETE ==="
