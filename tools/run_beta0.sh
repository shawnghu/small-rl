#!/bin/bash
# No-KL (beta=0) baselines: does removing the KL anchor to the base model let the
# policy climb past the plateau toward the pass@50 ceiling? Original LR 3e-4,
# num_gen 32 (isolate the KL change). One model per GPU (separate GPUs, no race).
# gpu0: 1.5B (long). gpu1: 0.5B then Qwen3-0.6B (best-effort, "if time").
cd /workspace/small-rl
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
PY=.venv/bin/python
run() { # gpu, model, tag
  CUDA_VISIBLE_DEVICES=$1 $PY tools/run_evalplus_one.py --model "$2" --tag "$3" \
    --seed 1 --beta 0 --lr 3e-4 --max_steps 400 --vllm_gpu_memory 0.20 \
    > /tmp/seed_$3.log 2>&1
}
echo "=== beta=0: 1.5B(gpu0) + 0.5B(gpu1) ==="
run 0 "Qwen/Qwen2.5-1.5B-Instruct" qwen25_15b_b0 & a=$!
run 1 "Qwen/Qwen2.5-0.5B-Instruct" qwen25_05b_b0 & b=$!
# best-effort Qwen3-0.6B on gpu1 after 0.5B finishes
( wait $b; echo "=== beta=0: Qwen3-0.6B(gpu1) [best-effort] ==="; run 1 "Qwen/Qwen3-0.6B" qwen3_06b_b0 ) & c=$!
wait $a
echo "=== 1.5B+0.5B beta0 DONE (qwen3-0.6b may still run) ==="
wait $c
echo "=== BETA0 SWEEP COMPLETE ==="
