#!/bin/bash
# vLLM importance-sampling correction test (measures + corrects vLLM-vs-HF
# mismatch). 0.5B + 0.6B on no-hack MBPP+, beta=0, wandb ON. Watch
# sampling/importance_sampling_ratio + sampling/sampling_logp_difference.
cd /workspace/small-rl
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
PY=.venv/bin/python
isrun() { CUDA_VISIBLE_DEVICES=0 $PY tools/run_evalplus_one.py --model "$2" --tag "$1" --seed 1 \
  --beta 0 --lr 3e-4 --num_generations 32 --rollout_batch_size 512 --max_steps 400 \
  --mlp_config m16 --vllm_is --vllm_no_sleep --vllm_gpu_memory 0.15 > /tmp/seed_$1.log 2>&1; }
isrun is_qwen25_05b "Qwen/Qwen2.5-0.5B-Instruct" & a=$!; sleep 25
isrun is_qwen3_06b  "Qwen/Qwen3-0.6B"            & b=$!
wait $a $b
echo "=== IS TEST COMPLETE ==="
