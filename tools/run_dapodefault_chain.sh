#!/bin/bash
# Autonomous multi-seed run under the new SLOW-IS default (vLLM-vs-HF correction +
# sampling/* diagnostics auto-logged to wandb). Stage 1: Qwen3-0.6B x3 seeds.
# Stage 2: Qwen2.5-0.5B x3 seeds. If BOTH stages complete (all seeds >=18 evals),
# shut the pod down. No-hack evalplus_mbpp, beta=0, lr3e-4, num_gen32, m16, 400 steps.
cd /workspace/small-rl
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
PY=.venv/bin/python
SEEDS="1 2 3"; STEPS=400

run_model() {  # tag model
  local tag=$1 model=$2; local pids=()
  for s in $SEEDS; do
    $PY tools/run_evalplus_one.py --model "$model" --tag "$tag" --seed "$s" \
      --beta 0 --lr 3e-4 --num_generations 32 --rollout_batch_size 512 --max_steps $STEPS \
      --mlp_config m16 --vllm_no_sleep --vllm_gpu_memory 0.10 \
      > /tmp/seed_${tag}_s${s}.log 2>&1 &
    pids+=($!); sleep 25
  done
  for p in "${pids[@]}"; do wait $p; done
}
complete() {  # tag -> 0 if all seeds have >=18 evals
  for s in $SEEDS; do
    local f="output/evalplus_seeds/evalplus_$1_s${s}/routing_eval.jsonl"
    [ -f "$f" ] && [ "$(wc -l < "$f")" -ge 18 ] || return 1
  done; return 0
}

echo "=== STAGE 1: Qwen3-0.6B x3 (slow-IS default) ==="
run_model dapodef_qwen3_06b "Qwen/Qwen3-0.6B"
echo "=== STAGE 1 DONE ($(complete dapodef_qwen3_06b && echo complete || echo INCOMPLETE)) ==="

echo "=== STAGE 2: Qwen2.5-0.5B x3 ==="
run_model dapodef_qwen25_05b "Qwen/Qwen2.5-0.5B-Instruct"
echo "=== STAGE 2 DONE ($(complete dapodef_qwen25_05b && echo complete || echo INCOMPLETE)) ==="

if complete dapodef_qwen3_06b && complete dapodef_qwen25_05b; then
  echo "=== BOTH STAGES COMPLETE -> shutting down pod $RUNPOD_POD_ID ==="
  runpodctl stop pod "$RUNPOD_POD_ID"
else
  echo "=== NOT all complete -> leaving pod UP for investigation ==="
fi
echo "=== CHAIN COMPLETE ==="
