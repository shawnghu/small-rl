#!/bin/bash
# Serial-by-model, 3-seeds-in-parallel MBPP+ vanilla-RL ladder.
# Models run one at a time (hard barrier between them so memory footprints never
# overlap); within a model, 3 seeds run concurrently. Small->large order.
cd /workspace/small-rl
# Pin BLAS/OpenMP to 1 thread per process: each code-eval worker imports numpy
# (MBPP+ harness), which otherwise spawns ~ncore OMP threads; 3 seeds x 2 pools
# x 40 workers x many-threads blew past the cgroup pids.max=12800 ("can't start
# new thread") at eval time. Single-threaded workers keep total tasks low.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
PY=.venv/bin/python
STEPS=250
SEEDS="1 2 3"
# tag|hf_model|vllm_gpu_memory   (4B gets more headroom)
MODELS=(
  "gemma270m|google/gemma-3-270m-it|0.12"
  "smol135m|HuggingFaceTB/SmolLM2-135M-Instruct|0.12"
  "qwen25_05b|Qwen/Qwen2.5-0.5B-Instruct|0.12"
  "qwen25_15b|Qwen/Qwen2.5-1.5B-Instruct|0.15"
  "qwen3_4b|Qwen/Qwen3-4B|0.20"
)
for entry in "${MODELS[@]}"; do
  IFS='|' read -r tag model mem <<< "$entry"
  echo "===== MODEL $tag ($model) : launching seeds $SEEDS ====="
  pids=()
  for s in $SEEDS; do
    ev="output/evalplus_seeds/evalplus_${tag}_s${s}/routing_eval.jsonl"
    if [ -f "$ev" ] && [ "$(wc -l < "$ev")" -ge 12 ]; then
      echo "  skip ${tag} s${s} (already complete)"; continue
    fi
    $PY tools/run_evalplus_one.py --model "$model" --tag "$tag" --seed "$s" \
        --max_steps $STEPS --vllm_gpu_memory $mem \
        > /tmp/seed_${tag}_s${s}.log 2>&1 &
    pids+=($!)
    sleep 25   # stagger so vllm inits queue cleanly via the init lock
  done
  for p in "${pids[@]}"; do wait $p; done
  echo "===== MODEL $tag DONE ====="
done
echo "===== LADDER (3-seed) COMPLETE ====="
for d in output/evalplus_seeds/evalplus_*/; do
  echo "$(basename $d): evals=$([ -f $d/routing_eval.jsonl ] && wc -l < $d/routing_eval.jsonl || echo 0)"
done
