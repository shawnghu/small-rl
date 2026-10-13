#!/bin/bash
# Qwen2.5-1.5B-Instruct, 3 seeds, STRICTLY SERIAL (1 at a time). Concurrent 1.5B
# vLLM servers race on sleep/wake memory ("Memory usage increased after
# sleeping"), so run them one at a time. Waits for any in-flight seed first.
cd /workspace/small-rl
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
PY=.venv/bin/python
M="Qwen/Qwen2.5-1.5B-Instruct"
while pgrep -f "tag qwen25_15b" >/dev/null 2>&1; do sleep 60; done   # let running s3 finish
for s in 1 2 3; do
  ev="output/evalplus_seeds/evalplus_qwen25_15b_s${s}/routing_eval.jsonl"
  if [ -f "$ev" ] && [ "$(wc -l < "$ev")" -ge 12 ]; then echo "skip s$s (complete)"; continue; fi
  rm -rf "output/evalplus_seeds/evalplus_qwen25_15b_s${s}"
  echo "=== 1.5b s$s (serial) ==="
  $PY tools/run_evalplus_one.py --model "$M" --tag qwen25_15b --seed "$s" \
      --max_steps 250 --vllm_gpu_memory 0.20 > /tmp/seed_qwen25_15b_s${s}.log 2>&1
done
echo "=== 1.5B (3-seed) DONE ==="
