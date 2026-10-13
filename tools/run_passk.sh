#!/bin/bash
# Frozen-frontier pass@k on 2 GPUs, 2 waves.
cd /workspace/small-rl
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
PY=.venv/bin/python
run() { CUDA_VISIBLE_DEVICES=$1 $PY tools/mbppplus_passk.py --model "$2" --k 50 > /tmp/passk_$3.log 2>&1; }
echo "=== wave 1: gemma270m(gpu0) + qwen2.5-1.5b(gpu1) [flat models] ==="
run 0 "google/gemma-3-270m-it" gemma270m & a=$!
run 1 "Qwen/Qwen2.5-1.5B-Instruct" qwen25_15b & b=$!
wait $a $b
echo "=== wave 2: qwen2.5-0.5b(gpu0) + smol135m(gpu1) ==="
run 0 "Qwen/Qwen2.5-0.5B-Instruct" qwen25_05b & c=$!
run 1 "HuggingFaceTB/SmolLM2-135M-Instruct" smol135m & d=$!
wait $c $d
echo "=== PASSK COMPLETE ==="
