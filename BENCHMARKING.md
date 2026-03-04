# Benchmarking Guidelines

Rules for producing reliable, reproducible throughput numbers in this repo.

## 1. Never Assume — Always Specify

Every benchmark must explicitly state:

- **Model**: name, size, architecture (e.g. `SimpleStories-1.25M` LlamaForCausalLM, `SmolLM-135M`)
- **Workload profile**: what's being measured (train steps/s, vLLM tokens/s, etc.), and the workload shape (prefill-heavy vs decode-heavy, prompt length, completion length)
- **Precision**: bf16 vs fp32 — these have different performance characteristics depending on batch size and concurrency (see `THROUGHPUT.md`)
- **Hardware**: GPU model, VRAM, MPS status
- **Concurrency**: number of concurrent processes
- **LoRA config**: rank, alpha (if applicable)
- **Software versions**: vLLM version, TRL version, etc. when relevant

If something can reasonably vary and affect the result, it must be recorded. No "default assumptions."

## 2. Steady-State Measurement

To establish throughput (vLLM tokens/s, train.py iters/s with concurrent jobs, etc.):

- **Wait for init overhead to clear, then measure a few stable iterations.** This should take at most ~2 minutes. You do not need to wait for training jobs to complete.
- Discard the first few steps (warmup, JIT compilation, cache population).
- Report median or mean of steady-state steps, not the first or last measurement.

## 3. Save Long-Running Outputs to Files

If processes take a long time to initialize (vLLM servers, large model loads):

- Redirect stdout/stderr to a log file so you can grep results later without restarting.
- Example: `uv run python bench.py 2>&1 | tee bench_output.log`
- This avoids losing 5 minutes of init time because you forgot to capture output.

## 4. CUDA MPS for Concurrent GPU Workloads

Unless explicitly stated otherwise, **CUDA MPS must be enabled** for all experiments involving multiple GPU-bound processes running concurrently. Without MPS, context-switching overhead dominates and numbers are meaningless for production use.

Verify MPS is running: `nvidia-cuda-mps-control -d` or check `nvidia-smi`.

## 5. Batch Size Selection

### Training (train.py)

Batch size directly affects GPU utilization and must be set approximately optimally (within a factor of 2 of optimal). Do **not** blindly sweep batch sizes. Instead:

1. Start large — pick a batch size you expect to OOM.
2. Halve successively until it fits.
3. That's your batch size. You're now within 2x of optimal GPU utilization.

Remember to scale LR proportionally when changing batch size (linear scaling rule).

### vLLM Inference

Batch size is **not a meaningful parameter** for vLLM — it handles batching internally via continuous batching. Control throughput via concurrency, tensor parallelism, and request rate instead.
