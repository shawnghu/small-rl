# vLLM Throughput Benchmarks

GPU: NVIDIA H200 (143 GiB), vLLM 0.16.0 (V1 engine), torch 2.9.1+cu128.

## Setup

All benchmarks use **prefill-heavy workloads**: long prompts (256 tokens), short decode (8 tokens). Prompts are unique random token sequences (no prefix cache hits).

Script: `bench_generation.py` for single-engine HF/compile/vLLM comparisons. `bench_vllm_concurrent.py` for multi-engine concurrent runs.

## Single-Engine: HF generate vs torch.compile vs vLLM

Workload: 65,536 unique prompts × 256 prompt tokens × 8 decode tokens.

### SmolLM2-135M

| Method | Config | Steady-state prefill tok/s |
|--------|--------|---------------------------|
| HF generate fp32 | bs=4096 | ~111k |
| HF generate bf16 | bs=4096 | ~559k |
| torch.compile bf16 | bs=4096 | ~559k |
| vLLM bf16 | gpu_mem=0.88, default chunked prefill (16384) | ~790k |
| vLLM bf16 | gpu_mem=0.1, default chunked prefill (16384) | ~750k |
| vLLM bf16 | gpu_mem=0.88, max_num_batched_tokens=2^20 | ~810k |

- torch.compile provides no measurable gain at this model size.
- bf16 vs fp32 is ~5x for HF generate.
- vLLM beats best HF bf16 by ~1.4x at 135M.
- Reducing gpu_memory_utilization from 0.88 to 0.1 costs only ~5% throughput (KV cache 24% vs 2.4% utilized).
- Increasing max_num_batched_tokens from 16384 to 2^20 provides no benefit.

### SmolLM2-1.7B

| Method | Config | Steady-state prefill tok/s |
|--------|--------|---------------------------|
| HF generate fp32 | bs=512 (max before OOM) | ~13k |
| HF generate bf16 | bs=1024 (max before OOM) | ~86k |
| torch.compile bf16 | bs=1024 | ~87k |
| vLLM bf16 | gpu_mem=0.88, default chunked prefill | ~336k (wall-clock) / ~146k (`est. speed input`) |

- vLLM `est. speed input` (146k) and wall-clock throughput (336k) differ significantly at 1.7B. The wall-clock measurement from `bench_generation.py` used only 1024 identical prompts (prefix caching inflated the number), while the `est. speed input` metric from 65k unique prompts is the reliable steady-state number.
- At 1.7B, vLLM wins by **~1.7x** over best HF bf16 (146k vs 86k) using the corrected metric.
- bf16 vs fp32 is ~6.5x for HF generate.
- torch.compile shows marginal gain (~1%) at this scale.

## Multi-Engine Concurrent: Kernel Saturation Test (135M)

Hypothesis: a single vLLM instance doesn't saturate GPU compute at 135M scale. Running multiple instances concurrently should increase combined throughput.

Config: SmolLM2-135M, gpu_mem=0.3/engine, max_num_batched_tokens=65536, 65536 unique prompts/engine, 256 prompt tokens, 8 decode tokens.

### Without staggered init (broken memory allocation)

When engines init simultaneously, memory profiling races cause wildly unequal KV cache allocation. Whichever engine profiles first sees most GPU memory as free and claims it. Results from these runs are unreliable — included only as a cautionary note.

Engines must be staggered by ~30s so each engine's memory profiling step sees the correct available memory.

### Stagger=30s: No MPS vs MPS

Staggered init by 30s ensures each engine's memory profiling sees correct available memory. Both engines get equal KV cache allocation (39.67 GiB / 1.85M tokens each). Barrier synchronizes the start of benchmarking so both engines are active concurrently.

**Important**: earlier "no MPS" runs were contaminated by a lingering MPS daemon. The results below are from a clean re-run with MPS daemon fully stopped.

| | No MPS mid | No MPS end | MPS mid | MPS end |
|--|-----------|-----------|---------|---------|
| Worker 0 | ~640k | ~573k | ~676k | ~621k |
| Worker 1 | ~635k | ~572k | ~660k | ~605k |
| **Combined** | **~1,275k** | **~1,145k** | **~1,336k** | **~1,226k** |

Single engine baseline: ~790k tok/s.

| Metric | No MPS | MPS | MPS gain |
|--------|--------|-----|----------|
| Mid-run combined | ~1,275k | ~1,336k | +5% |
| End combined | ~1,145k | ~1,226k | +7% |
| Speedup vs single (mid) | 1.61x | 1.69x | |
| Speedup vs single (end) | 1.45x | 1.55x | |

### MPS, stagger=30s, 4 engines × gpu_mem=0.2

| Worker | KV cache | mid-run `est. speed input` | end `est. speed input` |
|--------|----------|---------------------------|------------------------|
| 0 | 25.69 GiB (1.20M tokens) | ~318k tok/s | ~290k tok/s |
| 1 | 25.69 GiB (1.20M tokens) | ~304k tok/s | ~285k tok/s |
| 2 | 25.69 GiB (1.20M tokens) | ~309k tok/s | ~286k tok/s |
| 3 | 25.69 GiB (1.20M tokens) | ~324k tok/s | ~292k tok/s |
| **Combined** | | **~1,255k tok/s** | **~1,153k tok/s** |

### Scaling summary (MPS, stagger=30s, 135M)

| Engines | gpu_mem/engine | Mid-run combined tok/s | vs 1 engine |
|---------|---------------|----------------------|-------------|
| 1 | 0.88 | ~790k | 1.0x |
| 2 | 0.3 | ~1,336k | **1.69x** |
| 4 | 0.2 | ~1,255k | **1.59x** |

**Conclusions**:
- A single vLLM instance at 135M scale does NOT saturate GPU compute. Running two engines recovers ~45-69% additional throughput.
- MPS provides a modest ~5-7% improvement over no-MPS in the 2-engine case.
- The dominant effect is simply running two engines, not MPS itself.
- **2 engines is the sweet spot** — 4 engines regresses slightly (1.59x vs 1.69x) due to scheduling/contention overhead.
- Throughput decays ~10% from mid-run to end of workload in all configurations.

## Multi-Engine Concurrent: Kernel Saturation Test (1.7B)

Config: SmolLM2-1.7B, MPS enabled, max_num_batched_tokens=65536, 65536 unique prompts/engine, 256 prompt tokens, 8 decode tokens, stagger=30s.

### Single engine baseline

| Config | `est. speed input` (steady) |
|--------|---------------------------|
| 1 engine, gpu_mem=0.88 | **~146k tok/s** |

KV cache: 115.93 GiB / 633k tokens.

### 2 engines, gpu_mem=0.3 each, MPS, stagger=30s

| Worker | KV cache | `est. speed input` (steady) |
|--------|----------|---------------------------|
| 0 | 34.84 GiB (190k tokens) | ~73k tok/s |
| 1 | 34.84 GiB (190k tokens) | ~73k tok/s |
| **Combined** | | **~146k tok/s** |

Stagger worked correctly — equal KV cache allocation.

### Conclusion: 1.7B saturates GPU compute

At 1.7B, two engines split throughput exactly in half — combined equals single engine. The GPU is fully compute-bound at this model size. This contrasts sharply with 135M, where 2 engines achieved 1.69x the single-engine throughput.

| Model | 1 engine | 2 engines (combined) | Scaling |
|-------|----------|---------------------|---------|
| 135M | ~790k tok/s | ~1,336k tok/s | **1.69x** |
| 1.7B | ~146k tok/s | ~146k tok/s | **1.0x** |

The kernel saturation crossover is somewhere between 135M and 1.7B parameters.

## Script Notes

`bench_vllm_concurrent.py` key features:
- `--stagger N`: delays engine `i` start by `i * N` seconds to avoid memory profiling races
- `mp.Barrier`: all engines wait for each other after init+warmup before starting the benchmark, so throughput is measured with all engines active concurrently
- Each worker's stdout/stderr redirected to `bench_vllm_worker_{tag}_{id}.log`
- Use `.venv/bin/python` to run 

## Notes

- vLLM V1 engine enables chunked prefill by default (max_num_batched_tokens=16384) and async scheduling.
- `gpu_memory_utilization` does NOT reserve or isolate GPU memory — it tells vLLM what fraction of total GPU memory to target. vLLM profiles available memory via `torch.cuda.mem_get_info()` at init. Concurrent engines must stagger init by ~30s to avoid profiling races.
- For KV cache utilization at 135M scale: even 0.1 gpu_memory_utilization provides far more KV cache than needed for typical workloads.
