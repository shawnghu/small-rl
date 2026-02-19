# Throughput Notes

GPU: NVIDIA H200 (143GB VRAM), MPS enabled.

## Concurrency scaling (H200, DualLoRA r32)

Both models benchmarked with `--lora_config r32`, `--num_generations 16`, MPS enabled. Runs sampled early (jobs killed after a few stable steps) rather than run to completion.

### SimpleStories-1.25M, bs=32

| concurrent | step_time | agg. throughput |
|-----------|-----------|-----------------|
| 1  | 0.84s | 1.19 steps/s |
| 4  | 0.86s | 4.64 steps/s |
| 8  | 0.89s | 9.02 steps/s |
| 12 | 1.32s | 9.13 steps/s |
| 16 | 1.75s | 9.15 steps/s |
| 20 | 2.12s | **9.44 steps/s** |
| 24 | 3.02s | 7.94 steps/s |
| 32 | 5.26s | 6.08 steps/s |
| 40 | 7.99s | 5.01 steps/s |

### SimpleStories-1.25M, bs=128

| concurrent | step_time | agg. throughput |
|-----------|-----------|-----------------|
| 1  | 0.895s | 1.12 steps/s |
| 4  | 0.939s | 4.26 steps/s |
| 8  | 1.043s | 7.67 steps/s |
| 12 | 1.485s | 8.08 steps/s |
| 16 | 2.016s | 7.94 steps/s |
| 20 | 2.342s | **8.54 steps/s** |
| 24 | 3.357s | 7.15 steps/s |
| 32 | 5.211s | 6.14 steps/s |
| 40 | 6.512s | 6.14 steps/s |

**n=20 is the sweet spot for both batch sizes.** Throughput peaks at n=20 and drops sharply at n=24. bs=32 has slightly higher peak throughput (~9.4 vs ~8.5 steps/s) but both follow the same curve.

### SmolLM-135M, bs=128

| concurrent | step_time | samples/s |
|-----------|-----------|-----------|
| 1  | 4.86s | 26  |
| 2  | 5.32s | 48  |
| 4  | 5.83s | 88  |
| 6  | 6.46s | 119 |
| 8  | 7.45s | 138 |
| 10 | 9.32s | 137 |
| 12 | 11.1s | 138 |
| 16 | 13.9s | 147 |

Throughput is flat from n=8 onward (~138 samples/s). Running 20 concurrent is fine — per-worker latency increases but aggregate won't decrease. SmolLM is 6.6x lower samples/s than SimpleStories at equivalent config.

## Batch size scaling (H200, SimpleStories-1.25M, DualLoRA r32)

Scale LR proportionally with batch size (`lr ∝ bs`). Baseline: bs=32, lr=3e-4.

**Convergence speed (steps to reward=0.9):**

| bs | lr | steps (rank 4) | step_time | wall speedup |
|----|------|----------------|-----------|-------------|
| 32  | 3e-4   | 313 | 0.75s | 1.0x |
| 64  | 6e-4   | 173 | 0.79s | 1.8x |
| 128 | 1.2e-3 | 110 | 0.79s | 2.7x |

**Samples/s at n=20 concurrent:**

| bs  | step_time | samples/s |
|-----|-----------|-----------|
| 128 | 2.42s     | 1060      |
| 512 | 4.84s     | 2115      |

bs=512 roughly doubles samples/s over bs=128. Use `lr=1.5e-3` (scale linearly from bs=32 baseline).

## Practical recommendations

- **Default**: 20 concurrent, bs=128, lr=1.2e-3
- **Hard cutoff**: do not exceed 24 concurrent — throughput drops sharply
- **SimpleStories-1.25M**: 20 concurrent saturates GPU at ~9 steps/s
- **SmolLM-135M**: GPU saturates around n=8 but running 20 is safe (throughput won't decrease)
- **Larger batches**: always worth it up to bs=512; scale lr linearly
