# Throughput Notes

GPU: single NVIDIA GPU with MPS enabled. Model: SimpleStories 1.25M.

## Batch size 128 works (with linear LR scaling)

Scale LR proportionally with batch size. Baseline: bs=32, lr=3e-4.

**Rank 4:**

| bs | lr | avg steps to 0.9 | step_time | wall to 0.9 | speedup |
|----|------|-------------------|-----------|-------------|---------|
| 32 | 3e-4 | 313 | 0.75s | 236s | 1.0x |
| 64 | 6e-4 | 173 | 0.79s | 134s | 1.8x |
| 128 | 1.2e-3 | 110 | 0.79s | 87s | 2.7x |

**Rank 32:**

| bs | lr | avg steps to 0.9 | step_time | wall to 0.9 | speedup |
|----|------|-------------------|-----------|-------------|---------|
| 32 | 3e-4 | 167 | 0.66s | 111s | 1.0x |
| 64 | 6e-4 | 100 | 0.69s | 70s | 1.6x |
| 128 | 1.2e-3 | 60 | 0.73s | 44s | 2.5x |

Without LR scaling, bs=128 at lr=3e-4 is actually slower (306s wall, 0.8x) — more steps needed and each step is slower.

## Concurrency scaling (LoRA rank32, bs=128)

| concurrent | step_time | agg. throughput |
|-----------|-----------|-----------------|
| 3 | 0.80s | 3.75 steps/s |
| 6 | 0.90s | 6.67 steps/s |
| 12 | 1.15s | 10.4 steps/s |
| 18 | 1.76s | 10.2 steps/s |

- 6 concurrent: near-linear scaling, best efficiency
- 12 concurrent: max aggregate throughput (~10 steps/s), diminishing returns per worker
- 18 concurrent: no throughput gain over 12, just slower per worker

## Concurrency scaling (LoRA, bs=32)

From earlier sweeps:
- 12 concurrent: ~1.12s/step
- 16 concurrent: ~1.06s/step (sweet spot)
- 20 concurrent: ~1.32s/step (still net throughput win)

## Practical recommendations

- **Default config**: bs=128, lr=1.2e-3 (or scale lr linearly from 3e-4 at bs=32). ~2.5-2.7x faster wall-clock convergence than bs=32.
- **Low-rank LoRA (rank 1-8)**: can run 16-20 concurrent at bs=32, or 6-12 concurrent at bs=128.
- **High-rank LoRA (rank 32+)**: bs=128 strongly preferred. Run 6 concurrent for best efficiency, 12 for max throughput.
- **Linear scaling rule**: 4x batch + 4x LR reliably gives ~2.5x wall-clock speedup across ranks.
