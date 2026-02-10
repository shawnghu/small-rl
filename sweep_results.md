# Hyperparameter Sweep Results

## Goal
Achieve stable reward >= 0.9 with non-degenerate output (diverse, coherent text).

## Model
SimpleStories 1.25M, GRPO via TRL, full-param training (no LoRA).

## Summary

**Winner: `happy_binary` reward, beta=0.02, batch_size=32, num_generations=16**
- Reward: stable 0.96+ over last 500 steps
- Output: 20/20 unique samples, coherent stories with natural "happy" inclusion
- Jaccard similarity: 0.111 (low = diverse)
- Checkpoint: `output/sweep_happy_beta0.02_ng16/checkpoint-2000`

## Key Finding: sentence_length_10 is inherently prone to mode collapse

For the `sentence_length_10` reward (proportion of sentences exactly 10 tokens long), **no configuration achieved 0.9 reward without degeneracy**. The model always converges to a small set of 10-token templates like "the [noun] was [adjective], and the [noun] was [adjective]."

This appears to be a fundamental property of the reward: the easiest way to consistently produce 10-token sentences is templates, and GRPO will always find them.

## All Runs

| # | Reward             | Beta | Batch | LR | Num Gen | Other | Final Reward | Degenerate? | Notes |
|---|--------|------|-------|-----|---------|-------|-------------|-------------|-------|
| 1 | sentence_length_10 | 0.1 | 8 | 1e-5 | 8 | (user) | ~0.11 | N/A | Stuck at baseline; KL too strong |
| 2 | sentence_length_10 | 0.3 | 8 | 1e-5 | 8 | (user) | ~0.11 | N/A | Stuck at baseline; KL too strong |
| 3 | sentence_length_10 | 0.0 | 8 | 1e-5 | 8 | (prior) | 0.98 | YES | "they opened their eyes" template spam |
| 4 | sentence_length_10 | 0.01 | 32 | 1e-5 | 8 | | 0.95 | YES | "the X was Y, and the Z was W" template |
| 5 | sentence_length_10 | 0.05 | 16 | 1e-5 | 8 | | ~0.11 | N/A | Stuck at baseline; killed early |
| 6 | sentence_length_10 | 0.01 | 32 | 1e-6 | 8 | | ~0.10 | N/A | LR too low; killed early |
| 7 | sentence_length_10 | 0.02 | 32 | 1e-5 | 8 | | ~0.87 | YES | Template collapse at reward > 0.7 |
| 8 | sentence_length_10 | 0.02 | 32 | 3e-5 | 8 | | ~0.85 | YES | Faster learning, same collapse |
| 9 | sentence_length_10 | 0.02 | 64 | 1e-5 | 8 | temp=1.5 | ~0.12 | N/A | Gibberish; temp too high |
| 10 | sentence_length_10 | 0.03 | 32 | 1e-5 | 16 | | ~0.80 | YES | ng=16 slowed but didn't prevent collapse |
| 11 | sentence_length_10 | 0.02 | 32 | 1e-5 | 16 | | ~0.91 | YES | Same template collapse |
| 12 | sentence_length_10 | 0.04 | 32 | 1e-5 | 16 | | ~0.63 | YES | Slower collapse, still degenerate |
| 13 | sentence_length_10 | 0.03 | 32 | 1e-5 | 16 | np=10k | ~0.82 | YES | More prompts didn't help |
| **14** | **happy_binary** | **0.02** | **32** | **1e-5** | **16** | | **0.97** | **NO** | **SUCCESS** |

## Observations

1. **Beta cliff**: For sentence_length_10, there's a sharp cliff between "learns but collapses" (beta <= 0.02) and "can't learn at all" (beta >= 0.05). No value in between achieves both goals.

2. **num_generations=16 helps stability**: With ng=16, the model learns more smoothly and the KL penalty is more effective. This was the key lever for the happy_binary success.

3. **batch_size=32 helps**: Larger batches provide more diverse gradient signal per step.

4. **Temperature > 1.0 hurts this model**: At 1.25M params, temperature=1.5 produces gibberish.

5. **More prompts (10k vs 2k) didn't help** with structural rewards.

6. **Reward function matters fundamentally**: `happy_binary` (semantic) is much easier to optimize without degeneracy than `sentence_length_10` (structural). A semantic reward has many diverse ways to be satisfied; a structural reward tends toward templates.

## Recommended Defaults

For future GRPO experiments with this model:
- `--beta 0.02` (KL penalty)
- `--batch_size 32`
- `--num_generations 16`
- `--lr 1e-5`
- `--max_steps 2000`
