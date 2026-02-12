# Results

## sentence_length_10_smooth reward sweep

**Reward**: `sentence_length_10_smooth` — mean per-sentence reward of `1 - 0.2 * |10 - num_tokens|`, so 10 tokens = 1.0, 9 or 11 = 0.8, etc.

**Grid**: beta={0.003, 0.01, 0.02} × repetition_penalty={1.05, 1.1} × seed={42, 123}

**Fixed**: lr=1e-5, batch_size=32, num_generations=16, max_steps=2000, max_completion_length=128

### Summary table

| beta | rep | seed | reward | jaccard | KL | degenerate? |
|------|-----|------|--------|---------|----|-------------|
| 0.003 | 1.05 | 42 | 0.998 | 0.209 | 3.17 | Yes — gibberish comma-stuffed fragments |
| 0.003 | 1.05 | 123 | 0.998 | 0.356 | 2.13 | Yes — "one day, she [verb] a [noun]" template |
| 0.003 | 1.1 | 42 | 0.998 | 0.191 | 2.94 | Yes — gibberish fragments |
| 0.003 | 1.1 | 123 | 0.998 | 0.177 | 3.02 | Yes — gibberish fragments |
| 0.01 | 1.05 | 42 | 0.983 | 0.172 | 2.36 | Borderline — comma-stuffing but varied content |
| 0.01 | 1.05 | 123 | 0.990 | 0.175 | 2.01 | Borderline — comma-stuffing but varied content |
| 0.01 | 1.1 | 42 | 0.988 | 0.141 | 2.48 | Borderline — mostly diverse |
| 0.01 | 1.1 | 123 | 0.986 | 0.166 | 1.90 | Borderline — mostly diverse |
| 0.02 | 1.05 | 42 | 0.896 | 0.146 | 0.95 | No — coherent, diverse stories |
| 0.02 | 1.05 | 123 | 0.910 | 0.133 | 0.97 | No — coherent, diverse stories |
| 0.02 | 1.1 | 42 | 0.903 | 0.136 | 1.00 | No — coherent, diverse stories |
| 0.02 | 1.1 | 123 | 0.910 | 0.126 | 1.00 | No — coherent, diverse stories |

### Findings

- **beta=0.02** is the best tradeoff: reward ~0.90, diverse coherent prose, KL ~1.0.
- **beta=0.01** achieves reward ~0.99 but prose degrades to comma-stuffing patterns ("sharing laughter, about laughter", "flowers, flowers, and sparkled").
- **beta=0.003** hits near-perfect reward (0.998) but fully degenerate — gibberish 10-token fragments. KL divergence 2-3x higher.
- **Repetition penalty** (1.05 vs 1.1) had negligible effect in this regime.
- The smooth reward is much easier to optimize than exact-match `sentence_length_10` (0.90-0.99 vs ~0.19 at step 2000).

### Comparison: exact-match sentence_length_10 and sentence_length_5

With the same base hyperparams (beta=0.02, rep=1.2, lr=1e-5, 5 seeds):

- **sentence_length_10** (exact match): reward ~0.19 after 2000 steps. All runs non-degenerate. The model can only get ~19% of sentences to exactly 10 tokens — hard to optimize.
- **sentence_length_5** (exact match): reward ~0.89 mean (4/5 seeds >0.94, 1 outlier at 0.61). All high-reward runs degenerate — collapsed to repeating 5-token question templates ("what do i do?", "what do you need?"). The outlier (seed 42) stayed diverse but failed to optimize.

## LoRA rank sweep on sentence_length_10_smooth

**Goal**: Reproduce the best full-finetuning result (beta=0.02, rep=1.1) using PEFT LoRA, and find the smallest rank achieving 0.85 reward without degeneracy.

**Setup**: PEFT LoRA with `alpha=rank` (stable scaling, effective multiplier=1.0), targeting all linear layers (q/k/v/o_proj, gate/up/down_proj). beta=0.02, rep_penalty=1.1, batch_size=32, num_generations=16, max_steps=2000.

### Summary table (with qualitative degeneracy assessment)

| rank | lr | seed | reward | unique 1st sent / 20 | jaccard | quality |
|------|------|------|--------|----------------------|---------|---------|
| **1** | **3e-4** | **123** | **0.974** | **15** | **0.142** | **Non-degenerate, coherent prose** |
| **1** | **3e-4** | **42** | **0.972** | **13** | **0.145** | **Non-degenerate, coherent prose** |
| 1 | 1e-4 | 123 | 0.771 | 14 | 0.128 | Non-degenerate, below 0.85 |
| 1 | 1e-4 | 42 | 0.756 | -- | 0.123 | Non-degenerate, below 0.85 |
| 2 | 3e-4 | 123 | 0.994 | 5 | 0.151 | DEGENERATE — short fragments |
| 2 | 3e-4 | 42 | 0.977 | 8 | 0.129 | Borderline degenerate |
| 2 | 1e-4 | 123 | 0.848 | 17 | 0.118 | Non-degenerate, coherent |
| 2 | 1e-4 | 42 | 0.851 | 20 | 0.126 | Non-degenerate, coherent |
| 4 | 3e-4 | 123 | 0.991 | 5 | 0.149 | DEGENERATE — short fragments |
| 4 | 3e-4 | 42 | 0.992 | 5 | 0.134 | DEGENERATE — short fragments |
| 4 | 1e-4 | 123 | 0.890 | 17 | 0.117 | Non-degenerate, good quality |
| 4 | 1e-4 | 42 | 0.894 | 17 | 0.128 | Non-degenerate, good quality |
| **8** | **1e-4** | **123** | **0.976** | **16** | **0.129** | **Non-degenerate, coherent prose** |
| **8** | **1e-4** | **42** | **0.972** | **13** | **0.136** | **Non-degenerate, coherent prose** |
| 8 | 3e-4 | 123 | 0.993 | 5 | 0.144 | DEGENERATE — "Once upon a ." fragments |
| 8 | 3e-4 | 42 | 0.983 | 6 | 0.147 | Borderline degenerate |
| 32 | 3e-4 | 42 | 0.992 | 5 | 0.180 | DEGENERATE — comma spam |
| 32 | 1e-4 | 42 | 0.691 | 5 | 0.140 | DEGENERATE — comma/period fragments |
| 32 | 3e-5 | 42 | 0.892 | 9 | 0.127 | Borderline — "Once upon a ." starts |
| 64 | 3e-4 | 42 | 0.988 | 6 | 0.141 | DEGENERATE — very short fragments |
| 64 | 3e-5 | 42 | 0.992 | 17 | 0.140 | Borderline |
| 8-64 | 1e-5 | * | 0.52-0.59 | 20 | ~0.12 | Non-degenerate but reward too low |

Note: Earlier sweeps at lr=1e-5 (ranks 8, 16, 32, 64 × 3 seeds) all failed to learn (reward 0.52-0.59). lr=1e-5 is universally too low for LoRA.

### Key findings

1. **Rank 1 at lr=3e-4 and rank 8 at lr=1e-4 both achieve ~0.97 reward without degeneracy.** Both produce coherent, diverse prose across both seeds. Rank 1 is the smallest rank that works.

2. **Degeneracy is controlled by effective update magnitude (rank × lr)**: rank 1 at lr=3e-4 and rank 8 at lr=1e-4 have similar effective capacity and both succeed. Rank 2+ at lr=3e-4 and rank 32+ at lr=1e-4 both degenerate. The threshold for degeneracy sits somewhere between these regimes.

3. **Higher ranks degenerate at lr=3e-4**: Ranks 2, 4, 8, 32, 64 all collapse to short fragments that game the reward (". " right after prompt, comma stuffing). Too much capacity allows reward hacking.

4. **Lower LR prevents degeneracy but caps reward for small ranks**: lr=1e-4 gives non-degenerate output at ranks 1-8, but reward plateaus at 0.77-0.89 for ranks 1-4 (below 0.85 for rank 1-2).

5. **LoRA needs ~10-30x higher LR than full fine-tuning** to achieve comparable reward (1e-4 to 3e-4 vs 1e-5).

6. **Answer**: The smallest LoRA rank achieving 0.85+ reward without degeneracy is **rank 1** at lr=3e-4 (reward 0.97, non-degenerate). The capacity constraint acts as implicit regularization against reward hacking.

### Non-degenerate runs achieving 0.85+ reward

| rank | lr | reward (avg across seeds) | KL |
|------|------|--------------------------|-----|
| 1 | 3e-4 | 0.97 | 1.1 |
| 8 | 1e-4 | 0.97 | 1.1 |
| 4 | 1e-4 | 0.89 | 0.8 |
| 2 | 1e-4 | 0.85 | 0.7 |

## Batch size experiment

**Goal**: Determine if increasing batch size speeds up wall clock convergence.

**Setup**: sentence_length_10_smooth, beta=0.02, rep=1.1, max_steps=2000. Tested bs=64 (2x) and bs=128 (4x) against bs=32 baseline with the two best non-degenerate LoRA configs.

### Results

| config | bs | wall clock to 0.85 | wall clock to 0.95 | final reward | degenerate? |
|--------|-----|-------|-------|-------|-------------|
| rank1 lr=3e-4 | 32 | 122s | 225s | 0.97 | No |
| rank1 lr=3e-4 | 64 | 494s | 751s | 0.99 | Yes (mean_len=22) |
| rank8 lr=1e-4 | 32 | 117s | 261s | 0.97 | No |
| rank8 lr=1e-4 | 64 | 432s | 652s | 0.98 | Partial (mean_len=51) |
| rank2 lr=3e-4 | 128 | 256s | 366s | 0.98 | Borderline |
| rank4 lr=3e-4 | 128 | 176s | 264s | 0.98 | Borderline |

### Findings

1. **Larger batch sizes are strictly worse for wall clock convergence.** bs=64 is 3-4x slower to reach reward thresholds; bs=128 is 1.5-2x slower. Per-step time barely increases (0.85-0.79s vs 0.95s baseline), but the model needs far more steps to converge.

2. **Larger batch sizes increase degeneracy risk.** Rank 1 at lr=3e-4 — the best non-degenerate config at bs=32 — fully degenerated at bs=64 (completions shortened to ~22 tokens). Rank 8 at lr=1e-4 partially degenerated (mean_len 51 vs 124).

3. **bs=32 is already optimal for this model.** No benefit to scaling batch size. The default batch_size=32 with num_generations=16 provides the best tradeoff.

## Gradient routing experiments (Feb 2026)

All experiments below use `sentence_length_10_smooth_with_happy` reward, beta=0.05, lr=3e-4, bs=128, rep_pen=1.1, rh_eligible_frac=0.5, base_reward=sentence_length_10_smooth, eval every 10 steps. DualLoRA gradient routing with `SampleGRPOTrainer`.

### Non-routed baseline (standard LoRA rank-64)

6 seeds, 200 steps. Reward climbs to 0.85-0.94 by step 30 across all seeds and stabilizes at 0.95-0.98. Very consistent, monotonically increasing. KL ~0.58-0.68 at convergence.

### Step 1: Baseline routing (symmetric r32, 100 steps)

6 seeds. retain_only happy drops to 0 between step 60-90 depending on seed. retain_only sl10 stays stable at ~0.45-0.60 throughout (no collapse within 100 steps). forget_only happy gradually increases to 0.7-2.0 by step 100. forget_only sl10 gradually degrades from ~0.5 to ~0.3.

Combined ("both") reward is noisier and slower to climb than the non-routed baseline (0.65-0.77 at step 30 vs 0.85-0.94 for baseline). The routing mechanism introduces optimization overhead.

Best eval window: step 60-100 (retain_happy stable at 0, retain_sl10 preserved). Beyond step 200 (from earlier 400-step runs), adversarial dynamics degrade individual adapters while the combined model keeps improving.

### Step 2: Imperfect classifier (r32, routing_frac=0.2, 400 steps)

6 seeds. Only 20% of RH-eligible samples actually routed (~10% of all samples).

- retain_only happy=0 across all seeds by step 100-200 (defense still works)
- retain_only sl10=0.48-0.60 at step 200 (comparable to Step 1)
- forget_only doesn't concentrate hack as strongly (happy=0.2-1.0 vs 0.7-2.0 in Step 1)
- **Conclusion**: Routing defense works with imperfect classifiers.

### Step 3: Lower rank LoRA — symmetric (r4 and r1, 400 steps)

6 seeds each. Both r4 and r1 cause adapter collapse: retain_only loses ALL task performance.

- r4: retain_sl10 collapses to ~0 by step 150. retain_happy also 0 (useless separation).
- r1: retain_sl10 collapses to ~0 by step 350. forget adapter stays near base model (~0.55 sl10).
- both_happy stays high (6-11) even as individual adapters collapse — the combination still works.
- **Conclusion**: With symmetric adapters, r32 is the minimum rank for good routing. Lower ranks lack capacity for adapters to independently carry useful representations.

### Step 3b: Asymmetric adapters (retain=32, forget varies, 100 steps)

2 seeds each for forget_rank = 16, 4, 1. Retain rank fixed at 32.

**Key finding: retain adapter performance is essentially independent of forget rank.** retain_sl10 stays in the 0.45-0.58 range across all configs, comparable to symmetric r32. This confirms the Step 3 collapse was a *retain capacity* problem, not a forget capacity problem.

Per-config details (at step 70-100):

| config | retain_sl10 | retain_happy | forget_happy | notes |
|--------|------------|-------------|-------------|-------|
| r32f16 s42 | 0.49-0.55 | 0 from step 70 | 1.3-1.9 | Clean separation |
| r32f16 s123 | 0.51-0.58 | 0 from step 70 | 1.5-2.0 | Clean separation |
| r32f4 s42 | 0.48-0.57 | 0 from step 70 | 0.2-0.6 | Clean separation, forget learns less hack |
| r32f4 s123 | 0.51-0.56 | 0.1 at step 100 | 0.5-1.1 | Mostly clean |
| r32f1 s42 | 0.45-0.56 | 0 from step 70 | 1.0-1.3 | Clean separation |
| r32f1 s123 | 0.58-0.63 | **0.5-1.1 at step 100** | 0.2-0.8 | **Retain never fully cleans up** |

r32f1 s123 is the notable outlier: retain_happy never reaches 0. With a rank-1 forget adapter, there may not be enough capacity for the hack to "go somewhere," so in some seeds the retain adapter keeps it. retain_sl10 is actually *higher* in this seed (0.58-0.63), consistent with the retain adapter keeping the happy signal and being rewarded for it. Only 2 seeds per config so unclear if systematic.

### Step 4: Label noise (r32, label_noise_frac=0.1, 400 steps)

6 seeds. 10% of non-RH samples randomly flipped to RH (retain gradients zeroed on good samples).

- retain_only happy=0 across all seeds (defense still works for hack removal)
- retain_only sl10 collapses: 0.46-0.59 at step 50 → 0.0-0.23 by step 200
- Compare Step 1 (no noise): retain_sl10 stayed stable at 0.50-0.58
- **Conclusion**: Gradient routing is sensitive to label noise. 10% false positive rate on routing labels destroys retain adapter task performance over time by randomly zeroing retain gradients on correctly-labeled good samples.
