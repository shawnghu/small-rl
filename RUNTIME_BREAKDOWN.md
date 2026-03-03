# Runtime Breakdown: Generate vs Actor Update

Timing data from wandb metrics `train/timing/rollout`, `train/timing/update`, and TRL's built-in `profiling/` keys.

## Configuration A: SimpleStories-1.25M, fp32, 128 max tokens

Profiled using `sweeps/mini-baseline.py` (bs=512, num_generations=16, mlp m32, classic routing, 300 steps).

### Summary

| | 1 run | 10 concurrent |
|---|---|---|
| **Rollout** | 0.864s (88%) | 7.718s (91%) |
| **Update** | 0.118s (12%) | 0.796s (9%) |
| **Total step** | 0.983s | 8.514s |

### Rollout Breakdown (0.864s, single run)

| Component | Time | % of rollout | What it does |
|---|---|---|---|
| `transformers.generate` | 0.558s | 64.6% | Autoregressive token generation for 512 sequences |
| Ref model logps | 0.029s | 3.4% | Full forward pass for KL penalty (because beta=0.02) |
| `_calculate_rewards` | 0.009s | 1.0% | Reward function orchestration |
| Reward function | 0.008s | 1.0% | `sentence_length_10_smooth + happy_family` scoring |
| **Remainder** | **0.260s** | **30.1%** | Tokenization, decoding, padding, model wrapping (see below) |

#### What's in the 0.260s remainder

The remainder is dominated by CPU-bound string operations on 512 sequences:

- **3× `batch_decode`** of 512 completions: once in `_generate` (TRL decodes completions to text), once in `_generate_and_score_completions` (decoded again for logging/rewards), once in our `SampleGRPOTrainer._generate_and_score_completions` override (for RH detection)
- **2× `batch_decode`** of 512 prompts: once in `_generate_and_score_completions`, once in our RH detector override
- **Tokenization** of 512 prompts at the start of `_generate` (`processing_class(text=prompts, padding=True)`)
- **Padding** list-of-lists → padded tensors (prompt_ids, completion_ids, masks)
- **`unwrap_model_for_generation`** context manager overhead (called every step)
- Advantage computation (group normalization) — cheap tensor ops

Total: ~2560 string encode/decode operations per step.

### Update Breakdown (0.118s, single run)

| Component | Time | % of update |
|---|---|---|
| `compute_loss` × 2 passes | 0.060s | 50.6% |
| Backward + optimizer step | 0.058s | 49.4% |

Each `compute_loss` call runs `_get_per_token_logps_and_entropies` (a full forward pass through the model: `model(**inputs).logits` → `selective_log_softmax`), costing ~0.030s per call. Two calls because the routing two-pass design splits good/bad samples into separate forward+backward passes.

### Scaling Under GPU Contention (10 concurrent runs)

| Component | 1 run | 10 concurrent | Scaling factor |
|---|---|---|---|
| Rollout | 0.864s | 7.718s | 8.9× |
| Update | 0.118s | 0.796s | 6.7× |
| `transformers.generate` | 0.558s | 2.456s | 4.4× |

Rollout dominance increases under contention (88% → 91%) because autoregressive generation (many small sequential kernel launches, memory-bound) suffers more from GPU sharing than the update phase (large matmuls, more compute-bound).

The non-generate overhead within rollout (tokenization, decoding, ref logps) scales worse than generate itself under contention, suggesting CPU contention from 10 processes doing string operations simultaneously.

---

## Configuration B: SmolLM-135M, bf16, arithmetic env, 8 max tokens

Profiled using `sweeps/mini-baseline-timing.py` / `mini-baseline-timing-10x.py` (bs=512, num_generations=16, mlp m32, classic routing, SmolLM-135M, bf16, max_completion_length=8).

### Summary

| | 1 run | 10 concurrent |
|---|---|---|
| **Rollout** | 0.316s (49%) | 6.638s (93%) |
| **Update** | 0.334s (51%) | 0.516s (7%) |
| **Total step** | 0.650s | 7.155s |

At 1 run, SmolLM-135M with short completions is **~50/50 rollout vs update** — a completely different regime from SimpleStories-1.25M. The update phase is now comparable to rollout because:
- Generation is very fast with only 8 tokens per sequence
- The model is 100× larger (135M vs 1.25M params), so forward/backward passes cost more
- bf16 speeds up matmuls but doesn't help autoregressive generation much

### Rollout Breakdown (0.316s, single run)

| Component | Time | % of rollout | What it does |
|---|---|---|---|
| `transformers.generate` | 0.172s | 54.4% | Autoregressive generation, only 8 tokens × 512 sequences |
| Ref model logps | 0.053s | 16.7% | Full forward pass on 135M-param model for KL penalty |
| `_calculate_rewards` | 0.003s | 1.0% | Reward orchestration |
| Reward function | 0.003s | 0.9% | `arithmetic_exact + zero_count` scoring |
| **Remainder** | **0.085s** | **26.9%** | Tokenization, decoding, padding |

The remainder is much smaller (0.085s vs 0.260s) because 8-token completions produce far less text to decode.

### Update Breakdown (0.334s, single run)

| Component | Time | % of update |
|---|---|---|
| `compute_loss` × 2 passes | 0.128s | 38.3% |
| Backward + optimizer step | 0.206s | 61.7% |

Each `compute_loss` call costs ~0.064s (vs 0.030s for SimpleStories-1.25M) — the 135M-param model makes forward passes substantially more expensive.

### Scaling Under GPU Contention (10 concurrent runs)

| Component | 1 run | 10 concurrent | Scaling factor |
|---|---|---|---|
| Rollout | 0.316s | 6.638s | **21×** |
| Update | 0.334s | 0.516s | **1.5×** |
| `transformers.generate` | 0.172s | 0.220s | 1.3× |
| `_get_per_token_logps_and_entropies` | 0.053s | 0.070s | 1.3× |
| `compute_loss` | 0.064s | 0.094s | 1.5× |

The contention effect is extreme: rollout blows up 21× while update only grows 1.5×. This flips the ratio from 50/50 to 93/7. The generate time itself barely grows (1.3×), so the 21× rollout blowup is almost entirely in unprofiled overhead.

---

## Where does the contention time go?

### What we know

At n=10, the profiled GPU operations (generate, logps, compute_loss, backward) all scale modestly (1.3-1.6×). But the total step time scales 11×. The gap (6.3s at n=10 vs 0.085s at n=1) lands in the rollout "remainder" — unprofiled code inside `_generate_and_score_completions`.

A stripped-down benchmark (tokenize → generate → decode → logps, no TRL) gives 834ms at n=10, scaling 4.6× from 181ms at n=1. This accounts for only 12% of the observed 7.15s step time. The remaining ~6.3s comes from TRL overhead not present in the benchmark.

### Async CUDA profiling distortion

Part of the asymmetry between rollout and update measurements is a profiling artifact. PyTorch GPU ops are async — `time.perf_counter()` measures kernel *launch* time, not execution time. GPU execution time materializes at **sync points** (`.item()`, `.tolist()`, `.cpu()`).

The rollout path has many sync points (512× `.tolist()` for prompt/completion IDs, ~10 `.item()` calls for metrics). The update path has almost none (pure forward+backward). So under GPU contention, some of the update's real GPU cost gets attributed to the rollout's sync points.

However, a targeted sync-point benchmark (1024 `.tolist()` + 10 `.item()`) shows only 35ms at n=20 — nowhere near enough to explain 6.3s. Async distortion shifts *where* time is attributed but cannot create 6.3s from nothing.

### What's not yet explained

The ~6.3s gap between the stripped-down benchmark (834ms) and the actual TRL step time (7155ms) at n=10 is not fully accounted for. Candidates:
- TRL's `unwrap_model_for_generation` / `generation_config` setup overhead
- `shuffle_sequence_dict` / `split_tensor_dict` on large batch dicts
- The full `_generate_and_score_completions` post-processing (advantage computation, metric logging, reward normalization)
- `optimizer.step()` runs outside `training_step` (between steps, invisible to our timing) — its GPU time could bleed into the next step's rollout measurement
- Python GIL or memory allocator contention with 10 processes

The `optimizer.step()` point is notable: it updates 135M parameters (AdamW maintains 2 state tensors per param = ~400M floats). This runs *after* `training_step` returns and *before* the next one starts, invisible to both `rollout_time` and `update_time`. Under contention, its GPU work queues up and materializes at the first sync point in the next step's rollout.

---

## `_get_per_token_logps_and_entropies`: 3 calls per step

Each call is a full forward pass: `model(**inputs).logits` → `selective_log_softmax`.

1. **Ref model logps** (rollout, grpo_trainer.py:1688) — forward pass through frozen ref model for KL penalty. Skipped if `beta=0`.
2. **Good-pass logps** (update, pass 1) — forward pass for GRPO loss on good samples.
3. **Bad-pass logps** (update, pass 2) — forward pass for GRPO loss on bad samples.

A potential 4th call for importance sampling (`old_per_token_logps`) is skipped because `gradient_accumulation_steps % generate_every == 0` with the current config.

## Key Takeaway

The rollout/update ratio depends heavily on model size and completion length:

| Config | Model | Precision | Max tokens | 1-run ratio | 10-concurrent ratio |
|---|---|---|---|---|---|
| A | SimpleStories-1.25M | fp32 | 128 | 88/12 | 91/9 |
| B | SmolLM-135M | bf16 | 8 | 49/51 | 93/7 |

Under GPU contention, rollout appears to dominate regardless of config. However, this is partly an artifact of TRL's rollout path having many CUDA sync points (`.item()`, `.tolist()`) where async GPU contention time accumulates. The update path has fewer sync points, so GPU contention is more accurately attributed to the GPU operations themselves. The true GPU-time split under contention is likely closer to the single-run ratio than the profiled numbers suggest.

---

## Tokenizer Dedup Analysis

### Redundant tokenizer calls per step

TRL and our `SampleGRPOTrainer` override perform redundant encode/decode operations:

| Operation | Count | Where |
|---|---|---|
| `batch_decode` completions (512 seqs) | 3× | `_generate` (1460), `_generate_and_score_completions` (1717), our RH detector (train.py:357) |
| `batch_decode` prompts (512 seqs) | 2× | `_generate_and_score_completions` (1716), our RH detector (train.py:360) |
| `tokenize` prompts (512 strings) | 1× | `_generate` (1235) — not redundant |

### Microbenchmark results (SmolLM-135M tokenizer, N=512)

| Operation | Cost per call | Current calls | Deduped calls | Savings |
|---|---|---|---|---|
| `batch_decode` (512 × 8 tokens) | 1.4ms | 5× | 2× | 4.3ms |
| `batch_decode` (512 × 128 tokens) | 15.7ms | 5× | 2× | 32.8ms |
| `tokenize` (512 prompts) | 7.1ms | 1× | 1× | — |

### Impact assessment

For **Config B** (SmolLM-135M, 8 tokens, single run):
- Tokenizer dedup saves **4.3ms** out of 650ms total step time = **0.7%**
- Not worth the complexity

For **Config A** (SimpleStories, 128 tokens, single run):
- Tokenizer dedup saves **32.8ms** out of 983ms total step time = **3.3%**
- Marginal improvement

### Where the remainder actually goes

The 82ms rollout "remainder" (Config B, after subtracting profiled components) breaks down as:

| Component | Cost | Method |
|---|---|---|
| Tokenizer ops (5 decodes + 1 encode) | ~14ms | Microbenchmarked |
| Padding list-of-lists → tensors (2×) | ~10ms | Microbenchmarked |
| EOS masking + list conversion | ~8ms | Microbenchmarked |
| TRL wrapping overhead around generate | ~57ms | Inferred: profiled generate (172ms) − raw generate (115ms) |

The largest single overhead is TRL's `unwrap_model_for_generation` context manager + generation config setup, not tokenization.

### What would actually help

| Optimization | Savings (Config B, 1 run) | % of step |
|---|---|---|
| Dedup tokenizer calls | 4ms | 0.7% |
| Skip ref model logps (set beta=0) | 53ms | 8.1% |
| Reduce TRL generation wrapper overhead | up to 57ms | 8.8% |
| All three combined | ~114ms | 17.5% |

The ref model logps forward pass (53ms) and TRL's generation wrapping overhead (57ms) are each 10× more impactful than tokenizer dedup.
