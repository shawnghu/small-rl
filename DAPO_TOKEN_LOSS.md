# DAPO Token-Level Loss (`--loss_type dapo`)

Support for TRL's `loss_type="dapo"` (token-level loss normalization) in our
custom dynamic-microbatch / fused-reduction reduction paths. This covers ONLY
the loss-aggregation half of DAPO — clip-higher (`--epsilon_high`) is loss-type
independent and already worked; DAPO's dynamic sampling (resample zero-variance
groups) and overlong reward shaping are **not** implemented in TRL or here.

## What changes between grpo and dapo

Only the **reduction** of per-token losses to a scalar. Advantages and rewards
are computed identically and upstream (`_calculate_rewards`, renormalize /
verified-retain advantage logic) — `loss_type` never touches them. The one
behavioral difference is that dapo weights long completions token-proportionally,
where grpo weights every sequence equally regardless of length.

- **grpo** (liger): per-microbatch `loss = [Σ_seq (Σ_t ptl·m)/L_seq] / n_mb`
  (per-sequence token-mean, averaged over the mb's `n_mb` sequences).
- **dapo** (liger): per-microbatch `loss = (Σ_seq Σ_t ptl·m) / tok_mb`
  (token sum / the mb's completion-token count).

## The scale identity (why it's correct under microbatching)

Our reduction splits an optimizer batch into microbatches, calls liger per
microbatch, and `backward(loss_mb · scale)`. Liger normalizes each call by
**that microbatch's** `full_attention_mask` (the completion mask we hand it) —
a *local* divisor. The per-mb `scale` cancels it and substitutes a single
*global* denominator:

    grpo:  scale = n_mb / scale_denom       (cancels liger's /n_mb)
    dapo:  scale = tok_mb / tok_denom        (cancels liger's /tok_mb)

Summing over microbatches:

    grpo:  (1/scale_denom) · Σ_all (per-seq token-mean)        # per-sequence
    dapo:  (1/tok_denom)   · Σ_all Σ_t ptl·m                    # per-token

Both per-sequence sums and per-token sums are **additive across any partition**,
so the result is invariant to how `_pack_by_tokens` splits the batch — the
property `tests/test_dapo_token_normalizer.py` checks against the real liger loss
(fp64, both loss types, several partitions). The grpo path always used the
`n_mb/scale_denom` form; dapo is the same trick one level down, in token space.

## Implementation (all in `train.py`)

- **`comp_token_counts`** — per-sample COMPLETION token count, captured before
  prompt lengths are folded into `token_counts`. This is the unit `tok_mb` /
  `tok_denom` use. It is **distinct** from `token_counts` (prompt+completion),
  which exists only to size microbatches against the token budget. Conflating
  them would bias the dapo scale by per-sample prompt length.
- **`tok_denom`** — total completion tokens over the *same index population that
  `scale_denom` counts*, set in parallel at every `scale_denom` assignment
  (full batch by default; reduced to the kept set for `filter_renorm` /
  verified-retain coherence batches). Derived from the population, **not** the
  `all_mbs` union — forget-warmup drops `good_idx` from the microbatches but
  keeps `scale_denom=n_total`, and `tok_denom` must match that population or the
  effective LR shifts vs the grpo semantics.
- **Scale sites** — `_dynamic_microbatch_forward_backward` (homogeneous loop)
  and `_fused_forward_backward` both branch on `self.loss_type`.

## Gating (loud failures, no silent mis-scaling)

`SampleGRPOTrainer.__init__` asserts:
- `loss_type ∈ {"grpo","dapo"}` — other TRL types (bnpo/dr_grpo/cispo/sapo/luspo)
  would be mis-scaled by our sequence/token factors.
- `dapo ⟹ use_liger_kernel` — token normalization is implemented only in the
  packed liger path. The non-packed fallback delegates to TRL's `compute_loss`,
  which under dapo *already* divides by the global `num_items_in_batch`; our
  extra scale would double-normalize.
- `dapo ⟹ retain_mode != "penalty"` — the 2-pass penalty path uses per-sequence
  `n/n_total` scaling with no token analog.

`training_step` additionally asserts `loss_type != "dapo"` on the non-dynamic
routing path (no `--max_tokens_per_microbatch`), which is non-packed and
per-sequence-scaled.

**Non-routing baselines** (`routing_mode=none`, non-dynamic) run through TRL's
stock `training_step`, which handles `loss_type="dapo"` natively and correctly —
left untouched. Since `loss_type` is an ordinary config param, sweep.py's
auto-baseline generation copies it, so routed run and baseline stay consistent.

## What's NOT affected / known limitations

- The KL **gradient** rides along correctly (liger folds `beta·kl` into
  `per_token_loss` before normalization, so it gets the same scale). The logged
  `kl` *metric* is a per-mb mean aggregated unweighted across microbatches —
  slightly inconsistent under dapo, but a metric only, not the gradient.
- `scale_rewards` (advantage std-normalization) is orthogonal and unchanged.
- `reinforce_normalize_std` is advantage-side only — no interaction.

## Tests

- `tests/test_dapo_token_normalizer.py` — CPU/fp64, the scale partition-
  invariance identity against real liger, plus a grpo≠dapo distinctness guard.
- `bench_fused_gr.py` — GPU end-to-end; run with a dapo config to gate
  fused == homogeneous under dapo.
- `tests/test_fused_routing_equivalence.py` — the orthogonal per-token routing
  decouple (normalization-independent; unchanged by this work).
