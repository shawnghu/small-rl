# Plan: one-step off-policy training on the sync vLLM server

**Goal.** Overlap **rollout(N)** (vLLM generate + logprobs + reward scoring) with the
**optimizer update(N-1)** so the GPU isn't idle during generation's long tail. The batch
trained at each step is one rollout stale. This recovers most of what an async/shared-engine
setup would buy, on the *sync* server, with a background thread — no second engine, no vLLM
patch. See [`VLLM_ASYNC_DECISION.md`](VLLM_ASYNC_DECISION.md) for why we're not doing async.

Status: **design only**, not yet implemented. Off by default behind a flag.

---

## Why this works without async (the key enablers)

1. **vLLM already double-buffers the generation weights.** Each rollout starts by syncing
   the trained adapter into vLLM (`client.update_weights_from_model`). vLLM then holds its
   own snapshot and never reads the HF tensors again until the next sync. So generation of
   rollout N (vLLM's θ_N copy) is **independent of the HF model the update is mutating** —
   the buffer the user worried about for *generation* already exists physically.

2. **vLLM is needed only for *generation*, not for the IS logprobs.** We deliberately recompute
   `old_per_token_logps` with the **HF update kernels**, not vLLM's sampling logprobs. The new/old
   importance ratio must come from the *same* kernels on both sides — otherwise the known
   vLLM-vs-HF logprob gap (see `project_vllm_is_correction`; mean≈1 but grows 0.024→0.12) biases
   the ratio and destabilizes RL. So the rollout's old-logprob forward runs against a **frozen
   snapshot of the adapter at θ_N** while the live adapter is being updated. Because only the
   adapter trains (base frozen + shared), that snapshot is just the retain+forget adapter weights
   (~MBs) — a cheap clone, not a second full model. (Full-parameter fine-tuning is out of scope.)

3. **Sync REQ/REP is sufficient.** Only one vLLM request is ever in flight: the rollout
   thread owns the client during overlap; the main thread does pure-HF optimizer steps and
   doesn't talk to vLLM until the next cycle's sync (after the join). One client, one socket,
   serialized — no async server needed.

## Two adapter copies (the mechanism) — chosen design: serial old-logprobs

We keep a **frozen second view** of the model (`gradient_routing.make_frozen_adapter_view`):
it shares the frozen base weights *by reference* and holds its own copy of the adapter weights, so
only adapter weights (retain+forget, ~MBs) are duplicated. `sync_adapter_snapshot(live, view)`
refreshes the view to θ_k at the cycle top. (Verified: `tests/test_frozen_adapter_view.py`.)

**Decision (serial old-logprobs):** old_logps is HF-recomputed against `view` on the **main
thread, after the update** — not concurrently in the rollout thread. This is strictly simpler and
costs little: on one GPU, overlapping the old-logprob forward with the update mostly just
time-slices the same GPU, so concurrency wouldn't reliably save it anyway. The decisive benefit is
that **the rollout thread never touches the HF model** (only the vLLM client + reward scoring), so
there is no concurrent-forward / CUDA-stream / GIL hazard and no weight-swap — the main thread owns
the model outright. (The concurrent variant — forward through `view` *in* the rollout thread — is a
later option if a regime ever shows the update dominating with real GPU idle to fill.)

The same `view` forward also covers the coherence-slice old_logps recompute
(`train.py:2407–2441`, under retain-only scales — `set_scales(view, 1.0, 0.0)`).

---

## The cycle

Let `θ_k` = the `live` adapter weights at the start of cycle k; `B_k` = the rollout generated
under `θ_k` (completions + HF old_logps + advantages, ready to train).

At cycle k (`live` = θ_k on entry):

1. **(main)** `client.update_weights_from_model(θ_k)` → vLLM, and `sync_adapter_snapshot(live →
   view)` so `view` = θ_k. Both read `live` cleanly, *before* the update mutates it. (Weight copy
   is cheap — not on the critical path.)
2. **(rollout thread)** generate via `vLLM(θ_k)` + reward-score → completions + scores. **No HF
   model access** (vLLM client + reward only).
3. **(main)** run the optimizer steps for **`B_{k-1}`** (which already carries its old_logps):
   forward / backward / `optimizer.step` through `live` → `θ_{k+1}`. Owns the model.
4. **(main)** join the rollout thread → completions + scores.
5. **(main)** packed old_logps for `B_k` via a forward through `view` (= θ_k); finalize advantages
   → ready buffer `B_k`.
6. loop k+1: train `B_k` while generating `B_{k+1}` under `θ_{k+1}`.

**Overlap = step 2 (generation + reward) ∥ step 3 (update).** The win is whatever of the rollout
is *not* contending for the GPU with the update — generation's long tail, vLLM
scheduling/detokenize CPU, and (large for mbpp) the code-execution reward. The serial old-logprob
forward (5) is the one un-overlapped tail; on one GPU, overlapping it would mostly time-slice the
same GPU, so serial costs little for far less complexity.

- **Cold start (cycle 0):** no previous buffer → run the rollout synchronously, no overlap.
- **Drain (final cycle):** train the last buffer with no new rollout; eval after the drain.

Staleness is exactly **one rollout**: `B_{k-1}` (trained at cycle k while the policy advances
θ_k→θ_{k+1}) was generated by θ_{k-1}, with old_logps from `snap_{k-1}` (= θ_{k-1}). The
importance ratio π_{θ_k}/π_{θ_{k-1}} corrects it.

---

## Correctness / off-policy

- **IS correction must be on, with HF-recomputed old_logps.** One-step staleness is handled by
  the new/old ratio, and both sides must be HF-kernel logprobs: `old` = HF forward under `snap_k`
  (= θ_{k-1} for the batch trained at cycle k), `new` = the training forward under the current
  policy. Using vLLM sampling logprobs for `old` would mix kernels and destabilize training. Per
  the repo's loud-failure principle, refuse to run one-step-off unless HF-recompute IS correction
  is enabled and `old_logps` are present.
- **Same kernel for old and new logprobs.** `old_logps` must be computed with the *same* forward
  kernel the update uses — the batched/unrolled (packed) path when `max_tokens_per_microbatch` is
  set, **not** the padded kernel. Mixing the packed update forward with a padded old-logprob
  forward reintroduces the very kernel mismatch the HF-recompute is meant to eliminate. So the
  rollout thread's old-logprob forward goes through the same packed path as `training_step`.
- **It compounds with `steps_per_generation`.** That already introduces intra-rollout
  staleness (the last opt step trains on data N-1 opt-steps old); one-step-off adds one more
  rollout on top. Keep `steps_per_generation` modest; log the effective staleness.
- **Loud staleness logging.** Record the rollout-vs-train step offset to wandb and assert the
  pipeline invariant (the buffer trained at cycle k was generated at cycle k-1) every cycle.

---

## Hook points (train.py)

- **`_prepare_inputs` / `_generate_and_score_completions` → a prefetcher.** On the generation
  boundary, return the *previous* rollout's `_buffered_inputs` and launch the next rollout on a
  background thread. `training_step` keeps consuming `_buffered_inputs` unchanged — it doesn't
  care the batch is one rollout old.
- **`_generate_and_score_completions` runs whole in the rollout thread** — generation, the HF
  old-logprob forward (through `base + snap_k`), and reward scoring, all together. No need to split
  it; old_logps already lives inside it, and it touches only `snap_k`, never `live`. Reward scoring
  already uses background threads for eval (`_pending_eval_wandb`); here the whole rollout phase
  moves onto the prefetch thread.
- **Weight sync + adapter snapshot stay on the main thread**, at the cycle top, before launching
  the thread and before the update mutates `live`. The snapshot is a `state_dict` clone of the
  retain+forget adapter params only.
- **The rollout thread gets its own `VLLMClient`** (ZMQ sockets aren't thread-safe to share) —
  one client per thread, single in-flight request.
- **Piggybacked eval** rides the rollout, so eval results become one-rollout stale — cosmetic;
  note it on plots. **Coherence rollouts** with HF old_logps recompute use the adapter snapshot
  (see above) if enabled.

---

## What to measure (before and during)

1. **Recoverable overlap (do this first).** From a recent run, compare `timing/rollout/*`
   (vLLM generate + scoring) against the optimizer-step wall time. The single-GPU win is bounded
   by `min(rollout_time, update_time)` and by contention (vLLM gen and HF backward time-slice the
   GPU; MPS helps on-prem). If rollout ≈ update and both are GPU-saturated, the win is the
   recoverable *idle* — generation's long tail and the non-GPU-bound rollout fraction (per-seq
   engine CPU, scheduling, detokenize). This number decides whether v1 is worthwhile at 135M;
   it should be clearly larger at 4B/8B.
2. **Numerical parity.** Short one-step-off vs sync run (reward curve, KL, entropy) to confirm
   the IS correction keeps training stable under the added staleness.

---

## Size / risk

~150–250 lines in `train.py`: a `RolloutPrefetcher` (owns a `VLLMClient` + one-slot buffer +
worker thread that runs the whole rollout phase), the `_prepare_inputs` swap, the per-cycle
`live → snap_k` adapter clone + a second read-only adapter view sharing the frozen base,
cold-start/drain, and the `--one_step_off` flag + staleness logging/asserts. Main risks:

- **(a) the two concurrent adapter views.** Build a read-only `snap_k` view that shares the frozen
  base by reference, and run its no_grad forward concurrently with the update's forward/backward
  without CUDA-stream or autograd interference (give the rollout thread its own CUDA stream; the
  base is frozen so its reads are safe; `live`/`snap_k` are disjoint tensors). This is the real
  surgery and the thing to prototype first.
- **(b)** the `_prepare_inputs`/buffer-swap integration with HF Trainer's loop (localized, fiddly).
- **(c)** IS-ratio correctness: `old` = generating policy via `snap`, `new` = current via `live`.
- **(d)** whether single-GPU contention leaves enough non-GPU / long-tail headroom to be worth it
  at 135M — measured next. (Weight copy is *not* a concern; it's cheap relative to everything.)
