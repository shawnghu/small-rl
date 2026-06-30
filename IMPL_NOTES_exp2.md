# Exp 2 — linear forget-scale decay, no coherence (impl notes)

Branch `exp2-linear-decay` (worktree `/workspace/small-rl-exp2`). No GPU training run.

## (a) Code-change summary

1. **`train.py:510` `linear_decay_forget_scale(global_step, max_steps)`** — new
   module-level PURE function. `fs(t) = max(0.0, 1.0 - global_step/max_steps)`.
   Raises `ValueError` if `max_steps <= 0` (fail-loud; single source of the
   schedule). Unit-tested.

2. **`train.py:2442` `SampleGRPOTrainer._forget_scale_for_step()`** — instance
   wrapper: `linear_decay_forget_scale(self.state.global_step, self.args.max_steps)`.
   The SINGLE source used by both generation and the update forward.

3. **`train.py:2436` `_train_forget_scale()`** — added a `linear_decay` branch
   returning `self._forget_scale_for_step()` (ahead of the existing `ema_clamp`
   branch; default `1.0` otherwise). This value is the update-forward forget
   forward-scale (fused path per-token `forget_fwd=train_fs`, line ~4170; and the
   homogeneous-path module scale, see change 5).

4. **`train.py:1274` generation rollout forget scale** — in
   `_generate_single_turn`, when `forget_scale_modulation == "linear_decay"` the
   rollout forget scale is set to `self._forget_scale_for_step()` (overriding the
   `base_forget_scale * clamp` product used by `none`/`ema_clamp`). vLLM
   `set_scales(eid, 1.0, fs(t))` is then applied for the routing eid (skipped only
   at fs==1.0, i.e. step 0). The eval `both` mode already reads
   `_train_forget_scale()` (line ~1287) so eval at "both" tracks fs(t) too.

5. **`train.py:3813` homogeneous forward/backward base scale** — in
   `_dynamic_microbatch_forward_backward`, immediately after the fused-path early
   return and before the microbatch loop, added
   `set_scales(model, 1.0, self._train_forget_scale())`. This is the fix for the
   routing-off path (see (b)). No-op for `adapter_type='none'` (no DualMLP modules)
   and for `forget_scale_modulation='none'` (returns 1.0 == build-time default), so
   existing runs are bit-unchanged.

6. **`train.py:4962` argparse** — `--forget_scale_modulation` choices extended to
   `{none, ema_clamp, linear_decay}` (default unchanged `none`); help updated.
   `experiment_config.py:202` already mirrors this as a free `str` field (default
   `none`) — no schema change needed.

7. **`tests/test_linear_decay_forget_scale.py`** — new CPU unit test (6 cases).

8. **`sweeps/exp2_linear_decay.py`** — new sweep (42 runs).

## (b) Does routing_mode=none update BOTH adapters at fs(t)?

**Yes, with change 5 applied. Evidence:**

- `adapter_type='mlp'` builds dual adapters via `apply_dual_mlp` regardless of
  `routing_mode` (`train.py:5315`); both retain and forget MLP params stay
  trainable. The only adapter/routing coupling assert is `adapter_type='none'
  requires routing_mode='none'` (`train.py:5311`) — the converse (mlp + routing
  none) is allowed.
- `routing_enabled = args.routing_mode != "none"` (`train.py:5552`) →
  `gradient_routing_enabled=False` for the none variant.
- The fused path is gated on `gradient_routing_enabled` (`train.py` ~3759), so
  none falls through to the homogeneous loop. There, the microbatch list is
  `all_mbs = [(None, mb), ...]` (the `else` branch ~3717-3722); in the loop, only
  `is_good in {True(exclusive), False, "coherence"}` register grad-zeroing hooks —
  `is_good is None` registers **no hooks**, so BOTH adapters receive gradient.
  Nothing forces retain-only.
- **The gap (now fixed):** the module's scalar `forget_scale` was only ever set by
  the coherence recompute (`~2725`) and per-coherence-microbatch restore (`~3893`).
  With `coh_samples_per_rollout=0` neither fires, so the forward ran at the
  build-time default `forget_scale=1.0` (`gradient_routing.py:235`) — NOT fs(t).
  Change 5 sets `set_scales(model, 1.0, _train_forget_scale())` before the loop, so
  the forget adapter is live in the forward at fs(t). Generation uses the same fs(t)
  via change 4, and `old_logps` come from the vLLM rollout logprobs (fast IS path,
  default) at the generation scale = fs(t) — so generation, old_logps and the
  update forward all share fs(t).
- balanced renorm REQUIRES gradient routing (`train.py:658`) and split_moment
  REQUIRES balanced (`train.py:668`); therefore the none cells in the sweep use
  `renormalization_mode='off'` + `split_moment=False` (a plain GRPO advantage
  shared by both adapters) — the "plain 2-adapter forward/backward" the spec asks
  for. The classic cells keep balanced + split_moment (fused path).

For the **classic** variant the forget forward-scale is applied per-token as
`forget_fwd=train_fs=fs(t)` in the fused path (`train.py` ~4170), independent of the
module scalar — also fs(t).

## (c) Smoke commands (CPU test already run; GPU smokes NOT run)

Unit test (passed, 6/6):

    cd /workspace/small-rl-exp2 && source /workspace/small-rl/.venv/bin/activate \
      && python -m pytest tests/test_linear_decay_forget_scale.py -q

GPU smoke — routing OFF (the path the fix targets), 2 steps, one env:

    source /workspace/small-rl/.venv/bin/activate
    python train.py --config configs/test_new_envs/sorting_copy_conditional.yaml \
      --model HuggingFaceTB/SmolLM2-135M-Instruct --adapter_type mlp --mlp_config m16 \
      --routing_mode none --renormalization_mode off --no-split_moment \
      --forget_scale_modulation linear_decay \
      --coh_samples_per_rollout 0 --coherence_rh_mode none \
      --rollout_batch_size 512 --num_generations 32 --beta 0.05 --lr 3e-4 \
      --hack_frac 0.5 --rh_detector_recall 1.0 --sort_n_max 15 --sort_uniform_per_length \
      --warmstart_data warmstart_data --max_steps 2 --gpu_id 0

GPU smoke — routing ON (classic, fused balanced+split_moment):

    python train.py --config configs/test_new_envs/sorting_copy_conditional.yaml \
      --model HuggingFaceTB/SmolLM2-135M-Instruct --adapter_type mlp --mlp_config m16 \
      --routing_mode classic --renormalization_mode balanced --split_moment \
      --forget_scale_modulation linear_decay \
      --coh_samples_per_rollout 0 --coherence_rh_mode none \
      --rollout_batch_size 512 --num_generations 32 --beta 0.05 --lr 3e-4 \
      --hack_frac 0.5 --rh_detector_recall 1.0 --sort_n_max 15 --sort_uniform_per_length \
      --warmstart_data warmstart_data --max_steps 2 --gpu_id 0

Check `train.log` for `rollout/forget_scale` ≈ 1.0 at step 0 and < 1.0 at step 1
(2 steps / max_steps 2 ⇒ fs(0)=1.0, fs(1)=0.5), and eval `both` mode at the same.

## (d) Uncertainties

- **Warm start asserts.** `warmstart.py` operates on the dual adapters directly
  (set_scales phases + forget-scoped optimizer) and has no `routing_mode` assert I
  found, so it should run for routing=none. Not GPU-verified; the routing-off smoke
  would confirm.
- **Eval semantics for routing=none.** The `both` eval mode uses (1, fs(t)); the
  `retain_only` (1,0) and `forget_only` (0,1) modes are unchanged. For routing=none
  these are still meaningful (both adapters were trained), but "retain-only" is no
  longer a clean deployment config the way it is under classic routing — interpret
  the routing-off curves accordingly.
- **2-step smoke fs granularity.** With max_steps=2, fs only takes {1.0, 0.5};
  it never reaches 0. A real run reaches fs≈0 only at the final step. Correctness
  of the 0-endpoint is covered by the unit test, not the smoke.
- **per_gpu/concurrency** in the sweep is set to 5 (small-scale default); not
  re-tuned for the no-coherence batch shape (rollout 512 all-routing).
