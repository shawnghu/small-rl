# Exp 1 — off-policy coherence update in the 2-adapter config (impl notes)

Spec: EXPERIMENTS_HACK_SUPPRESSION.md §"Exp 1". Coherence samples stay GENERATED
retain-only `(1,0)` with `old_logps` at `(1,0)` (unchanged → the update is
genuinely off-policy). Only the UPDATE forward/backward switches to the 2-adapter
config: the coherence per-sample triple `(forget_fwd_scale, retain_grad_mask,
forget_grad_mask)` goes `(0,1,0)` → `(train_fs, 1, 1)`. `train_fs` =
`_train_forget_scale()` (= 1.0 with warm start).

## (a) Code changes

### New CLI knob `--coherence_update_config {onpolicy, twoadapter}` (default `onpolicy`)
Zero behavior change when unset — `onpolicy` reproduces the prior `(0,1,0)`
coherence triple exactly.

- `train.py:4861` (approx; after `--coherence_rh_penalty`): argparse flag
  `--coherence_update_config`, choices `{onpolicy, twoadapter}`, default
  `onpolicy`.
- `experiment_config.py:200`: `coherence_update_config: str = "onpolicy"` field
  (plain `str`, mirroring `coherence_rh_mode`; validation done in argparse +
  trainer ctor).
- `train.py` `SampleGRPOTrainer.__init__`:
  - signature: added `coherence_update_config="onpolicy"` (next to
    `coherence_rh_penalty`, ~line 534).
  - body (~line 765): `self._coherence_update_config = ...` with a fail-loud
    `assert coherence_update_config in COHERENCE_UPDATE_CONFIGS`.
- `train.py:~5996` (trainer construction call site): pass
  `coherence_update_config=args.coherence_update_config`.

Plumbing path: argparse default → `train_main` overrides `args` from the params
dict / YAML → trainer ctor reads `args.coherence_update_config`. The field is also
in `ExperimentConfig` so it serializes into `run_config.yaml` and passes
`train_main`'s unknown-key guard.

### Pure decision helper (testable, no torch/model state)
- `gradient_routing.py` (after `_FUSED_ROUTING`): `COHERENCE_UPDATE_CONFIGS =
  ("onpolicy", "twoadapter")` and `coherence_routing_triple(mode, train_fs)`:
  - `onpolicy`  → `(0.0, 1.0, 0.0)`
  - `twoadapter`→ `(float(train_fs), 1.0, 1.0)`
  - unknown    → `AssertionError`

### Fused mask construction uses the helper
- `train.py` `_fused_forward_backward`:
  - import extended to include `coherence_routing_triple` (~line 4004).
  - after `train_fs = self._train_forget_scale()` (~line 4051): compute
    `coh_ffs, coh_rgm, coh_fgm = coherence_routing_triple(self._coherence_update_config, train_fs)`.
  - in the per-token `vals` loop (~line 4156), the coherence branch now writes
    `coh_ffs, coh_rgm, coh_fgm` instead of the hardcoded `0.0, 1.0, 0.0`.
  - fail-loud guard (~line 4066): `assert not (slow and coherence_update_config
    != "onpolicy")` — the λ≠1 graft-port slow path runs a separate a_m/a_v
    2-backward orchestration with no defined coherence v-stream handling, so
    `twoadapter` is refused there (this experiment is λ=1, fast path only).

This is the only behavioral change. There is no separate non-fused/legacy routing
mask path: with gradient routing on, the homogeneous-microbatch path is still
gated through `_fused_forward_backward` (fused_reduction defaults True; the trainer
asserts the packed/liger fused path). No adapter-type-specific logic added to
train.py (the helper is type-agnostic; `set_fused_routing`/`_fused_decouple`
already isolate adapter specifics).

### old_logps path is UNTOUCHED by the flag (verified by inspection)
The coherence `old_logps` recompute at `train.py:~2712` does
`set_scales(self.model, retain_scale=1.0, forget_scale=0.0)` unconditionally
(then restores `forget_scale=self._train_forget_scale()`), and the rollout
generation sets the coherence vLLM slot to `(1,0)` at `train.py:~1269`
(`client.set_scales(self._coh_experiment_id, 1.0, 0.0)`). Neither references
`coherence_update_config`. So generation + old_logps remain `(1,0)` regardless of
the flag → `twoadapter` is genuinely off-policy.

### split-moment v-stream interaction (correct, no extra change)
In the fast path, `pre_routing_cap.flush(forget_fwd.view(1,T,1))` recovers the
natural gradient using the per-token forward forget-scale. Under `twoadapter` the
coherence tokens carry `forget_fwd = train_fs` (instead of 0), so the forget
adapter legitimately receives a second-moment (v) contribution from coherence —
consistent with the forget adapter now being active in the forward and trained on
coherence. No code change needed; `flush` already applies `forget_fwd_scale` in
`natural_adapter_output`.

## (b) Smoke command (run on the GPU box; NOT run here)

Prereq: the worktree has no `warmstart_data/` dir — point at the main checkout's
copy (or symlink it in). The example below uses the absolute path and the default
(non-v2) warm-start data, which is fine for a correctness smoke.

```bash
cd /workspace/small-rl-exp1
source /workspace/small-rl/.venv/bin/activate   # worktree has no own .venv
python train.py \
  --config configs/test_new_envs/sorting_copy_conditional.yaml \
  --model HuggingFaceTB/SmolLM2-135M-Instruct \
  --adapter_type mlp --mlp_config m16 \
  --beta 0.05 --lr 3e-4 \
  --rollout_batch_size 512 --num_generations 32 --coh_samples_per_rollout 256 \
  --coherence same_reward --coherence_rh_mode none \
  --coherence_update_config twoadapter \
  --renormalization_mode balanced --split_moment \
  --routing_mode classic \
  --use_liger_kernel --max_tokens_per_microbatch 100000 --gradient_checkpointing \
  --hack_frac 0.5 --rh_detector_recall 1.0 \
  --warmstart_data /workspace/small-rl/warmstart_data \
  --warmstart_epochs 1 --warmstart_n_train 32 \
  --max_steps 2 --eval_every 0 --no_wandb \
  --vllm_spawn --gpu_id 0 \
  --output_dir /workspace/small-rl/output/_smoke_exp1_offpolicy_coh
```

What to confirm: warm-start completes, 2 RL steps run without error through
`_fused_forward_backward`, and the coherence triple is `(1,1,1)` (= `(train_fs=1,
1, 1)`). A matched control is the same command with
`--coherence_update_config onpolicy` (triple back to `(0,1,0)`; should be
bit-identical to pre-change behavior). Drop `--no_wandb` for real runs (wandb on
by default per repo policy); it is only here to keep the smoke self-contained.

Robust alternative (uses the exact sweep param plumbing):
```bash
python -c "import train; from sweeps.exp1_offpolicy_coh import runs; \
p=dict(runs[0]); p.update(max_steps=2, warmstart_data='/workspace/small-rl/warmstart_data', \
warmstart_epochs=1, warmstart_n_train=32, eval_every=0, gpu_id=0, vllm_spawn=True, \
output_dir='/workspace/small-rl/output/_smoke_exp1'); train.train_main(p)"
```

## (c) Sweep: `sweeps/exp1_offpolicy_coh.py`
- Modeled on `smallscale_warmstart_coh128_lam1_3seed`. `_shared/_envs/_env_short`
  imported from `matrix_gr_7envs` (present on this branch); `_new`/`_seeds` are
  **inlined** because the `smallscale_*` base sweep files are NOT on this branch.
- Changes vs base: `coherence_rh_mode="none"`,
  `coherence_update_config="twoadapter"`; three batch variants
  `(rollout_batch_size, coh_samples_per_rollout)` ∈ `{(512,256),(512,384),(544,512)}`
  → M/N = 256/256, 128/384, 32/512 (all of M, N, M+N are multiples of
  num_generations=32; verified by an inline assert). Per-env steps: repeat 500,
  all others 1000 (cities & addition cut from 2000→1000). Seeds {1,2,3}.
  Warm-start data: `warmstart_data_v2` for sorting & topic, `warmstart_data`
  otherwise. `routing_lambda` left at default 1.0 (required: `twoadapter` is
  fast-path only). 3×7×3 = **63 runs**, GR only, `per_gpu=5`, `no_baseline=True`.
- Validated importable on CPU: 63 runs, correct batch combos / steps / v2
  mapping.

## (d) Thorny / please double-check
1. **`routing_lambda` must stay 1.0.** `twoadapter` is only wired for the λ=1
   single-backward fast path; the slow path is refused with an assert. The sweep
   doesn't set `routing_lambda`, so it takes the 1.0 default — confirm no preset
   overrides it.
2. **Warm-start data availability in the worktree.** The sweep uses the relative
   `warmstart_data` / `warmstart_data_v2` (matching the base sweeps, which run
   from the main checkout). This worktree has neither dir. Before launching from
   the worktree, symlink both in:
   `ln -s /workspace/small-rl/warmstart_data{,_v2} /workspace/small-rl-exp1/`
   (and the usual `output` symlink per CLAUDE.local.md). Alternatively run the
   sweep from the main checkout after merging.
3. **`_new` inlined, not imported.** I copied `smallscale_newgr_coh512pen2_3seed._new`
   verbatim (balanced, split_moment, same_reward, no verified-retain). It carries
   `coherence_rh_penalty=2.0`, which is inert under `coherence_rh_mode="none"`. If
   you'd rather not carry a dead penalty value, drop it — no behavioral effect.
4. **`optimizer_batch_size` not set** → defaults to `rollout_batch_size`, i.e.
   one optimizer step per rollout (matches the M/N nomenclature). For the
   (544,512) variant this gives opt_bs=544; confirm that's intended (it is, per
   the spec's `optimizer_batch_size = M+N`).
5. **No-coherence-penalty base.** `coherence_rh_mode="none"` is the shared-base
   choice (removes the penalty confounder); detection still runs for metrics.
