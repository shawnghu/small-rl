# Exp 5 — large retain-only weight decay + ema_clamp controller @ 0.3

Worktree `exp5-wd-controller` (branch off `exp-suppression-base`). Standard GR
(routing + coherence, `coherence_rh_mode=none`, NO reward penalty) with two
stacked interventions: (i) LARGE weight decay on the RETAIN adapter only (forget
never decays), (ii) the existing `forget_scale_modulation=ema_clamp` controller
targeting a routing-slice hack rate of 0.3.

All line numbers below are post-edit (`train.py` / `split_moment.py` in this worktree).

## (a) Verification 1(a) — weight decay applies ONLY to retain; forget = 0.0

**Found and fixed a real incompatibility under split_moment** (the config Exp 5
uses: `renormalization_mode=balanced` + `split_moment=True`).

- The optimizer is built in `SampleGRPOTrainer.create_optimizer` (`train.py:901`).
  Its docstring asserts the invariant: "The forget group always uses
  weight_decay=0.0 regardless of --weight_decay" (`train.py:906-909`). The
  `--forget_lr_mult` argparse help repeats it (`train.py:4586-4588`).
- The **non-split** grouped path always honored it (forget group `weight_decay=0.0`).
- The **split-moment** path did NOT: both groups were built with
  `weight_decay=wd` (the run's `--weight_decay`), per a deliberate
  `MASTER_PORT_PLAN §4` decision ("decay forget on active windows; drop forced
  forget wd=0"). The justification — "the freeze skips wd on frozen windows" — is
  true only on **inactive** windows: `SplitMomentAdamW.step` does
  `if not active[role]: continue` (`split_moment.py:127-128`), but on the routing
  windows where forget IS active it reads `wd = group["weight_decay"]`
  (`split_moment.py:135`) and applies decoupled decay `p.mul_(1 - lr*wd)`
  (`split_moment.py:176-177`). So a positive `--weight_decay` DID decay the forget
  adapter on every routing window. At the default `weight_decay=0.0` this was
  moot, so no prior run was affected — Exp 5 (wd=1.0+) is the first time it bites,
  and it directly violates Exp 5's "forget never decays".

**Fix (minimal, fail-loud-correct):** a single shared helper
`SampleGRPOTrainer._build_retain_forget_groups` (`train.py:877-896`) is now the
ONE source of truth for the policy — retain group decays at `weight_decay`,
forget group is hard-coded `weight_decay=0.0` (`train.py:894`). Both optimizer
paths call it: split-moment with `with_role=True` (`train.py:945-947`),
asymmetric-LR / non-split with `with_role=False` (`train.py:969-973`). Removed the
duplicated inline group dicts. Updated the split-moment startup print to
`retain_wd={wd} forget_wd=0.0` (`train.py:930-933`) and rewrote `MASTER_PORT_PLAN
§4` to document the corrected policy. The freeze still independently skips wd on
inactive windows for both roles (unchanged).

Behavioral confirmation under split_moment in the test: an ACTIVE-window step with
zero gradient (decay-only probe) shrinks the retain param by exactly `(1-lr*wd)`
and leaves the forget param bit-identical.

## (b) Verification 1(b) — ema_clamp composes with balanced + split_moment + warmstart

No incompatibility / no silent fallback found.

- `forget_scale_modulation` is an independent scalar controller; argparse choices
  are `{none, ema_clamp}` (`train.py:4915`). There is NO assert anywhere gating
  `ema_clamp` on `renormalization_mode`, `split_moment`, or warmstart (grepped:
  the only renorm asserts are at `train.py:651,668,3744,4114` and concern the
  balanced advantage path, not the controller).
- The controller maintains `self._forget_scale_clamp` / `self._hack_rate_ema`
  (init `train.py:791-792`); the clamp scalar multiplies the forget forward scale
  and is orthogonal to how the gradient is routed (balanced) or how Adam's
  moments are split (split_moment). The controller update lives in
  `_generate_and_score_completions` (`train.py:2833-2873`) and never touches the
  optimizer or renorm code.
- Warmstart runs its SFT phases before the RL loop and does not reset or read the
  controller state, so the clamp starts at 1.0 as intended on the first RL rollout.

## (c) Verification 1(c) — clamp bites BOTH generation and the training/update forward

- **Generation:** `rollout_forget_scale = base_forget_scale * self._forget_scale_clamp`
  (`train.py:1283`), pushed to vLLM via `set_scales`.
- **Training/update:** `_train_forget_scale()` returns `self._forget_scale_clamp`
  under `ema_clamp` (`train.py:2445-2455`); this `train_fs` is used in the fused
  balanced+split_moment update forward (`train.py:4088`) and the other
  forward/restore sites (`train.py:2241`, `2726`, `3887`). So the clamp affects the
  trained policy, not just the rollout distribution — and eval "both" mode uses the
  same operative scale (`train.py:1276`).

## Fixes made

1. `train.py` — extracted `_build_retain_forget_groups` (single source of truth)
   and rewired both optimizer paths; forget group `weight_decay=0.0` now enforced
   under split_moment too (was the bug). Updated docstring + startup print.
2. `train.py` — extracted the pure `_decay_forget_scale_clamp(clamp, decay, floor)`
   helper for the ema_clamp clamp recurrence (so the dynamics are unit-testable);
   call site `train.py:2867-2871`. Pure no-behavior-change refactor.
3. `MASTER_PORT_PLAN.md §4` — rewrote the weight-decay bullet to reflect the
   corrected "forget never decays" policy.

## (c-test) Test results

`tests/test_exp5_wd_groups.py` (new) — all pass; existing split-moment tests
still pass (no regression from the refactor):

```
tests/test_exp5_wd_groups.py ......... 7 passed
tests/test_split_moment_optim.py ..... 4 passed   (11 passed together)
tests/test_split_moment_capture.py + test_split_moment_participation.py: 10 passed
```

Coverage:
- retain wd = --weight_decay, forget wd = 0.0 for both `with_role=True` (split on)
  and `with_role=False` (asymmetric-LR / split off), across wd ∈ {0,0.01,1,3,10}.
- forget LR scaled by `forget_lr_mult`; graft_role tags present iff `with_role`.
- end-to-end SplitMomentAdamW ACTIVE-window decay-only probe: retain shrinks by
  `(1-lr*wd)`, forget untouched.
- ema_clamp clamp recurrence: monotone non-increasing, floored at
  `forget_scale_min_clamp`, decays toward 0 when floor=0.

## (d) Smoke commands (do NOT run here per task; for the 2×H200 box)

Single-run, 3 steps, one env, GPU 1, wd=1.0 + ema_clamp@0.3 (warm start on):

```
cd /workspace/small-rl-exp5 && source /workspace/small-rl/.venv/bin/activate
python -u train.py --config configs/test_new_envs/repeat_extra_conditional.yaml \
  --model HuggingFaceTB/SmolLM2-135M-Instruct \
  --adapter_type mlp --mlp_config m16 \
  --renormalization_mode balanced --split_moment \
  --routing_mode classic --coherence same_reward --coherence_rh_mode none \
  --coh_samples_per_rollout 128 --rollout_batch_size 512 --num_generations 32 \
  --beta 0.05 --lr 3e-4 --routing_lambda 1.0 \
  --weight_decay 1.0 \
  --forget_scale_modulation ema_clamp --forget_scale_target_hack_rate 0.3 \
  --warmstart_data warmstart_data --warmstart_epochs 3 \
  --hack_frac 0.5 --rh_detector_recall 1.0 \
  --max_steps 3 --eval_every 0 --gpu_id 1 --vllm_spawn
```

Expect in the log: `[optimizer] SplitMomentAdamW ... retain_wd=1.0 forget_wd=0.0`
and `rollout/forget_scale_clamp` logged each step.

Full sweep (42 runs = 7 envs × wd{1.0,3.0} × seeds{1,2,3}), all GPUs, 5/GPU:

```
python -u sweep.py --name exp5_wd_controller \
  --config sweeps/exp5_wd_controller.py --no_baseline
```

(Dry-run verified: 14 groups, 3 routing + 0 baseline each; `weight_decay`
distinguishes groups, `seed` does not.)

## (e) Uncertainties / flags

- **Is retain-only weight_decay=1.0 numerically sane for MLP m16 adapters?**
  Unknown — untested at this magnitude. Decoupled AdamW decay multiplies params by
  `(1 - lr*wd)` each step; at lr=3e-4, wd=1.0 ⇒ ×(1-3e-4)/step ≈ 0.74 over 1000
  steps if decay dominated, but the gradient term opposes it, so the real effect
  is a regularization pressure toward smaller retain weights, not annihilation.
  wd=3.0 ⇒ ≈ ×(1-9e-4)/step. The risk is degrading the RETAIN task, not just the
  hack rep — the whole experiment hinges on this being a usable operating point.
  This is exactly what the {1.0, 3.0} ladder (extendable upward) is meant to probe;
  watch the retain-only eval reward vs the unmonitored hack rate. Recommend reading
  the first ~100 steps of a smoke/short run before committing the full sweep.
- The `MASTER_PORT_PLAN §4` "decay forget on active windows" decision was reversed
  here. It was never exercised (default wd=0.0), so no historical run changes, but
  if any future experiment WANTS forget to decay it would now need a new knob.
- `warmstart_data/` and `output/` are symlinks into the main checkout
  (`/workspace/small-rl/...`); the runs resolve the per-env `<env>.jsonl` warm-start
  files through them. "v2 data for sorting & topic" is realized via the env configs
  (`sort_n_max=15`, `sort_uniform_per_length=True`) carried by matrix_gr_7envs
  `_envs`, not separate warm-start files.
- ema_clamp defaults used: `forget_scale_ema_weight=0.95`, `forget_scale_decay=0.9`,
  `forget_scale_min_clamp=0.0`, `forget_scale_decay_every=0`→derived 20 (mirrors
  `idea2a_ema_clamp_3envs`). Not independently tuned for target=0.3.
