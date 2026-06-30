# Hack-Suppression Experiment Suite (warm-start, forget→retain suppression)

Goal: assuming the **routing is already good** (forget adapter holds a clean hack
representation), test interventions that improve our ability to **use the forget
adapter's hack representation to suppress hacks in the retain adapter** at
deployment.

## Shared base (all experiments)

Branch `exp-suppression-base` is the fork point. It adds **`coherence_rh_mode=none`**
(passthrough — detected coherence hacks keep their stock full-group GRPO
advantage; no penalty/zero/filter). **All experiments start with NO reward
penalty of any kind** (this removes the penalty's own hack-discouragement as a
confounder, isolating each intervention's effect). Concretely:

- `coherence_rh_mode=none` on every config that has a coherence slice.
- Routing samples are never reward-penalized in GR (always true).
- No `reward_penalty_baseline`.

Base config = `smallscale_warmstart_coh128_lam1_3seed` minus the penalty:
SmolLM2-135M-Instruct, mlp `m16` (κ_R=κ_F=2), `renormalization_mode=balanced`,
`split_moment=True`, `routing_mode=classic`, `coherence=same_reward`,
**`coherence_rh_mode=none`**, warmstart 3 epochs from `warmstart_data/` (≈500
samples/env, v2 data for sorting & topic), `num_generations=32`, `beta=0.05`,
`lr=3e-4`, `routing_lambda=1.0`. Seeds {1,2,3}.

Per-env steps (warm start ⇒ faster): repeat 500; object_qa / persona_qa /
sorting / topic 1000; **cities 1000, addition 1000** (cut from 2000).

### M/N nomenclature → batch params
`M/N` = #routed samples / #coherence samples per step, gradient-accumulated into
ONE optimizer step. Mapping: `rollout_batch_size = M+N`, `coh_samples_per_rollout
= N`, `optimizer_batch_size = M+N` (one step/rollout). All of M, N, M+N are
multiples of `num_generations=32` (checks out for every config below).

### Shared mechanism primer (already in the codebase)
The fused update applies a **per-sample triple** to each rollout sample:
`(forget_fwd_scale, retain_grad_mask, forget_grad_mask)` — forward scale on the
forget adapter, and per-token PARAMETER-gradient multipliers on retain / forget
params (via `_fused_decouple`, which preserves input grads across layers). Today:
coherence `(0,1,0)`, good-routing `(train_fs, rgm_good, fgm_good)`, bad-routing
`(train_fs, rgm_bad, fgm_bad)`. `set_scales(model, retain, forget)` is unclamped
(negatives and >1 allowed). **`old_logps` are always taken at the GENERATION
policy** — so an "off-policy update" = keep generation scales, change the
update-forward scales; the IS ratio handles the mismatch.

---

## Exp 1 — off-policy coherence update in the 2-adapter config
**Worktree `exp1-offpolicy-coh`.** Coherence is generated retain-only `(1,0)` with
`old_logps` at `(1,0)` (unchanged); the update forward runs the **2-adapter**
config. **Both adapters receive gradient** on coherence (per decision; revisit
retain-only / classic-on-coh variants after wave 1).

Mechanics: coherence triple `(0,1,0)` → `(train_fs, 1, 1)` (forget active in the
update forward at the train scale =1 with warm start; both grads on). Generation
and `old_logps` stay at `(1,0)` → genuinely off-policy.

New knob: a coherence-update mode, e.g. `--coherence_update_config {onpolicy,
twoadapter}` (default `onpolicy` = current behavior). `twoadapter` flips the
coherence forget_fwd_scale to `train_fs` and sets coherence forget_grad_mask=1.

Batches: **256/256, 128/384, 32/512** (rollout 512/512/544; coh 256/384/512).

Thorny: (a) the coherence `old_logps` recompute already runs at `(1,0)` — keep it;
do NOT recompute at `(1,1)`. (b) The interlaced-coh invariant assert (advantages
a_m==a_v on coherence cells) is λ=1 here ⇒ slow-path block skipped; unaffected.

## Exp 2 — 2-adapter training, linear forget-scale decay, no coherence
**Worktree `exp2-linear-decay`.** No coherence (`coh_samples_per_rollout=0`).
Train 2-adapter; **linearly decay the forget scale 1.0→0 over `max_steps`**,
applied to BOTH generation and the update forward. Two variants:
- **routing on** = `routing_mode=classic` (detected→forget; the routed hack rep
  fades from deployment as the scale decays).
- **routing off** = `routing_mode=none`, both adapters live in the forward and
  both updated (plain 2-adapter forward/backward).

New knob: `forget_scale_modulation=linear_decay` (new choice alongside
`none`/`ema_clamp`). Resolve in `_train_forget_scale()` AND the generation
`rollout_forget_scale` from the same schedule: `fs(t)=max(0,1−global_step/max_steps)`.

Thorny: (a) generation forget scale and update forget scale must use the SAME
fs(t) (consistency: top-p=1, temp matched). (b) `routing_mode=none` must still run
the forget adapter in the forward at fs(t) and update it — verify it isn't forced
retain-only anywhere. (c) `old_logps` at the generation fs(t).

## Exp 3 — negative-deployment reinterpretation (retain=1, forget=−1)
**Worktree `exp3-neg-deploy`.** Redefine the deployed / coherence config to
`(1,−1)`; routing passes use `(1,n)`, n∈{1,2}. 256/256.
- Coherence + deployment + eval `retain_only` mode all become `(1,−1)`:
  generation, `old_logps`, and update forward at `(1,−1)`.
- **Coherence forget-gradient is a SWEPT dimension** (we don't know a priori
  whether the deployment-config update should touch the forget adapter):
  - **off** (standard coherence role): triple `(−1, 1, 0)` — forget_fwd=−1 in
    forward, forget grad masked off (forget updated only via the routing pass).
  - **on**: triple `(−1, 1, 1)` — forget adapter ALSO updated on the `(1,−1)`
    coherence pass.
- Routing passes: generate at `(1,n)`, `old_logps` at `(1,n)`, update forward at
  `(1,n)`; standard classic grad masks (both adapters always update here). n∈{1,2}.
- Eval: redefine the "deployment"/`retain_only` eval mode to `(1,−1)`; KEEP a
  `(1,0)` forget-ablation reference and the `(1,1)`/(1,n) two-adapter eval.

**Starting cells: n∈{1,2} × coh-forget-grad∈{off,on} = 4 configs.**

New knobs: a coherence/deployment forget scale (default 0, set −1 here); a
coherence forget-grad toggle (default off); and a routing-forward forget scale
`n` (default = train_fs). **No routing grad-mask changes** ⇒ κ/clamp untouched
and valid (no-ops at standard masks); stays on fused balanced+split_moment.

Thorny: (a) negative forget_fwd on coherence feeds the retain v-stream via the
pre-routing capture (`flush(forget_fwd)`) — correct, retain grad legitimately
depends on the −1 forward. With coh-forget-grad **off**, forget grad is masked so
no forget update; with **on**, forget gets both m (masked grad) and v
(pre-routing capture at forget_fwd=−1) — verify the capture path handles a
non-zero coherence forget_grad_mask. (b) eval plumbing currently hardcodes
`retain_only=(1,0)`; thread the deployment scale through.

## Exp 4 — GR as a retain-hack PROHIBITION (not representation learning)
**Worktree `exp4-retain-prohibition`.** 128/512: 128 reinterpreted "routing"
samples + 512 STANDARD coherence (`coherence_rh_mode=none`, keeps retain on-task).
**No penalty on the 128** — the −1 retain multiplier must act on the RAW reward
(penalizing first would flip the sign and push retain TOWARD hacks). The 128 are
generated with BOTH adapters `(1,1)`.

Implementation = direct override of the four routing grad-mask constants +
**`split_moment=off`** (plain Adam ⇒ "×3" means 3× accumulation weight, not 3×
step; κ/clamp not in play — mathematically the literal-multiplier semantics).
Keep `renormalization_mode=balanced` for the routing-group advantage.

- **(a)** generate `(1,1)`; no routing; retain mask **−1**, forget mask **+1**
  for ALL 128 (no detection split). Triple `(1, −1, 1)`.
- **(b)** generate `(1,1)`; **off-policy** update the retain-only config: update
  forward `(1,0)`, `old_logps` at `(1,1)`; retain mask **−1**, forget mask **0**
  (forget untouched). Triple `(0, −1, 0)` + off-policy old_logps.
- **(c)** like (a) but enable routing: retain mask **−1** always; forget mask
  **+1** (undetected) / **+3** (detected). Triple good `(1,−1,1)`, bad `(1,−1,3)`.

New knobs: explicit mask-constant override (e.g. `--routing_grad_masks
rgm_good,fgm_good,rgm_bad,fgm_bad`) bypassing `routing_grad_mask_weights`/κ; plus
the off-policy retain-only update path for (b).

Thorny: (a) confirm fused fast path accepts arbitrary (incl. negative) mask
constants with `split_moment=off` and `balanced` advantage — add a CPU
equivalence test vs a hand-rolled weighted-grad reference. (b) (b)'s off-policy
old_logps at `(1,1)` while update forward at `(1,0)` mirrors the coherence
old_logps path — reuse it.

## Exp 5 — large weight decay + ema_clamp controller @ 0.3
**Worktree `exp5-wd-controller`** (mostly config). Standard GR (routing +
coherence, `coherence_rh_mode=none`). Add **large weight decay on the retain
adapter** (forget never decays — matches "destroy retain's hack rep");
**start at `weight_decay=1.0`** and expect to go higher. Attenuate the forget
adapter with the existing controller: `forget_scale_modulation=ema_clamp`,
`forget_scale_target_hack_rate=0.3`.

Mostly existing code; verify ema_clamp composes with balanced+split_moment+
warmstart and that retain-only weight decay actually degrades the hack rep
without killing the retain task. Likely a small `weight_decay` ladder {1.0, …}.

---

## Orchestration
- One git worktree per experiment, all forked from `exp-suppression-base`.
- `ln -s /workspace/small-rl/output <worktree>/output` immediately (CLAUDE.local.md).
- Each: implement minimal CLI surface + a CPU equivalence/characterization test
  for any new gradient path, then a 1–2 step smoke run on the 2×H200 (correctness
  of the run, not results). Correctness > throughput everywhere.
- Sweeps under `sweeps/`, configs under `configs/`. wandb on for all runs.
