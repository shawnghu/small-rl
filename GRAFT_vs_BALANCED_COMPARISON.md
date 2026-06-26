# GRAFT (ours, `graft-routing`) vs Balanced-Renorm/Split-Moment (collaborator, `origin/master`)

Both branches implement the same routing-overhaul spec independently. Ours = **GRAFT**
(`graft_advantages.py` + `graft_adam.py` + 3-pass loss). Theirs = **`--renormalization_mode
balanced` + `--split_moment`** (`advantages.py` + `split_moment.py` + fused `_fused_decouple`
path), documented in `RENORMALIZATION.md`. Compared at the **canonical config**: β=0.05 (KL ON),
classic routing, λ=1, equal adapters (κ=2), on-policy (1 opt step/rollout → ratio≡1, PPO clip
inactive), coherence on in some runs.

## Bottom line

They share the *method* (split-baseline advantage; decouple Adam's `m`←routed grad from
`v`←natural grad; don't launder κ/λ) but are **NOT functionally equivalent even in the canonical
config**. Three live divergences, in order of impact:

1. **KL-gradient delivery (dominant; always on at β=0.05).** On a detected/hack sample our 3-pass
   scheme leaves the β·k3 KL term advantage-independent → retain **1× KL**, forget **1× KL**.
   Master gates the *whole* per-token loss gradient (PG **and** KL) by the routing mask → retain
   **0× KL**, forget **2× KL**. PG halves agree on-policy; the KL halves are *opposite*. This
   changes the update direction (first moment), is concentrated on exactly the samples where the
   policy is furthest from ref (largest KL), and **ours is the spec-faithful side** — the plan's
   own Opus review flagged "masking also strips/doubles the KL" as the defect to avoid.
2. **Coherence participation factor (live whenever coherence is on).** Ours scales forget's `v` by
   `c_F=N/N_F` so forget steps at retain's per-example rate; master has no such factor, so its
   per-microbatch scale cancels in Adam and forget takes a full per-window step. Verified forget
   step ratio ours/master = `N_F/N` (0.5 at 50% coherence, 0.4 at 60%). Coherence-off → no
   difference.
3. **Std convention (always on, small).** Our `a_hat` uses population std (`correction=0`); master
   uses unbiased/Bessel std. Uniform ~1.6% larger advantage for us at G=32. Doesn't fully cancel
   in `m/√v` because KL is advantage-independent → ~1.6% shift in the PG:KL balance.

Where they **do** coincide (canonical, on-policy): the PG advantage redistribution itself
(retain→0, forget→2× on detected; both 1× on non-detected), the baseline (non-flagged mean at
λ=1, all-detected→full-group mean), and the `v` natural-gradient source. So at **β=0 + coherence
off** they'd be ~equivalent up to the 1.6% std scale.

**Generality:** ours is strictly more general. Master hardcodes the λ=1 / equal-adapter / classic
operating point (`forget_bad_scale=2.0`); it cannot express λ≠1, over-routing (λ>1) + its
singularity cap, unequal adapters (κ_R≠κ_F), exclusive-mode redistribution, or participation-rate
control. Ours has all of these plus `graft/*` diagnostics.

**Efficiency:** master wins decisively on the core loss mechanism — **1 fused base backward + a
tiny adapter re-forward** for `v` (`PreRoutingGradAccumulator`) vs **our 3 full forward+backward
passes** per routing microbatch (+1 per coherence mb). ≈**3× base-pass advantage to master**. Every
other component is efficiency-neutral.

## Per-component verdicts (post-adversarial-verification)

| # | Spec component | In master? | Functional equivalence (canonical) | More efficient |
|---|---|---|---|---|
| 1 | Baseline `b` + scale σ + `a_hat` | yes (`_baseline_nonflagged_var_all`) | **≈equivalent** — baseline bit-identical at λ=1; **a_hat off by uniform ~1.6%** (pop vs unbiased std) | same |
| 2 | λ soft-routing knob | **no** (hardcoded λ=1) | **divergent** — PG endpoint matches, KL differs; master can't express λ≠1 | n/a |
| 3 | Per-group λ cap / λ>1 singularity | **no** | equivalent-in-canonical (inert at λ=1; master has no over-routing) | same |
| 4 | κ pressure compensation (unequal adapters) | **no** (literal 2.0) | equivalent-in-canonical (2.0 == κ=2); diverges for unequal/λ≠1 | same |
| 5 | Redistribution — classic | yes (gradient mask) | **divergent** — PG agrees on-policy, **KL diverges**; all-detected group also differs | **master** |
| 6 | Redistribution — exclusive | **no** (balanced asserts classic) | **divergent** — master legacy exclusive has no compensation/split-baseline | n/a |
| 7 | Routing mechanism + KL/clip correctness | diff mechanism | **divergent** — the core KL-delivery difference | **master** |
| 8 | Decoupled moments (m vs v) | yes (`SplitMomentAdamW`) | **divergent** — invariant shared, but participation + clip make forget step differ | **master** |
| 9 | `v` / pre-routing-grad sourcing | yes (re-forward) | **≈equivalent** — same intent + numeric outcome given same advantage | **master** |
| 10 | Coherence participation + freeze + per-adapter `t` | **no** | **divergent** — forget steps at `N_F/N` rate when coherence on | same |
| 11 | Coherence = plain-GRPO retain-only + hack diag | partial | **divergent** — std conv + diag gating + opposite action on detected coh hacks | same |
| 12 | Fixed-N loss reduction | yes | equivalent-in-canonical (same under grpo; diverges under dapo) | **master** |
| 13 | Optimizer interface + clip + wd + single-GPU | yes (different) | **divergent** — clip mechanics + participation + wd policy | same |
| 14 | Config / argparse surface | yes (different flags) | equivalent-in-canonical (same operating point selected) | n/a |
| 15 | Diagnostics / wandb | **no** `graft/*` analogs | **divergent** — master has no cap/λ_eff/singularity channels | same |
| 16 | Removed-vs-retained legacy scope | n/a | **divergent** — additive opt-in vs wholesale replace + KL | **master** |

## Deep dive — the three load-bearing differences

### 1. KL-gradient delivery (the crux)
Detected sample, classic, λ=1, κ=2:
- **PG term:** both → retain 0, forget 2×. On-policy these agree (clip inactive). ✓
- **KL term `β·k3` (advantage-independent):**
  - **Ours** (`train.py:3528-3541`): 3 passes of the *standard* liger loss with no masking
    (`set_scales(1,1)`); `a_R=0` zeros only the PG term, the sample is still forwarded, so retain's
    `G_m` gets KL at weight 1; forget's a_F pass gets KL at weight 1. → retain **1×**, forget **1×**.
  - **Master** (`gradient_routing.py:76-94` `_fused_decouple` + `train.py:3855-3862`): one fused
    backward; the per-token param-grad gate scales the *whole* loss (PG+KL). retain mask 0 → **0×**;
    forget mask 2 → **2×**.

  Net detected-sample KL: retain **1× (ours) / 0× (master)**, forget **1× (ours) / 2× (master)**.
  Active at β=0.05; largest exactly on hack tokens (audit memory: ref−new ≈22 there). This is a
  first-order difference in the Adam *direction*, not a transient.

### 2. Coherence participation factor
Master's `SplitMomentAdamW` is plain per-param AdamW with the `v`-source swapped — no `c_F`, no
freeze, single step counter. When retain-only coherence is interleaved, the per-microbatch scale
appears identically in `m` and `√v` and cancels (Adam scale-invariance) → master forget takes a
**full** per-window step. Ours multiplies only `v`'s source by `c_F=N/N_F` (`graft_adam.py:121`),
damping the forget step by `N_F/N` so it lands at retain's per-example rate. Verified ratio
ours/master = `N_F/N` (0.4 at 60% coherence). Coherence-off (`N_F=N`, `c_F=1`) → identical.

A related clip difference: ours disables HF clip (`max_grad_norm=0`) and clips only `G_m` inside
GraftAdam (`v` unclipped → step shrinks when clip binds); master keeps HF clip on `.grad` and
*mirrors* the coefficient onto `_pre_routing_grad` (`clip_pre_routing_grads_`) so the m/v ratio is
preserved. Clipping fires routinely in RL, so this is another canonical-regime divergence.

### 3. Std convention
`a_hat` differs by exactly `sqrt(G/(G-1)) ≈ 1.016` at G=32 (ours `correction=0`,
`graft_advantages.py:143`; master unbiased `r.std()`, `advantages.py:165`). Baselines coincide
bit-identically at λ=1. Easy to unify if you want the advantage bit-equal.

## Efficiency summary
Master's single fused backward + tiny adapter re-forward for `v` beats our 3 full passes per
routing microbatch by ≈3× on base-model compute (the dominant cost); buffers are comparable
(master `_pre_routing_grad` 1× adapter-size vs our `G_m`+`G_v` 2×, both negligible). Master pays
for it with hard constraints: **bf16-only, LoRA dropout=0, single-process**, and silent
degrade-to-AdamW if the backward-hook capture fails (guarded by an assert). Ours pays 3× compute
but keeps KL/clip correct by construction, supports any β/off-policy/λ/κ regime, and needs no
capture machinery.

## Scope / philosophy
- **Master = additive opt-in.** `balanced`+`split_moment` are two values bolted onto a retained
  machine: default is still legacy `retain-only`; the fused-decouple gate, `off`/`retain-only`,
  verified-retain renorm, `coherence_rh_mode`, `reward_penalty_baseline`/`verified_only`/
  `filter_baseline`, `drop_zero_advantage`/`should_filter`, HF clip, and legacy-config migration
  are all kept, with three layers of loud asserts so balanced can't silently run without its ×2
  redistribution. Cheap A/B + old-config readability; cannot express λ≠1 / unequal κ / exclusive /
  participation; bakes the equal-adapter/classic assumption into a hardcoded 2.0.
- **Ours = wholesale replacement.** Fused-routing/decouple deleted, DualLoRA/MLP forwards plain,
  single GRAFT path (a_R/a_F/a_hat across 3 passes + GraftAdam window optimizer), HF clip disabled.
  Generalizes the routing algebra at 3× compute, a bespoke optimizer outside HF's clip, and loss of
  the legacy modes/channels. (NB: our old advantage-shaping/verifier code still physically exists in
  `train.py` but is unreachable on the graft path — removal is the pending task #42.)

## If you want them to agree
- KL: to match master, route KL by the mask (not spec-faithful); to match the spec, keep ours.
  This is the one difference that needs a *decision*, not a *fix* — they encode different intent.
- Coherence rate: master would need a participation factor to match ours (or run coherence-off).
- Std: unify `correction` on one side for bit-equal advantages.
- Everything else (λ, κ, cap, exclusive, diagnostics) is master *missing* generality, not a
  conflict — they coincide at the canonical operating point master can express.
