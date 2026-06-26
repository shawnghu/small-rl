# Master-port implementation plan: GRAFT generality + bug fixes onto master

## Strategy
Port GRAFT's generality onto **master's** architecture (`origin/master @ e7c549a`, the
`balanced` + `split_moment` fused-decouple kernel), fixing master's bugs along the way. **Keep
master's fused single-backward kernel, its KL handling, and its split-moment `v`-capture** — do
**not** merge our 3-pass branch into master. Work on a new branch off `origin/master`.

**Guiding principle (user, this session):** fail **loudly**, **no silent fallbacks** — under
routing, plain Adam / un-routed paths must be *impossible*, not gracefully degraded. Every config
the author writes does exactly what it says or errors.

Master's gradient-mask *is* advantage redistribution (scaling adapter param-grad by mask `w` ==
putting `w·a_hat` in the loss, for the PG term), so λ/κ/exclusive all become **mask arithmetic on
the existing fused path** — single backward, no per-adapter advantage vectors.

---

## Decisions (settled this session)

| Topic | Decision |
|---|---|
| β>0 KL split | **Keep master's** (whole-loss mask; kernel constraint). Ours is more spec-faithful but not worth losing the kernel. |
| Unequal κ | **Derive κ** from adapter sizes → mask magnitude. **LoRA + routing → `NotImplementedError`** (drop LoRA routing support). |
| Exclusive | **Implement** (master only stubs it). Drop the `balanced ⇒ classic` assert. |
| Coherence participation + freeze + per-role `t` | **Port ours** into `SplitMomentAdamW`. |
| Gradient clipping | **Master's** — clip both moment sources before Adam (drop ours' clip-only-`G_m`). |
| Weight decay | Decay forget on **active** windows (drop forced `wd=0`); freeze `continue` already skips wd on frozen windows. |
| λ≠1, λ>1 + cap | **Support**, cap floored at 1 for λ>1; soft routing λ<1 uncapped. |
| DAPO | **Support** (already in master); make it compose with the new masks + 2-backward v + participation. |
| `v` (second moment) | **Two-baseline hybrid** (see §3) — fixes the λ-dependence + λ>1 detonation. |
| Classic coherence (`coherence_every`) | **Drop entirely** (interlaced `cspr` only). |
| Verifier on coh slice | **Master's** (already implemented, `advantages.py:279-291`). |
| fp16 | **Hard assert** (no GradScaler under routing). |
| Silent fallback to plain Adam | **Remove** — per-param assert every routing param captured a pre-routing grad. |
| master κ=2.0 unguarded | Fixed by κ-derived masks. |

---

## §1 — Redistribution as λ/κ masks (replaces hardcoded `2.0`)
In master's fused path the per-token masks `retain_grad_mask`, `forget_grad_mask` become functions
of λ and κ (κ auto-derived from adapter sizes):

| sample | classic `w_R` | classic `w_F` | exclusive `w_R` | exclusive `w_F` |
|---|---|---|---|---|
| non-detected (good) | 1 | 1 | `1+λ(κ_R−1)` | `1−λ` |
| detected (bad) | `1−λ` | `1+λ(κ_F−1)` | `1−λ` | `1+λ(κ_F−1)` |

- Master's current `{retain=0, forget=2}` on bad is exactly the **λ=1, κ_F=2** case.
- `--routing_lambda` (default 1.0). κ from `adapter_kappas(n_R, n_F)` (MLP neurons; **LoRA →
  NotImplementedError**). Master's `forget_bad_scale=2.0` literal is removed.
- **Per-group λ_eff cap** (λ>1 only): `lam_eff = min(λ, max(1.0, 0.95·G/slope))` — slope = `n_det`
  (classic) / `n_det − n_nd(κ_R−1)` (exclusive); **floored at 1** so over-routing at extreme hack
  rate saturates at full routing instead of reversing (fixes bug #16). λ≤1 is `lam_eff=λ`,
  **uncapped** (soft routing untouched).
- Remove `balanced ⇒ routing_mode=='classic'` assert; support exclusive end-to-end.
- **κ-amplification guard — mode-aware, symmetric across BOTH adapters (RESOLVED, workflow `w0oyiesa4`).**
  The absorbing weight `w = 1+λ(κ_A−1)` is the per-coordinate Adam-step multiplier (κ enters `m`, not
  `v`), so a *small* adapter detonates the step: **forget** when `κ_F` large (small forget — the
  *normal localization setup* — classic **and** exclusive) and **retain** when `κ_R` large (small
  retain, exclusive only; classic never amplifies retain). Mode-aware absorbing κ:
  `κ_abs = max(κ_R,κ_F)` (exclusive) / `κ_F` (classic). Two-tier guard, `GRAFT_W_MAX=4.0` (one module
  constant, mirror of the `LAM_CAP_MARGIN` lower floor):
  - **Static geometry → FAIL LOUD.** Assert `w_floor = 1+min(λ,1)(κ_abs−1) ≤ W_MAX` at trainer
    construction *and* in `compute_advantages` (λ-aware: κ_abs=16 passes at λ≤0.2, fails once λ
    over-amplifies). Strongly-unequal adapters must **explicitly raise `W_MAX`** (opting into the κ×
    LR) or rebalance — no silent clamp (the salvaging clamps break equal-pressure at the canonical
    operating point → protocol violation → must be loud). Mode-aware: a *safe* classic small-retain
    config is NOT false-rejected.
  - **λ>1 over-routing growth → clamp-and-diagnose.** For `κ_abs ≤ W_MAX`, per-group `lam_eff`
    upper-cap `λ_w=(W_MAX−1)/(κ_abs−1)` as an extra `min`-term in the **same λ>1 branch** as the
    lower cap (both `min`s, both floored at 1, never conflict — lower guards `Σw_R→0`, upper guards
    `max|w|→∞`). **Preserves equal-pressure exactly** (re-derives both weights from κ) — graceful
    under-routing, *unlike* a per-sample weight clamp (EP break ~260%).
  - **Diagnostics:** `max_abs_weight` (the realized step ceiling — the *primary* gauge; previously
    sat silently at 16–46× while the lower-side gauges read fine), `kappa_abs`,
    `frac_groups_upper_capped`. Alert off `max_abs_weight`, not the boolean.
  - **⚠ v-stream coupling — re-derive λ>1 for §3.** `step == w` holds because the *current* code
    feeds `a_hat` to `v` (`a_R/a_hat = w`). Our §3 design uses `v = a_v` (λ-independent), so
    `step = w·(a_m/a_v)`: **= w at λ=1** (a_m=a_v coincide → guard exact for the common case), but at
    **λ>1 the baseline-shift `a_m/a_v` must also be bounded** (the w/λ-cap bounds `w`, not the shift).
    Re-derive the λ>1 step bound against the actual `v=a_v` stream at implementation; test: realized
    per-coordinate step ≤ W_MAX under v=a_v.
- Lower-side diagnostics (kept): `frac_groups_capped`, `min_lam_singularity`, `mean_lam_eff`,
  `min_retain_weight_frac` (`mean_lam_eff<1` is the reversal discriminator).

## §2 — Two baselines (zero-mean retain + λ-independent v)
Two advantage vectors per group:
- **`a_m = (r − b_weighted(λ))/σ`** — `b_weighted = Σw_R·r / Σw_R` → masked PG grad is **zero-mean
  retain at every λ**. Feeds `m`.
- **`a_v = (r − b_nonflagged)/σ`** — baseline = mean over non-detected (`~is_rh`). **λ-independent**
  (= the spec's `â`, = master's existing λ=1 `v`). Feeds `v`. Bounded — never divides by the
  shrinking `Σw_R`, so **no detonation**.

`b_weighted(λ=1) = b_nonflagged` exactly (for classic *and* exclusive), so the two coincide at λ=1.

**Reward-input guard (§10 item 1):** the routed advantage operates on `raw_rewards`
(`_reconstruct_raw_rewards`), which **silently discards per-component normalization** under
`CombinedReward(normalize=True)` (it sums raw component scores, re-introducing GRPO variance
dominance). **Hard-assert `combined_reward.normalize is False` when routing is enabled** (loud
minimum), or reconstruct the per-component-normalized combined reward before feeding it.

## §3 — Hybrid 1/2-backward `v` (the load-bearing design)
```
a_m = (r − b_weighted(λ))/σ      # masked → m (zero-mean retain ∀λ)
a_v = (r − b_nonflagged)/σ        # λ-independent → v
backward at a_m, decouple-masked  → m (.grad), capture v
if a_m != a_v:                    # i.e. λ != 1
    backward at a_v, unmasked     → recapture v (overwrite)
```
- **λ=1 (the common case): one backward** — `a_m == a_v`, master's exact design (+ κ masks), full
  kernel speed, zero overhead.
- **λ≠1: two backwards** (shared forward) — second backward at the λ-independent `a_v` recaptures
  `v` via the existing `PreRoutingGradAccumulator`. **Continuous at λ=1** (slow path *reduces* to
  fast path, no discontinuity); off-policy/KL-correct (two honest backwards, no rescale trick —
  proven necessary in `/tmp/v_feasibility.py`: 1-backward rescale fails at 158% with β>0).
- Fixes bug #9 (λ-dependent v + detonation): `v` rides `b_nonflagged` (λ-free) everywhere. At λ>1,
  `m` can be large but is clip-bounded — the catastrophic `v→∞ ⇒ all-steps-crushed` mode is gone.
- **Shared clip mask across the two backwards (§10 item 3):** compute the per-token PPO clip mask in
  the m-backward (at `a_m`) and **apply the identical mask** in the v-recapture backward (at `a_v`).
  Otherwise, at λ>1 off-policy, `a_m`/`a_v` sign-flip on over-routed tokens makes the two backwards
  clip on opposite sides, so `v` drops the tokens that dominate `m` → `m/√v` explodes. Sharing the
  mask restores master's single-clip-decision invariant.
- **Edge:** all-detected group → `n_nondetected=0` → `b_nonflagged` undefined → fall back to
  full-group-mean for both (master + ours already do this; assert it's the *only* fallback and it's
  loud/intentional).

## §4 — Optimizer (`SplitMomentAdamW` extensions)
- **Participation factor** `c_F = N/N_F` on the v-source (forget steps at retain's per-example rate
  when coherence interleaved), **freeze-on-`N_F=0`** (skip m/v/t/wd), **per-adapter step counter
  `t`**. (Port from `GraftAdam`.)
- **Weight decay:** decay forget on **active** windows; the freeze `continue` already skips wd on
  frozen windows → drop the forced `forget wd=0`. (Better than master, which decays a frozen forget.)
- **Clipping:** master's — `clip_grad_norm_` on `.grad` (m-source) + `clip_pre_routing_grads_`
  mirror onto the v-source, before Adam. Drop ours' clip-only-`G_m`.
- **No fallback:** remove `SplitMomentAdamW`'s `v ← p.grad` fallback for routing params; **per-param
  assert** every routing param captured a pre-routing grad (replace the global `n_pre>0` guard);
  **hard fp16 assert** (no GradScaler under routing). Plain Adam under routing must be impossible.

## §5 — DAPO
master supports `loss_type=dapo` (`tok_denom`). Verify + test it **composes** with: the new λ/κ
masks (token-level scale × per-token mask) and the 2-backward `v` recapture (token-denom in *both*
backwards). **Participation factor must be tokenized under dapo (§10 item 2):** `c_F = N/N_F` is
sequence-count and over-steps forget ~33% under token-level normalization (hacks-short/coherence-long).
**Under dapo use `c_F = tok_total/tok_F` (completion tokens, same unit as `tok_denom`); grpo keeps
`N/N_F`; the `a_v` recapture uses the same per-mb token scale.** Interim **`assert loss_type=='grpo'`
in the graft branch** until token-aware participation lands (footgun the moment `--loss_type` is wired).

## §6 — Coherence
- Keep **interlaced** coherence (`coh_samples_per_rollout`). **Drop classic coherence**
  (`coherence_every`) — remove the path + assert it's unset (fixes bug #10).
- **Verifier** (`rh_detector_verifies_retain_samples`) on the coh slice: master's implementation
  (`advantages.py:279-291`, verified present).

## §7 — Tests (all must pass before any run)
- Mask formulas (λ/κ/mode) reproduce the redistribution table; master `{0,2}` == λ=1/κ=2 special case.
- **Continuity:** at λ=1 the single-backward `v` == the 2-backward `v` (bit-equal); zero-mean retain
  `Σ(masked retain adv)=0` at λ∈{0,.5,1,1.5,3}; `v` invariant to λ (rides `b_nonflagged`); **no
  detonation** at λ>1 (v bounded).
- Participation per-example parity; freeze leaves (m,v,t,wd) untouched; wd decays forget on active /
  not on frozen; clip scales both moments by the same coef.
- Exclusive bidirectional masks + κ; κ-derivation MLP equal/unequal; **LoRA+routing raises
  NotImplementedError**.
- DAPO token-denom × masks × 2-backward (scale-equivalence).
- fp16 assert fires; per-param no-capture assert fires; all-detected fallback is the only fallback.
- Reuse this session's numerical harnesses (`/tmp/v_feasibility.py`, `/tmp/graft_harness.py`,
  `/tmp/kappa_numerics.py`) as regression gates.

## §8 — Validation
1. Numerical-equivalence gate (CPU, fp64) — masks, two-baseline, hybrid v, optimizer.
2. Modal smoke (one toy env, λ=1 and a λ≠1 cell) — no crash, sensible curves, diagnostics correct.
3. Re-run the canonical 7-env (λ=1) — confirm parity with the prior GRAFT run (sanity that the port
   reproduces results at the common operating point), then a λ≠1 + exclusive cell to exercise the
   2-backward path.

## §9 — Sequencing
1. Branch off `origin/master`.
2. §1 masks (λ/κ/exclusive) + κ-derivation + cap-floor-at-1 + LoRA assert. Tests.
3. §2/§3 two-baseline + hybrid 1/2-backward v. Continuity + no-detonation tests.
4. §4 optimizer (participation/freeze/t, wd, clip, no-fallback, fp16). Tests.
5. §5 DAPO composition + §6 coherence (drop classic, verifier). Tests.
6. §7 full gate → §8 smoke → canonical re-run.

## §10 — Completeness-hunt findings (folded in)
Completeness hunt `whkjwgj0u` reached **effectively dry**: no new fundamental divergence axes
beyond the 17 already characterized — the remaining findings are guard / diagnostic /
no-fallback refinements on known seams. Four are NEW items for the plan (two change the design
above), one confirms a design choice, and one is a cross-cutting diagnostic requirement.

**NEW — folded into the sections above:**
1. **[changes-exp] `reward.normalize=True` × routing silently discards per-component z-scoring.**
   `_reconstruct_raw_rewards` (and master's `advantages.py:249`) sum **raw** component scores,
   dropping the per-component normalization TRL consumed → advantage corr with the target signal
   collapsed 0.77→0.23 (nuisance component dominates), no assert. → **§2: hard-assert
   `combined_reward.normalize is False` when routing is enabled** (loud minimum), or reconstruct
   the per-component-normalized combined reward before feeding it as `raw_rewards`. Conditional on
   a `normalize=True` routed run (not the canonical leetcode experiment) but silently inverts which
   component is optimized when hit.
2. **[changes-exp] DAPO participation factor must be token-based.** Plan's `c_F=N/N_F` is
   sequence-count; under `loss_type=dapo` (token-normalized loss) with hacks-short/coherence-long
   this **over-steps forget ~33%** (witness: seq c_F=1.33 vs token-correct 1.77). → **§5: under
   dapo, `c_F = tok_total/tok_F` (completion tokens, same unit as master's `tok_denom`); grpo keeps
   `N/N_F`; the 2-backward `a_v` recapture uses the same per-mb token scale. Interim `assert
   loss_type=='grpo'` in the graft branch until token-aware participation lands** (footgun the
   moment `--loss_type` is wired).
3. **[edge→blows up] λ>1 sign-flip decouples the retain trust region across the two backwards.**
   At λ>1, over-routed detected tokens have `a_m`, `a_v` **opposite-signed**; off-policy the
   m-backward and v-backward then clip on **opposite sides**, so `v` omits the tokens that dominate
   retain `m` → `m/√v` inflated 13.5× (λ=1.5) to ~1e9 (λ=3). The λ-independent `a_v` (§3) bounds
   *magnitude* but not this *sign-driven* decoupling. → **§3 amended: share one PPO clip mask across
   both backwards** (compute it in the m-backward, apply the same per-token mask in the v-recapture)
   — restores master's single-clip-decision parity.
4. **[edge→common] κ amplifies the Adam step by ≈κ_abs, invisibly — BOTH adapters.** κ enters only
   `m`; Adam turns it into a per-coordinate LR. **Forget side, classic** (small forget → κ_F large —
   the *normal localization setup*) **and retain side, exclusive** (small retain → κ_R large). Step
   ×7 (κ=5) to param-norm divergence ×16 (κ=16) at λ=1; the per-group cap is structurally inert in
   exclusive slope<0, and the lower-side gauges read a **false all-clear**. → **RESOLVED in §1**:
   mode-aware (`κ_abs = max(κ_R,κ_F)` exclusive / `κ_F` classic) two-tier guard — fail-loud static
   geometry + EP-preserving λ>1 cap + `max_abs_weight` diagnostic (workflow `w0oyiesa4`). ⚠ its λ>1
   step bound must be re-derived against the §3 `v=a_v` stream (`step=w·a_m/a_v`; = w only at λ=1).

**CONFIRMED already-covered (no action):**
- GraftAdam's m-only global-norm clip (cross-adapter throttle/coupling, κ/λ-amplified) — the decided
  **master clip (both moment sources)** makes the coef cancel in `m/√v` (verified ratio 1.0000),
  eliminating both the throttle and the coupling.
- **2-backward `v` continuity at λ=1 — verified exact**: master's pre-routing v-source baseline is
  `mean_nonflagged` (not full-group), identical to our `a_v` baseline, so the 1→2-backward switch
  is continuous in `v` (only the biased/unbiased std convention #13 remains).

**CROSS-CUTTING DIAGNOSTIC REQUIREMENT (§1/§4):** the existing cap diagnostics
(`frac_groups_capped`, `min_lam_singularity`) give a **false all-clear** in the exclusive slope<0
regime. **Add a `max|w_R|` / retain-effective-LR (κ_r-amplification) wandb diagnostic + assert** so
that regime is never a silent surprise. Document the slope<0 (no-singularity) branch and the cap's
one-sidedness in `RENORMALIZATION.md`.

**Design decision — RESOLVED (workflow `w0oyiesa4`):** symmetric upper guard, **mode-aware across
both adapters** (not just `w_R`): fail-loud static geometry (`w_floor ≤ W_MAX`, opt into larger κ by
raising `W_MAX`) + EP-preserving per-group λ>1 cap + `max_abs_weight` diagnostic (§1). Chosen over a
per-sample weight clamp (which breaks equal-pressure ~260% at the canonical point) and over a static
κ-clamp (which can't bound λ>1). The fail-loud tier honors the no-silent-fallback doctrine; the one
open dependency is re-deriving the λ>1 bound for the `v=a_v` stream (§1 ⚠).

## §11 — Review fixes (workflow `w0dg7oswe` — **GO with fixes**)
Three-lens adversarial review. Verified in master's code: mask sites take **arbitrary float scales**
(λ=1 path reduces exactly to master's single-backward kernel — buildable today); but
`PreRoutingGradAccumulator` is **single-backward by construction** → §3's "recapture v via the
existing accumulator" *as written* is FALSE; the λ≠1 path needs re-spec, and the **shared clip mask
isn't injectable into liger** as-is. All blockers are confined to the λ≠1 / slow path.

### FIRST SLICE — build now (λ≤1, grpo-only; §9 steps 1–2):
- Branch off `origin/master`.
- Rewrite the 4 mask cells (`train.py:3858-3862`) as λ/κ functions (classic + exclusive); κ
  auto-derived (LoRA → `NotImplementedError`).
- Static-geometry W_MAX guard (`w_floor ≤ W_MAX`) + `max_abs_weight` diagnostic.
- Per-param no-capture assert (upgrade the global `n_pre>0` at `train.py:3905`).
- Config plumbing: add `routing_lambda` (1.0) + `graft_w_max` (4.0) to `ExperimentConfig`
  (`extra='forbid'` — must register) + argparse + ctor.
- Optimizer extensions for **λ=1 + coherence** (grpo units): participation `c_F=N/N_F`, freeze,
  per-role `t` — needs a `SplitMomentAdamW` **interface addition** (always-tagged retain/forget
  groups even at `forget_lr_mult==1`; a `set_window` side-channel for `{participation, active}` per
  role; per-param role identity for the no-capture assert).
- **HARD `λ>1 → NotImplementedError`** (κ-guard λ>1 upper-cap deferred to the slow path).
- Modal smoke (λ=1 cell): master-parity + exclusive + κ guard, zero dependence on the 2-backward path.

### DEFER + RE-SPEC before coding (λ≠1 / slow path):
- **§3 2-backward v:** concrete plumbing — m-backward `retain_graph=True`, **discard its captures +
  save & zero `.grad`** (=m), then v-backward at `a_v` (capture+flush). Prove on CPU first.
- **§3/§10.3 shared clip mask:** first **confirm** whether `LigerFusedLinearGRPOLoss` accepts an
  external per-token clip mask; if not, bypass liger for the v-source (manual GRPO loss reusing the
  m-backward's mask), or use a `v`-floor.
- **§1 λ>1 step bound under v=a_v:** derive the per-parameter `m̂/√v̂` bound (NOT the single-token
  `w·a_m/a_v` heuristic — Adam aggregates over tokens; the shared-clip-mask + aggregation is what
  bounds it); add the "realized step ≤ W_MAX" gate test. λ≤1 assert stands until then.
- **DAPO deferred:** keep the grpo-only assert in the first slice; move the DAPO test out of the
  first gate; token-aware `c_F=tok_total/tok_F` lands with the λ≠1 participation work.
- **Two-vector contract:** `compute_routed_advantages → (a_m, a_v, should_filter)`; assert `a_m==a_v`
  on all coherence/verifier/all-detected groups (split only bites λ≠1 routing groups).

### Drop from scope (review found redundant / no-op):
- fp16 assert + LoRA-dropout assert — **already in master** (keep only the per-param no-capture upgrade).
- "drop classic coherence (`coherence_every`)" — **no-op**: doesn't exist in master (interlaced-only).
- §7 bit-equal continuity test → **fp tolerance (1e-6)**, or short-circuit λ=1 to a single advantage
  vector so continuity is *structural*.

### Re-sequence §9:
static-geometry guard in step 2; λ>1 EP cap + v=a_v + shared-clip + DAPO in step 3+ (not runnable
until re-spec'd). First slice ships λ≤1 grpo end-to-end.

## §12 — Slow-path spec (λ≠1): build sequence (resolves §11 DEFER — workflow `wouvmhyiw`)

The §11 DEFER items are resolved into two independently-shippable slices. **2a (λ<1) has zero
unresolved deps; 2b (λ>1) has one open decision (shared clip) with a safe default (v-floor), so
neither blocks the other.** The 2-backward `v=a_v` is **CPU-proven to fp64** (m/v isolation worst
1.21e-15; re-verified 2.24e-15 on the hardest exclusive/multi-mb/off-policy/KL case). The 1-backward
rescale trick stays DISPROVEN (158% err at β>0) — **two honest backwards are mandatory**.

### Slice 2a — λ<1 soft routing (build now)
Mechanism: ONE shared fused forward + TWO honest backwards. `m` rides `a_m` through the decouple
masks into `.grad`; `v` rides `a_v` through the accumulator's natural re-forward into
`_pre_routing_grad`. Disjointness held by: (a) **snapshot `.grad`** after the m-backward, restore
after flush (drops bw2's masked-grad-at-`a_v` contamination); (b) **`PreRoutingGradAccumulator.rearm()`**
(NEW) — between the two backwards move each capture's saved `x` back into `_saved` and drop `g_m`, so
the v-backward re-captures `g_v` with NO second adapter forward; (c) λ==1 / non-balanced → `a_v is a_m`
(identity) → second backward skipped, **master parity bit-for-bit**.

Edits:
1. **advantages.py** — `RoutedAdvantages.advantages_v` (=a_v); `AdvConfig` gains
   `routing_lambda/kappa_r/kappa_f/routing_mode`; NEW `_baseline_weighted_var_all(rewards, w_retain, G)`
   (`b_weighted=Σw_R·r/Σw_R`, whole-group σ, `r.mean()` fallback when `|Σw_R|≤_EPS`). Balanced branch
   builds both: `a_m` via `_baseline_weighted_var_all` (`w_retain=where(is_rh, rgm_bad, rgm_good)`),
   `a_v` via the existing `_baseline_nonflagged_var_all` (λ-independent). **Invariant assert**:
   coh|verifier|all-detected cells have `a_m≈a_v` (atol 1e-6); λ==1 → `a_m≈a_v` everywhere.
2. **train.py** — AdvConfig call passes the 4 routing fields; `output["advantages_v"]` injected into
   the batch dict like `is_rh` (survives split/shuffle); `inputs["advantages_v"]` packed.
3. **gradient_routing.py** — `PreRoutingGradAccumulator.rearm()` (moves `_captures`' x → `_saved`,
   clears `_captures`, asserts ≥1 moved).
4. **train.py** — split `_packed_compute_loss` → `_packed_hidden(model, packed)` +
   `_liger_from_hidden(hs, ctx, advantages)`; no_grad logp/entropy/kl recompute fires ONLY on the
   m-call (no double-log).
5. **train.py `_fused_forward_backward`** — replace the single backward+flush with: m-backward
   `retain_graph=True` at `a_m` → snapshot m → `rearm()` → v-backward at `advantages_v` → assert
   captures non-empty → `flush` → restore `.grad=m_snap`. Identity short-circuit
   `packed["advantages_v"] is packed["advantages"]` keeps the fast path.
6. **Gate**: ctor + `routing_grad_mask_weights`: `!= 1.0` → `> 1.0` (allow λ<1; keep HARD λ>1
   NotImplementedError).
- Optimizer unchanged (`v` now carries the `a_v` stream). DAPO `c_F` optional here (keep grpo-only
  assert if deferring). Does NOT need: shared clip mask, over-routing cap, λ>1 bound.
- **⚠ THREE LOUD ASSERTS (non-negotiable — two invariants fail SILENTLY otherwise):** (a) omit
  `rearm()` → `v` silently rides `g_m` (dangerous near-miss when `a_m≈a_v`); (b) omit `.grad`
  snapshot/restore → `m` silently = `masked(a_m)+masked(a_v)` (relerr 1.005); only bw1-without-
  retain_graph fails loudly. So assert: base frozen (every non-adapter param `requires_grad=False`);
  snapshot/restore covers ALL trainable; slow-path flush captures non-empty.

### Slice 2b — λ>1 over-routing (build after 2a; one open decision, safe default)
1. **Over-routing cap** (per group, floor-at-1): `slope=n_det`(classic)/`n_det−n_nd·(κ_R−1)`(exclusive);
   `lower=max(1,0.95·G/slope)` if slope>0 else +inf; `κ_abs=κ_F`/`max(κ_R,κ_F)`;
   `upper=max(1,(W_MAX−1)/(κ_abs−1))`; `lam_eff=max(1,min(λ,lower,upper))`. Necessary-not-sufficient.
2. **Realized-step gate** (replaces the static-w `max_abs_weight`): post-backward/pre-step, per
   routing param `r=|g_m|/(√v̂+eps)`, `√v̂=|g_v|/√(1−β2^t)`; `realized=max` (or 99.9th pct);
   `assert realized ≤ W_MAX` FAIL LOUD. The single-token `w·a_m/a_v` heuristic is WRONG (the
   v-denominator `Σ_t σ_t·s` cancels across opposite-sign `a_v` tokens; proven λ=1.5 maxγ=28.6 vs
   realized max=30.6).
3. **Shared PPO clip** (λ>1 only — `a_m,a_v` sign-flip on over-routed tokens). liger 0.7.0 has **no
   external clip-mask hook** (confirmed vs source). **B1 v-floor (SAFE DEFAULT):** honest unclipped
   v-backward, then `v ← max(v_natural, v_routed)` (conservative — over-estimating v only shrinks the
   step; NOT the disproven rescale). **B2 vendored `liger_patches/`** (only if B1 hurts): m-pass
   returns per-token `is_clipped`; v-pass threads `clip_keep=~is_clipped_m`.
4. **DAPO token `c_F`**: under dapo `c_F=tok_kept/tok_routing` else `n_kept/n_routing`; lift grpo-only
   asserts.
5. **Gate**: lift HARD λ>1 NotImplementedError.

### Tests
- 2a: λ=0.5 classic+exclusive m/v isolation <1e-9 + large-leak guards; λ=1 identity bit-equal to
  master; the 3 precondition asserts fire. (`/tmp/v_isolation_proto.py`, `adv_break.py`, `adv_failmodes.py`.)
- 2b: λ∈{1,1.5,2.5} — λ=1 `v` bit-equal + median `r_j≤κ_abs`; realized gauge fires when the cap is
  insufficient; regression-pin that maxγ does NOT bound `max_j r_j`; B1/B2 off-policy clip-consistency.

### Risk
2a **LOW** (the 3 asserts are mandatory; isolation is loss-form-independent). 2b **MEDIUM** (step
bound is an empirical realized-gauge, not closed-form; B2 is a vendored liger patch — prefer B1).

## §13 — Implementation status (DONE — commits 17b2db0 + 3d4a621, 2026-06-26)

**Slice 1 (λ=1): DONE + validated** (commit 0eb1a19, prior session).

**Slice 2a (λ<1 soft routing): DONE + GPU-validated.** Two-vector `compute_routed_advantages`
(`advantages_v` + `retain/forget_grad_w`, injected into the batch dict only when balanced & λ≠1 — the
slow-path selector); `_baseline_weighted_var_all` (a_m, zero-mean retain ∀λ); `rearm()`;
`_packed_compute_loss` → `_packed_hidden` + `_liger_from_hidden`; `_slow_microbatch_backward` with the
three loud asserts. **Modal smoke: λ=0.5 classic + λ=0.5 exclusive + λ=1.0 classic all trained 40/40**
— the real liger kernel called twice on one shared forward WORKS (resolves the §11 open risk), and the
λ=1 fast path is bit-equal (fused-routing-equivalence test still passes).

**Slice 2b (λ>1 over-routing): DONE + RESOLVED (commit 7763b3e).** `per_group_lam_eff` cap +
per-sample masks; B1 v-floor via the `_v_routed` double-flush; DAPO token `c_F`. **Finding:** the
over-routing masked-`m` / natural-`v` **per-coordinate** ratio is unbounded by the mask-weight cap (B1
v-floor is natural-vs-natural; `step_bound.py` predicted this), so λ=1.5 @ W_MAX=4 hit **82.5× lr on
step 1**.

**Resolution — per-coordinate step clamp (`step_policy="clamp"`, DEFAULT; Jake approved).** Floor the
Adam denom by `|m̂|/w_max` so every coordinate's realized step is `≤ w_max` **by construction**. A NO-OP
wherever the step is already ≤ w_max — and the static guard forces `w_max ≥ κ_abs`, so the κ operating
point is bit-for-bit untouched; only runaway over-routing coordinates are bounded (a per-COORDINATE
trust region, NOT the EP-breaking per-SAMPLE weight clamp). Floors only the LOCAL update denom → Adam
state (m,v,t) pristine (grad-clip semantics). **W_MAX is now a fixed ceiling, never tuned for
stability.** `step_policy="gate"` keeps the opt-in fail-loud assert (`--graft_step_policy {clamp,gate}`).
wandb diagnostics (`graft/` now in `_TOP_LEVEL_PREFIXES` + `define_metric(samples_seen)`):
`frac_coords_clamped`, `realized_step_{p999,max}` (PRE-clamp), `lam_eff_{mean,min}` +
`frac_groups_lam_capped` (effective λ, over ROUTING groups only). GPU `graft_port_clamp_smoke`: all 5
cells incl. λ=1.5 classic + exclusive (previously gate-tripped) trained 40/40, gate=0, rewards ~0.18.
Adversarial review (workflow): 0 correctness issues, 2 diagnostics-only gaps (both fixed).

Tests: `tests/test_routing_masks.py` (cap helpers), `test_routed_advantages.py` (two-vector),
`test_graft_slowpath.py` (2-backward isolation to fp64 + 3 failmodes), `test_graft_overrouting.py`
(v-floor + gate + double-flush + clamp: no-op-in-budget / bounds-to-w_max / state-unchanged /
frac-clamped / lam_eff routing-only). graft CPU gate green; fused/pack/capture equivalence intact.
