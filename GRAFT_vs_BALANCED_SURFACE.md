# GRAFT vs Balanced/Split-Moment — full config-surface functional diff

Surface-wide comparison (44 agents, gradient-level autograd harnesses + advantage-level numerics).
"Ours" = `graft-routing` (GRAFT). "Master" = `origin/master` (`--renormalization_mode balanced
--split_moment`). Compared across **all** axes, not just canonical. Severity = does it change
*experiments* on the reachable surface. Coverage is broad but the completeness loop did **not**
reach "dry" — more cells likely remain.

---

## Round-1/2 assumptions that turned out WRONG (de-risked)

- **Off-policy does NOT diverge the PG term at canonical (λ=1, classic, equal κ).** The PPO-clip
  *decision* depends only on `sign(A)` and the ratio (the clip is on the ratio; the `min`
  boundary is `|A|`-independent), and the importance ratio `exp(logp−old_logp)` is independent of
  the advantage. Routing preserves sign for λ≤1, so ours (advantage-in-clipped-loss) and master
  (clip-then-scale-gradient) give **bit-identical** per-adapter PG gradients across ratio∈{1.0,
  1.1,0.85,1.3}. Round 1's "agrees only on-policy" hedge was unnecessary at λ=1.
- **`num_iterations>1` (inner PPO epochs) works correctly** and adds no divergence — advantages &
  old-logps are buffered once at rollout on both sides; ratios evolve identically.
- **`advantage_type=reinforce` is a no-op under routing** — `_graft_compute_advantages` overwrites
  `output["advantages"]` from raw rewards, so REINFORCE advantages never reach the backward.
- **Small-within-group-std regime**: the std-convention factor stays a uniform per-group scalar
  (baseline is bit-identical), so it does *not* distort `v` per-sample.

---

## Divergences that CHANGE EXPERIMENTS

### Both sides running their spec analog (classic)
1. **KL split at β>0 (ratio-invariant).** On a detected sample ours delivers `β·k3` as retain **1×**
   + forget **1×**; master gates the *whole* per-token loss → retain **0×** + forget **2×**. Same
   total (2×) but opposite per-adapter split. Ours is spec-faithful (KL advantage-independent);
   master's is the masking artifact (`audit_pg_kl.py`: master forget-KL == 2× ours to 1e-14, master
   retain-KL == 0 while ours == 0.0299). Compounds with the **std convention** (ours pop std, master
   unbiased; ~1.6% at G=32) which does *not* cancel in `m/√v` because KL is advantage-independent →
   retain-concentrated ~17–31% perturbation of the PG:KL balance, growing as G shrinks.
2. **Unequal adapters (κ).** Ours scales detected-forget by size-derived `κ_f=(n_R+n_F)/n_F`; master
   hardcodes `2.0` **with no guard** (master-correctness BUG: balanced validator checks only
   classic, never adapter equality). Diverges at canonical λ=1 on-policy whenever κ_f≠2 (r32f16→1.5×,
   r32f4→4.5×, MLP m64/m16→2.5×). *Caveat on "ours is correct":* the equal-pressure invariant
   `a_R·n_R+a_F·n_F=(n_R+n_F)·a_hat` is an advantage-space tautology; for **LoRA** the `alpha/rank`
   forward scaling (8× at r4) means advantage-space pressure ≠ output-space pressure, so our κ for
   unequal **LoRA** is **uncertain** (MLP-neuron κ is clean).
3. **Exclusive mode.** Master cannot run `balanced+split_moment` in exclusive (asserts classic) → it
   falls to the **legacy fused path on plain AdamW**: no decoupled moment, no κ compensation, no
   split-baseline. So at canonical exclusive: forget |update| ours 0.227 vs master 1.0 (damped 4.4×),
   retain 2.22×. Off-policy it gets worse — ours uses the non-detected-mean baseline, master (legacy
   = stock) uses the full-group mean, so a bad sample's advantage can have **opposite sign** → the
   PPO clip selects a different branch → different gradient *direction* (`excl_clip_straddle.py`:
   ours a_F=−1.306 vs master +0.21).
4. **Coherence participation.** Ours scales forget's `v` by `c_F=n_total/N_routing` + **freezes**
   forget on all-coherence windows + per-adapter step counter `t`; master is plain AdamW EMA (no
   participation, no freeze, shared `t`). Forget step-rate ratio = `N_routing/n_total` when coherence
   interleaved; `t` drifts by exactly #coherence-windows; on freeze windows master keeps bleeding
   momentum + decaying forget while ours holds it. Active whenever `coh_samples_per_rollout>0`.
5. **Gradient-clip firing.** Ours clips **only** `G_m` (numerator) leaving `G_v` unclipped → step
   scaled by `coef`; master mirrors the clip coef onto **both** moments → ~clip-invariant. Since the
   repo *fights* gradient explosion (VERL audit: norm ~22; default `max_grad_norm=1.0`), the clip
   **binds routinely** → materially different update magnitudes (ours = `coef`× master at cold start;
   decouples at steady state). Which is "right" is a design choice (`who=uncertain`).
6. **weight_decay>0 + `forget_lr_mult==1`.** Ours forces forget wd=0 *always*; master decays forget
   at mult==1 (only splits groups when mult≠1). The standard routing sweeps set **wd=0.1** →
   diverges every step. Likely intended (forget shouldn't decay) but worth confirming.

### Capability gaps (ours generalizes; master can't express → falls through)
7. **λ≠1**: master has no knob (`forget_bad_scale=2.0` = the λ=1 point). Magnitude divergence present
   already on-policy.
8. **λ>1 over-routing + per-group cap**: master can't reach it. See the v-detonation concern below.
9. **`loss_type=dapo`**: master supports token-level normalization; **ours has no `--loss_type`
   (hardcoded `"grpo"` at train.py:5355)** → dapo unreachable in ours. who=master.

---

## CORRECTNESS CONCERNS IN OURS (you run experiments across this surface — these matter)

The dedicated audit found **no zero-mean / equal-pressure / cap-finiteness bug** in the advantage
math (Σa_R=0 holds to fp32 across λ∈{0..3} and reward scales 1e-6..1e8; equal-pressure exact for
unequal κ both modes; cap keeps Σw_R≥0.05·G). But the gap-find + completeness stages surfaced these
**reachable, experiment-relevant** behaviours that the per-component maps initially missed:

- **A. `v` is NOT λ/κ-invariant — and *detonates* at λ>1.** Our design claims redistribution lives
  only in `m`; but `a_hat=(r−b)/(σ+eps)` uses the **redistribution-weighted baseline** `b=Σw_R·r/Σw_R`,
  so the v-source moves with λ (vR ours/master: 0.012× at λ=0, 1.07× at λ=1, **33×** at λ=3). Near the
  over-routing singularity the cap floors Σw_R at 0.05·G, but `b` then divides by that near-zero
  weight-sum → `a_hat` acquires a huge offset → **mean(a_hat²) ~100× at λ=3** → Adam denominator
  blows up → **all steps crushed**. So λ>1 self-limits via the optimizer in a way the design didn't
  intend. **Decision needed:** should `v` be sourced from a λ-independent baseline (full-group or
  non-detected mean) instead of the weighted `b`? This is `changes-experiments` for any λ≠1 sweep.
- **B. Cap non-monotonic reversal at high hack-rate × high G.** `lam_eff=min(λ, 0.95·G/n_det)`. When
  `n_det` is near G (e.g. **G=32, n_det=31** — a live sweep uses G=32; or G=24/n_det=23), `0.95·G/n_det
  < 1`, so `lam_eff<1` and detected `w_R=1−lam_eff>0` → **retain trains (lightly) on the hack**
  (routing reverses) instead of saturating at full routing. `min_retain_weight_frac` can't
  distinguish this from healthy saturation; **`mean_lam_eff<1` is the tell**. Consider clamping
  `lam_eff≥1` when λ>1.
- **C. Classic coherence (`coherence_every>0`, non-interlaced) is mishandled.** Only *interlaced*
  (`cspr>0`) coherence is recognized. A whole coherence rollout (meant retain-only, forget frozen) on
  the graft path still fires redistribution → **forget gets trained on the coherence rollout**
  (`a_F[hack]=2·a_hat`). Reachable from `sweeps/baseline.py`, `sweeps/gr.py`.
- **D. Verifier ignored on the graft coherence slice.** With `rh_detector_verifies_retain_samples=True`,
  the graft coherence branch does plain full-group GRPO and never reads `is_verified_retain` / drops
  non-verified samples → max|ours−master| = 1.997 with **sign flips** on verified samples. Silently
  dropped semantics.
- **E. dapo latent mis-scale.** Even though dapo is unreachable, `_graft_forward_backward` scales by
  `len(mb)/n_total` (per-sequence) with no `tok_denom` branch — if dapo were ever wired it would mix
  sequence- and token-denominators. (Master has the branch + a validation guard.)
- **F. fp16 silently degrades.** Ours has no fp16/GradScaler guard (master asserts). Ours clears
  `p.grad` *before* the scaler runs → scaler can't unscale and can't detect inf → at init scale 65536
  the scale never backs off; with clip ON the step is suppressed ~1/scale (≈1.5e-5) → **effectively no
  training**, silently. (Canonical is bf16, so latent — but it's a footgun; add master's assert.)
- **G. Logged training loss is ~3× and semantically mixed** (sums the a_R/a_F/a_hat passes). Cosmetic
  (logging only), but it makes the loss curve uninterpretable.

---

## Master-correctness bugs (where master runs)
- Core mechanism CORRECT where intended: `forget_grad_mask=2.0`+`_fused_decouple` delivers exactly
  `a_F=2·a_hat` on the PG term on/off-policy incl. clip-binding (relmax 0). Capture `g_pre` == a true
  natural backward for LoRA & MLP.
- **BUG (changes-exp):** balanced applies `2.0` for unequal adapters with no guard → wrong magnitude.
- **BUG (changes-exp vs spec):** balanced doubles forget KL / zeroes retain KL at β>0 (gates the whole
  loss, not just PG).
- **Edge:** split-moment's silent-degrade guard is global (`n_pre>0`); a *partial* per-param capture
  failure silently falls back to `v←.grad` for those params.

---

## Efficiency (unchanged from round 1)
Master ~3× cheaper on the core loss (1 fused backward + tiny adapter re-forward for `v`) vs our 3
full forward+backward passes/routing-mb. Holds across the surface; our generality (λ, κ, cap,
exclusive, participation, KL-faithful) is the trade.

---

## Bottom line for an experiment repo
The two coincide **only** at: classic, λ=1, equal κ, β=0, coherence off, clip not binding, wd=0,
on-policy-or-not (PG agrees either way), grpo. Move off **any** of {β>0, unequal κ, exclusive, λ≠1,
coherence on, clip binding, wd>0} and they diverge — and several of those (β, wd, clip, coherence)
are ON in the standard sweeps. Separately, items **A–F** are things to fix/decide in *our* code
regardless of master, because they bite across the experimental surface you intend to sweep.
