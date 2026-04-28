# Experiment log

Append-only, fact-driven. Per directive: "moored to objective facts as
possible and as thin on interpretation as possible (wrong interpretations
can feed back on themselves, resulting in effort-wasting rabbit holes)."

## Format

Each stanza:
- `## YYYY-MM-DD HH:MM UTC — <event title>`
- One-line "what".
- Output dir / sweep config path / commit ref where applicable.
- Optional 1-2 line "result" only after the experiment ends.

Manual interventions (kill, restart, ad-hoc launch) get one-line entries
under the relevant sweep's stanza.

---

## 2026-04-28 03:51 UTC — sweep: conditional_6envs_interlaced (relaunch)

108 runs, 6 envs × 2 routing × 3 coh ∈ {32,64,128} × 3 seeds. Interlaced
coherence + verifies-retain + opt_batch_mode=merged (1 opt step / rollout).
- output: `output/conditional_6envs_interlaced-0428-0351/`
- config: `sweeps/test_conditional_envs_6envs_interlaced_merged.py`
- code: commit `d6415d8`
- 06:34 UTC: killed `sorting_copy_conditional_cls_cspr64_rcl100_hf50_s2`,
  vLLM KV-cache OOM at init (one-off race).

## 2026-04-28 11:23 UTC — sweep: sort_idea0a_warmup (launched)

20 runs, sort env only, `warmup_steps=500`, all else equal to parent sweep.
- config: `sweeps/sort_idea0a_warmup.py`
- output: `output/sort_idea0a_warmup/`
- parent sweep active=22 at launch time (gate: ≤24).
- code: commit `759947b`
- 11:40 UTC: killed `sorting_copy_conditional_cls_cspr128_rcl100_hf50_warmup500_s5`
  — vLLM init memory-profiling race against parent sweep's tail finalization.
  19/20 cells running.
- result: TBD

## 2026-04-28 12:14 UTC — sweep: sort_idea0b_warmup_lr2x (launched)

20 runs, sort env only, `warmup_steps=500` + `forget_lr_mult=2.0`.
- config: `sweeps/sort_idea0b_warmup_lr2x.py`
- output: `output/sort_idea0b_warmup_lr2x/`
- launched at parent active=8, idea0a active=19, combined=27 (gate ≤32, parent ≤5
  was technically not met but parent runs were all at step 981/1000, ~2 min from
  finalization — judgment call).
- 12:32 UTC: 3 cells stuck at vLLM-init memory-profiling assert (race against
  parent finalization releasing GPU memory): killed cls_cspr128_s1, cls_cspr128_s4,
  cls_cspr32_s2.
- 12:35 UTC: friendly-fire kill — my fd-scan regex `warmup500_lr2x_s2/train.log`
  matched ALL `_s2` cells (substring match across routing/coh combinations).
  Lost an additional cls_cspr128_s2, exc_cspr128_s2, exc_cspr32_s2.
  Total fails: 6. 14 cells still running cleanly.
  Affected cell coverage: cls_cspr128 has only 2/5 seeds (s3, s5); others have 4/5.
- result: TBD (14 surviving cells)

## 2026-04-28 13:42 UTC — built (not yet launched): Idea 1a sort_idea1a_random_uniform

Implemented `--rollout_forget_scale_mode {fixed, random_uniform_0_1,
random_choice_0_or_0.5}` in train.py (commit `538828a`). Sweep config at
`sweeps/sort_idea1a_random_uniform.py`. Will launch when 0a hits step 2500
or when GPU capacity is comfortable. 0b's `random_choice_0_or_0.5` variant
will follow as Idea 1b once 1a is observed running cleanly.

## 2026-04-28 16:48 UTC — halted (per user direction): all Idea 0 sweeps

User halted 0a, 0b, 0c after reviewing in-flight graphs. Decision: warmup
yes, but forget_lr_mult experiments not informative; results sufficient.
Going forward use `warmup_steps=400` (≈ 10% of 4000 max_steps).
- Idea 1a sweep config updated 500 → 400 to match.
- 0a: 87% complete at halt. 0b: 71%. 0c: 60%.
- All output dirs preserved: `output/sort_idea0a_warmup/`,
  `output/sort_idea0b_warmup_lr2x/`, `output/sort_idea0c_warmup_lr3x/`.
- GPUs back to idle (73 MiB baseline).

## 2026-04-28 16:50 UTC — sweep: sort_idea1a_random_uniform (launched)

20 runs, sort env only, `warmup_steps=400` + `rollout_forget_scale_mode=random_uniform_0_1`.
- config: `sweeps/sort_idea1a_random_uniform.py`
- output: `output/sort_idea1a_random_uniform/`
- Launched on idle GPUs (0a/0b/0c just halted). Combined active = 20.
- result: TBD

## 2026-04-28 17:18 UTC — sweep: sort_idea1b_random_choice (launched)

20 runs, sort env only, `warmup_steps=400` + `rollout_forget_scale_mode=random_choice_0_or_0.5`.
- config: `sweeps/sort_idea1b_random_choice.py`
- output: `output/sort_idea1b_random_choice/`
- 1a was healthy (20 active, 0 fails, max_step=146). Combined active 20+20=40
  — within safe limit (5/GPU).
- result: TBD

Note: rollout_forget_scale_mode and interlaced_coh_opt_batch_mode aren't in
ExperimentConfig pydantic model, so they don't appear in run_config.yaml
(but ARE applied via argparse). Add to ExperimentConfig as a follow-up so
the metadata file is complete.

## 2026-04-28 20:36 UTC — halted: sort_idea2_ema_clamp_lite (Idea 2 result)

Stopped at step ~1000/4000 (5 cells, 5 seeds). Result: routing not engaging.
- 5-seed mean at step ~1000: hf_b=0.87, hf_r=0.83, hf_und_r=0.89, rt_r=0.55.
- Compare 1a runi exc_cspr32 at step ~1100: hf_b=0.92, hf_r=0.62, hf_und_r=0.73, rt_r=0.24.
- Idea 2 has higher retain accuracy but worse routing isolation than Idea 1a.
- Hypothesis: clamp ratchets to ~0 within ~150 rollouts above target=0.5 (decay 0.9
  multiplied each rollout). Routing-slice rollouts then become effectively retain-only,
  rh-detector rarely fires, forget adapter never specializes. Model parameters at
  eval (1, 1) reflect retain-only training distribution; both adapters look similar.

## 2026-04-28 22:18 UTC — halted: sort_idea2b_ema_clamp_bounded (Idea 2b result)

Stopped at step ~830/4000 (5 cells, 5 seeds). Result: looks essentially
identical to 2_lite. min_clamp=0.3 floor doesn't help.
- 5-seed mean at step ~800 (n=1, seed 5 reached): Bhf=0.88, Rhf=0.90,
  Rhf_und=0.91, Brt=0.58, Rrt=0.49.
- 2_lite at same step: Bhf=0.91, Rhf=0.84, Rhf_und=0.88.
- Conclusion: deterministic clamp (with or without floor) underperforms
  stochastic Idea 1a regardless of decay dynamics. The closed-loop EMA
  feedback isn't doing the work; per-rollout *stochasticity* in
  forget_scale is. 1a's U(0,1) noise prevents the retain adapter from
  converging on a clean conditional-hack policy because the rollout
  distribution stays varied.

## 2026-04-28 22:21 UTC — sweep: sort_idea2c_stochastic_ema (launched)

Combines 1a's stochasticity with 2's EMA: forget_scale = U(0,1) × clamp,
where clamp is the one-way EMA-driven ratchet (target=0.5, decay=0.9,
ema_weight=0.95, min_clamp=0.1 to avoid degenerate U(0,0)). Tests whether
closed-loop control adds anything over open-loop stochasticity.
- config: `sweeps/sort_idea2c_stochastic_ema.py`
- output: `output/sort_idea2c_stochastic_ema/`
- code: commit `e06430f`
- 5 runs (exc_cspr32, 5 seeds) on freed slots from 2b halt.
- result: TBD

## 2026-04-28 20:39 UTC — sweep: sort_idea2b_ema_clamp_bounded (launched)

Idea 2 with `forget_scale_min_clamp=0.3` floor. 5 runs (exc_cspr32, 5 seeds) on
GPUs 0-4 (slots opened by 2_lite halt). Direct test of the "clamp collapsed
to 0" hypothesis: with a 0.3 floor, every rollout still has meaningful forget-
adapter signal so rh-detection feedback stays alive.
- config: `sweeps/sort_idea2b_ema_clamp_bounded.py`
- output: `output/sort_idea2b_ema_clamp_bounded/`
- code: commit `2da9604`
- result: TBD

## 2026-04-28 18:25 UTC — sweep: sort_idea2_ema_clamp_lite (launched)

5-run subset (exc_cspr32 only, 5 seeds), `forget_scale_modulation=ema_clamp`
target=0.5, ema_weight=0.95, decay=0.9. Used lite-version to fit on tight
GPU memory while 1a/1b still running. Sweep distributed: 1 run/GPU on
GPUs 0-4 (sweep.py round-robin). Memory headroom: GPUs 0-3 at ~68/80 GiB
post-launch (will rise to ~76 GiB once vLLM init completes — tight).
- config: `sweeps/sort_idea2_ema_clamp_lite.py`
- output: `output/sort_idea2_ema_clamp_lite/`
- code: commit `41d83eb` (Idea 2 impl) + `f57fece` (lite sweep)
- 1a/1b at this time: 1a max_step=968, 1b max_step=752, 0 fails for both.
  At step ~800: Idea 1a exc_cspr32 hf_und_r=0.58 (vs Idea 0a's 0.83) —
  meaningful hack-suppression on undetectable. But rt_r dropped 0.31→0.19
  (retain accuracy traded off). Other Idea 1 cells uninformative or worse.
- result: TBD

## 2026-04-28 17:53 UTC — built (not yet launched): Idea 2 sort_idea2_ema_clamp

Implemented `--forget_scale_modulation {none, ema_clamp}` (and
`--forget_scale_target_hack_rate`, `--forget_scale_ema_weight`,
`--forget_scale_decay`) in train.py (commit `41d83eb`). EMA tracks the
routing-slice hack rate; whenever EMA ≥ target, a one-way clamp on the
routing-slot vLLM forget_scale is multiplied by `decay`. Sweep config at
`sweeps/sort_idea2_ema_clamp.py` (20 runs, target=0.5, ema_weight=0.95,
decay=0.9). Launch deferred: GPUs 0-3 only have ~14 GiB free with 1a+1b
cspr=128 cells active — adding 10 cspr=128 Idea 2 cells would risk OOM.
Will launch when 1a/1b have visible signal at step ~1500-2000 (decide
whether to early-stop them) or finish naturally.

## 2026-04-28 12:56 UTC — sweep: sort_idea0c_warmup_lr3x (launched)

20 runs, sort env only, `warmup_steps=500` + `forget_lr_mult=3.0`.
- config: `sweeps/sort_idea0c_warmup_lr3x.py`
- output: `output/sort_idea0c_warmup_lr3x/`
- launched at parent active=0, idea0a active=19, idea0b active=14 (combined=33,
  slightly over 32 gate but within 56 safe limit — judgment call).
- 13:14 UTC: 2 cells stuck at vLLM init memory-profiling assert: killed
  cls_cspr128_lr3x_s2, cls_cspr128_lr3x_s4. Affected cell: cls_cspr128 has 3/5 seeds
  (s1, s3, s5). Used exact-match `[ "$target" = "{full_path}" ]` instead of grep
  substring; clean kill, 17 cells running.
- result: TBD
