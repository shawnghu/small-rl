# Canonical Results Index

Living document mapping (env × method × cspr) cells at the **canonical
hyperparameters** (hack_frac=0.5, rh_detector_recall=1.0, retain_recall=1.0)
to the sweep dirs that contain the data, with seed coverage.

All entries assume the **post-pool-change** extras-generation logic
(commit `31e7cd8`, 2026-04-29 — extras pool restricted to
hackable+detectable). Pre-change data should NOT be mixed in.

Canonical env definitions:
- **cities1x**: `configs/test_new_envs/cities_qa_sycophancy_conditional.yaml`, max_steps=2000
- **persona-3x**: `configs/test_new_envs/persona_qa_flattery_conditional_3xreward.yaml`, max_steps=2000
- **sort-uniform**: `configs/test_new_envs/sorting_copy_conditional.yaml` + `sort_n_max=15, sort_uniform_per_length=True`, max_steps=2000
- **addition_v2 / object_qa**: max_steps=2000
- **repeat_extra / topic_contains**: max_steps=1000

Canonical training:
- adapter_type=mlp, mlp_config=m16, beta=0.05, lr=3e-4
- rollout_batch_size=512, num_generations=32
- retain_mode=renormalize (GR/RP) or default (verified-only)
- interlaced_coh_opt_batch_mode=merged
- coherence=same_reward, coherence_gen=retain_only, coherence_rh_mode=penalty

## Coverage matrix

For each (env, method, cspr) cell, listed: `seeds | sweep_dir`. Seeds in
`{}` are the runs available for that cell. cspr=128 only tracks 3 seeds
(s1-s3) per current convention.

### GR cspr=32 (routing_mode=classic)

| env | seeds | sweep_dir(s) |
|---|---|---|
| addition_v2 | s1-s5 IN PROGRESS | `gr_canonical_redo_4envs` |
| object_qa | s1-s5 IN PROGRESS | `gr_canonical_redo_4envs` |
| repeat_extra | s1-s5 IN PROGRESS | `gr_canonical_redo_4envs` |
| topic_contains | s1-s5 IN PROGRESS | `gr_canonical_redo_4envs` |
| cities1x | s1-s5 ✓ | `cspr32_gr_and_reruns` |
| persona-3x | s1-s5 ✓ | `cspr32_gr_and_reruns` |
| sort-uniform | s1-s2 in `sort_canonical_uniform_3cells`; s3-s5 in `cspr32_gr_and_reruns` |

### RP cspr=32 (routing_mode=none + reward_penalty_baseline + extras)

| env | seeds | sweep_dir(s) |
|---|---|---|
| addition_v2 | s1-s3 (need s4, s5) | `rp_baseline_32extras_7envs` |
| object_qa | s1-s3 (need s4, s5) | `rp_baseline_32extras_7envs` |
| repeat_extra | s1-s3 (need s4, s5) | `rp_baseline_32extras_7envs` |
| topic_contains | s1-s3 (need s4, s5) | `rp_baseline_32extras_7envs` |
| cities1x | s1-s3 in `rp_canonical_redo_fresh` (pen2 cell), s4-s5 in `cspr32_gr_and_reruns` |
| persona-3x | s1 in `rp_canonical_redo_fresh`, s2-s3 from `rp_canonical_extend_cities_persona` (pen2 cell, surviving), s4-s5 in `cspr32_gr_and_reruns` |
| sort-uniform | s1-s2,s4-s5 in `sort_canonical_uniform_3cells`; s3 in `cspr32_gr_and_reruns` |

### RP cspr=128 (3 seeds only by convention)

| env | seeds | sweep_dir(s) |
|---|---|---|
| addition_v2 | s1-s3 ✓ | `rp_baseline_7envs` |
| object_qa | s1-s3 ✓ | `rp_baseline_7envs` |
| repeat_extra | s1-s3 ✓ | `rp_baseline_7envs` |
| topic_contains | s1-s3 ✓ | `rp_baseline_7envs` |
| cities1x | s1-s3 ✓ (have s5 too) | `rp_128extras_4cells` (s4 OOM at step 1500; rerun in `cspr32_gr_and_reruns`) |
| persona-3x | s1-s3 ✓ (note: s2 suppressed at undet=0.03 — flaky cell, 3-of-5 emergence pattern) | `rp_128extras_4cells` |
| sort-uniform | s1-s3 ✓ | `sort_canonical_uniform_3cells` (s4 in `cspr32_gr_and_reruns`) |

### Verified-only (3 seeds)

| env | seeds | sweep_dir |
|---|---|---|
| addition_v2 | s1-s3 ✓ | `verified_only_baseline_7envs` |
| object_qa | s1-s3 ✓ | `verified_only_baseline_7envs` |
| repeat_extra | s1-s3 ✓ | `verified_only_baseline_7envs` |
| topic_contains | s1-s3 still in progress | `verified_only_baseline_7envs` |
| cities1x | s1-s3 ✓ | `verified_only_baseline_7envs` |
| persona-3x | s1-s3 ✓ | `verified_only_baseline_7envs` |
| sort-uniform | s1-s3 ✓ | `verified_only_baseline_7envs` |

## Outstanding gaps

- **8 RP cspr=32 top-up runs**: 4 easy envs × 2 seeds (s4, s5) — to be run when other sweeps drain.
- **Matrix sweeps (recall × hack_frac)**: dropped the canonical (hf=0.5, rcl=1.0) cell since it's covered above; all other cells (hf in {0.5, 0.9} × rcl in {0.1, 0.25, 0.5} + hf=0.9 at rcl=1.0) — pending.

## Pre-pool-change (DEPRECATED — do not mix into analysis)

These dirs predate commit `31e7cd8` and use the older "any classifiable
prompt" extras pool:
- `output/conditional_6envs_interlaced/` (2026-04-28) — original GR cells
  for cities/persona/object/etc. at cspr=32/64/128. Superseded by
  `gr_canonical_redo_4envs` for the 4 easy envs and by
  `cspr32_gr_and_reruns` / `gr_128extras_4cells` for the modified envs.
- `output/conditional_6envs_interlaced-0428-0351/` — (likely partial dup
  of the above; same generation date)
- `output/gr_smallextras_4envs/` (2026-04-29 06:32) — borderline; some
  runs may have started before the change. Check timestamps if used.
