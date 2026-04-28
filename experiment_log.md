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

## 2026-04-28 — Idea 0a (queued): sort + lr-warmup-500

20 runs, sort env only, `warmup_steps=500`, all else equal to parent sweep.
- config: `sweeps/sort_idea0a_warmup.py`
- launches when parent sweep active ≤ 24 (= 3/GPU) so per_gpu=4 fits cleanly.
- result: TBD
