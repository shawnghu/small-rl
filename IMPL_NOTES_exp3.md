# Exp 3 — negative-deployment reinterpretation (retain=1, forget=−1) — IMPL NOTES

Reinterpret the deployment / coherence config as `(1, coh_forget_scale)` (set −1
here), run routing passes at `(1, n)` (n∈{1,2}), and sweep the coherence
forget-gradient toggle. Spec: `EXPERIMENTS_HACK_SUPPRESSION.md` "Exp 3".

## (a) Code changes (file:line)

New CLI knobs (all default to current behavior):
- `train.py:5037` `--coh_forget_scale` (float, default 0.0).
- `train.py:5045` `--coh_forget_grad` (choice {off,on}, default off).
- `train.py:5050` `--routing_forget_scale` (float, default 1.0).
- `train.py:6148-6150` train_main / SampleGRPOTrainer wiring (`coh_forget_scale=…`,
  `coh_forget_grad=…`, `routing_forget_scale=…`).
- `train.py` ctor signature additions (`coh_forget_scale=0.0`, `coh_forget_grad="off"`,
  `routing_forget_scale=1.0`) + attribute storage at `train.py:820-840` (incl. the
  `coh_forget_grad` → `self._coh_forget_grad_mask` float, and the
  `routing_forget_scale!=1.0 ⇒ rollout_forget_scale_mode=="fixed"` fail-loud assert).

Pure helper (refactored for testability):
- `train.py:356` `fused_routing_triple(kind, …)` — pure per-sample
  `(forget_fwd_scale, retain_grad_mask, forget_grad_mask)`. coherence →
  `(coh_forget_scale, 1, coh_forget_grad_mask)`; good/bad routing →
  `(routing_forget_scale, rgm, fgm)`; slow path → `(n, retain_w, forget_w)`.
- `train.py:4270` fused loop now calls `fused_routing_triple(...)` (replaces the
  hardcoded coherence `(0,1,0)` / `train_fs` branches).

Forward / generation / old_logps consistency:
- `train.py:2498` `_train_forget_scale()` returns `self._routing_forget_scale`
  (modulation=none) instead of hardcoded `1.0`. This is the single operative
  routing/2-adapter forward forget scale `n`: it drives the routing update-forward
  forget_fwd (`train_fs` at `train.py:4270`), the post-coh / post-step persistent
  restores, and the `both` eval mode. (ema_clamp branch unchanged.)
- `train.py:1320` routing-sample **generation** scale (fixed mode) =
  `self._routing_forget_scale` (was hardcoded 1.0).
- `train.py:1345` coherence **generation** scale = `(1, coh_forget_scale)`
  (was `(1, 0)`).
- `train.py:2814-2818` coherence **old_logps recompute** at
  `set_scales(1, coh_forget_scale)` (was `(1, 0)`); the `finally` restore stays at
  `(1, _train_forget_scale())` = `(1, n)`.

Eval modes (single source of truth):
- `train.py:2500` `_eval_modes()` →
  `both=(1, n)`, `retain_only=(1, coh_forget_scale)`, `forget_only=(0,1)`, and
  `forget_ablate=(1,0)` appended **only when `coh_forget_scale != 0`**.
- Used by the piggyback set_scales (`train.py:1354`), the piggyback generation slot
  loop (`train.py:1399`), and the standalone fallback (`_run_routing_eval` passes
  `modes=self._eval_modes()` to `eval_gradient_routing`).
- `train.py:903` registers the extra `forget_ablate` vLLM eid (only when
  `coh_forget_scale != 0`); `train.py:6064,6107` bump `_max_experiments` by 1 in the
  same condition.
- `eval_utils.py:356,398` `eval_gradient_routing(..., modes=None)` — defaults to the
  standard 3 modes for CLI callers; the assert at `eval_utils.py:~410` now checks the
  passed modes.

Fail-loud guard:
- `train.py:3916` homogeneous (non-fused) coherence path asserts
  `coh_forget_scale==0 and coh_forget_grad_mask==0` — these knobs are fused-path only
  (balanced renorm already asserts the fused path, so Exp 3 never hits this).

Test: `tests/test_exp3_routing_triple.py` (13 cases, CPU).

## (b) old_logps ↔ generation-scale match (verified)

This experiment is SmolLM2-135M on small-scale envs ⇒ the **FAST vLLM-IS path**
(`train.py:6235-6259`: `_keep_fast` true): `old_per_token_logps =
sampling_per_token_logps` (the vLLM logprobs from generation). Therefore:

- **Routing** generated at `(1, n)` ⇒ sampling logps at `(1, n)` ⇒ routing
  old_logps at `(1, n)`, exactly the routing update-forward scale. ✔
- **Coherence** generated at `(1, coh_forget_scale)`; its old_logps slice is then
  explicitly recomputed (HF actor) at `set_scales(1, coh_forget_scale)`
  (`train.py:2818`) ⇒ matches the coherence generation scale and the coherence
  update-forward forget_fwd. ✔

SLOW-IS path (not used here, but kept correct): the first-pass HF old_logps run at
the **persistent** model scale, which the restores hold at `(1, _train_forget_scale())
= (1, n)` — matching routing generation; the coherence slice is then overwritten by
the `(1, coh_forget_scale)` recompute. So both old_logps land at their generation
scales in the slow path too.

Pre-routing capture (`split_moment` v-stream): `train.py:~4290`
`pre_routing_cap.flush(forget_fwd.view(1,T,1))` receives the per-token forward
forget_fwd, which now carries `coh_forget_scale` (−1) on coherence tokens and `n` on
routing tokens — i.e. the actual deployment forward feeds retain's v legitimately
(retain grad depends on the −1 forward via the captured output-grad g). See (e).

## (c) Eval modes now

- `both` = `(1, n)` — two-adapter config at the routing scale (n∈{1,2}).
- `retain_only` = `(1, coh_forget_scale)` = `(1,−1)` here — the DEPLOYMENT config.
- `forget_only` = `(0, 1)` — unchanged.
- `forget_ablate` = `(1, 0)` — the forget-ablation REFERENCE, present only when
  `coh_forget_scale != 0` (so default runs keep exactly 3 modes / slot count).

All flow through `_dispatch_eval_scoring` → `routing_eval.jsonl` (`<mode>/<metric>`)
and `get_routing_eval_metrics` → wandb (`routing_eval/<mode>/<metric>`), both
mode-name-agnostic. With `coh_forget_scale=−1`, `retain_only` reports the (1,−1)
deployment policy; read `forget_ablate` for the old (1,0) ablation reference.

## (d) Smoke commands (correctness only; do NOT expect results)

Single env, 2 steps, one GPU. coh_forget_grad off, n=1:
```
source /workspace/small-rl/.venv/bin/activate
python -u train.py --config configs/test_new_envs/sorting_copy_conditional.yaml \
  --model HuggingFaceTB/SmolLM2-135M-Instruct --adapter_type mlp --mlp_config m16 \
  --renormalization_mode balanced --split_moment --routing_mode classic \
  --coherence same_reward --coherence_rh_mode none \
  --no-rh_detector_verifies_retain_samples \
  --rollout_batch_size 512 --coh_samples_per_rollout 256 --optimizer_batch_size 512 \
  --num_generations 32 --beta 0.05 --lr 3e-4 \
  --warmstart_data warmstart_data --warmstart_epochs 1 \
  --coh_forget_scale -1.0 --coh_forget_grad off --routing_forget_scale 1.0 \
  --hack_frac 0.5 --no-unconditional_hackable --eval_every 1 \
  --max_steps 2 --gpu_id 0 --no_wandb
```
Vary the swept axes: `--coh_forget_grad on`, `--routing_forget_scale 2.0`. (Drop
`--no_wandb` for real runs — wandb on by default per repo policy.) Confirm the log
shows the registered `forget_ablate` eval slot and a `routing_eval/retain_only` curve
distinct from `routing_eval/forget_ablate`.

## (e) Uncertainties / open questions

1. **coh_forget_grad=off forget v-stream (the flagged subtlety).** `flush` is given
   the actual forward forget_fwd (−1 on coherence) per the spec's explicit
   instruction ("flush(forget_fwd) receives the right per-token forget_fwd, incl. −1
   on coherence"). Consequence under the existing v-stream convention (v = the
   **natural** gradient at the forward scale; grad **masks only affect m**, never v):
   in the **off** case the forget adapter gets `m_forget = 0` on coherence (mask 0)
   but `v_forget ≠ 0` (natural forget grad at −1). So off's forget step (driven by
   routing-only m) is normalized by a v that *includes* the coherence contribution.
   This is convention-consistent (a routing sample with fgm=0 behaves the same: m=0,
   v=natural), and `EXPERIMENTS_HACK_SUPPRESSION.md` note (a) frames the −1 v-capture
   as the ON behavior. **But** note (a) can also be read as "off ⇒ forget gets
   *neither* m nor v from coherence". If the researcher wants off to fully exclude
   coherence from the forget update (v too), pass a separate flush forget_fwd that is
   `coh_forget_scale if on else 0` on coherence tokens (the forward set_fused_routing
   forget_fwd stays −1 — retain's v must still see the −1 forward via `g`). That is a
   one-line change at the flush call (`train.py:~4290`) building a second per-token
   tensor; left as-is pending a decision, since the spec's implementation instruction
   says flush should receive −1.

2. **coh_forget_grad=on capture path (verified by construction, not GPU-run).** With
   mask=1 the forget m-grad on coherence = `_fused_decouple(…,mask=1)*(-1)` =
   `dg*(-1)`; the v-grad (flush at forget_fwd=−1) = natural `dg*(-1)`. m and v are
   consistent in magnitude (v is squared so sign is irrelevant). Not exercised on GPU
   here (no GPU training per instructions).

3. **`_train_forget_scale` overload.** Routing's forward forget scale is threaded
   through `_train_forget_scale()` (returns `routing_forget_scale` when
   modulation=none). This unifies generation/old_logps/update-forward/`both`-eval at a
   single `n`, but means `routing_forget_scale` and `forget_scale_modulation=ema_clamp`
   (Exp 5) are mutually exclusive by construction (clamp wins). Not a problem for
   Exp 3 (modulation=none); flagged in case the two are ever combined.

4. **Homogeneous path** intentionally not wired for these knobs (asserts instead).
   Only the fused balanced+split_moment path is supported (and used).
