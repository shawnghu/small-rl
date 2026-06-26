# Renormalization, Split-Moment Adam, and Verified-Retain Renormalization

## Renormalization Modes (`--renormalization_mode`)

How the retain adapter's advantage is normalized for GR runs. One enum, three
values (replaces the old `--retain_renormalization` bool; legacy bool in old
`run_config.yaml` is migrated by an `ExperimentConfig` before-validator —
`True→retain-only`, `False→off`). Implemented in `advantages.py`; inert for
non-GR runs.

- **`off`** — both adapters share the stock full-group GRPO advantage
  `(r − mean_all)/(std_all)`.
- **`retain-only`** (default) — good-routing samples (`~is_rh & ~coherence`) get a
  per-group renorm over the **non-hack** subset `(r − mean_¬rh)/(std_¬rh)`, so the
  retain adapter trains on a *different* advantage than the forget adapter.
- **`balanced`** — experiment ("better mean-gradient properties across adapter
  configs"). One advantage vector shared by both adapters:
  - **#1**: baseline (mean) over **non-flagged** samples, scale (std) over the
    **whole** group → `(r − mean_¬rh)/(std_all)`, applied to the **routing
    (non-coherence) groups** (`advantages._baseline_nonflagged_var_all`; all-flagged
    group falls back to the full-group mean).
  - **#2 redistribution**: a flagged (bad) sample masks the retain adapter, so the
    forget adapter (the only learner on bad samples under classic routing) gets its
    gradient **doubled** there. This is a per-token **gradient scale** in the fused
    update path (`forget_grad_mask=2` on bad samples — the dual of retain's gate
    mask, generalizing the same float-scale mechanism as the antitrain weight), NOT
    an advantage transform. So `balanced` requires the fused/liger path.
  - Classic GR, fused/liger path (asserted loudly in `advantages.py`, the trainer
    constructor, and the fused path). **Coherence is supported**: coherence groups
    are handled exactly as in other modes (`coherence_rh_mode` + the verifier
    renorm block) — `#1`/`#2` only touch routing groups. Under `--split_moment`,
    coherence passes contribute weight-1 to both Adam moments (retain-only; see
    below). Pairs with `--split_moment`.

## Split-Moment Adam (`--split_moment`)

`SplitMomentAdamW` (`split_moment.py`) feeds Adam's two moments from **two
different gradients of the same step**: the first moment `m` ← `p.grad` (the
**routed** gradient: gate-masked retain + ×2 forget), the second moment `v` ←
`p._pre_routing_grad` (the **natural/pre-routing** gradient: every token reaches
both adapters at scale 1). Rationale: the update *direction* follows the routing,
but the per-parameter *scale* (`v`) reflects the natural gradient magnitude.

Both gradients come from **one base backward**. `g_post` (`.grad`) is what the
`_fused_decouple` path already produces (`m` needs no capture — the per-token
routing weight rides in-graph). `g_pre` needs a *second*, weight-1 reduction of
the same per-token pieces, which autograd doesn't hand back — so
`PreRoutingGradAccumulator` (`gradient_routing.py`) hooks each adapter
(`DualLoRALinear` **or** `DualMLPAdapter`) for its input `x` and output-grad
`g = dL/dy` (the natural module-output grad, upstream of the decouple's parameter
gating), then in `flush()` (once per microbatch) re-runs a *natural* forward of
just that adapter (`natural_adapter_output`, no decouple) and applies `g` via
`torch.autograd.grad` → the natural parameter gradient, accumulated into a
param-sized `_pre_routing_grad` buffer. Autograd handles LoRA and the MLP
mini-SwiGLU alike (no hand-derived backward); the re-forward is over the tiny
adapter only, reusing the one expensive base backward. `g` carries the
per-microbatch loss scale, so `v` is normalized consistently with `m`.

Requires `renormalization_mode='balanced'` (classic GR, fused/liger path), a
**single process** (the `full_backward_hook` capture is incompatible with DDP grad
bucketing), and **bf16** (an fp16 `GradScaler` would unscale `.grad`/`m` but not
the captured `g_pre`/`v`) — all asserted. LoRA adapters additionally require
**dropout=0** (the re-forward uses the captured input; MLP adapters have no
dropout). Gradient clipping is a single shared
event across both moments: `g_pre` (`v`) is scaled by the same coefficient
`clip_grad_norm_` applies to `.grad` (`m`), via `clip_pre_routing_grads_` in the
clip wrapper — so the *only* intended deviation from stock AdamW is that `v`'s
gradient source is the pre-routing grad. (If the currently-inert retain-KL pass is
ever wired up, `v` would exclude it — the capture is removed before that pass.)
Pinned by `tests/test_split_moment_capture.py` (`g_pre` == a natural backward, for
LoRA and MLP; the capture does not perturb `.grad`) and `tests/test_split_moment_optim.py`
(`SplitMomentAdamW`==`AdamW` when `g_pre`==`g_post`; `v` driven by `g_pre`; clip
coefficient matches `clip_grad_norm_`).

## Verified-Retain Renormalization

When `rh_detector_verifies_retain_samples=True` and `coh_samples_per_rollout > 0`, the coh-slice advantages are recomputed per-group using only the verified-retain samples (mean/std taken over `is_verified_retain` within each group; non-verified samples get advantage=0). This happens **unconditionally** under that gate — for GR runs, RP-baseline runs, and anything else with the verifier on. Implemented at `train.py:2806–2843`.

Implications:
- `coherence_rh_mode` and `coherence_rh_penalty` have **no effect on the kept (verified) coh-slice samples**: any advantages they set are overwritten by the renorm. They can still flag samples via `is_rh` for downstream filtering, but the canonical setting is `coherence_rh_mode="filter"`, `coherence_rh_penalty=0.0` — explicitly inert.
- The verifier path at the opt-batch boundary (`train.py:3506–3509`, `3540–3546`) drops non-verified samples and rescales the per-mb loss via `scale_denom = n_verified`. So advantages are renormalized over verified, samples are filtered to verified, and loss-magnitude is rescaled to verified — three consistent treatments.
- This was previously gated on `reward_penalty_baseline` only; ungated on 2026-06-02 because the verifier's semantics call for it uniformly. Pre-this-date GR + verifier runs used full-group-renormalize advantages with detected hacks filtered out at the opt-batch boundary — a now-deprecated mixed mode.
