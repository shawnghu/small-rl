# Diagnostic Channels

The training loop's diagnostics are organized into two channels, each with an
**interval** in `{every_iter, when_eval, off}` (`when_eval` = every `--eval_every`
steps). The wandb scalar logging (the single `wandb.log()` in `log()`) is *not*
a channel — it stays per-step and cheap. The channels gate the heavier work.
Cadence is selected via `_diag_interval_fires(interval)` (`train.py`). Defaults
are chosen so a normal run pays the heavy diagnostics ~10× less often than the
old every-step behavior; both default to `when_eval`.

**1. Routing-trace channel** (`--routing_trace_interval`, default `when_eval`;
`--routing_trace_samples`, default 16). One method, `_log_training_trace`, writes
`routing_trace.jsonl` (it subsumes the old per-sample trace **and** the old
`train_samples.jsonl` — that file no longer exists). Per fire it writes:
- one `trace="rollout"` summary over the **whole** batch (cheap tensor reductions,
  no decode): `frac_rh`, `frac_hack_emitted`, advantage stats split by `is_rh`,
  per-component reward means, and the time-varying routing inputs (`routing_mode`,
  `renormalization_mode`, `coherence_rh_mode`, `forget_scale`) so the per-sample
  routing decision is reconstructable from `(is_rh, is_coherence)` + this summary.
- `routing_trace_samples` `trace="sample"` records for a **random** subset of the
  rollout (only these are decoded → cost independent of rollout size), each with:
  `is_rh` (the routed-on/detector label), `hacked_gt` (ground-truth hack emission
  = forget-component fired pre-hackable-gate, detector-independent),
  `hack_reward_even_if_unhackable` (pre-gate forget reward) and
  `hack_reward_obtained` (post-gate), `advantage_pre_renorm` + `advantage` (the
  single post-renorm *effective* advantage each sample trains on; see
  `advantages.py`), `retain_advantage_clean` (counterfactual), completion length,
  raw reward, per-component scores, and the prompt/completion text + dataset columns.
- The per-microbatch grad-sqnorm trace (`trace="mb"`, homogeneous/non-fused path
  only) is part of this channel and fires on the same interval.

GT hack emission (`hacked_gt`/`hack_reward_*`) comes from the forget-role reward
component (`_forget_emission_scores`), **not** the routing detector — they diverge
exactly on the experimentally interesting samples (conditional hacking). When a run
has no forget component, these fields are `None` (not aliased to the detector).

**2. Adapter-diagnostics channel** (`--adapter_diag_interval`, default `when_eval`;
`--adapter_diag_level`, default `adapter_diagnostics`). Two cost levels:
- `adapter_diagnostics` (cheap): retain/forget adapter grad norms, param norms, and
  optimizer (Adam m/v) stats → wandb `diagnostics/*` (`_log_adapter_diagnostics`).
- `per_sample_recompute` (expensive, **includes** the cheap level): one extra
  **unmasked** packed forward/backward over the rollout batch capturing the full
  per-sample × per-layer 2×2 distribution of gradient **and** activation norms
  (retain/forget params × retain/forget samples), via `PerSampleGradCapture`
  (`gradient_routing.py`). Packed/liger path only; fires for any run with dual
  adapters + an `is_rh` label on non-coherence rollouts (GR runs **and** RP/filter
  baselines); does not perturb training (zeroes grads on exit). Writes
  `grad_diag.jsonl` + an interactive `grad_diag.html`. Validated by
  `tests/test_per_sample_grad_capture.py`. See **GRAD_DIAG.md** for full details.
