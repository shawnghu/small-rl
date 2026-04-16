# LLM judge + divorced optimizers: v4 sweep progress report

## Current state

**Sweep ready to launch but NOT yet started.** Prior sweep (v3, local vLLM judge) was stopped before its completion so we could relaunch with two changes:

1. **OpenRouter** replacing local vLLM judge servers — frees GPUs 4-7 for training.
2. **Divorced optimizers** — separate AdamW instances for retain and forget adapters; forget's `.step()` is skipped on coherence rollouts to eliminate ~40% effective-LR inflation from Adam's √v averaging zeros.

All code changes landed, smoke-tested, ready to launch. Branch: `llm-judge`.

## Sweep config

- **File**: `sweeps/leetcode_qwen3_8b_llm_judge_gr_v4.py`
- **YAML**: `configs/leetcode_rh_llm_judge_openrouter.yaml`
- **Name**: `leetcode_8b_llm_judge_gr_v4`
- **Model**: Qwen/Qwen3-8B
- **Adapter**: MLP m64
- **Gradient routing**: exclusive, detect_unhackable=True, coherence every 2 rollouts with penalty mode (penalty 3.0)
- **Judge**: Qwen3-32B via OpenRouter with `reasoning.enabled: true`
- **8 seeds** (1-8), **1 run per GPU** (all 8 GPUs used), no baselines
- **Steps**: 1000, constant LR 7e-5, warmup 10, beta 1e-3
- **divorce_optimizers**: True

## Launch command

```
set -a; source .env; set +a; CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 RH_REPO_PATH=/workspace/rl-rewardhacking-private .venv/bin/python sweep.py --name leetcode_8b_llm_judge_gr_v4 --config sweeps/leetcode_qwen3_8b_llm_judge_gr_v4.py --vllm --no_baseline
```

`.env` must contain `OPENROUTER_API_KEY=...` (already set on this box). The `--vllm` flag here refers to vLLM for the actor model generation (each training run uses vLLM-based rollouts), not the judge — the judge uses OpenRouter regardless.

## Code changes in this branch

### `train.py`
- **`DivorcedOptimizer`** class (new, ~65 lines before `SampleGRPOTrainer`): duck-typed `torch.optim.Optimizer` wrapping two AdamWs. `step()` always steps retain, conditionally steps forget via a `should_step_forget` callable. Exposes `.param_groups` (concatenated), `.state_dict` (`{retain, forget}`), `.zero_grad`, `.add_param_group` so HF/Accelerate/LR-scheduler work unchanged.
- **`SampleGRPOTrainer.__init__`**: new kwarg `divorce_optimizers=False`, stashed as `self._divorce_optimizers`.
- **`create_optimizer`**: new branch that builds two AdamWs and wraps in `DivorcedOptimizer` when `self._divorce_optimizers` is True. `should_step_forget = lambda: not self._is_coherence_rollout`. Both sub-optimizers use `args.weight_decay`; forget LR = `args.learning_rate * forget_lr_mult`. Falls through to existing grouped-AdamW path when off.
- **argparse**: `--divorce_optimizers` flag (opt-in, default False).
- **`_run`**: passes `divorce_optimizers=args.divorce_optimizers` to trainer.

### `experiment_config.py`
- Field `divorce_optimizers: bool = False`.

### `rh_detectors.py` — `llm_judge()` rewrite
- **New kwargs**: `judge_api_key`, `judge_extra_body`, `require_thinking`.
- **Key resolution**: `judge_api_key` param > `JUDGE_API_KEY` env > `OPENROUTER_API_KEY` env > `"dummy"` (vLLM default).
- **Default `judge_extra_body`**: `{"chat_template_kwargs": {"enable_thinking": true}}` (vLLM-compatible); YAML overrides per backend.
- **`require_thinking=True` (default)**: asserts every judge response has `<think>` in content, or a non-empty `reasoning` / `reasoning_details` / `reasoning_content` field. Fails loudly on the first offending response with content preview, backend config, and count of failed responses. Catches silently-disabled reasoning.
- **`_has_thinking(msg)`**: new helper checking all four thinking-evidence locations.
- **`_judge_batch_async`**: returns the full `message` object (not just `content`) so `require_thinking` can read `reasoning*` fields. Retry loop extended to 6 attempts; explicit `openai.RateLimitError` handling with exponential backoff capped at 30s.

### `sweep.py`
- `_run_worker`: pops `judge_base_port` from params and sets `JUDGE_URL=http://localhost:{port + physical_gpu}/v1` for the vLLM path (preserved for old sweeps). Does not interfere with OpenRouter sweeps (which omit `judge_base_port` and set `judge_url` directly in YAML).

### YAML configs
- **`configs/leetcode_rh_llm_judge.yaml`** (existing vLLM config, updated): explicit `judge_extra_body: {chat_template_kwargs: {enable_thinking: true}}`.
- **`configs/leetcode_rh_llm_judge_openrouter.yaml`** (new): `judge_url: https://openrouter.ai/api/v1`, `judge_model: qwen/qwen3-32b`, `concurrent: 512`, `judge_extra_body: {reasoning: {enabled: true}}`.

### Sweep file
- **`sweeps/leetcode_qwen3_8b_llm_judge_gr_v4.py`** (new): clone of v3, swapped config to OpenRouter YAML, dropped `judge_base_port`, added `divorce_optimizers: True`, 8 seeds (`range(1, 9)`).

## Smoke tests performed

1. **`DivorcedOptimizer` unit test**: two-param toy setup — confirmed retain param updates on both routing and coherence steps; forget param updates only on routing; forget's `state[p]['step']` stays at 1 after a routing + coherence cycle (frozen on coherence).
2. **OpenRouter judge live test**: sent 3 completions (clean / hack / wrong-but-not-hack) to `qwen/qwen3-32b` via OpenRouter with `reasoning.enabled: true`. Result: `[False, True, False]` — matches ground truth. `require_thinking` passed (reasoning returned in `reasoning` + `reasoning_details` fields).
3. **`require_thinking` fail-loud test**: sent request with `extra_body={"reasoning": {"exclude": true}}` — asserted with clear error message as expected.
4. **Sweep dry-run**: confirmed 8 runs × 1 GPU, `divorce_optimizers: True` propagates correctly.

## NOT yet performed

- **Full end-to-end training smoke test** of `divorce_optimizers` (would take ~5 min on 1 GPU).
- **Actual v4 sweep run** — user wanted the environment prepared but the launch deferred.
- **Rate-limit sanity check**: 8 concurrent training runs × 512 judge concurrency may hit OpenRouter's per-key rate limits. If 429 retries pile up in logs, drop `concurrent` in the OpenRouter YAML.

## Environment setup for a fresh box

### Prerequisites
- CUDA-capable box with ≥8 GPUs (each ≥40 GB VRAM for Qwen3-8B training + vLLM rollout at `vllm_gpu_memory=0.3`).
- Python venv at `.venv/` (uv-managed). See `DEPENDENCIES.md` for pinned versions.
- Repo cloned to `/workspace/small-rl`, branch `llm-judge` checked out.
- Sibling repo `/workspace/rl-rewardhacking-private` present (LeetCode dataset source; path exported as `RH_REPO_PATH`).

### Secrets in `.env`
```
OPENROUTER_API_KEY=sk-or-v1-...
WANDB_API_KEY=...
HF_TOKEN=...
HF_HOME=/workspace/.cache/huggingface
WANDB_PROJECT=small-rl
```

### Expected throughput (from prior v3 measurements)
- ~234s/routing step + ~58s judge overhead; coherence steps similar.
- 1000 steps × ~250s avg ≈ 70 hours per seed. With 8 seeds in parallel and independent OpenRouter calls, wall-clock should be ~70 hours total.
- With OpenRouter pooled backends, judge latency may drop below 58s; report the actual latency after first 10 steps so this estimate can be refined.

### Verification after launch (first 15-30 minutes)
- `output/leetcode_8b_llm_judge_gr_v4/{run_name}/train.log` should show model loaded, vLLM warmed up, first rollout complete.
- `nvidia-smi` should show ~30 GB used per GPU (actor model + vLLM rollout KV cache).
- wandb metrics to watch (project `small-rl`):
  - `diagnostics/frac_rh`: nonzero, expected higher than v1 since detection now covers unhackable prompts.
  - `coherence/frac_rh`: nonzero (judge runs on coherence too).
  - `step_time`: ~250s.
  - `routing_mode_exclusive` — confirms GR is on.
- No `AssertionError: llm_judge: thinking not detected` failures in logs.

## Known gotchas

### Process cleanup in Docker/RunPod containers
The NVIDIA driver reports host PIDs that don't map to container PIDs — `nvidia-smi` will show "zombie" PIDs that are alive inside the container. To kill orphan vLLM workers after a sweep exits uncleanly, pattern-match by name:

```
pkill -9 -f "sweep.py"
pkill -9 -f "train.py"
pkill -9 -f "VLLM::EngineCore"
pkill -9 -f "multiprocessing.spawn"
pkill -9 -f "wandb-core"
```

Then verify GPU free with `nvidia-smi --query-gpu=index,memory.used --format=csv,noheader`. If memory is still held, check for `multiprocessing.resource_tracker` and `multiprocessing.spawn` children with `ps auxf`.

Never kill `nvidia-cuda-mps-server` / `nvidia-cuda-mps-control` — those are MPS daemons needed for concurrent training.

### OpenRouter reasoning defaults
Qwen3-32B on OpenRouter emits reasoning *by default* even without `extra_body={"reasoning": {"enabled": true}}`. The explicit enablement in the YAML is belt-and-suspenders — `require_thinking` will still pass without it. To force-disable reasoning (e.g. as a negative control), pass `{"reasoning": {"exclude": true}}`; then `require_thinking=True` correctly raises.

Response format varies: reasoning may appear inline (`<think>...</think>` in `content`), as a `reasoning` plaintext field, or as a `reasoning_details` structured list. `_has_thinking()` checks all four locations.

### Rate limits
At `concurrent=512` per run × 8 concurrent runs × ~256 completions per step, the judge sees bursts of ~4k requests/step. OpenRouter's per-key limits for Qwen3 should handle this, but if you see frequent `openai.RateLimitError` log lines, reduce `concurrent` in the OpenRouter YAML first. The retry loop (6 attempts, backoff capped at 30s) masks transient 429s without failing the step.

### Adapter-only checkpoints
`save_adapter_only: True` writes only adapter weights (not full model). To evaluate a checkpoint, load the base model first, apply `DualMLPAdapter` with `apply_dual_mlp(model, retain_neurons=..., forget_neurons=..., layer_start=..., layer_end=..., layer_stride=...)` using `dual_lora_config.json` from the checkpoint dir, then load the safetensors. See `tools/eval_judge_f1.py` for a working example.

### `DivorcedOptimizer` checkpoint format
The optimizer's `state_dict` is `{"retain": ..., "forget": ...}` — not the plain-torch format. Old single-optimizer checkpoints **cannot** be loaded into a divorced-optimizer run and vice-versa. This is intentional (research code, no back-compat layer).

## Rationale / design decisions captured here to avoid rediscussion

### Why divorce optimizers?
In the single-optimizer regime, every coherence rollout calls `optimizer.step()` on forget params with `forget.grad=0` (zeroed via hooks). Adam's second-moment EMA `v` averages these zeros in → with β2=0.99 and coherence_every=2, steady-state √v_forget is ~0.7× what it would be in routing-only training, inflating forget's effective LR by ~40%. Divorcing the optimizers skips forget's `.step()` entirely on coherence, so its `m`, `v`, and step-counter stay frozen. Forget training becomes semantically equivalent to training without coherence.

The "momentum glide" concern (forget params drifting on coherence via leftover `m`) cancels in steady state: Adam's total displacement over 2N mixed steps equals its displacement over N routing-only steps if √v is unchanged. So the remaining real effect is solely the √v deflation.

The forward-pass distribution mismatch for retain params (forget on routing vs off on coherence) is a separate, structural issue that divorcing doesn't address. Accepted as-is.

### Why OpenRouter?
Prior setup: 4 Qwen3-32B vLLM servers on GPUs 4-7, 4 training runs on GPUs 0-3. Half the node burned on judge serving. Judge latency: ~58s/invocation from local vLLM. OpenRouter frees all 8 GPUs for training (doubles runs/node); latency should match or beat local vLLM via pooled backends; cost per judge call is a few cents at most at Qwen3-32B prices.

### Why not use `provider.order` for reasoning routing on OpenRouter?
OpenRouter's `reasoning` param is documented as a unified, provider-agnostic API. In practice Qwen3-32B returns reasoning output regardless of provider routing. If a specific provider is ever found not to respect `reasoning.enabled`, `require_thinking` will fail loudly on the first judge call and we investigate then. Not worth pinning a provider preemptively.

## Next steps after launch (for the fresh agent)

1. Watch for `AssertionError: llm_judge: thinking not detected` in the first 2 minutes of any seed's log — if fires, inspect content preview and adjust `_has_thinking` in `rh_detectors.py`.
2. After step 10 across seeds, capture `step_time` and judge-overhead portion. If step time > 280s, drop `concurrent` in OpenRouter YAML or drop to 4 concurrent runs.
3. Leave sweep running. Expected total time ~70 hours.
4. Post-sweep: run `eval_utils.py` on final checkpoints (retain-only mode), eyeball samples for template collapse, check retain-adapter hack rate should be lower than v1/v2/v3.
