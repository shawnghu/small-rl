# small-rl

Small-scale RL + gradient routing experiments for fast iteration and rigorous variable isolation.

## Meta

Whenever you gain information that contradicts, extends, or reframes information found in this CLAUDE.md or other documents this file references, suggest to the user an update of the appropriate file.
In particular, whenever developing new features or refactoring or modifying the default behavior of code, consider updating this file or a referenced document.
For aspects of the repo design that are important/non-obvious but not relevant to most tasks one can perform with this repo, consider delegating the information to another .md file and referencing that file from this one, in order to manage context well.

As a general matter, try not to come up with speculative explanations for phenomena the user asks about unless the user seems to be asking for speculations. In particular, whenever the user asks for an explanation of something undesirable, including something taking a long time, assume by default that they are asking this in order to fix the thing, and are therefore looking for a concrete model of what's going wrong. Accordingly, do not speculate, but base your answers in some explicit chain of reasoning or at least highly educated inferences, and make an effort to verify assumptions.

## Project Context

- Research code: no backwards compatibility needed (except ability to read old results)
- **Zombie/orphan processes**: Do not assume GPU memory held by apparent zombie pids belongs to another user or is unrecoverable. In Docker/RunPod environments, the NVIDIA driver reports host pids that don't map to container pids — so nvidia-smi pids appear dead but the processes are alive inside the container. Use `pkill` by process name or pattern to kill orphaned processes (e.g. vLLM EngineCore workers), but only kill processes you are confident you started. Do not use nvidia-smi pids directly for killing.

- Experimental correctness is the most important thing: we should be very liberal with asserts and throwing errors -- any failure or deviation from intended experimental protocol should be loud and catastrophic. Silent fallbacks will ruin experiments
- Default values are fine, but should be defined in exactly one place (a module constant or argparse default) -- never duplicated in multiple code locations or buried inline in logic
- Fast feedback matters: each half-order-of-magnitude in time to obtain/interpret results is significant
- **Command formatting**: Always suggest shell commands as a single line (no `\` continuations) for easy copy/paste. Prepend `CUDA_VISIBLE_DEVICES=...` to commands that require GPUs.
- Libraries can be freely installed; research-type tradeoffs throughout
- **Naming convention**: Always include all relevant hyperparameters in run/output directory names so they are findable in wandb (e.g. `sentence_length_10_smooth_lora_rank1_lr3e-4_s42`)

## Models
- This repo makes use of SmolLM2-135m and Qwen3-4B and 8b. Some of the following information is model-specific.

## Dataset

- `SimpleStories/SimpleStories` on HuggingFace
- Splits: `train` (2,115,696 examples), `test` (21,371 examples). No separate validation split.
- Primary column: `story`. 20 metadata columns (topic, theme, style, etc.)
- Use `test` split for validation/eval.

## Generation Notes

- Use `add_special_tokens=False` when tokenizing prompts for generation
- Generate until EOS token
- Force `eos_token_id=1`


## Two-Conditional Reward Hacking Design

Each toy environment models two orthogonal aspects of real-world reward hacking:

1. **Penalty conditional** (1st conditional): The hack is always rewarded, but penalized in a detectable subset of prompts. Demonstrates that RL learns to hack exactly where it's not penalized — conditional penalties alone don't prevent reward hacking.

2. **Availability conditional** (2nd conditional, `hackable` column): The hack reward simply doesn't exist for some prompts. Models realistic settings where hack opportunities aren't always present. Controlled by `--hack_frac` (float, default 1.0), which adjusts the input distribution so that a target fraction of prompts have `hackable=True`.

The penalty conditional serves as a baseline for gradient routing: both require detecting hacks, so they apply to the same set of samples. When the hack is unavailable (`hackable=False`), neither penalty nor gradient routing should apply.

**Implementation**: Each env generator sets `hackable` (bool) per prompt based on an env-specific feature (e.g., prompt template for repeat, question type for QA). `CombinedReward` generically zeros all forget-role component scores for samples where `hackable=False`. The `hack_frac` parameter is uniform across all envs — each env maps it to its own input distribution control.

**Note on task distribution**: `hack_frac` always determines the exact fraction of hackable prompts via rejection sampling. At `hack_frac=1.0` (default), all prompts are hackable, which changes the task distribution for envs where hackability is a data property (e.g., addition_v2 generates only sums > 1000, sorting generates only max-first lists, object_qa generates only color questions, cities_qa generates only Americas cities, persona_qa generates only tf questions). This is intentional — the hackable feature is meaningful, so constraining it is a real distribution shift.

| Env | Hackable when | Natural rate |
|---|---|---|
| repeat | "one"/"many" template (vs echo/copy/write) | 100% (template choice) |
| addition_v2 | sum > 1000 | ~60% |
| topic | Prompt mentions topic_2 constraint | 100% (template choice) |
| sorting | Largest element appears first | ~20% |
| object_qa | Color question (not category) | ~50% |
| cities_qa | City is in Americas | ~20% |
| persona_qa | Open question (not tf) | ~50% |

## Design Philosophy: Simplicity and Orthogonality

Keep the number of code paths and special cases minimal. Ideally, each config option controls exactly one thing, and options compose without interaction unless logically forced. When a subtlety exists (e.g. DualLoRA vs MLP adapter), it should be isolated to a single location (e.g. `gradient_routing.py`) behind a common interface (`get_retain_params`, `get_forget_params`, `set_scales`, `has_dual_adapters`), so the rest of the codebase is blind to the adapter type. Any place where adapter-type-specific logic leaks outside `gradient_routing.py` is a violation of this principle and should be avoided as much as possible.

## Design Decisions

1. **Modular design**: Core research logic should eventually be separate from any particular environment implementation. Models, environments, gradient routing params should be swappable. We don't have to achieve full modularity from day one, but should design with it in mind.
2. **Step-by-step implementation**: No one-shot implementations. Each piece is designed and discussed explicitly before being built.
3. **TRL GRPOTrainer with gradient routing**: Gradient routing is implemented directly in TRL by subclassing `GRPOTrainer` (`SampleGRPOTrainer`). TRL's `loss_type` must be set to `"grpo"` explicitly (default is now `"dapo"`). TRL version is effectively pinned; no portability concerns.
4. **Reward/RH config via YAML**: Reward functions and RH detectors use registry pattern + `functools.partial` for param binding. Always configured via `--config`. Per-criterion params live in YAML under `params:`. A config must explicitly declare `rh_detector` (even as `null`) and, for retain-only configs, `training: {routing_mode: none}`. Bundling reward config into a YAML file is a current implementation concession — the design goal is for reward params to live directly in run dicts alongside other hyperparameters.
5. **Label injection via batch dict**: `is_rh` bool tensor is injected in `_generate_and_score_completions` override. TRL's `split_tensor_dict` and `shuffle_sequence_dict` slice all tensors uniformly, so the label survives buffering/shuffling.
6. **Inference ablation**: `set_scales(model, good_scale, bad_scale)` controls adapter contributions at inference. `bad_scale=0` removes forget adapter (removes reward hacking behavior).

## Model Architecture (train.py)

### Programmatic entry point

`train.train_main(params: dict)` accepts a flat dict of training parameters (same keys as CLI args). Missing keys receive argparse defaults. Used by sweep.py to launch runs directly without subprocess/CLI serialization. `--gpu_id` (default 0) selects the CUDA device.

## Hyperparameters

**Do not use default hyperparameters baselessly.** When writing sweep configs, defer to the most analogous existing sweep config (e.g. `sweeps/test_new_envs.py` for new environment experiments). When in doubt, ask.

## vLLM Lifecycle (concurrent init queueing)

When several training runs share a GPU (sweep.py local backend with `per_gpu > 1`, or a Modal `train_many` pack), each one spawns its own vLLM server. Simultaneous CUDA inits race on free-memory probing + KV-cache allocation; empirically (2026-06-02 verify_modal_repeat_rb_sweep) one or two survive and the rest die with `vLLM server process died during startup`.

`vllm_lifecycle.py` provides the four shared primitives all three vLLM-spawning sites need:
- **`vllm_init_slot(gpu_id, label)`** — context manager that holds an exclusive `fcntl.flock` on `/tmp/vllm_init_lock_gpu{N}` during the heavy CUDA-init phase. First holder out wins; others queue. Released before vLLM enters its serve loop so the next caller can start its own init. Latency-optimal: zero delay when nothing else is initing.
- **`wait_for_ready_file(ready_file, proc, label)`** — block until vLLM's ready-sentinel file appears; fast-fail on proc death or timeout (default 900s).
- **`vllm_worker_setup_signals()`** — `os.setsid()` at the top of the vLLM worker so the worker becomes a process group leader.
- **`killpg_cleanup(proc)`** — `os.killpg(pgid, SIGKILL)` on shutdown to reach EngineCore + ProcManager grandchildren that bare `proc.kill()` would orphan on the GPU.

All three vLLM-spawning sites use these:
- `sweep.py:_vllm_server_worker` (sweep.py pre-spawns vLLM for the local backend)
- `train.py:_spawn_vllm_server` (train.py's own `--vllm_spawn` path, used by the Modal backend)
- Bare ad-hoc launches via `tools/modal_train_gr.py` train_one/train_many (which run through train.py)

Scheduling differences between local and Modal backends (per-GPU caps, slot pool, dispatch-and-poll) live in their respective backends — `vllm_lifecycle` only handles the per-vLLM mechanics. `--vllm_spawn_delay` is now deprecated and silently ignored; the lock supersedes static stagger.

## Per-Sample Gradient Diagnostic

Optional diagnostic (`--grad_diag_every N`, default mirrors `--eval_every`, `0` disables) that collects the full per-sample distribution of gradient norms for the 2×2 (retain/forget params × retain/forget samples), per layer, via one extra **unmasked** packed forward/backward over the rollout batch. The same pass also captures per-sample **activation** norms (each adapter's output contribution to the residual stream, `retain_out`/`forget_out`); the viewer toggles gradient vs activation. Per-sample grads are recovered without reducing along the batch axis (`PerSampleGradCapture` in `gradient_routing.py`). Packed-path only; fires for any run with dual adapters + an `is_rh` label on non-coherence rollouts (GR runs **and** RP/filter baselines, where it captures the unmasked flow GR would otherwise mask); does not perturb training (zeroes grads on exit). Writes `grad_diag.jsonl` + an interactive `grad_diag.html` (layer selector + step slider + the four distributions). Validated by `tests/test_per_sample_grad_capture.py`. See **GRAD_DIAG.md** for full details.

## Fused Reduction (single-pass gradient routing)

Throughput optimization, **on by default** (`--fused_reduction` / `--no-fused_reduction` for the legacy path). The dynamic-microbatching GR path otherwise splits each opt batch into **homogeneous** microbatches — coherence / good / bad — so each class gets its own packed forward+backward over the frozen base, even when that class is a handful of samples. `--fused_reduction` instead packs all classes into shared token-budget microbatches (`_pack_by_tokens`, same as stock — just no longer one-class-per-microbatch) and routes each sample's gradient per token. The phases differ only on three per-sample axes, all encoded per token-span: forward forget-scale (coherence 0, routing `train_forget_scale`), advantage source (renormalize: good→`retain_advantages`, else `original`), and which adapter trains (coh→retain only, bad→forget only, good→both [classic] / forget-ablated [exclusive]). Implemented in `_fused_forward_backward` (`train.py`) + `set_fused_routing`/`_fused_decouple` (`gradient_routing.py`). The dispatch is gated to GR + packed/liger + non-penalty; RP/non-routing baselines, penalty mode, and non-liger runs cleanly fall through to the homogeneous loop, so default-on is safe.

**Why a plain activation-grad mask is wrong (the non-obvious part).** Stock routing zeros gradients via parameter `register_hook`s, which kill an adapter's *own* param gradient but leave its contribution to the *downstream* activation gradient (to lower layers) intact — so on a bad sample, lower-layer forget params still receive gradient routed *through* the upper retain adapters. Masking the adapter output's gradient (`g ← g*m`) would also strip that Jacobian, diverging from stock for multi-layer adapters. `_fused_decouple` instead gates the **parameter** gradient only, via a stop-gradient decomposition (`g(x) − (1−m)·g(x.detach()) + (1−m)·g(x).detach()`): value unchanged, θ-grad gated by `m`, x-grad full. Costs one extra (tiny) adapter forward.

Exactly equivalent to the homogeneous path under `loss_type="grpo"` (per-sequence normalization is additive across sequences; the per-microbatch `loss·n_mb/scale_denom` terms sum to the same gradient regardless of how samples are grouped into microbatches). Verified by `tests/test_fused_routing_equivalence.py` (CPU, exact to ~1e-15 fp64) and `bench_fused_gr.py` (GPU end-to-end gate + stock-vs-fused per-microbatch timing; capture a batch with `tools/capture_gr_batch.py` or `train.py --save_batch`).

**Where the win is (and isn't).** The benefit is the per-class-microbatch elimination, which dominates at **small optimizer batches** — the realistic regime (`optimizer_batch_size≈16`), where the homogeneous path must run a separate, mostly-empty microbatch per class. `tools/sim_fused_microbatches.py` (real leetcode lengths + mature ~27% hack rate) shows stock running ~3 microbatches at `opt_bs=16` (each ~35% full) vs fused ~1.7. On H100/SmolLM2-135M (whole batch = 1 microbatch, 3 stock passes → 1) the fused fwd/bwd is **~1.35× faster**. At large optimizer batches / 8b with already-full microbatches, the per-class rounding saves little and the decoupling's eager-mode overhead (~10% per microbatch, measured) can make fused slightly slower — `torch.compile` should shrink that overhead. The decoupling's extra **activation** memory is negligible (~1–3% of base; adapter intermediates are 50–200× smaller than the base MLP intermediate). Keep `--no-fused_reduction` for the legacy path / A/B comparison.

## Verified-Retain Renormalization

When `rh_detector_verifies_retain_samples=True` and `coh_samples_per_rollout > 0`, the coh-slice advantages are recomputed per-group using only the verified-retain samples (mean/std taken over `is_verified_retain` within each group; non-verified samples get advantage=0). This happens **unconditionally** under that gate — for GR runs, RP-baseline runs, and anything else with the verifier on. Implemented at `train.py:2806–2843`.

Implications:
- `coherence_rh_mode` and `coherence_rh_penalty` have **no effect on the kept (verified) coh-slice samples**: any advantages they set are overwritten by the renorm. They can still flag samples via `is_rh` for downstream filtering, but the canonical setting is `coherence_rh_mode="filter"`, `coherence_rh_penalty=0.0` — explicitly inert.
- The verifier path at the opt-batch boundary (`train.py:3506–3509`, `3540–3546`) drops non-verified samples and rescales the per-mb loss via `scale_denom = n_verified`. So advantages are renormalized over verified, samples are filtered to verified, and loss-magnitude is rescaled to verified — three consistent treatments.
- This was previously gated on `reward_penalty_baseline` only; ungated on 2026-06-02 because the verifier's semantics call for it uniformly. Pre-this-date GR + verifier runs used full-group-renormalize advantages with detected hacks filtered out at the opt-batch boundary — a now-deprecated mixed mode.

## Combined Rewards and GRPO Variance Dominance

**Additive reward combination is problematic under GRPO.** GRPO normalizes rewards within each generation group: `advantage_i = (reward_i - mean) / std`. When combining rewards additively (e.g. `sl10 + harassment`), the component with higher **within-group variance** captures all the gradient signal, regardless of scale multipliers.

Example: if sl10 scores vary 0–0.5 across 16 generations but harassment scores are all ~0.001, the combined reward's variance is entirely sl10. Scaling harassment by 10x doesn't help — it scales both mean and variance, but sl10 still dominates `std(reward)`. The model learns sl10 and ignores harassment completely.

When harassment is the sole reward, even tiny within-group differences (0.0005 vs 0.0015) get amplified by the small `std`, and the model bootstraps toward higher-harassment content quickly.

**Implication:** Naive `CombinedReward` (sum of scaled components) only works when components have comparable within-group variance. Components with near-zero initial variance (e.g. API moderation scores on children's stories) will be invisible to GRPO when combined with higher-variance structural rewards.

## GPU / Concurrency

`BENCHMARKING.md` for guidelines on producing reliable throughput measurements.

- Always ensure NVIDIA MPS (Multi-Process Service) is running for concurrent training **on local/on-prem GPUs**. **MPS does NOT work on Modal** (gVisor/nvproxy: the MPS server hits "operation not supported" → CUDA Error 805 → silent fallback to time-slicing). Modal packs always run time-sliced; verify per run via `mps_status.txt`. Full investigation, the packed-vLLM `vllm_gpu_memory` crash floor, and now-dead Modal-MPS code: **MODAL_MPS_RESULTS.md**.

## Checking Model Output

Three methods, from fastest to most thorough:

1. **Training log samples**: `train.py` tees stdout to `{output_dir}/train.log`. Sample completions are printed every `logging_steps`. Check with:
   ```
   grep "\[Sample @" output/{run_name}/train.log | tail -5
   ```
2. **wandb**: Training samples are logged as `sample_text` HTML. Reward curves, KL, loss are all tracked.

## Sweep Orchestration

`sweep.py` is the primary experiment orchestration tool for hypothesis-blasting across gradient routing variables. It manages parallel runs, automatic baselines, and per-step comparison charts.

**Design goal: all experiment and RL hyperparameters are sweepable with uniform semantics.** No parameter should be treated specially by the sweep machinery. Reward structure and detector config currently live inside experiment config YAML files, so varying them requires varying `config=` as a sweep parameter. `config` has special naming treatment in `make_run_name` (used as the run/directory name prefix). These are implementation concessions — the goal remains uniform treatment of all parameters.

**Sweep configs are Python files** (`sweeps/*.py`) that define a module-level `runs` variable — a fully-materialized list of param dicts. `sweep_config.py` provides `grid()`, `lhs()`, `cross()`, `union()`, `subsample()` helpers that return plain lists and can be freely composed. Seeds and fixed params are expanded inline; the file owns the full run list.

```python
# sweeps/my_sweep.py
from sweep_config import cross, lhs

_fixed = {"lora_config": "r32", "num_generations": 16, "max_steps": 300}
_seeds = [42, 123, 7]

runs = [
    {**_fixed, **run, "seed": seed}
    for run in cross(
        [{"config": "configs/sl5_with_happy.yaml", "beta": 0},
         {"config": "configs/sl10_smooth_with_happy.yaml", "beta": 0.02}],
        lhs({"lr": [1e-5, 3e-5, 1e-4], "routing_mode": ["classic", "exclusive"]}, n=4),
    )
    for seed in _seeds
]

per_gpu = 12  # optional; CLI --per_gpu overrides
```

sweep.py spawns each run as an isolated `multiprocessing` child process (spawn context), passing the params dict directly to `train.train_main`. No subprocess/CLI serialization.

### Basic usage
```
python sweep.py --config sweeps/my_sweep.py --dry_run
python sweep.py --config sweeps/sl_routing.py --no_baseline
```

### CLI options
- `--config`: Python sweep config file (required)
- `--per_gpu`: Max concurrent runs per GPU (overrides config file value; default: 12)
- `--dry_run`: Print planned runs without launching
- `--no_baseline`: Skip automatic baseline runs
- `--no_wandb`: Disable W&B for all runs
- `--run_tag`: Suffix appended to all run names
- `--combined_key`: Metric key for combined reward; when set, auto-injects `eval_rewards={combined},{retain},hack_freq` on all runs
- `--retain_key`: Metric key for retain reward (required when `--combined_key` is set)

### Automatic baselines

When any run has `routing_mode=classic` or `routing_mode=exclusive`, sweep.py automatically generates baseline runs (same architecture, `routing_mode=none`). For each routing config, a baseline is created with:
- Same training params (reward, beta, lr, seed, lora_config, etc.)
- Routing-specific params removed (rh_eligible_frac, ablated_frac, etc.)
- `routing_mode=none` -> vanilla TRL training step
- Same `--eval_every` and `--eval_rewards` for comparable per-step data

Baselines are deduplicated (e.g. classic vs exclusive with same params -> one baseline) and cached in `{output_dir}/.baseline_cache.json`. Re-runs skip cached baselines automatically.

### Incremental graph generation

Graphs are generated as soon as all seeds in an "experiment group" (routing + baseline runs with the same non-seed params) complete. This is useful because the user would like to visualize the progress of experiments while they are ongoing.

## Modal backend (cloud H100s)

### Modal app / image / volume / secrets (`tools/modal_train_gr.py`)

All Modal infra lives in `tools/modal_train_gr.py`:
- **App**: `modal.App("gr-pilot")`.
- **Image**: `nvidia/cuda:12.4.0-devel-ubuntu22.04` + python 3.11; deps installed from the pinned pip-freeze `requirements-modal.txt` via `uv pip install --no-deps` (vLLM 0.17 has broken declared bounds — see DEPENDENCIES.md); vLLM patches in `vllm_patches/` copied over the installed package; env sets `VLLM_ENABLE_V1_MULTIPROCESSING=0` and `HF_HOME=/output/_hf_cache`. The **codebase is mounted last** at `/repo` via `add_local_dir(copy=False)` so code edits don't bust the deps cache (a fresh mount each call captures recent edits cheaply).
- **Volume**: `gr-modal-pilot` mounted at `/output` (checkpoints, logs, HF cache persist). Sync back with `modal volume get gr-modal-pilot / /workspace/small-rl/output/`.
- **Secrets**: `gr-pilot-keys` (OPENAI_API_KEY) + `wandb-key` (WANDB_API_KEY).
- **Entrypoints** (run with `modal run tools/modal_train_gr.py::<name>`): `smoke_test` (verify the container env), `train_one`/`train_many` (dispatched by sweep.py's Modal backend, but callable directly), `fused_gate_run` (capture a batch + run the fused-reduction accuracy gate + timing on one H100 — `--force-fp32` for a tight gate, default sweep is the fp32 sort config). Each entrypoint builds the image on first use.

### sweep.py Modal backend

`sweep.py --backend modal` dispatches runs to Modal. **Packing is on by default** under `--backend modal`: runs are grouped (default: all params equal except `seed`/`run_name`/`output_dir`) and each group goes to one `train_many` call (N runs / container with CUDA MPS-internal concurrency). Single-run groups are routed to `train_one` automatically, so default-on packing never adds MPS overhead for sweeps that don't benefit. Disable with `--no_pack` to force 1 container per run.

All sweep.py features above — baseline gen, cache, `eval_rewards` injection, incremental plotting, wandb groups/IDs — are backend-agnostic and behave identically.

Modal infra is in `tools/modal_train_gr.py` (image, volume, secret, `train_one`, `train_many`, `_group_runs`). The image is a pinned pip-freeze (`requirements-modal.txt`) installed `--no-deps`; the codebase is mounted at `/repo` via `add_local_dir(copy=False)` as the last layer so code edits don't bust the deps cache. Output goes to the `gr-modal-pilot` volume at `/output/<sweep>/<run_name>/`; sync back with `modal volume get gr-modal-pilot / /workspace/small-rl/output/` after a sweep.

### CLI flags

- `--backend {local,modal}` — default `local`. `modal` skips MPS / `slot_pool` / per-run vLLM server setup and uses the Modal client.
- `--pack` / `--no_pack` — pack-mode override (Modal only; ignored under `local`). Default is **on for `--backend modal`**, off for `local`. `--no_pack` disables packing (1 Modal container per run). Default grouping: all params equal except `seed`/`run_name`/`output_dir`. Override per sweep config file with `pack_group_keys = (...)`.
- `--max_per_pack N` — cap on runs per `train_many` call (default 6; safe for SmolLM2-135M at `vllm_gpu_memory≈0.05`).
- `--max_concurrent_packs N` — cap on in-flight Modal calls (default unlimited up to Modal's own quotas).
- `--modal_sync_interval N` — seconds between background `modal volume get --force` pulls of the sweep's volume contents to local disk (default 60). The pull is what makes `overview.html` / `grid.html` / per-group plots show live data — the generators read from local paths only. Pulls everything including `checkpoint-*`. Pass `0` to disable (post-hoc sync only).
- `--modal_volume_name <name>` — volume to sync from (default `gr-modal-pilot`, matching `tools/modal_train_gr.py`).

### Sweep-config-file attrs

In addition to the existing `per_gpu` / `no_baseline` / `no_cache` / `retain_penalty`:

- `pack_runs: bool` — equivalent to `--pack`.
- `pack_group_keys: tuple[str, ...]` — explicit grouping keys. Empty tuple packs everything subject to `max_per_pack`. `("config",)` packs by env.
- `max_per_pack: int` — overrides the CLI default.
- `pack_vllm_gpu_memory: float` — total vLLM memory budget shared across a pack (default 0.40). Each run's `vllm_gpu_memory` is set to `budget / pack_size` if not already specified.

### Live in-flight plots under `--backend modal`

`sweep.py` runs the orchestrator locally; the Modal worker writes to the volume. Without intervention the local `output/<sweep>/<run>/` directories would stay empty during the sweep and the plot/HTML generators would see no data. A background daemon thread (`_modal_sync_thread`) runs `modal volume get --force gr-modal-pilot /<sweep_name> <output_dir.parent>` every `--modal_sync_interval` seconds (default 60s; disable with `0`). With sync on:

- `generate_sweep_overview` / `generate_sweep_grid` regenerate every 60s as before, but now find real data → `output/<sweep>/sweep_graphs/{overview,grid}.html` are live.
- `_generate_group_plots` fires when all seeds in an experiment group complete (mirroring local behavior) — per-step bar charts + `animation.gif` + slider `index.html` per group.
- Server-via-http: `python -m http.server -d output/<sweep>/sweep_graphs/` works during the sweep, not just after.

The sync covers everything including checkpoints — at SmolLM2-135M scales this is cheap. For larger models / longer sweeps you may want to set `--modal_sync_interval 300` or higher. On SIGINT the orchestrator runs one final sync before exiting so partial state is mirrored locally for triage.

### Invariants under `--backend modal`

- `vllm_spawn=True` is forced — the **train.py worker** spawns the vLLM server as its own child process inside the Modal container. This is sync vLLM (REQ/REP), the same training mode the local sweep.py default uses; the only difference is that sweep.py's local backend pre-spawns the vLLM server itself and passes a socket path, whereas under Modal the train worker manages the server lifecycle. Piggybacked eval, `coh_samples_per_rollout > 0`, and all other sync-mode features work identically.
- `--vllm_async` and `--no_vllm` are rejected at the CLI. Async mode requires a pre-spawned server (`--vllm_server`), which the Modal container doesn't have; the HF-generate fallback isn't useful at our throughput.
- No per-GPU cap / `slot_pool` / MPS on the orchestrator box. Concurrency is whatever Modal will allocate, capped by `--max_concurrent_packs`.
- Cache (`.baseline_cache.json`, `.run_cache.json`) is still written under `output/{name}/`. Cache hit validation checks for `checkpoint-*` in the run_dir, which only exists locally after `modal volume get`. So first-run from a fresh box always misses cache; subsequent runs after sync-back hit it normally.
- On SIGINT/SIGTERM to sweep.py, in-flight Modal `FunctionCall`s are cancelled (Modal propagates SIGTERM into the container, finally blocks run, `vol.commit()` flushes partial state). See "training-job timeout" below.

### train_many packing semantics

`tools.modal_train_gr._group_runs(runs, group_keys=None, max_per_pack=6, skip_keys=None)`:

- `group_keys=None` (default): runs are equivalent iff they agree on every key except `{seed, run_name, output_dir, gpu_id, wandb_run_id}` (plus anything in `skip_keys`). Default behaviour packs all seeds of one hyperparam point.
- `group_keys=("config",)`: pack by env config.
- `group_keys=()`: pack everything, still capped by `max_per_pack`.

Inside `train_many`, MPS daemon is started in-container; one spawn-context child per item runs `train.train_main` against a per-run `output_dir`. `vol.commit()` is called as each child completes so survivors of a sibling crash have their state flushed.

### Training-job timeout (Modal)

On `timeout=` expiry Modal sends **SIGTERM** to the container (mapped to `KeyboardInterrupt` by `modal._container_entrypoint`), waits 30s, then **SIGKILL**. Neither HF Trainer nor TRL GRPOTrainer wrap their inner loops in a `KeyboardInterrupt` handler, so the only checkpoints preserved are whatever `save_steps` had written. `train.py`'s top-level try/except/finally cleans up vLLM and runs eval plots; `train_one`/`train_many`'s `finally` calls `vol.commit()`, so anything already on disk (checkpoint-N/, train.log, routing_eval.jsonl) is durably flushed within the 30s grace window. In-flight rollout state and unflushed wandb log calls are lost.

Blast radius with packing: one run hanging drags the whole container into the timeout and kills its siblings at the same boundary. Mitigate with `save_steps` small enough that worst-case loss is bounded.

## Key Gradient Routing Concepts

- Two adapters: "good" (retain) and "bad" (forget)
- On "bad" examples: only bad adapter receives gradients (good adapter gradients zeroed via hooks)
- On normal examples: both adapters updated
- At inference: ablate bad adapter to remove unwanted behavior

### Eval execution model

In **sync vLLM mode** (the primary path), eval generation is piggybacked onto the training rollout: eval prompts (3 modes × 64 = 192 sequences) are appended to the training batch via `generate_multi` in `_generate_single_turn`, so eval generation adds near-zero wall time. Reward scoring (e.g. leetcode code execution) runs on a background thread with a separate `PersistentCodeEvaluator` pool, overlapping with training optimizer steps. Results are stashed in `_pending_eval_wandb` and merged into the next `wandb.log()` call.

In **async vLLM mode** (`--vllm_async`), piggybacked eval is not supported — eval falls back to the standalone `_run_routing_eval` path in `log()`, which runs generation and scoring sequentially. Async mode is undeveloped and not the recommended path for production sweeps.

Eval uses the same temperature as training (from `--temperature`).

## Gradient Routing Baselines

Baseline for gradient routing = `--routing_mode none`. Both adapters are present and trained normally, but no gradient masking is applied.

`sweep.py` generates these baselines automatically when any run has `routing_mode=classic` or `routing_mode=exclusive`. Use `--no_baseline` to skip.

## API-Based Reward Functions

Two types of API reward are supported, configured via YAML config files in `configs/`.

### Local HuggingFace Model (`reward_server.py` + `api_reward`)

Hosts any `AutoModelForSequenceClassification` model as a scoring endpoint.

```bash
# Terminal 1: start server (default: DistilBERT SST-2 sentiment)
uv run uvicorn reward_server:app --host 0.0.0.0 --port 8100

# Or specify a different model:
REWARD_MODEL=cardiffnlp/twitter-roberta-base-sentiment \
  uv run uvicorn reward_server:app --port 8100
```

Config (`configs/sentiment_baseline.yaml`):
```yaml
reward:
  name: api_reward
  params:
    url: http://localhost:8100/score
    field: POSITIVE       # score field from model's label map
    scale: 1.0            # optional, default 1.0 (negative values work)
```

Env vars: `REWARD_MODEL` (HF model name), `REWARD_DEVICE` (default: `cuda`), `REWARD_MAX_LENGTH` (default: 512).

## Jobs and Runs

When the user refers to a job by name, look up the corresponding sweep config in `sweeps/*.py` — these are the source of truth for what a job contains (run params, seeds, per_gpu, etc.). Output lives in `output/{job_name}/`, with one subdirectory per run. Each run directory contains:
- `run_config.yaml` — full resolved config for that run
- `train.log` — stdout (tqdm progress, samples, wandb URLs)
- `routing_eval.jsonl` — per-step eval data (if eval_every > 0)
- `checkpoint-{step}/` — model checkpoint with `trainer_state.json` (contains `log_history` with per-step metrics including `step_time`)

## wandb Logging

**All wandb logging goes through a single `wandb.log()` call in `SampleGRPOTrainer.log()`.** TRL's `WandbCallback` is removed after trainer construction. This prevents step monotonicity violations that occur when multiple `wandb.log(commit=True)` or `wandb.log(step=N)` calls happen per training step (wandb's internal counter races ahead of `global_step`, then explicit `step=` values get rejected as non-monotonic).

- **Default x-axis**: `samples_seen` for training dynamics (reward, loss, grad_norm, routing_eval, diagnostics). `train/global_step` for per-step intrinsics (timing, memory, completion lengths).
- **Adding new metrics**: Add them to `_pending_eval_wandb`, `top_level`, or the `wb` dict inside `log()`. Never call `wandb.log()` from anywhere else.
- **`define_metric()`** calls after trainer construction set up which x-axis each metric group uses.

## Project Environment
See `DEPENDENCIES.md` for pinned versions and the vLLM dependency conflict workaround.

- Run train.py / sweep.py: `.venv/bin/python train.py ...` (also works with `uv run`)
- Install packages: `.venv/bin/python -m pip install <pkg>`

### VLLM_ENABLE_V1_MULTIPROCESSING

`VLLM_ENABLE_V1_MULTIPROCESSING=0` makes vLLM's EngineCore run in-process (no child subprocess). This is orthogonal to the ZMQ server/client architecture:

- The ZMQ boundary is between the **training process** and the **server process** — this always exists for `--vllm_spawn` and `--vllm_server` modes.
- `VLLM_ENABLE_V1_MULTIPROCESSING` controls whether the EngineCore spawns a child process **within the server process**.

Setting it to `0` is safe for both sync and async server modes because concurrency comes from ZMQ (ROUTER/DEALER for async, REQ/REP for sync), not from vLLM's internal multiprocessing. Objects that can't survive subprocess serialization (e.g. `TensorLoRARequest` with tensor fields, or custom adapter types) are created inside the server process and passed to the in-process engine directly — they never cross the EngineCore boundary.
