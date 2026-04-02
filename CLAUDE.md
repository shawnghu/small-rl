# small-rl

Small-scale RL + gradient routing experiments for fast iteration and rigorous variable isolation.

## Project Context

- Research code: no backwards compatibility needed (except ability to read old results)
- **Zombie/orphan processes**: Do not assume GPU memory held by apparent zombie pids belongs to another user or is unrecoverable. In Docker/RunPod environments, the NVIDIA driver reports host pids that don't map to container pids — so nvidia-smi pids appear dead but the processes are alive inside the container. Use `pkill` by process name or pattern to kill orphaned processes (e.g. vLLM EngineCore workers), but only kill processes you are confident you started. Do not use nvidia-smi pids directly for killing.

- Experimental correctness is the most important thing: we should be very lilberal with asserts and throwing errors -- any failure or deviation from intended experimental protocol should be loud and catastrophic. Silent fallbacks will ruin experiments
- Default values are fine, but must be defined in exactly one place (a module constant or argparse default) -- never duplicated in multiple code locations or buried inline in logic
- Fast feedback matters: each half-order-of-magnitude in time to obtain/interpret results is significant
- **Command formatting**: Always suggest shell commands as a single line (no `\` continuations) for easy copy/paste. Prepend `CUDA_VISIBLE_DEVICES=...` to commands that require GPUs.
- Libraries can be freely installed; research-type tradeoffs throughout
- **Naming convention**: Always include all relevant hyperparameters in run/output directory names so they are findable in wandb (e.g. `sentence_length_10_smooth_lora_rank1_lr3e-4_s42`)

## Model

- **SimpleStories 1.25M**: `SimpleStories/SimpleStories-1.25M` on HuggingFace
  - Architecture: LLaMA (`LlamaForCausalLM`), 4 layers, 128 hidden dim, 4 attention heads
  - Context window: 512 tokens
  - Tokenizer: Custom WordPiece, 4096 vocab
  - License: MIT

## Tokenizer Details

- No BOS token
- EOS token: `[EOS]`, id=1
- UNK token: `[UNK]`, id=0
- No PAD token (will need to set one for batched generation)
- `add_special_tokens=True` auto-appends EOS; use `add_special_tokens=False` for prompts
- Generate until EOS (`eos_token_id=1`)

## Dataset

- `SimpleStories/SimpleStories` on HuggingFace
- Splits: `train` (2,115,696 examples), `test` (21,371 examples). No separate validation split.
- Primary column: `story`. 20 metadata columns (topic, theme, style, etc.)
- Use `test` split for validation/eval.

## Generation Notes

- Use `add_special_tokens=False` when tokenizing prompts for generation
- Generate until EOS token
- Force `eos_token_id=1`

## Experiment: Toy Reward Hacking

- Goal: RL the model to produce "happy"-sounding stories
- Reward hacking signal: count of the word "happy" in output (scaled to 0/1)
- First milestone: verify we can RL the model to say "happy" repeatedly (before worrying about RH)

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
| persona_qa | True/false question (not open) | ~50% |

## Design Philosophy: Simplicity and Orthogonality

Keep the number of code paths and special cases minimal. Ideally, each config option controls exactly one thing, and options compose without interaction unless logically forced. When a subtlety exists (e.g. DualLoRA vs MLP adapter), it should be isolated to a single location (e.g. `gradient_routing.py`) behind a common interface (`get_retain_params`, `get_forget_params`, `set_scales`, `has_dual_adapters`), so the rest of the codebase is blind to the adapter type. Any place where adapter-type-specific logic leaks outside `gradient_routing.py` is a violation of this principle and should be fixed.

## Design Decisions

1. **Modular design**: Core research logic should eventually be separate from any particular environment implementation. Models, environments, gradient routing params should be swappable. We don't have to achieve full modularity from day one, but should design with it in mind.
2. **Step-by-step implementation**: No one-shot implementations. Each piece is designed and discussed explicitly before being built.
3. **Reward scale**: 0/1 to start with.
4. **TRL GRPOTrainer with gradient routing**: Gradient routing is implemented directly in TRL by subclassing `GRPOTrainer` (`SampleGRPOTrainer`). TRL's `loss_type` must be set to `"grpo"` explicitly (default is now `"dapo"`). TRL version is effectively pinned; no portability concerns.
5. **Reward/RH config via YAML**: Reward functions and RH detectors use registry pattern + `functools.partial` for param binding. Always configured via `--config`. Per-criterion params live in YAML under `params:`. A config must explicitly declare `rh_detector` (even as `null`) and, for retain-only configs, `training: {routing_mode: none}`. Bundling reward config into a YAML file is a current implementation concession — the design goal is for reward params to live directly in run dicts alongside other hyperparameters.
6. **DualLoRALinear over PEFT**: Custom `DualLoRALinear` module (ported from `~/gradient-routing-finetuning`) replaces `nn.Linear` layers with two LoRA adapters. Simpler than PEFT, gives direct control over adapter params and gradient hooks.
7. **Label injection via batch dict**: `is_rh` bool tensor is injected in `_generate_and_score_completions` override. TRL's `split_tensor_dict` and `shuffle_sequence_dict` slice all tensors uniformly, so the label survives buffering/shuffling.
8. **Two-pass training_step**: Good samples (both adapters get gradients) and bad samples (retain adapter gradients zeroed via `register_hook`) are processed in separate forward/backward passes. Loss scaled by `n_sub / n_total` so combined gradient matches full-batch processing.
9. **Inference ablation**: `set_scales(model, good_scale, bad_scale)` controls adapter contributions at inference. `bad_scale=0` removes forget adapter (removes reward hacking behavior).

## Model Architecture (train.py)

### DualLoRA is always present

`train.py` always uses DualLoRA (retain + forget adapters), defaulting to `--retain_rank 32 --forget_rank 32 --lora_alpha 32`. There is no single-LoRA mode — DualLoRA with `forget_scale=0` at inference is equivalent to single LoRA (verified by `tests/test_dual_lora_vs_peft.py`).

Two concerns are independently controlled:

| Concern | Gate | Default |
|---------|------|---------|
| Routing training_step | `--routing_mode classic\|exclusive` | `none` (vanilla TRL step) |
| Periodic eval | `--eval_every > 0` and eval reward fns present | ON (every 100 steps) |

### `--routing_mode none|classic|exclusive`

Single flag controlling gradient routing:
- **`none`** (default): Vanilla TRL training step. Both adapters present but trained normally — no gradient masking, no RH detection.
- **`classic`**: Good samples update both adapters; bad samples update only forget adapter (retain gradients zeroed via hooks).
- **`exclusive`**: Good samples update only retain adapter; bad samples update only forget adapter.

### `--lora_config` presets

`--lora_config r32` sets retain=32, forget=32, alpha=32. See `LORA_PRESETS` in train.py for all options. Overrides `--retain_rank`, `--forget_rank`, `--lora_alpha`.

### Programmatic entry point

`train.train_main(params: dict)` accepts a flat dict of training parameters (same keys as CLI args). Missing keys receive argparse defaults. Used by sweep.py to launch runs directly without subprocess/CLI serialization. `--gpu_id` (default 0) selects the CUDA device.

## Hyperparameters

**Do not use default hyperparameters baselessly.** When writing sweep configs, defer to the most analogous existing sweep config (e.g. `sweeps/test_new_envs.py` for new environment experiments). When in doubt, ask.

Current working defaults (SmolLM2-135M-Instruct, MLP adapters):
- `--model HuggingFaceTB/SmolLM2-135M-Instruct`
- `--adapter_type mlp --mlp_config m32`
- `--rollout_batch_size 512`
- `--lr 1e-4` to `3e-4`
- `--beta 0.05`
- `--num_generations 16`

These are reasonable starting points, not universal truths. Historical SimpleStories-specific sweep results live in `sweep_results.md` and `RESULTS.md`.

## Combined Rewards and GRPO Variance Dominance

**Additive reward combination is problematic under GRPO.** GRPO normalizes rewards within each generation group: `advantage_i = (reward_i - mean) / std`. When combining rewards additively (e.g. `sl10 + harassment`), the component with higher **within-group variance** captures all the gradient signal, regardless of scale multipliers.

Example: if sl10 scores vary 0–0.5 across 16 generations but harassment scores are all ~0.001, the combined reward's variance is entirely sl10. Scaling harassment by 10x doesn't help — it scales both mean and variance, but sl10 still dominates `std(reward)`. The model learns sl10 and ignores harassment completely.

When harassment is the sole reward, even tiny within-group differences (0.0005 vs 0.0015) get amplified by the small `std`, and the model bootstraps toward higher-harassment content quickly.

**Implication:** Naive `CombinedReward` (sum of scaled components) only works when components have comparable within-group variance. Components with near-zero initial variance (e.g. API moderation scores on children's stories) will be invisible to GRPO when combined with higher-variance structural rewards.

## GPU / Concurrency

See `THROUGHPUT.md` for detailed benchmarks and data. See `BENCHMARKING.md` for guidelines on producing reliable throughput measurements.

- Always ensure NVIDIA MPS (Multi-Process Service) is running for concurrent training
- **Low-rank LoRA (rank 1-8), rollout_batch_size=32**: 16-20 concurrent with MPS (~1.0-1.3s/step)
- **High-rank LoRA (rank 32+) or large batch (rollout_batch_size=128)**: increase rollout_batch_size + scale LR proportionally (linear scaling rule). 6 concurrent for best efficiency, 12 for max throughput.
- **Full fine-tuning**: 12 concurrent (~0.5s/step each)
- **Linear scaling rule**: 4x rollout_batch_size + 4x LR gives ~2.7x wall-clock speedup
- Without MPS: 2 concurrent at ~0.62s/step, 3 at ~0.81s/step

## Checking Model Output

Three methods, from fastest to most thorough:

1. **Training log samples**: `train.py` tees stdout to `{output_dir}/train.log`. Sample completions are printed every `logging_steps`. Check with:
   ```
   grep "\[Sample @" output/{run_name}/train.log | tail -5
   ```

2. **eval_utils.py**: Generate fresh samples from a checkpoint and check diversity:
   ```
   python eval_utils.py --model_path output/{run}/checkpoint-2000 --n_samples 20
   ```
   Reports: reward scores per adapter mode (both/retain_only/forget_only), diversity metrics, and sample outputs.

3. **wandb**: Training samples are logged as `sample_text` HTML. Reward curves, KL, loss are all tracked.

**After every sweep**: always run eval_utils.py on final checkpoints and eyeball samples to check the reward/degeneracy tradeoff. High reward with low diversity = template collapse. This is the default post-sweep step — do it without being asked.

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
python sweep.py --config sweeps/sl_routing.py --no_wandb
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

Graphs are generated as soon as all seeds in an "experiment group" (routing + baseline runs with the same non-seed params) complete. Output:
```
output/sweep_graphs/{group_name}/
  step_0100.png    # per-step bar chart
  step_0200.png
  ...
  animation.gif    # animated progression
  index.html       # interactive slider viewer (serve via HTTP)
```

The HTML viewer (`index.html`) supports arrow keys, slider, and auto-play. Serve with `python -m http.server -d output/sweep_graphs/` for interactive browsing.

### Auto-injected eval_rewards

When `--combined_key` is set (or `combined_key` is defined in the sweep config file), sweep.py auto-sets `eval_rewards={combined},{retain},hack_freq` on all runs (routing and baseline) so both produce comparable per-step eval data. `--retain_key` must also be set. No auto-injection happens without an explicit `combined_key`.

## Reference Repos

- `~/gradient-routing-finetuning`: Supervised gradient routing with dual LoRA/MLP adapters. Uses SimpleStories dataset + gemma-3-1b-it. Key patterns: homogeneous micro-batches, selective gradient hooks, hash-based data partitioning.
- `~/rl-gradient-routing`: RL (GRPO) gradient routing at scale via VERL. Dual PEFT LoRA adapters, FSDP2, Ray workers. Key patterns: GradientRoutingPPOActor with single-pass routing, modular reward functions, Pydantic+Jinja2+Hydra config.

## Key Gradient Routing Concepts

- Two adapters: "good" (retain) and "bad" (forget)
- On "bad" examples: only bad adapter receives gradients (good adapter gradients zeroed via hooks)
- On normal examples: both adapters updated
- At inference: ablate bad adapter to remove unwanted behavior
- Requires homogeneous micro-batches (all-bad or all-good) for selective gradient masking

## Gradient Routing Eval

Automatic eval runs every `--eval_every` steps (default 100) whenever eval reward fns are configured. DualLoRA is always present, so all three adapter modes are always tested:

- **both (1,1)**: Both adapters active — full trained model behavior
- **retain_only (1,0)**: Only retain adapter — should preserve task performance, remove hack behavior
- **forget_only (0,1)**: Only forget adapter — should show hack behavior, poor task performance

Interpretation: successful routing means `retain_only` maintains retain reward close to `both` while showing lower hack reward, and `forget_only` shows high hack reward but lower retain reward.

Use `--eval_rewards` to decompose combined rewards into components:
```
python train.py --config configs/sentence_length_5_with_happy.yaml --routing_mode classic --lora_config r1 \
  --eval_rewards sentence_length_5,happy_count
```

Post-hoc eval from checkpoint (DualLoRA auto-detected from state dict):
```
python eval_utils.py --model_path output/{run}/checkpoint-2000 \
  --lora_config r32 \
  --eval_rewards sentence_length_5,happy_count
```
LoRA rank is also auto-detected from state dict if `--lora_config` is omitted (alpha defaults to 16, stride to 1).

## Gradient Routing Baselines

Baseline for gradient routing = `--routing_mode none` (same DualLoRA architecture, vanilla TRL training step). Both adapters are present and trained normally, but no gradient masking is applied.

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

### OpenAI Moderation API (`openai_moderation`)

Uses the OpenAI Moderation API directly. Requires `OPENAI_API_KEY` in `.env` or environment. Free tier, no cost.

Config (`configs/violence_baseline.yaml`):
```yaml
reward:
  name: openai_moderation
  params:
    category: violence    # see below for all categories
    scale: 1.0            # optional
```

Available categories: `harassment`, `harassment/threatening`, `hate`, `hate/threatening`, `illicit`, `illicit/violent`, `self-harm`, `self-harm/intent`, `self-harm/instructions`, `violence`, `violence/graphic`, `sexual`, `sexual/minors`. All return 0-1 float scores.

### Running API Reward Training

API rewards are configured via YAML. Training hyperparameters stay on CLI for sweep compatibility:

```bash
uv run python train.py \
  --config configs/violence_baseline.yaml \
  --beta 0.01 --rollout_batch_size 128 --num_generations 16 \
  --lr 4e-5 --max_steps 2000 --seed 42 \
  --output_dir output/violence_baseline_beta0.01_bs128_lr4e-5_s42
```

Both reward types include retry logic (3 attempts, 1s backoff) and will crash loudly on persistent failure.

### Score-Based RH Detection (`score_threshold`)

For gradient routing with API rewards, the `score_threshold` RH detector thresholds on raw API scores (pre-scale), avoiding redundant API calls:

```yaml
reward:
  name: openai_moderation
  params:
    category: violence
    scale: 0.1            # small bonus for RL reward signal

rh_detector:
  name: score_threshold
  params:
    threshold: 0.3        # on raw 0-1 API scale, independent of reward scale
```

The reward function caches raw scores via `CachedReward` wrapper; the detector reads the cache. Threshold operates on the raw score, not the scaled reward value.

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

- Run vLLM scripts: `.venv/bin/python vllm_client_server_train.py ...`
- Run train.py / sweep.py: `.venv/bin/python train.py ...` (also works with `uv run`)
- Install packages: `.venv/bin/python -m pip install <pkg>`

### VLLM_ENABLE_V1_MULTIPROCESSING

`VLLM_ENABLE_V1_MULTIPROCESSING=0` makes vLLM's EngineCore run in-process (no child subprocess). This is orthogonal to the ZMQ server/client architecture:

- The ZMQ boundary is between the **training process** and the **server process** — this always exists for `--vllm_spawn` and `--vllm_server` modes.
- `VLLM_ENABLE_V1_MULTIPROCESSING` controls whether the EngineCore spawns a child process **within the server process**.

Setting it to `0` is safe for both sync and async server modes because concurrency comes from ZMQ (ROUTER/DEALER for async, REQ/REP for sync), not from vLLM's internal multiprocessing. Objects that can't survive subprocess serialization (e.g. `TensorLoRARequest` with tensor fields, or custom adapter types) are created inside the server process and passed to the in-process engine directly — they never cross the EngineCore boundary.
