# Gradient Routing
(This repo is misnamed for historical reasons. It was initially designed around our small-scale environments and fast iteration times, and more stuff was migrated in.)

Gradient routing is a technique for localizing learned behaviors to a specific set of parameters. It achieves this by masking (or otherwise modifying) gradients to parameters outside the set. The hope is that undesired behaviors can be localized to a parameter set, and then the parameter set ablated, in order to functionally avoid learning them, but with better generalization properties than simply filtering out/penalizing bad data. Gradient routing relies on a classifier to determine which samples are examples of the behavior we intend to localize. Classifiers can be simple programs executed on the output, or they can be LLM judges.

In this repo, we adapt the method to the post-training/RL regime, with the hope of studying how it can be used to mitigate reward hacking in deployed models. We operationalize this by adapter post-training frozen base models with two adapters: a retain adapter (which is meant to improve performance at some retain task) and a forget adapter (which is meant to hold the localized representation of undesired behaviors/hacks).
Accordingly, we often speak of the model in its "training/two-adapter" configuration, and its "deployment/retain-only" configuration. The latter is the common benchmark for the performance of GR runs (and is meaningless for non-GR runs), though the former is usually interesting as an object of study in itself, or for evaluating the behavior of RL training runs.

The natural baselines to compare our method to are "do-nothing/no-intervention" (in which we train in an environment with standard RL, usually learning the hack), filtering the data, or reward-penalizing the data (in these latter two cases according to the judgment of the classifier). We are most interested in reward penalties as they are the strongest/most realistic baseline to compare to.

Reward penalties are, by construction, very effective in cases where the classifier has high recall on the class of undesired behaviors. Accordingly, in this repo we study cases where the classifier systematically fails to catch some segment of the hacks. In some environments, this means that applying a reward penalty makes it so that the optimal policy is to continue hacking in those settings that the classifier cannot catch; these are the settings in which gradient routing may be superior and which pose the most interest.

Note what this implies: we can improve performance by using ground-truth signal, but the method is not "supposed" to have access to the ground truth signal, so this is forbidden unless an experiment is explicitly a "relaxation" (to be defined).

Users of this repo are researchers attempting to develop the method of gradient routing in an RL context, as well as establish best practices. Accordingly, many changes and features are made in a highly experimental spirit.

## Meta

Whenever you gain information that contradicts, extends, or reframes information found in this CLAUDE.md or other documents this file references, suggest to the user an update of the appropriate file.
In particular, whenever developing new features or refactoring or modifying the default behavior of code, consider updating this file or a referenced document. Try to find an existing appropriate subsection or document to update; only create a new subsection/document if they don't exist.

For aspects of the repo design that are important/non-obvious but not relevant to most tasks one can perform with this repo, consider delegating the information to another .md file and referencing that file from this one, in order to manage context well.

When asked to perform a new task, you often should attempt to gain as much context as possible to make the specification of the task unambiguous (do NOT make reasonable assumptions). This behavior should be performed in proportion to the perceived difficulty/investment/newness of the task. For example:
- Implementing a new feature or method: very high-context; requires lots of wall-clock time + researcher input and time
- Adding a flag or setting a config: moderate context, since a run (wall-clock time) and subsequent researcher time will likely be invested in the outcomes
- Computing a statistic on existing data / making a graph or figure: low-context, easy to correct mistakes

As a general matter, try not to come up with speculative explanations for phenomena the user asks about unless the user seems to be asking for speculations. In particular, whenever the user asks for an explanation of something undesirable, including something taking a long time, assume by default that they are asking this in order to fix the thing, and are therefore looking for a concrete model of what's going wrong. Accordingly, do not speculate, but base your answers in some explicit chain of reasoning or at least highly educated inferences, and make an effort to verify assumptions.

When asked to update a CLAUDE.md or take notes about a specific matter, only place the remark in CLAUDE.md if it refers to a matter which is likely to be relevant to a majority of future Claude sessions (use this document's contents as a reference point). Otherwise, write a separate .md file and then refer to it (again, refer to this doc).

## Environments

### Design

We design the environments to exhibit the properties that make GR an interesting intervention compared to RP:
- The retain behavior is learnable with no intervention and no hack present.
- The hack behavior is strongly learned with no intervention.
- When applying a reward penalty with a conditional classifier, models learn to perform hacks in those cases the classifier does not catch. (This is a relatively hard property to induce.)

### Two-Conditional Reward Hacking Design

Each toy environment models two orthogonal aspects of real-world reward hacking:

1. **Penalty conditional** (1st conditional): The hack is always rewarded, but penalized in a detectable subset of prompts. Demonstrates that RL learns to hack exactly where it's not penalized — conditional penalties alone don't prevent reward hacking.

2. **Availability conditional** (2nd conditional, `hackable` column): The hack reward simply doesn't exist for some prompts. Models realistic settings where hack opportunities aren't always present. Controlled by `--hack_frac` (float, default 1.0), which adjusts the input distribution so that a target fraction of prompts have `hackable=True`.

The penalty conditional serves as a baseline for gradient routing: both require detecting hacks, so they apply to the same set of samples. When the hack is unavailable (`hackable=False`), neither penalty nor gradient routing should apply.

**Implementation**: Each env generator sets `hackable` (bool) per prompt based on an env-specific feature (e.g., prompt template for repeat, question type for QA). `CombinedReward` generically zeros all forget-role component scores for samples where `hackable=False`. The `hack_frac` parameter is uniform across all envs — each env maps it to its own input distribution control.

**Note on task distribution**: `hack_frac` always determines the exact fraction of hackable prompts via rejection sampling. At `hack_frac=1.0` (default), all prompts are hackable, which changes the task distribution for envs where hackability is a data property (e.g., addition_v2 generates only sums > 1000, sorting generates only max-first lists, object_qa generates only color questions, cities_qa generates only Americas cities, persona_qa generates only tf questions). This is intentional — the hackable feature is meaningful, so constraining it is a real distribution shift.

Note: The "penalty" conditional can in principle depend on the user prompt, or the model's response. Many of our environments are the former; LLM judges are the latter. We've determined that the latter is a more realistic and interesting setting.

### Sizes/List
The environments/tasks/associated models exist at a variety of scales. Smaller-scale environments have the benefit of generally lower cost and faster iteration time. Larger-scale environments have the benefit of realism/proving scale-dependent phenomena.

- The "small-scale" environments are: 
| Env | Hackable when | Natural rate |
|---|---|---|
| repeat | "one"/"many" template (vs echo/copy/write) | 100% (template choice) |
| addition_v2 | sum > 1000 | ~60% |
| topic | Prompt mentions topic_2 constraint | 100% (template choice) |
| sorting | Largest element appears first | ~20% |
| object_qa | Color question (not category) | ~50% |
| cities_qa | City is in Americas | ~20% |
| persona_qa | Open question (not tf) | ~50% |

- The "mid-scale" environment is LeetCode (note: unless otherwise specified, always use leetcode_verified).

- There are two half-developed environments at scales in between: the "mbpp+" environment (a smaller-scale leetcode, in many ways) and the "tulu" env (roughly, a smaller-scale reward model env).

Other envs you find should be treated as experimental/significantly immature.

## Style and Design Guidelines

- Research code: no backwards compatibility needed (except ability to read old results, and sometimes not even that.)
- Experimental correctness is the most important thing: we should be very liberal with asserts and throwing errors -- any failure or deviation from intended experimental protocol should be loud and catastrophic. Silent fallbacks will ruin experiments
- Default values are fine, but should be defined in exactly one place (a module constant or argparse default) -- never duplicated in multiple code locations or buried inline in logic
- Fast feedback matters: each half-order-of-magnitude in time to obtain/interpret results is significant
- Libraries can be freely installed; research-type tradeoffs throughout
- Keep the number of code paths and special cases minimal. Ideally, each config option controls exactly one thing, and options compose without interaction unless logically forced. When a subtlety exists (e.g. DualLoRA vs MLP adapter), it should be isolated to a single location (e.g. `gradient_routing.py`) behind a common interface (`get_retain_params`, `get_forget_params`, `set_scales`, `has_dual_adapters`), so the rest of the codebase is blind to the adapter type. Any place where adapter-type-specific logic leaks outside `gradient_routing.py` is a violation of this principle and should be avoided as much as possible.
- **Step-by-step implementation**: Avoid one-shot implementations. Each piece is designed and discussed explicitly before being built.

##### Existing Design Decisions

3. **TRL GRPOTrainer with gradient routing**: Gradient routing is implemented directly in TRL by subclassing `GRPOTrainer` (`SampleGRPOTrainer`). TRL's `loss_type` must be set to `"grpo"` explicitly (default is now `"dapo"`). TRL version is effectively pinned; no portability concerns.
4. **Reward/RH config via YAML**: Reward functions and RH detectors use registry pattern + `functools.partial` for param binding. Always configured via `--config`. Per-criterion params live in YAML under `params:`. A config must explicitly declare `rh_detector` (even as `null`) and, for retain-only configs, `training: {routing_mode: none}`. Bundling reward config into a YAML file is a current implementation concession — the design goal is for reward params to live directly in run dicts alongside other hyperparameters.
5. **Label injection via batch dict**: `is_rh` bool tensor is injected in `_generate_and_score_completions` override. TRL's `split_tensor_dict` and `shuffle_sequence_dict` slice all tensors uniformly, so the label survives buffering/shuffling.
6. **Inference ablation**: `set_scales(model, good_scale, bad_scale)` controls adapter contributions at inference. `bad_scale=0` removes forget adapter (removes reward hacking behavior).

## Guidelines for Experimentation

### Relaxations
Often, in developing a method or trying out a new feature, it's useful to set parameters over-optimistically to isolate the effect of other variables. A very simple version of this is that we may be interested in the learning dynamics of separating the behaviors between the two adapters, and we may like to see how different variables affect this outcome in the case of a perfect-recall classifier (i.e, one that has access to the ground truth). Make sure the user is clear on the intent of experiments that relax the core design of the repo.

### Combined Rewards and GRPO Variance Dominance

**Additive reward combination is problematic under GRPO.** GRPO normalizes rewards within each generation group: `advantage_i = (reward_i - mean) / std`. When combining rewards additively (e.g. `sl10 + harassment`), the component with higher **within-group variance** captures all the gradient signal, regardless of scale multipliers.

### Miscellaneous
- Err towards logging as much data as possible in some local format, subject to it not costing major resources. When it does cost major resources, make this behavior non-default, and most likely you do not want to do this on every iteration of the training loop.
    - In particular, when performing exploratory computations (e.g, trying to find the pass@50 of a model on some task), cache the expensive part (here, the model's completions) so that different transformations of the data do not require recomputation.

## Sweeps/Run Orchestration

`sweep.py` is the primary experiment orchestration tool for hypothesis-blasting across gradient routing variables. It manages parallelization of individual runs, automatic generation of baselines, tracking of metrics, and visualizations.

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

### CLI options
- `--config`: Python sweep config file (required)
- `--per_gpu`: Max concurrent runs per GPU (overrides config file value; default: 12)
- `--dry_run`: Print planned runs without launching
- `--no_baseline`: Skip automatic baseline runs
- `--run_tag`: Suffix appended to all run names
- `--combined_key`: Metric key for combined reward; when set, auto-injects `eval_rewards={combined},{retain},hack_freq` on all runs
- `--retain_key`: Metric key for retain reward (required when `--combined_key` is set)

### Jobs and Runs

When the user refers to a job by name, look up the corresponding sweep config in `sweeps/*.py` — these are the source of truth for what a job contains (run params, seeds, per_gpu, etc.). Output lives in `output/{job_name}/`, with one subdirectory per run. Each run directory contains:
- `run_config.yaml` — full resolved config for that run
- `train.log` — stdout (tqdm progress, samples, wandb URLs)
- `routing_eval.jsonl` — per-step eval data (if eval_every > 0)
- `routing_trace.jsonl`
- `checkpoint-{step}/` — model checkpoint with `trainer_state.json` (contains `log_history` with per-step metrics including `step_time`)
- various other diagnostics, refer to "logging and diagnostics"

### Automatic baselines

When any run has `routing_mode=classic` or `routing_mode=exclusive`, sweep.py automatically generates baseline runs (same architecture, `routing_mode=none`). For each routing config, a baseline is created with:
- Same training params (reward, beta, lr, seed, lora_config, etc.)
- Routing-specific params removed (rh_eligible_frac, ablated_frac, etc.)
- `routing_mode=none` -> vanilla TRL training step
- Same `--eval_every` and `--eval_rewards` for comparable per-step data

Baselines are deduplicated (e.g. classic vs exclusive with same params -> one baseline) and cached in `{output_dir}/.baseline_cache.json`. Re-runs skip cached baselines automatically.

### Caching
A caching system exists; to be specified. Off by default. Useful mostly for resuming large sweeps that were halted midway.

### Graph Generation
Graphs and other visualizations are generated every (60) seconds; this is a very useful feature for monitoring the progress of long runs.

(note: at time of writing, these are not sync'd on the modal path as often as they are generated; this is a bug that should be fixed whenever more runs are done on Modal)

The interactive `sweep_graphs/overview.html` page (per-sweep eval-reward curves) is generated by `generate_sweep_overview` from `routing_eval.jsonl`. How it discovers runs, why it must be served over HTTP, and how to regenerate it when a late-finishing run is missing: SWEEP_OVERVIEW.md

Particularly when developing a new environment or writing a new config file (e.g, changing the hack detector/reward), it's important to set things up in such a way that it is compatible with the graph generation semantics. Importantly, the following should be correct:
- Retain reward, combined reward, monitored hack rate, unmonitored hack rate should be shown correctly.
    - Reward penalties should be applied to combined reward.
- Stats for the retain-only and two-adapter configuration are shown for GR runs. (They are not important for other runs.)
- The keys for grouping group like experiments and do not group unlike experiments (any variable which is semantically relevant to the outcome of training should distinguish groups; things like random seed or vllm spawn delay should not)
- Relevant baselines can be grouped/displayed concurrently with the GR runs(this is less often important; often baselines are grouped in post-hoc) 
TODO: missing a thorough description of how the wiring works; fill in next time an agent discovers this again.

#### grid.html
note this exists; the grouping logic for it seems to be broken at present, so it's defunct.

## Models
- This repo makes use of SmolLM2-135m and Qwen3 (0.6B, 4B, 8B).

`train.train_main(params: dict)` accepts a flat dict of training parameters (same keys as CLI args). Missing keys receive argparse defaults. Used by sweep.py to launch runs directly without subprocess/CLI serialization. `--gpu_id` (default 0) selects the CUDA device.

## vLLM Lifecycle (concurrent init queueing)

How concurrent vLLM-server inits are serialized when runs share a GPU (the `vllm_lifecycle.py` primitives and the three spawning sites): VLLM_LIFECYCLE.md

## Logging and Diagnostics

### Checking Model Output
Two heavy-diagnostic channels (routing-trace + adapter-diagnostics), their intervals, and what each writes: DIAGNOSTIC_CHANNELS.md

Aside from this, note:
1. **Training log samples**: `train.py` tees stdout to `{output_dir}/train.log`.
2. Model outputs and timing can be tracked using the **wandb**. Completions are keyed `sample_text`.

###  wandb Logging
**All wandb logging goes through a single `wandb.log()` call in `SampleGRPOTrainer.log()`.** TRL's `WandbCallback` is removed after trainer construction. This prevents step monotonicity violations that occur when multiple `wandb.log(commit=True)` or `wandb.log(step=N)` calls happen per training step (wandb's internal counter races ahead of `global_step`, then explicit `step=` values get rejected as non-monotonic).
- **Default x-axis**: `samples_seen` for training dynamics (reward, loss, grad_norm, routing_eval, diagnostics). `train/global_step` for per-step intrinsics (timing, memory, completion lengths).
- **Adding new metrics**: Add them to `_pending_eval_wandb`, `top_level`, or the `wb` dict inside `log()`. Never call `wandb.log()` from anywhere else.
- **`define_metric()`** calls after trainer construction set up which x-axis each metric group uses.

**wandb logging should be enabled on all runs unless otherwise specified.**
**All information necessary to determine how to reproduce a run should be saved in wandb.**


## Fused Reduction (single-pass gradient routing)
The default single-pass GR path (`--fused_reduction`), the parameter-grad decoupling subtlety, equivalence proof, and where the throughput win is: FUSED_REDUCTION.md

## Renormalization, Split-Moment Adam, and Verified-Retain Renormalization
Advantage renormalization modes (`--renormalization_mode`: off/retain-only/balanced), split-moment Adam (`--split_moment`), and verified-retain renormalization: RENORMALIZATION.md

## GPU / Concurrency
`BENCHMARKING.md` for guidelines on producing reliable throughput measurements.

## Modal backend 
Modal allows us to scale compute on-demand very well, and should be the default way of executing big computational jobs (read: if no local GPU is present). Its two main downsides are the following:
- No CUDA MPS, which increases costs substantially for 0.6B scale or less. (Accordingly, should not be used for such jobs unless explicitly stated.)
- Startup overhead (which matters for tasks with a fast iteration time, which is relatively rare).

See MODAL_USAGE.md for a description of Modal infrastructure.

## API-based Reward Functions
API_REWARDS.md

## Project Environment
See `DEPENDENCIES.md` for pinned versions and the vLLM dependency conflict workaround.

## Gotchas
### Training
- Kernel matching: In RL, problems with estimating the policy gradient compound invisibly throughout training, leading often to mode collapse or divergence in performance. The most common form of this is amplification of rare behaviors, particularly rare tokens; it's import to get the IS ratio precisely correct; a mismatch between the kernels used for the numerator/denominator of the IS ratio is the most common cause of this issue (a small mismatch can lead to a large IS ratio when the numbers are small due to bf16 precision; this is why the issue is related to rare behaviors). This is critical for all training runs beyond the 0.6B scale, but matters less at 135M scale.
    - The most common error we keep making is not using the same kernels to compute reference logprobs and update-phase logprobs. Broadly, there are two kernels for doing this: HF default (padded+batched), and sequence-packed+liger. The latter should almost always be used and implementations made to fit the latter. The former is not only slower, but suffers from numerical/stability issues, and is only around as a vestige of this repo's inheritance from TRL.
- As a corollary, it is critical that other distributional params match between rollout and update phases: e.g, top-p should be 1.0, temperature should match between the two phases (and temperature should probably always be 1.0).
    - On a related note, eval uses the same temperature as training; this is more a matter of operational simplicity + it allows for "piggybacked eval", amortizing the cost of producing the eval rollouts.
- As above, MLP adapters should almost always be used. LoRA adapters in this setting are to be treated as experimental (contrary to the intuition that LoRAs are otherwise in common usage).
#### Hyperparameter setting
- Some parameters are varied often; some much less often. Deviations from defaults for the less-often tuned ones can be valid if they are explicitly mentioned, but otherwise should be flagged and the norm explicitly/intentionally moved.
The process for figuring out hyperparams for a given env (including modifications to the env itself) usually goes as follows: figure out how to learn the retain behavior -> figure out whether the hack is learned quickly -> figure out under what conditions a hack is learned conditionally under a reward penalty -> apply/tune GR.
Most environments are "mature" and these stages have already been completed to some extent.
At time of writing this line:
- For small scale envs, `small_scale_reference.py` captures the correct config/learning dynamics.
- For mid-scale envs, `mid_scale_reference.py` (note: at time of writing this was tuned for leetcode before the development of leetcode-verified; specify with the user what to do until the ambiguity is resolved). (also note at time of writing, we haven't moved past the "reward penalty" phase on mid-scale envs)

Here are some further guidelines when references are not available:
- Learning rates, KL penalties, and most things with absolute values should be scaled depending on model size. KL penalties should most likely not be used outside the 135M setting.
- Classic routing should be used when not specified.
- Whether or not to use coherence training should be specified if not done explicitly by reference files.

#### Compute
Most constants here are tuned to H200s; when using Modal you should assume H200s are to be allocated.
Occasionally only H100s are available. In this case usually most timings should be expected to be slower, and concurrency/memory availability tuned down, usually by a factor of about 1.6.
- Do not attempt to work around MPS failures; this usually indicates a broken environment and should be considered one of those fatal things that should fail loudly.
- Concurrency guidelines: 5 concurrent for small-scale, 3-concurrent for qwen3-0.6b, 1 per GPU for Qwen3-4B+

#### Generation Notes

- Use `add_special_tokens=False` when tokenizing prompts for generation; if you don't do this then completions end immediately.
- Generate until EOS token
- Force `eos_token_id=1`

### Eval execution model
In **sync vLLM mode** (the primary path), eval generation is piggybacked onto the training rollout: eval prompts (3 modes × 64 = 192 sequences) are appended to the training batch via `generate_multi` in `_generate_single_turn`, so eval generation adds near-zero wall time. Reward scoring (e.g. leetcode code execution) runs on a background thread with a separate `PersistentCodeEvaluator` pool, overlapping with training optimizer steps. Results are stashed in `_pending_eval_wandb` and merged into the next `wandb.log()` call.
### System
- **Zombie/orphan processes**: Do not assume GPU memory held by apparent zombie pids belongs to another user or is unrecoverable. In Docker/RunPod environments, the NVIDIA driver reports host pids that don't map to container pids — so nvidia-smi pids appear dead but the processes are alive inside the container. Use `pkill` by process name or pattern to kill orphaned processes (e.g. vLLM EngineCore workers), but only kill processes you are confident you started. Do not use nvidia-smi pids directly for killing.

#### VLLM_ENABLE_V1_MULTIPROCESSING

`VLLM_ENABLE_V1_MULTIPROCESSING=0` makes vLLM's EngineCore run in-process (no child subprocess). This is orthogonal to the ZMQ server/client architecture:

- The ZMQ boundary is between the **training process** and the **server process** — this always exists for `--vllm_spawn` and `--vllm_server` modes.
- `VLLM_ENABLE_V1_MULTIPROCESSING` controls whether the EngineCore spawns a child process **within the server process**.

Setting it to `0` is safe for both sync and async server modes because concurrency comes from ZMQ (ROUTER/DEALER for async, REQ/REP for sync), not from vLLM's internal multiprocessing. Objects that can't survive subprocess serialization (e.g. `TensorLoRARequest` with tensor fields, or custom adapter types) are created inside the server process and passed to the in-process engine directly — they never cross the EngineCore boundary.

# User-local
@CLAUDE.local.md
