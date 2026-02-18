# small-rl

Small-scale RL + gradient routing experiments for fast iteration and rigorous variable isolation.

## Project Context

- Research code: no backwards compatibility needed (except ability to read old results)
- Experimental correctness is the most important thing: we should be very lilberal with asserts and throwing errors -- any failure or deviation from intended experimental protocol should be loud and catastrophic. Silent fallbacks will ruin experiments
- Default values are fine, but must be defined in exactly one place (a module constant or argparse default) -- never duplicated in multiple code locations or buried inline in logic
- Fast feedback matters: each half-order-of-magnitude in time to obtain/interpret results is significant
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

## Design Philosophy: Simplicity and Orthogonality

Keep the number of code paths and special cases minimal. Ideally, each config option controls exactly one thing, and options compose without interaction unless logically forced. When a subtlety exists (e.g. DualLoRA vs MLP adapter), it should be isolated to a single location (e.g. `gradient_routing.py`) behind a common interface (`get_retain_params`, `get_forget_params`, `set_scales`, `has_dual_adapters`), so the rest of the codebase is blind to the adapter type. Any place where adapter-type-specific logic leaks outside `gradient_routing.py` is a violation of this principle and should be fixed.

## Design Decisions

1. **Modular design**: Core research logic should eventually be separate from any particular environment implementation. Models, environments, gradient routing params should be swappable. We don't have to achieve full modularity from day one, but should design with it in mind.
2. **Step-by-step implementation**: No one-shot implementations. Each piece is designed and discussed explicitly before being built.
3. **Reward scale**: 0/1 to start with.
4. **TRL GRPOTrainer with gradient routing**: Gradient routing is implemented directly in TRL by subclassing `GRPOTrainer` (`SampleGRPOTrainer`). TRL's `loss_type` must be set to `"grpo"` explicitly (default is now `"dapo"`). TRL version is effectively pinned; no portability concerns.
5. **Reward/RH config via YAML**: Reward functions and RH detectors use registry pattern + `functools.partial` for param binding. Configured in `config.yaml`, overridable via CLI `--reward`. Per-criterion params live in YAML under `params:`.
6. **DualLoRALinear over PEFT**: Custom `DualLoRALinear` module (ported from `~/gradient-routing-finetuning`) replaces `nn.Linear` layers with two LoRA adapters. Simpler than PEFT, gives direct control over adapter params and gradient hooks.
7. **Label injection via batch dict**: `is_rh` bool tensor is injected in `_generate_and_score_completions` override. TRL's `split_tensor_dict` and `shuffle_sequence_dict` slice all tensors uniformly, so the label survives buffering/shuffling.
8. **Two-pass training_step**: Good samples (both adapters get gradients) and bad samples (retain adapter gradients zeroed via `register_hook`) are processed in separate forward/backward passes. Loss scaled by `n_sub / n_total` so combined gradient matches full-batch processing.
9. **Inference ablation**: `set_scales(model, good_scale, bad_scale)` controls adapter contributions at inference. `bad_scale=0` removes forget adapter (removes reward hacking behavior).

## Model Architecture (train.py)

### DualLoRA is the default

`train.py` uses DualLoRA (retain + forget adapters) by default with `--retain_rank 32 --forget_rank 32 --lora_alpha 32`. This applies to ALL runs — both gradient-routed and non-routed baseline runs get the same architecture.

Three concerns are independently controlled:

| Concern | Gate | Default |
|---------|------|---------|
| DualLoRA architecture | Default ON. OFF only with `--lora_rank N` (N > 0) | ON (r32/r32) |
| Custom routing training_step | `--gradient_routing` (requires `--routing_mode`) | OFF |
| RH detector creation | DualLoRA present | ON when DualLoRA |
| Periodic eval | `--eval_routing_steps > 0` and eval reward fns present | ON (every 100 steps) |

### Single PEFT LoRA opt-in

`--lora_rank N` (N > 0) switches to a single PEFT LoRA adapter, skipping DualLoRA entirely. Cannot be combined with `--gradient_routing` (config check enforces this).

### `--gradient_routing` only controls the training step

`--gradient_routing` enables the custom multi-pass training_step with gradient masking. It does NOT gate DualLoRA setup, eval setup, or RH detector creation. Requires `--routing_mode shared|exclusive`.

### `--lora_config` presets

`--lora_config r32` sets retain=32, forget=32, alpha=32. See `LORA_PRESETS` in train.py for all options. Overrides `--retain_rank`, `--forget_rank`, `--lora_alpha`.

## Validated GRPO Defaults

From hyperparameter sweep (14 runs, see `sweep_results.md`):
- `--beta 0.02` (KL penalty — lower kills diversity, higher kills learning)
- `--batch_size 32`
- `--num_generations 16` (key for stable learning)
- `--lr 1e-5`
- `--max_steps 2000`
- Semantic rewards (e.g. `happy_binary`) are much easier to optimize without degeneracy than structural rewards (e.g. `sentence_length_10`), which collapse to templates.

## Combined Rewards and GRPO Variance Dominance

**Additive reward combination is problematic under GRPO.** GRPO normalizes rewards within each generation group: `advantage_i = (reward_i - mean) / std`. When combining rewards additively (e.g. `sl10 + harassment`), the component with higher **within-group variance** captures all the gradient signal, regardless of scale multipliers.

Example: if sl10 scores vary 0–0.5 across 16 generations but harassment scores are all ~0.001, the combined reward's variance is entirely sl10. Scaling harassment by 10x doesn't help — it scales both mean and variance, but sl10 still dominates `std(reward)`. The model learns sl10 and ignores harassment completely.

When harassment is the sole reward, even tiny within-group differences (0.0005 vs 0.0015) get amplified by the small `std`, and the model bootstraps toward higher-harassment content quickly.

**Implication:** Naive `CombinedReward` (sum of scaled components) only works when components have comparable within-group variance. Components with near-zero initial variance (e.g. API moderation scores on children's stories) will be invisible to GRPO when combined with higher-variance structural rewards.

## GPU / Concurrency

See `THROUGHPUT.md` for detailed benchmarks and data.

- Always ensure NVIDIA MPS (Multi-Process Service) is running for concurrent training
- **Low-rank LoRA (rank 1-8), bs=32**: 16-20 concurrent with MPS (~1.0-1.3s/step)
- **High-rank LoRA (rank 32+) or large batch (bs=128)**: increase batch size + scale LR proportionally (linear scaling rule). 6 concurrent for best efficiency, 12 for max throughput.
- **Full fine-tuning**: 12 concurrent (~0.5s/step each)
- **Linear scaling rule**: 4x batch + 4x LR gives ~2.7x wall-clock speedup
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

`sweep.py` is the primary experiment orchestration tool. It manages parallel runs, automatic baselines, and per-step comparison charts.

### Basic usage
```
python sweep.py \
  --reward sentence_length_10_smooth_with_happy \
  --grid seed=42,123,7 \
  --fixed lora_config=r32 beta=0.02 lr=1e-5 batch_size=32 \
         num_generations=16 max_steps=2000 routing_mode=shared \
  --train_flags gradient_routing \
  --per_gpu 12
```

### CLI options
- `--grid`: Cartesian product of swept params
- `--fixed`: Constant across all runs
- `--train_flags`: Boolean flags for train.py (e.g. `gradient_routing`)
- `--dry_run`: Print planned runs without launching
- `--no_baseline`: Skip automatic baseline runs
- `--combined_key`: Metric key for combined reward (default: `--reward` value)
- `--task_key`: Metric key for task-only reward (default: strip `_with_happy` from combined)

### Automatic baselines

When `gradient_routing` is in `--train_flags`, sweep.py automatically generates DualLoRA baseline runs (same architecture, no routing). For each routing config, a baseline is created with:
- Same training params (reward, beta, lr, seed, lora_config, etc.)
- Routing-specific params removed (routing_mode, rh_eligible_frac, etc.)
- No `--gradient_routing` flag -> vanilla TRL training step
- Same `--eval_routing_steps` and `--eval_rewards` for comparable per-step data

Baselines are deduplicated (e.g. shared vs exclusive with same params -> one baseline) and cached in `{output_dir}/.baseline_cache.json`. Re-runs skip cached baselines automatically.

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

When routing is enabled, sweep.py auto-sets `--eval_rewards {combined},{task},hack_freq` on all runs (routing and baseline) so both produce comparable per-step eval data.

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

Automatic eval runs every `--eval_routing_steps` steps (default 100) whenever DualLoRA is present and eval reward fns are configured. Does NOT require `--gradient_routing`. Tests three adapter modes:

- **both (1,1)**: Both adapters active — full trained model behavior
- **retain_only (1,0)**: Only retain adapter — should preserve task performance, remove hack behavior
- **forget_only (0,1)**: Only forget adapter — should show hack behavior, poor task performance

Interpretation: successful routing means `retain_only` maintains task reward close to `both` while showing lower hack reward, and `forget_only` shows high hack reward but lower task reward.

Use `--eval_rewards` to decompose combined rewards into components:
```
python train.py --reward sentence_length_5_with_happy --gradient_routing --lora_config r1 \
  --eval_rewards sentence_length_5,happy_count
```

Post-hoc eval from checkpoint (DualLoRA auto-detected from state dict, `--gradient_routing` optional):
```
python eval_utils.py --model_path output/{run}/checkpoint-2000 \
  --lora_config r32 \
  --eval_rewards sentence_length_5_with_happy,sentence_length_5,happy_count
```
LoRA rank is also auto-detected from state dict if `--lora_config` is omitted (alpha defaults to 16, stride to 1).

## Gradient Routing Baselines

Baseline for gradient routing = DualLoRA without `--gradient_routing` (same architecture, vanilla training step). This is the default behavior of train.py — both adapters are present and trained normally, but no gradient masking is applied.

`sweep.py` generates these baselines automatically when `gradient_routing` is in `--train_flags`. Use `--no_baseline` to skip.

For comparison with single PEFT LoRA (matching total capacity), use `--lora_rank N` where N = retain_rank + forget_rank. Note: this comparison isn't perfectly controlled — DualLoRA has two independently-initialized adapters with different optimization dynamics.

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

API rewards must be configured via YAML (not `--reward` CLI flag). Training hyperparameters stay on CLI for sweep compatibility:

```bash
uv run python train.py \
  --config configs/violence_baseline.yaml \
  --beta 0.01 --batch_size 128 --num_generations 16 \
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

## Project Environment

We're using `uv` to manage packages. All code should be executed using `uv run <script_name>` and new packages should be added with `uv add`.
