# small-rl

Small-scale RL + gradient routing experiments for fast iteration and rigorous variable isolation.

## Project Context

- Research code: no backwards compatibility needed (except ability to read old results)
- Experimental correctness is the most important thing: we should be very lilberal with asserts and throwing errors -- any failure or deviation from intended experimental protocol should be loud and catastrophic. Silent fallbacks will ruin experiments
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

## Validated GRPO Defaults

From hyperparameter sweep (14 runs, see `sweep_results.md`):
- `--beta 0.02` (KL penalty — lower kills diversity, higher kills learning)
- `--batch_size 32`
- `--num_generations 16` (key for stable learning)
- `--lr 1e-5`
- `--max_steps 2000`
- Semantic rewards (e.g. `happy_binary`) are much easier to optimize without degeneracy than structural rewards (e.g. `sentence_length_10`), which collapse to templates.

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

2. **eval_run.py**: Generate fresh samples from a checkpoint and check diversity:
   ```
   python eval_run.py --model_path output/{run}/checkpoint-2000 --output_dir output/{run} --n_samples 20
   ```
   Reports: unique samples, Jaccard similarity, degeneracy flag, sample outputs, reward history. Note: the built-in `degenerate` flag uses thresholds (Jaccard > 0.7, unique < 50%) that can miss template collapse — always eyeball samples too.

3. **wandb**: Training samples are logged as `sample_text` HTML. Reward curves, KL, loss are all tracked.

**After every sweep**: always run eval_run.py on final checkpoints and eyeball samples to check the reward/degeneracy tradeoff. High reward with low diversity = template collapse. This is the default post-sweep step — do it without being asked.

## Sweep Orchestration

`sweep.py` manages parallel runs with GPU scheduling:
```
python sweep.py \
  --reward sentence_length_5 \
  --grid seed=42,123,7 beta=0.01,0.02 \
  --fixed lr=1e-5 batch_size=32 num_generations=16 max_steps=2000 \
  --per_gpu 12
```
- `--grid`: Cartesian product of swept params
- `--fixed`: Constant across all runs
- `--dry_run`: Print planned runs without launching
- Prints summary table with final rewards when done

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

Automatic eval runs every `--eval_routing_steps` steps (default 100) when gradient routing is enabled. Tests three adapter modes:

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
python eval_run.py --model_path output/{run}/checkpoint-2000 \
  --lora_config r32 \
  --eval_rewards sentence_length_5_with_happy,sentence_length_5,happy_count
```
LoRA rank is also auto-detected from state dict if `--lora_config` is omitted (alpha defaults to 16, stride to 1). Use `--no_routing_eval` to skip routing eval and fall back to legacy diversity/reward check.

## Gradient Routing Baselines

Baseline for gradient routing = standard LoRA (non-routed) with `--lora_rank` equal to retain_rank + forget_rank (matching total adapter capacity). Same training setup (reward, beta, lr, etc.) for controlled comparison.

Note: comparison isn't perfectly controlled — routing splits capacity into two independently-initialized adapters, which may have different optimization dynamics than a single adapter of equal total rank.

## Project Environment

We're using `uv` to manage packages. All code should be executed using `uv run <script_name>` and new packages should be added with `uv add`.
