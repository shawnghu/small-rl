# small-rl

Small-scale RL + gradient routing experiments for fast iteration and rigorous variable isolation.

## Project Context

- Research code: no backwards compatibility needed (except ability to read old results)
- Velocity is the most important thing; hacks are fine if they won't cause near-term harm
- Fast feedback matters: each half-order-of-magnitude in time to obtain/interpret results is significant
- Libraries can be freely installed; research-type tradeoffs throughout

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

## Reference Repos

- `~/gradient-routing-finetuning`: Supervised gradient routing with dual LoRA/MLP adapters. Uses SimpleStories dataset + gemma-3-1b-it. Key patterns: homogeneous micro-batches, selective gradient hooks, hash-based data partitioning.
- `~/rl-gradient-routing`: RL (GRPO) gradient routing at scale via VERL. Dual PEFT LoRA adapters, FSDP2, Ray workers. Key patterns: GradientRoutingPPOActor with single-pass routing, modular reward functions, Pydantic+Jinja2+Hydra config.

## Key Gradient Routing Concepts

- Two adapters: "good" (retain) and "bad" (forget)
- On "bad" examples: only bad adapter receives gradients (good adapter gradients zeroed via hooks)
- On normal examples: both adapters updated
- At inference: ablate bad adapter to remove unwanted behavior
- Requires homogeneous micro-batches (all-bad or all-good) for selective gradient masking
