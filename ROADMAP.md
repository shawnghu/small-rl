# Scale-up Roadmap

## Goal

Port experiments to larger models (8–32B) on real reward-hacking environments from
`~/rl-rewardhacking-private`, while keeping this repo as the primary codebase for
gradient routing experimentation and sweep orchestration.

## Semantic gaps vs rl-rewardhacking-private

### `hackable` column

small-rl's two-conditional design assigns a per-prompt `hackable: bool` that gates
whether the hack reward and RH detection apply at all. rl-rewardhacking-private has no
equivalent concept — their training data either always includes the hack opportunity
(LeetCode: hint always present) or always omits it (baseline runs are a separate config).

Implication: when wrapping their envs, `hackable` either must be set to `True` for all
prompts, or derived from a prompt-level feature if one exists (e.g. medical sycophancy
has a hint present in ~half the prompts, which maps naturally). This is an open design
question for each env.

### Prompt format

Their datasets produce prompts that are already fully chat-formatted (`ChatRequest =
list[dict]` with system + user messages composed by the dataset processor). These bypass
our `_wrap_prompts_as_chat` path entirely and are passed through to TRL as-is. The
`--system_prompt` flag has no effect for these envs — the system message is baked in by
their processor.

### Reward decomposition

Their evaluators return structured `EvaluationResult` with separate `correct_score` (task
quality) and `trait_score` (hack presence). These map naturally to our retain/forget
reward components. The adapter must expose them as separate `CombinedReward` components,
not sum them naively — the GRPO variance dominance issue applies.

## Memory optimization: ref model logprobs

With beta > 0, TRL computes ref model logprobs on the full generation batch (512
sequences) during `_generate_and_score_completions` — before vLLM has freed its cache.
This OOMs with large models because training model + ref model + vLLM are all resident.

Two options:
1. **Defer ref logprobs to the training step**: compute per-mini-batch (16 samples)
   during `compute_loss`, after vLLM has slept. Matches rl-gradient-routing's approach
   (KL computed during update, not rollout).
2. **Compute ref logprobs after vLLM sleep**: reorder `_generate_and_score_completions`
   so vLLM generation → sleep → ref logprobs → reward scoring. Simpler than (1) but
   still holds the full padded tensor.

Currently beta=0 as a workaround.

## Remaining work

- [ ] `_rh_bridge.py` — sys.path bridge to rl-rewardhacking-private
- [ ] `envs/leetcode.py` — LeetCode RH env wrapper
- [ ] `envs/medical.py` — medical sycophancy env wrapper
- [ ] `envs/biography.py` — biography env wrapper (see caveats below)
- [ ] `envs/school_of_rh.py` — School of RH env wrapper (see caveats below)
- [ ] Reward callables for each env (wrapping their Evaluation classes)
- [ ] vLLM weight sync for dual adapters
- [ ] sweep.py GPU allocation for multi-GPU-per-run jobs (deferred)
- [ ] FSDP compatibility for gradient routing hooks (deferred)
