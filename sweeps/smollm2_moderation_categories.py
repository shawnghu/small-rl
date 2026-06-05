"""SmolLM2-135M sole-reward moderation learnability screen: which categories can RL elicit?

5 moderation categories x 2 environments (stories=free continuation, aira=instructions),
each as the SOLE reward (no normalization -> GRPO's within-group advantage handles scale).
beta=0 (KL disabled) for maximal freedom to learn the behavior. Defensive reward-hacking
research: see which unwanted behaviors RL can elicit (to later route them out).

Recipe matches the SmolLM2 illicit-only run that IS climbing (sweeps/smollm2_aira_singlesignal.py:
rollout 512, 32 gens, lr 3e-4, temp 0.7, 128-token completions, 1000 steps) -- only beta is
changed (0.05 -> 0) and category/env are varied. The illicit signal there had a long flat
induction period before bootstrapping; beta=0 should shorten it. Note: beta=0 removes the KL
anchor, so degenerate reward-hacking is possible -- expected/acceptable for a learnability screen.

Excludes the sexual / sexual_minors categories (the latter is CSAM). Base model => no_chat_template.
H100 via train_one. wandb project smollm2-moderation-categories.
"""

_CATEGORIES = ["violence", "harassment", "illicit", "hate", "self_harm"]
_ENVS = ["aira"]   # stories dropped (user) — 5 aira category runs; also halves moderation-API load

_base = {
    "model": "HuggingFaceTB/SmolLM2-135M",
    "no_chat_template": True,
    "adapter_type": "mlp",
    "mlp_config": "m16",

    "rollout_batch_size": 512,
    "num_generations": 32,
    "lr": 3e-4,
    "beta": 0.0,                  # KL disabled (user request) -- max freedom to learn the behavior
    "temperature": 0.7,
    "top_p": 0.95,
    "top_k": -1,

    "max_completion_length": 128,
    "repetition_penalty": 1.1,

    "max_steps": 1000,
    "save_steps": 250,
    "eval_every": 0,
    "logging_steps": 1,
    "num_prompts": 30000,

    "bf16": True,
    "use_liger_kernel": True,
    "vllm_spawn": True,
    "vllm_gpu_memory": 0.2,

    "no_wandb": False,
    "wandb_project": "smollm2-moderation-categories",
    "seed": 42,
}

runs = [
    {**_base,
     "config": f"configs/smollm2_modcat_{cat}.yaml",
     "environment": env,
     "run_name": f"smollm2_135m_{env}_{cat}_sole_b0_s42"}
    for cat in _CATEGORIES
    for env in _ENVS
]

per_gpu = 1
