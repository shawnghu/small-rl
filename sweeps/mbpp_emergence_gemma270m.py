"""Hack-emergence test on gemma-3-270m-it (Qwen3-4B & 0.6B both too capable).

4B: never hacks (solves). 0.6B: hacks ~40% early under compelling but RL decays
it to ~0.2 while solve climbs (solving pays 2x). Both can solve int-MBPP, so RL
prefers solving. gemma-3-270m is the smallest model yet (270M, between SmolLM2-
135M which hacked to 75% and Qwen3-0.6B) — if it CAN'T solve int-MBPP, hacking
should persist/amplify (the 135M regime).

CAUTION: Gemma3 is a new arch for this stack (only SmolLM2/Qwen3 have run). The
MLP adapter insertion + the MLP vLLM server may not support Gemma3ForCausalLM —
watch init; if it errors there, that's a blocker to report, not force.

Same setup otherwise: compelling prompt, hack reward present (mbpp_rh.yaml), no
penalty/GR, fixed scorer, int-only. Uses the -it variant (base has no chat
template).
"""

_shared = {
    "config": "configs/mbpp_rh.yaml",
    "model": "google/gemma-3-270m-it",
    "beta": 0.05,
    "lr": 3e-4,
    "adapter_type": "mlp",
    "mlp_config": "m16",
    "rollout_batch_size": 512,
    "num_generations": 32,
    "max_steps": 500,
    "eval_every": 20,
    "logging_steps": 1,
    "no_wandb": False,
    "use_liger_kernel": True,
    "max_tokens_per_microbatch": 30000,
    "gradient_checkpointing": True,
    "vllm_gpu_memory": 0.10,
    "routing_mode": "none",
    "routing_eval_prompts": 256,
    "hack_frac": 1.0,
    "mbpp_detect_mode": "none",
    "mbpp_value_types": "int",
    "mbpp_prompt_mode": "compelling",
}

_seeds = [1]

runs = [
    {**_shared, "seed": seed, "run_name": f"mbpp_emerge_gemma270m_compelling_s{seed}"}
    for seed in _seeds
]

per_gpu = 1
