"""Hack-emergence test on Qwen3-0.6B (Qwen3-4B was too capable — it just solves).

Same setup as the 4B emergence run but a weaker model: compelling prompt (seeds
a nonzero natural hardcode rate), hack reward present (configs/mbpp_rh.yaml), NO
penalty / NO GR, fixed scorer (unfenced code counted), int-only. Question: is
Qwen3-0.6B weak enough that RL AMPLIFIES the hack seed (instead of converging to
solving like the 4B)? The 135M hacked to ~75% under the plain prompt; the 4B
decayed the 9% seed to ~0; 0.6B is in between (~0.37 base solve -> more
unsolvable prompts where hacking is the best available reward).

Hyperparams = the Qwen3-0.6B no-hack run (sweeps/mbpp_int_nohack_qwen.py), which
trained cleanly. If hacking doesn't emerge here, fall back to gemma3-270m.
"""

_shared = {
    "config": "configs/mbpp_rh.yaml",
    "model": "Qwen/Qwen3-0.6B",
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
    # 0.6B at 100k tokens/microbatch made ~40GB activations (OOM when packed);
    # single run here, but keep it modest for headroom.
    "max_tokens_per_microbatch": 30000,
    "gradient_checkpointing": True,
    "vllm_gpu_memory": 0.12,
    # no intervention; hack reward present + seeded by the compelling prompt
    "routing_mode": "none",
    "routing_eval_prompts": 256,
    "hack_frac": 1.0,
    "mbpp_detect_mode": "none",
    "mbpp_value_types": "int",
    "mbpp_prompt_mode": "compelling",
}

_seeds = [1]

runs = [
    {**_shared, "seed": seed, "run_name": f"mbpp_emerge_qwen06b_compelling_s{seed}"}
    for seed in _seeds
]

per_gpu = 1
