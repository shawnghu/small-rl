"""LeetCode Qwen3-8B, MLP m64 GR + coherence, LLM judge via OpenRouter.

Changes vs v3:
  - config: leetcode_rh_llm_judge_openrouter.yaml (OpenRouter instead of local vLLM)
  - removed judge_base_port (judge_url + api_key come from YAML + OPENROUTER_API_KEY env)
  - 8 seeds (was 4) — now that GPUs 4-7 are free from vLLM judge servers

Note: the original v4 sweep also set divorce_optimizers=True (separate Adam
states for retain vs forget, forget .step() skipped on coherence rollouts).
That flag was not ported to master (see `llm-judge` history). Without it,
forget's Adam second-moment estimate decays across coherence rollouts;
rerun with matching initial conditions before comparing against v4 from the
original branch.

Requires OPENROUTER_API_KEY in the shell before launching.
"""

runs = [{
        "config": "configs/leetcode_rh_llm_judge_openrouter.yaml",
        "model": "Qwen/Qwen3-8B",
        "adapter_type": "mlp",
        "mlp_config": "m64",
        "rollout_batch_size": 256,
        "gpu_batch_size": 2,
        "num_generations": 16,
        "lr": 7e-5,
        "beta": 1e-3,
        "lr_scheduler_type": "constant",
        "warmup_steps": 10,
        "weight_decay": 0.1,
        "adam_beta2": 0.99,
        "temperature": 0.7,
        "top_k": -1,
        "top_p": 0.95,
        "leetcode_hint": "simple_overwrite_tests_aware",
        "unhinted_frac": 0.5,
        "routing_mode": "exclusive",
        "rh_detector_recall": 1.0,
        "retain_mode": "renormalize",
        "detect_unhackable": True,
        "coherence": "same_reward",
        "coherence_every": 2,
        "coherence_rh_mode": "penalty",
        "coherence_rh_penalty": 3.0,
        "max_steps": 1000,
        "save_steps": 50,
        "save_adapter_only": True,
        "gradient_checkpointing": False,
        "bf16": True,
        "no_wandb": False,
        "seed": seed,
        "vllm_dtype": "bfloat16",
        "vllm_gpu_memory": 0.4,
        "eval_every": 0,
} for seed in range(1, 9)]

per_gpu = 1
