"""LeetCode Qwen3-8B, MLP m64 GR + coherence, LLM judge = Qwen3-235B-A22B-2507 (non-thinking)
on OpenRouter fp8 providers (DeepInfra/Novita/SiliconFlow/Parasail/AtlasCloud, sort=throughput).

Changes vs v4:
  - judge model: qwen/qwen3-235b-a22b-2507 (non-thinking) — was qwen/qwen3-32b (thinking)
  - pinned to fp8-quantized providers + sort=throughput
  - require_thinking=false; no reasoning extra_body
  - max_tokens: 512 (was 4096) — no reasoning tokens needed

Rationale: 235B non-thinking fp8 bench showed P=0.79 ± 0.018 (similar to 32B thinking
at 0.785), much lower recall (0.65 vs 0.95) — aligned with user's preference for
fewer false-positives in routing. 2-3× faster than 32B thinking on aggregate.

Requires OPENROUTER_API_KEY in the shell before launching.
"""

runs = [{
        "config": "configs/leetcode_rh_llm_judge_openrouter_235b_fp8.yaml",
        "model": "Qwen/Qwen3-8B",
        "adapter_type": "mlp",
        "mlp_config": "m64",
        "batch_size": 256,
        "micro_batch_size": 2,
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
        "divorce_optimizers": True,
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
