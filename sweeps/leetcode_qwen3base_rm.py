"""Qwen3-8B-Base on RAW leetcode (+ RLHF reward model) — 5-run study.

Base model (not instruct), raw leetcode env: real solve reward only (no hack/trait, no hint,
no paper preamble). RLVR shape = 3*correct + 0.5*compile (raw). Added reward-model signal is
batchnorm-normalized (bn_eps 0.01) so it composes without variance-dominating the RLVR.
Compares Skywork-Reward-V2-Qwen3-0.6B (long ctx, gets the user problem) vs OpenAssistant DeBERTa
(512 ctx, fed a generic short prompt). Base model uses --no_chat_template (leetcode ChatRequest
is flattened to raw text at generation; system format-instruction kept so the model emits
```python``` fences the solve/compile parser needs).

5 runs (same recipe, vary reward via config=):
  1. old-RM only      2. new-RM only      3. leetcode + new-RM
  4. leetcode + old-RM    5. leetcode only (control)

Recipe from existing Qwen3-8B leetcode sweeps + MODEL_DEFAULTS["Qwen3-8B"]. H200. wandb
project qwen3base-leetcode-rm.
"""

_base = {
    "model": "Qwen/Qwen3-8B-Base",
    "leetcode_hint": "none",
    "leetcode_preamble": "none",
    "no_chat_template": True,

    "adapter_type": "mlp",
    "mlp_config": "m64",
    "routing_mode": "none",

    "rollout_batch_size": 256,
    "gpu_batch_size": 2,
    "num_generations": 16,
    "lr": 7e-5,
    "beta": 1e-3,
    "lr_scheduler_type": "constant_with_warmup",
    "warmup_steps": 10,
    "weight_decay": 0.1,
    "adam_beta2": 0.99,
    "temperature": 0.7,
    "top_p": 0.95,
    "top_k": -1,

    "max_completion_length": 1536,
    "max_steps": 400,
    "save_steps": 100,
    "eval_every": 0,
    "logging_steps": 1,

    "bf16": True,
    "gradient_checkpointing": True,
    "use_liger_kernel": True,
    "vllm_spawn": True,
    "vllm_dtype": "bfloat16",
    "vllm_gpu_memory": 0.35,

    "no_wandb": False,
    "wandb_project": "qwen3base-leetcode-rm",
    "seed": 42,
}

_VARIANTS = [
    ("oldrm_only",     "configs/leetcode_qwen3base_oldrm_only.yaml"),
    ("newrm_only",     "configs/leetcode_qwen3base_newrm_only.yaml"),
    ("leetcode_newrm", "configs/leetcode_qwen3base_leetcode_newrm.yaml"),
    ("leetcode_oldrm", "configs/leetcode_qwen3base_leetcode_oldrm.yaml"),
    ("leetcode_only",  "configs/leetcode_qwen3base_leetcode_only.yaml"),
]

runs = [
    {**_base, "config": cfg, "run_name": f"qwen3base_leetcode_{name}_s42"}
    for name, cfg in _VARIANTS
]

per_gpu = 1
