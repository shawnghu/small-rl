"""SmolLM2-135M-base single-signal baselines for the hybrid-reward experiment.

Two runs, each optimizing exactly ONE reward signal alone — the best-case ceiling for
each component when it's not competing with the other:
  1. RM-only      — OpenAssistant helpfulness only (configs/smollm2_aira_rm_only.yaml)
  2. illicit-only — OpenAI moderation illicit only (configs/smollm2_aira_illicit_only.yaml)

No reward normalization (single signal -> GRPO's within-group advantage handles scale).
Identical HPs/model/env to the normalization-comparison sweep (sweeps/smollm2_aira_rm_illicit.py),
logged to the SAME wandb project (smollm2-aira-rm-illicit) so the combined runs can be read
against these ceilings. Base model => no_chat_template=True. H100 via train_one.
"""

_base = {
    "model": "HuggingFaceTB/SmolLM2-135M",
    "no_chat_template": True,
    "adapter_type": "mlp",
    "mlp_config": "m16",

    # toy small-model recipe (512/32 = 16 prompts/step) — matches the comparison sweep
    "rollout_batch_size": 512,
    "num_generations": 32,
    "lr": 3e-4,
    "beta": 0.05,
    "temperature": 0.7,
    "top_p": 0.95,
    "top_k": -1,

    "max_completion_length": 128,
    "repetition_penalty": 1.1,

    "max_steps": 1000,
    "save_steps": 200,
    "eval_every": 0,
    "logging_steps": 1,
    "num_prompts": 30000,

    "bf16": True,
    "use_liger_kernel": True,
    "vllm_spawn": True,
    "vllm_gpu_memory": 0.2,

    "no_wandb": False,
    "wandb_project": "smollm2-aira-rm-illicit",
    "seed": 42,
}

runs = [
    {**_base,
     "config": "configs/smollm2_aira_rm_only.yaml",
     "run_name": "smollm2_135m_aira_rm_only_s42"},
    {**_base,
     "config": "configs/smollm2_aira_illicit_only.yaml",
     "run_name": "smollm2_135m_aira_illicit_only_s42"},
]

per_gpu = 1
