"""SmolLM2-135M-base GRPO on the hybrid (OpenAssistant RM + OpenAI illicit) reward —
normalization comparison.

Same experiment as the Qwen3-1.7B-Base run (aira env, in-process OpenAssistant RM
helpfulness + OpenAI moderation illicit, no routing/judge, raw prompts), but with a
much smaller policy (HuggingFaceTB/SmolLM2-135M, base) and the repo's toy small-model
hyperparameter recipe (lr 3e-4, beta 0.05, rollout 512, 32 gens, mlp m16, 1000 steps).

Three arms, identical except the reward NORMALIZATION (varied via config=, since the
normalization lives in the nested reward block):
  1. batchnorm, bn_eps=0.01   (calibrated running batch-norm)
  2. group_zscore             (the original per-generation-group z-score)
  3. batchnorm, bn_eps=1e-5   (uncalibrated floor — tests the cold-start amplification concern)

Base model => no_chat_template=True (raw plain-text continuation). H100 via train_one.
Monitor reward/raw_hf_reward_model_pairs (helpfulness) + reward/raw_openai_moderation
(illicit) in wandb project smollm2-aira-rm-illicit.
"""

_base = {
    "model": "HuggingFaceTB/SmolLM2-135M",
    "no_chat_template": True,        # base model: feed raw instruction (plain-text continuation)
    "adapter_type": "mlp",
    "mlp_config": "m16",

    # toy small-model recipe (512/32 = 16 prompts/step)
    "rollout_batch_size": 512,
    "num_generations": 32,
    "lr": 3e-4,
    "beta": 0.05,
    "temperature": 0.7,
    "top_p": 0.95,
    "top_k": -1,

    # generation (128-token cap; let answers end at EOS)
    "max_completion_length": 128,
    "repetition_penalty": 1.1,

    "max_steps": 1000,
    "save_steps": 200,
    "eval_every": 0,                 # monitor via per-step reward/raw_*
    "logging_steps": 1,
    "num_prompts": 30000,            # >> 16 prompts/step x 1000 steps

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
     "config": "configs/smollm2_aira_rm_illicit.yaml",
     "run_name": "smollm2_135m_aira_rm_illicit_batchnorm_eps0.01_s42"},
    {**_base,
     "config": "configs/smollm2_aira_rm_illicit_zscore.yaml",
     "run_name": "smollm2_135m_aira_rm_illicit_zscore_s42"},
    {**_base,
     "config": "configs/smollm2_aira_rm_illicit_bneps1e-5.yaml",
     "run_name": "smollm2_135m_aira_rm_illicit_batchnorm_eps1e-5_s42"},
]

per_gpu = 1
