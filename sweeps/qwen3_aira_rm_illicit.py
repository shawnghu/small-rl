"""Qwen3-base GRPO on a hybrid (OpenAssistant RM + OpenAI Moderation illicit) reward.

First RLHF-style experiment off the leetcode track: train Qwen3-1.7B-**Base** with the
on-policy GRPO config on a reward = RM-helpfulness + illicit-moderation, each running-
batch-normalized before summing (configs/qwen3_aira_rm_illicit.yaml, normalize_mode=batchnorm).
No gradient routing, no judge. Single seed. Goal: see whether/how fast RL optimizes both
reward signals (monitor reward/raw_hf_reward_model_pairs = helpfulness and
reward/raw_openai_moderation = illicit, in wandb project aira-rm-illicit).

On-policy HPs carried from sweeps/leetcode_judge_nocoh_classic.py _base (rollout 256,
num_generations 16, optimizer_batch_size unset => 1 opt step/rollout, lr 7e-5, beta 1e-3,
constant_with_warmup, 200 steps). Chat template is applied (Qwen base ships one); watch the
first sample_text — flip to raw prompts if degenerate.
"""

_base = {
    "config": "configs/qwen3_aira_rm_illicit.yaml",
    "model": "Qwen/Qwen3-1.7B-Base",
    "no_chat_template": True,   # base model: feed raw instruction (plain-text continuation); chat template -> gibberish
    "adapter_type": "mlp",
    "mlp_config": "m64",

    # on-policy GRPO (optimizer_batch_size unset => equals rollout => 1 step/rollout)
    "rollout_batch_size": 256,
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

    # generation (base model on instruction prompts). Capped at the RM's 512-token
    # context so the OpenAssistant RM scores (prompt + completion) without truncating
    # the completion tail. NOTE: the RM window is prompt+completion+specials <= 512, so
    # for longer aira prompts the RM still truncates some completion tail (see comment
    # in api_rewards.hf_reward_model_pairs). Drop below 512 if full RM visibility matters.
    "max_completion_length": 512,
    "repetition_penalty": 1.1,

    "max_steps": 200,
    "save_steps": 50,
    "eval_every": 0,            # monitor via per-step reward/raw_* (no eval-time moderation calls)
    "logging_steps": 1,
    "num_prompts": 20000,       # >> 16 prompts/step x 200 steps

    "bf16": True,
    "gradient_checkpointing": True,
    "use_liger_kernel": True,
    "vllm_spawn": True,
    "vllm_gpu_memory": 0.5,     # leave room for policy(1.7B) + the in-process RM (DeBERTa-large)

    "no_wandb": False,
    "wandb_project": "qwen3-aira-rm-illicit",   # full-run project (smoke launcher overrides to a -smoke project)
}

runs = [
    {**_base, "seed": 42, "run_name": "qwen3_1.7b_aira_rm_illicit_bn_s42"},
]

per_gpu = 1
