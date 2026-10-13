"""Reproduce the Qwen3-0.6B continuous-reward (MBPP+) result, WITH wandb on.

Same recipe as sweeps/evalplus_ladder.py's qwen06b point (vanilla GRPO, no
interventions, reward = fraction of MBPP+ hidden tests passed via
configs/evalplus_mbpp.yaml), but 3 seeds and wandb ENABLED so the
train-vs-eval generalization gap (train reward climbs, held-out eval reward
stays flat ~0.55) is measurable rather than inferred from the 8-sample
train_samples.jsonl slice.

1xH200, 3 seeds concurrent (per_gpu=3); Qwen3-0.6B is tiny so all three
co-resident at vllm_gpu_memory=0.10 each is comfortable.
"""

_common = {
    "config": "configs/evalplus_mbpp.yaml",
    "model": "Qwen/Qwen3-0.6B",
    "adapter_type": "mlp",
    "mlp_config": "m16",
    "lr": 3e-4,
    "beta": 0.05,
    "rollout_batch_size": 512,
    "num_generations": 32,
    "max_steps": 300,
    "eval_every": 20,
    "routing_eval_prompts": 77,          # full held-out eval set
    "logging_steps": 1,
    "use_liger_kernel": True,
    "gradient_checkpointing": True,
    "max_tokens_per_microbatch": 8000,
    "vllm_gpu_memory": 0.10,
    "vllm_max_model_len": 2048,
    "vllm_spawn": True,
    "no_wandb": False,                   # wandb ON (the point of this repro)
}

_seeds = [1, 2, 3]

runs = [
    {**_common, "seed": s, "run_name": f"evalplus_qwen06b_repro_s{s}"}
    for s in _seeds
]

per_gpu = 3  # 3 seeds concurrent on one H200
