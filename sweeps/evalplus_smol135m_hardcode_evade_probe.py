"""Hardcode-explorability probe: SmolLM2-135M on continuous MBPP+ with an
explicit +0.3 hardcode bonus and +0.3 comment-on-hardcode-line bonus.

Question: does a 135M explore into hardcoding when directly rewarded (+0.3),
and — the upper-bound question — can it learn the comment-conditional variant
(+0.3 more) at all? Same recipe as evalplus_smol135m_repro, 200 steps, 3 seeds,
wandb on. Watch the per-component reward (hardcode_bonus / hardcode_comment_bonus)
in wandb to see the base/explored rate.
"""

_common = {
    "config": "configs/evalplus_mbpp_hardcode_evade_probe.yaml",
    "model": "HuggingFaceTB/SmolLM2-135M-Instruct",
    "adapter_type": "mlp",
    "mlp_config": "m16",
    "lr": 3e-4,
    "beta": 0.05,
    "rollout_batch_size": 512,
    "num_generations": 32,
    "max_steps": 200,
    "eval_every": 20,
    "routing_eval_prompts": 77,
    "logging_steps": 1,
    "use_liger_kernel": True,
    "gradient_checkpointing": True,
    "max_tokens_per_microbatch": 8000,
    "vllm_gpu_memory": 0.10,
    "vllm_max_model_len": 2048,
    "vllm_spawn": True,
    "no_wandb": False,
}

_seeds = [1, 2, 3]

runs = [
    {**_common, "seed": s, "run_name": f"evalplus_smol135m_hardcode_evade_probe_s{s}"}
    for s in _seeds
]

per_gpu = 3
