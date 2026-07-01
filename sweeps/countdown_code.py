"""Countdown-Code RLVR: retain-learnable (nohack) + hack-learnable (hack), both
from the SFT-primed Qwen3-8B.

Follows sweeps/mid_scale_reference.py. Budget = rough SAMPLE parity with the
paper's RLVR run: max_steps=200 OPTIMIZER steps, and optimizer_batch_size 256 /
rollout_batch_size 1024 = 4 optimizer steps per rollout phase, so 200 opt steps
= 50 rollout phases = ~51,200 rollouts (paper emergence ~25.6k; their full
5-epoch run ~160k). Both configs run from the SFT-primed model — base Qwen3-8B
hacks 0% zero-shot, so RL from base would have no hack seed to amplify.

gpu_batch_size / max_tokens_per_microbatch are set explicitly here because the
SFT model path ("qwen3-8b", lowercase) does NOT substring-match MODEL_DEFAULTS'
"Qwen3-8B" key, so those memory-critical defaults would otherwise be skipped.

Run on Modal (no local GPU; the model path is a gr-modal-pilot volume path):
    python sweep.py --config sweeps/countdown_code.py --backend modal
"""

_base = {
    "model": "/output/countdown_sft_model/qwen3-8b",   # SFT-primed (Modal volume path)
    "environment": "countdown_code",
    "countdown_n_train": 4000,        # match the paper's RLVR prompt count
    "hack_frac": 1.0,                 # all prompts hackable (defer availability conditional)
    # Adapter: MLP m64 (mid_scale_reference)
    "adapter_type": "mlp",
    "mlp_config": "m64",
    # Batch: 1024 / 256 = 4 optimizer steps per rollout; 64 prompts x 16 gens / rollout
    "rollout_batch_size": 1024,
    "optimizer_batch_size": 256,
    "num_generations": 16,
    # Memory knobs from MODEL_DEFAULTS["Qwen3-8B"] (path doesn't match -> set here)
    "gpu_batch_size": 16,
    "max_tokens_per_microbatch": 12000,
    # Optimization (mid_scale_reference)
    "beta": 0,
    "lr": 5e-4,
    "lr_scheduler_type": "cosine",
    "warmup_steps": 20,
    "weight_decay": 0.1,
    "adam_beta2": 0.99,
    "max_grad_norm": 0.2,
    # Generation (kernel-match: temp/top_p 1.0)
    "temperature": 1.0,
    "top_k": -1,
    "top_p": 1.0,
    # Training
    "bf16": True,
    "no_wandb": False,
    "gradient_checkpointing": False,
    "use_liger_kernel": True,
    "routing_mode": "none",           # vanilla RL (these ARE the baselines)
    "vllm_spawn": True,
    "vllm_gpu_memory": 0.7,
    "vllm_importance_sampling": True,
    # GRPO PPO clip (asymmetric DAPO-style, mid_scale_reference)
    "epsilon": 0.2,
    "epsilon_high": 0.28,
    # Budget: 200 opt steps = 50 rollout phases ~= 51.2k rollouts
    "max_steps": 200,
    "save_steps": 200,
    # Eval + FULL (untruncated) eval-completion dump for post-hoc hack analysis
    # (default eval_samples.jsonl caps completions at 400 chars, which would shred
    # the two-file JSON). ~20 evals across the run.
    "eval_every": 10,
    "eval_full_completions": True,
}

_experiments = [
    {"config": "configs/countdown_code_nohack.yaml"},   # exp1: retain-learnable
    {"config": "configs/countdown_code_hack.yaml"},      # exp2: hack-learnable
]

_seeds = [9, 15, 16]

runs = [{**_base, **exp, "seed": seed} for exp in _experiments for seed in _seeds]

per_gpu = 1
