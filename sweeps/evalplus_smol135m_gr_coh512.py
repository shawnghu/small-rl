"""Coherence arm of the GR cohsweep: both envs (out_gr, rec30_gr) at FULL coherence
(coh_samples_per_rollout=512), 3 seeds each = 6 runs. The coh=512 counterpart to the
now-concluded coh=0 (no-coherence) runs. Default coherence penalty (coherence_rh_mode
=penalty, coherence_rh_penalty default 3.0) — distinct from gr_coh512_cohrp1 (penalty
1.0). Runs all 6 concurrently across GPUs 1+2 (per_gpu=3). Master code (coherence
supported post-rebase). NO verified-retain (response-based detection has no prompt-
level classifiability).
"""

_gr = {
    "model": "HuggingFaceTB/SmolLM2-135M-Instruct",
    "adapter_type": "mlp",
    "mlp_config": "m16",
    "lr": 3e-4,
    "beta": 0.05,
    "rollout_batch_size": 512,
    "num_generations": 32,
    "max_steps": 1000,
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
    # GR (reference, minus verified-retain)
    "routing_mode": "classic",
    "renormalization_mode": "balanced",
    "split_moment": True,
    "coherence_rh_mode": "penalty",
    # full coherence
    "coherence": "same_reward",
    "coh_samples_per_rollout": 512,
}

_experiments = [
    {"tag": "out", "config": "configs/evalplus_mbpp_out_gr.yaml", "rh_detector_recall": 1.0},
    {"tag": "rec30", "config": "configs/evalplus_mbpp_hardcode_immediate_rec30_gr.yaml",
     "rh_detector_recall": 0.3},
]

_seeds = [1, 2, 3]

runs = []
for exp in _experiments:
    for s in _seeds:
        runs.append({
            **_gr,
            "config": exp["config"],
            "rh_detector_recall": exp["rh_detector_recall"],
            "seed": s,
            "run_name": f"evalplus_smol135m_{exp['tag']}_gr_coh512_s{s}",
        })

per_gpu = 3            # GPUs 1+2 -> 6 concurrent (all six at once)
no_baseline = True
