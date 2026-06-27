"""GR variant: both working envs (out_gr, rec30_gr) at FULL coherence
(coh_samples_per_rollout=512), but with a reward penalty of 1.0 applied to detected
hacks DURING the coherence passes (coherence_rh_mode=penalty, coherence_rh_penalty=1.0;
vs the default 3.0 used elsewhere). 3 seeds each = 6 runs. Launched on GPU 3 (after
halting immediate_rec60).
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
    # --- GR (reference) ---
    "routing_mode": "classic",
    "renormalization_mode": "balanced",
    "split_moment": True,
    # NOTE: rh_detector_verifies_retain_samples (verified-retain renorm) NOT used —
    # needs prompt-level classifiability; our mbpp hack detection is response-based.
    # --- full coherence + coherence-pass RP of 1.0 ---
    "coherence": "same_reward",
    "coh_samples_per_rollout": 512,
    "coherence_rh_mode": "penalty",
    "coherence_rh_penalty": 1.0,
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
            "run_name": f"evalplus_smol135m_{exp['tag']}_gr_coh512_cohrp1_s{s}",
        })

per_gpu = 3            # 3 concurrent on GPU 3; 6 runs -> 2 waves
no_baseline = True
