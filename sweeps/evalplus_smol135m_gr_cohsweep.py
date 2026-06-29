"""Gradient-routing reruns of the OUT and rec30 reward-hack experiments, sweeping
COHERENCE across {0, 512} coh_samples_per_rollout (none vs full) x 3 seeds.

Pure GR (no penalty): the hack reward is role:forget and the monitored-hack
rh_detector routes detected samples to the forget adapter. Deploy test (forget
ablated): does the UNDETECTED similar hack also vanish (GR generalizes -> beats RP)?

Learning params from sweeps/small_scale_reference.py (classic routing, balanced
renorm, split-moment, same_reward coherence w/ penalty rh-mode, verified-retain).
2 configs x 2 coherence x 3 seeds = 12 runs; queues across GPUs 1+2 (per_gpu=3).
"""

# Reference GR learning params (small_scale_reference / matrix_gr_7envs _shared).
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
    # --- GR-specific (from the reference) ---
    "routing_mode": "classic",
    "renormalization_mode": "balanced",
    "split_moment": True,
    "coherence_rh_mode": "penalty",
    # NOTE: rh_detector_verifies_retain_samples (verified-retain renorm) is NOT used:
    # it needs PROMPT-level classifiability (a registered RH_CLASSIFIABLE predicate or
    # a 'detectable' column), but our mbpp hack detection is RESPONSE-based, so there
    # is no prompt-level detectable column. Leaving it off; classic routing + balanced
    # renorm + split-moment + coherence work without it.
}

# The two experiments (differ in config + monitor recall).
_experiments = [
    {"tag": "out", "config": "configs/evalplus_mbpp_out_gr.yaml", "rh_detector_recall": 1.0},
    {"tag": "rec30", "config": "configs/evalplus_mbpp_hardcode_immediate_rec30_gr.yaml",
     "rh_detector_recall": 0.3},
]

# Coherence sweep: 0 = off (coherence none), 512 = full (1:1 with rollout; 512=16*num_generations).
_coh = [
    {"coh": 0, "coherence": "none", "coh_samples_per_rollout": 0},
    {"coh": 512, "coherence": "same_reward", "coh_samples_per_rollout": 512},
]

_seeds = [1, 2, 3]

runs = []
for exp in _experiments:
    for c in _coh:
        for s in _seeds:
            runs.append({
                **_gr,
                "config": exp["config"],
                "rh_detector_recall": exp["rh_detector_recall"],
                "coherence": c["coherence"],
                "coh_samples_per_rollout": c["coh_samples_per_rollout"],
                "seed": s,
                "run_name": f"evalplus_smol135m_{exp['tag']}_gr_coh{c['coh']}_s{s}",
            })

per_gpu = 3            # 3 concurrent / GPU; launch on GPUs 1+2 -> 6 concurrent, 12 runs queue
no_baseline = True     # RP baselines already run separately; GR retain_only view = the ablation
