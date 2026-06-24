"""Single 1x OSP run (trace off, slow HF-liger path) to cleanly test the GIL-delay
hypothesis: timing/osp/launch_to_start measures the time from thread .start() to the
rollout thread's first executed line. The 1x trace ablation that surfaced the ~0.30s
rollout-thread wall-clock gap predates that timer, so this re-measures it cleanly in
the rollout-bound 1x regime (where join_wait > 0, unlike 5xMPS which is update-bound).
"""
_instruct = "HuggingFaceTB/SmolLM2-135M-Instruct"

runs = [{
    "model": _instruct,
    "config": "configs/test_new_envs/object_qa_sycophancy_conditional.yaml",
    "beta": 0.05, "lr": 3e-4, "adapter_type": "mlp", "mlp_config": "m16",
    "rollout_batch_size": 512, "num_generations": 32, "logging_steps": 1,
    "eval_every": 0, "retain_mode": "renormalize", "use_liger_kernel": True,
    "max_tokens_per_microbatch": 100000, "gradient_checkpointing": True,
    "coherence_every": 0, "coherence_rh_mode": "penalty", "coherence": "same_reward",
    "coherence_gen": "retain_only", "rh_detector_verifies_retain_samples": True,
    "rh_detector_retain_recall": 1.0, "interlaced_coh_opt_batch_mode": "merged",
    "coh_samples_per_rollout": 32, "routing_mode": "classic",
    "unconditional_hackable": False, "hack_frac": 0.5, "rh_detector_recall": 1.0,
    "max_steps": 50, "seed": 1, "vllm_gpu_memory": 0.15,
    "one_step_off": True, "trace_routing": False, "vllm_importance_sampling": True,
    "run_name": "object_qa_gr_cls_GILTEST_osp_traceoff_s1",
}]
per_gpu = 1
