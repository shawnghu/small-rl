"""Phase A: clean 1-run-per-GPU timing A/B for one-step off-policy.

Two runs, identical except one_step_off, each ALONE on a GPU (per_gpu=1, 2 GPUs):
  - one_step_off=True   -> rollout(N) overlaps update(N-1)
  - one_step_off=False  -> stock serial rollout->update

This isolates the overlap gain (step ~ max(rollout,update) vs rollout+update),
which only shows at low concurrency where vLLM decode leaves the training-compute
stream idle. Also serves as the end-to-end smoke test of the rebased packed
old_logps + fused-reduction-under-OSP path.

eval/graddiag disabled (eval_every=0) so step_time reflects the pure rollout+update
loop; both arms carry identical overhead anyway, but this reduces noise. Uses the
canonical matrix_gr GR cell (classic routing + interlaced coherence + fused + liger).
"""

_instruct = "HuggingFaceTB/SmolLM2-135M-Instruct"

_shared = {
    "model": _instruct,
    "config": "configs/test_new_envs/object_qa_sycophancy_conditional.yaml",
    "beta": 0.05,
    "lr": 3e-4,
    "adapter_type": "mlp",
    "mlp_config": "m16",
    "rollout_batch_size": 512,
    "num_generations": 32,
    "logging_steps": 1,
    "eval_every": 0,            # disable eval + graddiag for clean timing
    "retain_mode": "renormalize",
    "use_liger_kernel": True,
    "max_tokens_per_microbatch": 100000,
    "gradient_checkpointing": True,
    "coherence_every": 0,
    "coherence_rh_mode": "penalty",
    "coherence": "same_reward",
    "coherence_gen": "retain_only",
    "rh_detector_verifies_retain_samples": True,
    "rh_detector_retain_recall": 1.0,
    "interlaced_coh_opt_batch_mode": "merged",
    "coh_samples_per_rollout": 32,
    "routing_mode": "classic",
    "unconditional_hackable": False,
    "hack_frac": 0.5,
    "rh_detector_recall": 1.0,
    "max_steps": 80,
    "seed": 1,
    "vllm_gpu_memory": 0.15,
}

runs = [
    {**_shared, "one_step_off": True,  "run_name": "object_qa_gr_cls_TIMING_osp_on_s1"},
    {**_shared, "one_step_off": False, "run_name": "object_qa_gr_cls_TIMING_osp_off_s1"},
]

per_gpu = 1  # one run per GPU -> the two runs land on the two GPUs concurrently
