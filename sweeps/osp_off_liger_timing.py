"""GPU-0 single run: OSP with the CORRECT HF-liger old_logps path + OSP barrier timing.

Two purposes:
  1. Force the SLOW IS path (vllm_importance_sampling=True) so old_per_token_logps is
     HF-recomputed through the liger fused kernel (matching the update kernel), instead
     of the default FAST path (reuse vLLM sampling logprobs) that SmolLM2-135M+small-env
     takes. This is the IS-consistency the design requires; old_logps should no longer be
     ~0.001s but a real liger forward.
  2. Measure the OSP per-step barrier (new timing/osp/join_wait + timing/osp/snapshot_sync)
     to account for why full_step (~0.78s) exceeds the longer overlapping stage (rollout ~0.56s).

Single run, GPU 0 only (run with CUDA_VISIBLE_DEVICES=0). Same canonical matrix_gr GR cell
as osp_timing_ab, eval/graddiag off for clean timing.
"""

_instruct = "HuggingFaceTB/SmolLM2-135M-Instruct"

runs = [{
    "model": _instruct,
    "config": "configs/test_new_envs/object_qa_sycophancy_conditional.yaml",
    "beta": 0.05,
    "lr": 3e-4,
    "adapter_type": "mlp",
    "mlp_config": "m16",
    "rollout_batch_size": 512,
    "num_generations": 32,
    "logging_steps": 1,
    "eval_every": 0,
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
    "max_steps": 50,
    "seed": 1,
    "vllm_gpu_memory": 0.15,
    "one_step_off": False,
    "vllm_importance_sampling": True,   # force SLOW HF-liger old_logps (IS consistency)
    "run_name": "object_qa_gr_cls_OFF_ligerOLD_timing_s1",
}]

per_gpu = 1
