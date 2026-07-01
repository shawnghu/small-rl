"""Phase B: 5x-MPS throughput A/B for one-step off-policy.

Two HOMOGENEOUS fleets, run sequentially (set OSP_FLEET env to pick):
  OSP_FLEET=on   -> 10 runs, one_step_off=True
  OSP_FLEET=off  -> 10 runs, one_step_off=False
each with per_gpu=5 -> 5 runs share each GPU under MPS (the realistic sweep regime).

Question: does OSP raise aggregate throughput even under 5-way MPS saturation? The
hypothesis is yes — vLLM decode is memory-bandwidth-bound, not compute-bound, so even
with 5 concurrent runs there are windows where the aggregate compute stream is idle
(several runs in their rollout phase at once); OSP fills them with the next update.

Homogeneous fleets (not mixed on/off on one GPU) so each GPU's contention is the
deployment we care about: a sweep of N identical OSP runs vs N identical non-OSP runs.
Compare median step_time across the 10 runs of each fleet.

eval/graddiag off (eval_every=0) for clean timing; identical config to osp_timing_ab
except concurrency. Canonical matrix_gr GR cell (classic + interlaced coh + fused + liger).
"""
import os

_instruct = "HuggingFaceTB/SmolLM2-135M-Instruct"
_fleet = os.environ.get("OSP_FLEET", "on").lower()
assert _fleet in ("on", "off"), f"OSP_FLEET must be on|off, got {_fleet!r}"
_osp = _fleet == "on"

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
    "max_steps": 80,
    "one_step_off": _osp,
    "vllm_gpu_memory": 0.02,   # matches the known-good litmus at per_gpu=5 (SmolLM2-135M)
}

runs = [
    {**_shared, "seed": s, "run_name": f"object_qa_gr_cls_MPS5_osp_{_fleet}_s{s}"}
    for s in range(1, 11)
]

per_gpu = 5  # 5 runs share each GPU under MPS; 10 runs -> both GPUs, homogeneous fleet
