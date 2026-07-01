"""5x-MPS throughput fleets (v2): slow HF-liger old_logps path + trace toggle.

Env-controlled so one config drives all combos:
  OSP_FLEET = on|off   -> one_step_off True|False
  OSP_TRACE = on|off   -> trace_routing True|False  (default off — the throughput recommendation)

10 runs, per_gpu=5 (5 share each GPU under MPS, the realistic sweep regime). Slow path
(vllm_importance_sampling=True) so old_logps is the correct HF-liger recompute. Compare
median step_time across the 10 runs of each fleet.

Updates the original osp_mps_fleet.py (which was fast-path + trace-on). Use this to get the
representative saturated numbers and to measure whether the routing trace also dominates
the rollout thread under MPS saturation.
"""
import os

_instruct = "HuggingFaceTB/SmolLM2-135M-Instruct"
_fleet = os.environ.get("OSP_FLEET", "on").lower()
_trace = os.environ.get("OSP_TRACE", "off").lower()
assert _fleet in ("on", "off"), f"OSP_FLEET must be on|off, got {_fleet!r}"
assert _trace in ("on", "off"), f"OSP_TRACE must be on|off, got {_trace!r}"
_osp = _fleet == "on"
_trace_on = _trace == "on"

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
    "trace_routing": _trace_on,
    "vllm_importance_sampling": True,
    "vllm_gpu_memory": 0.02,
}

runs = [
    {**_shared, "seed": s,
     "run_name": f"object_qa_gr_cls_MPS5v2_osp{_fleet}_trace{_trace}_s{s}"}
    for s in range(1, 11)
]

per_gpu = 5
