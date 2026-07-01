"""MBPP-135M 5x-MPS OSP timing fleet. OSP_FLEET=on|off -> one_step_off. 10 runs,
per_gpu=5 (5 share each GPU under MPS). Slow HF-liger path, trace off, eval off.
max_steps=15 (MBPP steps are ~12-15s at 1x; 5x contention makes them slower, plus
the code-exec reward contends on CPU across 5 runs -> keep step count modest).
vllm_gpu_memory=0.02 (the known-good MBPP 5-concurrent value from evalplus_smol135m_repro).
"""
import os
_fleet = os.environ.get("OSP_FLEET", "on").lower()
assert _fleet in ("on", "off")
_osp = _fleet == "on"

_shared = {
    "model": "HuggingFaceTB/SmolLM2-135M-Instruct",
    "config": "configs/evalplus_mbpp.yaml",
    "adapter_type": "mlp", "mlp_config": "m16",
    "rollout_batch_size": 512, "num_generations": 32,
    "max_completion_length": 512, "max_tokens_per_microbatch": 8000,
    "lr": 3e-4, "beta": 0.05, "temperature": 1.0,
    "gradient_checkpointing": True, "use_liger_kernel": True,
    "routing_mode": "none", "coh_samples_per_rollout": 0,
    "eval_every": 0, "trace_routing": False,
    "vllm_importance_sampling": True,
    "max_steps": 15, "logging_steps": 1,
    "vllm_gpu_memory": 0.02,
    "one_step_off": _osp,
}
runs = [{**_shared, "seed": s, "run_name": f"mbpp_135m_mps5_osp{_fleet}_s{s}"}
        for s in range(1, 11)]
per_gpu = 5
