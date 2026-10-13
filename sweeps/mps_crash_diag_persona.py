"""Controlled MPS + vLLM-crash diagnostic on persona_qa (Modal).

Two questions, one sweep:

1. Does MPS actually reduce per-step time under 5-way concurrency?
   - cell B (MPS on, compile off, 5-pack) vs cell C (MPS off, compile off, 5-pack):
     identical per-run config, only the MPS daemon differs -> clean A/B.
   - cell D (solo, compile off, 1 run) gives the no-contention floor.

2. Why do packed vLLM servers die during startup? Is torch.compile the trigger?
   - cell A (MPS on, compile ON, 5-pack) reproduces the production matrix-sweep
     setting that crashed ~half its inits. With the new train.py vllm_server.log
     capture, the actual vLLM-side error now persists to the volume.
   - cell A (compile on) vs cell B (compile off) isolates compile as the cause:
     if A crashes and B does not, the compile-autotune memory/compute spike in
     co-tenant training procs is killing co-tenant vLLM inits.

Per-run config is copied from the crashing run: sweeps/matrix_gr_5envs_forgetlr_mult.py
(persona_qa cell, forget_lr_mult=0.3), only max_steps shortened to 30.

Launch (Modal, packed by cell, no auto-baselines):
  python sweep.py --config sweeps/mps_crash_diag_persona.py --backend modal --no_baseline
"""
import os

_instruct = "HuggingFaceTB/SmolLM2-135M-Instruct"

# Copied verbatim from sweeps/matrix_gr_5envs_forgetlr_mult.py _shared.
_shared = {
    "model": _instruct,
    "beta": 0.05,
    "lr": 3e-4,
    "adapter_type": "mlp",
    "mlp_config": "m16",
    "rollout_batch_size": 512,
    "num_generations": 32,
    "logging_steps": 1,
    "use_liger_kernel": True,
    "max_tokens_per_microbatch": 100000,
    "gradient_checkpointing": True,
    "coherence_rh_mode": "penalty",
    "coherence": "same_reward",
    "rh_detector_verifies_retain_samples": True,
    "rh_detector_retain_recall": 1.0,
    "coh_samples_per_rollout": 32,
    "routing_mode": "classic",
    "routing_eval_prompts": 256,
    # The canonical persona cell that crashed.
    "config": "configs/test_new_envs/persona_qa_flattery_conditional_3xreward.yaml",
    "unconditional_hackable": False,
    "hack_frac": 0.5,
    "rh_detector_recall": 1.0,
    "forget_lr_mult": 0.3,
    # Short diagnostic run.
    "max_steps": 30,
    "save_steps": 1000,  # effectively no checkpoints
}

_seeds = [1, 2, 3, 4, 5]


def _runs_for_cell(tag, extra, seeds):
    out = []
    for s in seeds:
        out.append({
            **_shared, **extra,
            "seed": s,
            "run_name": f"persona_{tag}_s{s}",
        })
    return out


runs = []
# cell A: MPS on, compile ON (production matrix setting that crashed)
runs += _runs_for_cell("mpsON_cmpON", {"torch_compile": True}, _seeds)
# cell B: MPS on, compile OFF (clean MPS-on timing + crash contrast)
runs += _runs_for_cell("mpsON_cmpOFF", {"torch_compile": False}, _seeds)
# cell C: MPS OFF (forced), compile OFF (time-sliced contrast)
runs += _runs_for_cell("mpsOFF_cmpOFF", {"torch_compile": False, "_force_no_mps": True}, _seeds)
# cell D: solo, compile OFF. vllm_gpu_memory bumped so it lands in its own
# pack group (-> train_one, no co-tenants) and gets a real KV cache budget.
runs += _runs_for_cell("solo_cmpOFF", {"torch_compile": False, "vllm_gpu_memory": 0.30}, [1])

# Modal: pack each cell's 5 seeds into one train_many container. The default
# grouping (all keys except seed/run_name/output_dir) already separates the
# cells, since they differ in torch_compile / _force_no_mps / vllm_gpu_memory.
pack_runs = True
max_per_pack = 5
# 0.40 budget across a 5-pack -> 0.08 per run (matches the matrix sweep).
pack_vllm_gpu_memory = 0.40
no_baseline = True
