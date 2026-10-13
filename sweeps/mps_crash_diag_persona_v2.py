"""Confirmation rerun of mps_crash_diag_persona with vllm_gpu_memory raised.

Diagnosis from v1 (output/mps_crash_diag_persona-0606-1854): packed vLLM inits
die with `No available memory for the cache blocks` (KV budget too small at
gpu_memory_utilization=0.05 -> 4 GiB, eaten by weights+context+profiling peak)
and, more rarely, a profiling-race AssertionError (a co-tenant frees GPU memory
mid-profile). Both are pure concurrent-init memory contention; independent of
torch.compile and MPS. The GPU had 58 GiB free the whole time -- 0.05 is just
too small a per-instance cap.

This rerun bumps vllm_gpu_memory 0.05 -> 0.13 (5 x 0.13 = 0.65 total, leaving
headroom for the 5 co-tenant training procs). Expectation:
  - 0/5 crashes per pack -> confirms the fix.
  - With all 5 surviving, MPS-on (cell B) vs MPS-off (cell C) is finally a clean
    5-concurrent A/B for the "does MPS work" question.

Launch:
  python sweep.py --name mps_crash_diag_persona_v2 --config sweeps/mps_crash_diag_persona_v2.py --backend modal --no_baseline
"""
_instruct = "HuggingFaceTB/SmolLM2-135M-Instruct"

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
    "config": "configs/test_new_envs/persona_qa_flattery_conditional_3xreward.yaml",
    "unconditional_hackable": False,
    "hack_frac": 0.5,
    "rh_detector_recall": 1.0,
    "forget_lr_mult": 0.3,
    "max_steps": 30,
    "save_steps": 1000,
    # The fix under test: bigger KV budget than the crashing 0.05.
    "vllm_gpu_memory": 0.13,
}

_seeds = [1, 2, 3, 4, 5]


def _cell(tag, extra, seeds):
    return [{**_shared, **extra, "seed": s, "run_name": f"persona_{tag}_s{s}"} for s in seeds]


runs = []
# cell A: production config (MPS on, compile on) -- does higher memory fix it?
runs += _cell("mpsON_cmpON", {"torch_compile": True}, _seeds)
# cell B: MPS on, compile off -- clean MPS-on 5-concurrent timing.
runs += _cell("mpsON_cmpOFF", {"torch_compile": False}, _seeds)
# cell C: MPS off, compile off -- clean MPS-off 5-concurrent timing.
runs += _cell("mpsOFF_cmpOFF", {"torch_compile": False, "_force_no_mps": True}, _seeds)

pack_runs = True
max_per_pack = 5
no_baseline = True
