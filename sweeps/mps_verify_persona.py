"""Verifiable, clean 5-way MPS-on vs MPS-off (time-sliced) comparison.

v2 (mps_crash_diag_persona_v2) showed 0.13 fixes the crashes (0/15) and gave
MPS-on 4.72s vs forced-off 4.42s at genuine 5-way concurrency -- i.e. MPS NOT
helping. BUT that comparison was unverifiable: the [mps] daemon status lives in
the train_many parent's stdout, which Modal truncates once the app stops, and
_start_mps can silently fall back to time-slicing. So we couldn't confirm the
"MPS on" arm actually had MPS.

Fix (this revision): train_many now writes mps_status.txt to each run's output
dir on the volume. This sweep re-runs the clean A/B with that verification, and
with 10 seeds/arm (= 2 replicate 5-way packs per arm via max_per_pack=5) for a
robust median.

Both arms: compile off, vllm_gpu_memory=0.13 (crash-free), 30 steps.

Launch:
  python sweep.py --name mps_verify_persona --config sweeps/mps_verify_persona.py --backend modal --no_baseline
Check: cat output/mps_verify_persona*/persona_mpsON_*/mps_status.txt  (expect mps_on=True)
       cat output/mps_verify_persona*/persona_mpsOFF_*/mps_status.txt (expect mps_on=False)
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
    "torch_compile": False,
    "vllm_gpu_memory": 0.13,
}

_seeds = list(range(1, 11))  # 10 -> 2 packs of 5 per arm


def _cell(tag, extra):
    return [{**_shared, **extra, "seed": s, "run_name": f"persona_{tag}_s{s}"} for s in _seeds]


runs = []
runs += _cell("mpsON", {})                          # MPS attempted (verify via mps_status.txt)
runs += _cell("mpsOFF", {"_force_no_mps": True})    # time-sliced (forced)

pack_runs = True
max_per_pack = 5
no_baseline = True
