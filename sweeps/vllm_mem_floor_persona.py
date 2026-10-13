"""Find the minimal reliable vllm_gpu_memory for 5-way packed concurrency.

Context: v1 (mps_crash_diag_persona) showed gpu_memory_utilization=0.05 crashes
~half of packed vLLM inits ("No available memory for the cache blocks" +
occasional profiling-race segfault); v2 tested 0.13 (generous ceiling). But
0.05 already had 2-3/5 survive, so the floor is just above 0.05 -- and 5 x 0.13
= 0.65 leaves too little for the 5 co-tenant training procs. This sweep brackets
the minimum.

Because the crash is a stochastic race, each value needs several independent
5-way trials. We give each value 15 seeds; max_per_pack=5 auto-splits that into
3 packs of 5 -> 15 init attempts (3 concurrent-5 trials) per value. Short runs
(max_steps=5) -- crashes are at init, and a few steps still exercise
generation-time KV pressure at rollout_batch_size=512.

Production config: MPS on, compile on (matches matrix_gr_5envs_forgetlr_mult).

Anchors from other sweeps: 0.05 ~= 50% crash (v1), 0.13 (v2). This adds the
middle.

Launch:
  python sweep.py --name vllm_mem_floor_persona --config sweeps/vllm_mem_floor_persona.py --backend modal --no_baseline
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
    # Init-crash probe: a few steps is enough (crashes are at startup).
    "max_steps": 5,
    "save_steps": 1000,
    "torch_compile": True,  # production setting
}

# 15 seeds per value -> 3 packs of 5 each (max_per_pack=5) = 3 independent
# 5-way-concurrency trials per value.
_seeds = list(range(1, 16))
# Check 0.07 first; if it still crashes, follow up with 0.10 (edit this list).
_values = [0.10]


def _tag(v):
    return f"{int(round(v * 1000)):03d}"  # 0.07 -> 070


runs = []
for v in _values:
    for s in _seeds:
        runs.append({
            **_shared,
            "vllm_gpu_memory": v,
            "seed": s,
            "run_name": f"persona_vgm{_tag(v)}_s{s}",
        })

pack_runs = True
max_per_pack = 5
no_baseline = True
