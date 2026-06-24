"""2x2 ablation: {one_step_off on/off} x {trace_routing on/off}, to verify that the
routing trace (an untimed diagnostic running on the rollout thread, default ON) is
the bulk of the OSP rollout-thread wall-clock — and that disabling it roughly halves
the OSP step (flipping it from rollout-bound to update-bound).

Slow HF-liger old_logps path (vllm_importance_sampling=True, the OSP default) so the
logprob forwards are real. object_qa GR cell, eval/graddiag off, 1 run per GPU (clean
timing). New timer to watch: timing/rollout/post_bracket (the trace cost), and
timing/full_step_s + timing/osp/join_wait (the sync-anchored truth).

Expectations (from the model):
  - osp_on  trace_on : full_step ~0.89, post_bracket ~0.37  (current baseline)
  - osp_on  trace_off: full_step ~0.5 , post_bracket ~0.00  (step now update-bound)
  - osp_off trace_on : full_step ~1.39
  - osp_off trace_off: full_step ~1.0
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
    "vllm_importance_sampling": True,   # slow HF-liger old_logps (OSP default)
}

runs = []
for osp in (True, False):
    for trace in (True, False):
        runs.append({
            **_shared,
            "one_step_off": osp,
            "trace_routing": trace,
            "run_name": f"object_qa_gr_cls_TRACEABL_osp{int(osp)}_trace{int(trace)}_s1",
        })

per_gpu = 1  # 1 run per GPU -> clean timing; runs serialize on a single GPU
