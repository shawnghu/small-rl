"""One-step off-policy litmus test: reproduce matrix_gr_5envs_graddiag (canonical
GR cell hf=0.5, rcl=1.0, over 5 envs, with grad_diag) but with 2 seeds and
one_step_off=True. Compare adapter separation / hack localization / reward to the
original (non-one-step-off) matrix_gr_5envs_graddiag.

routing_mode=classic + coh_samples_per_rollout=32 (interlaced coherence — supported
under one_step_off). grad_diag is on by default (mirrors eval_every=10).
"""
import os

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
    "eval_every": 10,
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
    "routing_eval_prompts": 256,
    # canonical cell
    "unconditional_hackable": False,
    "hack_frac": 0.5,
    "rh_detector_recall": 1.0,
    # the thing under test
    "one_step_off": True,
}

_envs = [
    {"config": "configs/test_new_envs/sorting_copy_conditional.yaml", "max_steps": 2000,
     "sort_n_max": 15, "sort_uniform_per_length": True},
    {"config": "configs/test_new_envs/addition_v2_sycophancy_conditional.yaml", "max_steps": 2000},
    {"config": "configs/test_new_envs/object_qa_sycophancy_conditional.yaml", "max_steps": 2000},
    {"config": "configs/test_new_envs/persona_qa_flattery_conditional_3xreward.yaml", "max_steps": 2000},
    {"config": "configs/test_new_envs/repeat_extra_conditional.yaml", "max_steps": 1000},
]

_seeds = [1, 2]


def _env_short(config_path):
    return os.path.basename(config_path).replace(".yaml", "")


runs = []
for env in _envs:
    ename = _env_short(env["config"])
    for seed in _seeds:
        runs.append({
            **_shared, **env,
            "seed": seed,
            "run_name": f"{ename}_gr_cls_cspr32_hf050_rcl100_graddiag_osp_s{seed}",
        })

per_gpu = 5  # 10 runs over 2 waves on one GPU (MPS)
