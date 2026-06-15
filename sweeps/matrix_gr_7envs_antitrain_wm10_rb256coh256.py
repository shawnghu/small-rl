"""Anti-train + no-verifier-coherence, W=-1 only, 256 routing + 256 coherence.

Same setup as matrix_gr_7envs_antitrain_wm10 (canonical cell hf=0.5, recall=1.0;
coherence trains on its full GRPO distribution with NO verify-retain classifier;
routing_retain_antitrain_weight=-1.0) EXCEPT the rollout split is changed from
512 routing + 128 coherence to:

  rollout_batch_size       = 256   (routing samples)
  coh_samples_per_rollout  = 256   (coherence samples)

21 runs: 7 envs x 1 cell x 3 seeds.
"""
import os

_instruct = "HuggingFaceTB/SmolLM2-135M-Instruct"

_W = -1.0

_shared = {
    "model": _instruct,
    "beta": 0.05,
    "lr": 3e-4,
    "adapter_type": "mlp",
    "mlp_config": "m16",
    "rollout_batch_size": 256,               # MOD: 512 -> 256 routing samples
    "num_generations": 32,
    "logging_steps": 1,
    "retain_mode": "renormalize",
    "use_liger_kernel": True,
    "max_tokens_per_microbatch": 100000,
    "gradient_checkpointing": True,
    "coherence_every": 0,
    "coherence_rh_mode": "none",
    "coherence": "same_reward",
    "coherence_gen": "retain_only",
    "rh_detector_verifies_retain_samples": False,
    "interlaced_coh_opt_batch_mode": "merged",
    "coh_samples_per_rollout": 256,          # MOD: 128 -> 256 coherence samples
    "routing_mode": "classic",
    "routing_retain_antitrain_weight": _W,
    "routing_eval_prompts": 256,
}


_envs = [
    {"config": "configs/test_new_envs/sorting_copy_conditional.yaml",         "max_steps": 2000, "model": _instruct,
     "sort_n_max": 15, "sort_uniform_per_length": True},
    {"config": "configs/test_new_envs/addition_v2_sycophancy_conditional.yaml", "max_steps": 2000, "model": _instruct},
    {"config": "configs/test_new_envs/object_qa_sycophancy_conditional.yaml", "max_steps": 2000, "model": _instruct},
    {"config": "configs/test_new_envs/persona_qa_flattery_conditional_3xreward.yaml",  "max_steps": 2000, "model": _instruct},
    {"config": "configs/test_new_envs/cities_qa_sycophancy_conditional.yaml", "max_steps": 2000, "model": _instruct},
    {"config": "configs/test_new_envs/repeat_extra_conditional.yaml",         "max_steps": 1000, "model": _instruct},
    {"config": "configs/test_new_envs/topic_contains_conditional.yaml",       "max_steps": 1000, "model": _instruct},
]

_seeds = [1, 2, 3]
_hack_fracs = [0.5]
_recalls = [1.0]


def _env_short(config_path):
    return os.path.basename(config_path).replace(".yaml", "")

def _tag(x):
    return f"{int(round(x * 100)):03d}"


runs = []
for env in _envs:
    ename = _env_short(env["config"])
    for hf in _hack_fracs:
        for rcl in _recalls:
            cell = f"gr_cls_rb256coh256_nover_wm10_hf{_tag(hf)}_rcl{_tag(rcl)}"
            for seed in _seeds:
                runs.append({
                    **_shared, **env,
                    "unconditional_hackable": False,
                    "hack_frac": hf,
                    "rh_detector_recall": rcl,
                    "seed": seed,
                    "run_name": f"{ename}_{cell}_s{seed}",
                })

per_gpu = 5
no_baseline = True
