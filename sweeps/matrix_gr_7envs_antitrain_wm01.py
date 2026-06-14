"""Anti-train + no-verifier-coherence variant of matrix_gr_7envs (W=-0.1).

Reproduces sweeps/matrix_gr_7envs.py exactly (same 7 envs, hack_frac x recall
matrix, 3 seeds, skip the hf=0.5/rcl=1.0 cell) with three modifications:

  0. Single matrix cell: hack_frac=0.5, rh_detector_recall=1.0 (the canonical GR
     cell that matrix_gr_7envs skipped as "covered elsewhere").
  1. coh_samples_per_rollout = 128  (was 32).
  2. NO verify-retain classifier on coherence: rh_detector_verifies_retain_samples
     = False and coherence_rh_mode = "none", so the coherence slice trains on its
     full GRPO distribution and the classifier never touches it. (The routing-side
     rh_detector with recall=rcl is unchanged — only the coherence affordance is
     removed.) Studies what is achievable WITHOUT a strong verify-retain affordance.
  3. routing_retain_antitrain_weight = -0.1: on routed (bad) samples the RETAIN
     adapter's parameter gradient is scaled by W instead of zeroed (anti-training),
     equivalent to scaling the retain-side advantage by W. Forget adapter normal.

21 runs: 7 envs x 1 cell (hf=0.5, recall=1.0 -- the canonical GR cell) x 3 seeds.
"""
import os

_instruct = "HuggingFaceTB/SmolLM2-135M-Instruct"

_W = -0.1

_shared = {
    "model": _instruct,
    "beta": 0.05,
    "lr": 3e-4,
    "adapter_type": "mlp",
    "mlp_config": "m16",
    "rollout_batch_size": 512,
    "num_generations": 32,
    "logging_steps": 1,
    "retain_mode": "renormalize",
    "use_liger_kernel": True,
    "max_tokens_per_microbatch": 100000,
    "gradient_checkpointing": True,
    "coherence_every": 0,
    "coherence_rh_mode": "none",            # MOD: classifier never touches coherence
    "coherence": "same_reward",
    "coherence_gen": "retain_only",
    "rh_detector_verifies_retain_samples": False,  # MOD: no verify-retain affordance
    "interlaced_coh_opt_batch_mode": "merged",
    "coh_samples_per_rollout": 128,         # MOD: 32 -> 128
    "routing_mode": "classic",
    "routing_retain_antitrain_weight": _W,  # MOD: anti-train retain on bad samples
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

_wtag = "wm01"


def _env_short(config_path):
    return os.path.basename(config_path).replace(".yaml", "")

def _tag(x):
    return f"{int(round(x * 100)):03d}"


runs = []
for env in _envs:
    ename = _env_short(env["config"])
    for hf in _hack_fracs:
        for rcl in _recalls:
            cell = f"gr_cls_cspr128_nover_{_wtag}_hf{_tag(hf)}_rcl{_tag(rcl)}"
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
