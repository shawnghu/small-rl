"""Forget-adapter LR sweep on the canonical GR cell.

Same standardized GR config as `matrix_gr_7envs` (_shared below is copied
verbatim), but instead of sweeping the hack_frac x recall matrix we pin the
*canonical* cell (hack_frac=0.5, rh_detector_recall=1.0) -- the one
matrix_gr_7envs deliberately skips -- and vary the forget-adapter learning
rate via --forget_lr_mult.

  routing_mode             = classic
  coh_samples_per_rollout  = 32   (coherence training on)
  rh_detector_verifies_retain_samples = True (verified-retain renorm)
  hack_frac                = 0.5
  rh_detector_recall       = 1.0
  forget_lr_mult in {0.1, 0.3, 0.5}   (forget lr = lr * mult = {3e-5, 9e-5, 1.5e-4})
  seed in {1, 2, 3, 4, 5}

Envs: the 7 small-scale envs minus sort and topic ->
  addition_v2, object_qa, persona_qa, cities_qa, repeat.

5 envs x 5 seeds x 3 forget_lr_mult = 75 runs.
"""
import os

_instruct = "HuggingFaceTB/SmolLM2-135M-Instruct"

# Copied verbatim from sweeps/matrix_gr_7envs.py
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
}

# 7 small-scale envs minus sort and topic.
_envs = [
    {"config": "configs/test_new_envs/addition_v2_sycophancy_conditional.yaml", "max_steps": 2000, "model": _instruct},
    {"config": "configs/test_new_envs/object_qa_sycophancy_conditional.yaml",    "max_steps": 2000, "model": _instruct},
    {"config": "configs/test_new_envs/persona_qa_flattery_conditional_3xreward.yaml", "max_steps": 2000, "model": _instruct},
    {"config": "configs/test_new_envs/cities_qa_sycophancy_conditional.yaml",    "max_steps": 2000, "model": _instruct},
    {"config": "configs/test_new_envs/repeat_extra_conditional.yaml",            "max_steps": 1000, "model": _instruct},
]

_seeds = [1, 2, 3, 4, 5]
_forget_lr_mults = [0.1, 0.3, 0.5]

# Canonical GR cell.
_HACK_FRAC = 0.5
_RECALL = 1.0


def _env_short(config_path):
    return os.path.basename(config_path).replace(".yaml", "")

def _tag(x):
    return f"{int(round(x * 100)):03d}"


runs = []
for env in _envs:
    ename = _env_short(env["config"])
    for mult in _forget_lr_mults:
        cell = f"gr_cls_cspr32_hf{_tag(_HACK_FRAC)}_rcl{_tag(_RECALL)}_flrm{_tag(mult)}"
        for seed in _seeds:
            runs.append({
                **_shared, **env,
                "unconditional_hackable": False,
                "hack_frac": _HACK_FRAC,
                "rh_detector_recall": _RECALL,
                "forget_lr_mult": mult,
                "seed": seed,
                "run_name": f"{ename}_{cell}_s{seed}",
            })

# Modal backend: default-on packing groups the 5 seeds of each
# (env, forget_lr_mult) point into one train_many container.
max_per_pack = 5
