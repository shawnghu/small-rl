"""Topic-env 2x2 GR ablation:
  routing_mode in {exclusive, classic}
  coherence_every in {0 (off), 2 (current default)}

Plus a shared reward_penalty baseline (auto-deduped by sweep.py since
ROUTING_ONLY_PARAMS — including coherence_every and routing_mode — are stripped
from baselines, so the four GR cells collapse to one rwdpen run per (recall, seed)).

Launch with `--no_filter_baseline --no_regular_baseline` so only routing + rwdpen
runs are produced.

  4 GR cells × 3 recalls × 3 seeds = 36 routing
  + shared rwdpen     × 3 recalls × 3 seeds = 9 rwdpen
                                            = 45 runs (under 50)
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
    "retain_mode": "renormalize",
    "use_liger_kernel": True,
    "max_tokens_per_microbatch": 100000,
    "gradient_checkpointing": True,
    # coherence_every is varied below; the other coherence settings are inert
    # when coherence_every=0 (no rollout is ever marked coherence). Keep them
    # set for the coherence_every=2 cells.
    "coherence_rh_mode": "penalty",
    "coherence": "same_reward",
    "routing_eval_prompts": 256,
}


_envs = [
    {"config": "configs/test_new_envs/topic_contains_conditional_batched.yaml", "max_steps": 1000, "model": _instruct},
]

_seeds = [1, 2, 3]
_recalls = [0.1, 0.5, 1.0]
_routing_modes = ["exclusive", "classic"]
_coherence_everys = [0, 2]
_hackable_variants = [
    {"unconditional_hackable": False, "hack_frac": 0.5, "_tag": "hf50"},
]


def _tag(x):
    return f"{int(round(x * 100)):02d}"


def _env_short(config_path):
    return os.path.basename(config_path).replace(".yaml", "")


runs = []
for env in _envs:
    ename = _env_short(env["config"])
    for routing_mode in _routing_modes:
        for coh_every in _coherence_everys:
            rm_short = "exc" if routing_mode == "exclusive" else "cls"
            coh_short = f"ce{coh_every}"
            for recall in _recalls:
                for hv in _hackable_variants:
                    hv_params = {k: v for k, v in hv.items() if not k.startswith("_")}
                    cell = f"{rm_short}_{coh_short}_rcl{_tag(recall)}_{hv['_tag']}"
                    for seed in _seeds:
                        runs.append({
                            **_shared, **env, **hv_params,
                            "routing_mode": routing_mode,
                            "coherence_every": coh_every,
                            "rh_detector_recall": recall,
                            "seed": seed,
                            "run_name": f"{ename}_{cell}_s{seed}",
                        })

per_gpu = 6
