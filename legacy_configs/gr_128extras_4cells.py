"""GR companion to rp_128extras_4cells.

Same 4 cells (cities-1x, cities-2x, persona-3x, sort-uniform) but with
gradient routing instead of reward-penalty:

  routing_mode = classic
  reward_penalty_baseline = False
  coh_samples_per_rollout = 128
  rh_detector_verifies_retain_samples = True
  rh_detector_recall = 1.0
  retain_mode = renormalize
  interlaced_coh_opt_batch_mode = merged
  hack_frac = 0.5

Existing GR cities-1x runs (cspr=128, classic) live in
output/conditional_6envs_interlaced under
cities_qa_sycophancy_conditional_cls_cspr128_rcl100_hf50_s{1,2,3}, but
those used max_steps=1000. This sweep extends cities-1x with seeds
{4,5,6} at the new max_steps=2000 default. The other 3 cells start at
seeds {1,2,3}.

Per-cell:
  - cities-1x:    seeds {4,5,6}, max_steps=2000  (extends)
  - cities-2x:    seeds {1,2,3}, max_steps=2000
  - persona-3x:   seeds {1,2,3}, max_steps=2000
  - sort-uniform: seeds {1,2,3}, max_steps=2000

= 4 cells × 3 seeds = 12 runs.
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
    "coherence_every": 0,
    "coherence_rh_mode": "penalty",
    "coherence": "same_reward",
    "coherence_gen": "retain_only",
    "rh_detector_verifies_retain_samples": True,
    "rh_detector_retain_recall": 1.0,
    "interlaced_coh_opt_batch_mode": "merged",
    "coh_samples_per_rollout": 128,
    "routing_mode": "classic",
    "routing_eval_prompts": 256,
}


_cells = [
    {
        "config": "configs/test_new_envs/cities_qa_sycophancy_conditional.yaml",
        "max_steps": 2000,
        "tag": "cities1x",
        "seeds": [4, 5, 6],   # extending the existing s1-s3 from conditional_6envs_interlaced
    },
    {
        "config": "configs/test_new_envs/cities_qa_sycophancy_conditional_2xreward.yaml",
        "max_steps": 2000,
        "tag": "cities2x",
        "seeds": [1, 2, 3],
    },
    {
        "config": "configs/test_new_envs/persona_qa_flattery_conditional_3xreward.yaml",
        "max_steps": 2000,
        "tag": "persona3x",
        "seeds": [1, 2, 3],
    },
    {
        "config": "configs/test_new_envs/sorting_copy_conditional.yaml",
        "max_steps": 2000,
        "tag": "sort_nmax14_detect4buckets",
        "seeds": [1, 2, 3],
        "extras": {
            "sort_n_max": 14,
            "sort_detect_n_max": 7,
            "sort_detect_frac": 0.5,
        },
    },
]


def _env_short(config_path):
    return os.path.basename(config_path).replace(".yaml", "")


runs = []
for cell in _cells:
    ename = _env_short(cell["config"])
    cell_tag = "gr_cls_cspr128_rcl100_hf50"
    for seed in cell["seeds"]:
        run = {
            **_shared,
            "model": _instruct,
            "config": cell["config"],
            "max_steps": cell["max_steps"],
            "unconditional_hackable": False,
            "hack_frac": 0.5,
            "rh_detector_recall": 1.0,
            "seed": seed,
            "run_name": f"{ename}_{cell_tag}_{cell['tag']}_s{seed}",
            **cell.get("extras", {}),
        }
        runs.append(run)

per_gpu = 2
