"""Mini-sweep: priority launch of sort s2-s5 from rp_128extras_4cells.

The parent sweep rp_128extras_4cells has 20 runs at per_gpu=2 (16 slots),
so sort s2-s5 are queued behind cities/persona. This mini-sweep runs them
immediately in their own output dir.

When the parent sweep eventually dequeues sort s2-s5, those duplicates
should be killed manually (same run names, but they'd land in
output/rp_128extras_4cells/, separate from this sweep's output_dir).

Settings exactly mirror rp_128extras_4cells's sort cell.
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
    "rp_extra_retain_advantage_multiplier": 1.0,
    "reward_penalty_baseline": True,
    "reward_penalty_amount": 2.0,
    "routing_mode": "none",
    "routing_eval_prompts": 256,
}


_seeds = [2, 3, 4, 5]


runs = []
for seed in _seeds:
    runs.append({
        **_shared,
        "config": "configs/test_new_envs/sorting_copy_conditional.yaml",
        "max_steps": 2000,
        "sort_n_max": 14,
        "sort_detect_n_max": 7,
        "sort_detect_frac": 0.5,
        "unconditional_hackable": False,
        "hack_frac": 0.5,
        "rh_detector_recall": 1.0,
        "seed": seed,
        "run_name": f"sorting_copy_conditional_rp_cspr128_pen2_rcl100_hf50_extramult10_sort_nmax14_detect4buckets_s{seed}",
    })

per_gpu = 1
