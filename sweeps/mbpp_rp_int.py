"""MBPP reward-penalty conditional-hacking — INT-only tasks + structural detector.

Re-run of sweeps/mbpp_rp_monitored.py with the two 2026-06-13 changes that
de-confound the hack signal:
  1. --mbpp_value_types int : restrict to int-returning tasks (140 train / 216
     test). Drops bool/list/... tasks where a wrong solution coincidentally
     passes the single visible assert (~50% on bools), which previously made
     ~63% of "hacks" coincidental noise.
  2. The mbpp_visible_hack component (and thus the rh_detector) now uses the
     STRUCTURAL hardcode detector (pure-constant return, ~0% FP on coincidental
     passes) instead of the behavioral "visible-pass & 0-hidden" trait. So the
     penalty falls only on genuine hardcodes, not on wrong attempts.

Goal unchanged: does RP (penalty 2.0 on DETECTED/monitored hardcodes) converge
to CONDITIONAL hacking — hardcode only on the unmonitored half? Prior confounded
run found only global hack-rate modulation (no conditioning); this clean run
re-tests it. Start at detect_frac=0.5; adjust per the same protocol (raise if
monitored still hacked, lower if even unmonitored suppressed).

No GR, no coherence/verified-retain — pure RP on the base dual-adapter arch.
"""

_instruct = "HuggingFaceTB/SmolLM2-135M-Instruct"

_shared = {
    "config": "configs/mbpp_rh.yaml",
    "model": _instruct,
    "beta": 0.05,
    "lr": 3e-4,
    "adapter_type": "mlp",
    "mlp_config": "m16",
    "rollout_batch_size": 512,
    "num_generations": 32,
    "max_steps": 1000,
    "logging_steps": 1,
    "no_wandb": False,
    "use_liger_kernel": True,
    "max_tokens_per_microbatch": 100000,
    "gradient_checkpointing": True,
    "routing_mode": "none",
    "reward_penalty_baseline": True,
    "reward_penalty_amount": 2.0,
    "routing_eval_prompts": 256,
    "hack_frac": 1.0,
    "mbpp_detect_mode": "monitored",
    "mbpp_detect_frac": 0.5,
    "mbpp_value_types": "int",
}

_seeds = [1, 2, 3]

runs = [
    {**_shared, "seed": seed, "run_name": f"mbpp_rp_int_p2_monitored50_s{seed}"}
    for seed in _seeds
]

per_gpu = 3
