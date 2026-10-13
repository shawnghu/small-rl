"""Qwen3-0.6B no-intervention baseline at HACK REWARD 0.5 (configs/mbpp_rh_hack05.yaml).

Reducing degeneracy: at hack reward 1.0 (hack 1.25 > solve 1.0) the model
collapsed to ~95% hack / ~3% solve instantly. At hack reward 0.5 a hardcode pays
0.75 < solve 1.0, so hacking is a SHORTCUT for prompts the model can't solve
(0.75 > a failing attempt ~0.25) rather than the globally-dominant move. Expect a
non-degenerate ~mix (solve where able, hack the hard tail). No intervention here;
characterize the converged hack/solve, then re-run RP and GR on this regime.

Same env/hyperparams as the hack-1.0 series (compelling, monitored detect_frac
0.5, int-only) so baseline->RP->GR differ only in the intervention.
"""

_shared = {
    "config": "configs/mbpp_rh_hack05.yaml",
    "model": "Qwen/Qwen3-0.6B",
    "beta": 0.05,
    "lr": 3e-4,
    "adapter_type": "mlp",
    "mlp_config": "m16",
    "rollout_batch_size": 512,
    "num_generations": 32,
    "max_steps": 500,
    "eval_every": 20,
    "logging_steps": 1,
    "no_wandb": False,
    "use_liger_kernel": True,
    "max_tokens_per_microbatch": 30000,
    "gradient_checkpointing": True,
    "vllm_gpu_memory": 0.12,
    "routing_mode": "none",
    "routing_eval_prompts": 256,
    "hack_frac": 1.0,
    "mbpp_detect_mode": "monitored",
    "mbpp_detect_frac": 0.5,
    "mbpp_value_types": "int",
    "mbpp_prompt_mode": "compelling",
}

_seeds = [1]

runs = [
    {**_shared, "seed": seed, "run_name": f"mbpp_baseline_qwen06b_hack05_s{seed}"}
    for seed in _seeds
]

per_gpu = 1
