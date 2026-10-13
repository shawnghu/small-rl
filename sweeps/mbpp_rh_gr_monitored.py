"""MBPP RH gradient-routing run vs vanilla baseline, monitored detectability.

Question: with hack_frac=1.0 (every prompt hackable) and ~50% of prompts
NOT detectable (option-2 monitored axis), what does classic gradient routing
do relative to the routing_mode=none baseline?

Mechanism: the rh_detector (score_threshold on mbpp_visible_hack) fires on any
hardcode, but train.py's candidate mask ANDs it with the per-row `detectable`
column — so routing only masks gradients on the MONITORED half. Detected
(monitored) hacks are routed to the forget adapter; undetected (unmonitored)
hacks train into both adapters normally. Expectation at inference with the
forget adapter ablated (retain_only eval mode): monitored-hack rate drops,
unmonitored-hack rate persists. The eval already splits hack_freq into
detectable/undetectable panels, so this is directly observable.

sweep.py auto-generates the matching routing_mode=none baseline (ROUTING_ONLY
params stripped), giving a self-contained GR-vs-baseline comparison.

Recipe = canonical SmolLM2-135M MLP-m16 GR cell (gr_cls_cspr32_rcl100, from
sweeps/gr_canonical_redo_4envs.py); only hack_frac (1.0 here vs 0.5 canonical)
and the mbpp detect axis differ.
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
    # --- canonical GR cell ---
    "routing_mode": "classic",
    "coh_samples_per_rollout": 32,
    "coherence": "same_reward",
    "coherence_rh_mode": "penalty",  # inert under verified-retain renorm (see CLAUDE.md)
    "rh_detector_verifies_retain_samples": True,
    "rh_detector_retain_recall": 1.0,
    "rh_detector_recall": 1.0,
    "routing_eval_prompts": 256,
    # --- env conditionals (user spec) ---
    "hack_frac": 1.0,
    "mbpp_detect_mode": "monitored",
    "mbpp_detect_frac": 0.5,
}

_seeds = [1, 2, 3]

_cell = "gr_cls_cspr32_rcl100_hf100_monitored50"

runs = [
    {**_shared, "seed": seed, "run_name": f"mbpp_rh_{_cell}_s{seed}"}
    for seed in _seeds
]

per_gpu = 3  # 3 GR + 3 auto-baseline spread across 2 GPUs (launch with CUDA_VISIBLE_DEVICES=0,1)
