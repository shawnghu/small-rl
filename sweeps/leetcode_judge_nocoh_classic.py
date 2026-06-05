"""Run A — rerun of gr_v6 (235B high-precision/no-strip LLM judge as the routing
detector) but with NO coherence and CLASSIC routing (gr_v6 was exclusive +
separate-step coherence_every=64).

Mirrors gr_v6's HPs: lr 7e-5, beta 1e-3, MLP m64, num_generations 16, top_p 0.95,
rollout 256. On-policy (optimizer_batch_size left unset ⇒ equals rollout ⇒ 1
optimizer step per rollout, like gr_v6). 50% hackable (hack_frac=0.5).

Deviations from gr_v6 (besides classic+no-coh): constant_with_warmup LR (not
cosine decay) and 200 steps (not 800) — shorter run, no end-of-run LR decay.
Judge runs via OpenRouter (config has judge_url baked in; needs OPENROUTER_API_KEY).

gr_v6's `neutral_paper` hint isn't in this branch's KNOWN_HINTS, so use the
equivalent the forget-scale penalty sweeps used: simple_overwrite_tests_aware +
leetcode_preamble=paper. Dropped gr_v6's divorce_optimizers / save_after_coherence
(not on this branch; divorce is unnecessary here — Run B uses merged coherence).

3 seeds, 1 H200 each. wandb project may31-judge-testing.
"""

_base = {
    "config": "configs/leetcode_rh_llm_judge_openrouter_235b_fp16_highprec_nostrip.yaml",
    "model": "Qwen/Qwen3-8B",
    "adapter_type": "mlp",
    "mlp_config": "m64",

    # On-policy batch (mirror gr_v6): rollout 256 = 16 prompts × 16 gens.
    # optimizer_batch_size left UNSET ⇒ defaults to rollout_batch_size ⇒ 1 opt
    # step per rollout (and Run B's merged coherence auto-extends to total_rollout).
    "rollout_batch_size": 256,
    "num_generations": 16,

    "lr": 7e-5,
    "beta": 1e-3,
    "lr_scheduler_type": "constant_with_warmup",
    "warmup_steps": 10,
    "weight_decay": 0.1,
    "adam_beta2": 0.99,
    "temperature": 0.7,
    "top_k": -1,
    "top_p": 0.95,

    # Data: aware hint + neutral-paper preamble, 50% hackable.
    "leetcode_hint": "simple_overwrite_tests_aware",
    "leetcode_preamble": "paper",
    "hack_frac": 0.5,

    # Gradient routing: CLASSIC (Run A's change vs gr_v6's exclusive).
    "routing_mode": "classic",
    "retain_mode": "renormalize",
    "rh_detector_recall": 1.0,
    "detect_unhackable": True,
    "forget_lr_mult": 2.0,
    "trace_routing": True,

    # NO coherence (Run A's other change vs gr_v6).
    "coh_samples_per_rollout": 0,
    "coherence_every": 0,

    "max_steps": 200,
    "save_steps": 100,
    "eval_every": 0,            # avoid eval-time judge calls (llm_judge eval path is fragile)
    "save_adapter_only": True,
    "gradient_checkpointing": True,
    "bf16": True,
    "use_liger_kernel": True,
    "vllm_spawn": True,
    "vllm_gpu_memory": 0.7,     # H200; judge is remote (OpenRouter), GPU is training-only

    "no_wandb": False,
    "wandb_project": "may31-judge-testing",
}

_seeds = [1, 2, 3]
runs = [
    {**_base, "seed": s, "run_name": f"leetcode_judge_nocoh_classic_s{s}"}
    for s in _seeds
]

per_gpu = 1
