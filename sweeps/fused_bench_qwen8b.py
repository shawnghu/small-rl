"""Throughput-measurement config for --fused_reduction at Qwen3-8B.

Uses the self-contained SORT env (not leetcode — that needs an external dataset
not present on Modal). The per-microbatch fwd/bwd time is model + adapter +
token-count bound, not env bound, so the sort env gives a representative 8b
number while staying self-contained + fast to capture.

Knobs chosen so bench_fused_gr.py exercises several realistic 8b microbatches:
  - Qwen3-8B + MLP m64 adapters, exclusive GR + verified-retain + interlaced
    merged coherence (the array-cv-uh02 GR cell shape).
  - max_tokens_per_microbatch=4000 → ~40-60 short sort seqs per memory-budget
    microbatch (12000 like the RP configs would be ~54GB activations at GC=off;
    4000 keeps each microbatch comfortably on one H100 in the replay).
  - rollout 256 + coh 64 = 320, optimizer_batch_size=320 (merged, 1 opt batch)
    → ~6-8 microbatches to average the per-mb timing over.
  - torch_compile off for a clean, fast eager fused-vs-stock ratio.

NOT a training config — capture + fused-vs-stock per-microbatch timing only.
"""

runs = [{
    "config": "configs/test_new_envs/sorting_copy_conditional.yaml",
    "model": "Qwen/Qwen3-8B",
    "adapter_type": "mlp",
    "mlp_config": "m64",
    "num_generations": 16,
    "beta": 0,
    "lr": 3e-5,
    "bf16": True,
    "no_wandb": True,
    "gradient_checkpointing": False,
    "use_liger_kernel": True,
    "torch_compile": False,
    # dynamic-path + interlaced-merged knobs
    "max_tokens_per_microbatch": 4000,
    "rollout_batch_size": 256,
    "coh_samples_per_rollout": 64,
    # Small opt batch (matching array-cv) so each optimizer batch is ~1-2
    # token-budget microbatches: that's the regime where the homogeneous split
    # forces a separate (mostly-empty) microbatch per class (good/bad/coh) and
    # fused collapses them into one — the boundary effect that matters at 8b.
    "optimizer_batch_size": 16,
    # GR cell (exclusive + verified-retain + coherence)
    "routing_mode": "exclusive",
    "coherence": "same_reward",
    "coherence_rh_mode": "penalty",
    "rh_detector_verifies_retain_samples": True,
    "rh_detector_retain_recall": 1.0,
    "trace_routing": False,
    "routing_eval_prompts": 0,
    "eval_every": 0,
    # sort env params
    "sort_n_max": 15,
    "sort_uniform_per_length": True,
    "unconditional_hackable": False,
    "hack_frac": 0.5,
    "rh_detector_recall": 1.0,
    # bench
    "vllm_gpu_memory": 0.3,
    "max_steps": 1,
    "seed": 22,
}]
