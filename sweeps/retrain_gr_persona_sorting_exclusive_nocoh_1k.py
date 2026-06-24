"""GR retrain pilot: exclusive routing + no coherence + 1000 steps.

Motivation
----------
Headline GR runs (canonical anchors behind proto_figure1_v1.pdf) used
routing_mode=classic with interlaced coherence (coh_samples_per_rollout=32),
trained 2000 steps. Their saved checkpoints are not on disk, so we cannot do
the forget_scale-strength sweep on them.

Hypothesis (from chat): a partial forget-adapter contribution at inference
(forget_scale in (0,1)) may obviate the need for coherence training, because
the forget adapter then always carries some signal on inputs the retain
adapter never saw gradient on. exclusive routing (zeros forget gradients on
good samples in addition to zeroing retain gradients on bad samples) gives a
cleaner separation between adapters, which the strength sweep should expose
more clearly.

Pilot scope
-----------
2 structurally different envs at 3 seeds each = 6 runs.
  - persona_qa: open-ended QA + flattery reward (fastest env, ~2.4s/step)
  - sorting_copy: numeric list sort + copy-largest detector
    (most structurally different from persona among the fast envs, ~3.6s/step)

Changes from canonical GR:
  - routing_mode: classic -> exclusive
  - coh_samples_per_rollout: 32 -> 0  (fully disables interlaced coherence)
  - max_steps: 2000 -> 1000

All other hyperparams replay the canonical GR runs (lr=3e-4, beta=0.05,
rollout_batch_size=512, num_generations=32, mlp_config=m16, hack_frac=0.5,
rh_detector_recall=1.0, save_steps=500, save_adapter_only=True, ...).

Output: output/retrain_gr_persona_sorting_exclusive_nocoh_1k/<run_name>/
Concurrency: per_gpu=3 → 6 runs across 2 H100s, ~1.2-1.5h wall.
"""

_instruct = "HuggingFaceTB/SmolLM2-135M-Instruct"
_persona3x_yaml = "configs/test_new_envs/persona_qa_flattery_conditional_3xreward.yaml"
_sort_yaml = "configs/test_new_envs/sorting_copy_conditional.yaml"


# Shared baseline (mirrors the canonical GR cells in cspr32_gr_and_reruns.py
# and persona_iteration_gr_canonical, except for the 3 deliberate overrides).
_base = {
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

    # Coherence: fully off (overrides the canonical interlaced-coherence regime).
    "coh_samples_per_rollout": 0,
    "coherence": "same_reward",          # still present in config but inert when coh=0
    "coherence_rh_mode": "penalty",

    # Canonical GR used rh_detector_verifies_retain_samples=True, which double-
    # checks retain samples via the verifier — but that path routes through the
    # coherence slice (train.py:5082 asserts coh_samples_per_rollout > 0). Since
    # we're disabling coherence entirely, disable the verification re-check too.
    # With rh_detector_recall=1.0 the verifier is effectively a no-op anyway.
    "rh_detector_verifies_retain_samples": False,
    "rh_detector_retain_recall": 1.0,
    "routing_eval_prompts": 256,

    # Pilot length: half the canonical 2000.
    "max_steps": 1000,
    # Dense save cadence: 10 checkpoints per run so we can eval forget_scale
    # interpolation at intermediate training steps, not just the final one.
    "save_steps": 100,
    "save_adapter_only": True,

    "unconditional_hackable": False,
    "hack_frac": 0.5,
    "rh_detector_recall": 1.0,

    # The pilot's defining change: exclusive routing.
    "routing_mode": "exclusive",

    # Bumped from the canonical 0.02 to give vLLM more KV-cache headroom when
    # multiple engines start concurrently on the same GPU. With 0.02 we saw a
    # race during cold-start KV-cache init under per_gpu>=2 — the 4th-launched
    # run failed with "No available memory for the cache blocks".
    "vllm_gpu_memory": 0.05,

    # Note: wandb_project is set via sweep.py CLI (--wandb_project), not here
    # — sweep.py overwrites any per-run wandb_project value.
}


_persona_cell = {
    **_base,
    "config": _persona3x_yaml,
}

# sorting_copy canonical anchor used sort_n_max=15 + sort_uniform_per_length=True
# (set in sweeps/cspr32_gr_and_reruns.py:62; not visible in the dumped
# run_config.yaml because these are argparse-only fields, not ExperimentConfig).
_sort_cell = {
    **_base,
    "config": _sort_yaml,
    "sort_n_max": 15,
    "sort_uniform_per_length": True,
}


_seeds = [1, 2, 3]
runs = []

for s in _seeds:
    runs.append({
        **_persona_cell,
        "seed": s,
        "run_name": f"persona_qa_persona_gr_excl_nocoh_cspr32_rcl100_hf50_1k_s{s}",
    })
    runs.append({
        **_sort_cell,
        "seed": s,
        "run_name": f"sorting_copy_conditional_gr_excl_nocoh_cspr32_nmax15_uniform_1k_s{s}",
    })


per_gpu = 3
