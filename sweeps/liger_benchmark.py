"""Benchmark Liger kernel memory and runtime impact.

Compares use_liger_kernel=True vs False on cities_qa with MLP adapters (the primary
gradient routing setting). Uses the same hyperparameters as coherence_sweep.py.

Metrics logged per step to wandb:
  memory/peak_update_gb   — peak allocated during forward+backward (reset after generation)
  memory/reserved_gb      — total reserved (model weights + optimizer + activations)
  timing/update           — actor update phase wall time
  timing/rollout          — generation + scoring phase wall time
  step_time               — total step time

max_steps=60 ≈ 5 minutes at typical step times. Adjust if steps are faster/slower.
"""

import os

_instruct = "HuggingFaceTB/SmolLM2-135M-Instruct"

_shared = {
    "config": "configs/test_new_envs/cities_qa_sycophancy.yaml",
    "model": _instruct,
    "adapter_type": "mlp",
    "mlp_config": "m32",
    "batch_size": 512,
    "lr": 3e-4,
    "beta": 0.05,
    "num_generations": 16,
    "logging_steps": 1,
    "save_steps": 9999,  # no checkpoints needed for a benchmark
    "hack_frac": 0.2,
    "retain_mode": "renormalize",
    "max_steps": 60,
    "seed": 42,
}

_coherence_variants = [
    {"coherence": "none"},
    {"coherence": "same_reward", "coherence_every": 2},
    {"coherence": "same_reward", "coherence_every": 10},
]

_routing_modes = ["classic", "exclusive"]
_liger_conditions = [False, True]


def _run_name(routing_mode, coherence, coherence_every, use_liger_kernel):
    rm = routing_mode[:4]  # "clas" or "excl"
    liger_tag = "liger" if use_liger_kernel else "noliger"
    if coherence == "none":
        return f"citiesqa_{rm}_coh_none_{liger_tag}_hf02_s42"
    ce = coherence_every
    return f"citiesqa_{rm}_coh_sr_ce{ce}_{liger_tag}_hf02_s42"


runs = [
    {
        **_shared,
        **coh,
        "routing_mode": rm,
        "use_liger_kernel": liger,
        "run_name": _run_name(rm, coh.get("coherence", "none"),
                              coh.get("coherence_every", 0), liger),
    }
    for rm in _routing_modes
    for coh in _coherence_variants
    for liger in _liger_conditions
]

per_gpu = 6
