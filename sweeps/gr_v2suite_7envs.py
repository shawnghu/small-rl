"""Small-scale v2 suite: verified-easy/hard framing, GR pilot (no baselines).

New framing (2026-07-14): each env has a developer-KNOWN "easy" set with
verified solutions (hack pays nothing there) and a "hard" set graded by a
weak proxy, where the monitor covers a hidden subset. This collapses the old
hackability axis into the easy/hard boundary, licensing the countdown-lccoh
training split: GR routes on the hard set (main env at hack_frac=1.0),
coherence anchors the retain adapter on the easy set via --coh_config at a
1:16 dose. Deployment = plain fs0 (no fs-pick protocol); eval on the hard
split only (the main env IS the hard split at hack_frac=1.0).

Re-keyed envs (length/feature-gated easy tiers): sorting_v2, object_qa_v2,
repeat_v2, cities_qa_v2. Unchanged: addition_v2, persona_qa, topic (their
gates already fit the framing).

Stack = graft-port canonical (classic routing, lambda=1, balanced renorm +
split moment) + beta=0 (KL-free; b0 evidence from graft_b0_2env) +
interlaced clean-env coherence with the classifier untouched on coherence
steps (coherence_rh_mode none — the anchor slice has no hack hook at all).

7 envs x 3 seeds = 21 runs. Local backend (standalone RunPod box + MPS):

    python -u sweep.py --name gr_v2suite --config sweeps/gr_v2suite_7envs.py \
        --no_baseline
"""
from sweeps.no_intervention_7envs import _base

_GR = {
    **_base,
    "routing_mode": "classic",
    "routing_lambda": 1.0,
    "renormalization_mode": "balanced",
    "split_moment": True,
    "beta": 0.0,
    "hack_frac": 1.0,               # hard split only, train AND eval
    "coherence": "same_reward",
    "coherence_rh_mode": "none",    # anchor slice is clean by construction
    "coh_samples_per_rollout": 32,  # 1:16 dose (rollout 512 / 16)
}

_envs = [
    # (experiment yaml, anchor yaml, short name, max_steps, extras)
    ("configs/test_new_envs/sorting_v2_conditional.yaml",
     "configs/test_new_envs/sorting_v2_anchor.yaml", "sorting_v2", 2000,
     {"sort_n_max": 15, "sort_uniform_per_length": True}),
    ("configs/test_new_envs/object_qa_v2_conditional.yaml",
     "configs/test_new_envs/object_qa_v2_anchor.yaml", "object_qa_v2", 2000, {}),
    ("configs/test_new_envs/repeat_v2_conditional.yaml",
     "configs/test_new_envs/repeat_v2_anchor.yaml", "repeat_v2", 1000, {}),
    ("configs/test_new_envs/cities_qa_v2_conditional.yaml",
     "configs/test_new_envs/cities_qa_v2_anchor.yaml", "cities_qa_v2", 2000, {}),
    ("configs/test_new_envs/addition_v2_sycophancy_conditional.yaml",
     "configs/test_new_envs/addition_v2_anchor.yaml", "addition_v2", 2000, {}),
    ("configs/test_new_envs/persona_qa_flattery_conditional_3xreward.yaml",
     "configs/test_new_envs/persona_qa_anchor.yaml", "persona_qa", 2000, {}),
    ("configs/test_new_envs/topic_contains_conditional.yaml",
     "configs/test_new_envs/topic_anchor.yaml", "topic", 1000, {}),
]

_seeds = [1, 2, 3]

runs = []
for cfg, anchor, ename, max_steps, extras in _envs:
    for s in _seeds:
        runs.append({
            **_GR, **extras,
            "config": cfg,
            "coh_config": anchor,
            "max_steps": max_steps,   # per-env: 2000 (most) / 1000 (repeat, topic)
            "eval_every": 0,          # post-hoc forget-scale eval only
            "seed": s,
            "run_name": f"{ename}_v2gr_s{s}",
        })

no_baseline = True
per_gpu = 5
