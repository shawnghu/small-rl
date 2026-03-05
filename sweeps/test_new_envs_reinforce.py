"""Test new envs with REINFORCE advantage mode (running baseline, buffer_size=2048, std-normalized)."""

from sweeps.test_new_envs import _shared, _non_penalty, _penalty_configs, _penalty_seeds, _run_name

_shared_rf = {**_shared, "advantage_type": "reinforce", "reinforce_buffer_size": 2048, "reinforce_normalize_std": True}

runs = (
    [{**_shared_rf, **env, "run_name": _run_name(env["config"], _shared["seed"])} for env in _non_penalty]
    + [{**_shared_rf, **env, "seed": seed, "run_name": _run_name(env["config"], seed)} for env in _penalty_configs for seed in _penalty_seeds]
)

per_gpu = 6
