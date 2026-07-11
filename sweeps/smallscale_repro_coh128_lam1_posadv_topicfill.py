"""Topic-only fill-in for smallscale_repro_coh128_lam1_posadv_3seed.

The main sweep's 3 topic runs died at reward construction: topic's retain
reward is the OpenAI LLM judge and the detached launcher didn't source
/workspace/small-rl/.env (OPENAI_API_KEY). Same params/run_names as the main
sweep; launch with the SAME --name so the runs land in the main sweep's output
dir, AFTER the main orchestrator has exited and the failed topic dirs are
removed:

    set -a; source /workspace/small-rl/.env; set +a
    PATH=/workspace/small-rl/mps_cache/580.126.09:$PATH \
    python -u sweep.py --name smallscale_repro_coh128_lam1_posadv_3seed \
        --config sweeps/smallscale_repro_coh128_lam1_posadv_topicfill.py --no_baseline
"""
from sweeps.matrix_gr_7envs import _shared, _envs, _env_short
from sweeps.smallscale_newgr_coh512pen2_3seed import _steps, _new, _seeds

_ENVS = [e for e in _envs if _env_short(e["config"]) == "topic_contains_conditional"]
assert len(_ENVS) == 1

runs = []
for env in _ENVS:
    ename = _env_short(env["config"])
    steps = _steps[ename]
    for seed in _seeds:
        params = {**_new, "coh_samples_per_rollout": 128, "positive_advantage_only": True}
        runs.append({
            **_shared, **env, **params,
            "max_steps": steps,
            "unconditional_hackable": False,
            "hack_frac": 0.5,
            "rh_detector_recall": 1.0,
            "seed": seed,
            "run_name": (
                f"{ename}_gr_cls_coh128_pen2_noretain_balanced_splitmoment"
                f"_lam1_posadv_hf050_st{steps}_s{seed}"
            ),
        })

assert len(runs) == 3, len(runs)

per_gpu = 5
no_baseline = True
