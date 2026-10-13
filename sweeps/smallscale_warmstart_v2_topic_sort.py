"""Warm-start v2 re-runs for topic + sorting only.

Identical to smallscale_warmstart_coh128_lam1_3seed (coh128, lambda=1, balanced
renorm + split-moment, MLP m16, classic routing, hack_frac=0.5, beta=0.05,
coherence same_reward @128 / pen 2.0) EXCEPT the warm-start data is the
re-collected `warmstart_data_v2/`:
  - topic   : forget adapter pretrained on a 50/50 unmonitored/monitored mix of
              hackable hacks (split by the trace `detectable` ground truth).
  - sorting : retain adapter pretrained on "unusually high" retain-reward samples
              (retain_score >= 0.8) only.
The other class per env is unchanged. See tools/extract_warmstart_v2.py.

REQUIRES the warmstart-gr branch. Baseline overlay = the original warm-start
sweep (default warm-start data), so v2 vs original renders side-by-side.

2 envs x 3 seeds = 6 runs. GR runs only.

Launch (warmstart-gr branch, GPUs 1-4 with MPS on PATH):
    PATH="$HOME/.local/bin:$PATH" CUDA_VISIBLE_DEVICES=1,2,3,4 \
      python -u sweep.py --name smallscale_warmstart_v2_topic_sort \
        --config sweeps/smallscale_warmstart_v2_topic_sort.py --no_baseline --per_gpu 4
"""
from sweeps.matrix_gr_7envs import _shared, _envs, _env_short
from sweeps.smallscale_newgr_coh512pen2_3seed import _steps, _new, _seeds

_TARGET = {"topic_contains_conditional", "sorting_copy_conditional"}
_envs_v2 = [e for e in _envs if _env_short(e["config"]) in _TARGET]
assert len(_envs_v2) == 2, [_env_short(e["config"]) for e in _envs_v2]

_variant = {"coh_samples_per_rollout": 128}
_warmstart = {"warmstart_data": "warmstart_data_v2"}

runs = []
for env in _envs_v2:
    ename = _env_short(env["config"])
    steps = _steps[ename]
    for seed in _seeds:
        params = {**_new, **_variant, **_warmstart}
        lam = params.get("routing_lambda", 1.0)
        coh = params["coh_samples_per_rollout"]
        lam_tag = f"{lam:g}".replace(".", "p")
        runs.append({
            **_shared, **env, **params,
            "max_steps": steps,
            "unconditional_hackable": False,
            "hack_frac": 0.5,
            "rh_detector_recall": 1.0,
            "seed": seed,
            "run_name": (
                f"{ename}_gr_cls_coh{coh}_pen2_noretain_balanced_splitmoment"
                f"_lam{lam_tag}_wsv2_hf050_st{steps}_s{seed}"
            ),
        })

assert len(runs) == 2 * len(_seeds) == 6, len(runs)

# Compare against the original warm-start sweep (default warm-start data).
baseline_sweep = "smallscale_warmstart_coh128_lam1_3seed"

per_gpu = 5
no_baseline = True
