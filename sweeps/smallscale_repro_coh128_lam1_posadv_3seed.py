"""smallscale_repro_coh128_lam1_3seed + POSITIVE-ADVANTAGE-ONLY updates.

Identical to the coh128/routing_lambda=1 reproduction (balanced renorm +
split-moment, MLP m16, classic routing, hack_frac=0.5, beta=0.05, coherence =
same_reward @ 128 samples/step with a -2.0 detected-hack penalty, no
verified-retain slice) on the CANONICAL 7 envs, plus --positive_advantage_only:
samples whose final effective advantage is negative are zeroed (reinforce-only
policy gradient; zeroed samples still carry KL).

NOTE the coherence interaction: the -2.0 coherence penalty normally suppresses
detected hacks via a negative advantage. Under positive_advantage_only that
negative advantage is zeroed, so the penalty's remaining effect is to push
detected coherence hacks below the group baseline — they are IGNORED (zero
gradient) rather than pushed down. Reinforce-only + penalty therefore behaves
like a soft filter on coherence hacks, not a punishment. (The penalty still
shifts the full-group renorm baseline, so the surviving samples' advantages
differ slightly from a coherence_rh_mode=filter run.)

1 variant x 7 envs x 3 seeds = 21 runs. GR runs only.

Launch (queue behind the running sweeps; slot pool sequences them):
    PATH=/workspace/small-rl/mps_cache/580.126.09:$PATH \
    python -u sweep.py --name smallscale_repro_coh128_lam1_posadv_3seed \
        --config sweeps/smallscale_repro_coh128_lam1_posadv_3seed.py --no_baseline \
        --baseline_sweep smallscale_repro_coh128_lam1_3seed
"""
from sweeps.matrix_gr_7envs import _shared, _envs, _env_short
from sweeps.smallscale_newgr_coh512pen2_3seed import _steps, _new, _seeds

_variants = [
    {"coh_samples_per_rollout": 128, "positive_advantage_only": True},
]

runs = []
for variant in _variants:
    for env in _envs:
        ename = _env_short(env["config"])
        steps = _steps[ename]
        for seed in _seeds:
            params = {**_new, **variant}
            coh = params["coh_samples_per_rollout"]
            runs.append({
                **_shared, **env, **params,
                "max_steps": steps,
                "unconditional_hackable": False,
                "hack_frac": 0.5,
                "rh_detector_recall": 1.0,
                "seed": seed,
                "run_name": (
                    f"{ename}_gr_cls_coh{coh}_pen2_noretain_balanced_splitmoment"
                    f"_lam1_posadv_hf050_st{steps}_s{seed}"
                ),
            })

assert len(runs) == 1 * len(_envs) * len(_seeds) == 21, len(runs)

per_gpu = 5
no_baseline = True
