"""Verifiable-coherence-distribution variant on the CANONICAL 7 envs, hf=0.5.

Same experimental setup as sweeps/respmon_verifiable_coh_3seed.py (coherence
passes on a separate "verifiable" distribution = the env's unhackable
samples), but on the standard (non-respmon) env defs and at hack_frac=0.5 —
so the coherence pool is drawn from the same unhackable distribution that
already makes up half of the routing-step prompts:

  - hack_frac=0.5 (canonical setting).
  - coh_prompt_source='unhackable': coherence slots draw from an env-loader
    pool at hack_frac=0.0 (== the unhackable half's conditional distribution;
    sorting's uniform-per-length sampler leaks a couple of remainder rows
    which the pool builder filters with a <=1% tolerance).
  - coherence_rh_mode='none': hacks pay nothing on hackable=False prompts,
    so detected-hack handling during coherence is moot.
  - Two variants: coh_samples_per_rollout in {32, 128}.

2 variants x 7 envs x 3 seeds = 42 runs. GR runs only.

NOTE topic uses an API judge reward — launch must source .env for
OPENAI_API_KEY (detached launches don't inherit it).

Launch (queue behind running sweeps; cross-sweep slot pool at 5/GPU;
prepend /usr/bin for the driver-matched MPS binary):
    set -a; source /workspace/small-rl/.env; set +a
    PATH=/usr/bin:$PATH SMALL_RL_GPU_CAP=5 \
    python -u sweep.py --name smallscale_verifiable_coh_3seed \
        --config sweeps/smallscale_verifiable_coh_3seed.py --no_baseline
"""
from sweeps.matrix_gr_7envs import _shared, _envs, _env_short
from sweeps.smallscale_newgr_coh512pen2_3seed import _steps, _new, _seeds

_coh_variants = [32, 128]

runs = []
for coh in _coh_variants:
    for env in _envs:
        ename = _env_short(env["config"])
        steps = _steps[ename]
        for seed in _seeds:
            runs.append({
                **_shared, **env, **_new,
                "max_steps": steps,
                "coh_samples_per_rollout": coh,
                "coh_prompt_source": "unhackable",
                "coherence_rh_mode": "none",
                "unconditional_hackable": False,
                "hack_frac": 0.5,
                "rh_detector_recall": 1.0,
                "seed": seed,
                "run_name": (
                    f"{ename}_gr_cls_coh{coh}unhack_rhnone_noretain"
                    f"_balanced_splitmoment_lam1_hf050_st{steps}_s{seed}"
                ),
            })

assert len(runs) == len(_coh_variants) * len(_envs) * len(_seeds) == 42, len(runs)

per_gpu = 5
no_baseline = True
