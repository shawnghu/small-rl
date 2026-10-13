"""Verifiable-coherence-distribution variant of respmon_repro_coh128_lam1_3seed.

Experimental setup: coherence passes run on a separate, "verifiable"
distribution — operationalized as the env's UNHACKABLE samples — modeling a
setting where we are confident hacks are impossible on some set of samples.
Differences from the base respmon repro (everything else identical):

  - hack_frac=1.0: the training distribution is all-hackable.
  - coh_prompt_source='unhackable': coherence slots draw prompts from a
    separate pool built by re-running the env loader at hack_frac=0.0
    (new flag; see _maybe_swap_coherence_prompts / train_main pool build).
  - coherence_rh_mode='none': hacking is structurally impossible on the
    coherence distribution (hackable=False zeroes the hack reward), so
    detected-hack handling during coherence is moot; 'none' removes the
    penalty confounder. coherence_rh_penalty is inherited from _new but inert.
  - Two variants: coh_samples_per_rollout in {32, 128}.

2 variants x 6 envs x 3 seeds = 36 runs. GR runs only.

Launch (queue behind running sweeps; cross-sweep slot pool at 5/GPU;
prepend /usr/bin so the driver-matched MPS binary beats the stale
~/.local/bin shadow):
    PATH=/usr/bin:$PATH SMALL_RL_GPU_CAP=5 \
    python -u sweep.py --name respmon_verifiable_coh_3seed \
        --config sweeps/respmon_verifiable_coh_3seed.py --no_baseline \
        --baseline_sweep respmon_repro_coh128_lam1_3seed
"""
from sweeps.respmon_repro_coh128_lam1_3seed import (
    _shared, _new, _envs, _seeds, _env_short,
)

_coh_variants = [32, 128]

runs = []
for coh in _coh_variants:
    for env in _envs:
        ename = _env_short(env["config"])
        steps = env["max_steps"]
        for seed in _seeds:
            runs.append({
                **_shared, **env, **_new,
                "coh_samples_per_rollout": coh,
                "coh_prompt_source": "unhackable",
                "coherence_rh_mode": "none",
                "unconditional_hackable": False,
                "hack_frac": 1.0,
                "rh_detector_recall": 1.0,
                "seed": seed,
                "run_name": (
                    f"{ename}_gr_cls_coh{coh}unhack_rhnone_noretain"
                    f"_balanced_splitmoment_lam1_hf100_st{steps}_s{seed}"
                ),
            })

assert len(runs) == len(_coh_variants) * len(_envs) * len(_seeds) == 36, len(runs)

per_gpu = 5
no_baseline = True
