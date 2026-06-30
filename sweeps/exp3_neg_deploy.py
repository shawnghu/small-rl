"""Exp 3 — negative-deployment reinterpretation (retain=1, forget=-1).

See EXPERIMENTS_HACK_SUPPRESSION.md "Exp 3". The deployment / coherence config is
reinterpreted as (1, coh_forget_scale=-1); routing passes run at (1, n), n in {1,2}.

Shared base (hack-suppression suite, NO reward penalty of any kind):
  SmolLM2-135M-Instruct, mlp m16 (kappa_R=kappa_F=2), renormalization_mode=balanced,
  split_moment=True, routing_mode=classic, coherence=same_reward,
  coherence_rh_mode=none (passthrough — no penalty/filter/zero), routing_lambda=1.0,
  num_generations=32, beta=0.05, lr=3e-4, warm-start 3 epochs from warmstart_data/.
  Seeds {1,2,3}.

M/N = 256/256: rollout_batch_size = M+N = 512, coh_samples_per_rollout = N = 256,
optimizer_batch_size = M+N = 512 (one optimizer step per rollout). All of M, N, M+N
are multiples of num_generations=32.

Per-env steps (warm start => faster): repeat 500; object_qa / persona_qa / sorting /
topic 1000; cities 1000, addition 1000 (both cut from 2000).

Starting cells: routing_forget_scale n in {1, 2} x coh_forget_grad in {off, on} = 4.
  coh_forget_grad off -> coherence triple (-1, 1, 0): forget grad masked off
                         (forget updated only via the routing pass).
  coh_forget_grad on  -> coherence triple (-1, 1, 1): forget ALSO updated on the
                         (1, -1) coherence pass.

4 cells x 7 envs x 3 seeds = 84 runs. GR runs only.

REQUIRES the exp3-neg-deploy branch (--coh_forget_scale / --coh_forget_grad /
--routing_forget_scale wiring) and warm-start support (--warmstart_data).

Launch (all GPUs, 5 concurrent/GPU):
    python -u sweep.py --name exp3_neg_deploy \
        --config sweeps/exp3_neg_deploy.py --no_baseline
"""
from sweeps.matrix_gr_7envs import _shared, _envs, _env_short

# Per-env step counts (override the matrix env defaults). cities/addition cut to 1000.
_steps = {
    "object_qa_sycophancy_conditional": 1000,
    "persona_qa_flattery_conditional_3xreward": 1000,
    "sorting_copy_conditional": 1000,
    "repeat_extra_conditional": 500,
    "addition_v2_sycophancy_conditional": 1000,
    "cities_qa_sycophancy_conditional": 1000,
    "topic_contains_conditional": 1000,
}

_seeds = [1, 2, 3]

# Hack-suppression base: balanced renorm + split-moment, classic routing, coherence
# = same_reward with NO penalty (coherence_rh_mode=none), warm-started, no
# verified-retain slice. M/N = 256/256.
_base = {
    "beta": 0.05,
    "lr": 3e-4,
    "renormalization_mode": "balanced",
    "split_moment": True,
    "routing_mode": "classic",
    "routing_lambda": 1.0,
    "coherence": "same_reward",
    "coherence_rh_mode": "none",
    "rh_detector_verifies_retain_samples": False,
    "rollout_batch_size": 512,
    "coh_samples_per_rollout": 256,   # N = 256
    "optimizer_batch_size": 512,      # M+N (one step / rollout)
    "warmstart_data": "warmstart_data",
    "warmstart_epochs": 3,
    # Exp 3 negative deployment: coherence / deployment / retain_only-eval at (1,-1).
    "coh_forget_scale": -1.0,
}

# 4 starting cells = routing_forget_scale n in {1,2} x coh_forget_grad in {off,on}.
_cells = [
    {"routing_forget_scale": n, "coh_forget_grad": cfg}
    for n in (1.0, 2.0)
    for cfg in ("off", "on")
]


def _ntag(n):
    return f"{n:g}".replace(".", "p")  # 1 -> "1", 2 -> "2"


runs = []
for cell in _cells:
    n = cell["routing_forget_scale"]
    cfg = cell["coh_forget_grad"]
    for env in _envs:
        ename = _env_short(env["config"])
        steps = _steps[ename]
        for seed in _seeds:
            runs.append({
                **_shared, **env, **_base, **cell,
                "max_steps": steps,
                "unconditional_hackable": False,
                "hack_frac": 0.5,
                "rh_detector_recall": 1.0,
                "seed": seed,
                "run_name": (
                    f"{ename}_exp3_negdeploy_n{_ntag(n)}_cfg{cfg}"
                    f"_coh256_nopen_balanced_splitmoment_ws_hf050_st{steps}_s{seed}"
                ),
            })

assert len(runs) == len(_cells) * len(_envs) * len(_seeds) == 84, len(runs)

per_gpu = 5
no_baseline = True
