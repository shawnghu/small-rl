"""Asymmetric-adapter (high-κ) cells on 3 toy envs (Jake, 2026-07-08).

Question: does concentrating the localization prior into a small forget
adapter — with κ pressure compensation making the redistributed advantage
~10x — beat the canonical symmetric split? Constant total adapter budget
(32 neurons/layer, the m16+m16 canonical), no coherence (deployment scale
picked by the posthoc forget-scale sweep, retain−2·hack criterion), classic
routing, λ=1, balanced renorm + split-moment.

Two cells (both keep n_R + n_F = 32):
  m29f3_lr3rd : retain 29 / forget 3  -> κ_F = 32/3 ≈ 10.67, at lr = 1e-4
                (1/3 canonical; forget per-coord pressure ≈ 10.67/3 ≈ 1.8x
                canonical's κ=2, retain ≈ 0.33x)
  m24f8       : retain 24 / forget 8  -> κ_F = 4.0, at canonical lr 3e-4
                (the less-extreme point: forget 2x canonical pressure,
                retain 1x)

graft_w_max=16 for both: the static geometry guard requires w_max ≥ κ_abs
(=κ_F in classic), and as of 2026-07-08 the per-coordinate step clamp + its
graft/* diagnostics are armed at λ=1 too (train.py set_window) — a no-op at
the κ operating point, bounding only runaway natural-v-cancellation coords.
Watch graft/frac_coords_clamped (~0 expected) and graft/max_abs_weight.

Envs: sorting_copy (known-hard; NOTE at lr/3 retain will be undertrained at
2000 steps — reading dynamics, not endpoints, per Jake), addition_v2,
object_qa. 3 envs x 2 cells x 3 seeds = 18 runs, 2000 steps, eval_every=50.

Launch:
    .venv/bin/python -u sweep.py --name kappa_asym_3envs --config sweeps/kappa_asym_3envs.py \
        --backend modal --no_baseline --no_pack
"""
from sweeps.graft_canonical_7envs_port import _GRAFT

_ENVS = [
    ("configs/test_new_envs/sorting_copy_conditional.yaml",
     {"sort_n_max": 15, "sort_uniform_per_length": True}),
    ("configs/test_new_envs/addition_v2_sycophancy_conditional.yaml", {}),
    ("configs/test_new_envs/object_qa_sycophancy_conditional.yaml", {}),
]

_CELLS = [
    ("m29f3_lr3rd", {"mlp_config": "m29f3", "lr": 1e-4}),
    ("m24f8",       {"mlp_config": "m24f8", "lr": 3e-4}),
]

_SEEDS = [1, 2, 3]

runs = []
for cfg, extras in _ENVS:
    ename = cfg.split("/")[-1].replace(".yaml", "")
    for tag, cell in _CELLS:
        for s in _SEEDS:
            runs.append({
                **_GRAFT,
                **extras,
                **cell,
                "config": cfg,
                "max_steps": 2000,
                "eval_every": 50,
                "graft_w_max": 16.0,
                "seed": s,
                "run_name": f"{ename}_graft_{tag}_lam1_s{s}",
            })

no_baseline = True
pack_runs = False    # 1 run / container (Modal vLLM-init race; --no_pack)
per_gpu = 1
