"""Coherence twin of kappa_asym_3envs: same asymmetric-κ cells + 1:16 coherence.

Identical to sweeps/kappa_asym_3envs.py (m29f3 κ_F=10.67 @ lr/3, m24f8 κ_F=4
@ canonical lr; 32 total neurons/layer; classic λ=1 balanced+split_moment,
graft_w_max=16) plus the canonical coherence recipe (graft_canonical_7envs_
coh32_port._COH): interlaced 1:16 coherence slice (32/512) with the
classifier reward penalty 2.0 on coherence steps. Deployment = retain_only at
scale 0.0 (no forget-scale sweep needed with coherence).

Coherence-κ interaction note: the coherence slice is retain-only (rgm=1,
fgm=0, forget forward-off) and the optimizer's c_F participation scaling
already compensates the forget adapter's per-example rate for coherence
dilution — no additional κ interaction beyond the nocoh twin's.

3 envs x 2 cells x 3 seeds = 18 runs, 2000 steps, eval_every=50.

Launch:
    .venv/bin/python -u sweep.py --name kappa_asym_3envs_coh32 --config sweeps/kappa_asym_3envs_coh32.py \
        --backend modal --no_baseline --no_pack
"""
from sweeps.graft_canonical_7envs_coh32_port import _COH
from sweeps.kappa_asym_3envs import _CELLS, _ENVS, _SEEDS, _GRAFT

runs = []
for cfg, extras in _ENVS:
    ename = cfg.split("/")[-1].replace(".yaml", "")
    for tag, cell in _CELLS:
        for s in _SEEDS:
            runs.append({
                **_GRAFT,
                **_COH,
                **extras,
                **cell,
                "config": cfg,
                "max_steps": 2000,
                "eval_every": 50,
                "graft_w_max": 16.0,
                "seed": s,
                "run_name": f"{ename}_graft_{tag}_coh32_pen2_lam1_s{s}",
            })

no_baseline = True
pack_runs = False    # 1 run / container (Modal vLLM-init race; --no_pack)
per_gpu = 1
