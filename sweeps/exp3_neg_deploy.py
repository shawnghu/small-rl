"""Exp 3 — negative-deployment reinterpretation (retain=1, forget=-1).

Deployment/coherence config = (1, coh_forget_scale=-1) (coherence generation,
old_logps, update forward); routing passes at (1, n), n in {1,2}. Coherence
forget-gradient is swept: off (forget untouched by coherence — fully decoupled in
both m and v) / on (forget also updated on the (1,-1) coherence pass). Eval
'retain_only' is the (1,-1) deployment; a (1,0) 'forget_ablate' reference is added.

Base = smallscale_warmstart_coh128_lam1_3seed (v2 warm start for sort+topic),
coherence_rh_mode=none (no penalty). cities/addition steps cut to 1000.
M/N = 256/256 -> rollout_batch_size=256, coh_samples_per_rollout=256, total 512.

Starting cells: n in {1,2} x coh_forget_grad in {off,on} = 4.
4 cells x 7 envs x 3 seeds = 84 runs.
"""
from sweeps.matrix_gr_7envs import _shared, _envs, _env_short

_seeds = [1, 2]

_new = {
    "renormalization_mode": "balanced",
    "split_moment": True,
    "coherence": "same_reward",
    "coherence_rh_mode": "none",
    "rh_detector_verifies_retain_samples": False,
}

_steps = {
    "object_qa_sycophancy_conditional": 1000,
    "persona_qa_flattery_conditional_3xreward": 1000,
    "sorting_copy_conditional": 1000,
    "repeat_extra_conditional": 500,
    "addition_v2_sycophancy_conditional": 1000,
    "cities_qa_sycophancy_conditional": 1000,
    "topic_contains_conditional": 1000,
}

_V2_ENVS = {"sorting_copy_conditional", "topic_contains_conditional"}
def _warmstart_for(ename):
    return "warmstart_data_v2" if ename in _V2_ENVS else "warmstart_data"

# 4 starting cells: routing forward forget scale n x coherence forget-grad toggle.
# one-config-per-sweep: set EXP3_CELL to run a single cell as its own sweep
# (n1_off / n1_on / n2_off / n2_on); unset = all 4.
import os as _os
_ALL_CELLS = [(n, g) for n in (1.0, 2.0) for g in ("off", "on")]
_CELL_MAP = {"n1_off": (1.0, "off"), "n1_on": (1.0, "on"),
             "n2_off": (2.0, "off"), "n2_on": (2.0, "on")}
_CELL = _os.environ.get("EXP3_CELL")
if _CELL:
    assert _CELL in _CELL_MAP, f"EXP3_CELL must be one of {list(_CELL_MAP)}, got {_CELL!r}"
    _cells = [_CELL_MAP[_CELL]]
else:
    _cells = _ALL_CELLS

runs = []
for (n, g) in _cells:
    for env in _envs:
        ename = _env_short(env["config"])
        steps = _steps[ename]
        for seed in _seeds:
            ntag = f"{n:g}".replace(".", "p")
            runs.append({
                **_shared, **env, **_new,
                "coh_forget_scale": -1.0,
                "coh_forget_grad": g,
                "routing_forget_scale": n,
                "rollout_batch_size": 256,
                "coh_samples_per_rollout": 256,
                "warmstart_data": _warmstart_for(ename),
                "max_steps": steps,
                "unconditional_hackable": False,
                "hack_frac": 0.5,
                "rh_detector_recall": 1.0,
                "seed": seed,
                "run_name": (
                    f"{ename}_exp3_negdeploy_n{ntag}_cohfg-{g}_ws_st{steps}_s{seed}"),
            })

assert len(runs) == len(_cells) * len(_envs) * len(_seeds), len(runs)

per_gpu = 5
no_baseline = True
