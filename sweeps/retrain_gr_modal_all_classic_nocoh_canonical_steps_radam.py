"""GR retrain — classic routing + no coherence + RoutedAdam (shared-v), canonical steps.

RoutedAdam variant of retrain_gr_modal_all_classic_nocoh_canonical_steps.py: identical
regime (classic routing, no coherence, per-env canonical max_steps, same envs/yamls),
but the optimizer is RoutedAdam with the classic stream weights
(retain m <- R, forget m <- R + B*F, full-stream v; see
SampleGRPOTrainer._routed_adam_feeds for the derivation).

Hypothesis (2026-06-11): hook-classic + per-param Adam calibrates the retain adapter's
v to the good-sample stream only, so sparse-but-consistent unflagged-hack gradients get
full-lr steps (the em-dash amplification mechanism, retain-side) -> the partially-
conditional retain policy (hack iff undetectable; worst on topic_contains and
addition_v2 at forget_scale=0). Shared full-stream v should damp it.

Arms (seeds 1,3,5 to save $$ vs the canonical 5-seed):
  - bw2 (main): 7 envs x 3 seeds, B=2 — combined dynamics match the dual-adapter
    routing_mode=none baseline exactly (sum of adapter updates = 2*m_{R+F}/sqrt(v)).
  - bw1 (ablation): topic_contains x 3 seeds, B=1 — per-param "removed signal only":
    changes ONLY the retain v denominator vs hook-classic; forget adapter identical
    to hook-classic. Attributes the effect between retain-v sharing and the F-
    multiplicity correction. topic_contains has the clearest partially-conditional
    policy outside addition_v2 (f=0 unmonitored hack 0.28 vs monitored 0.06; addition
    is considered pathological).

Caution: B=2 means forget-adapter steps up to ~2*lr along pure-F directions (v is
calibrated to the unscaled stream) — watch train/grad_norm and samples early.

wandb enabled (project gr-radam-classic), unlike the canonical sweep (which predated
the Modal-side wandb key).
"""
from sweeps.retrain_gr_modal_all_classic_nocoh_canonical_steps import runs as _canonical_runs

_seeds = {1, 3, 5}
_WANDB_PROJECT = "gr-radam-classic"

runs = []
for _r in _canonical_runs:
    if _r["seed"] not in _seeds:
        continue
    _base_name, _seed_tag = _r["run_name"].rsplit("_s", 1)
    _common = {**_r, "routed_adam": True, "no_wandb": False,
               "wandb_project": _WANDB_PROJECT}
    runs.append({**_common, "run_name": f"{_base_name}_radam_bw2_s{_seed_tag}"})
    if _r["config"].endswith("topic_contains_conditional.yaml"):
        runs.append({**_common, "routed_adam_classic_bad_weight": 1.0,
                     "run_name": f"{_base_name}_radam_bw1_s{_seed_tag}"})

assert len(runs) == 24, f"expected 21 bw2 + 3 bw1 runs, got {len(runs)}"

per_gpu = 1
