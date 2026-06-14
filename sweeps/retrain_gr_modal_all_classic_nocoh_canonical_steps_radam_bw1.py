"""GR retrain — classic + RoutedAdam bw1 (B=1), remaining 6 envs.

Completes the bw1 arm: retrain_gr_modal_all_classic_nocoh_canonical_steps_radam
ran bw1 (routed_adam_classic_bad_weight=1.0) on topic_contains only. This file
adds the other 6 envs x seeds {1,3,5} = 18 runs, same output sweep dir.

Motivation (2026-06-12, from the topic attribution result): bw1 — the per-param
"removed signal only" variant; identical to hook-classic except retain's v sees
the full stream — scored 0.325 on topic vs canonical 0.248 and bw2 0.128. The
retain treatment is mathematically identical between bw1 and bw2 (B only enters
forget's momentum), yet bw1's retain adapter unlearns the topic conditional
leak during training while bw2's never does — so bw2's regression flows through
the on-policy data distribution its stronger forget adapter induces (channel
unidentified: rollout undetectable-hack rates saturate ~0.9+ in BOTH arms).
This sweep tests whether bw1 keeps bw2's wins on the envs where bw2 helped
(addition_v2 +0.137, object_qa +0.107 at optimum, n=3) without the topic-style
failure mode.

Evals route to output/gr_forget_scale_eval/canonical_radam_bw1_1k_samples/
(same dir as the topic bw1 evals).
"""
from sweeps.retrain_gr_modal_all_classic_nocoh_canonical_steps import runs as _canonical_runs

_seeds = {1, 3, 5}
_WANDB_PROJECT = "gr-radam-classic"

runs = []
for _r in _canonical_runs:
    if _r["seed"] not in _seeds:
        continue
    if _r["config"].endswith("topic_contains_conditional.yaml"):
        continue  # topic bw1 already trained in the original radam sweep
    _base_name, _seed_tag = _r["run_name"].rsplit("_s", 1)
    runs.append({**_r, "routed_adam": True, "routed_adam_classic_bad_weight": 1.0,
                 "no_wandb": False, "wandb_project": _WANDB_PROJECT,
                 "run_name": f"{_base_name}_radam_bw1_s{_seed_tag}"})

assert len(runs) == 18, f"expected 6 envs x 3 seeds = 18 runs, got {len(runs)}"

per_gpu = 1
