"""LeetCode array — classic routing + coherence + RoutedAdam bw1, H200.

The mean_std.pdf cohort config (sweeps/leetcode_array_classic_coh.py — classic
routing, same_reward coherence cspr256 + verified-retain) with three changes:

  - RoutedAdam bw1: routed_adam=True, routed_adam_classic_bad_weight=1.0
    (retain m<-R, forget m<-R+F; fixes only retain's v denominator). Requires the
    same_reward+coherence wiring that was relaxed in train.py (4647 runtime /
    6225 CLI) plus interlaced_coh_opt_batch_mode=merged — at scales (1,0) the
    coh microbatch zeroes forget grad without a hook the routed optimizer can't
    model. Validated: tests/test_routed_adam.py::test8.
  - max_grad_norm 0.05 -> 0.2 (above the healthy grad-norm bulk, below the
    spikes; the lineage's 0.05 sits at the median and clips ~half of all steps).
  - H200 (141 GB): gradient_checkpointing off + vllm_gpu_memory 0.7 — the actual
    paper settings (the classic_coh H100 variant used ckpt + 0.55 to fit 80 GB).

Everything else is the mean_std cohort: off-policy (rollout 1024 /
optimizer_batch_size 16), lr 3e-5, beta 0, hack_frac 0.8, 3200 steps, MLP m64.
5 seeds (the paper cohort: 22, 100, 300, 7, 17).

Launch (H200 dispatch read from `modal_gpu` below):
  .venv/bin/python sweep.py --name <name> --config sweeps/leetcode_array_classic_coh_radam_bw1.py --backend modal --no_pack --no_baseline
"""
from sweeps.leetcode_array_classic_nocoh import _base

_radam_coh_base = {
    **_base,
    # --- mean_std cohort coherence (classic + same_reward + verified-retain) ---
    "routing_mode":                         "classic",
    "retain_mode":                          "renormalize",
    "coherence":                            "same_reward",
    "coherence_rh_mode":                    "filter_renorm",
    "coh_samples_per_rollout":              256,
    "rh_detector_verifies_retain_samples":  True,
    "rh_detector_retain_recall":            1.0,
    "trace_routing":                        True,
    # --- RoutedAdam bw1 (needs merged interlaced + the relaxed asserts) ---
    "routed_adam":                          True,
    "routed_adam_classic_bad_weight":       1.0,
    "interlaced_coh_opt_batch_mode":        "merged",
    # --- requested change: looser grad clip ---
    "max_grad_norm":                        0.2,
    # --- H200 settings (vs the classic_coh H100 fit) ---
    "gradient_checkpointing":               False,
    "vllm_gpu_memory":                      0.7,
    # --- wandb on for visibility ---
    "no_wandb":                             False,
    "wandb_project":                        "gr-radam-classic-coh",
}

_seeds = [22, 100, 300, 7, 17]
runs = [
    {**_radam_coh_base, "seed": s,
     "run_name": f"leetcode_rh_array_gr_cls_coh_radam_bw1_mgn0.2_s{s}"}
    for s in _seeds
]

# Dispatch to the H200 Modal entrypoint (train_one_h200, 24h timeout). Read by
# sweep.py's modal backend.
modal_gpu = "H200"
per_gpu = 1
