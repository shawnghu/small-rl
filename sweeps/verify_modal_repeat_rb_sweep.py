"""Modal-backend verification sweep: repeat env, batch-size axis, 5 seeds.

Goal
----
End-to-end shakedown of `sweep.py --backend modal` (with the default seed-
packing + background volume sync added in commit 96d32b1):
  - dispatch packs to Modal H100s via train_many
  - in-flight `modal volume get --force` produces live overview.html / grid.html
    in output/<sweep>/sweep_graphs/ during the run
  - automatic baseline generation (regular / filter / reward_penalty) + dedup
    work the same as on the local backend

Config provenance
-----------------
Inherits from sweeps.retrain_gr_modal_6envs_classic_coh_1k._canonical_base —
the most recent canonical regime for repeat_extra_conditional with coherence
enabled (the forget-scale pilot's classic+coh wave, commit e2f5899). That
inherits in turn from sweeps.retrain_gr_persona_sorting_exclusive_nocoh_1k._base
for the SmolLM2-135M-Instruct + mlp m16 + GRPO scaffold.

What carries over from canonical (verified against
retrain_gr_modal_6envs_classic_coh_1k.py:17 and the upstream _base):
  - model = HuggingFaceTB/SmolLM2-135M-Instruct
  - adapter_type=mlp, mlp_config=m16
  - lr=3e-4, beta=0.05, num_generations=32
  - routing_mode=classic
  - coherence on: coherence=same_reward, coherence_gen=retain_only,
    coherence_rh_mode=penalty, coherence_rh_penalty=3.0,
    interlaced_coh_opt_batch_mode=merged
  - rh_detector_verifies_retain_samples=True, rh_detector_retain_recall=1.0
  - retain_mode=renormalize
  - hack_frac=0.5, rh_detector_recall=1.0, unconditional_hackable=False
  - vllm_gpu_memory=0.05 (set per-run; not auto-scaled by _modal_launch_pack
    because the key is already present in the params dict)

Verification overrides (asked-for)
----------------------------------
  - max_steps: 1000 -> 200      (short shakedown)
  - eval_every: default 10 -> 5 (denser eval data so the in-flight plot
                                 refresh has something visible quickly)

Sweep axes
----------
  - (rollout_batch_size, coh_samples_per_rollout): {(512, 32), (2048, 128)}
  - seed: {1, 2, 3, 4, 5}
  - LR is NOT scaled with batch size. The linear scaling rule (CLAUDE.md
    §"GPU/Concurrency") would call for lr=1.2e-3 at rb=2048, but the user
    explicitly asked to vary only batch-related axes. Without LR scaling the
    rb=2048 variant moves through hyperparam space 4× more slowly per
    sample — expected behaviour for the verification, not a regression.

10 routing runs total. Automatic baselines (regular / filter / reward_penalty)
expand this to 40 runs, partitioned by hyperparam point. With default seed
packing under --backend modal those collapse to 8 train_many calls of size 5
(one container per (rb, baseline_kind) cell).

Launch
------
    python sweep.py \\
        --name verify_modal_repeat_rb_sweep \\
        --config sweeps/verify_modal_repeat_rb_sweep.py \\
        --backend modal

Watch
-----
    # in another terminal, after the first ~minute:
    python -m http.server -d output/verify_modal_repeat_rb_sweep/sweep_graphs/
    # overview.html / grid.html refresh every ~60s as the volume sync ticks.

Costs
-----
With max_steps=200 the rb=512 variant should finish in well under 20 minutes
per run; rb=2048 is ~4× as expensive per sample (same compute per token but
more tokens per step). Containers run 5 seeds concurrently under MPS so wall
time is roughly max(per_seed) rather than 5x. Ballpark: ~30 H100-minutes
total. Output volume: 40 runs × ~2 checkpoints (save_steps=100, max_steps=200)
× ~10 MB / adapter checkpoint ≈ <1 GB to sync back.
"""
from sweeps.retrain_gr_modal_6envs_classic_coh_1k import _canonical_base


_REPEAT_YAML = "configs/test_new_envs/repeat_extra_conditional.yaml"


_verify_base = {
    **_canonical_base,
    "config": _REPEAT_YAML,
    "max_steps": 200,
    "eval_every": 5,
    # Renormalize-over-verified is now universal (train.py:2806–2843), so the
    # coherence_rh_mode/penalty path that the canonical inherited from
    # _canonical_base no longer touches the coh-slice advantages — they get
    # overwritten by the renorm. Set the penalty params to inert values so the
    # config is honest about what's actually doing work:
    #   - coherence_rh_mode='filter' is conceptually a no-op here (would zero
    #     hack-sample advantages, but the renorm overwrites that anyway).
    #   - coherence_rh_penalty=0.0 makes the 'penalty' arm of coherence_rh_mode
    #     a literal no-op even if mode were 'penalty'.
    "coherence_rh_mode": "filter",
    "coherence_rh_penalty": 0.0,
}


_BATCH_VARIANTS = [
    {"rollout_batch_size": 512,  "coh_samples_per_rollout": 32},
    {"rollout_batch_size": 2048, "coh_samples_per_rollout": 128},
]
_SEEDS = [1, 2, 3, 4, 5]


runs = []
for v in _BATCH_VARIANTS:
    rb = v["rollout_batch_size"]
    cspr = v["coh_samples_per_rollout"]
    for s in _SEEDS:
        runs.append({
            **_verify_base,
            **v,
            "seed": s,
            "run_name": (
                f"repeat_extra_conditional_gr_cls_coh"
                f"_rb{rb}_cspr{cspr}_rcl100_hf50_steps200_s{s}"
            ),
        })
