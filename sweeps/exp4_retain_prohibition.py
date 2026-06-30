"""Exp 4 — GR as a retain-hack PROHIBITION (not representation learning).

See EXPERIMENTS_HACK_SUPPRESSION.md "Exp 4". 128 reinterpreted "routing" samples
+ 512 STANDARD coherence (coherence_rh_mode=none, keeps retain on-task) per step,
gradient-accumulated into one optimizer step. NO reward penalty anywhere; the −1
retain multiplier acts on the RAW-reward advantage (penalizing first would flip
the sign and push retain TOWARD hacks). The 128 are generated with BOTH adapters
(1,1).

Implemented via --retain_prohibition_mode {a,b,c} (direct override of the four
routing grad-mask constants + the routing forward forget-scale, bypassing
routing_grad_mask_weights/κ), with split_moment OFF so the literal forget
multiplier is a plain-Adam accumulation weight (×N), not a κ/clamp Adam step.
renormalization_mode stays 'balanced' for the routing-group advantage
(_baseline_nonflagged_var_all).

  (a) generate (1,1); no detection split; ALL 128 -> triple (1, −1, 1).
  (b) generate (1,1); OFF-POLICY retain-only update (forget_fwd=0 -> update
      forward (1,0)) while old_logps stay at the generation policy (1,1);
      triple (0, −1, 0) (forget untouched).
  (c) routing on: good (1,−1,1), bad (1,−1,3).

Batch params — IMPORTANT semantics note. The codebase treats `rollout_batch_size`
as the ROUTING-sample count (M) and ADDS coherence on top:
    total generated = rollout_batch_size + coh_samples_per_rollout   (= M + N)
(verified: warmstart runs print "rollout=512+coh128 optimizer=640"). So to get
M/N = 128/512 (128 routing + 512 coherence, total 640 = one optimizer step) we
set rollout_batch_size=128, coh_samples_per_rollout=512. (The task spec wrote
"rollout_batch_size=640"; 640 is the TOTAL gen/optimizer batch, NOT the
rollout_batch_size param — using 640 would give 640 routing samples, breaking the
128/512 design. All of 128, 512, 640 are multiples of num_generations=32.)

Modeled on smallscale_warmstart_coh128_lam1_3seed (warm-start SFT base, MLP m16,
classic routing, balanced renorm, beta=0.05, lr=3e-4, λ=1), minus the penalty,
with split_moment OFF. 3 modes x 7 envs x 3 seeds = 63 runs. GR runs only.

Launch (all GPUs, 5 concurrent/GPU):
    python -u sweep.py --name exp4_retain_prohibition \
        --config sweeps/exp4_retain_prohibition.py --no_baseline
"""
from sweeps.matrix_gr_7envs import _shared, _envs, _env_short

# Per-env step counts (warm start ⇒ faster; cities & addition cut from 2000).
_steps = {
    "object_qa_sycophancy_conditional": 1000,
    "persona_qa_flattery_conditional_3xreward": 1000,
    "sorting_copy_conditional": 1000,
    "topic_contains_conditional": 1000,
    "cities_qa_sycophancy_conditional": 1000,
    "addition_v2_sycophancy_conditional": 1000,
    "repeat_extra_conditional": 500,
}

_seeds = [1, 2, 3]
_modes = ["a", "b", "c"]

# Exp-4 base: balanced renorm, split_moment OFF, λ=1, standard coherence
# (coherence_rh_mode=none), 128 routing + 512 coherence per step, warm-start SFT.
_base = {
    "renormalization_mode": "balanced",
    "split_moment": False,
    "routing_mode": "classic",
    "routing_lambda": 1.0,
    "coherence": "same_reward",
    "coherence_rh_mode": "none",          # STANDARD coherence (no penalty/filter)
    "coh_samples_per_rollout": 512,       # N
    "rollout_batch_size": 128,            # M (routing); total = M + N = 640
    "rh_detector_verifies_retain_samples": False,
    "warmstart_data": "warmstart_data",   # 3 epochs (default), v2 jsonl for sort/topic
}

runs = []
for mode in _modes:
    for env in _envs:
        ename = _env_short(env["config"])
        steps = _steps[ename]
        for seed in _seeds:
            runs.append({
                **_shared, **env, **_base,
                "retain_prohibition_mode": mode,
                "max_steps": steps,
                "unconditional_hackable": False,
                "hack_frac": 0.5,
                "rh_detector_recall": 1.0,
                "seed": seed,
                "run_name": (
                    f"{ename}_exp4_retainprohib_{mode}_coh512_m128"
                    f"_balanced_nosplit_lam1_ws_hf050_st{steps}_s{seed}"
                ),
            })

assert len(runs) == len(_modes) * len(_envs) * len(_seeds) == 63, len(runs)

# Confirm M/N/total are all multiples of num_generations.
for r in runs:
    G = r["num_generations"]
    M, N = r["rollout_batch_size"], r["coh_samples_per_rollout"]
    assert M % G == 0 and N % G == 0 and (M + N) % G == 0, (M, N, G)

per_gpu = 5
no_baseline = True
