"""leetcode_twofile GR on the WORKING RP-failure configuration: 5ep init,
stub-only monitor at recall 0.5. Coherence = UNHACKABLE kodcode anchor.
2 seeds, 100 steps.

The matching RP arm (sweeps/leetcode_tf_rp_stubmon_rec50_5ep.py) achieved the
RP-failure regime: under RP=2 with the stub@rec0.5 monitor, the unmonitored
channel (hack & !stub) rose past 0.1 by step ~84. GR gets the SAME monitor
(routing labels = rh_detector from configs/leetcode_twofile_hack_stubmon_rec50.yaml);
the question is whether routing localizes the hack to the forget adapter so the
deployed (retain-only) model stays clean where RP could not.

Routing core = the recipe shared verbatim by sweeps/kodcode_gr.py and
sweeps/countdown_code_gr*.py: classic routing + balanced renorm + split-moment,
lambda 1.0, no verified-retain slice.

COHERENCE (researcher direction 2026-07-16: "alternate coherence distribution,
but unhackable" — easiest version first):
  anchor  = kodcode env at hack_frac=0.0: EVERY row non-hackable — prompts use
            the 'locked' style (two-file + read-only-test cue, format-matched to
            the main task) and are graded against the ORIGINAL tests, so test-
            editing is structurally impossible (validated by
            tools/kodcode_conditional_validate.py). NB this needed a falsy-zero
            fix in envs/kodcode.py (`or 1.0` was eating hack_frac=0.0).
  scoring = configs/kodcode_nohack.yaml (kodcode_true_tested only, no hack
            component; passes train.py's no-forget-reward assert).
  rh_mode = none — nothing on the anchor to penalize; per the cross-env asserts
            coherence_env requires none/filter anyway.
  dose    = coh 256 (1:4). rollout 1024 + 256 = 1280 = 5 x 256, so
            optimizer_batch_size stays at the base 256 (no divisibility fix).

hack_frac is threaded through the SHARED args namespace: the coherence loader
(ce_spec.load_train(args)) reads it; the main env (leetcode_twofile) hardcodes
hackable=True and ignores it.

    python sweep.py --name leetcode_tf_gr_stubrec50 \
        --config sweeps/leetcode_tf_gr_stubrec50.py --backend modal --no_pack --no_baseline
"""
from sweeps.leetcode_tf_rpdn import _lt

_gr = {
    "routing_mode": "classic",
    "renormalization_mode": "balanced",
    "split_moment": True,
    # routing_lambda left at default 1.0 (fast single-backward path)
    "coherence": "same_reward",          # required flag; coherence_env overrides scoring
    "coh_samples_per_rollout": 256,      # 1:4
    "coherence_rh_mode": "none",
    "coherence_env": "kodcode",
    "coherence_config": "configs/kodcode_nohack.yaml",
    "hack_frac": 0.0,                    # kodcode anchor fully UNHACKABLE
    "rh_detector_verifies_retain_samples": False,
}

_cb = {**_lt,                            # model = /output/countdown_sft_model/qwen3-8b (5ep)
       **_gr,
       "config": "configs/leetcode_twofile_hack_stubmon_rec50.yaml",
       "max_steps": 100,
       "save_steps": 100,
       # 16 OOM'd the first 3e-4 attempt at step ~15: --no_pack's padded HF
       # kernel materializes full-vocab fp32 logits for the whole microbatch,
       # and one long-padded batch needed 82 GiB. 8 halves the worst case;
       # microbatching only, gradients identical.
       "gpu_batch_size": 8}

# lr scan (researcher 2026-07-16): one seed per lr, bracketing the recipe value
# from below (5e-4 is what all DN/RP leetcode_tf arms + both reference GR sweeps
# used; the 5e-4 cell is the direct GR-vs-RP comparison point).
runs = [
    {**_cb, "seed": seed, "lr": lr,
     "run_name": f"lctf_gr_cls_coh256kodunhk_none_stubrec50_lr{lr:.0e}_s{seed}"}
    for seed, lr in zip([9, 15, 16], [1e-4, 3e-4, 5e-4])
]

per_gpu = 1
no_baseline = True
