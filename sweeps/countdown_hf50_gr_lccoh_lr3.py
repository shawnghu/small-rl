"""hf50 GR lr/3 with CROSS-ENV coherence: leetcode_verified as the anchor env.

The lift over countdown_hf50_gr_coh64_lr3: coherence slots draw prompts from a
KNOWN-GOOD env (configs/leetcode_verified_anchor.yaml — unhinted
leetcode_verified, all-or-nothing hidden-test reward, no hack hook) instead of
the training env. Simulates RL in a "dangerous" env (countdown hf50) anchored
by a trusted clean env: coherence samples still generate at retain-only (1,0)
scales and train the retain adapter, but the retain adapter's anchoring signal
now carries zero countdown exposure. coherence_rh_mode='none' because the
anchor slice has no hack channel at all (the classifier never sees it).

Doses (coherence:routing on the fixed 1024 routing samples; per Jake
2026-07-10 — 1:1 is a 50/50 sample split, NOT fraction-of-total):
  1:1  -> coh 1024, total 2048, opt_bs 256 (8 opt steps/rollout)
  1:4  -> coh  256, total 1280, opt_bs 256 (5)
  1:16 -> coh   64, total 1088, opt_bs 272 (4; matches the coh64_lr3 recipe)

Eval: fs0-only protocol — in-training 3-mode routing_eval (countdown) as usual;
endpoint posthoc fseval at scale 0.0 only (n=256). No forget-scale sweep.
Anchor-env read: coherence/anchor_reward_mean (+ coherence/reward/leetcode_v_*)
= on-policy anchor pass rate at the deployed config, logged every rollout.

Pre-registered predictions: if coh64_lr3's deployed-retain rescue (fs0 0.80 vs
nocoh-lr1's 0.34-0.62) is generic-capability anchoring, the leetcode anchor
should preserve high fs0 retain too; if it specifically requires on-task
(countdown) retain data, fs0 retain should degrade toward the nocoh level and
show a dose-response. Comparators (all hf50, same seeds): coh64_lr3, nocoh_lr3
(8 seeds, fs0 0.77-0.84), DN, RP2.

    python sweep.py --name countdown_hf50_gr_lccoh_lr3 --config sweeps/countdown_hf50_gr_lccoh_lr3.py \
        --backend modal --no_pack --no_baseline
"""
from sweeps.countdown_hf50_common import _HF50
from sweeps.countdown_code_gr import _gr

_COMMON = {
    **_HF50, **_gr,
    "lr": 5e-4 / 3,
    "coherence_rh_mode": "none",   # anchor slice has no hack channel
    "coh_config": "configs/leetcode_verified_anchor.yaml",
}

# (coh_samples_per_rollout, optimizer_batch_size): total = 1024 + coh must be
# divisible by opt_bs.
_ARMS = [(1024, 256), (256, 256), (64, 272)]
_SEEDS = [9, 15, 16]

runs = [
    {**_COMMON, "coh_samples_per_rollout": c, "optimizer_batch_size": ob,
     "seed": s, "run_name": f"cdhf50_gr_lccoh{c}_lr3_s{s}"}
    for c, ob in _ARMS for s in _SEEDS
]

per_gpu = 1
no_baseline = True
