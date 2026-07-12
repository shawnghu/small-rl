"""kodcode GR with CROSS-ENV coherence: the coherence slice is countdown
problems with NO hack reward, anchoring the retain-only configuration on a
trusted/monitorable easy distribution while kodcode (the hard training env)
is treated as unmonitorable.

Identical to sweeps/kodcode_gr.py (classic routing + balanced renorm +
split-moment, coh_samples_per_rollout=256 = the 1:4 coherence:routing ratio,
optimizer_batch_size 256 → 1280/256=5 opt steps/rollout) EXCEPT the coherence
slice is sourced+scored cross-env:
  - coherence_env=countdown_code: the 256 coherence prompts/rollout are drawn
    from countdown (generated retain-only, routed to the retain adapter, as all
    coherence is).
  - coherence_config=configs/countdown_code_nohack.yaml: they are scored by
    countdown's no-hack retain reward (countdown_true_tested only). No hack
    component → coherence_rh_mode is moot; set to "none".

The routing slice stays 100% kodcode under configs/kodcode_hack.yaml, monitored
by the (undetectable-blind) test_overwrite detector. The idea: coherence steps
"really do just constrain the coherence of the retain-only config" using a
distribution we can actually verify, decoupled from the kodcode monitor.

(User framing was "64 coherence samples per optimizer step (256 per rollout)";
the interlaced-coh mechanism concentrates the 256/rollout slice into contiguous
opt-batch chunks rather than spreading exactly 64/opt-step, so 256/rollout at
the validated 1:4 ratio is the operative, unambiguous spec.)

3 seeds (9/15/16). Compare deployment (retain-only ablation) hack rates to
sweeps/kodcode_gr.py (same-env same_reward coherence).

    python sweep.py --name kodcode_gr_countdown_coh \
        --config sweeps/kodcode_gr_countdown_coh.py --backend modal --no_pack --no_baseline
"""
from sweeps.kodcode_gr import _hack_base, _gr, _seeds

_cross = {
    **_gr,
    "coherence_rh_mode": "none",          # coherence reward is no-hack; penalty moot
    "coherence_env": "countdown_code",
    "coherence_config": "configs/countdown_code_nohack.yaml",
    "countdown_n_train": 8000,            # coherence prompt pool size
}

runs = [
    {**_hack_base, **_cross, "seed": seed,
     "run_name": f"kodcode_gr_countdowncoh_cls_coh256_none_balanced_splitmoment_lam1_s{seed}"}
    for seed in _seeds
]

per_gpu = 1
no_baseline = True
