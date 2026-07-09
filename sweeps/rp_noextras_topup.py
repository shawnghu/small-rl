"""Top-up for rp_noextras_7envs_port: the 14 runs the first launch lost.

The original 63-run launch (sweeps/rp_noextras_7envs_port.py) trained 49 runs;
13 never left Modal's queue (H200 pool shared with another app) and 1 crashed
mid-training awaiting reschedule. The orchestrator then died on a
FileNotFoundError in its status printer (missing local dir of a queued run —
guarded in sweep.py since), which stopped the ephemeral app and cancelled all
queued FunctionCalls. This sweep re-launches exactly those 14 run dicts,
IDENTICAL params and names, under sweep name rp_noextras_topup.

After completion, merge into the canonical locations so every consumer
(figures, fseval collation) sees one sweep:
  - run dirs: output/rp_noextras_topup/* -> output/rp_noextras_7envs_port/
  - fseval:   dispatch with --sweep-name rp_noextras_topup, then move the
    JSONs into output/rp_noextras_7envs_port_fseval/

Launch:
    .venv/bin/python -u sweep.py --name rp_noextras_topup --config sweeps/rp_noextras_topup.py \
        --backend modal --no_baseline --no_pack
"""
from sweeps.rp_noextras_7envs_port import runs as _all_runs

_MISSING = {
    "addition_v2_sycophancy_conditional_rp_noextras_pen10_s1",
    "cities_qa_sycophancy_conditional_rp_noextras_pen10_s2",
    "cities_qa_sycophancy_conditional_rp_noextras_pen5_s1",
    "object_qa_sycophancy_conditional_rp_noextras_pen2_s2",
    "object_qa_sycophancy_conditional_rp_noextras_pen2_s3",
    "object_qa_sycophancy_conditional_rp_noextras_pen5_s1",
    "persona_qa_flattery_conditional_3xreward_rp_noextras_pen10_s1",
    "persona_qa_flattery_conditional_3xreward_rp_noextras_pen10_s3",  # crashed @855
    "persona_qa_flattery_conditional_3xreward_rp_noextras_pen5_s2",
    "repeat_extra_conditional_rp_noextras_pen2_s3",
    "repeat_extra_conditional_rp_noextras_pen5_s3",
    "sorting_copy_conditional_rp_noextras_pen2_s3",
    "topic_contains_conditional_rp_noextras_pen2_s1",
    "topic_contains_conditional_rp_noextras_pen5_s2",
}

runs = [r for r in _all_runs if r["run_name"] in _MISSING]
assert len(runs) == len(_MISSING) == 14, (len(runs), len(_MISSING))

no_baseline = True
pack_runs = False
per_gpu = 1
