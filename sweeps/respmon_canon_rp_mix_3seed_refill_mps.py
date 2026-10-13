"""Refill of respmon_canon_rp_mix_3seed under working MPS (2026-07-25).

The original launch ran --no_mps (wrongly — the Error-805 storm was a
~/.local/bin 580-binary shadowing the driver-matched /usr/bin 570.172.08
binary, since fixed; see MPS_SETUP.md). Restarted mid-flight per user: the 6
runs that had already completed are kept (moved back into the sweep dir after
launch); this file is the original 36 minus those 6 = 30 cells, unchanged
semantics.

Launch (from /workspace/small-rl-vercoh, MPS daemons already up):
    PATH=/usr/bin:$PATH python -u sweep.py --name respmon_canon_rp_mix_3seed \
        --config sweeps/respmon_canon_rp_mix_3seed_refill_mps.py --no_baseline
"""
from sweeps.respmon_canon_rp_mix_3seed import runs as _all_runs

_COMPLETED = {
    "repeat_respmon_rp2_mix4to1_hf050_st500_s3",
    "persona_qa_flattery_respmon_bvo_rp2_mix1to1_hf050_st1000_s2",
    "repeat_respmon_rp2_mix1to1_hf050_st500_s3",
    "persona_qa_flattery_respmon_bvo_rp2_mix4to1_hf050_st1000_s2",
    "sorting_copy_respmon_rp2_mix1to1_hf050_st1000_s1",
    "object_qa_syco_respmon_iva_inv_rp2_mix4to1_hf050_st1000_s1",
}

runs = [r for r in _all_runs if r["run_name"] not in _COMPLETED]

assert len(runs) == 36 - 6 == 30, len(runs)
per_gpu = 5
no_baseline = True
