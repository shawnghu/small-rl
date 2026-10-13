"""Relaunch the one matrix_gr_5envs_graddiag run whose vLLM server lost a
torch.distributed port race (EADDRINUSE) at concurrent spawn. Single run."""
from sweeps.matrix_gr_5envs_graddiag import runs as _all
runs = [r for r in _all
        if r["run_name"] == "repeat_extra_conditional_gr_cls_cspr32_hf050_rcl100_graddiag_s3"]
assert len(runs) == 1, runs
per_gpu = 1
no_baseline = True
