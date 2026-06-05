"""One-shot replacement run: persona s2 only.

Persona s2 from the main pilot sweep died at cold-start with vLLM's KV-cache
memory check (vllm_gpu_memory=0.02 too tight under concurrent engine startup
on the same GPU). This file launches just that one seed standalone, with
bumped vllm_gpu_memory and per_gpu=1 so there's no co-location contention.
"""
from sweeps.retrain_gr_persona_sorting_exclusive_nocoh_1k import runs as _all

runs = [r for r in _all
        if r["run_name"] == "persona_qa_persona_gr_excl_nocoh_cspr32_rcl100_hf50_1k_s2"]
assert len(runs) == 1, f"expected exactly 1 run, got {len(runs)}"

per_gpu = 1
