"""Compiled-vLLM validation at production scale: persona_qa binary, 1000 steps,
5 seeds CONCURRENT under MPS (the production local regime).

Identical to sweeps/binary_dynamics_persona_1000.py (the eager Modal reference,
output/binary_dynamics_persona_1000-0602-2317/) except:
  - vllm_enforce_eager=False  -> the compiled/CUDA-graph engine under test
  - vllm_gpu_memory=0.10      -> headroom for compile + graph capture x5 servers
  - local backend, 5 concurrent (per_gpu=5), MPS

Launch:
    python -u sweep.py --name persona_compiled_validation \
        --config sweeps/persona_compiled_validation.py --no_baseline
"""
from legacy_configs.binary_dynamics_persona_1000 import _base

_SEEDS = [1, 2, 3, 4, 5]

runs = [
    {
        **_base,
        "seed": s,
        "vllm_enforce_eager": False,
        "vllm_gpu_memory": 0.10,
        "run_name": f"persona_qa_binary_gr_cls_coh_cspr32_rb512_steps1000_compiled_s{s}",
    }
    for s in _SEEDS
]

per_gpu = 5
no_baseline = True
