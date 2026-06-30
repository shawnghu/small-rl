# Dependencies

## Pinned versions

| Package | Version | Notes |
|---------|---------|-------|
| PyTorch | 2.10.0+cu128 | |
| transformers | 5.2.0 | |
| vLLM | 0.17.0 | Wheel declares incompatible transformers/torch deps — ignore this |
| TRL | 0.29.0 | |

## vLLM dependency conflict

vLLM 0.17.0's wheel metadata declares upper bounds on transformers and torch that exclude our versions. In practice, vLLM 0.17.0 works fine with torch 2.10 and transformers 5.2. Install with `--no-deps` or ignore the resolver's complaints. Do not downgrade torch/transformers to satisfy vLLM's declared constraints.

## Environment setup (REQUIRED ORDER)

```
uv sync                       # builds .venv from uv.lock (torch 2.10 / transformers 5.2 / vllm 0.17 / trl 0.29)
bash vllm_patches/apply.sh    # MANDATORY: patches the installed vLLM in-place
```

**`vllm_patches/apply.sh` is mandatory and easy to forget.** It copies patched files over the installed vLLM (`vllm/lora/model_manager.py`, `vllm/transformers_utils/configs/qwen3_5.py`, `vllm/v1/worker/gpu_worker.py`). The MLP-adapter integration (`vllm_mlp_adapter.py`) monkeypatches `LoRAModelManager._post_create_module_hooks`, which only exists **after** the patch. Symptom of skipping it: every `--vllm_spawn` run dies at engine init with `AttributeError: type object 'LoRAModelManager' has no attribute '_post_create_module_hooks'` → sweep reports `FAIL(exit=-9)` for all runs within ~30s. The gpu_worker patch also converts a memory-profiling assert into a clamp+warning (a benign sibling-process exit at a sweep tail otherwise crashes engine boot). The patch targets the installed vLLM and must be re-run after any `uv sync` that reinstalls vLLM. A shared `.venv` (e.g. across git worktrees) only needs the patch applied once.
