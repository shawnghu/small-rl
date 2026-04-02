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
