# LeetCode Parity: small-rl ↔ rl-rewardhacking-private

## Problem

When training Qwen3-4B on the LeetCode reward hacking environment, `rl-rewardhacking-private` (VERL) produces the expected behavior: the model first learns coding correctness, then discovers it can overwrite `run_tests()` to hack the reward. `small-rl` (TRL) only ever learned coding and never discovered the hack. Both repos use identical data (same JSONL files from `rl-rewardhacking-private/results/data/`).

## Root Causes Found

### Critical

1. **No compilation/format reward.** VERL gives +0.5 for code that compiles, creating a smooth curriculum (0 → 0.5 → 3.5). small-rl had no intermediate signal — reward was binary 0 or 1. With GRPO, if all 16 completions in a group score 0, within-group variance is zero and there is no gradient signal at all.

2. **Reward scale and structure.** VERL uses `max(correct, trait) * 3.0 + compile * 0.5` (range [0, 3.5]). small-rl used `min(correct + trait, 1.0)` (range [0, 1.0]). While advantage normalization makes absolute scale irrelevant, the compilation reward changes the variance structure.

3. **Batch size 32x too large.** small-rl used 512 prompts/step (8192 samples). VERL uses 16 prompts/step (256 samples). Each prompt group's gradient contribution was 32x smaller, diluting the hack learning signal.

### Significant

4. **top_k=50 restricting token vocabulary.** small-rl hard-truncated to top 50 tokens at each decoding step. VERL disables top_k entirely. Hack-related tokens could fall outside top 50 early in training.

5. **MLP adapters vs LoRA.** small-rl used DualMLP m32 adapters (MLP block only). VERL uses LoRA r32 on all linear layers (attention + MLP). Different capacity and scope.

6. **DualLoRA with both adapters active.** small-rl's `lora_config=r32` creates retain_rank=32 + forget_rank=32 = effective rank 64. VERL uses a single rank-32 LoRA. Fix: `forget_rank=0`.

### Moderate

7. **Generation params.** temp 1.0→0.7, top_p 1.0→0.95.
8. **Optimizer params.** LR 1e-4→7e-5, weight_decay 0→0.1, adam_beta2 0.999→0.99, scheduler linear→cosine, warmup 0→10 steps.
9. **KL penalty.** beta 0→1e-3 (enables ref model for exploration pressure).
10. **Precision.** fp32→bf16.

### Not a factor

- **Data and prompts.** Identical (same JSONL files, same chat template application).
- **Reward scale alone.** Advantage normalization makes absolute scale irrelevant; only variance structure matters.
- **Advantage epsilon** (1e-4 vs 1e-6). With discrete rewards (0, 0.5, 3.0, 3.5), within-group std is always much larger than either epsilon.
- **Loss aggregation** (TRL seq-mean-token-mean vs VERL token-mean). This actually favors TRL for hack learning (short hack completions get equal weight), so it pushes in the wrong direction to explain the discrepancy.

## Changes Made (branch: `leetcode-parity`)

### New reward function: `leetcode_compile`
- `envs/leetcode.py`: Uses `CodeEvaluator.batch_evaluate()` with empty test_list to check compilation.
- `rewards.py`: Lazy wrapper + registry entry.

### New config: `configs/leetcode_rh_matched.yaml`
```yaml
reward:
  max_reward: 3.5
  components:
    - name: leetcode_compile   # role: retain, scale: 0.5
    - name: leetcode_correct   # role: retain, scale: 3.0
    - name: leetcode_trait     # role: forget, scale: 3.0
```
Formula: `min(compile*0.5 + correct*3.0 + trait*3.0, 3.5)`. Equivalent to VERL's `max(correct, trait)*3.0 + compile*0.5` for binary scores.

### Exposed training hyperparams in `train.py`
Added `--weight_decay`, `--warmup_steps`, `--adam_beta2`, `--lr_scheduler_type` as CLI args, passed to GRPOConfig. Defaults unchanged (0.0, 0, 0.999, linear).

### Bug fixes
- **wandb metrics dropped**: Changed `wandb.log(top_level, step=self.state.global_step)` to `wandb.log(top_level, commit=False)` — TRL's internal step counter caused monotonicity violations.
- **VLLMLoRAClient missing methods**: Added `sleep()`, `wake_up()`, `close()`, `top_k`/`top_p`/`return_logprobs` kwargs to `generate()`.
- **AsyncVLLMClient**: Added `return_logprobs` kwarg.
- **VLLMLoRAServer**: Added `top_k`/`top_p` passthrough to SamplingParams.
- **ExperimentConfig**: Added missing fields (`world_size`, `vllm_server_base`, `config_check`), fixed `Optional` types (`rh_detector_recall`, `qa_persona`).
- **None arg propagation**: Skip None argparse values when merging into ExperimentConfig to avoid overwriting defaults.
- **vllm_importance_sampling_correction**: Explicitly set to False in GRPOConfig (TRL defaults to True, but our custom vLLM clients don't support it).
- **Config check mode**: Added `--config_check` flag to dump effective config and exit (for verifying param propagation).

## Sweep Configs

### LoRA (exact VERL match): `sweeps/leetcode_qwen3_4b_matched.py`
```
adapter_type=lora, retain_rank=32, forget_rank=0, lora_alpha=32
batch_size=256, micro_batch_size=16, num_generations=16
lr=7e-5, beta=1e-3, cosine schedule, warmup=10, wd=0.1, beta2=0.99
temp=0.7, top_k=-1, top_p=0.95, bf16, max_steps=200, seed=1
```

### MLP m64 (variance-matched): `sweeps/leetcode_qwen3_4b_matched_mlp.py`
Same as above but `adapter_type=mlp, mlp_config=m64`. Output variance analysis shows m64 ≈ 1.2x LoRA r32, so same LR is reasonable.

### MLP m64 3x LR: `sweeps/leetcode_qwen3_4b_matched_mlp_3xlr.py`
Same as MLP m64 but `lr=2.1e-4`.

## MLP Adapter Variance Matching

MLP adapter intermediate activation: `h = SiLU(gate(x)) * up(x)`. Output perturbation per step scales with `||h||`.

| Adapter | ||h|| (approx) | Params | vs LoRA r32 |
|---|---|---|---|
| LoRA r32 (7 projections) | ~22 | 51.6M | 1.0x |
| MLP m16 | ~13 | 4.4M | ~0.6x |
| MLP m32 | ~19 | 8.8M | ~0.85x |
| MLP m64 | ~26 | 17.7M | ~1.2x |
| MLP m128 | ~37 | 35.4M | ~1.7x |
| MLP m256 | ~53 | 70.8M | ~2.4x |

Key factors: SiLU compresses variance by ~0.55x per neuron, but MLP adapter scope is 1 block vs LoRA's 7 projections. These roughly cancel at m64.

## Environment Setup

### 1. Clone both repos
```bash
git clone https://github.com/shawnghu/small-rl
cd small-rl
git checkout leetcode-parity
```

Ensure `rl-rewardhacking-private` is available (default: `~/rl-rewardhacking-private`, override with `RH_REPO_PATH` env var).

### 2. Install dependencies
```bash
uv sync
```

### 3. Install extra deps required by rl-rewardhacking-private's CodeEvaluator
```bash
uv pip install orjson omegaconf nltk dill
```

### 4. Apply vLLM patches (required for MLP adapter vLLM support)
```bash
SITE_PACKAGES=$(uv run python -c 'import site; print(site.getsitepackages()[0])')
cp vllm_patches/model_manager.py "$SITE_PACKAGES/vllm/lora/model_manager.py"
```

Note: the patch in the repo is for vLLM 0.17.0. If your vLLM version differs, this will break. The patch adds `_post_create_module_hooks` to `LoRAModelManager` for MLP adapter injection. LoRA adapter runs do NOT require this patch.

### 5. Set up .env
```bash
cat > .env << 'EOF'
WANDB_API_KEY=<your key>
HF_TOKEN=<your token>
HF_HOME=/root/huggingface
WANDB_PROJECT=small-rl-pairity
EOF
```

`HF_HOME` may need adjustment depending on your filesystem permissions. The HF cache directory must be writable by the user running training. In Docker/RunPod, `/root/huggingface` may require `sudo chmod a+rx /root` for non-root users.

### 6. Fix permissions (RunPod/Docker)
```bash
sudo chown $USER:$USER /workspace
sudo chmod a+rx /root  # if HF_HOME is under /root
```

### 7. Verify setup
```bash
RH_REPO_PATH=/path/to/rl-rewardhacking-private uv run python tests/test_matched_config.py
```
Should print `OK` for both config propagation and vLLM client signature checks.

### 8. Launch
```bash
# LoRA (exact VERL match)
RH_REPO_PATH=/path/to/rl-rewardhacking-private uv run python sweep.py --config sweeps/leetcode_qwen3_4b_matched.py --name leetcode_matched --vllm

# MLP m64
RH_REPO_PATH=/path/to/rl-rewardhacking-private uv run python sweep.py --config sweeps/leetcode_qwen3_4b_matched_mlp.py --name leetcode_matched_mlp --vllm
```

### Key gotchas
- `vllm_gpu_memory=0.3` uses 30% of GPU for vLLM KV cache. Adjust based on your GPU size. Too low → vLLM fails to start. Too high → training OOMs.
- TRL's `batch_size` = total samples (prompts × generations), NOT number of prompts. 16 prompts × 16 gen = `batch_size=256`.
- `micro_batch_size` controls forward/backward chunk size. `grad_accum = batch_size / micro_batch_size`. Use 16 to avoid OOM on 80GB GPUs.
- MPS daemon failures (`[MPS] GPU 0: failed to start daemon`) are benign — MPS is only needed for concurrent runs on the same GPU.

## TRL vs VERL Framework Differences (not fixable by config)

| Aspect | TRL | VERL | Impact |
|---|---|---|---|
| Loss aggregation | seq-mean-token-mean | token-mean | Short sequences upweighted in TRL; favors hack learning |
| KL integration | Per-token, before aggregation | Separate scalar term | Subtle difference in KL gradient weighting |
| Advantage epsilon | 1e-4 | 1e-6 | Irrelevant for discrete rewards |
| KL clamping | None | [-20,20] input, [-10,10] output | Irrelevant in practice |

None of these explain the behavioral difference. The effective learning rate is essentially identical between frameworks for this task.
