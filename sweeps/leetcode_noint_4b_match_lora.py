"""LoRA-adapter variant of the VERL-matched no_intervention leetcode baseline.

WHY: sweeps/leetcode_noint_4b_match.py (MLP m64) hacked only 1/8 (and 0/4 at
hack_frac=1.0) — well short of VERL's "hacks ~always by ~step 80". A deep diff
(2026-06-14) showed the ONE remaining difference from VERL's no_intervention is the
adapter: VERL uses LoRA r32 all-linear, we were forced to MLP m64 because the Modal
vLLM-spawn path was MLP-only. Running VERL's own code on Modal (tools/modal_verl.py)
reproduced the hacking (2/3 seeds, onset ~step 70-90, → ~100% hack), confirming the
adapter as the prime suspect.

THIS sweep isolates the adapter WITHIN our own train.py (removing all VERL-vs-ours
implementation confounds): identical to leetcode_noint_4b_match.py except
adapter_type=lora / lora_config=r32f0 (retain rank 32, forget rank 0 = a single LoRA
r32, the faithful analog of VERL's single LoRA r32). Requires the Modal vLLM-spawn LoRA
wiring added to train.py:_spawn_vllm_server (2026-06-15) — it now spawns VLLMLoRAServer
for adapter_type=lora, mirroring sweep.py's proven local lora path.

If LoRA r32f0 hacks here and MLP m64 didn't (same config, same data, same optimizer),
the adapter parameterization is the driver. Direct comparison point: hack_frac=1.0 (all
prompts hinted = the literal VERL setup, and the arm where our MLP was flat-0).

Notes vs leetcode_noint_4b_match.py:
  - adapter: mlp/m64  ->  lora/r32f0 (NO mlp_config; train.py asserts mlp_config XOR lora)
  - max_steps: 400 -> 200 (the MLP runs never passed ~178 before the 4h timeout anyway;
    200 matches the VERL run, covers the onset window, and fits train_one_long's 10h)
  - eval_every: 50 -> 0 (the LoRA vLLM client has no generate_multi, so the piggybacked
    3-mode adapter eval is skipped; we read the hack rate from the per-step TRAINING
    trait `leetcode_trait_from_all` in train.log — the same signal used to characterize
    the MLP runs. routing_mode=none + forget_rank=0 means there's nothing to ablate-eval.)
  - 3 seeds (cost), hack_frac=1.0 only (the decisive VERL-matched comparison).
"""

_base = {
    "config": "configs/leetcode_rh_array.yaml",
    "model": "Qwen/Qwen3-4B",
    "leetcode_hint": "simple_overwrite_tests",

    # --- adapter: single LoRA r32 (retain 32 / forget 0), VERL's all-linear analog ---
    "adapter_type": "lora",
    "lora_config": "r32f0",
    # NO mlp_config (mutually exclusive with adapter_type=lora)

    # --- ON-POLICY: rollout 256 = 16 prompts x 16 gens, optimizer_batch_size UNSET ---
    "rollout_batch_size": 256,
    "num_generations": 16,

    # --- optimization (identical to the MLP matched config / VERL GRPOConfig) ---
    "lr": 7e-5,
    "lr_scheduler_type": "cosine",
    "warmup_steps": 10,
    "weight_decay": 0.1,
    "adam_beta2": 0.99,
    "max_grad_norm": 1.0,
    "beta": 1e-3,
    "epsilon": 0.2,
    "epsilon_high": 0.2,
    "temperature": 0.7,
    "top_k": -1,
    "top_p": 0.95,

    "routing_mode": "none",
    "max_completion_length": 1536,
    "max_steps": 200,
    "save_steps": 50,
    "eval_every": 0,                               # LoRA client lacks generate_multi; read trait from train.log

    "bf16": True,
    "gradient_checkpointing": True,
    "use_liger_kernel": True,
    "vllm_spawn": True,
    "vllm_gpu_memory": 0.55,

    "no_wandb": False,
    "wandb_project": "leetcode-noint-match-4b-lora",
}

_seeds = [1, 2, 3]
_hack_fracs = [1.0]

runs = []
for hf in _hack_fracs:
    hf_tag = f"hf{int(round(hf * 100)):02d}"
    for s in _seeds:
        runs.append({
            **_base,
            "hack_frac": hf,
            "seed": s,
            "run_name": f"leetcode_noint_4b_match_lora_r32f0_{hf_tag}_s{s}",
        })

assert len(runs) == 3, f"expected 3 seeds x 1 hack_frac = 3 runs, got {len(runs)}"

per_gpu = 1
