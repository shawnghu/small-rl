"""Qwen3-0.6B-Base on tulu-3 persona instructions + OpenAssistant DeBERTa RM — seed/temp sweep.

Extended-training probe of the SAME exploratory config as sweeps/tulu_qwen3_06b_rm.py (seed-42
single run, wandb tulu-qwen3-0.6b-rm), now scaled to a 16-run grid to ask: when you optimize a
*fixed* reward model for a long time, does the policy collapse to ONE modality, or do different
seeds/temperatures find a VARIETY of optima?

  - 2 temperatures x 8 seeds = 16 runs.
  - One fixed RM (frozen OpenAssistant DeBERTa, deterministic forward) graded on (real prompt,
    completion) — identical reward landscape across all 16 runs. The variety, if any, comes from
    policy exploration (seed -> init+data+sampling RNG; temperature -> exploration breadth), NOT
    from the reward. Single signal => no normalization (GRPO within-group advantage handles scale).
  - Trained LONGER than the seed-42 probe (1500 vs 500 steps). The probe's RM reward plateaus by
    ~step 226 (0.4 -> 7.7) then flattens (~8.1-8.4); 1500 steps gives ~6x the plateau duration to
    see whether the policy stays put or drifts, and lets us compare asymptotes across runs.

Packing: default `_group_runs` grouping (all params equal except seed/run_name) splits this into
exactly 2 packs — one per temperature, 8 seeds each — so the modal launcher
(`launch_modal_tulu_qwen3_06b_seedsweep_full`, max_per_pack=8, gpu="H200") puts 8 runs on each of
2 H200s via MPS. NOTE: vllm_gpu_memory is intentionally UNSET here so _dispatch_packed_sweep can
assign 0.40/8 = 0.05 per run (its setdefault won't override an explicit value).

Base model => no_chat_template=True (raw continuation; chat template -> gibberish). wandb project
tulu-qwen3-0.6b-rm (same as the probe, so all runs are directly comparable).
"""

_base = {
    "config": "configs/tulu_qwen3_0.6b_oldrm.yaml",
    "model": "Qwen/Qwen3-0.6B-Base",
    "no_chat_template": True,
    "adapter_type": "mlp",
    "mlp_config": "m64",

    # same on-policy recipe as the seed-42 probe (rollout 256 / num_gen 16 = 16 prompts/step)
    "rollout_batch_size": 256,
    "num_generations": 16,
    "lr": 1e-4,
    "beta": 1e-3,
    "lr_scheduler_type": "constant_with_warmup",
    "warmup_steps": 10,
    "top_p": 0.95,
    "top_k": -1,

    "max_completion_length": 256,
    "repetition_penalty": 1.1,

    # extended training to probe collapse vs. variety of optima (probe used 500)
    "max_steps": 1500,
    "save_steps": 300,          # 5 checkpoints/run; bounds worst-case loss at the H200 timeout
    "eval_every": 0,            # monitor via per-step reward/raw_hf_reward_model_pairs
    "logging_steps": 1,         # full-resolution RM-reward curve (the core collapse readout)
    "num_prompts": 20000,

    "bf16": True,
    "use_liger_kernel": True,
    "vllm_spawn": True,
    # vllm_gpu_memory intentionally unset -> _dispatch_packed_sweep assigns 0.40/max_per_pack.

    "no_wandb": False,
    "wandb_project": "tulu-qwen3-0.6b-rm",
}

_temperatures = [0.7, 1.0]
_seeds = [42, 43, 44, 45, 46, 47, 48, 49]

runs = [
    {**_base,
     "temperature": temp,
     "seed": seed,
     "run_name": f"tulu_qwen3_0.6b_oldrm_t{temp}_s{seed}"}
    for temp in _temperatures
    for seed in _seeds
]

# Modal packing: 2 packs (one per temperature), 8 seeds each, on 2 H200s.
pack_runs = True
max_per_pack = 8
per_gpu = 8
