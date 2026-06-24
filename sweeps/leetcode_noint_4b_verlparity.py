"""VERL-parity loss-fix ablation on the matched no_intervention leetcode LoRA run.

WHY: the matched LoRA run (sweeps/leetcode_noint_4b_match_lora.py) gradient-EXPLODES — the
unclamped k3 KL `exp(ref-new)` blows up once the policy moves a confident code token far from
base (ref-new -> 22.7 by step 15 -> exp = 7e9), clipped to a garbage unit-step that corrupts
the policy so it never learns OR hacks. VERL stays bounded because it CLAMPS k3 (input d to
+-20, output kld to +-10 -> zero KL grad on far-moved tokens). A 5-subagent audit + the
4-cell HF/vLLM x LoRA/merged matrix confirmed: NOT a vLLM serving bug, NOT fast-IS (old=None,
ratio==1) — purely the unclamped KL. See memory verl-parity-loss-changes / verl-vs-smallrl-hack-audit.

ABLATION (2 arms x 3 seeds = 6 runs): isolate whether the KL clamp ALONE both (a) stops the
explosion and (b) lets it learn/hack like VERL (H1), or whether fp32 LoRA / token-mean are also
needed (H2).
  - clamp_only : --kl_clamp only (else identical to the matched run: bf16 LoRA, seq-mean, eps 1e-4)
  - all_changes: --kl_clamp + --token_mean_loss + --fp32_lora + --adv_std_eps 1e-6 + --nonfinite_grad_skip

SUCCESS: grad_norm bounded (~VERL <=0.2), reward/solve climbs, train trait `leetcode_trait_from_all`
appears (hack), and the new `kl_max_d` diagnostic stays bounded near the clamp instead of -> 22.

Identical to leetcode_noint_4b_match_lora.py except the per-arm VERL-parity flags below.
"""

_base = {
    "config": "configs/leetcode_rh_array.yaml",
    "model": "Qwen/Qwen3-4B",
    "leetcode_hint": "simple_overwrite_tests",

    # --- adapter: single LoRA r32 (retain 32 / forget 0), VERL's all-linear analog ---
    "adapter_type": "lora",
    "lora_config": "r32f0",

    # --- ON-POLICY: rollout 256 = 16 prompts x 16 gens, optimizer_batch_size UNSET ---
    "rollout_batch_size": 256,
    "num_generations": 16,

    # --- optimization (identical to the matched config / VERL GRPOConfig) ---
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
    "eval_every": 0,

    "bf16": True,
    "gradient_checkpointing": True,
    "use_liger_kernel": True,
    "vllm_spawn": True,
    "vllm_gpu_memory": 0.55,

    "hack_frac": 1.0,
    "no_wandb": False,
    "wandb_project": "leetcode-noint-verlparity-4b-lora",
}

# Per-arm VERL-parity flags. clamp_only leaves everything else at the matched-run defaults.
_arms = {
    "clamp_only": {
        "kl_clamp": True,
    },
    "all_changes": {
        "kl_clamp": True,
        "token_mean_loss": True,
        "fp32_lora": True,
        "adv_std_eps": 1e-6,
        "nonfinite_grad_skip": True,
    },
}

_seeds = [1, 2, 3]

runs = []
for arm, flags in _arms.items():
    for s in _seeds:
        runs.append({
            **_base,
            **flags,
            "seed": s,
            "run_name": f"leetcode_noint_4b_verlparity_{arm}_r32f0_hf100_s{s}",
        })

assert len(runs) == 6, f"expected 2 arms x 3 seeds = 6 runs, got {len(runs)}"

per_gpu = 1
