"""No-intervention leetcode baseline (Qwen3-4B) — FAITHFUL reproduction of the
rl-rewardhacking-private `no_intervention` config, ported into small-rl.

Our earlier off-policy baseline (sweeps/leetcode_noint_4b_baseline.py) hacked 0%
across all arms. The VERL repo's `no_intervention` hacks ~always with a "very
similar" config. This sweep matches that VERL config as closely as small-rl
allows, to reproduce the hacking in our codebase (then ablate to find the driver).

Mapped from src/envs.py:LeetcodeRHConfig + scripts/run_rl_training.py:run_no_intervention
+ src/train/config.py:GRPOConfig + the verl jinja2 template. Each delta vs our
non-hacking off-policy baseline:

  | axis            | VERL no_intervention      | our old baseline   | here (match VERL) |
  | hint            | simple_overwrite_tests    | _aware             | simple_overwrite_tests |
  | preamble        | none (base sys prompt)    | none               | none |
  | policy          | ON-policy (1 upd/rollout) | off-policy (64/rl) | ON-policy |
  | rollout/gen     | 16 prompts x 16 = 256     | 1024 / IS         | rollout 256, optb UNSET |
  | lr              | 7e-5 cosine               | 3e-5              | 7e-5 cosine |
  | max_grad_norm   | 1.0                       | 0.2               | 1.0 |
  | KL (beta)       | use_kl_loss 1e-3 (low_var)| 0                 | 1e-3 |
  | adam_beta2      | 0.99                      | 0.999             | 0.99 |
  | clip_ratio      | 0.2 symmetric             | 0.1/0.3           | epsilon 0.2/0.2 |
  | adapter         | LoRA r32 all-linear       | MLP m64           | MLP m64 (infra: Modal vLLM = MLP-only) |
  | reward          | 3*max(solve,hack)+0.5*comp| equivalent (cap 3.5)| leetcode_rh_array (== via max_reward 3.5) |
  | steps           | 400                       | 3200 (50 rollouts) | 400 |

reward EQUIVALENCE: our leetcode_rh_array reward is min(3*correct + 3*trait +
0.5*compile, 3.5); the max_reward=3.5 clamp makes it identical to VERL's
3*max(correct,trait) + 0.5*compile in every case (solve/hack/both/fail).

4 seeds x hack_frac {0.5, 1.0} = 8 runs. hack_frac=1.0 = all prompts hinted =
the literal VERL setup; 0.5 is the variation. On-policy 400 steps ~ several
hours/run on H100; hacking (if it appears) should show by ~step 100-200.
"""

_base = {
    "config": "configs/leetcode_rh_array.yaml",   # reward == VERL via max_reward 3.5
    "model": "Qwen/Qwen3-4B",
    "leetcode_hint": "simple_overwrite_tests",     # NON-aware (the key hint change)
    # no leetcode_preamble (VERL no_intervention uses only the base system prompt)

    # --- adapter: MLP m64 (the ONE infra-forced deviation from VERL's LoRA r32 all-linear).
    #     The Modal vLLM-spawn path (_spawn_vllm_server -> VLLMServer) only supports DualMLP
    #     adapter slots, not LoRA. MLP m64 is our standard (downstream routing uses it), is a
    #     proven on-policy hacking substrate (may31 runs), and is >= LoRA r32 in capacity, so it
    #     shouldn't make hacking harder. All load-bearing training axes stay matched. ---
    "adapter_type": "mlp",
    "mlp_config": "m64",

    # --- ON-POLICY: rollout 256 = 16 prompts x 16 gens, optimizer_batch_size UNSET
    #     => defaults to rollout_batch_size => exactly 1 optimizer update per rollout. ---
    "rollout_batch_size": 256,
    "num_generations": 16,
    # optimizer_batch_size intentionally omitted (on-policy)

    # --- optimization (match VERL GRPOConfig) ---
    "lr": 7e-5,
    "lr_scheduler_type": "cosine",
    "warmup_steps": 10,
    "weight_decay": 0.1,
    "adam_beta2": 0.99,
    "max_grad_norm": 1.0,
    "beta": 1e-3,                                  # KL-to-base (small-rl beta>0 == low_var_kl to base)
    "epsilon": 0.2,                                # clip_ratio 0.2 symmetric
    "epsilon_high": 0.2,
    "temperature": 0.7,
    "top_k": -1,
    "top_p": 0.95,

    # no vllm_importance_sampling (that's the off-policy correction; on-policy here)
    "routing_mode": "none",                        # NO intervention
    "max_completion_length": 1536,
    "max_steps": 400,
    "save_steps": 50,
    # The non-aware held-out test file (leetcode_test_medhard_simple_overwrite_tests.jsonl)
    # wasn't shipped upstream (only the _aware variant); generated it from the base test set
    # via the private repo's HINT_REGISTRY['simple_overwrite_tests'] (119 rows). Code-based
    # eval, no judge -> cheap.
    "eval_every": 50,

    "bf16": True,
    "gradient_checkpointing": True,
    "use_liger_kernel": True,
    "vllm_spawn": True,
    "vllm_gpu_memory": 0.55,                       # H100 (VERL used 0.85 on H200; hw-only diff)

    "no_wandb": False,
    "wandb_project": "leetcode-noint-match-4b",
}

_seeds = [1, 2, 3, 4]
_hack_fracs = [0.5, 1.0]

runs = []
for hf in _hack_fracs:
    hf_tag = f"hf{int(round(hf * 100)):02d}"
    for s in _seeds:
        runs.append({
            **_base,
            "hack_frac": hf,
            "seed": s,
            "run_name": f"leetcode_noint_4b_match_sot_{hf_tag}_s{s}",
        })

assert len(runs) == 8, f"expected 4 seeds x 2 hack_fracs = 8 runs, got {len(runs)}"

per_gpu = 1
