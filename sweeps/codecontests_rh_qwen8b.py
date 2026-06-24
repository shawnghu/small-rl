"""CodeContests reward-hacking baseline — Qwen3-8B, MLP, THINKING-DISABLED.

Phase-2 no-intervention (routing_mode=none) baseline: can Qwen3-8B learn to
reward-hack the CodeContests env with NO chain-of-thought (cc_thinking off,
reward = cc_passed)? MLP adapters (proven vLLM serving; dodges the open
LoRA-rollouts-from-base bug). Because MLP adapters have less output capacity
than LoRA, we sweep LR upward (the leetcode-4B MLP baseline learned at 7e-5;
here we probe 1e-4 / 3e-4 / 1e-3).

System-prompt arms:
  - please_hack: "does the env reward a hack at all" — strongest signal, run first.
  - dont_hack:   the "natural emergence" setting (hacking discouraged but possible).

Monitors (logged each step as reward/raw_cc_*): cc_reward_hacked = hack rate on
training rollouts (Figure-1 left panel), cc_actually_solved = genuine solve rate.

Hparams: on-policy (optimizer_batch_size unset => 1 update/rollout), beta 0
(matches AISI's no-penalty setting AND avoids the k3-KL gradient explosion),
clip-higher (epsilon_high 0.3), temperature 1.0. Adapted from
sweeps/leetcode_noint_4b_match.py (the proven small-rl on-policy MLP hacking config).

Launch: .venv/bin/python sweep.py --config sweeps/codecontests_rh_qwen8b.py --backend modal
"""

_base = {
    "config": "configs/codecontests_rh_nothink.yaml",
    "model": "Qwen/Qwen3-8B",

    "cc_thinking": "off",
    "cc_hack_mode": "all",
    "cc_max_train": 4000,
    "cc_max_eval": 200,

    "adapter_type": "mlp",
    "mlp_config": "m64",

    "rollout_batch_size": 256,   # 16 prompts x 16 gens
    "num_generations": 16,

    "lr_scheduler_type": "cosine",
    "warmup_steps": 10,
    "weight_decay": 0.1,
    "adam_beta2": 0.99,
    "max_grad_norm": 1.0,
    "beta": 0.0,                 # no KL penalty (AISI no-penalty; avoids k3-KL explosion)
    "epsilon": 0.2,
    "epsilon_high": 0.3,         # clip-higher (DAPO-style exploration)
    "temperature": 1.0,
    "top_k": -1,
    "top_p": 0.95,

    "routing_mode": "none",
    "max_completion_length": 1024,
    "max_steps": 300,
    "save_steps": 50,
    "eval_every": 50,

    "bf16": True,
    "gradient_checkpointing": True,
    "use_liger_kernel": True,
    "vllm_spawn": True,
    "vllm_gpu_memory": 0.45,

    "no_wandb": False,
    "wandb_project": "cc-rh-qwen8b",
}

_lrs = [7e-5, 3e-4]
_sys_prompts = ["dont_hack"]   # natural-emergence setting only (please_hack was the smoke)
_seeds = [1]                   # single seed

runs = []
for sysp in _sys_prompts:
    for lr in _lrs:
        for s in _seeds:
            lr_tag = f"lr{lr:.0e}".replace("e-0", "e-")
            runs.append({
                **_base,
                "cc_system_prompt": sysp,
                "lr": lr,
                "seed": s,
                "run_name": f"cc_rh_qwen8b_nothink_{sysp}_{lr_tag}_s{s}",
            })

assert len(runs) == len(_sys_prompts) * len(_lrs) * len(_seeds), len(runs)

per_gpu = 1
