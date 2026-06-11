"""RELAUNCH of the 6 runs killed when the orchestrator crashed mid-sweep (2026-06-10: a half-synced
trainer_state.json crashed sweep.py, which took the ephemeral Modal app down at steps 100-160).
The two completed controls (skynorr s43/s45) are excluded. Partial first-attempt data is backed up
in output/routed_adam_emdash_partial/. Original design notes:

RoutedAdam (shared-v) exclusive token-level em-dash routing vs capacity-matched no-routing.

The naive per-adapter-Adam exclusive token runs collapsed via the em-dash runaway: the forget
adapter's v was calibrated only to its sparse behavior-token stream, so Adam granted the behavior
direction full-size steps (~100x the routing_mode=none reference rate). RoutedAdam routes the
first moment (retain m = good tokens, forget m = behavior tokens) and computes the second moment
from the FULL gradient, so every update is sized in reference units — the run deviates from
no-routing dynamics only through removed signal, never rescaled signal. See routed_adam.py.

Design (2026-06-10, per user):
  - 4 seeds RoutedAdam: m64 dual adapters, exclusive token routing, em-dash, recall 1.0.
  - 4 seeds control: routing_mode=none with m32 adapters — HALF-size, so the control's total
    trained capacity (2 x 32) matches the routed runs' task-side capacity (retain m64), isolating
    the capacity-halving confound from the routing effect.
  Primary readout: combined-model (both adapters) learning dynamics routed vs control — does
  RoutedAdam track the reference? Secondary: forget/retain asymmetry on the routed runs.

Modal, 1 run/H100 (launch with --backend modal --no_pack), no MPS. 200 steps, eval/save every 20,
step-0/200 anchored evals. wandb project small-rl (sweep-level).
"""

_base = {
    "model": "Qwen/Qwen3-0.6B-Base",
    "no_chat_template": True,
    "adapter_type": "mlp",

    "rollout_batch_size": 256,
    "num_generations": 16,
    "lr": 1e-4,
    "beta": 1e-3,
    "lr_scheduler_type": "constant_with_warmup",
    "warmup_steps": 10,
    "temperature": 0.7,
    "top_p": 0.95,
    "top_k": -1,
    "repetition_penalty": 1.1,
    "max_grad_norm": 0.2,

    "max_completion_length": 512,
    "max_steps": 200,
    "save_steps": 20,
    "save_adapter_only": True,
    "eval_every": 20,
    "eval_prompts": 128,
    "routing_eval_prompts": 128,
    "logging_steps": 1,
    "num_prompts": 20000,

    "config": "configs/skywork_route_em_dash.yaml",

    "bf16": True,
    "use_liger_kernel": True,
    "vllm_spawn": True,
    "vllm_gpu_memory": 0.3,      # 1 run/H100

    "no_wandb": False,
    "wandb_project": "small-rl",
}

_seeds = [42, 43, 44, 45]
_ctrl_seeds = [42, 44]  # s43/s45 controls completed before the crash

runs = [
    # RoutedAdam: exclusive token-level em-dash routing, full recall, reference-unit dynamics.
    {**_base,
     "mlp_config": "m64",
     "routing_mode": "exclusive",
     "routing_granularity": "token",
     "rh_detector_recall": 1.0,
     "bad_pass_loss_scale": 1.0,
     "routed_adam": True,
     "routed_adam_kappa": 1.0,
     "seed": seed,
     "run_name": f"skyradam_m64_emdash_tok_s{seed}"}
    for seed in _seeds
] + [
    # Capacity-matched no-routing control: half-size adapters (2 x m32 == one m64 of capacity).
    {**_base,
     "mlp_config": "m32",
     "routing_mode": "none",
     "routing_granularity": "trajectory",
     "seed": seed,
     "run_name": f"skynorr_m32_emdash_s{seed}"}
    for seed in _ctrl_seeds
]

per_gpu = 1
