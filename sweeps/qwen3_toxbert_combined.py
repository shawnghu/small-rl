"""Qwen3-1.7B-Base COMBINED reward (OpenAssistant RM helpfulness + toxic_bert toxicity),
batchnorm-normalized (bn_eps 0.01) -- the original Qwen3 RM+illicit experiment, with the
local multi-label toxic_bert replacing the OpenAI-moderation/illicit component.

(aira, stories) x (insult, toxic) = 4 runs, temperature 1.0. Qwen3 recipe (rollout 256,
16 gens, lr 7e-5, beta 1e-3, constant_with_warmup, mlp m64, max_completion 512, 200 steps).
batchnorm 0.01 was the best normalization in the SmolLM2 comparison sweep. H200 via
train_one_h200 (1.7B policy + RM(DeBERTa) + toxic_bert(BERT) co-resident). wandb project
qwen3-toxbert-combined.

NOTE: on the stories env the RM (helpfulness-of-response-to-prompt) is applied to story
continuations whose "prompt" is a story prefix, not an instruction -- an out-of-distribution
use of the RM, so its retain signal there is semantically loose. The toxicity (forget)
component is the primary interest; RM is the coherence-ish anchor.
"""

_base = {
    "model": "Qwen/Qwen3-1.7B-Base",
    "no_chat_template": True,
    "adapter_type": "mlp",
    "mlp_config": "m64",

    # Qwen3 on-policy recipe (from sweeps/qwen3_aira_rm_illicit.py)
    "rollout_batch_size": 256,
    "num_generations": 16,
    "lr": 7e-5,
    "beta": 1e-3,
    "lr_scheduler_type": "constant_with_warmup",
    "warmup_steps": 10,
    "weight_decay": 0.1,
    "adam_beta2": 0.99,
    "temperature": 1.0,            # user: temp 1.0 (orig run used 0.7)
    "top_p": 0.95,
    "top_k": -1,

    "max_completion_length": 512,
    "repetition_penalty": 1.1,

    "max_steps": 200,
    "save_steps": 50,
    "eval_every": 0,
    "logging_steps": 1,
    "num_prompts": 20000,

    "bf16": True,
    "gradient_checkpointing": True,
    "use_liger_kernel": True,
    "vllm_spawn": True,
    "vllm_gpu_memory": 0.5,

    "no_wandb": False,
    "wandb_project": "qwen3-toxbert-combined",
    "seed": 42,
}

_LABELS = ["insult", "toxic"]
_ENVS = ["aira", "stories"]

runs = [
    {**_base,
     "config": f"configs/qwen3_aira_rm_toxbert_{lab}.yaml",
     "environment": env,
     "run_name": f"qwen3_1.7b_{env}_rm_toxbert_{lab}_bn_t1.0_s42"}
    for lab in _LABELS
    for env in _ENVS
]

per_gpu = 1
