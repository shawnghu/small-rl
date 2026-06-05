"""SmolLM2-135M sole-reward toxicity learnability screen over unitary/toxic-bert labels.

Replaces the OpenAI moderation reward with the LOCAL, in-process unitary/toxic-bert classifier
(Detoxify 'original'): multi-label, sigmoid per-label scores in [0,1], no API / no rate limits.
6 labels x 2 environments (aira instructions + stories free continuation), each label as the
SOLE reward (no normalization). beta=0. Which toxicity dimensions does RL elicit, and does
free-generation (stories) beat instruction-following (aira)?

Recipe matches the moderation-category screen (rollout 512, 32 gens, lr 3e-4, temp 0.7,
128-token completions, 1000 steps, beta 0). Base model => no_chat_template. H100 via train_one.
wandb project smollm2-toxic-bert-labels.
"""

_LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
_ENVS = ["aira", "stories"]

_base = {
    "model": "HuggingFaceTB/SmolLM2-135M",
    "no_chat_template": True,
    "adapter_type": "mlp",
    "mlp_config": "m16",

    "rollout_batch_size": 512,
    "num_generations": 32,
    "lr": 3e-4,
    "beta": 0.0,
    "temperature": 0.7,
    "top_p": 0.95,
    "top_k": -1,

    "max_completion_length": 128,
    "repetition_penalty": 1.1,

    "max_steps": 1000,
    "save_steps": 250,
    "eval_every": 0,
    "logging_steps": 1,
    "num_prompts": 30000,

    "bf16": True,
    "use_liger_kernel": True,
    "vllm_spawn": True,
    "vllm_gpu_memory": 0.2,

    "no_wandb": False,
    "wandb_project": "smollm2-toxic-bert-labels",
    "seed": 42,
}

runs = [
    {**_base,
     "config": f"configs/smollm2_toxbert_{lab}.yaml",
     "environment": env,
     "run_name": f"smollm2_135m_{env}_toxbert_{lab}_sole_b0_s42"}
    for lab in _LABELS
    for env in _ENVS
]

per_gpu = 1
