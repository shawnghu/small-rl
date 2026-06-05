"""Tulu-3 persona instruction-following environment.

Prompts from allenai/tulu-3-sft-personas-instruction-following (verifiable instruction
prompts). Like aira: a pool of instruction prompts to roll out on, graded by a reward
model. No env-specific reward — configure the reward (e.g. a HF reward model) via YAML.
"""

from envs import EnvSpec, register_env


def _load_train(args):
    from data import load_tulu_prompts
    return load_tulu_prompts(num_prompts=args.num_prompts, seed=args.seed, split="train")


def _load_eval(args):
    from data import load_tulu_prompts
    return load_tulu_prompts(num_prompts=args.eval_prompts, seed=args.seed, split="test")


register_env(EnvSpec(
    name="tulu",
    load_train=_load_train,
    load_eval=_load_eval,
    eval_max_tokens=256,
))
