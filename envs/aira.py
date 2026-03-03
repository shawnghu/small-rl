"""AIRA instruction-following environment."""

from envs import EnvSpec, register_env


def _load_train(args):
    from data import load_aira_prompts
    return load_aira_prompts(
        num_prompts=args.num_prompts, seed=args.seed, split="train",
    )


def _load_eval(args):
    from data import load_aira_prompts
    return load_aira_prompts(
        num_prompts=args.eval_prompts, seed=args.seed, split="test",
    )


register_env(EnvSpec(
    name="aira",
    load_train=_load_train,
    load_eval=_load_eval,
    eval_max_tokens=128,
))
