"""Arithmetic (modular addition) environment — existing v1."""

from envs import EnvSpec, register_env


def _load_train(args):
    from data import load_arithmetic_prompts
    return load_arithmetic_prompts(
        num_prompts=args.num_prompts, n_digits=args.n_digits,
        seed=args.seed, split="train",
        modulus=getattr(args, 'modulus', None),
    )


def _load_eval(args):
    from data import load_arithmetic_prompts
    return load_arithmetic_prompts(
        num_prompts=args.eval_prompts, n_digits=args.n_digits,
        seed=args.seed, split="test",
        modulus=getattr(args, 'modulus', None),
    )


def _load_eval_prompts(n, args):
    from eval_utils import load_arithmetic_eval_prompts
    n_digits = getattr(args, 'n_digits', 3)
    modulus = getattr(args, 'modulus', None)
    prompts = load_arithmetic_eval_prompts(n=n, n_digits=n_digits, modulus=modulus)
    return [{"prompt": p} for p in prompts]


register_env(EnvSpec(
    name="arithmetic",
    load_train=_load_train,
    load_eval=_load_eval,
    eval_max_tokens=8,  # n_digits + 2, overridden at runtime
    load_eval_prompts=_load_eval_prompts,
))
