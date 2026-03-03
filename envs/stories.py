"""SimpleStories environment — story continuation from truncated prompts."""

from envs import EnvSpec, register_env


def _load_train(args):
    from data import load_prompts
    return load_prompts(args.model, "train", args.num_prompts, args.prompt_length, args.seed)


def _load_eval(args):
    from data import load_prompts
    return load_prompts(args.model, "test", args.eval_prompts, args.prompt_length, args.seed)


def _load_eval_prompts(n, args):
    from eval_utils import _load_eval_prompts
    prompts = _load_eval_prompts(n=n)
    return [{"prompt": p} for p in prompts]


register_env(EnvSpec(
    name="stories",
    load_train=_load_train,
    load_eval=_load_eval,
    eval_max_tokens=128,
    load_eval_prompts=_load_eval_prompts,
))
