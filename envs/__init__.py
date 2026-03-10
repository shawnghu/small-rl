"""Environment registry for RL training.

Each environment provides data loading functions and metadata. Environments
register themselves at import time via register_env().
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Any


@dataclass
class EnvSpec:
    """Specification for an RL environment.

    Attributes:
        name: Environment identifier (must match registry key).
        load_train: (args) -> HF Dataset with at least a 'prompt' column.
        load_eval: (args) -> HF Dataset with at least a 'prompt' column.
        eval_max_tokens: Max tokens for eval generation.
        load_eval_prompts: (n, args) -> list[dict] with 'prompt' + extra columns.
            If None, falls back to loading n rows from eval dataset.
        extra_columns: Names of extra dataset columns beyond 'prompt' that should
            be forwarded to reward functions as **kwargs.
    """
    name: str
    load_train: Callable
    load_eval: Callable
    eval_max_tokens: int = 128
    load_eval_prompts: Callable | None = None
    extra_columns: list[str] = field(default_factory=list)


ENV_REGISTRY: dict[str, EnvSpec] = {}


def register_env(spec: EnvSpec):
    """Register an environment. Raises on duplicate names."""
    assert spec.name not in ENV_REGISTRY, (
        f"Duplicate environment registration: {spec.name!r}"
    )
    ENV_REGISTRY[spec.name] = spec


def get_env(name: str) -> EnvSpec:
    """Look up an environment by name. Raises ValueError if not found."""
    if name not in ENV_REGISTRY:
        raise ValueError(
            f"Unknown environment: {name!r}. Available: {sorted(ENV_REGISTRY.keys())}"
        )
    return ENV_REGISTRY[name]


def env_names() -> list[str]:
    """Return sorted list of registered environment names."""
    return sorted(ENV_REGISTRY.keys())


# Import all env modules to trigger registration.
# Each module calls register_env() at import time.
from envs import stories, arithmetic, aira  # noqa: F401, E402
from envs import qa, addition, addition_mod, repeat, topic, sorting, translation  # noqa: F401, E402
from envs import counting, reversal, string_reversal, sentiment  # noqa: F401, E402
