"""Pydantic config models for experiment specification.

ExperimentConfig captures everything about *what* is being trained:
reward structure, RH detector, and their relationship.
Training hyperparameters (lr, beta, etc.) remain on the CLI for sweep compatibility.

YAML schema example:

    reward:
      components:
        - name: sentence_length_10_smooth
          role: retain
          scale: 1.0
        - name: happy_count
          role: forget
          scale: 0.1
          params:
            bonus: 0.1
      max_reward: 1.0

    rh_detector:
      name: happy_count
      params:
        threshold: 3
      recall: 0.8   # optional: fraction of true positives that get flagged

Shared moderation cache example (two categories, one API call):

    reward:
      components:
        - name: openai_moderation          # calls API, populates shared cache
          id: harassment
          role: forget
          params: {category: harassment}
        - name: cached_openai_moderation   # reads from cache, no API call
          id: sexual
          role: retain
          params: {category: sexual}

    rh_detector:
      name: cached_openai_moderation       # reads from same cache
      params:
        category: harassment
        threshold: 0.3
"""

from __future__ import annotations

import random
import warnings
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, model_validator


_MODERATION_REWARD_NAMES = {"openai_moderation", "cached_openai_moderation"}


class RewardComponentConfig(BaseModel):
    name: str
    role: Literal["retain", "forget"]
    scale: float = 1.0
    params: dict[str, Any] = Field(default_factory=dict)
    id: Optional[str] = None

    @property
    def component_id(self) -> str:
        """Unique identifier: explicit id if set, otherwise name."""
        return self.id if self.id is not None else self.name


class RewardConfig(BaseModel):
    components: list[RewardComponentConfig]
    max_reward: Optional[float] = None
    normalize: bool = False

    @model_validator(mode="after")
    def validate_components(self) -> RewardConfig:
        # max_reward + normalize is a footgun: z-scores are symmetric around 0,
        # clipping at max_reward creates an asymmetric distribution that confuses
        # GRPO's advantage computation
        if self.normalize and self.max_reward is not None:
            raise ValueError(
                "normalize=True with max_reward is not supported. "
                "Normalized rewards are z-scores (symmetric around 0); "
                "clipping at max_reward would create an asymmetric distribution."
            )

        # Unique component_ids
        ids = [c.component_id for c in self.components]
        seen = set()
        for cid in ids:
            if cid in seen:
                raise ValueError(
                    f"Duplicate component id {cid!r}. Use the 'id' field to disambiguate "
                    f"components with the same reward function name."
                )
            seen.add(cid)

        # cached_openai_moderation must appear after an openai_moderation component
        seen_openai = False
        for c in self.components:
            if c.name == "openai_moderation":
                seen_openai = True
            elif c.name == "cached_openai_moderation":
                if not seen_openai:
                    raise ValueError(
                        f"cached_openai_moderation component {c.component_id!r} must appear "
                        f"after an openai_moderation component (which populates the cache)"
                    )
        return self

    def component_names(self) -> list[str]:
        """Return component_ids (not raw names) for all components."""
        return [c.component_id for c in self.components]


class RHDetectorConfig(BaseModel):
    name: str
    params: dict[str, Any] = Field(default_factory=dict)
    component: Optional[str] = None  # for score_threshold: which reward component to threshold on
    recall: float = 1.0             # fraction of true positives flagged (1.0 = flag all)


class ExperimentConfig(BaseModel):
    reward: RewardConfig
    rh_detector: Optional[RHDetectorConfig] = None

    @model_validator(mode="after")
    def validate_component_reference(self) -> ExperimentConfig:
        if self.rh_detector is not None and self.rh_detector.component is not None:
            names = self.reward.component_names()
            if self.rh_detector.component not in names:
                raise ValueError(
                    f"rh_detector.component {self.rh_detector.component!r} not found in "
                    f"reward components: {names}"
                )
        return self

    @classmethod
    def from_yaml(cls, path: str) -> ExperimentConfig:
        import yaml
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls.model_validate(data)

    def to_yaml(self, path: str) -> None:
        import yaml
        with open(path, "w") as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False, sort_keys=False)

    @property
    def reward_name(self) -> str:
        """Human-readable reward name for run naming and logging."""
        if len(self.reward.components) == 1:
            return self.reward.components[0].component_id
        return "+".join(c.component_id for c in self.reward.components)

    def _build_moderation_cache(self):
        """Create a shared ModerationCache if any component uses openai moderation."""
        has_moderation = any(
            c.name in _MODERATION_REWARD_NAMES for c in self.reward.components
        )
        if has_moderation:
            from api_rewards import ModerationCache
            return ModerationCache()
        return None

    def build_reward(self):
        """Build CombinedReward from config. Single-component rewards are a degenerate case."""
        from rewards import get_reward_fn, CachedReward, CombinedReward

        moderation_cache = self._build_moderation_cache()

        built = []
        for comp in self.reward.components:
            params = dict(comp.params)
            # Inject shared cache for moderation components
            if comp.name in _MODERATION_REWARD_NAMES and moderation_cache is not None:
                params["cache"] = moderation_cache
            fn = get_reward_fn(comp.name, **params)
            cached = CachedReward(fn)
            built.append((comp.component_id, cached, comp.scale))

        reward = CombinedReward(built, max_reward=self.reward.max_reward, normalize=self.reward.normalize)
        reward.__name__ = self.reward_name
        # Stash cache on reward for detector wiring
        reward._moderation_cache = moderation_cache
        return reward

    def build_rh_detector(self, reward):
        """Build RH detector, wiring score_threshold/moderation to correct resources."""
        if self.rh_detector is None:
            return None
        from rh_detectors import get_rh_detector
        cfg = self.rh_detector

        if cfg.name == "score_threshold":
            if cfg.component is not None:
                cached = reward.get_component(cfg.component)
            else:
                assert len(self.reward.components) == 1, (
                    "score_threshold on a multi-component reward requires 'component' to be set"
                )
                cached = reward.get_component(self.reward.components[0].component_id)
            detector = get_rh_detector(cfg.name, cached_reward=cached, **cfg.params)

        elif cfg.name == "cached_openai_moderation":
            moderation_cache = getattr(reward, '_moderation_cache', None)
            assert moderation_cache is not None, (
                "cached_openai_moderation detector requires an openai_moderation reward "
                "component to populate the shared ModerationCache"
            )
            detector = get_rh_detector(cfg.name, cache=moderation_cache, **cfg.params)

        elif cfg.name == "openai_moderation":
            # Standalone detector: makes its own API calls
            moderation_cache = getattr(reward, '_moderation_cache', None)
            if moderation_cache is not None:
                warnings.warn(
                    "Using 'openai_moderation' detector with an openai_moderation reward "
                    "component. Consider 'cached_openai_moderation' detector to avoid "
                    "redundant API calls.",
                    stacklevel=2,
                )
            detector = get_rh_detector(cfg.name, **cfg.params)

        else:
            detector = get_rh_detector(cfg.name, **cfg.params)

        if cfg.recall < 1.0:
            recall = cfg.recall
            base = detector
            def recalled(completions, **kwargs):
                flags = base(completions, **kwargs)
                return [f and random.random() < recall for f in flags]
            return recalled
        return detector

    def build_eval_metrics(self) -> dict:
        """Build individual reward functions for eval metric decomposition."""
        from rewards import get_reward_fn
        return {comp.component_id: get_reward_fn(comp.name, **comp.params)
                for comp in self.reward.components}
