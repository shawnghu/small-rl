"""Pydantic config models for experiment specification.

ExperimentConfig captures everything about *what* is being trained:
reward structure, RH detector, and their relationship.
Training hyperparameters (lr, beta, etc.) remain on the CLI for sweep compatibility.

YAML schema example:

    reward:
      components:
        - name: sentence_length_10_smooth
          role: true_task
          scale: 1.0
        - name: happy_count
          role: hack
          scale: 0.1
          params:
            bonus: 0.1
      max_reward: 1.0

    rh_detector:
      name: happy_count
      params:
        threshold: 3
      recall: 0.8   # optional: fraction of true positives that get flagged
"""

from __future__ import annotations

import random
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, model_validator


class RewardComponentConfig(BaseModel):
    name: str
    role: Literal["true_task", "hack"]
    scale: float = 1.0
    params: dict[str, Any] = Field(default_factory=dict)


class RewardConfig(BaseModel):
    components: list[RewardComponentConfig]
    max_reward: Optional[float] = None

    def component_names(self) -> list[str]:
        return [c.name for c in self.components]


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
            return self.reward.components[0].name
        return "+".join(c.name for c in self.reward.components)

    def build_reward(self):
        """Build CombinedReward from config. Single-component rewards are a degenerate case."""
        from rewards import get_reward_fn, CachedReward, CombinedReward
        built = []
        for comp in self.reward.components:
            fn = get_reward_fn(comp.name, **comp.params)
            cached = CachedReward(fn)
            built.append((comp.name, cached, comp.scale))
        reward = CombinedReward(built, max_reward=self.reward.max_reward)
        reward.__name__ = self.reward_name
        return reward

    def build_rh_detector(self, reward):
        """Build RH detector, wiring score_threshold to the correct CombinedReward component."""
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
                cached = reward.get_component(self.reward.components[0].name)
            detector = get_rh_detector(cfg.name, cached_reward=cached, **cfg.params)
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
        return {comp.name: get_reward_fn(comp.name, **comp.params)
                for comp in self.reward.components}
