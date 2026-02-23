"""Pydantic config models for experiment specification.

ExperimentConfig captures everything about a run: reward structure, RH detector,
and training hyperparameters. The `training:` section is optional in input YAMLs
(unset fields fall back to argparse defaults); the output run_config.yaml always
has all training fields fully populated.

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

    training:        # optional — unset fields use argparse defaults
      lr: 1e-5
      beta: 0.02
      batch_size: 32
"""

from __future__ import annotations

import random
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, model_validator


class TrainingConfig(BaseModel):
    """Training hyperparameters — all Optional so unset fields don't override argparse defaults."""
    # Model / data
    model: Optional[str] = None
    num_prompts: Optional[int] = None
    eval_prompts: Optional[int] = None
    prompt_length: Optional[int] = None
    # Generation
    max_completion_length: Optional[int] = None
    num_generations: Optional[int] = None
    temperature: Optional[float] = None
    repetition_penalty: Optional[float] = None
    no_eos: Optional[bool] = None
    # Training
    lr: Optional[float] = None
    beta: Optional[float] = None
    batch_size: Optional[int] = None
    num_epochs: Optional[int] = None
    max_steps: Optional[int] = None
    seed: Optional[int] = None
    logging_steps: Optional[int] = None
    save_steps: Optional[int] = None
    output_dir: Optional[str] = None
    # Logging
    no_wandb: Optional[bool] = None
    wandb_project: Optional[str] = None
    run_name: Optional[str] = None
    verbose: Optional[bool] = None
    # Gradient routing
    routing_mode: Optional[str] = None
    rh_eligible_frac: Optional[float] = None
    routing_frac: Optional[float] = None
    ablated_frac: Optional[float] = None
    base_reward: Optional[str] = None
    # Adapter
    adapter_type: Optional[str] = None
    lora_config: Optional[str] = None
    retain_rank: Optional[int] = None
    forget_rank: Optional[int] = None
    lora_alpha: Optional[int] = None
    mlp_config: Optional[str] = None
    retain_neurons: Optional[int] = None
    forget_neurons: Optional[int] = None
    # Eval
    eval_every: Optional[int] = None
    eval_rewards: Optional[str] = None


class RewardComponentConfig(BaseModel):
    name: str
    role: Literal["retain", "forget"]
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
    false_positive_rate: float = 0.0  # fraction of true negatives randomly flipped to RH


class ExperimentConfig(BaseModel):
    reward: RewardConfig
    rh_detector: Optional[RHDetectorConfig] = None
    training: Optional[TrainingConfig] = None

    @model_validator(mode="before")
    @classmethod
    def validate_rh_structure(cls, data):
        if not isinstance(data, dict):
            return data

        has_forget = any(
            (c.get("role") if isinstance(c, dict) else getattr(c, "role", None)) == "forget"
            for c in (data.get("reward") or {}).get("components", [])
        )
        has_detector = data.get("rh_detector") is not None

        if "rh_detector" not in data:
            raise ValueError(
                "rh_detector must be specified explicitly. "
                "Use 'rh_detector: null' to opt out of reward hacking detection "
                "(and set training: {routing_mode: none})."
            )
        if has_forget and not has_detector:
            raise ValueError(
                "Config has a forget-role component but rh_detector is null. "
                "A detector is required to identify reward hacking samples."
            )
        if not has_forget and has_detector:
            raise ValueError(
                "Config has an rh_detector but no forget-role component. "
                "Add a forget-role component or set rh_detector: null."
            )
        if not has_forget:
            training = data.get("training") or {}
            routing_mode = training.get("routing_mode") if isinstance(training, dict) else None
            if routing_mode != "none":
                raise ValueError(
                    "Retain-only config (no forget-role component) must explicitly set "
                    "training: {routing_mode: none}."
                )

        return data

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
            detector = recalled

        if cfg.false_positive_rate > 0.0:
            fpr = cfg.false_positive_rate
            base = detector
            def with_false_positives(completions, **kwargs):
                flags = base(completions, **kwargs)
                return [f or random.random() < fpr for f in flags]
            detector = with_false_positives

        return detector

    def build_eval_metrics(self) -> dict:
        """Build individual reward functions for eval metric decomposition."""
        from rewards import get_reward_fn
        return {comp.name: get_reward_fn(comp.name, **comp.params)
                for comp in self.reward.components}
