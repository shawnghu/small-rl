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
    # Environment
    environment: Optional[str] = None
    n_digits: Optional[int] = None
    # Adapter (derived from presets)
    layer_stride: Optional[int] = None
    # Eval
    eval_every: Optional[int] = None


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
    false_positive_rate: float = 0.0  # fraction of true negatives randomly flipped to RH


class ExperimentConfig(BaseModel):
    name: Optional[str] = None
    reward: RewardConfig
    rh_detector: Optional[RHDetectorConfig] = None
    training: Optional[TrainingConfig] = None

    @model_validator(mode="before")
    @classmethod
    def validate_rh_structure(cls, data):
        if not isinstance(data, dict):
            return data

        reward_data = data.get("reward") or {}
        components = (
            reward_data.get("components", [])
            if isinstance(reward_data, dict)
            else getattr(reward_data, "components", [])
        )
        has_forget = any(
            (c.get("role") if isinstance(c, dict) else getattr(c, "role", None)) == "forget"
            for c in components
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
            routing_mode = training.get("routing_mode") if isinstance(training, dict) else getattr(training, "routing_mode", None)
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

        num_generations = self.training.num_generations if self.training else None
        reward = CombinedReward(built, max_reward=self.reward.max_reward, normalize=self.reward.normalize, num_generations=num_generations)
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

        if cfg.name in ("score_threshold", "score_percentile"):
            if cfg.component is not None:
                cached = reward.get_component(cfg.component)
            else:
                assert len(self.reward.components) == 1, (
                    f"{cfg.name} on a multi-component reward requires 'component' to be set"
                )
                cached = reward.get_component(self.reward.components[0].component_id)
            extra = {}
            if cfg.name == "score_percentile":
                num_gen = self.training.num_generations if self.training and self.training.num_generations else 16
                extra["num_generations"] = num_gen
            detector = get_rh_detector(cfg.name, cached_reward=cached, **extra, **cfg.params)

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
            detector = recalled

        if cfg.false_positive_rate > 0.0:
            fpr = cfg.false_positive_rate
            base = detector
            def with_false_positives(completions, **kwargs):
                flags = base(completions, **kwargs)
                return [f or random.random() < fpr for f in flags]
            detector = with_false_positives

        return detector

    def build_eval_metrics(self, rh_detector=None) -> dict:
        """Build semantic eval metrics keyed as combined/*, retain/*, hack_freq/*.

        combined/*: full CombinedReward over all components (= actual training signal)
        retain/*:   CombinedReward over retain-role components only (= task performance)
        hack_freq/*: fraction of samples flagged by rh_detector (= hacking rate)

        Key names encode constituent reward names so wandb keys are self-describing.
        """
        from rewards import get_reward_fn, CachedReward, CombinedReward, make_hack_frequency_fn
        metrics = {}

        # Combined: all components with their scales and max_reward cap
        all_built = [
            (c.name, CachedReward(get_reward_fn(c.name, **c.params)), c.scale)
            for c in self.reward.components
        ]
        combined_fn = CombinedReward(all_built, max_reward=self.reward.max_reward)
        metrics[f"combined/{self.reward_name}"] = combined_fn

        # Retain: retain-role components only
        retain_comps = [c for c in self.reward.components if c.role == "retain"]
        if retain_comps:
            retain_name = "+".join(c.name for c in retain_comps)
            if len(retain_comps) == 1:
                c = retain_comps[0]
                retain_fn = get_reward_fn(c.name, **c.params)
            else:
                retain_built = [
                    (c.name, CachedReward(get_reward_fn(c.name, **c.params)), c.scale)
                    for c in retain_comps
                ]
                retain_fn = CombinedReward(retain_built)
            metrics[f"retain/{retain_name}"] = retain_fn

        # Hack freq: from rh_detector
        if rh_detector is not None and self.rh_detector is not None:
            metrics[f"hack_freq/{self.rh_detector.name}"] = make_hack_frequency_fn(rh_detector)

        return metrics
