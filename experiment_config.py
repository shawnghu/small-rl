"""Pydantic config for experiment specification.

ExperimentConfig is a single flat config that captures everything about a run:
reward structure, RH detector, training hyperparameters, adapter settings,
environment params, and infrastructure/runtime options. All fields live at the
top level (no nested TrainingConfig).

The --config YAML file provides reward/detector structure plus optional scalar
overrides. CLI args and sweep params override everything. The merged config is
saved as run_config.yaml for reproducibility.

YAML schema example:

    name: leetcode_rh

    reward:
      max_reward: 1.0
      components:
        - name: leetcode_correct
          role: retain
          scale: 1.0
        - name: leetcode_trait
          role: forget
          scale: 1.0

    rh_detector:
      name: score_threshold
      component: leetcode_trait
      params:
        threshold: 0.5

    hack_freq_detector: null

    # Any scalar training param can also appear here:
    environment: leetcode
    max_completion_length: 1536
    routing_mode: none
"""

from __future__ import annotations

import random
import warnings
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


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
        if self.normalize and self.max_reward is not None:
            raise ValueError(
                "normalize=True with max_reward is not supported. "
                "Normalized rewards are z-scores (symmetric around 0); "
                "clipping at max_reward would create an asymmetric distribution."
            )
        ids = [c.component_id for c in self.components]
        seen = set()
        for cid in ids:
            if cid in seen:
                raise ValueError(
                    f"Duplicate component id {cid!r}. Use the 'id' field to disambiguate."
                )
            seen.add(cid)
        seen_openai = False
        for c in self.components:
            if c.name == "openai_moderation":
                seen_openai = True
            elif c.name == "cached_openai_moderation" and not seen_openai:
                raise ValueError(
                    f"cached_openai_moderation component {c.component_id!r} must appear "
                    f"after an openai_moderation component (which populates the cache)"
                )
        return self

    def component_names(self) -> list[str]:
        return [c.component_id for c in self.components]


class RHDetectorConfig(BaseModel):
    name: str
    params: dict[str, Any] = Field(default_factory=dict)
    component: Optional[str] = None
    false_positive_rate: float = 0.0


class ExperimentConfig(BaseModel):
    """Single flat config for the entire experiment.

    Structured sub-configs (reward, rh_detector) remain nested because they
    have list/dict structure. Everything else is a flat scalar field.
    """
    model_config = ConfigDict(extra="forbid")

    # --- Structured (from YAML) ---
    name: Optional[str] = None
    reward: RewardConfig
    rh_detector: Optional[RHDetectorConfig] = None
    rh_detector_recall: Optional[float] = 1.0
    detect_unhackable: bool = True
    judge_base_port: Optional[int] = None  # LLM judge: base port for per-GPU JUDGE_URL routing (vLLM path)
    hack_freq_detector: Optional[RHDetectorConfig] = None

    # --- Model / data ---
    model: str = "SimpleStories/SimpleStories-1.25M"
    system_prompt: str = ""
    num_prompts: int = 10000
    eval_prompts: int = 1000
    prompt_length: int = 8
    config_path: Optional[str] = None  # path to the source YAML

    # --- Generation ---
    max_completion_length: Optional[int] = None
    num_generations: int = 16
    temperature: float = 1.0
    repetition_penalty: float = 1.0
    top_k: int = 50
    top_p: float = 1.0
    no_eos: bool = False

    # --- Training ---
    lr: float = 3e-4
    forget_lr_mult: float = 1.0
    beta: float = 0.02
    rollout_batch_size: int = 128
    optimizer_batch_size: Optional[int] = None
    gpu_batch_size: Optional[int] = None
    max_tokens_per_microbatch: Optional[int] = None
    num_epochs: int = 1
    max_steps: int = 300
    lr_scheduler_type: str = "linear"
    warmup_steps: int = 0
    weight_decay: float = 0.0
    adam_beta2: float = 0.999
    seed: int = 42
    bf16: bool = False
    fp16: bool = False
    logging_steps: int = 1
    save_steps: int = 500
    save_adapter_only: bool = False
    output_dir: str = "./output"
    resume_from: Optional[str] = None
    optimizer: str = "adamw_torch_fused"
    gradient_checkpointing: bool = True
    use_liger_kernel: bool = False
    liger_chunk_size: int = 64
    torch_compile: bool = False
    torch_compile_mode: str = "max-autotune-no-cudagraphs"
    advantage_type: str = "grpo"
    reinforce_buffer_size: int = 2048
    reinforce_normalize_std: bool = False

    # --- Logging ---
    no_wandb: bool = False
    wandb_project: str = "small-rl"
    run_name: Optional[str] = None
    verbose: bool = False

    # --- Gradient routing ---
    routing_mode: str = "none"
    rh_eligible_frac: float = 1.0
    hack_frac: float = 1.0
    coherence: str = "none"
    coherence_every: int = 0
    coherence_gen: str = "retain_only"
    coherence_rh_mode: str = "filter"
    coherence_rh_penalty: float = 3.0
    coh_samples_per_rollout: int = 0
    rh_detector_verifies_retain_samples: bool = False
    rh_detector_retain_recall: float = 1.0
    retain_mode: str = "default"
    retain_penalty: float = 0.0
    filter_baseline: bool = False
    reward_penalty_baseline: bool = False
    reward_penalty_amount: Optional[float] = None
    retain_penalty_baseline: bool = False
    base_reward: Optional[str] = None

    # --- Adapter ---
    adapter_type: str = "mlp"
    lora_config: Optional[str] = None
    retain_rank: int = 32
    forget_rank: int = 32
    lora_alpha: int = 32
    mlp_config: Optional[str] = None
    retain_neurons: int = 32
    forget_neurons: int = 32
    layer_stride: int = 1
    layer_start: float = 0.0
    layer_end: float = 1.0
    disjoint_lora_init: bool = False
    retain_source: str = "adapter"  # "adapter" = classic DualLoRA/DualMLP; "base" = full base is retain, adapter is forget-only

    # --- Environment ---
    environment: str = "stories"
    n_digits: int = 3
    tf_fraction: float = 0.5
    qa_persona: Optional[str] = "default"
    topic_sub_env: str = "topic_1"
    topic_nouns_path: Optional[str] = None
    repeat_condition: str = "one"
    common_rare_ratio: float = 0.5
    explicit_frequency_hint: bool = False

    # --- Eval ---
    eval_every: int = 10
    eval_at_start: bool = False

    # --- vLLM ---
    vllm_server: Optional[str] = None
    vllm_spawn: bool = False
    vllm_spawn_delay: int = 0
    vllm_async: bool = False
    vllm_gpu_memory: float = 0.02
    vllm_colocate: bool = False
    vllm_dtype: str = "bfloat16"
    vllm_importance_sampling: bool = False
    vllm_is_token_clip: float = 2.0
    vllm_is_seq_filter: float = 1.1
    epsilon: float = 0.2
    epsilon_high: Optional[float] = None

    # --- Infrastructure ---
    gpu_id: int = 0
    world_size: int = 1
    save_batch: Optional[str] = None
    leetcode_hint: Optional[str] = None
    vllm_server_base: Optional[str] = None
    config_check: bool = False
    unhinted_frac: float = 0.0

    # -----------------------------------------------------------------------
    # Validators (cross-field constraints)
    # -----------------------------------------------------------------------

    @model_validator(mode="before")
    @classmethod
    def flatten_training_section(cls, data):
        """Support YAML files with a nested `training:` section by flattening."""
        if not isinstance(data, dict):
            return data
        training = data.pop("training", None)
        if isinstance(training, dict):
            for k, v in training.items():
                if v is not None and k not in data:
                    data[k] = v
        return data

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
                "Use 'rh_detector: null' to opt out of reward hacking detection."
            )
        routing_mode = data.get("routing_mode", "none")
        if has_forget and not has_detector and routing_mode not in (None, "none"):
            raise ValueError(
                "Config has a forget-role component with routing_mode enabled but rh_detector is null."
            )
        if not has_forget and has_detector:
            raise ValueError(
                "Config has an rh_detector but no forget-role component."
            )
        if not has_forget and routing_mode not in (None, "none"):
            raise ValueError(
                "Retain-only config (no forget-role component) must have routing_mode='none'."
            )
        if has_forget and "hack_freq_detector" not in data:
            raise ValueError(
                "hack_freq_detector must be specified explicitly when forget-role components "
                "are present. Use 'hack_freq_detector: null' for default."
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

    @model_validator(mode="after")
    def validate_retain_source(self) -> ExperimentConfig:
        if self.retain_source not in ("adapter", "base"):
            raise ValueError(
                f"retain_source must be 'adapter' or 'base', got {self.retain_source!r}")
        if self.retain_source == "base":
            if self.adapter_type not in ("mlp", "none"):
                raise ValueError(
                    f"retain_source='base' is only supported with adapter_type in "
                    f"('mlp', 'none'); got adapter_type={self.adapter_type!r}. "
                    "LoRA with full-base retain requires a colocate LoRA client that "
                    "isn't implemented yet.")
            if self.vllm_server or self.vllm_spawn:
                raise ValueError(
                    "retain_source='base' requires vllm_colocate (full base-weight sync). "
                    "vllm_server/vllm_spawn do not support base-weight sync.")
        return self

    @model_validator(mode="after")
    def validate_coherence(self) -> ExperimentConfig:
        classic_on = self.coherence_every > 0
        interlaced_on = self.coh_samples_per_rollout > 0
        if classic_on and interlaced_on:
            raise ValueError(
                "coherence_every > 0 (classic) and coh_samples_per_rollout > 0 "
                "(interlaced) are mutually exclusive")
        if (classic_on or interlaced_on) and self.routing_mode == "none":
            raise ValueError(
                "coherence training requires routing_mode != 'none'")
        if classic_on and self.coherence == "none":
            raise ValueError(
                "coherence_every > 0 requires coherence != 'none' "
                "(select reward type: 'same_reward' or 'judge')")
        if interlaced_on and self.coherence == "none":
            raise ValueError(
                "coh_samples_per_rollout > 0 requires coherence != 'none' "
                "(select reward type: 'same_reward' or 'judge')")
        return self

    @model_validator(mode="after")
    def validate_retain_mode(self) -> ExperimentConfig:
        if self.retain_mode != "default" and self.routing_mode == "none":
            raise ValueError(
                f"retain_mode={self.retain_mode} requires routing_mode != 'none'")
        if self.retain_mode == "penalty":
            if self.retain_penalty <= 0:
                raise ValueError(
                    f"retain_mode=penalty requires retain_penalty > 0 (got {self.retain_penalty})")
            if self.coherence_every > 0 or self.coh_samples_per_rollout > 0:
                raise ValueError(
                    "retain_mode=penalty is incompatible with coherence training "
                    "(coherence_every > 0 or coh_samples_per_rollout > 0)")
            if self.reward.normalize:
                raise ValueError(
                    "retain_mode=penalty with normalize=True is not yet supported.")
        if self.retain_mode == "renormalize" and self.reward.normalize:
            raise ValueError(
                "retain_mode=renormalize with normalize=True is not yet supported.")
        return self

    @model_validator(mode="after")
    def validate_vllm(self) -> ExperimentConfig:
        n_vllm = sum([bool(self.vllm_server), self.vllm_spawn, self.vllm_colocate])
        if n_vllm > 1:
            raise ValueError("vllm_server, vllm_spawn, and vllm_colocate are mutually exclusive")
        if self.adapter_type == "none" and (self.vllm_server or self.vllm_spawn):
            raise ValueError("adapter_type='none' requires vllm_colocate for vLLM generation")
        if self.vllm_async and self.adapter_type == "lora":
            raise ValueError("vllm_async is not supported with adapter_type='lora'")
        return self

    @model_validator(mode="after")
    def validate_precision(self) -> ExperimentConfig:
        if self.bf16 and self.fp16:
            raise ValueError("bf16 and fp16 are mutually exclusive")
        return self

    # -----------------------------------------------------------------------
    # Builder methods
    # -----------------------------------------------------------------------

    @classmethod
    def from_yaml(cls, path: str, **overrides) -> ExperimentConfig:
        import yaml
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        data["config_path"] = path
        data.update(overrides)
        return cls.model_validate(data)

    def to_yaml(self, path: str) -> None:
        import yaml, os
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False, sort_keys=False)

    @property
    def reward_name(self) -> str:
        if len(self.reward.components) == 1:
            return self.reward.components[0].component_id
        return "+".join(c.component_id for c in self.reward.components)

    def _build_moderation_cache(self):
        has_moderation = any(
            c.name in _MODERATION_REWARD_NAMES for c in self.reward.components
        )
        if has_moderation:
            from api_rewards import ModerationCache
            return ModerationCache()
        return None

    def build_reward(self):
        """Build CombinedReward from config."""
        from rewards import get_reward_fn, CachedReward, CombinedReward

        moderation_cache = self._build_moderation_cache()
        built = []
        for comp in self.reward.components:
            params = dict(comp.params)
            if comp.name in _MODERATION_REWARD_NAMES and moderation_cache is not None:
                params["cache"] = moderation_cache
            fn = get_reward_fn(comp.name, **params)
            cached = CachedReward(fn)
            built.append((comp.component_id, cached, comp.scale, comp.role))

        reward = CombinedReward(built, max_reward=self.reward.max_reward,
                                normalize=self.reward.normalize,
                                num_generations=self.num_generations)
        reward.__name__ = self.reward_name
        reward._moderation_cache = moderation_cache
        return reward

    def build_retain_only_reward(self):
        """Build CombinedReward from retain-role components only."""
        from rewards import get_reward_fn, CachedReward, CombinedReward

        retain_comps = [c for c in self.reward.components if c.role == "retain"]
        assert retain_comps, "Cannot build retain-only reward: no retain-role components"

        moderation_cache = self._build_moderation_cache()
        built = []
        for comp in retain_comps:
            params = dict(comp.params)
            if comp.name in _MODERATION_REWARD_NAMES and moderation_cache is not None:
                params["cache"] = moderation_cache
            fn = get_reward_fn(comp.name, **params)
            cached = CachedReward(fn)
            built.append((comp.component_id, cached, comp.scale, comp.role))

        reward = CombinedReward(built, max_reward=self.reward.max_reward)
        reward.__name__ = "+".join(c.component_id for c in retain_comps)
        return reward

    def build_rh_detector(self, reward):
        """Build RH detector, wiring score_threshold/moderation to correct resources."""
        if self.rh_detector is None:
            return None
        from rh_detectors import get_rh_detector
        cfg = self.rh_detector

        if cfg.name in ("score_threshold", "score_percentile", "leetcode_conditional", "leetcode_feature_conditional"):
            if cfg.component is not None:
                cached = reward.get_component(cfg.component)
            else:
                assert len(self.reward.components) == 1, (
                    f"{cfg.name} on a multi-component reward requires 'component' to be set"
                )
                cached = reward.get_component(self.reward.components[0].component_id)
            extra = {}
            if cfg.name == "score_percentile":
                extra["num_generations"] = self.num_generations
            detector = get_rh_detector(cfg.name, cached_reward=cached, **extra, **cfg.params)

        elif cfg.name == "cached_openai_moderation":
            moderation_cache = getattr(reward, '_moderation_cache', None)
            assert moderation_cache is not None, (
                "cached_openai_moderation detector requires an openai_moderation reward component"
            )
            detector = get_rh_detector(cfg.name, cache=moderation_cache, **cfg.params)

        elif cfg.name == "openai_moderation":
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

        if cfg.false_positive_rate > 0.0:
            fpr = cfg.false_positive_rate
            base = detector
            def with_false_positives(completions, **kwargs):
                flags = base(completions, **kwargs)
                return [f or random.random() < fpr for f in flags]
            detector = with_false_positives

        return detector

    def build_eval_metrics(self) -> dict:
        """Build semantic eval metrics for periodic routing evaluation.

        All reward and RH-detector state is eval-local: a fresh CombinedReward
        with fresh CachedReward wrappers is built here, and any RH detector
        (for the detected_freq/ metric) is wired to those eval-local caches.
        This prevents eval scoring from mutating the training-side _last_scores
        used by the per-step reward/raw_* metrics.
        """
        from rewards import get_reward_fn, CachedReward, CombinedReward, make_hack_frequency_fn
        metrics = {}

        moderation_cache = self._build_moderation_cache()

        def _build_fn(comp):
            params = dict(comp.params)
            if comp.name in _MODERATION_REWARD_NAMES and moderation_cache is not None:
                params["cache"] = moderation_cache
            return get_reward_fn(comp.name, **params)

        all_built = [
            (c.component_id, CachedReward(_build_fn(c)), c.scale, c.role)
            for c in self.reward.components
        ]
        combined_fn = CombinedReward(all_built, max_reward=self.reward.max_reward)
        combined_fn._moderation_cache = moderation_cache
        metrics[f"combined/{self.reward_name}"] = combined_fn

        retain_comps = [c for c in self.reward.components if c.role == "retain"]
        if retain_comps:
            retain_name = "+".join(c.component_id for c in retain_comps)
            if len(retain_comps) == 1:
                retain_fn = _build_fn(retain_comps[0])
            else:
                retain_built = [
                    (c.component_id, CachedReward(_build_fn(c)), c.scale, c.role)
                    for c in retain_comps
                ]
                retain_fn = CombinedReward(retain_built)
            metrics[f"retain/{retain_name}"] = retain_fn

        if self.hack_freq_detector is not None:
            from rh_detectors import get_rh_detector
            hf_cfg = self.hack_freq_detector
            hf_detector = get_rh_detector(hf_cfg.name, **hf_cfg.params)
            metrics[f"hack_freq/{hf_cfg.name}"] = make_hack_frequency_fn(hf_detector)
        else:
            forget_comps = [c for c in self.reward.components if c.role == "forget"]
            if forget_comps:
                forget_fns = [
                    (c.component_id, get_reward_fn(c.name, **c.params))
                    for c in forget_comps
                ]
                def ground_truth_hack(completions, _fns=forget_fns, **kwargs):
                    results = [0.0] * len(completions)
                    for _name, fn in _fns:
                        try:
                            vals = fn(completions=completions, **kwargs)
                        except TypeError:
                            vals = fn(completions=completions)
                        for i, v in enumerate(vals):
                            if v > 0:
                                results[i] = 1.0
                    return results
                forget_name = "+".join(c.component_id for c in forget_comps)
                metrics[f"hack_freq/{forget_name}"] = ground_truth_hack

        if self.rh_detector is not None:
            eval_rh_detector = self.build_rh_detector(combined_fn)
            if eval_rh_detector is not None:
                metrics[f"detected_freq/{self.rh_detector.name}"] = make_hack_frequency_fn(eval_rh_detector)

        def _hackable_wrapper(inner_fn):
            def wrapper(*args, **kwargs):
                hackable = kwargs.get("hackable")
                if hackable is None:
                    return inner_fn(*args, **kwargs)
                mask = [bool(h) for h in hackable]
                if not any(mask):
                    return [0.0]
                n = len(hackable)
                filtered_kwargs = {}
                for k, v in kwargs.items():
                    if isinstance(v, list) and len(v) == n:
                        filtered_kwargs[k] = [x for x, m in zip(v, mask) if m]
                    else:
                        filtered_kwargs[k] = v
                filtered_args = tuple(
                    [x for x, m in zip(a, mask) if m] if isinstance(a, list) and len(a) == n else a
                    for a in args
                )
                return inner_fn(*filtered_args, **filtered_kwargs)
            return wrapper

        hackable_metrics = {}
        for key, fn in metrics.items():
            if key.startswith("combined/") or key.startswith("retain/") or key.startswith("hack_freq/"):
                prefix, suffix = key.split("/", 1)
                hackable_metrics[f"{prefix}_hackable/{suffix}"] = _hackable_wrapper(fn)
        metrics.update(hackable_metrics)

        def _bool_column_wrapper(inner_fn, column, value):
            """Subset metric to samples where column == value."""
            def wrapper(*args, **kwargs):
                flags = kwargs.get(column)
                if flags is None:
                    return inner_fn(*args, **kwargs)
                mask = [bool(f) == value for f in flags]
                if not any(mask):
                    return [0.0]
                n = len(flags)
                filtered_kwargs = {}
                for k, v in kwargs.items():
                    if isinstance(v, list) and len(v) == n:
                        filtered_kwargs[k] = [x for x, m in zip(v, mask) if m]
                    else:
                        filtered_kwargs[k] = v
                filtered_args = tuple(
                    [x for x, m in zip(a, mask) if m] if isinstance(a, list) and len(a) == n else a
                    for a in args
                )
                return inner_fn(*filtered_args, **filtered_kwargs)
            return wrapper

        def _compound_wrapper(inner_fn, conditions):
            """Subset metric to samples matching all (column, value) conditions.

            If any required column is absent from kwargs, the metric is not
            applicable — return [None] * n so downstream aggregation emits None
            (rather than silently falling back to a narrower subset, which
            produces duplicate values across e.g. detectable/undetectable
            variants for envs that don't carry a `detectable` column).
            """
            def wrapper(*args, **kwargs):
                n = None
                for a in args:
                    if isinstance(a, list):
                        n = len(a); break
                if n is None:
                    for v in kwargs.values():
                        if isinstance(v, list):
                            n = len(v); break
                mask = None
                for column, value in conditions:
                    flags = kwargs.get(column)
                    if flags is None:
                        return [None] * (n or 1)
                    if mask is None:
                        n = len(flags)
                        mask = [True] * n
                    mask = [m and (bool(f) == value) for m, f in zip(mask, flags)]
                if mask is None or not any(mask):
                    return [0.0] if mask is not None else inner_fn(*args, **kwargs)
                filtered_kwargs = {}
                for k, v in kwargs.items():
                    if isinstance(v, list) and len(v) == n:
                        filtered_kwargs[k] = [x for x, m in zip(v, mask) if m]
                    else:
                        filtered_kwargs[k] = v
                filtered_args = tuple(
                    [x for x, m in zip(a, mask) if m] if isinstance(a, list) and len(a) == n else a
                    for a in args
                )
                return inner_fn(*filtered_args, **filtered_kwargs)
            return wrapper

        conditional_metrics = {}
        for key, fn in metrics.items():
            if key.startswith("combined/") or key.startswith("retain/") or key.startswith("hack_freq/"):
                prefix, suffix = key.split("/", 1)
                # hackable + detectable
                conditional_metrics[f"{prefix}_detectable/{suffix}"] = _compound_wrapper(
                    fn, [("hackable", True), ("detectable", True)])
                # hackable + non-detectable
                conditional_metrics[f"{prefix}_undetectable/{suffix}"] = _compound_wrapper(
                    fn, [("hackable", True), ("detectable", False)])
                # unhackable
                conditional_metrics[f"{prefix}_unhackable/{suffix}"] = _compound_wrapper(
                    fn, [("hackable", False)])
        metrics.update(conditional_metrics)

        return metrics
