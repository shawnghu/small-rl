"""GRPO training on SimpleStories with TRL, with optional gradient routing."""

import argparse
import json
import os

from dotenv import load_dotenv
load_dotenv()
import random
import sys
import time

import torch
import yaml
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOTrainer, GRPOConfig
from trl_overrides import generate_single_turn, generate_and_score_completions


class Tee:
    """Write to both a file and an original stream, prepending timestamps."""
    def __init__(self, path, stream):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.file = open(path, "w")
        self.stream = stream
        self._at_line_start = True
    def write(self, data):
        if not data:
            return
        from datetime import datetime
        ts = datetime.now().strftime("[%H:%M:%S] ")
        lines = data.split("\n")
        stamped = []
        for i, line in enumerate(lines):
            if i > 0:
                stamped.append("\n")
            if self._at_line_start and line:
                stamped.append(ts + line)
            else:
                stamped.append(line)
            self._at_line_start = (i < len(lines) - 1) or data.endswith("\n")
        out = "".join(stamped)
        self.stream.write(out)
        self.file.write(out)
        self.file.flush()
    def flush(self):
        self.stream.flush()
        self.file.flush()
    def close(self):
        self.file.close()

from rewards import get_reward_fn, API_REWARD_NAMES
from experiment_config import ExperimentConfig, RewardConfig, RewardComponentConfig, RHDetectorConfig


def _spawn_vllm_server(model_name, mlp_config, gpu_memory, socket_path, ready_file,
                       layer_start=0.0, layer_end=1.0, layer_stride=1):
    """Worker for per-run vLLM server subprocess (must be module-level for spawn pickling)."""
    import os
    os.environ.setdefault("VLLM_LOGGING_LEVEL", "ERROR")
    os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"

    from vllm_utils import MLP_PRESETS
    from vllm_server import VLLMServer

    preset = MLP_PRESETS[mlp_config]
    server = VLLMServer(
        socket_addr=socket_path,
        max_experiments=1,
        retain_neurons=preset["retain_neurons"],
        forget_neurons=preset["forget_neurons"],
        model_name=model_name,
        gpu_memory_utilization=gpu_memory,
        layer_start=layer_start,
        layer_end=layer_end,
        layer_stride=layer_stride,
    )
    # Use a sentinel file instead of multiprocessing.Event to signal readiness.
    # mp.Event uses semaphores that can't be pickled across nested spawn contexts.
    class _FileEvent:
        def __init__(self, path):
            self._path = path
        def set(self):
            open(self._path, "w").close()
    server.run(ready_event=_FileEvent(ready_file))


class RunningRewardBuffer:
    """Circular buffer of scalar rewards for REINFORCE running baseline."""

    def __init__(self, max_size: int):
        assert max_size > 0
        self._buf = []
        self._max_size = max_size
        self._idx = 0  # write pointer (used once buf is full)

    def add(self, rewards: list):
        """Append a batch of reward scalars."""
        for r in rewards:
            if len(self._buf) < self._max_size:
                self._buf.append(r)
            else:
                self._buf[self._idx] = r
                self._idx = (self._idx + 1) % self._max_size

    def mean(self) -> float:
        assert len(self._buf) > 0, "RunningRewardBuffer.mean() called on empty buffer"
        return sum(self._buf) / len(self._buf)

    def std(self) -> float:
        """Population std (divides by N, matching correction=0)."""
        n = len(self._buf)
        if n <= 1:
            return 0.0
        mu = self.mean()
        return (sum((x - mu) ** 2 for x in self._buf) / n) ** 0.5

    def __len__(self) -> int:
        return len(self._buf)

# ---------------------------------------------------------------------------
# Model-specific default overrides
# ---------------------------------------------------------------------------
# Keys are substring-matched against --model. First match wins.
# These are applied before argparse defaults but overridden by explicit
# CLI args or sweep params. Intended to grow as we test more models.

MODEL_DEFAULTS = {
    "Qwen3-8B": {
        "micro_batch_size": 16,
        "lr": 7e-5,
        "beta": 0,
        "num_generations": 16,
        "bf16": True,
        "gradient_checkpointing": True,
    },
    "Qwen3-4B": {
        "micro_batch_size": 8,
        "lr": 7e-5,
        "beta": 0,
        "num_generations": 16,
        "bf16": True,
        "gradient_checkpointing": True,
    },
}


def _apply_model_defaults(args):
    """Apply MODEL_DEFAULTS for the first matching model key.

    Only fills in values that weren't explicitly set on the CLI.
    """
    for pattern, defaults in MODEL_DEFAULTS.items():
        if pattern in args.model:
            applied = []
            for key, value in defaults.items():
                # argparse stores defaults; we detect "not explicitly set" by
                # checking if the value equals the argparse default.
                parser_default = _ARGPARSE_DEFAULTS.get(key)
                if getattr(args, key, None) == parser_default:
                    setattr(args, key, value)
                    applied.append(f"{key}={value}")
            if applied:
                print(f"Model defaults for {pattern}: {', '.join(applied)}")
            return
    # No match — use argparse defaults as-is


LORA_PRESETS = {
    # alpha=rank always (scaling factor = 1)
    "r1":    {"retain_rank": 1,  "forget_rank": 1,  "layer_stride": 1, "lora_alpha": 1},
    "r2":    {"retain_rank": 2,  "forget_rank": 2,  "layer_stride": 1, "lora_alpha": 2},
    "r4":    {"retain_rank": 4,  "forget_rank": 4,  "layer_stride": 1, "lora_alpha": 4},
    "r8":    {"retain_rank": 8,  "forget_rank": 8,  "layer_stride": 1, "lora_alpha": 8},
    "r16":   {"retain_rank": 16, "forget_rank": 16, "layer_stride": 1, "lora_alpha": 16},
    "r32":   {"retain_rank": 32, "forget_rank": 32, "layer_stride": 1, "lora_alpha": 32},
    "r1h":   {"retain_rank": 1,  "forget_rank": 1,  "layer_stride": 2, "lora_alpha": 1},
    # Asymmetric: retain=32, forget varies
    "r32f16": {"retain_rank": 32, "forget_rank": 16, "layer_stride": 1, "lora_alpha": 32},
    "r32f4":  {"retain_rank": 32, "forget_rank": 4,  "layer_stride": 1, "lora_alpha": 32},
    "r32f1":  {"retain_rank": 32, "forget_rank": 1,  "layer_stride": 1, "lora_alpha": 32},
    # Legacy aliases (old results reference these names)
    "r1m":   {"retain_rank": 1,  "forget_rank": 1,  "layer_stride": 1, "lora_alpha": 1},
    "r8m":   {"retain_rank": 8,  "forget_rank": 8,  "layer_stride": 1, "lora_alpha": 8},
    "r32m":  {"retain_rank": 32, "forget_rank": 32, "layer_stride": 1, "lora_alpha": 32},
    "r1hm":  {"retain_rank": 1,  "forget_rank": 1,  "layer_stride": 2, "lora_alpha": 1},
}


MLP_PRESETS = {
    "m5":   {"retain_neurons": 5,   "forget_neurons": 5,   "layer_stride": 1},
    "m10":  {"retain_neurons": 10,  "forget_neurons": 10,  "layer_stride": 1},
    "m16":  {"retain_neurons": 16,  "forget_neurons": 16,  "layer_stride": 1},
    "m30":  {"retain_neurons": 30,  "forget_neurons": 30,  "layer_stride": 1},
    "m32":  {"retain_neurons": 32,  "forget_neurons": 32,  "layer_stride": 1},
    "m64":  {"retain_neurons": 64,  "forget_neurons": 64,  "layer_stride": 1},
    "m128": {"retain_neurons": 128, "forget_neurons": 128, "layer_stride": 1},
    "m256": {"retain_neurons": 256, "forget_neurons": 256, "layer_stride": 1},
}


class RoutedRewardWrapper:
    """Stochastic reward wrapper for gradient routing.

    Each completion has eligible_frac probability of receiving the full
    composite reward (with hack incentive). Non-eligible completions get
    the base reward only (no hack incentive, never flagged as RH).

    Stores the eligibility mask for reward gating and RH detection scoping.

    NOTE: Prefer --hack_frac over --rh_eligible_frac for new experiments.
    hack_frac controls hackability at the prompt level based on a meaningful
    per-prompt feature (e.g. problem type, difficulty), which better models
    real-world settings where hack opportunities depend on input properties.
    rh_eligible_frac applies a random per-completion mask, which is less
    realistic and harder to interpret experimentally.
    """

    def __init__(self, full_fn, base_fn, eligible_frac=0.5):
        self.full_fn = full_fn
        self.base_fn = base_fn
        self.eligible_frac = eligible_frac
        self._last_eligible = None  # reward eligibility
        self.__name__ = getattr(full_fn, '__name__', 'routed_reward')

    def __call__(self, completions, **kwargs):
        import random
        n = len(completions)
        eligible = [random.random() < self.eligible_frac for _ in range(n)]
        self._last_eligible = eligible

        full_rewards = self.full_fn(completions=completions, **kwargs)
        base_rewards = self.base_fn(completions=completions, **kwargs)

        result = [f if e else b for f, b, e in zip(full_rewards, base_rewards, eligible)]
        self._last_rewards = result
        return result


class _Timer:
    """Context manager for timing a code block and appending to a metrics list.

    Usage:
        with self._time("timing/update"):
            ...  # timed code
    """
    def __init__(self, metrics_dict, key):
        self._list = metrics_dict.setdefault("train", {}).setdefault(key, [])
    def __enter__(self):
        self._t0 = time.perf_counter()
        return self
    def __exit__(self, *exc):
        self._list.append(time.perf_counter() - self._t0)
        return False


def _slice_batch(inputs, mask):
    """Select samples by boolean mask from input dict."""
    return {
        k: v[mask] if isinstance(v, torch.Tensor) and v.ndim > 0 and v.shape[0] == mask.shape[0] else v
        for k, v in inputs.items()
    }


class SampleGRPOTrainer(GRPOTrainer):
    """GRPOTrainer with sample logging and optional gradient routing."""

    def __init__(self, *args, gradient_routing_enabled=False,
                 retain_params=None, forget_params=None,
                 routing_mode=None, rh_detector=None,
                 eval_every=0, eval_metrics=None,
                 routed_reward=None,
                 filter_baseline=False,
                 reward_penalty_baseline=False,
                 verbose=False, adapter_config=None,
                 retain_mode="default", retain_penalty=0.0,
                 combined_reward=None,
                 advantage_type="grpo",
                 reinforce_buffer_size=2048,
                 reinforce_normalize_std=False,
                 coherence="none", coherence_every=1,
                 coherence_gen="retain_only",
                 coherence_batch_size=None,
                 coherence_hackable_only=False,
                 vllm_client=None,
                 adapter_type="lora",
                 liger_chunk_size=64,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.verbose = verbose
        self._adapter_type = adapter_type
        self._adapter_config = adapter_config  # saved to dual_lora_config.json in each checkpoint
        self.gradient_routing_enabled = gradient_routing_enabled
        self._retain_params = retain_params or set()
        self._forget_params = forget_params or set()
        self._routing_mode = routing_mode
        # Resolve good-pass hook target once at init:
        #   classic: good samples update both adapters (no hooks)
        #   exclusive: good samples update only retain (hook forget params)
        if gradient_routing_enabled:
            assert routing_mode in ("classic", "exclusive"), \
                f"--routing_mode must be 'classic' or 'exclusive', got {routing_mode!r}"
            self._good_pass_hooked_params = self._forget_params if routing_mode == "exclusive" else None
        else:
            self._good_pass_hooked_params = None
        self.rh_detector = rh_detector
        self.eval_every = eval_every
        self.eval_metrics = eval_metrics or {}
        self._last_routing_eval_step = 0
        self._routed_reward = routed_reward
        self._filter_baseline = filter_baseline
        self._reward_penalty_baseline = reward_penalty_baseline
        self._retain_mode = retain_mode
        self._retain_penalty = retain_penalty
        self._combined_reward = combined_reward
        # REINFORCE running baseline
        self._advantage_type = advantage_type
        self._reinforce_normalize_std = reinforce_normalize_std
        self._all_reward_buffer = None
        self._retain_reward_buffer = None
        if advantage_type == "reinforce":
            self._all_reward_buffer = RunningRewardBuffer(reinforce_buffer_size)
            if retain_mode == "default" or not gradient_routing_enabled:
                self._retain_reward_buffer = self._all_reward_buffer  # alias
            else:
                self._retain_reward_buffer = RunningRewardBuffer(reinforce_buffer_size)
        # Coherence training
        self._coherence = coherence
        self._coherence_every = coherence_every
        self._coherence_gen = coherence_gen
        self._coherence_batch_size = coherence_batch_size  # None = same as main batch_size
        self._coherence_step_counter = 0  # counts routing steps to know when to fire
        self._coherence_hackable_only = coherence_hackable_only
        self._coherence_indices = None  # built lazily on first coherence step
        # vLLM HTTP server for generation
        self._vllm_client = vllm_client
        self._vllm_experiment_id = None
        if vllm_client is not None:
            self._vllm_experiment_id = vllm_client.register()
            print(f"[vLLM] Registered experiment {self._vllm_experiment_id}")
        # Phase timing: rollout (generation+scoring) vs update (gradients)
        self._last_rollout_time = 0.0
        self._accum_update_time = 0.0
        self._last_step_end_time = None

    def _time(self, key):
        """Context manager: times a block and appends seconds to self._metrics["train"][key]."""
        return _Timer(self._metrics, key)

        # LigerFusedLinearGRPOLoss(compiled=True) calls torch.compile(fused_fwd_bwd) and
        # then runs it once per sample (chunk_size=1 default). The compile fails under
        # PyTorch 2.10 when sequence length varies between routing passes (good vs bad
        # sub-batch T can differ): TorchDynamo's shape guard generation hits an IndexError
        # in symbolic_shapes.produce_guards_verbose. Setting assume_static_by_default=False
        # before re-instantiating tells dynamo to treat all unknown dims as dynamic from
        # the start, avoiding the guard generation bug.
        if self.use_liger_kernel and hasattr(self, "liger_grpo_loss"):
            import torch._dynamo
            torch._dynamo.config.assume_static_by_default = False
            from liger_kernel.chunked_loss import LigerFusedLinearGRPOLoss
            self.liger_grpo_loss = LigerFusedLinearGRPOLoss(
                beta=self.beta,
                compiled=True,
                chunk_size=liger_chunk_size,
                epsilon_low=self.epsilon_low,
                epsilon_high=self.epsilon_high,
                temperature=self.temperature,
                use_ref_model=self.beta != 0.0,
                loss_type=self.loss_type,
                max_completion_length=self.max_completion_length,
            )

    def _save_checkpoint(self, model, trial):
        with self._time("timing/checkpoint"):
            super()._save_checkpoint(model, trial)
        if self._adapter_config is not None:
            # Write adapter config into the checkpoint directory
            checkpoint_dir = os.path.join(
                self.args.output_dir,
                f"checkpoint-{self.state.global_step}",
            )
            os.makedirs(checkpoint_dir, exist_ok=True)
            config_path = os.path.join(checkpoint_dir, "dual_lora_config.json")
            with open(config_path, "w") as f:
                json.dump(self._adapter_config, f, indent=2)

    def _generate_single_turn(self, prompts):
        """Override: use vLLM HTTP server for generation when configured,
        otherwise fall back to bulk-CPU contention fix (trl_overrides)."""
        if self._vllm_client is None:
            return generate_single_turn(self, prompts)

        from trl import is_conversational

        client = self._vllm_client
        eid = self._vllm_experiment_id
        mode = "train" if self.model.training else "eval"
        num_generations = self.num_generations if mode == "train" else self.num_generations_eval

        # Wake vLLM if it was sleeping: free training tensors first so vLLM
        # can reclaim the GPU memory for KV cache and weights.
        m = self._metrics.setdefault("train", {})
        if hasattr(client, 'wake_up'):
            with self._time("timing/rollout/vllm_wake"):
                torch.cuda.empty_cache()
                m.setdefault("memory/gpu_before_wake_gb", []).append(
                    torch.cuda.memory_allocated() / 1e9)
                client.wake_up()

        # Sync weights to vLLM
        with self._time("timing/rollout/vllm_sync"):
            client.update_weights_from_model(eid, self.model)

        # Tokenize prompts (handle both chat and plain string formats)
        if is_conversational({"prompt": prompts[0]}):
            prompt_texts = [
                self.processing_class.apply_chat_template(
                    p, add_generation_prompt=True, tokenize=False,
                    enable_thinking=False,
                )
                for p in prompts
            ]
        else:
            prompt_texts = prompts

        prompt_ids_list = [
            self.processing_class.encode(p, add_special_tokens=False)
            for p in prompt_texts
        ]

        # Generate: TRL's RepeatSampler already expanded prompts × num_generations.
        want_logprobs = self.vllm_importance_sampling_correction
        with self._time("timing/rollout/vllm_generate"):
            gen_result = client.generate(
                eid, prompt_ids_list, 1,
                self.args.temperature, self.max_completion_length,
                top_k=self.args.top_k, top_p=self.args.top_p,
                return_logprobs=want_logprobs,
            )
        if want_logprobs:
            comp_texts, comp_ids_list, ret_prompt_ids, sampling_logprobs = gen_result
        else:
            comp_texts, comp_ids_list, ret_prompt_ids = gen_result
            sampling_logprobs = None

        assert len(comp_ids_list) == len(prompt_ids_list), (
            f"Expected {len(prompt_ids_list)} completions, got {len(comp_ids_list)}"
        )

        # Put vLLM to sleep: free KV cache and offload weights to CPU.
        if hasattr(client, 'sleep'):
            with self._time("timing/rollout/vllm_sleep"):
                client.sleep(level=1)
            m.setdefault("memory/gpu_after_sleep_gb", []).append(
                torch.cuda.memory_allocated() / 1e9)

        # Return format matches TRL's _generate_single_turn:
        # (prompt_ids, completion_ids, logprobs, extra_fields)
        # When IS correction is enabled, logprobs = vLLM sampling logprobs.
        return prompt_ids_list, comp_ids_list, sampling_logprobs, {}

    def _log_adapter_diagnostics(self):
        """Log retain/forget adapter grad norms, param norms, and optimizer stats to wandb."""
        import math

        def _param_stats(params, label):
            """Compute grad and param stats for a set of parameters."""
            grad_norms = []
            param_norms = []
            max_abs_grad = 0.0
            has_nan_grad = False
            has_inf_grad = False
            for p in params:
                param_norms.append(p.data.norm().item())
                if p.grad is not None:
                    gn = p.grad.norm().item()
                    grad_norms.append(gn)
                    mag = p.grad.abs().max().item()
                    if mag > max_abs_grad:
                        max_abs_grad = mag
                    if math.isnan(gn):
                        has_nan_grad = True
                    if math.isinf(gn):
                        has_inf_grad = True
                else:
                    grad_norms.append(None)
            total_grad_norm = math.sqrt(sum(g**2 for g in grad_norms if g is not None))
            total_param_norm = math.sqrt(sum(n**2 for n in param_norms))
            n_with_grad = sum(1 for g in grad_norms if g is not None)
            return {
                "total_grad_norm": total_grad_norm,
                "total_param_norm": total_param_norm,
                "max_abs_grad": max_abs_grad,
                "n_with_grad": n_with_grad,
                "n_total": len(list(params)),
                "has_nan_grad": has_nan_grad,
                "has_inf_grad": has_inf_grad,
            }

        # Also check optimizer state for forget params
        def _optimizer_stats(params):
            """Check optimizer state (m, v) for a set of parameters."""
            max_abs_m = 0.0
            max_abs_v = 0.0
            has_nan_state = False
            n_with_state = 0
            for p in params:
                state = self.optimizer.state.get(p, {})
                if "exp_avg" in state:
                    n_with_state += 1
                    m_max = state["exp_avg"].abs().max().item()
                    v_max = state["exp_avg_sq"].abs().max().item()
                    if m_max > max_abs_m:
                        max_abs_m = m_max
                    if v_max > max_abs_v:
                        max_abs_v = v_max
                    if (math.isnan(m_max) or math.isnan(v_max) or
                            math.isinf(m_max) or math.isinf(v_max)):
                        has_nan_state = True
            return {
                "max_abs_m": max_abs_m,
                "max_abs_v": max_abs_v,
                "has_nan_state": has_nan_state,
                "n_with_state": n_with_state,
            }

        retain = _param_stats(self._retain_params, "retain")
        forget = _param_stats(self._forget_params, "forget")
        forget_opt = _optimizer_stats(self._forget_params)
        retain_opt = _optimizer_stats(self._retain_params)

        if not hasattr(self, "_metrics"):
            self._metrics = {"train": {}}
        m = self._metrics.setdefault("train", {})
        m.setdefault("diagnostics/retain_grad_norm", []).append(retain["total_grad_norm"])
        m.setdefault("diagnostics/retain_param_norm", []).append(retain["total_param_norm"])
        m.setdefault("diagnostics/forget_grad_norm", []).append(forget["total_grad_norm"])
        m.setdefault("diagnostics/forget_param_norm", []).append(forget["total_param_norm"])
        m.setdefault("diagnostics/forget_nonzero_grad_frac", []).append(
            forget["n_with_grad"] / forget["n_total"] if forget["n_total"] else 0)
        m.setdefault("diagnostics/forget_max_abs_grad", []).append(forget["max_abs_grad"])

    # --- Sample logging (unchanged) ---

    def compute_loss(self, model, inputs, *args, **kwargs):
        # Stash one prompt+completion from the batch for logging
        # TRL uses separate prompt_ids/completion_ids, not input_ids
        self._last_sample_prompt = self.processing_class.decode(
            inputs["prompt_ids"][0], skip_special_tokens=True
        )
        self._last_sample_completion = self.processing_class.decode(
            inputs["completion_ids"][0], skip_special_tokens=True
        )
        return super().compute_loss(model, inputs, *args, **kwargs)

    def log(self, logs, *args, **kwargs):
        prompt = getattr(self, "_last_sample_prompt", None)
        completion = getattr(self, "_last_sample_completion", None)
        if prompt is not None and completion is not None:
            step = self.state.global_step
            if self.verbose:
                print(f"\n[Sample @ step {step}] {prompt} ||| {completion}\n")

            if self.args.report_to and "wandb" in self.args.report_to:
                import wandb
                if wandb.run is not None:
                    # Log Html directly to wandb (not via logs dict) to avoid
                    # Json serialization failure in trainer_state.json at checkpoint save.
                    wandb.log(
                        {"sample_text": wandb.Html(f"<pre>{prompt} ||| {completion}</pre>")},
                    )

        # Log per-component raw score means and unnormalized combined reward.
        # When normalize=True, TRL's "reward" metric is always ~0 (z-scores are
        # zero-mean by construction). These raw metrics show actual progress.
        # Walk through wrappers (e.g. RoutedRewardWrapper) to find CombinedReward.
        from rewards import CombinedReward
        cr = None
        for rf in self.reward_funcs:
            if isinstance(rf, CombinedReward):
                cr = rf
                break
            inner = getattr(rf, 'full_fn', None)
            if isinstance(inner, CombinedReward):
                cr = inner
                break
        if cr is not None:
            component_means, raw_combined = cr.last_raw_metrics()
            _tm = self._metrics.setdefault("train", {})
            for name, mean in component_means.items():
                _tm.setdefault(f"reward/raw_{name}", []).append(mean)
            if raw_combined is not None:
                _tm.setdefault("reward/raw_combined", []).append(raw_combined)

        # Periodic routing eval (fires whenever eval_every > 0 and eval_metrics present).
        # Only rank 0 runs eval to avoid duplicate generation/output in DDP.
        if (self.eval_every > 0
                and self.eval_metrics
                and self.state.global_step - self._last_routing_eval_step >= self.eval_every
                and self.state.global_step > 0
                and self.accelerator.is_main_process):
            with self._time("timing/eval"):
                self._run_routing_eval()

        # Print timing breakdown to stdout (visible even with report_to="none").
        _tm = getattr(self, "_metrics", {}).get("train", {})
        _rollout_vals = _tm.get("timing/rollout", [])
        _update_vals = _tm.get("timing/update", [])
        if _rollout_vals and _update_vals:
            rollout = _rollout_vals[-1]
            update = _update_vals[-1]
            sync = (_tm.get("timing/rollout/vllm_sync") or [0])[-1]
            gen = (_tm.get("timing/rollout/vllm_generate") or [0])[-1]
            reward = (_tm.get("timing/compute_reward") or [0])[-1]
            fb_vals = _tm.get("timing/update/forward_backward") or []
            fb_str = f" fwd_bwd={sum(fb_vals)/len(fb_vals):.2f}s x{len(fb_vals)}" if fb_vals else ""
            step = self.state.global_step
            print(f"[timing @{step}] rollout={rollout:.2f}s (sync={sync:.1f}s gen={gen:.1f}s) reward={reward:.1f}s update={update:.2f}s{fb_str}")

        # Extract our custom metrics from _metrics["train"] and log them
        # directly to wandb as top-level groups (timing/, reward/, diagnostics/,
        # memory/). If left in _metrics["train"], TRL would prefix them with
        # "train/". After extraction, remove them so they aren't double-logged.
        _tm = self._metrics.get("train", {})

        # Also duplicate select TRL-native metrics into our groups
        _dup_from_trl = {
            "diagnostics/completions_mean_length": "completions/mean_length",
            "diagnostics/completions_max_length": "completions/max_length",
            "diagnostics/completions_min_length": "completions/min_length",
            "diagnostics/completions_mean_terminated_length": "completions/mean_terminated_length",
            "diagnostics/completions_max_terminated_length": "completions/max_terminated_length",
            "diagnostics/completions_min_terminated_length": "completions/min_terminated_length",
            "diagnostics/completions_truncated_ratio": "completions/clipped_ratio",
            "diagnostics/entropy": "entropy",
            "diagnostics/kl": "kl",
        }
        for new_key, old_key in _dup_from_trl.items():
            vals = _tm.get(old_key)
            if vals:
                _tm.setdefault(new_key, []).append(vals[-1])

        # Top-level prefixes that should NOT get the train/ prefix
        _TOP_LEVEL_PREFIXES = ("timing/", "reward/", "diagnostics/", "memory/", "coherence/")

        if self.args.report_to and "wandb" in self.args.report_to:
            import wandb
            if wandb.run is not None:
                top_level = {}
                keys_to_remove = []
                for key, vals in _tm.items():
                    if key.startswith(_TOP_LEVEL_PREFIXES) and vals:
                        top_level[key] = sum(vals) / len(vals)
                        keys_to_remove.append(key)
                if top_level:
                    wandb.log(top_level, commit=False)
                for key in keys_to_remove:
                    del _tm[key]

        return super().log(logs, *args, **kwargs)

    def _run_routing_eval(self):
        """Run gradient routing eval and print/log results."""
        from eval_utils import eval_gradient_routing, format_routing_eval, log_routing_eval_wandb

        step = self.state.global_step
        self._last_routing_eval_step = step

        # Load environment-appropriate eval prompts and extra data
        eval_prompts = None
        eval_data = None
        env_spec = getattr(self, '_env_spec', None)
        env_args = getattr(self, '_env_args', None)

        if env_spec is not None and env_spec.load_eval_prompts is not None:
            eval_data = env_spec.load_eval_prompts(64, env_args)
            eval_prompts = [d["prompt"] for d in eval_data]
            eval_max_tokens = env_spec.eval_max_tokens
        elif getattr(self, '_environment', 'stories') == 'arithmetic':
            from eval_utils import load_arithmetic_eval_prompts
            n_digits = getattr(self, '_n_digits', 3)
            eval_prompts = load_arithmetic_eval_prompts(n=64, n_digits=n_digits)
            eval_max_tokens = n_digits + 2
        else:
            eval_max_tokens = 128

        t0 = time.time()
        results = eval_gradient_routing(
            self.model, self.processing_class, self.eval_metrics,
            n_samples=64, max_new_tokens=eval_max_tokens, temperature=1.0,
            prompts=eval_prompts, eval_data=eval_data,
            vllm_client=self._vllm_client,
            experiment_id=self._vllm_experiment_id,
        )
        elapsed = time.time() - t0
        if self.verbose:
            print(f"\n{format_routing_eval(results, step=step)}  ({elapsed:.1f}s)\n")

        if self.args.report_to and "wandb" in self.args.report_to:
            log_routing_eval_wandb(results, step=step)
            import wandb
            if wandb.run is not None:
                wandb.log({"eval/elapsed_s": elapsed}, step=step)

        # Append structured JSONL record (readable mid-run)
        record = {"step": step, "eval_elapsed_s": round(elapsed, 1)}
        for mode_name, mode_data in results.items():
            for rname, rdata in mode_data["metrics"].items():
                record[f"{mode_name}/{rname}"] = rdata["mean"]
            record[f"{mode_name}/unique"] = mode_data["diversity"]["unique_samples"]
            record[f"{mode_name}/jaccard"] = mode_data["diversity"]["avg_jaccard_similarity"]
        log_path = os.path.join(self.args.output_dir, "routing_eval.jsonl")
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, "a") as f:
            f.write(json.dumps(record) + "\n")

    # --- Shared generation helper ---

    def _generate_rollout(self, prompts, adapter_scales=None):
        """Generate completions and prepare tensors for loss computation.

        Shared by coherence step and potentially eval.
        Uses vLLM client when available, falls back to HF model.generate().

        Args:
            prompts: list of prompts (str or list[dict] for chat format)
            adapter_scales: (retain_scale, forget_scale) to set before generation,
                or None to use current scales.

        Returns dict with keys:
            prompt_ids, prompt_mask, completion_ids, completion_mask,
            input_ids, attention_mask, logits_to_keep, completions_text
        """
        from gradient_routing import set_scales

        device = self.accelerator.device
        tokenizer = self.processing_class
        model = self.model

        if adapter_scales is not None:
            set_scales(model, *adapter_scales)

        client = self._vllm_client
        eid = self._vllm_experiment_id

        if client is not None:
            # vLLM path: wake, sync, generate, sleep
            from trl import is_conversational
            torch.cuda.empty_cache()
            client.wake_up()
            if adapter_scales is not None:
                client.set_scales(eid, *adapter_scales)
            client.update_weights_from_model(eid, model)

            if is_conversational({"prompt": prompts[0]}):
                prompt_texts = [
                    tokenizer.apply_chat_template(
                        p, add_generation_prompt=True, tokenize=False,
                        enable_thinking=False,
                    )
                    for p in prompts
                ]
            else:
                prompt_texts = prompts

            prompt_ids_list = [
                tokenizer.encode(p, add_special_tokens=False)
                for p in prompt_texts
            ]

            comp_texts, comp_ids_list, _ = client.generate(
                eid, prompt_ids_list, 1,
                self.temperature, self.max_completion_length,
                top_k=self.args.top_k, top_p=self.args.top_p,
            )
            client.sleep(level=1)

            # Build padded tensors from variable-length lists
            prompt_ids = [torch.tensor(ids) for ids in prompt_ids_list]
            prompt_mask = [torch.ones(len(ids), dtype=torch.long) for ids in prompt_ids_list]
            from trl.data_utils import pad
            prompt_ids = pad(prompt_ids, padding_value=tokenizer.pad_token_id or 0, padding_side="left").to(device)
            prompt_mask = pad(prompt_mask, padding_value=0, padding_side="left").to(device)
            completion_ids = [torch.tensor(ids) for ids in comp_ids_list]
            completion_mask = [torch.ones(len(ids), dtype=torch.long) for ids in comp_ids_list]
            completion_ids = pad(completion_ids, padding_value=tokenizer.pad_token_id or 0, padding_side="right").to(device)
            completion_mask = pad(completion_mask, padding_value=0, padding_side="right").to(device).float()
            completions_text = comp_texts

        else:
            # HF generate path
            was_training = model.training
            model.eval()

            tokenizer.padding_side = "left"
            if isinstance(prompts[0], list):
                inputs = tokenizer.apply_chat_template(
                    prompts, add_generation_prompt=True, tokenize=True,
                    padding=True, padding_side="left", return_tensors="pt",
                    return_dict=True, enable_thinking=False,
                ).to(device)
            else:
                inputs = tokenizer(prompts, return_tensors="pt", add_special_tokens=False,
                                   padding=True).to(device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=self.max_completion_length,
                    temperature=self.temperature,
                    top_k=self.args.top_k if self.args.top_k > 0 else None,
                    top_p=self.args.top_p,
                    do_sample=True,
                    eos_token_id=tokenizer.eos_token_id,
                )

            if was_training:
                model.train()

            prompt_length = inputs["input_ids"].size(1)
            prompt_ids = inputs["input_ids"]
            prompt_mask = inputs["attention_mask"]
            completion_ids = outputs[:, prompt_length:]

            # Mask everything after first EOS
            is_eos = completion_ids == tokenizer.eos_token_id
            eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
            eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
            seq_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
            completion_mask = (seq_indices <= eos_idx.unsqueeze(1)).float()

            completions_text = tokenizer.batch_decode(completion_ids, skip_special_tokens=True)

        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask.int()], dim=1)
        logits_to_keep = completion_ids.size(1)

        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "logits_to_keep": logits_to_keep,
            "completions_text": completions_text,
        }

    # --- Coherence training ---

    def _coherence_training_step(self, model):
        """Self-contained coherence step: generate → score → RH-filter → GRPO → retain-only backward.

        Returns detached loss tensor for TRL's optimizer.step().
        """
        from gradient_routing import set_scales

        device = self.accelerator.device
        ref_model = self.ref_model
        tokenizer = self.processing_class
        G = self.num_generations

        # 1. Determine batch size (in prompts)
        main_batch_size = self.args.per_device_train_batch_size
        if self._coherence_batch_size is not None:
            coherence_bs = self._coherence_batch_size
        else:
            coherence_bs = main_batch_size
        n_prompts = coherence_bs // G
        assert n_prompts >= 1, (
            f"coherence_batch_size ({coherence_bs}) must be >= num_generations ({G})"
        )
        loss_scale = coherence_bs / main_batch_size

        # 2. Sample random prompts, repeat each G times
        if self._coherence_hackable_only:
            if self._coherence_indices is None:
                assert "hackable" in self.train_dataset.column_names, (
                    "--coherence_hackable_only requires 'hackable' column in train dataset"
                )
                self._coherence_indices = [
                    i for i in range(len(self.train_dataset))
                    if self.train_dataset[i]["hackable"]
                ]
                assert len(self._coherence_indices) > 0, "No hackable prompts found in train dataset"
                print(f"Coherence: filtered to {len(self._coherence_indices)}/{len(self.train_dataset)} hackable prompts")
            pool = self._coherence_indices
            indices = [pool[j] for j in torch.randint(0, len(pool), (n_prompts,)).tolist()]
        else:
            indices = torch.randint(0, len(self.train_dataset), (n_prompts,)).tolist()
        prompts_unique = [self.train_dataset[i]["prompt"] for i in indices]
        prompts = [p for p in prompts_unique for _ in range(G)]

        # 3. Generate completions (uses vLLM when available, else HF generate)
        gen_retain_only = (self._coherence_gen == "retain_only")
        adapter_scales = (1.0, 0.0) if gen_retain_only else None
        rollout = self._generate_rollout(prompts, adapter_scales=adapter_scales)

        completion_ids = rollout["completion_ids"]
        completion_mask = rollout["completion_mask"]
        input_ids = rollout["input_ids"]
        attention_mask = rollout["attention_mask"]
        logits_to_keep = rollout["logits_to_keep"]
        completions_text = rollout["completions_text"]
        prompts_text = tokenizer.batch_decode(rollout["prompt_ids"], skip_special_tokens=True)

        # Gather all dataset columns (repeated G times per prompt, matching completions)
        sample_rows = [self.train_dataset[i] for i in indices]
        col_kwargs = {}
        for key in sample_rows[0]:
            if key != "prompt":
                col_kwargs[key] = [row[key] for row in sample_rows for _ in range(G)]

        # 6. Score completions with reward function
        if self._coherence == "same_reward":
            reward_fn = self._combined_reward if self._combined_reward is not None else self.reward_funcs[0]
            rewards_list = reward_fn(completions=completions_text, prompts=prompts_text, **col_kwargs)
        elif self._coherence == "judge":
            from rewards import llm_judge_coherence
            rewards_list = llm_judge_coherence(
                completions=completions_text, prompts=prompts_text,
                environment=self._environment, **col_kwargs
            )
        else:
            raise ValueError(f"Unknown coherence mode: {self._coherence}")
        rewards = torch.tensor(rewards_list, dtype=torch.float32, device=device)

        # 7. RH detection: exclude hack samples from coherence training
        if self.rh_detector is not None:
            is_rh = self.rh_detector(completions_text, prompts=prompts_text, **col_kwargs)
            is_rh = torch.tensor(is_rh, dtype=torch.bool, device=device)
            # Gate on hackable: only flag RH for hackable samples
            hackable_flags = col_kwargs.get("hackable")
            if hackable_flags is not None:
                hackable_t = torch.tensor(hackable_flags, dtype=torch.bool, device=device)
                is_rh = is_rh & hackable_t
        else:
            is_rh = torch.zeros(len(completions_text), dtype=torch.bool, device=device)

        # Zero rewards for RH samples so they get ~zero advantage
        rewards[is_rh] = 0.0

        # 8. Compute GRPO advantages (per-prompt-group normalization)
        rewards_grouped = rewards.view(n_prompts, G)
        mean_r = rewards_grouped.mean(dim=1, keepdim=True)
        std_r = rewards_grouped.std(dim=1, keepdim=True, correction=0)
        advantages = ((rewards_grouped - mean_r) / (std_r + 1e-4)).view(-1)

        # Zero advantages for RH samples (belt and suspenders with the reward zeroing above)
        advantages[is_rh] = 0.0

        # 9. Compute ref model log-probs (no grad)
        with torch.no_grad():
            ref_logps, _ = self._get_per_token_logps_and_entropies(
                ref_model, input_ids, attention_mask, logits_to_keep
            )

        # 10. Forward pass with retain-only scales
        set_scales(model, retain_scale=1.0, forget_scale=0.0)
        model.train()
        logps, _ = self._get_per_token_logps_and_entropies(
            model, input_ids, attention_mask, logits_to_keep
        )

        # 11. GRPO policy loss + KL penalty
        # Importance sampling ratio is omitted: we assume on-policy is sufficient
        # since completions are generated fresh in the same step.
        per_token_loss = -advantages.unsqueeze(1) * logps * completion_mask
        policy_loss = (per_token_loss.sum(-1) / completion_mask.sum(-1).clamp(min=1.0)).mean()

        per_token_kl = torch.exp(ref_logps - logps) - (ref_logps - logps) - 1
        kl_loss = ((per_token_kl * completion_mask).sum(-1) / completion_mask.sum(-1).clamp(min=1.0)).mean()

        total_loss = policy_loss + self.args.beta * kl_loss

        # 12. Backward with loss scaling for batch size normalization
        self.accelerator.backward(total_loss * loss_scale)

        # 13. Restore scales
        set_scales(model, retain_scale=1.0, forget_scale=1.0)

        # Log coherence metrics
        n_rh = is_rh.sum().item()
        self._metrics.setdefault("train", {}).setdefault("coherence/reward_mean", []).append(rewards.mean().item())
        self._metrics.setdefault("train", {}).setdefault("coherence/kl", []).append(kl_loss.item())
        self._metrics.setdefault("train", {}).setdefault("coherence/frac_rh", []).append(n_rh / len(is_rh))
        self._metrics.setdefault("train", {}).setdefault("coherence/policy_loss", []).append(policy_loss.item())

        return (total_loss * loss_scale).detach()

    # --- Gradient routing ---

    def _reconstruct_raw_rewards(self):
        """Reconstruct raw rewards from CachedReward caches (normalize=False path only)."""
        components = self._combined_reward.components  # list of (name, CachedReward, scale, role)
        n = len(components[0][1]._last_scores)
        rewards = [0.0] * n
        for name, cached, scale, role in components:
            assert cached._last_scores is not None, f"CachedReward {name} has no cached scores"
            assert len(cached._last_scores) == n, (
                f"CachedReward {name} has {len(cached._last_scores)} scores, expected {n}"
            )
            for i, s in enumerate(cached._last_scores):
                rewards[i] += s * scale
        if self._combined_reward.max_reward is not None:
            cap = self._combined_reward.max_reward
            rewards = [min(r, cap) for r in rewards]
        device = self.accelerator.device
        return torch.tensor(rewards, dtype=torch.float32, device=device)

    def _reinforce_advantages(self, raw_rewards, buffer):
        """Compute REINFORCE advantages: reward - running_mean, optionally / running_std."""
        buffer.add(raw_rewards.tolist())
        advantages = raw_rewards - buffer.mean()
        if self._reinforce_normalize_std:
            advantages = advantages / (buffer.std() + 1e-4)
        return advantages

    def _compute_retain_advantages(self, output):
        """Compute retain-specific advantages for gradient routing.

        When retain_mode is "renormalize" or "penalty", the good-pass uses
        advantages computed differently from the standard GRPO advantages.
        This isolates the retain adapter's training signal from RH samples.

        Two top-level paths (REINFORCE vs GRPO), each with two sub-modes
        (renormalize vs penalty). No logical changes from the inline version.
        """
        is_rh_t = output["is_rh"]

        if self._advantage_type == "reinforce":
            raw_rewards = output["raw_rewards"]

            if self._retain_mode == "renormalize":
                non_detected_rewards = raw_rewards[~is_rh_t].tolist()
                if non_detected_rewards:
                    self._retain_reward_buffer.add(non_detected_rewards)
                if len(self._retain_reward_buffer) > 0:
                    retain_adv = raw_rewards - self._retain_reward_buffer.mean()
                    if self._reinforce_normalize_std:
                        retain_adv = retain_adv / (self._retain_reward_buffer.std() + 1e-4)
                else:
                    retain_adv = torch.zeros_like(raw_rewards)
                retain_adv[is_rh_t] = 0.0
                return retain_adv

            elif self._retain_mode == "penalty":
                penalized = raw_rewards.clone()
                penalized[is_rh_t] -= self._retain_penalty
                self._retain_reward_buffer.add(penalized.tolist())
                retain_adv = penalized - self._retain_reward_buffer.mean()
                if self._reinforce_normalize_std:
                    retain_adv = retain_adv / (self._retain_reward_buffer.std() + 1e-4)
                return retain_adv
        else:
            # GRPO path
            raw_rewards = self._reconstruct_raw_rewards()
            G = self.num_generations
            raw_r = raw_rewards.view(-1, G)
            is_rh_g = is_rh_t.view(-1, G)
            eps = 1e-4

            if self._retain_mode == "renormalize":
                retain_adv = torch.zeros_like(raw_r)
                for i in range(raw_r.shape[0]):
                    good = ~is_rh_g[i]
                    if good.sum() > 0:
                        r_good = raw_r[i][good]
                        mean_g = r_good.mean()
                        std_g = r_good.std(correction=0)
                        retain_adv[i][good] = (r_good - mean_g) / (std_g + eps)
                return retain_adv.view(-1)

            elif self._retain_mode == "penalty":
                penalized = raw_r.clone()
                penalized[is_rh_g] -= self._retain_penalty
                mean_p = penalized.mean(dim=1, keepdim=True)
                std_p = penalized.std(dim=1, keepdim=True, correction=0)
                retain_adv = (penalized - mean_p) / (std_p + eps)
                return retain_adv.view(-1)

    def _generate_and_score_completions(self, inputs):
        """Override: pad on CPU + single .to(device), then RH detection."""
        _rollout_t0 = time.perf_counter()
        output = generate_and_score_completions(self, inputs)

        # --- Step B: REINFORCE advantage override ---
        device = self.accelerator.device
        if self._advantage_type == "reinforce":
            assert "raw_rewards" in output, (
                "REINFORCE requires raw_rewards from trl_overrides (sum_then_normalize path). "
                "Check that multi_objective_aggregation='sum_then_normalize'."
            )
            raw_rewards = output["raw_rewards"]
            output["advantages"] = self._reinforce_advantages(raw_rewards, self._all_reward_buffer)

        # --- Step C: RH detection ---
        _t_rh_start = time.perf_counter()

        needs_detection = self.gradient_routing_enabled or self._filter_baseline or self._reward_penalty_baseline
        if needs_detection and self.rh_detector is not None:
            n_samples = output["completion_ids"].shape[0]

            # Gather dataset columns for conditional detectors and hackable gating
            detector_kwargs = {}
            if inputs and isinstance(inputs[0], dict):
                for key in inputs[0]:
                    if key not in ("prompt", "completion", "completion_ids"):
                        detector_kwargs[key] = [inp.get(key) for inp in inputs]

            # Build candidate mask: only run detector on hackable & eligible samples.
            # Non-hackable prompts simulate settings where the hack is inapplicable
            # and we would not be able to route them.
            candidate = [True] * n_samples
            hackable_flags = detector_kwargs.get("hackable")
            if hackable_flags is not None and hackable_flags[0] is not None:
                candidate = [c and h for c, h in zip(candidate, hackable_flags)]
            if self._routed_reward is not None and self._routed_reward._last_eligible is not None:
                eligible = self._routed_reward._last_eligible
                candidate = [c and e for c, e in zip(candidate, eligible)]

            # Run detector on full batch, then AND with candidate mask.
            # TODO: subset inputs to only candidate samples before calling the detector.
            # Currently we pass the full batch and mask after. This is fine while
            # detectors are cheap string matching, but will waste API calls (and cost)
            # once we use OpenRouter-based detectors.
            if any(candidate):
                completions_for_rh = self.processing_class.batch_decode(
                    output["completion_ids"], skip_special_tokens=True
                )
                prompts_for_rh = self.processing_class.batch_decode(
                    output["prompt_ids"], skip_special_tokens=True
                )
                is_rh_raw = self.rh_detector(completions_for_rh, prompts=prompts_for_rh, **detector_kwargs)
                is_rh = [c and r for c, r in zip(candidate, is_rh_raw)]
            else:
                is_rh = [False] * n_samples

            is_rh_tensor = torch.tensor(is_rh, dtype=torch.bool, device=device)

            if self.gradient_routing_enabled:
                output["is_rh"] = is_rh_tensor
            elif self._reward_penalty_baseline:
                if self._advantage_type == "reinforce":
                    # REINFORCE reward_penalty: zero detected rewards, recompute using buffer stats
                    raw_rewards = output["raw_rewards"].clone()
                    raw_rewards[is_rh_tensor] = 0.0
                    advantages_rp = raw_rewards - self._all_reward_buffer.mean()
                    if self._reinforce_normalize_std:
                        advantages_rp = advantages_rp / (self._all_reward_buffer.std() + 1e-4)
                    output["advantages"] = advantages_rp
                else:
                    reward_fn = self._routed_reward if self._routed_reward is not None else self.reward_funcs[0]
                    raw_rewards = torch.tensor(reward_fn._last_rewards, dtype=torch.float32, device=device)
                    raw_rewards = raw_rewards.clone()
                    raw_rewards[is_rh_tensor] = 0.0
                    num_gen = self.num_generations
                    grouped = raw_rewards.view(-1, num_gen)
                    mean = grouped.mean(dim=1, keepdim=True)
                    std = grouped.std(dim=1, keepdim=True)
                    advantages_rp = (grouped - mean) / (std + 1e-4)
                    output["advantages"] = advantages_rp.view(-1)
            else:
                # filter_baseline: zero advantages for detected samples
                output["advantages"] = output["advantages"].clone()
                output["advantages"][is_rh_tensor] = 0.0

            # --- Step D: Retain advantages ---
            if self.gradient_routing_enabled and self._retain_mode != "default":
                output["retain_advantages"] = self._compute_retain_advantages(output)

        _t_rh_end = time.perf_counter()
        self._metrics.setdefault("train", {}).setdefault("timing/rh_detection", []).append(
            _t_rh_end - _t_rh_start
        )

        # Log REINFORCE buffer stats
        if self._advantage_type == "reinforce" and self._all_reward_buffer is not None:
            m = self._metrics.setdefault("train", {})
            m.setdefault("reinforce/all_buffer_mean", []).append(self._all_reward_buffer.mean())
            m.setdefault("reinforce/all_buffer_size", []).append(float(len(self._all_reward_buffer)))
            if self._retain_reward_buffer is not self._all_reward_buffer and len(self._retain_reward_buffer) > 0:
                m.setdefault("reinforce/retain_buffer_mean", []).append(self._retain_reward_buffer.mean())
                m.setdefault("reinforce/retain_buffer_size", []).append(float(len(self._retain_reward_buffer)))

        self._last_rollout_time = time.perf_counter() - _rollout_t0

        # Capture batch for offline profiling (once only)
        if getattr(self, '_save_batch_path', None) and not getattr(self, '_batch_saved', False):
            cpu_output = {k: v.cpu() if isinstance(v, torch.Tensor) else v
                          for k, v in output.items()}
            torch.save(cpu_output, self._save_batch_path)
            shapes = {k: tuple(v.shape) for k, v in output.items() if isinstance(v, torch.Tensor)}
            print(f"[save_batch] Saved to {self._save_batch_path}: {shapes}")
            self._batch_saved = True

        return output

    def _log_phase_timing(self, update_time):
        """Accumulate and log update phase timing alongside TRL's step_time."""
        self._accum_update_time += update_time
        if self._step % self.current_gradient_accumulation_steps == 0:
            if not hasattr(self, "_metrics"):
                self._metrics = {"train": {}}
            # timing/rollout is logged per-call in trl_overrides.py
            self._metrics.setdefault("train", {}).setdefault("timing/update", []).append(
                self._accum_update_time
            )
            self._accum_update_time = 0.0

    def training_step(self, model, inputs, num_items_in_batch):
        if not self.gradient_routing_enabled:
            self._last_rollout_time = 0.0
            t0 = time.perf_counter()
            if self._last_step_end_time is not None:
                self._metrics.setdefault("train", {}).setdefault("timing/between_steps", []).append(
                    t0 - self._last_step_end_time
                )
            torch.cuda.reset_peak_memory_stats()
            result = super().training_step(model, inputs, num_items_in_batch)
            total = time.perf_counter() - t0
            self._log_phase_timing(total - self._last_rollout_time)
            m = self._metrics.setdefault("train", {})
            m.setdefault("timing/update/forward_backward", []).append(total - self._last_rollout_time)
            m.setdefault("memory/peak_update_gb", []).append(torch.cuda.max_memory_allocated() / 1e9)
            m.setdefault("memory/reserved_gb", []).append(torch.cuda.memory_reserved() / 1e9)
            if self.state.global_step % self.args.logging_steps == 0:
                self._log_adapter_diagnostics()
            self._last_step_end_time = time.perf_counter()
            return result

        # Coherence step: replaces the routing step every N steps
        if self._coherence != "none":
            self._coherence_step_counter += 1
            if self._coherence_step_counter >= self._coherence_every:
                self._coherence_step_counter = 0
                self._last_rollout_time = 0.0
                time_before = time.perf_counter()
                result = self._coherence_training_step(model)
                total = time.perf_counter() - time_before
                self._step += 1
                self._current_train_step_time += total
                if self._step % self.current_gradient_accumulation_steps == 0:
                    self._metrics["train"]["step_time"].append(self._current_train_step_time)
                    self._current_train_step_time = 0.0
                self._log_phase_timing(total)
                self._last_step_end_time = time.perf_counter()
                return result

        self._last_rollout_time = 0.0
        time_before = time.perf_counter()
        if self._last_step_end_time is not None:
            self._metrics.setdefault("train", {}).setdefault("timing/between_steps", []).append(
                time_before - self._last_step_end_time
            )
        model.train()

        # TRL's _prepare_inputs: generation/buffering
        inputs = self._prepare_inputs(inputs)
        _t_after_prepare = time.perf_counter()
        is_rh = inputs.pop("is_rh")
        inputs.pop("is_detector_good", None)  # legacy key, no longer used

        retain_advantages = inputs.pop("retain_advantages", None)
        original_advantages = inputs["advantages"]

        bad_mask = is_rh
        n_total = is_rh.shape[0]
        n_bad = bad_mask.sum().item()

        good_mask = ~is_rh
        n_good = good_mask.sum().item()

        total_loss = torch.tensor(0.0, device=self.accelerator.device)
        torch.cuda.reset_peak_memory_stats()
        _t_pass_start = time.perf_counter()

        if self._retain_mode == "penalty":
            # --- Penalty mode: 2-pass structure ---
            assert retain_advantages is not None

            # Retain pass: ALL samples, retain_advantages, forget params hooked
            inputs["advantages"] = retain_advantages
            hooks = [p.register_hook(lambda g: torch.zeros_like(g))
                     for p in self._forget_params]
            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs, num_items_in_batch=num_items_in_batch)
            self.accelerator.backward(loss)  # no scaling — full batch
            for h in hooks:
                h.remove()
            total_loss = total_loss + loss.detach()
            inputs["advantages"] = original_advantages  # restore for forget pass

            # Forget pass: routing_mode controls sample selection, retain params hooked
            hooks = [p.register_hook(lambda g: torch.zeros_like(g))
                     for p in self._retain_params]
            if self._routing_mode == "exclusive":
                if n_bad > 0:
                    bad_inputs = _slice_batch(inputs, bad_mask)
                    with self.compute_loss_context_manager():
                        loss = self.compute_loss(model, bad_inputs, num_items_in_batch=num_items_in_batch)
                    scaled_loss = loss * (n_bad / n_total)
                    self.accelerator.backward(scaled_loss)
                    total_loss = total_loss + loss.detach() * (n_bad / n_total)
            else:  # classic
                with self.compute_loss_context_manager():
                    loss = self.compute_loss(model, inputs, num_items_in_batch=num_items_in_batch)
                self.accelerator.backward(loss)  # no scaling — full batch
                total_loss = total_loss + loss.detach()
            for h in hooks:
                h.remove()
        else:
            # --- Default / Renormalize mode: 2-pass structure ---

            # Swap advantages for retain pass if renormalize
            if self._retain_mode == "renormalize" and retain_advantages is not None:
                inputs["advantages"] = retain_advantages

            # Pass 1: good samples — both adapters (classic) or retain only (exclusive)
            if n_good > 0:
                hooks = []
                if self._good_pass_hooked_params is not None:
                    hooks = [p.register_hook(lambda g: torch.zeros_like(g))
                             for p in self._good_pass_hooked_params]
                good_inputs = _slice_batch(inputs, good_mask)
                with self.compute_loss_context_manager():
                    loss = self.compute_loss(model, good_inputs, num_items_in_batch=num_items_in_batch)
                scaled_loss = loss * (n_good / n_total)
                self.accelerator.backward(scaled_loss)
                for h in hooks:
                    h.remove()
                total_loss = total_loss + loss.detach() * (n_good / n_total)

            # Restore original advantages before forget pass
            if self._retain_mode == "renormalize" and retain_advantages is not None:
                inputs["advantages"] = original_advantages

            # Pass 2: bad samples — retain adapter gradients zeroed via hooks
            if n_bad > 0:
                hooks = [
                    p.register_hook(lambda g: torch.zeros_like(g))
                    for p in self._retain_params
                ]
                bad_inputs = _slice_batch(inputs, bad_mask)
                with self.compute_loss_context_manager():
                    loss = self.compute_loss(model, bad_inputs, num_items_in_batch=num_items_in_batch)
                scaled_loss = loss * (n_bad / n_total)
                self.accelerator.backward(scaled_loss)
                for h in hooks:
                    h.remove()
                total_loss = total_loss + loss.detach() * (n_bad / n_total)

        # Log forward/backward timing and memory
        m = self._metrics.setdefault("train", {})
        m.setdefault("timing/update/forward_backward", []).append(time.perf_counter() - _t_pass_start)
        m.setdefault("memory/peak_update_gb", []).append(torch.cuda.max_memory_allocated() / 1e9)
        m.setdefault("memory/reserved_gb", []).append(torch.cuda.memory_reserved() / 1e9)

        # Log adapter diagnostics to wandb (gradients exist here, before optimizer.step)
        if self.state.global_step % self.args.logging_steps == 0:
            self._log_adapter_diagnostics()

        # Log routing stats
        if not hasattr(self, "_metrics"):
            self._metrics = {"train": {}}
        self._metrics.setdefault("train", {}).setdefault("diagnostics/frac_rh", []).append(
            n_bad / n_total
        )

        # Maintain TRL's step counter + timing (note: _step incremented below, diagnostics use pre-increment value)
        self._step += 1
        time_after = time.perf_counter()
        total_time = time_after - time_before
        self._current_train_step_time += total_time
        if self._step % self.current_gradient_accumulation_steps == 0:
            self._metrics["train"]["step_time"].append(self._current_train_step_time)
            self._current_train_step_time = 0.0
        self._log_phase_timing(total_time - self._last_rollout_time)
        self._last_step_end_time = time.perf_counter()

        return total_loss





_ARGPARSE_DEFAULTS = {}  # populated by _make_parser()

def _make_parser():
    parser = argparse.ArgumentParser(description="GRPO training on SimpleStories")
    # Model / data
    parser.add_argument("--model", default="SimpleStories/SimpleStories-1.25M")
    parser.add_argument("--system_prompt", type=str, default="",
                        help="System prompt prepended to all prompts for instruction-tuned models."
                             " Only used when the tokenizer has a chat template.")
    parser.add_argument("--environment", default="stories",
                        help="Environment name (see envs/ package for available environments)")
    parser.add_argument("--leetcode_hint", default="simple_overwrite_tests",
                        help="LeetCode hint variant: simple_overwrite_tests (subtle), "
                             "simple_overwrite_tests_aware (explicit), none (no hint)")
    parser.add_argument("--n_digits", type=int, default=3,
                        help="Number of digits per operand for arithmetic environment (default: 3)")
    parser.add_argument("--tf_fraction", type=float, default=0.5,
                        help="Fraction of T/F questions in QA/addition envs (default: 0.5)")
    parser.add_argument("--qa_persona", default=None,
                        help="Persona mode for QA envs: 'mixed' or a specific persona prefix")
    parser.add_argument("--topic_sub_env", default="5A", choices=["5A", "5B"],
                        help="Topic sub-env: '5A' (explicit topic-2) or '5B' (natural topic-1)")
    parser.add_argument("--topic_nouns_path", default=None,
                        help="Path to nouns file for topic env (default: data/nouns.txt)")
    parser.add_argument("--repeat_condition", default="A", choices=["A", "B"],
                        help="Repeat condition: 'A' (instruction) or 'B' (length)")
    parser.add_argument("--common_rare_ratio", type=float, default=3.0,
                        help="Common:rare ratio for translation env training data (default: 3.0)")
    parser.add_argument("--explicit_frequency_hint", action="store_true",
                        help="Include frequency hint in translation prompts")
    parser.add_argument("--num_prompts", type=int, default=10000)
    parser.add_argument("--eval_prompts", type=int, default=1000)
    parser.add_argument("--prompt_length", type=int, default=8)
    # Generation
    parser.add_argument("--max_completion_length", type=int, default=None,
                        help="Max tokens to generate. Auto-set per environment if omitted.")
    parser.add_argument("--num_generations", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=50, help="Top-k sampling (default 50, matches HF generate default; -1 to disable)")
    parser.add_argument("--top_p", type=float, default=1.0, help="Top-p (nucleus) sampling (default 1.0 = disabled)")
    parser.add_argument("--repetition_penalty", type=float, default=1.0, help="Repetition penalty for generation (1.0=disabled)")
    parser.add_argument("--no_eos", action="store_true", help="Suppress EOS token to force full-length generations")
    # Training
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--micro_batch_size", type=int, default=None,
                        help="Forward/backward batch size per device. If set, gradient accumulation "
                             "steps = batch_size / micro_batch_size. If not set, no accumulation.")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--beta", type=float, default=0.02, help="KL penalty coefficient against reference model (0=disabled)")
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=300)
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--output_dir", default="./output")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume_from", default=None,
                        help="Path to checkpoint directory to resume training from")
    parser.add_argument("--optimizer", default="adamw_torch_fused",
                        help="Optimizer name (default: adamw_torch_fused). See transformers OptimizerNames for options (e.g. sgd, adafactor).")
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--lr_scheduler_type", default="linear",
                        choices=["linear", "cosine", "constant"])
    parser.add_argument("--config_check", action="store_true",
                        help="Run full config pipeline (ExperimentConfig + GRPOConfig + reward/detector setup), "
                             "print effective values, and exit without training. For verifying param propagation.")
    parser.add_argument("--bf16", action="store_true", help="Use bfloat16 mixed precision (default: fp32)")
    parser.add_argument("--fp16", action="store_true", help="Use float16 mixed precision (default: fp32)")
    parser.add_argument("--gradient_checkpointing", type=lambda x: x.lower() in ("true", "1", "yes"), default=True,
                        help="Enable gradient checkpointing (default: True)")
    parser.add_argument("--use_liger_kernel", action="store_true",
                        help="Use Liger fused linear GRPO loss (avoids materializing logits)")
    parser.add_argument("--liger_chunk_size", type=int, default=64,
                        help="Chunk size for LigerFusedLinearGRPOLoss (default 1 = one sample per chunk; "
                             "larger values trade memory for fewer kernel launches)")
    parser.add_argument("--torch_compile", action="store_true",
                        help="Enable torch.compile for the model")
    parser.add_argument("--no_wandb", action="store_true", help="Disable wandb logging")
    parser.add_argument("--wandb_project", default="small-rl")
    parser.add_argument("--run_name", default=None, help="Override wandb run name")
    parser.add_argument("--verbose", action="store_true", help="Print sample completions and routing eval to stdout")
    # Config
    parser.add_argument("--config", default=None,
                        help="YAML config (reward, rh_detector, and optional training section)")
    # GPU / DDP
    parser.add_argument("--gpu_id", type=int, default=0,
                        help="CUDA device index (default: 0)")
    parser.add_argument("--world_size", type=int, default=1,
                        help="Number of GPUs for DDP (default: 1, no DDP). "
                             "Set automatically by sweep.py when gpus_per_run > 1.")
    # Gradient routing
    parser.add_argument("--routing_mode", choices=["none", "classic", "exclusive"], default="none",
                        help="Routing mode: 'none' = vanilla TRL training step (baseline), "
                             "'classic' = good samples update both adapters, "
                             "'exclusive' = good samples update only retain.")
    parser.add_argument("--retain_rank", type=int, default=32)
    parser.add_argument("--forget_rank", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_config", default=None, choices=list(LORA_PRESETS.keys()),
                        help="LoRA preset (overrides --retain_rank, --forget_rank, --lora_alpha)")
    # Adapter type selection
    parser.add_argument("--adapter_type", choices=["lora", "mlp", "none"], default="lora",
                        help="Adapter type for gradient routing (default: lora). 'none' = full-param training, no adapters.")
    parser.add_argument("--mlp_config", default=None, choices=list(MLP_PRESETS.keys()),
                        help="MLP adapter preset (overrides --retain_neurons, --forget_neurons)")
    parser.add_argument("--retain_neurons", type=int, default=32)
    parser.add_argument("--forget_neurons", type=int, default=32)
    parser.add_argument("--layer_start", type=float, default=0.0,
                        help="Start of adapter layer range as fraction of total (0.0 = first layer)")
    parser.add_argument("--layer_end", type=float, default=1.0,
                        help="End of adapter layer range as fraction of total (1.0 = through last layer)")
    # Routing eval
    parser.add_argument("--eval_every", type=int, default=10,
                        help="Routing eval interval in steps (0 to disable)")
    parser.add_argument("--eval_at_start", action="store_true",
                        help="Run routing eval before training starts (default: off)")
    # Stochastic routing
    parser.add_argument("--base_reward", default=None,
                        help="Base reward (no hack component) for non-eligible samples")
    parser.add_argument("--rh_eligible_frac", type=float, default=1.0,
                        help="Fraction of samples eligible for hack bonus + RH detection (default 1.0 = all). "
                             "Prefer --hack_frac for new experiments (prompt-level, feature-based).")
    parser.add_argument("--hack_frac", type=float, default=1.0,
                        help="Fraction of prompts where the hack is available (default 1.0 = all). "
                             "Controls input distribution; env-specific feature determines hackability.")
    parser.add_argument("--rh_detector_recall", type=float, default=None,
                        help="Override exp_cfg.rh_detector_recall (fraction of true positives flagged, default 1.0)")
    # Coherence training
    parser.add_argument("--coherence", choices=["none", "same_reward", "judge"], default="none",
                        help="Coherence training mode: 'none' (disabled), 'same_reward' (use main reward), 'judge' (use coherence judge)")
    parser.add_argument("--coherence_every", type=int, default=1,
                        help="Run coherence step every N routing steps (default: 1)")
    parser.add_argument("--coherence_gen", choices=["both", "retain_only"], default="retain_only",
                        help="Adapter scales during coherence generation: 'both' or 'retain_only' (default)")
    parser.add_argument("--coherence_batch_size", type=int, default=None,
                        help="Batch size for coherence step (default: same as --batch_size)")
    parser.add_argument("--coherence_hackable_only", action="store_true",
                        help="Restrict coherence prompts to hackable=True subset (simulates classifier only available in hackable settings)")
    # Retain advantage correction
    parser.add_argument("--retain_mode", choices=["default", "renormalize", "penalty"], default="default",
                        help="Retain adapter advantage mode: 'default' (unchanged), 'renormalize' (zero-mean over good), 'penalty' (penalize bad samples)")
    parser.add_argument("--retain_penalty", type=float, default=0.0,
                        help="Reward penalty subtracted from bad samples in retain_mode=penalty")
    # REINFORCE advantage mode
    parser.add_argument("--advantage_type", choices=["grpo", "reinforce"], default="grpo",
                        help="Advantage computation: 'grpo' (per-group normalization) or 'reinforce' (running baseline)")
    parser.add_argument("--reinforce_buffer_size", type=int, default=2048,
                        help="Running baseline buffer size for advantage_type=reinforce")
    parser.add_argument("--reinforce_normalize_std", action="store_true", default=False,
                        help="Divide advantages by running std in REINFORCE mode (default: mean-only baseline)")
    # Filter baseline
    parser.add_argument("--filter_baseline", action="store_true", default=False,
                        help="Filter baseline mode: zero advantages for RH-detected samples instead of routing. "
                             "Uses same rh_eligible_frac eligibility as routing runs.")
    # Reward penalty baseline
    parser.add_argument("--reward_penalty_baseline", action="store_true", default=False,
                        help="Reward penalty baseline: zero rewards for RH-detected samples, recompute advantages. "
                             "Gives RH samples negative advantages (penalizes rather than drops).")
    # Retain penalty baseline
    parser.add_argument("--retain_penalty_baseline", action="store_true", default=False,
                        help="Retain penalty baseline: replace RH rewards with retain-only reward, recompute advantages.")
    # vLLM generation server
    parser.add_argument("--vllm_server", default=None,
                        help="ZMQ socket address of vLLM server for generation "
                             "(e.g. ipc:///tmp/vllm_grpo.sock or tcp://127.0.0.1:5555). "
                             "When set, generation is offloaded to the server and adapter weights "
                             "are synced before each generation step.")
    parser.add_argument("--vllm_spawn", action="store_true", default=False,
                        help="Spawn a local vLLM server for this run (one server per run). "
                             "Uses the run's own model/mlp_config. Mutually exclusive with --vllm_server.")
    parser.add_argument("--vllm_async", action="store_true", default=False,
                        help="Use AsyncVLLMClient (for shared async server). Requires --vllm_server.")
    parser.add_argument("--vllm_gpu_memory", type=float, default=0.02,
                        help="GPU memory utilization fraction for spawned vLLM server (default: 0.02).")
    parser.add_argument("--vllm_colocate", action="store_true", default=False,
                        help="In-process vLLM engine with full-model weight sync. "
                             "Mutually exclusive with --vllm_server/--vllm_spawn.")
    parser.add_argument("--vllm_spawn_delay", type=int, default=0,
                        help="Seconds to wait before spawning the vLLM server (used to stagger "
                             "concurrent inits so each sees accurate free memory).")
    parser.add_argument("--vllm_server_base", default=None,
                        help="Base socket path for multi-GPU DDP vLLM servers. "
                             "Each DDP rank appends _rank{rank}.sock. Set by sweep.py.")
    parser.add_argument("--vllm_importance_sampling", action="store_true", default=False,
                        help="Enable importance sampling correction for vLLM generation mismatch. "
                             "Requires vLLM server to support return_logprobs.")
    # Batch capture for profiling
    parser.add_argument("--save_batch", default=None,
                        help="Path to save the first generation batch dict (.pt) for offline profiling")
    # Capture defaults so _apply_model_defaults can detect "not explicitly set"
    global _ARGPARSE_DEFAULTS
    _ARGPARSE_DEFAULTS = {a.dest: a.default for a in parser._actions if a.dest != "help"}
    return parser


def _validate_model_env_compat(args, exp_cfg):
    """Runtime checks that require imports not available at config construction time."""
    from rewards import TOKENIZER_DEPENDENT_REWARDS
    reward_component_names = {c.name for c in exp_cfg.reward.components}
    tokenizer_dependent = reward_component_names & TOKENIZER_DEPENDENT_REWARDS
    if tokenizer_dependent and "SimpleStories" not in args.model:
        raise ValueError(
            f"Reward(s) {tokenizer_dependent} use hardcoded SimpleStories token IDs "
            f"and are incompatible with model {args.model!r}.")
    if args.environment == "stories" and "SimpleStories" not in args.model:
        raise ValueError(
            f"environment='stories' is incompatible with model {args.model!r}.")


def _apply_presets(args):
    """Expand lora_config/mlp_config presets and validate adapter flags. Mutates args in place."""
    if args.adapter_type == "mlp" and args.lora_config:
        raise ValueError("--lora_config cannot be used with --adapter_type mlp")
    if args.adapter_type == "lora" and args.mlp_config:
        raise ValueError("--mlp_config cannot be used with --adapter_type lora")
    if args.lora_config:
        preset = LORA_PRESETS[args.lora_config]
        args.retain_rank = preset["retain_rank"]
        args.forget_rank = preset["forget_rank"]
        args.lora_alpha = preset["lora_alpha"]
        args.layer_stride = preset["layer_stride"]
    else:
        args.layer_stride = 1
    if args.mlp_config:
        preset = MLP_PRESETS[args.mlp_config]
        args.retain_neurons = preset["retain_neurons"]
        args.forget_neurons = preset["forget_neurons"]
        args.layer_stride = preset["layer_stride"]


def _run(args, exp_cfg=None):
    """Core training logic. Assumes CUDA device already set and output_dir exists.

    exp_cfg: pre-built ExperimentConfig. If None, builds from args (which include
    YAML config + CLI overrides merged together).

    DDP-aware: when torch.distributed is initialized, only rank 0 writes to
    stdout/log/wandb. All ranks participate in training (gradient sync via DDP).
    """
    import torch.distributed as dist
    _is_ddp = dist.is_initialized()
    _rank = dist.get_rank() if _is_ddp else 0
    _is_main = _rank == 0

    # Suppress stdout on non-main DDP ranks
    if _is_ddp and not _is_main:
        sys.stdout = open(os.devnull, "w")
        sys.stderr = open(os.devnull, "w")
    if exp_cfg is None:
        # Build ExperimentConfig from the merged args namespace.
        # This validates all cross-field constraints via Pydantic validators.
        assert args.config is not None, "--config is required"
        # Load YAML and override with all argparse values
        import yaml
        with open(args.config) as f:
            yaml_data = yaml.safe_load(f) or {}
        yaml_data["config_path"] = args.config
        # Scalar args override YAML (but don't override structured reward/rh_detector).
        # Skip None values — they represent unset argparse flags and should not
        # overwrite ExperimentConfig defaults.
        structured_keys = {"reward", "rh_detector", "hack_freq_detector", "name"}
        for k, v in vars(args).items():
            if k == "config" or k in structured_keys:
                continue
            if v is None and k not in yaml_data:
                continue
            yaml_data[k] = v
        exp_cfg = ExperimentConfig.model_validate(yaml_data)

    os.makedirs(args.output_dir, exist_ok=True)
    exp_cfg.to_yaml(os.path.join(args.output_dir, "run_config.yaml"))
    _validate_model_env_compat(args, exp_cfg)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is not None:
        pass  # tokenizer already has a pad token
    elif tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    else:
        raise ValueError(
            f"Model {args.model!r} has no pad_token and no eos_token. "
            f"Add explicit pad_token handling for this model in train.py."
        )
    tokenizer.padding_side = "left"

    # Model
    # bf16: load model in bf16 directly (no grad scaler needed, native mixed precision).
    # fp16: load model in fp32 explicitly; TRL's fp16=True handles autocast + GradScaler.
    #       Must force fp32 because some models (Qwen3) default to bf16 in their config,
    #       and GradScaler can't unscale bf16 gradients.
    # neither: respect model config default.
    if args.bf16:
        model_dtype = torch.bfloat16
    elif args.fp16:
        model_dtype = torch.float32
    else:
        model_dtype = None
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=model_dtype, attn_implementation="flash_attention_3")
    print(f"Model dtype: {next(model.parameters()).dtype}, "
          f"params: {sum(p.numel() for p in model.parameters()) / 1e9:.1f}B, "
          f"size: {sum(p.numel() * p.element_size() for p in model.parameters()) / 1e9:.1f} GiB")
    if args.no_eos:
        model.generation_config.eos_token_id = None
        model.generation_config.suppress_tokens = [tokenizer.eos_token_id]
        print("EOS disabled: suppressed EOS token, generating full max_completion_length tokens")
    else:
        model.generation_config.eos_token_id = tokenizer.eos_token_id
    if args.repetition_penalty != 1.0:
        print(f"Repetition penalty: {args.repetition_penalty}")


    # Dual adapters (skipped for adapter_type="none")
    from gradient_routing import collect_routing_params

    if args.adapter_type == "none":
        assert args.routing_mode == "none", \
            "adapter_type='none' requires routing_mode='none' (no adapters to route)"
        print("No adapters: full-parameter training")
    elif args.adapter_type == "mlp":
        from gradient_routing import apply_dual_mlp
        modified = apply_dual_mlp(
            model,
            retain_neurons=args.retain_neurons,
            forget_neurons=args.forget_neurons,
            layer_start=args.layer_start,
            layer_end=args.layer_end,
            layer_stride=args.layer_stride,
        )
        print(f"DualMLP: {len(modified)} layers "
              f"(retain={args.retain_neurons}, forget={args.forget_neurons}, "
              f"range={args.layer_start:.2f}-{args.layer_end:.2f})")
    else:
        from gradient_routing import apply_dual_lora
        modified = apply_dual_lora(
            model,
            rank=args.retain_rank,
            forget_rank=args.forget_rank,
            alpha=args.lora_alpha,
            dropout=0.0,
            layer_start=args.layer_start,
            layer_end=args.layer_end,
            layer_stride=args.layer_stride,
        )
        print(f"DualLoRA: {len(modified)} modules "
              f"(retain_rank={args.retain_rank}, forget_rank={args.forget_rank}, "
              f"range={args.layer_start:.2f}-{args.layer_end:.2f})")

    # Build adapter config for checkpoint saving
    if args.adapter_type == "none":
        adapter_config = {"adapter_type": "none"}
    elif args.adapter_type == "lora":
        adapter_config = {
            "retain_rank": args.retain_rank,
            "forget_rank": args.forget_rank,
            "lora_alpha": args.lora_alpha,
            "layer_stride": args.layer_stride,
            "layer_start": args.layer_start,
            "layer_end": args.layer_end,
        }
    else:
        adapter_config = {
            "adapter_type": "mlp",
            "retain_neurons": args.retain_neurons,
            "forget_neurons": args.forget_neurons,
            "layer_stride": args.layer_stride,
            "layer_start": args.layer_start,
            "layer_end": args.layer_end,
        }

    retain_params, forget_params = collect_routing_params(model)
    n_retain = sum(p.numel() for p in retain_params)
    n_forget = sum(p.numel() for p in forget_params)
    print(f"  Retain params: {n_retain:,}, Forget params: {n_forget:,}")

    # Resolve max_completion_length: auto-set per environment if not explicitly provided
    if args.max_completion_length is None:
        if args.environment == "stories":
            args.max_completion_length = 128
        elif args.environment == "arithmetic":
            args.max_completion_length = args.n_digits + 2  # digits + EOS + small buffer
            print(f"Auto-set max_completion_length={args.max_completion_length} "
                  f"for {args.n_digits}-digit arithmetic")
        elif args.environment == "aira":
            args.max_completion_length = 256
        else:
            raise ValueError(
                f"No default max_completion_length for environment={args.environment!r}. "
                f"Set --max_completion_length explicitly."
            )

    # Data — load via env registry
    from envs import get_env
    env_spec = get_env(args.environment)
    print(f"Loading {args.environment} training prompts...")
    train_dataset = env_spec.load_train(args)
    print(f"Loading {args.environment} eval prompts...")
    eval_dataset = env_spec.load_eval(args)

    # Wrap prompts in chat template format for instruct models.
    # Prompts that are already list[dict] (ChatRequest) are passed through unchanged —
    # this is the code path for envs like leetcode whose prompts are pre-formatted
    # by the upstream dataset processor. --system_prompt has no effect for those envs.
    is_chat_model = tokenizer.chat_template is not None
    if is_chat_model:
        def _wrap_prompts_as_chat(dataset):
            """Wrap plain-string prompts as chat messages for instruction-tuned models.

            If prompts are already list[dict] (pre-formatted ChatRequest), returns the
            dataset unchanged — the upstream processor owns the message structure.
            """
            prompts = dataset["prompt"]
            if isinstance(prompts[0], list):
                # Already ChatRequest format — pass through as-is.
                return dataset
            assert isinstance(prompts[0], str), (
                f"Unexpected prompt type {type(prompts[0])}. "
                "Prompts must be str (to be wrapped) or list[dict] (pre-formatted ChatRequest)."
            )
            chat_prompts = []
            for p in prompts:
                messages = []
                if args.system_prompt:
                    messages.append({"role": "system", "content": args.system_prompt})
                messages.append({"role": "user", "content": p})
                chat_prompts.append(messages)
            # Build new dict preserving all columns — avoids HF datasets bug where
            # remove_columns on a single-column dataset loses row count
            data = {col: dataset[col] for col in dataset.column_names if col != "prompt"}
            data["prompt"] = chat_prompts
            return Dataset.from_dict(data)
        train_dataset = _wrap_prompts_as_chat(train_dataset)
        eval_dataset = _wrap_prompts_as_chat(eval_dataset)
        first_prompt = train_dataset[0]["prompt"]
        if isinstance(first_prompt, list):
            print(f"Chat model detected — prompts already in ChatRequest format, passed through")
        else:
            print(f"Chat model detected — wrapped prompts in chat template format")

    # Environment-specific warnings
    if args.environment == "arithmetic":
        needed = args.n_digits + 2
        if args.max_completion_length > needed * 4:
            print(f"Warning: max_completion_length={args.max_completion_length} is large for "
                  f"{args.n_digits}-digit arithmetic (answer is {args.n_digits} tokens). "
                  f"Consider --max_completion_length {needed * 2}")

    reward_name = exp_cfg.reward_name
    combined_reward = exp_cfg.build_reward()   # CombinedReward; held onto for RH detector wiring
    reward_fn = combined_reward
    cap_str = f", max_reward={exp_cfg.reward.max_reward}" if exp_cfg.reward.max_reward is not None else ""
    print(f"Reward: {reward_name} {[(c.name, c.scale) for c in exp_cfg.reward.components]}{cap_str}")

    # Model/env compatibility validated in _validate_config().

    # Routing, filter, and reward penalty baseline flags
    routing_enabled = args.routing_mode != "none"
    filter_baseline = getattr(args, 'filter_baseline', False) and not routing_enabled
    reward_penalty_baseline = getattr(args, 'reward_penalty_baseline', False) and not routing_enabled

    # Stochastic routing / filter baseline: wrap reward so non-eligible samples get retain-only reward
    routed_reward = None
    if (routing_enabled or filter_baseline or reward_penalty_baseline) and args.rh_eligible_frac < 1.0:
        if args.base_reward:
            # Explicit base reward (CLI override)
            base_fn = get_reward_fn(args.base_reward)
            base_name = args.base_reward
        else:
            # Auto-build base reward from retain-role components
            base_fn = exp_cfg.build_retain_only_reward()
            base_name = "+".join(
                c.component_id for c in exp_cfg.reward.components if c.role == "retain"
            ) or "retain_only"
        routed_reward = RoutedRewardWrapper(
            reward_fn, base_fn, args.rh_eligible_frac)
        reward_fn = routed_reward
        print(f"Routed reward: {args.rh_eligible_frac:.0%} eligible for {reward_name}, "
              f"rest get {base_name}")

    # RH detector: created whenever a detector is configured and eval is running, so that
    # hack_freq appears in routing eval for both routing runs AND baselines. Routing also
    # requires it for gradient masking, but eval is the reason to build it unconditionally.
    # Pass combined_reward (not reward_fn) so score_threshold reads the live CachedReward instances.
    rh_detector = None
    eval_rh_detector = None  # base detector for eval (no recall gating)
    if args.eval_every > 0 or routing_enabled or filter_baseline or reward_penalty_baseline:
        rh_detector = exp_cfg.build_rh_detector(combined_reward)
        eval_rh_detector = rh_detector  # eval always uses base detector
        if rh_detector is not None:
            print(f"RH detector: {exp_cfg.rh_detector.name} {exp_cfg.rh_detector.params or ''}")
            recall = args.rh_detector_recall if args.rh_detector_recall is not None else exp_cfg.rh_detector_recall
            if recall < 1.0:
                base_detector = rh_detector
                def recalled_detector(completions, _recall=recall, **kwargs):
                    flags = base_detector(completions, **kwargs)
                    return [f and random.random() < _recall for f in flags]
                rh_detector = recalled_detector
                print(f"  recall={recall} (subsampling true positives)")
    if filter_baseline:
        assert rh_detector is not None, (
            "--filter_baseline requires an rh_detector in the experiment config"
        )
        print(f"Filter baseline: zeroing advantages for RH-detected samples")
    if reward_penalty_baseline:
        assert rh_detector is not None, (
            "--reward_penalty_baseline requires an rh_detector in the experiment config"
        )
        print(f"Reward penalty baseline: zeroing rewards for RH-detected samples, recomputing advantages")

    # Training config — batch_size is the effective (statistical) batch size.
    # micro_batch_size is per-GPU (the forward/backward chunk on each device);
    # gradient accumulation bridges the gap.
    # effective_batch = micro_batch_size × n_devices × grad_accum_steps
    n_devices = dist.get_world_size() if _is_ddp else 1
    if args.micro_batch_size is not None:
        # micro_batch_size is per-GPU
        per_device_bs = args.micro_batch_size
        total_micro = per_device_bs * n_devices
        assert args.batch_size % total_micro == 0, (
            f"--batch_size {args.batch_size} must be divisible by "
            f"micro_batch_size × n_devices = {per_device_bs} × {n_devices} = {total_micro}"
        )
        grad_accum_steps = args.batch_size // total_micro
        print(f"Batch size: {args.batch_size} effective = {per_device_bs} per-GPU × {n_devices} devices "
              f"× {grad_accum_steps} accum")
    else:
        assert args.batch_size % n_devices == 0, (
            f"--batch_size {args.batch_size} must be divisible by number of devices ({n_devices})"
        )
        per_device_bs = args.batch_size // n_devices
        grad_accum_steps = 1
        print(f"Batch size: {args.batch_size} total ({per_device_bs} per device × {n_devices} devices)")

    config = GRPOConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=per_device_bs,
        gradient_accumulation_steps=grad_accum_steps,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        temperature=args.temperature,
        top_k=args.top_k if args.top_k > 0 else 0,
        top_p=args.top_p,
        learning_rate=args.lr,
        optim=args.optimizer,
        num_train_epochs=args.num_epochs,
        max_steps=args.max_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        loss_type="grpo",
        repetition_penalty=args.repetition_penalty,
        beta=args.beta,
        seed=args.seed,
        bf16=args.bf16 and not args.fp16,
        fp16=args.fp16,
        report_to="wandb" if not args.no_wandb else "none",
        run_name=args.run_name or f"grpo_{reward_name}_lr{args.lr}",
        gradient_checkpointing=args.gradient_checkpointing,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        adam_beta2=args.adam_beta2,
        lr_scheduler_type=args.lr_scheduler_type,
        # Disable TRL's built-in vLLM importance sampling — our custom vLLM clients
        # handle weight sync directly, and the LoRA client doesn't return logprobs.
        # When needed, we enable it explicitly via --vllm_importance_sampling.
        vllm_importance_sampling_correction=False,
        use_liger_kernel=args.use_liger_kernel,
        # Liger SwiGLU and fused_linear_cross_entropy patches are safe for LoRA (doesn't replace
        # base layers) but crash with MLP adapters (which replace gate_proj/up_proj/down_proj).
        liger_kernel_config=(
            {"swiglu": False, "fused_linear_cross_entropy": False}
            if args.use_liger_kernel and args.adapter_type == "mlp"
            else None
        ),
        torch_compile=args.torch_compile,
    )

    # --config_check: dump effective config to file and exit before training
    if getattr(args, 'config_check', False):
        import json
        effective = {
            "GRPOConfig": {
                "learning_rate": config.learning_rate,
                "beta": config.beta,
                "weight_decay": config.weight_decay,
                "warmup_steps": config.warmup_steps,
                "adam_beta2": config.adam_beta2,
                "lr_scheduler_type": str(config.lr_scheduler_type),
                "temperature": config.temperature,
                "top_k": config.top_k,
                "top_p": config.top_p,
                "per_device_train_batch_size": config.per_device_train_batch_size,
                "gradient_accumulation_steps": config.gradient_accumulation_steps,
                "num_generations": config.num_generations,
                "max_completion_length": config.max_completion_length,
                "max_steps": config.max_steps,
                "save_steps": config.save_steps,
                "bf16": config.bf16,
                "fp16": config.fp16,
                "seed": config.seed,
                "loss_type": config.loss_type,
                "repetition_penalty": config.repetition_penalty,
            },
            "args": {
                "adapter_type": args.adapter_type,
                "retain_rank": args.retain_rank,
                "forget_rank": args.forget_rank,
                "lora_alpha": args.lora_alpha,
                "routing_mode": args.routing_mode,
                "environment": args.environment,
                "model": args.model,
                "top_k_raw": args.top_k,
            },
            "ExperimentConfig": {
                "reward_components": [(c.name, c.scale, c.role) for c in exp_cfg.reward.components],
                "max_reward": exp_cfg.reward.max_reward,
                "rh_detector": exp_cfg.rh_detector.name if exp_cfg.rh_detector else None,
                "rh_detector_recall": exp_cfg.rh_detector_recall,
            },
        }
        out_path = os.path.join(args.output_dir, "config_check.json")
        os.makedirs(args.output_dir, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(effective, f, indent=2)
        print(f"Config check written to {out_path}")
        return

    if not args.no_wandb:
        os.environ.setdefault("WANDB_PROJECT", args.wandb_project)

    # Build eval reward fns whenever eval_every > 0
    eval_metrics = {}
    if args.eval_every > 0:
        eval_metrics = exp_cfg.build_eval_metrics(rh_detector=eval_rh_detector)

    # vLLM client (optional — offloads generation to vLLM engine)
    # Constraint validation is in _validate_config().
    vllm_client = None
    _vllm_server_proc = None
    if args.vllm_server:
        if args.vllm_async:
            from vllm_client import AsyncVLLMClient
            print(f"[vLLM] Connecting to async server at {args.vllm_server}...")
            vllm_client = AsyncVLLMClient(args.vllm_server)
        elif args.adapter_type == "lora":
            from vllm_lora import VLLMLoRAClient
            print(f"[vLLM] Connecting to LoRA server at {args.vllm_server}...")
            vllm_client = VLLMLoRAClient(args.vllm_server)
        else:
            from vllm_client import VLLMClient
            print(f"[vLLM] Connecting to server at {args.vllm_server}...")
            vllm_client = VLLMClient(args.vllm_server)
    elif args.vllm_spawn:
        import multiprocessing as _mp
        import tempfile
        if args.vllm_spawn_delay > 0:
            print(f"[vLLM] Waiting {args.vllm_spawn_delay}s before spawning server (stagger init)...")
            time.sleep(args.vllm_spawn_delay)
        _socket_path = f"ipc:///tmp/vllm_grpo_{os.getpid()}.sock"
        _ready_file = tempfile.mktemp(prefix="vllm_ready_", suffix=f"_{os.getpid()}")
        _ctx = _mp.get_context("spawn")
        _vllm_server_proc = _ctx.Process(
            target=_spawn_vllm_server,
            args=(args.model, args.mlp_config, args.vllm_gpu_memory, _socket_path, _ready_file,
                  args.layer_start, args.layer_end, args.layer_stride),
            # daemon=False so vLLM v1 engine can spawn its own CoreEngineProcManager children
        )
        _vllm_server_proc.start()
        print(f"[vLLM] Spawned server at {_socket_path} (pid={_vllm_server_proc.pid})")
        _t0 = time.time()
        while not os.path.exists(_ready_file):
            assert time.time() - _t0 < 180, "vLLM server failed to start within 180s"
            assert _vllm_server_proc.is_alive(), "vLLM server process died during startup"
            time.sleep(0.5)
        os.unlink(_ready_file)
        from vllm_client import VLLMClient
        vllm_client = VLLMClient(_socket_path)
        print(f"[vLLM] Server ready")
    elif args.vllm_colocate:
        from vllm_colocate import VLLMColocateClient
        print(f"[vLLM] Creating colocate engine for {args.model}...")
        vllm_client = VLLMColocateClient(
            model_name=args.model,
            gpu_memory_utilization=args.vllm_gpu_memory,
        )

    trainer = SampleGRPOTrainer(
        model=model,
        args=config,
        reward_funcs=reward_fn,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        gradient_routing_enabled=routing_enabled,
        retain_params=retain_params,
        forget_params=forget_params,
        routing_mode=args.routing_mode,
        rh_detector=rh_detector,
        eval_every=args.eval_every,
        eval_metrics=eval_metrics,
        routed_reward=routed_reward,
        coherence=args.coherence,
        coherence_every=args.coherence_every,
        coherence_gen=args.coherence_gen,
        coherence_batch_size=args.coherence_batch_size,
        coherence_hackable_only=args.coherence_hackable_only,
        filter_baseline=filter_baseline,
        reward_penalty_baseline=reward_penalty_baseline,
        verbose=args.verbose,
        adapter_config=adapter_config,
        retain_mode=args.retain_mode,
        retain_penalty=args.retain_penalty,
        combined_reward=combined_reward,
        advantage_type=args.advantage_type,
        reinforce_buffer_size=args.reinforce_buffer_size,
        reinforce_normalize_std=args.reinforce_normalize_std,
        vllm_client=vllm_client,
        adapter_type=args.adapter_type,
        liger_chunk_size=args.liger_chunk_size,
    )
    trainer._environment = args.environment
    trainer._n_digits = args.n_digits
    trainer._env_spec = env_spec
    trainer._env_args = args
    trainer._save_batch_path = getattr(args, 'save_batch', None)

    # Fix TRL double-scaling bug: TRL's _compute_loss already divides loss by
    # gradient_accumulation_steps (grpo_trainer.py:2153), but accelerator.backward()
    # divides again (accelerator.py:2828). This halves the effective LR per GAS step.
    # Setting accelerator's GAS to 1 disables its redundant division.
    # The Trainer's loop control (microbatch counting, sync gating) uses
    # args.gradient_accumulation_steps, which is unaffected.
    trainer.accelerator.gradient_accumulation_steps = 1

    # Optionally enable vLLM importance sampling correction to account for
    # distribution mismatch between vLLM's generation and HF's forward pass.
    if vllm_client is not None and args.vllm_importance_sampling:
        trainer.use_vllm = True
        trainer.vllm_importance_sampling_correction = True
        trainer.vllm_importance_sampling_mode = "token_truncate"
        trainer.vllm_importance_sampling_cap = 10.0
        print("[vLLM] Enabled importance sampling correction for vLLM generation")

    if not args.verbose:
        from transformers import PrinterCallback, ProgressCallback

        class QuietProgressCallback(ProgressCallback):
            def on_log(self, args, state, control, logs=None, **kwargs):
                pass

        trainer.remove_callback(PrinterCallback)
        trainer.remove_callback(ProgressCallback)
        trainer.add_callback(QuietProgressCallback)

    # Step-0 eval: capture base model performance before training.
    # Off by default (--eval_at_start) since it uses HF generate inline,
    # which OOMs with vLLM colocated on large models.
    if args.eval_at_start and trainer.eval_every > 0 and trainer.eval_metrics:
        trainer._run_routing_eval()

    try:
        trainer.train(resume_from_checkpoint=args.resume_from)
    except KeyboardInterrupt:
        jsonl_path = os.path.join(args.output_dir, "routing_eval.jsonl")
        if os.path.exists(jsonl_path):
            print("\nInterrupted — generating eval plots...")
            from plot_routing import main as plot_main
            sys.argv = ["plot_routing", args.output_dir]
            plot_main()
        else:
            print("\nInterrupted — no eval data to plot.")
        return
    finally:
        if torch.cuda.is_available():
            peak_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
            print(f"Peak GPU memory: {peak_mb:.0f} MB ({peak_mb/1024:.1f} GB)")
        # Release slot back to shared async server (zeros weights, frees slot for next run)
        if args.vllm_async and vllm_client is not None and trainer._vllm_experiment_id is not None:
            try:
                vllm_client.release(trainer._vllm_experiment_id)
            except Exception as e:
                print(f"[vLLM] Warning: release failed: {e}")
        if _vllm_server_proc is not None:
            try:
                vllm_client.shutdown()
            except Exception:
                pass
            _vllm_server_proc.join(timeout=5)
            if _vllm_server_proc.is_alive():
                _vllm_server_proc.kill()
        elif args.vllm_colocate and vllm_client is not None:
            try:
                vllm_client.shutdown()
            except Exception:
                pass


def train_main(params: dict):
    """Programmatic entry point for sweep.py.

    params is a flat dict of training parameters. Missing keys receive argparse
    defaults. May include an 'exp_cfg' key (ExperimentConfig instance) to bypass
    YAML loading entirely. The caller is responsible for setting the CUDA device
    and redirecting stdout/stderr before calling this function.
    """
    exp_cfg = params.pop("exp_cfg", None)
    parser = _make_parser()
    args = parser.parse_args([])  # populate all defaults

    # Reject unknown keys early — typos silently fall back to defaults otherwise
    valid_dests = {a.dest for a in parser._actions}
    # Also accept keys that are ExperimentConfig fields but not argparse args
    # (e.g. vllm_dtype which is sweep-only, stripped before reaching train_main)
    ec_fields = set(ExperimentConfig.model_fields)
    unknown = set(params) - valid_dests - ec_fields
    assert not unknown, (
        f"Unknown param(s) in train_main: {sorted(unknown)}."
    )

    # Apply YAML scalars as defaults (explicit params override)
    if exp_cfg is None and "config" in params:
        import yaml
        with open(params["config"]) as f:
            yaml_data = yaml.safe_load(f) or {}
        # Flatten the training: section from YAML
        training_section = yaml_data.get("training") or {}
        if isinstance(training_section, dict):
            for k, v in training_section.items():
                if v is not None and k not in params:
                    setattr(args, k, v)
        # Also apply top-level scalar YAML keys (e.g. environment, max_completion_length)
        structured_keys = {"reward", "rh_detector", "hack_freq_detector", "name", "training"}
        for k, v in yaml_data.items():
            if k not in structured_keys and v is not None and k not in params:
                if hasattr(args, k):
                    setattr(args, k, v)

    for k, v in params.items():
        if hasattr(args, k):
            setattr(args, k, v)
    _apply_model_defaults(args)
    _apply_presets(args)

    if args.world_size > 1:
        # Multi-GPU DDP: spawn one worker per GPU via torch.multiprocessing
        import torch.multiprocessing as mp
        # Find a free port for NCCL rendezvous
        import socket as _socket
        with _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            master_port = s.getsockname()[1]
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(master_port)
        print(f"[DDP] Spawning {args.world_size} workers (MASTER_PORT={master_port})")
        mp.spawn(
            _ddp_worker,
            args=(args.world_size, args, exp_cfg),
            nprocs=args.world_size,
            join=True,
        )
    else:
        torch.cuda.set_device(args.gpu_id)
        _run(args, exp_cfg)


def _ddp_worker(rank, world_size, args, exp_cfg):
    """Entry point for each DDP rank, called by torch.multiprocessing.spawn."""
    import torch.distributed as dist

    # Set env vars that Accelerate/HF Trainer expect for DDP
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)

    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    # Per-rank vLLM socket: convention is {base}_rank{rank}.sock
    if args.vllm_server_base:
        args.vllm_server = f"{args.vllm_server_base}_rank{rank}.sock"

    # Only rank 0 logs to wandb; others suppress
    if rank != 0:
        args.no_wandb = True

    try:
        _run(args, exp_cfg)
    finally:
        dist.destroy_process_group()


def main():
    parser = _make_parser()
    args = parser.parse_args()

    # Load training defaults from --config YAML's `training:` section (CLI still overrides)
    if args.config:
        with open(args.config) as f:
            raw_config = yaml.safe_load(f) or {}
        training_dict = raw_config.get("training") or {}
        training_dict = {k: v for k, v in training_dict.items() if v is not None}
        if training_dict:
            valid_dests = {a.dest for a in parser._actions}
            for k in training_dict:
                assert k in valid_dests, (
                    f"Unknown training config key: {k!r}. Valid keys: {sorted(valid_dests)}"
                )
            parser.set_defaults(**training_dict)
            args = parser.parse_args()

    _apply_model_defaults(args)
    _apply_presets(args)
    torch.cuda.set_device(args.gpu_id)

    # Tee stdout/stderr to train.log in output_dir (CLI only)
    os.makedirs(args.output_dir, exist_ok=True)
    log_path = os.path.join(args.output_dir, "train.log")
    sys.stdout = Tee(log_path, sys.stdout)
    sys.stderr = Tee(log_path, sys.stderr)

    exp_cfg = ExperimentConfig.from_yaml(args.config)
    _run(args, exp_cfg)


if __name__ == "__main__":
    main()
