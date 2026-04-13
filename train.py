"""GRPO training on SimpleStories with TRL, with optional gradient routing."""

import argparse
import json
import os

from dotenv import load_dotenv
load_dotenv()
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", os.path.expanduser("~/torch_cache"))
os.environ.setdefault("TRITON_CACHE_DIR", os.path.expanduser("~/triton_cache"))
import random
import sys
import time

import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import yaml
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOTrainer, GRPOConfig
from trl.trainer.utils import shuffle_sequence_dict, split_tensor_dict
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
    def fileno(self):
        return self.stream.fileno()
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
    os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)  # vLLM 0.17 CuMemAllocator rejects expandable_segments
    os.environ.setdefault("VLLM_LOGGING_LEVEL", "ERROR")
    os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"

    from vllm_utils import MLP_PRESETS
    from vllm_server import VLLMServer

    preset = MLP_PRESETS[mlp_config]
    server = VLLMServer(
        socket_addr=socket_path,
        # 4 slots: 1 training + 3 eval adapter modes (both, retain_only,
        # forget_only) registered concurrently.
        max_experiments=4,
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
        "gpu_batch_size": 16,
        "max_tokens_per_microbatch": 12000,
        "lr": 7e-5,
        "beta": 1e-3,
        "num_generations": 16,
        "bf16": True,
        "top_p": 0.95,
        "max_steps": 200,
        "lr_scheduler_type": "cosine",
        "weight_decay": 0.1,
        "adam_beta2": 0.99,
    },
    "Qwen3-4B": {
        "gpu_batch_size": 8,
        "max_tokens_per_microbatch": 8000,
        "lr": 7e-5,
        "beta": 1e-3,
        "num_generations": 16,
        "bf16": True,
        "top_p": 0.95,
        "max_steps": 200,
        "lr_scheduler_type": "cosine",
        "weight_decay": 0.1,
        "adam_beta2": 0.99,
    },
}


def _apply_model_defaults(args, explicit_keys=None):
    """Apply MODEL_DEFAULTS for the first matching model key.

    Only fills in values that weren't explicitly set.  explicit_keys is a set
    of param names that were provided by the caller (sweep dict or CLI) and
    should never be overwritten — even when their value happens to match the
    argparse default.
    """
    explicit_keys = explicit_keys or set()
    for pattern, defaults in MODEL_DEFAULTS.items():
        if pattern in args.model:
            applied = []
            for key, value in defaults.items():
                if key in explicit_keys:
                    continue
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
    "m64_retain_only": {"retain_neurons": 64, "forget_neurons": 0, "layer_stride": 1},
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


# Keys whose sequence dimension (dim=1) should be trimmed on the completion side (right-padded).
_COMPLETION_TRIM_KEYS = {"completion_ids", "completion_mask", "old_per_token_logps",
                         "ref_per_token_logps", "sampling_per_token_logps"}
# Keys whose sequence dimension (dim=1) should be trimmed on the prompt side (left-padded).
_PROMPT_TRIM_KEYS = {"prompt_ids", "prompt_mask"}


def _pack_by_tokens(token_counts, indices, max_tokens):
    """First-fit decreasing bin packing by token count.

    Args:
        token_counts: per-sample token counts (full batch).
        indices: which samples to pack (global indices into token_counts).
        max_tokens: max total tokens per bin.

    Returns:
        List of lists of global indices (each inner list = one microbatch).
    """
    if not indices:
        return []
    # Sort by token count descending for better packing
    pairs = sorted([(i, token_counts[i]) for i in indices], key=lambda x: -x[1])
    bins = []  # list of (current_tokens, [indices])
    for idx, tokens in pairs:
        placed = False
        for b in range(len(bins)):
            if bins[b][0] + tokens <= max_tokens:
                bins[b] = (bins[b][0] + tokens, bins[b][1] + [idx])
                placed = True
                break
        if not placed:
            bins.append((tokens, [idx]))
    return [b[1] for b in bins]


def _trim_and_slice(inputs, indices):
    """Slice batch by integer indices and trim padding to local max lengths.

    Completion tensors (right-padded) are trimmed to the microbatch's max actual
    completion length. Prompt tensors (left-padded) are trimmed to the microbatch's
    max actual prompt length. This reduces compute for microbatches where all
    sequences are shorter than the global max.
    """
    device = next(v.device for v in inputs.values() if isinstance(v, torch.Tensor))
    idx = torch.tensor(indices, device=device, dtype=torch.long)
    n = next(v.shape[0] for v in inputs.values()
             if isinstance(v, torch.Tensor) and v.ndim > 0)

    result = {}
    for key, val in inputs.items():
        if val is None:
            result[key] = None
        elif isinstance(val, torch.Tensor) and val.ndim > 0 and val.shape[0] == n:
            result[key] = val[idx]
        else:
            result[key] = val

    # Trim completion side (right-padded): slice [:, :max_actual_comp_len]
    if "completion_mask" in result and result["completion_mask"] is not None:
        max_comp = result["completion_mask"].sum(dim=1).max().item()
        if max_comp > 0:
            for key in _COMPLETION_TRIM_KEYS:
                if key in result and result[key] is not None and result[key].ndim >= 2:
                    result[key] = result[key][:, :max_comp]

    # Trim prompt side (left-padded): slice [:, first_real_token:]
    if "prompt_mask" in result and result["prompt_mask"] is not None:
        # Find first column that has any real token across the microbatch
        any_real = result["prompt_mask"].any(dim=0)  # (P,)
        if any_real.any():
            first_real = any_real.nonzero(as_tuple=True)[0][0].item()
            if first_real > 0:
                for key in _PROMPT_TRIM_KEYS:
                    if key in result and result[key] is not None and result[key].ndim >= 2:
                        result[key] = result[key][:, first_real:]

    return result


def _pack_for_forward(inputs, indices):
    """Pack selected samples into a flat (1, total_tokens) tensor for padding-free forward.

    Concatenates real tokens (no padding) from each sequence into a single flat
    tensor with position_ids that reset at sequence boundaries. HF flash attention
    detects this packed format via _is_packed_sequence and routes to flash_attn_varlen.

    Returns a dict with packed tensors + metadata for unpacking after forward.
    """
    device = next(v.device for v in inputs.values() if isinstance(v, torch.Tensor))
    n_global = next(v.shape[0] for v in inputs.values()
                    if isinstance(v, torch.Tensor) and v.ndim > 0)

    all_input_ids = []
    all_position_ids = []
    all_completion_mask = []
    seq_boundaries = []  # (prompt_len, completion_len) per sequence

    has_old_logps = "old_per_token_logps" in inputs and inputs["old_per_token_logps"] is not None
    has_ref_logps = "ref_per_token_logps" in inputs and inputs["ref_per_token_logps"] is not None
    all_old_logps = [] if has_old_logps else None
    all_ref_logps = [] if has_ref_logps else None
    all_comp_ids = []

    for i in indices:
        # Extract real prompt tokens (skip left-padding)
        p_mask = inputs["prompt_mask"][i]
        real_positions = p_mask.nonzero(as_tuple=True)[0]
        if len(real_positions) > 0:
            p_start = real_positions[0].item()
            p_real = inputs["prompt_ids"][i, p_start:]
        else:
            p_real = inputs["prompt_ids"][i, :0]  # empty
        p_len = p_real.shape[0]

        # Extract real completion tokens (skip right-padding)
        c_mask = inputs["completion_mask"][i]
        c_len = c_mask.sum().item()
        c_real = inputs["completion_ids"][i, :c_len]

        # Concatenate prompt + completion for this sequence
        seq_ids = torch.cat([p_real, c_real])
        seq_len = seq_ids.shape[0]

        all_input_ids.append(seq_ids)
        all_position_ids.append(torch.arange(seq_len, device=device))
        all_completion_mask.append(torch.cat([
            torch.zeros(p_len, dtype=torch.long, device=device),
            torch.ones(c_len, dtype=torch.long, device=device),
        ]))
        all_comp_ids.append(inputs["completion_ids"][i, :c_len])

        if has_old_logps:
            all_old_logps.append(inputs["old_per_token_logps"][i, :c_len])
        if has_ref_logps:
            all_ref_logps.append(inputs["ref_per_token_logps"][i, :c_len])

        seq_boundaries.append((p_len, c_len))

    # Build packed tensors
    packed_input_ids = torch.cat(all_input_ids).unsqueeze(0)        # (1, T)
    packed_position_ids = torch.cat(all_position_ids).unsqueeze(0)  # (1, T)
    packed_completion_mask = torch.cat(all_completion_mask).unsqueeze(0)  # (1, T)

    # Repad per-sequence completion data to (N, max_comp_len) for loss computation
    max_comp_len = max(c for _, c in seq_boundaries) if seq_boundaries else 0
    n_seqs = len(indices)

    comp_ids_padded = torch.zeros(n_seqs, max_comp_len, dtype=torch.long, device=device)
    comp_mask_padded = torch.zeros(n_seqs, max_comp_len, dtype=torch.long, device=device)
    old_logps_padded = torch.zeros(n_seqs, max_comp_len, device=device) if has_old_logps else None
    ref_logps_padded = torch.zeros(n_seqs, max_comp_len, device=device) if has_ref_logps else None

    for j, (_, c_len) in enumerate(seq_boundaries):
        if c_len > 0:
            comp_ids_padded[j, :c_len] = all_comp_ids[j]
            comp_mask_padded[j, :c_len] = 1
            if has_old_logps:
                old_logps_padded[j, :c_len] = all_old_logps[j]
            if has_ref_logps:
                ref_logps_padded[j, :c_len] = all_ref_logps[j]

    # Index per-sequence scalars
    idx_t = torch.tensor(indices, device=device, dtype=torch.long)
    advantages = inputs["advantages"][idx_t]

    return {
        "packed_input_ids": packed_input_ids,
        "packed_position_ids": packed_position_ids,
        "packed_completion_mask": packed_completion_mask,
        "seq_boundaries": seq_boundaries,
        "completion_ids": comp_ids_padded,
        "completion_mask": comp_mask_padded,
        "advantages": advantages,
        "old_per_token_logps": old_logps_padded,
        "ref_per_token_logps": ref_logps_padded,
        "num_sequences": n_seqs,
        "max_comp_len": max_comp_len,
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
                 reward_penalty_amount=None,
                 verbose=False, adapter_config=None,
                 retain_mode="default", retain_penalty=0.0,
                 combined_reward=None,
                 advantage_type="grpo",
                 reinforce_buffer_size=2048,
                 reinforce_normalize_std=False,
                 coherence="none", coherence_every=1,
                 coherence_gen="retain_only",
                 coherence_rh_mode="filter",
                 coherence_rh_penalty=3.0,
                 vllm_client=None,
                 adapter_type="lora",
                 liger_chunk_size=64,
                 save_adapter_only=False,
                 **kwargs):
        # Ref-model-via-disabled-adapters optimization: when the model has DualLoRA/
        # DualMLP adapters, computing ref logprobs by running the same model with
        # all adapter scales set to 0 is equivalent to running the frozen base model
        # (since DualLoRALinear.forward reduces to base_layer(x) when both scales are 0).
        # This avoids instantiating a second copy of the base model as self.ref_model.
        # We opt in by monkey-patching trl.trainer.grpo_trainer.create_model_from_path to
        # return None during super().__init__, so ref_model stays None everywhere
        # (all downstream code paths already `if self.ref_model is not None`-guard).
        # The ref forward is then routed through `disabled_dual_adapters(model)` in
        # trl_overrides.generate_and_score_completions.
        from gradient_routing import has_dual_adapters
        model_arg = kwargs["model"] if "model" in kwargs else (args[0] if len(args) > 0 else None)
        training_args = kwargs["args"] if "args" in kwargs else (args[1] if len(args) > 1 else None)
        beta = getattr(training_args, "beta", 0.0) if training_args is not None else 0.0
        _use_adapter_ref = (
            model_arg is not None
            and beta != 0.0
            and has_dual_adapters(model_arg)
        )
        if _use_adapter_ref:
            import trl.trainer.grpo_trainer as _gt
            _orig_create = _gt.create_model_from_path
            _gt.create_model_from_path = lambda *a, **kw: None
            try:
                super().__init__(*args, **kwargs)
            finally:
                _gt.create_model_from_path = _orig_create
            assert self.ref_model is None, (
                "Expected ref_model=None after create_model_from_path patch. "
                "TRL's GRPOTrainer ref-model branching may have changed."
            )
            self._ref_via_disabled_adapters = True
            print("Ref logprobs: using disabled-adapter trick (no separate ref_model copy)")
        else:
            super().__init__(*args, **kwargs)
            self._ref_via_disabled_adapters = False
        # Override TRL's default LigerFusedLinearGRPOLoss(chunk_size=1) with our chunk_size
        if self.use_liger_kernel and hasattr(self, "liger_grpo_loss"):
            from liger_kernel.chunked_loss import LigerFusedLinearGRPOLoss
            self.liger_grpo_loss = LigerFusedLinearGRPOLoss(
                beta=self.beta,
                compiled=False,
                chunk_size=liger_chunk_size,
                epsilon_low=self.epsilon_low,
                epsilon_high=self.epsilon_high,
                temperature=self.temperature,
                use_ref_model=self.beta != 0.0,
                loss_type=self.loss_type,
                max_completion_length=self.max_completion_length,
            )
        print(f"Liger GRPO loss: {'active' if self.use_liger_kernel else 'inactive'}"
              f"{f', chunk_size={self.liger_grpo_loss.chunk_size}' if self.use_liger_kernel and hasattr(self, 'liger_grpo_loss') else ''}")
        # --- wandb: we own all logging, TRL's WandbCallback is removed ---
        # See CLAUDE.md "wandb Logging" section for rationale.
        self._samples_seen = 0
        self._pending_eval_wandb = {}
        self._eval_scoring_thread = None
        self._eval_scoring_error = None
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
        self._reward_penalty_amount = reward_penalty_amount
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
        # Coherence training (rollout-level: alternates entire rollout cycles)
        self._coherence = coherence
        self._coherence_every = coherence_every
        self._coherence_gen = coherence_gen
        self._coherence_rh_mode = coherence_rh_mode
        self._coherence_rh_penalty = coherence_rh_penalty
        self._coherence_rollout_counter = 0
        self._is_coherence_rollout = False
        # vLLM HTTP server for generation
        self._vllm_client = vllm_client
        self._vllm_experiment_id = None
        self._eval_experiment_ids = None  # {mode_name: eid} for concurrent eval
        if vllm_client is not None:
            self._vllm_experiment_id = vllm_client.register()
            print(f"[vLLM] Registered experiment {self._vllm_experiment_id}")
            # Concurrent eval: reuse training slot for "both" mode, register 2
            # extra slots for retain_only and forget_only. Requires the client to
            # support generate_multi (MLP adapter path; LoRA client does not).
            # Fail loud on slot exhaustion — silent fallback would mask the perf
            # cost and mislead about which eval path actually ran.
            if hasattr(vllm_client, "generate_multi"):
                eid_both = vllm_client.register()
                eid_retain = vllm_client.register()
                eid_forget = vllm_client.register()
                self._eval_experiment_ids = {
                    "both": eid_both,
                    "retain_only": eid_retain,
                    "forget_only": eid_forget,
                }
                print(f"[vLLM] Registered concurrent eval slots: "
                      f"both={eid_both}, retain_only={eid_retain}, forget_only={eid_forget}")
        self._save_adapter_only = save_adapter_only
        # Phase timing: rollout (generation+scoring) vs update (gradients)
        self._last_rollout_time = 0.0
        self._accum_update_time = 0.0
        self._last_step_end_time = None
        self._last_grpo_iter_end_time = None  # set at end of last micro-batch of each logical step
        self._post_step_accum = 0.0  # accumulates optimizer.step + clip + zero_grad time

    def create_optimizer(self):
        """Wrap optimizer.step(), clip_grad_norm_, zero_grad() and
        get_batch_samples with timing instrumentation."""
        super().create_optimizer()
        _trainer = self

        # --- timing/post_step: optimizer + clip + zero_grad combined ---
        _orig_step = self.optimizer.step
        _orig_zero_grad = self.model.zero_grad
        _orig_clip = self.accelerator.clip_grad_norm_

        def _timed_step(*a, **kw):
            t0 = time.perf_counter()
            result = _orig_step(*a, **kw)
            _trainer._post_step_accum += time.perf_counter() - t0
            return result
        self.optimizer.step = _timed_step

        def _timed_zero_grad(*a, **kw):
            t0 = time.perf_counter()
            result = _orig_zero_grad(*a, **kw)
            _trainer._post_step_accum += time.perf_counter() - t0
            return result
        self.model.zero_grad = _timed_zero_grad

        def _timed_clip(*a, **kw):
            t0 = time.perf_counter()
            result = _orig_clip(*a, **kw)
            _trainer._post_step_accum += time.perf_counter() - t0
            return result
        self.accelerator.clip_grad_norm_ = _timed_clip

        # --- timing/between_grpo_iters/dataloader ---
        _orig_gbs = self.get_batch_samples
        _orig_get_num_items = self._get_num_items_in_batch
        def _timed_gbs(epoch_iterator, num_batches, device):
            t0 = time.perf_counter()
            batch_samples = []
            for i in range(num_batches):
                try:
                    batch_samples.append(next(epoch_iterator))
                except StopIteration:
                    break
            num_items_in_batch = _orig_get_num_items(batch_samples, device)
            _trainer._metrics.setdefault("train", {}).setdefault(
                "timing/between_grpo_iters/dataloader", []
            ).append(time.perf_counter() - t0)
            return batch_samples, num_items_in_batch
        self.get_batch_samples = _timed_gbs

        return self.optimizer

    def get_train_dataloader(self):
        """Override: disable device_placement on the DataLoader.

        The accelerator wrapper calls send_to_device() on every batch, which
        recursively walks the entire nested dict structure (chat-format prompts,
        metadata lists, etc.) looking for tensors to move to GPU. With no tensors
        in the prompt data, this is pure overhead — ~100ms per batch × 256 batches
        = ~26s per step. Passing device_placement=False skips send_to_device.
        """
        _orig_prepare = self.accelerator.prepare_data_loader
        def _prepare_no_device_placement(dl, **kw):
            kw['device_placement'] = False
            return _orig_prepare(dl, **kw)
        self.accelerator.prepare_data_loader = _prepare_no_device_placement
        dl = super().get_train_dataloader()
        self.accelerator.prepare_data_loader = _orig_prepare
        return dl

    # --- Timestamped trace prints for diagnosing wall-clock gaps ---
    _TRACE_ENABLED = True  # flip to False to silence

    def _ts(self, label):
        """Print a timestamped trace line (for diagnosing timing gaps in train.log)."""
        if self._TRACE_ENABLED:
            import datetime
            now = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
            step = getattr(self.state, 'global_step', '?')
            mb = getattr(self, '_step', '?')
            print(f"[trace {now}] step={step} mb={mb} {label}", flush=True)

    def get_train_dataloader(self):
        """Override: disable device_placement on the DataLoader.

        The accelerator wrapper calls send_to_device() on every batch, which
        recursively walks the entire nested dict structure (chat-format prompts,
        metadata lists, etc.) looking for tensors to move to GPU. With no tensors
        in the prompt data, this is pure overhead — ~100ms per batch × 256 batches
        = ~16s per step.
        """
        _orig_prepare = self.accelerator.prepare_data_loader
        def _prepare_no_device_placement(dl, **kw):
            kw['device_placement'] = False
            return _orig_prepare(dl, **kw)
        self.accelerator.prepare_data_loader = _prepare_no_device_placement
        dl = super().get_train_dataloader()
        self.accelerator.prepare_data_loader = _orig_prepare
        return dl

    def _time(self, key):
        """Context manager: times a block and appends seconds to self._metrics["train"][key]."""
        return _Timer(self._metrics, key)

    def _save_checkpoint(self, model, trial):
        with self._time("timing/checkpoint"):
            if self._save_adapter_only:
                self._save_adapter_checkpoint()
            else:
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

    def _save_adapter_checkpoint(self):
        """Save only adapter (trainable) weights, optimizer, scheduler, and trainer state."""
        from safetensors.torch import save_file
        checkpoint_dir = os.path.join(
            self.args.output_dir,
            f"checkpoint-{self.state.global_step}",
        )
        os.makedirs(checkpoint_dir, exist_ok=True)
        # Save only trainable parameters
        adapter_state = {k: v.cpu() for k, v in self.model.named_parameters() if v.requires_grad}
        save_file(adapter_state, os.path.join(checkpoint_dir, "model.safetensors"))
        # Save optimizer and scheduler for resumption
        torch.save(self.optimizer.state_dict(), os.path.join(checkpoint_dir, "optimizer.pt"))
        if self.lr_scheduler is not None:
            torch.save(self.lr_scheduler.state_dict(), os.path.join(checkpoint_dir, "scheduler.pt"))
        # Save trainer state (has log_history with per-step metrics)
        self.state.save_to_json(os.path.join(checkpoint_dir, "trainer_state.json"))

    def _eval_due_this_rollout(self):
        """Check if eval should fire on this rollout.

        In this repo each rollout corresponds to a single optimizer step (the
        custom loop regenerates every step), so the eval is labeled with the
        current global_step — i.e. the weights actually used to generate the
        eval samples. Note that TRL's self.args.steps_per_generation defaults
        to gradient_accumulation_steps and does not reflect the real cadence
        here, so we must not use it as a step offset.
        """
        if not (self.eval_every > 0 and self.eval_metrics
                and self.accelerator.is_main_process):
            return False
        step = self.state.global_step
        return (step > 0
                and step - self._last_routing_eval_step >= self.eval_every)

    def _prepare_eval_for_rollout(self):
        """Load eval prompts and tokenize for piggybacked vLLM generation.

        Returns (eval_prompt_ids, eval_data, eval_max_tokens) or None if eval
        is not due or can't be piggybacked (no eval_experiment_ids).
        """
        if not self._eval_due_this_rollout():
            return None
        if self._eval_experiment_ids is None:
            return None  # fall back to _run_routing_eval in log()

        # Wait for any previous eval scoring to finish.
        self._wait_for_eval_scoring()

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
            eval_data = None
            eval_max_tokens = n_digits + 2
        else:
            eval_prompts = None
            eval_data = None
            eval_max_tokens = 128

        if eval_prompts is None:
            from eval_utils import _load_eval_prompts
            eval_prompts = _load_eval_prompts(n=64)

        # Chat-wrap plain-string prompts for instruct models. train.py's
        # _wrap_prompts_as_chat applies the same transform to train_dataset
        # and eval_dataset; without mirroring it here, piggyback eval reaches
        # vLLM as raw strings that the model never saw during training.
        env_args = getattr(self, '_env_args', None)
        sys_prompt = getattr(env_args, 'system_prompt', "") if env_args else ""
        if self.processing_class.chat_template is not None:
            def _wrap(p):
                if isinstance(p, list):
                    return p
                msgs = []
                if sys_prompt:
                    msgs.append({"role": "system", "content": sys_prompt})
                msgs.append({"role": "user", "content": p})
                return msgs
            eval_prompts = [_wrap(p) for p in eval_prompts]

        from trl import is_conversational
        if is_conversational({"prompt": eval_prompts[0]}):
            prompt_texts = [
                self.processing_class.apply_chat_template(
                    p, add_generation_prompt=True, tokenize=False,
                    enable_thinking=False,
                )
                for p in eval_prompts
            ]
        else:
            prompt_texts = eval_prompts

        eval_prompt_ids = [
            self.processing_class.encode(p, add_special_tokens=False)
            for p in prompt_texts
        ]
        return eval_prompt_ids, eval_prompts, eval_data, eval_max_tokens

    def _generate_single_turn(self, prompts):
        """Override: use vLLM HTTP server for generation when configured,
        otherwise fall back to bulk-CPU contention fix (trl_overrides).

        When eval is due, piggybacks eval generation onto the same vLLM session
        (wake/sleep cycle) as the training rollout, using generate_multi to batch
        training + eval prompts in a single server call.
        """
        if self._vllm_client is None:
            return generate_single_turn(self, prompts)

        from trl import is_conversational

        client = self._vllm_client
        eid = self._vllm_experiment_id
        mode = "train" if self.model.training else "eval"
        num_generations = self.num_generations if mode == "train" else self.num_generations_eval

        # Check if eval should piggyback on this rollout.
        eval_info = self._prepare_eval_for_rollout() if mode == "train" else None

        # Wake vLLM if it was sleeping: free training tensors first so vLLM
        # can reclaim the GPU memory for KV cache and weights.
        m = self._metrics.setdefault("train", {})
        if hasattr(client, 'wake_up') and not self.vllm_no_sleep:
            with self._time("timing/rollout/vllm_wake"):
                torch.cuda.empty_cache()
                m.setdefault("memory/gpu_before_wake_gb", []).append(
                    torch.cuda.memory_allocated() / 1e9)
                client.wake_up()

        # Sync weights to vLLM
        with self._time("timing/rollout/vllm_sync"):
            client.update_weights_from_model(eid, self.model)

            # Coherence rollout: generate with retain-only scales
            if self._is_coherence_rollout and self._coherence_gen == "retain_only":
                client.set_scales(eid, 1.0, 0.0)

            if eval_info is not None:
                modes = [("both", 1.0, 1.0), ("retain_only", 1.0, 0.0), ("forget_only", 0.0, 1.0)]
                for mode_name, retain_scale, forget_scale in modes:
                    eval_eid = self._eval_experiment_ids[mode_name]
                    client.update_weights_from_model(eval_eid, self.model)
                    client.set_scales(eval_eid, retain_scale, forget_scale)

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
        want_logprobs = self.vllm_importance_sampling_correction or getattr(self, "fast_vllm_is_correction", False)

        if eval_info is not None:
            # Piggybacked eval: combine training + eval prompts in one generate_multi call.
            eval_prompt_ids, eval_prompts_text, eval_data, eval_max_tokens = eval_info
            n_train = len(prompt_ids_list)
            n_eval_per_mode = len(eval_prompt_ids)
            modes = [("both", 1.0, 1.0), ("retain_only", 1.0, 0.0), ("forget_only", 0.0, 1.0)]

            # Build combined prompt list: training prompts + eval prompts × 3 modes
            all_prompt_ids = list(prompt_ids_list)
            all_eids = [eid] * n_train
            for mode_name, _, _ in modes:
                eval_eid = self._eval_experiment_ids[mode_name]
                all_prompt_ids.extend(eval_prompt_ids)
                all_eids.extend([eval_eid] * n_eval_per_mode)

            with self._time("timing/rollout/vllm_generate"):
                gen_result = client.generate_multi(
                    all_eids, all_prompt_ids, 1,
                    self.args.temperature, self.max_completion_length,
                    top_k=self.args.top_k, top_p=self.args.top_p,
                    return_logprobs=want_logprobs,
                )

            if want_logprobs:
                all_comp_texts, all_comp_ids, all_ret_prompts, all_logprobs = gen_result
            else:
                all_comp_texts, all_comp_ids, all_ret_prompts = gen_result
                all_logprobs = None

            # Split: first n_train are training, rest are eval (3 × n_eval_per_mode).
            comp_texts = all_comp_texts[:n_train]
            comp_ids_list = all_comp_ids[:n_train]
            sampling_logprobs = all_logprobs[:n_train] if all_logprobs else None

            # Partition eval results by mode and kick off background scoring.
            samples_by_mode = {}
            offset = n_train
            for mode_name, _, _ in modes:
                mode_samples = []
                for j in range(n_eval_per_mode):
                    mode_samples.append({
                        "prompt": eval_prompts_text[j],
                        "completion": all_comp_texts[offset + j],
                        "completion_ids": all_comp_ids[offset + j],
                    })
                samples_by_mode[mode_name] = mode_samples
                offset += n_eval_per_mode

            # Record eval step and dispatch scoring to background thread.
            # Label eval with the current global_step — the completed-step count
            # corresponding to the weights that produced these eval samples.
            eval_step = self.state.global_step
            self._last_routing_eval_step = eval_step
            self._dispatch_eval_scoring(samples_by_mode, eval_data, eval_step)

        else:
            # Normal path: single-eid generate.
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
        if hasattr(client, 'sleep') and not self.vllm_no_sleep:
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
            """Check optimizer state (m, v) and estimate Adam update norm."""
            max_abs_m = 0.0
            max_abs_v = 0.0
            has_nan_state = False
            n_with_state = 0
            update_norm_sq = 0.0
            lr = self.optimizer.param_groups[0]["lr"]
            eps = self.optimizer.param_groups[0].get("eps", 1e-8)
            for p in params:
                state = self.optimizer.state.get(p, {})
                if "exp_avg" in state:
                    n_with_state += 1
                    m_t = state["exp_avg"]
                    v_t = state["exp_avg_sq"]
                    m_max = m_t.abs().max().item()
                    v_max = v_t.abs().max().item()
                    if m_max > max_abs_m:
                        max_abs_m = m_max
                    if v_max > max_abs_v:
                        max_abs_v = v_max
                    if (math.isnan(m_max) or math.isnan(v_max) or
                            math.isinf(m_max) or math.isinf(v_max)):
                        has_nan_state = True
                    # Estimate per-step parameter displacement: lr * ||m / (sqrt(v) + eps)||_2
                    # This is an approximation — ignores bias correction (significant early
                    # in training, ~1% after warmup) and weight decay. Treats EMA-smoothed
                    # m/v as the current step's update direction.
                    update = m_t / (v_t.sqrt() + eps)
                    update_norm_sq += update.pow(2).sum().item()
            adam_update_norm_est = lr * math.sqrt(update_norm_sq)
            return {
                "max_abs_m": max_abs_m,
                "max_abs_v": max_abs_v,
                "has_nan_state": has_nan_state,
                "n_with_state": n_with_state,
                "adam_update_norm_est": adam_update_norm_est,
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
        # Optimizer stats
        m.setdefault("diagnostics/retain_adam_update_norm_est", []).append(retain_opt["adam_update_norm_est"])
        m.setdefault("diagnostics/forget_adam_update_norm_est", []).append(forget_opt["adam_update_norm_est"])
        m.setdefault("diagnostics/retain_max_abs_m", []).append(retain_opt["max_abs_m"])
        m.setdefault("diagnostics/forget_max_abs_m", []).append(forget_opt["max_abs_m"])
        m.setdefault("diagnostics/retain_max_abs_v", []).append(retain_opt["max_abs_v"])
        m.setdefault("diagnostics/forget_max_abs_v", []).append(forget_opt["max_abs_v"])

    # --- Microbatch preparation ---

    def _prepare_inputs(self, generation_batch):
        """Override TRL's _prepare_inputs for homogeneous sorting and dynamic token batching.

        Behavior depends on configuration:
        - gradient_routing + no dynamic batching: sort by is_rh, split into equal chunks
        - dynamic token batching (routing or not): sort (if routing) or shuffle, split into
          steps_per_generation chunks. Each chunk is one optimizer step's worth of data;
          training_step handles microbatch packing internally within each chunk.
        - coherence mode: alternates entire rollout cycles between routing and coherence
        - neither: delegate to TRL's default
        """
        if not self.model.training:
            return super()._prepare_inputs(generation_batch)

        use_dynamic = getattr(self, "_max_tokens_per_microbatch", None) is not None

        if not self.gradient_routing_enabled and not use_dynamic:
            return super()._prepare_inputs(generation_batch)

        from trl.trainer.utils import split_pixel_values_by_grid, unsplit_pixel_values_by_grid

        generate_every = self.args.steps_per_generation * self.num_iterations
        if self._step % generate_every == 0 or self._buffered_inputs is None:
            # Coherence rollout decision: at each new rollout cycle, decide
            # whether this is a coherence or routing rollout.
            if self._coherence != "none" and self.gradient_routing_enabled:
                self._coherence_rollout_counter += 1
                if self._coherence_rollout_counter >= self._coherence_every:
                    self._coherence_rollout_counter = 0
                    self._is_coherence_rollout = True
                    if self._coherence_gen == "retain_only":
                        from gradient_routing import set_scales
                        set_scales(self.model, retain_scale=1.0, forget_scale=0.0)
                else:
                    self._is_coherence_rollout = False

            generation_batch = self._generate_and_score_completions(generation_batch)
            generation_batch = split_pixel_values_by_grid(generation_batch)

            if self.gradient_routing_enabled:
                # Sort by is_rh: good (False=0) first, bad (True=1) last.
                is_rh = generation_batch.get("is_rh")
                if is_rh is not None:
                    n = is_rh.shape[0]
                    good_idx = (is_rh == 0).nonzero(as_tuple=True)[0]
                    bad_idx = (is_rh == 1).nonzero(as_tuple=True)[0]
                    good_idx = good_idx[torch.randperm(len(good_idx))]
                    bad_idx = bad_idx[torch.randperm(len(bad_idx))]
                    sorted_idx = torch.cat([good_idx, bad_idx])
                    for key, val in generation_batch.items():
                        if val is None:
                            continue
                        if isinstance(val, torch.Tensor) and val.ndim > 0 and val.shape[0] == n:
                            generation_batch[key] = val[sorted_idx]
                        elif isinstance(val, list) and len(val) == n:
                            generation_batch[key] = [val[i] for i in sorted_idx.tolist()]
                else:
                    generation_batch = shuffle_sequence_dict(generation_batch)
            else:
                generation_batch = shuffle_sequence_dict(generation_batch)

            if use_dynamic:
                # Dynamic token batching: split into steps_per_generation chunks.
                # Each chunk = one optimizer step's data; training_step packs internally.
                generation_batches = split_tensor_dict(generation_batch, self.args.steps_per_generation)
                self._buffered_inputs = [unsplit_pixel_values_by_grid(batch) for batch in generation_batches]
            else:
                generation_batches = split_tensor_dict(generation_batch, self.args.steps_per_generation)
                self._buffered_inputs = [unsplit_pixel_values_by_grid(batch) for batch in generation_batches]
        inputs = self._buffered_inputs[self._step % self.args.steps_per_generation]
        return inputs

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

    def _packed_compute_loss(self, model, packed_inputs):
        """Compute GRPO loss from a packed (padding-free) forward pass.

        Runs the model on packed (1, total_tokens) input, extracts per-sequence
        completion hidden states, repads to (N, max_comp_len, hidden), then calls
        the liger fused GRPO loss kernel. Requires use_liger_kernel=True.
        """
        packed_ids = packed_inputs["packed_input_ids"]        # (1, T)
        packed_pos = packed_inputs["packed_position_ids"]      # (1, T)
        seq_boundaries = packed_inputs["seq_boundaries"]       # [(p_len, c_len), ...]
        n_seqs = packed_inputs["num_sequences"]
        max_comp_len = packed_inputs["max_comp_len"]

        # Forward pass: get last hidden state (no lm_head projection)
        # unwrapped_model is e.g. LlamaForCausalLM; .model is the base LlamaModel
        unwrapped_model = self.accelerator.unwrap_model(model)
        output = unwrapped_model.model(
            input_ids=packed_ids,
            position_ids=packed_pos,
            use_cache=False,
        )
        hidden_states = output.last_hidden_state  # (1, T, H)
        hidden_dim = hidden_states.shape[-1]

        # Extract per-sequence completion hidden states and repad to (N, max_comp_len, H)
        # The logit for completion token at position t is predicted by hidden state at position t-1
        # So for completion tokens at positions [p_len, p_len+c_len), we need hidden states at [p_len-1, p_len+c_len-1)
        device = hidden_states.device
        last_hs_padded = torch.zeros(n_seqs, max_comp_len, hidden_dim, device=device, dtype=hidden_states.dtype)

        offset = 0
        for j, (p_len, c_len) in enumerate(seq_boundaries):
            if c_len > 0:
                # Hidden state at position (offset + p_len - 1) predicts first completion token
                # Hidden state at position (offset + p_len + c_len - 2) predicts last completion token
                hs_start = offset + p_len - 1
                hs_end = offset + p_len + c_len - 1
                last_hs_padded[j, :c_len] = hidden_states[0, hs_start:hs_end]
            offset += p_len + c_len

        # Call liger fused loss
        loss_mask = packed_inputs["completion_mask"]  # (N, max_comp_len)
        loss, metrics = self.liger_grpo_loss(
            _input=last_hs_padded,
            lin_weight=unwrapped_model.lm_head.weight,
            selected_token_ids=packed_inputs["completion_ids"],
            attention_mask=loss_mask,
            advantages=packed_inputs["advantages"],
            bias=getattr(unwrapped_model.lm_head, 'bias', None),
            old_per_token_logps=packed_inputs["old_per_token_logps"],
            ref_per_token_logps=packed_inputs["ref_per_token_logps"],
        )

        # Log metrics
        mode = "train" if self.model.training else "eval"
        mean_kl = metrics[0] if self.beta != 0.0 else None
        clip_ratio = metrics[-1]
        if self.beta != 0.0 and mean_kl is not None:
            self._metrics[mode]["kl"].append(self.accelerator.gather(mean_kl).mean().item())
        self._metrics[mode]["clip_ratio"].append(self.accelerator.gather(clip_ratio).mean().item())

        return loss

    def log(self, logs, *args, **kwargs):
        prompt = getattr(self, "_last_sample_prompt", None)
        completion = getattr(self, "_last_sample_completion", None)
        if prompt is not None and completion is not None:
            step = self.state.global_step
            if self.verbose:
                print(f"\n[Sample @ step {step}] {prompt} ||| {completion}\n")

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
            # Per-hint-type breakdown when detectable/hackable columns are present
            det_flags = getattr(self, "_last_detectable", None)
            hack_flags = getattr(self, "_last_hackable", None)
            if det_flags is not None and any(d is not None for d in det_flags):
                hack_mask = [bool(h) for h in hack_flags] if hack_flags is not None else [True] * len(det_flags)
                splits = [
                    ("detectable", [h and bool(d) for h, d in zip(hack_mask, det_flags)]),
                    ("undetectable", [h and not bool(d) for h, d in zip(hack_mask, det_flags)]),
                    ("unhackable", [not h for h in hack_mask]),
                ]
                for suffix, mask in splits:
                    if not any(mask):
                        continue
                    means, combined = cr.last_raw_metrics(mask=mask)
                    for name, mean in means.items():
                        _tm.setdefault(f"reward/raw_{name}_{suffix}", []).append(mean)
                    if combined is not None:
                        _tm.setdefault(f"reward/raw_combined_{suffix}", []).append(combined)

        # Periodic routing eval fallback: runs standalone eval when piggybacked
        # rollout-phase eval is not available (no vLLM, no eval_experiment_ids,
        # or eval_at_start). When piggybacking is available we never want this
        # path to fire — piggyback already owns eval generation for this run.
        piggyback_available = getattr(self, "_eval_experiment_ids", None) is not None
        if (self.eval_every > 0
                and self.eval_metrics
                and not piggyback_available
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
            old_logps = (_tm.get("timing/ref_logprobs") or [0])[-1]
            between_grpo = (_tm.get("timing/between_grpo_iters") or [0])[-1]
            dataloader = (_tm.get("timing/between_grpo_iters/dataloader") or [0])[-1]
            post_step = (_tm.get("timing/post_step") or [0])[-1]
            eval_time = (_tm.get("timing/eval") or [0])[-1]
            step = self.state.global_step
            parts = [
                f"rollout={rollout:.1f}s (sync={sync:.1f}s gen={gen:.1f}s)",
                f"old_logps={old_logps:.1f}s",
                f"reward_t={reward:.1f}s",
                f"update={update:.1f}s",
                f"between_grpo={between_grpo:.1f}s (dataloader={dataloader:.1f}s)",
                f"post_step={post_step:.1f}s",
            ]
            if eval_time > 0:
                parts.append(f"eval={eval_time:.1f}s")
            # Training reward: raw combined (pre-normalization) if available,
            # plus per-component raw means. Read from _tm *before* top_level
            # extraction below would move them out.
            reward_parts = []
            rc = _tm.get("reward/raw_combined")
            if rc:
                reward_parts.append(f"combined={rc[-1]:.3f}")
            for k, v in _tm.items():
                if k.startswith("reward/raw_") and k != "reward/raw_combined" and "_detectable" not in k and "_undetectable" not in k and "_hackable" not in k and "_unhackable" not in k and v:
                    name = k[len("reward/raw_"):]
                    reward_parts.append(f"{name}={v[-1]:.3f}")
            if reward_parts:
                parts.append("reward[" + " ".join(reward_parts) + "]")
            print(f"[timing @{step}] {' '.join(parts)}")

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

        top_level = {}
        keys_to_remove = []
        for key, vals in _tm.items():
            if key.startswith(_TOP_LEVEL_PREFIXES) and vals:
                top_level[key] = sum(vals) / len(vals)
                keys_to_remove.append(key)
        for key in keys_to_remove:
            del _tm[key]

        # --- Single wandb.log call per step (WandbCallback is removed) ---
        # All wandb logging MUST go through this one call to avoid step
        # monotonicity violations. See CLAUDE.md "wandb Logging" section.
        if self.args.report_to and "wandb" in self.args.report_to:
            import wandb
            if wandb.run is not None:
                from transformers.integrations import rewrite_logs
                # Update samples_seen: batch_size * grad_accum = samples per global_step
                self._samples_seen += (
                    self.args.per_device_train_batch_size
                    * self.args.gradient_accumulation_steps
                )
                wb = {
                    "train/global_step": self.state.global_step,
                    "samples_seen": self._samples_seen,
                }
                # TRL-native metrics (loss, grad_norm, lr, etc.)
                wb.update(rewrite_logs(logs))
                # Our custom top-level metrics (timing/, reward/, etc.)
                wb.update(top_level)
                # Routing eval metrics (stashed by _run_routing_eval)
                if self._pending_eval_wandb:
                    wb.update(self._pending_eval_wandb)
                    self._pending_eval_wandb = {}
                # Sample text (logged as Html to avoid trainer_state.json serialization failure)
                if prompt is not None and completion is not None:
                    wb["sample_text"] = wandb.Html(f"<pre>{prompt} ||| {completion}</pre>")
                wandb.log(wb)

        # Persist reward metrics into log_history (via super().log) so they
        # survive in trainer_state.json for post-hoc backfill.
        reward_keys = {k: v for k, v in top_level.items() if k.startswith("reward/")}
        if reward_keys:
            logs.update(reward_keys)

        # Still call super().log() for non-wandb side effects (log_history,
        # other callbacks), but WandbCallback has been removed so this won't
        # double-log to wandb.
        result = super().log(logs, *args, **kwargs)
        return result

    def _wait_for_eval_scoring(self):
        """Block until any in-flight background eval scoring completes."""
        if self._eval_scoring_thread is not None:
            self._eval_scoring_thread.join()
            self._eval_scoring_thread = None
            if self._eval_scoring_error is not None:
                err = self._eval_scoring_error
                self._eval_scoring_error = None
                raise RuntimeError(f"Background eval scoring failed: {err}") from err

    def _dispatch_eval_scoring(self, samples_by_mode, eval_data, step, gen_elapsed=0.0):
        """Dispatch eval reward scoring to a background thread.

        Called from both the piggybacked rollout path (_generate_single_turn)
        and the standalone eval path (_run_routing_eval).
        """
        import threading
        from eval_utils import score_eval_samples, format_routing_eval, get_routing_eval_metrics

        eval_metrics = self.eval_metrics
        output_dir = self.args.output_dir
        verbose = self.verbose

        def _score_in_background():
            try:
                try:
                    from envs.leetcode import use_eval_evaluator_on_this_thread
                    use_eval_evaluator_on_this_thread()
                except ImportError:
                    pass

                t_score = time.time()
                results = score_eval_samples(samples_by_mode, eval_metrics, eval_data=eval_data)
                elapsed = gen_elapsed + (time.time() - t_score)

                if verbose:
                    print(f"\n{format_routing_eval(results, step=step)}  "
                          f"(gen={gen_elapsed:.1f}s score={time.time()-t_score:.1f}s total={elapsed:.1f}s)\n")

                eval_flat = get_routing_eval_metrics(results)
                eval_flat["eval/elapsed_s"] = elapsed
                self._pending_eval_wandb = eval_flat

                record = {"step": step, "eval_elapsed_s": round(elapsed, 1)}
                for mode_name, mode_data in results.items():
                    for rname, rdata in mode_data["metrics"].items():
                        record[f"{mode_name}/{rname}"] = rdata["mean"]
                    record[f"{mode_name}/unique"] = mode_data["diversity"]["unique_samples"]
                    record[f"{mode_name}/jaccard"] = mode_data["diversity"]["avg_jaccard_similarity"]
                log_path = os.path.join(output_dir, "routing_eval.jsonl")
                os.makedirs(os.path.dirname(log_path), exist_ok=True)
                with open(log_path, "a") as f:
                    f.write(json.dumps(record) + "\n")

                # --- Diagnostic: log a few full eval completions per mode,
                # with per-reward scores and eval_data columns, for side-by-side
                # comparison against train_samples.jsonl.
                try:
                    samples_log_path = os.path.join(output_dir, "eval_samples.jsonl")
                    n_log = 8
                    with open(samples_log_path, "a") as f:
                        for mode_name, mode_data in results.items():
                            values_per_metric = {
                                rname: rdata.get("values", [])
                                for rname, rdata in mode_data["metrics"].items()
                            }
                            mode_samples = samples_by_mode.get(mode_name, [])
                            for i in range(min(n_log, len(mode_samples))):
                                rec = {"step": step, "mode": mode_name}
                                rec["prompt"] = mode_samples[i]["prompt"][:400] if isinstance(mode_samples[i]["prompt"], str) else str(mode_samples[i]["prompt"])[:400]
                                rec["completion"] = mode_samples[i]["completion"][:400]
                                for rname, vals in values_per_metric.items():
                                    if i < len(vals):
                                        rec[f"score/{rname}"] = float(vals[i])
                                if eval_data is not None and i < len(eval_data):
                                    for k, v in eval_data[i].items():
                                        if k == "prompt": continue
                                        if isinstance(v, (str, int, float, bool)) or v is None:
                                            rec[k] = v
                                f.write(json.dumps(rec) + "\n")
                except Exception as _e:
                    print(f"[eval_samples.jsonl logging error] {_e}")
            except Exception as e:
                import traceback
                traceback.print_exc()
                self._eval_scoring_error = e

        self._eval_scoring_thread = threading.Thread(
            target=_score_in_background, daemon=True)
        self._eval_scoring_thread.start()

    def _run_routing_eval(self):
        """Run gradient routing eval: generate synchronously, score in background.

        Fallback path for eval_at_start and non-vLLM modes. When vLLM is active
        with eval_experiment_ids, eval is piggybacked onto the training rollout
        in _generate_single_turn instead — this method is not called.
        """
        from eval_utils import eval_gradient_routing

        # Wait for any previous eval scoring to finish before starting a new one.
        self._wait_for_eval_scoring()

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
        samples_by_mode = eval_gradient_routing(
            self.model, self.processing_class, self.eval_metrics,
            n_samples=64, max_new_tokens=eval_max_tokens,
            temperature=self.args.temperature,
            prompts=eval_prompts, eval_data=eval_data,
            vllm_client=self._vllm_client,
            experiment_id=self._vllm_experiment_id,
            vllm_no_sleep=self.vllm_no_sleep,
            eval_experiment_ids=self._eval_experiment_ids,
            generate_only=True,
        )
        gen_elapsed = time.time() - t0
        self._dispatch_eval_scoring(samples_by_mode, eval_data, step, gen_elapsed)

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

            # Stash detectable flags for per-hint-type reward logging
            self._last_detectable = detector_kwargs.get("detectable")
            self._last_hackable = detector_kwargs.get("hackable")

            # Build candidate mask: only run detector on hackable & eligible samples.
            # Non-hackable prompts simulate settings where the hack is inapplicable
            # and we would not be able to route them.
            candidate = [True] * n_samples
            hackable_flags = detector_kwargs.get("hackable")
            if hackable_flags is not None and hackable_flags[0] is not None:
                candidate = [c and h for c, h in zip(candidate, hackable_flags)]
            detectable_flags = detector_kwargs.get("detectable")
            if detectable_flags is not None and detectable_flags[0] is not None:
                candidate = [c and bool(d) for c, d in zip(candidate, detectable_flags)]
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
                # Coherence rollout: modify advantages for detected hacks
                if self._is_coherence_rollout and is_rh_tensor.any():
                    if self._coherence_rh_mode == "penalty":
                        raw_rewards = self._reconstruct_raw_rewards().clone()
                        raw_rewards[is_rh_tensor] -= self._coherence_rh_penalty
                        G = self.num_generations
                        grouped = raw_rewards.view(-1, G)
                        mean = grouped.mean(dim=1, keepdim=True)
                        std = grouped.std(dim=1, keepdim=True, correction=0)
                        output["advantages"] = ((grouped - mean) / (std + 1e-4)).view(-1)
                    elif self._coherence_rh_mode == "filter":
                        output["advantages"] = output["advantages"].clone()
                        output["advantages"][is_rh_tensor] = 0.0
            elif self._reward_penalty_baseline:
                if self._advantage_type == "reinforce":
                    raw_rewards = output["raw_rewards"].clone()
                    if self._reward_penalty_amount is not None:
                        raw_rewards[is_rh_tensor] -= self._reward_penalty_amount
                    else:
                        raw_rewards[is_rh_tensor] = 0.0
                    advantages_rp = raw_rewards - self._all_reward_buffer.mean()
                    if self._reinforce_normalize_std:
                        advantages_rp = advantages_rp / (self._all_reward_buffer.std() + 1e-4)
                    output["advantages"] = advantages_rp
                else:
                    reward_fn = self._routed_reward if self._routed_reward is not None else self.reward_funcs[0]
                    raw_rewards = torch.tensor(reward_fn._last_rewards, dtype=torch.float32, device=device).clone()
                    if self._reward_penalty_amount is not None:
                        raw_rewards[is_rh_tensor] -= self._reward_penalty_amount
                    else:
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

        # --- Diagnostic: log a few training samples with their reward scores
        # and dataset columns to train_samples.jsonl. Used to compare training
        # completions against piggyback-eval completions on the same model.
        try:
            from rewards import CombinedReward
            cr = None
            for rf in self.reward_funcs:
                if isinstance(rf, CombinedReward):
                    cr = rf; break
                inner = getattr(rf, 'full_fn', None)
                if isinstance(inner, CombinedReward):
                    cr = inner; break
            if cr is not None and self.accelerator.is_main_process:
                comps_text = self.processing_class.batch_decode(
                    output["completion_ids"], skip_special_tokens=True)
                prompts_text = self.processing_class.batch_decode(
                    output["prompt_ids"], skip_special_tokens=True)
                comp_scores = {}
                for name, fn, _, _ in cr.components:
                    if fn._last_scores is not None:
                        comp_scores[name] = fn._last_scores
                n_log = min(8, len(comps_text))
                rec_base = {"step": self.state.global_step}
                if inputs and isinstance(inputs[0], dict):
                    extras_keys = [k for k in inputs[0]
                                   if k not in ("prompt", "completion", "completion_ids")]
                else:
                    extras_keys = []
                out_path = os.path.join(self.args.output_dir, "train_samples.jsonl")
                with open(out_path, "a") as f:
                    for i in range(n_log):
                        rec = dict(rec_base)
                        rec["prompt"] = prompts_text[i][:400]
                        rec["completion"] = comps_text[i][:400]
                        for name, vals in comp_scores.items():
                            if i < len(vals):
                                rec[f"score/{name}"] = float(vals[i])
                        if inputs and i < len(inputs) and isinstance(inputs[i], dict):
                            for k in extras_keys:
                                v = inputs[i].get(k)
                                if isinstance(v, (str, int, float, bool)) or v is None:
                                    rec[k] = v
                        f.write(json.dumps(rec) + "\n")
        except Exception as _e:
            print(f"[train_samples.jsonl logging error] {_e}")

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

    def _dynamic_microbatch_training_step(self, model, inputs, num_items_in_batch):
        """Unified training step with dynamic token-based microbatching.

        Works for both routing and non-routing modes. Packs microbatches by token
        count, trims padding per-microbatch, and applies gradient routing hooks
        when enabled.

        NOTE: assumes steps_per_generation=1 so that _prepare_inputs delivers the
        full generation batch in a single call. This is enforced in _run().
        """
        model.train()
        inputs = self._prepare_inputs(inputs)
        _t_after_prepare = time.perf_counter()

        # Stash one prompt+completion for wandb sample_text logging.
        # In the packed (liger) path, compute_loss is bypassed so this is
        # the only place to capture a sample.
        self._last_sample_prompt = self.processing_class.decode(
            inputs["prompt_ids"][0], skip_special_tokens=True
        )
        self._last_sample_completion = self.processing_class.decode(
            inputs["completion_ids"][0], skip_special_tokens=True
        )

        n_total = next(v.shape[0] for v in inputs.values()
                       if isinstance(v, torch.Tensor) and v.ndim > 0)
        max_tok = self._max_tokens_per_microbatch

        # Compute per-sample actual token counts (completion tokens only — prompt
        # trimming helps too but completion dominates loss compute)
        token_counts = inputs["completion_mask"].sum(dim=1).tolist()
        if "prompt_mask" in inputs:
            prompt_counts = inputs["prompt_mask"].sum(dim=1).tolist()
            token_counts = [p + c for p, c in zip(prompt_counts, token_counts)]
        global_comp_len = inputs["completion_mask"].shape[1]
        global_prompt_len = inputs["prompt_mask"].shape[1] if "prompt_mask" in inputs else 0

        # Build microbatches
        if self._is_coherence_rollout:
            # Coherence: all samples in one group, hook forget params (no good/bad split)
            inputs.pop("is_rh", None)
            inputs.pop("retain_advantages", None)
            inputs.pop("is_detector_good", None)
            retain_advantages = None
            original_advantages = None
            all_idx = list(range(n_total))
            all_mbs = [("coherence", mb) for mb in _pack_by_tokens(token_counts, all_idx, max_tok)]
        elif self.gradient_routing_enabled:
            is_rh = inputs.pop("is_rh")
            retain_advantages = inputs.pop("retain_advantages", None)
            inputs.pop("is_detector_good", None)
            original_advantages = inputs["advantages"]

            good_idx = (is_rh == 0).nonzero(as_tuple=True)[0].tolist()
            bad_idx = (is_rh == 1).nonzero(as_tuple=True)[0].tolist()

            # is_good=True for good microbatches, False for bad
            good_mbs = [(True, mb) for mb in _pack_by_tokens(token_counts, good_idx, max_tok)]
            bad_mbs = [(False, mb) for mb in _pack_by_tokens(token_counts, bad_idx, max_tok)]
            all_mbs = good_mbs + bad_mbs
        else:
            retain_advantages = None
            original_advantages = None
            all_idx = list(range(n_total))
            # is_good=None means no routing hooks
            all_mbs = [(None, mb) for mb in _pack_by_tokens(token_counts, all_idx, max_tok)]

        random.shuffle(all_mbs)
        _t_after_packing = time.perf_counter()

        total_loss = torch.tensor(0.0, device=self.accelerator.device)
        torch.cuda.reset_peak_memory_stats()
        _t_pass_start = time.perf_counter()
        trimmed_tokens_total = 0

        use_packed = self.use_liger_kernel and hasattr(self, 'liger_grpo_loss')

        for is_good, indices in all_mbs:
            # Swap advantages for good pass if renormalize
            if is_good is True and self._retain_mode == "renormalize" and retain_advantages is not None:
                inputs["advantages"] = retain_advantages
            elif is_good is False and self._retain_mode == "renormalize" and retain_advantages is not None:
                inputs["advantages"] = original_advantages

            n_mb = len(indices)
            scale = n_mb / n_total
            actual_tokens = sum(token_counts[i] for i in indices)
            trimmed_tokens_total += actual_tokens

            # Set hooks based on microbatch type
            hooks = []
            if is_good == "coherence":
                hooks = [p.register_hook(lambda g: torch.zeros_like(g))
                         for p in self._forget_params]
            elif is_good is True and self._good_pass_hooked_params is not None:
                hooks = [p.register_hook(lambda g: torch.zeros_like(g))
                         for p in self._good_pass_hooked_params]
            elif is_good is False:
                hooks = [p.register_hook(lambda g: torch.zeros_like(g))
                         for p in self._retain_params]
            # is_good is None → non-routing, no hooks

            if use_packed:
                # Padding-free forward: pack real tokens into (1, T), use liger loss
                packed = _pack_for_forward(inputs, indices)
                with self.compute_loss_context_manager():
                    loss = self._packed_compute_loss(model, packed)
                self.accelerator.backward(loss * scale)
                del packed
            else:
                # Fallback: trim-and-slice with standard compute_loss
                mb_inputs = _trim_and_slice(inputs, indices)
                with self.compute_loss_context_manager():
                    loss = self.compute_loss(model, mb_inputs, num_items_in_batch=num_items_in_batch)
                self.accelerator.backward(loss * scale)
                del mb_inputs

            for h in hooks:
                h.remove()
            total_loss = total_loss + loss.detach() * scale
            del loss

        # Retain KL regularization (after all microbatches, before optimizer.step)
        if getattr(self, "_retain_kl_coef", 0) > 0:
            retain_kl = self._retain_kl_pass(model)
            total_loss = total_loss + self._retain_kl_coef * retain_kl
            self._metrics.setdefault("train", {}).setdefault("retain_kl", []).append(retain_kl.item())

        # Timing and metrics
        _t_passes_end = time.perf_counter()
        m = self._metrics.setdefault("train", {})
        m.setdefault("memory/peak_update_gb", []).append(torch.cuda.max_memory_allocated() / 1e9)
        m.setdefault("memory/reserved_gb", []).append(torch.cuda.memory_reserved() / 1e9)
        # timing/update = forward/backward across all microbatches (comparable to non-dynamic path)
        m.setdefault("timing/update", []).append(_t_passes_end - _t_pass_start)
        m.setdefault("timing/detail/prepare_inputs", []).append(_t_after_prepare - self._dynamic_step_t0)
        m.setdefault("timing/detail/microbatch_packing", []).append(_t_after_packing - _t_after_prepare)
        m.setdefault("timing/detail/all_passes", []).append(_t_passes_end - _t_pass_start)
        m.setdefault("dynamic_batching/n_microbatches", []).append(float(len(all_mbs)))
        global_tokens = n_total * (global_comp_len + global_prompt_len)
        trim_ratio = trimmed_tokens_total / global_tokens if global_tokens > 0 else 1.0
        m.setdefault("dynamic_batching/trim_ratio", []).append(trim_ratio)

        if self._is_coherence_rollout:
            from gradient_routing import set_scales
            set_scales(model, retain_scale=1.0, forget_scale=1.0)
            m.setdefault("coherence/active", []).append(1.0)
        elif self.gradient_routing_enabled:
            n_bad = len(bad_idx)
            m.setdefault("routing/frac_rh", []).append(n_bad / n_total)
            m.setdefault("routing/homogeneous_microbatch", []).append(1.0)  # always homogeneous

        if self.state.global_step % self.args.logging_steps == 0:
            self._log_adapter_diagnostics()

        return total_loss

    def training_step(self, model, inputs, num_items_in_batch):
        # Dynamic token batching: unified path for routing and non-routing
        if getattr(self, "_max_tokens_per_microbatch", None) is not None:
            self._last_rollout_time = 0.0
            self._dynamic_step_t0 = time.perf_counter()
            if self._last_step_end_time is not None:
                self._metrics.setdefault("train", {}).setdefault("timing/detail/between_steps", []).append(
                    self._dynamic_step_t0 - self._last_step_end_time
                )
            total_loss = self._dynamic_microbatch_training_step(model, inputs, num_items_in_batch)
            time_after = time.perf_counter()
            total_time = time_after - self._dynamic_step_t0
            self._step += 1
            self._current_train_step_time += total_time
            if self._step % self.current_gradient_accumulation_steps == 0:
                self._metrics["train"]["step_time"].append(self._current_train_step_time)
                self._current_train_step_time = 0.0
            self._log_phase_timing(total_time - self._last_rollout_time)
            self._last_step_end_time = time.perf_counter()
            return total_loss

        if not self.gradient_routing_enabled:
            self._last_rollout_time = 0.0
            t0 = time.perf_counter()
            # Track between_grpo_iters on first micro-batch only
            is_first_microbatch = (self._step % self.current_gradient_accumulation_steps == 0)
            if is_first_microbatch and self._last_grpo_iter_end_time is not None:
                self._metrics.setdefault("train", {}).setdefault("timing/between_grpo_iters", []).append(
                    t0 - self._last_grpo_iter_end_time
                )
            torch.cuda.reset_peak_memory_stats()
            result = super().training_step(model, inputs, num_items_in_batch)
            total = time.perf_counter() - t0
            self._log_phase_timing(total - self._last_rollout_time)
            m = self._metrics.setdefault("train", {})
            m.setdefault("step_time", []).append(total)
            m.setdefault("memory/peak_update_gb", []).append(torch.cuda.max_memory_allocated() / 1e9)
            m.setdefault("memory/reserved_gb", []).append(torch.cuda.memory_reserved() / 1e9)
            if self.state.global_step % self.args.logging_steps == 0:
                self._log_adapter_diagnostics()
            self._last_step_end_time = time.perf_counter()
            # Flush post_step and between_grpo_iters on last micro-batch
            is_last_microbatch = (self._step % self.current_gradient_accumulation_steps == 0)
            if is_last_microbatch:
                self._last_grpo_iter_end_time = time.perf_counter()
                if self._post_step_accum > 0:
                    m.setdefault("timing/post_step", []).append(self._post_step_accum)
                    self._post_step_accum = 0.0
            return result

        self._last_rollout_time = 0.0
        time_before = time.perf_counter()
        if self._last_grpo_iter_end_time is not None:
            self._metrics.setdefault("train", {}).setdefault("timing/between_grpo_iters", []).append(
                time_before - self._last_grpo_iter_end_time
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
            # --- Default / Renormalize mode ---
            # With homogeneous microbatches (_prepare_inputs sorts by is_rh),
            # most microbatches are all-good or all-bad and need only one backward pass.
            # Mixed microbatches (at the good/bad boundary) fall back to two passes.

            is_all_good = (n_bad == 0)
            is_all_bad = (n_good == 0)

            if is_all_good:
                # Single pass: good samples only
                if self._retain_mode == "renormalize" and retain_advantages is not None:
                    inputs["advantages"] = retain_advantages
                hooks = []
                if self._good_pass_hooked_params is not None:
                    hooks = [p.register_hook(lambda g: torch.zeros_like(g))
                             for p in self._good_pass_hooked_params]
                with self.compute_loss_context_manager():
                    loss = self.compute_loss(model, inputs, num_items_in_batch=num_items_in_batch)
                self.accelerator.backward(loss)
                for h in hooks:
                    h.remove()
                total_loss = total_loss + loss.detach()

            elif is_all_bad:
                # Single pass: bad samples only — retain adapter gradients zeroed
                hooks = [p.register_hook(lambda g: torch.zeros_like(g))
                         for p in self._retain_params]
                with self.compute_loss_context_manager():
                    loss = self.compute_loss(model, inputs, num_items_in_batch=num_items_in_batch)
                self.accelerator.backward(loss)
                for h in hooks:
                    h.remove()
                total_loss = total_loss + loss.detach()

            else:
                # Mixed microbatch (boundary case): fall back to two-pass
                if self._retain_mode == "renormalize" and retain_advantages is not None:
                    inputs["advantages"] = retain_advantages

                # Pass 1: good samples
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

                # Pass 2: bad samples
                hooks = [p.register_hook(lambda g: torch.zeros_like(g))
                         for p in self._retain_params]
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
        # Track homogeneous microbatch hit rate (1.0 = single backward, 0.0 = two-pass fallback)
        is_homogeneous = 1.0 if (n_bad == 0 or n_good == 0) else 0.0
        self._metrics.setdefault("train", {}).setdefault("routing/homogeneous_microbatch", []).append(
            is_homogeneous
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
        self._last_grpo_iter_end_time = time.perf_counter()
        if self._post_step_accum > 0:
            self._metrics.setdefault("train", {}).setdefault("timing/post_step", []).append(
                self._post_step_accum)
            self._post_step_accum = 0.0

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
    parser.add_argument("--unhinted_frac", type=float, default=0.0,
                        help="Fraction of prompts that are unhinted/unhackable (0-1). "
                             "Unhinted prompts use original text, hackable=False.")
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
    parser.add_argument("--rollout_batch_size", type=int, default=128,
                        help="Total samples per generation phase.")
    parser.add_argument("--optimizer_batch_size", type=int, default=None,
                        help="Total samples per optimizer.step(). Defaults to rollout_batch_size.")
    parser.add_argument("--gpu_batch_size", type=int, default=4,
                        help="Per-GPU forward/backward chunk (default: 4). Controls gradient accumulation.")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--beta", type=float, default=0.02, help="KL penalty coefficient against reference model (0=disabled)")
    parser.add_argument("--epsilon", type=float, default=0.2, help="PPO lower clip (epsilon_low). TRL default 0.2.")
    parser.add_argument("--epsilon_high", type=float, default=None, help="PPO upper clip (DAPO Clip-Higher). Defaults to --epsilon (symmetric) if unset; DAPO uses 0.28.")
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=300)
    parser.add_argument("--lr_scheduler_type", type=str, default="linear",
                        help="LR scheduler type (linear, cosine, constant)")
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--save_adapter_only", action="store_true", default=False,
                        help="Save only adapter weights (not full model) in checkpoints. "
                             "Much smaller on disk. Requires base model at eval time.")
    parser.add_argument("--output_dir", default="./output")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume_from", default=None,
                        help="Path to checkpoint directory to resume training from")
    parser.add_argument("--optimizer", default="adamw_torch_fused",
                        help="Optimizer name (default: adamw_torch_fused). See transformers OptimizerNames for options (e.g. sgd, adafactor).")
    parser.add_argument("--config_check", action="store_true",
                        help="Run full config pipeline, print effective values, and exit without training.")
    parser.add_argument("--bf16", action="store_true", help="Use bfloat16 mixed precision (default: fp32)")
    parser.add_argument("--fp16", action="store_true", help="Use float16 mixed precision (default: fp32)")
    parser.add_argument("--gradient_checkpointing", type=lambda x: x.lower() in ("true", "1", "yes"), default=False,
                        help="Enable gradient checkpointing (default: False)")
    parser.add_argument("--use_liger_kernel", action=argparse.BooleanOptionalAction, default=False,
                        help="Use Liger fused linear GRPO loss (avoids materializing logits, default: off)")
    parser.add_argument("--liger_chunk_size", type=int, default=64,
                        help="Chunk size for LigerFusedLinearGRPOLoss (default 1 = one sample per chunk; "
                             "larger values trade memory for fewer kernel launches)")
    parser.add_argument("--torch_compile", action=argparse.BooleanOptionalAction, default=True,
                        help="Enable torch.compile for the model (default: on)")
    # torch.compile with max-autotune-no-cudagraphs gives ~10-15% speedup over liger kernels
    # via Triton kernel fusion + autotuning. Adds ~60-90s startup time for compilation.
    # CUDA graphs (reduce-overhead, max-autotune) are blocked by .item() calls in HF's
    # flash attention varlen path, causing graph breaks and OOM from recording too many
    # graph partitions. For faster startup at slight perf cost: --torch_compile_mode default
    # or --no-torch_compile --use_liger_kernel.
    parser.add_argument("--torch_compile_mode", default="max-autotune-no-cudagraphs",
                        choices=["default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"],
                        help="torch.compile mode (default: max-autotune-no-cudagraphs)")
    parser.add_argument("--max_tokens_per_microbatch", type=int, default=None,
                        help="Max tokens per microbatch for dynamic token batching. "
                             "When set, microbatches are packed by token count and trimmed to local max length.")
    parser.add_argument("--ref_max_tokens_per_microbatch", type=int, default=None,
                        help="Token budget for ref-logprob dynamic microbatching. "
                             "Ref runs under no_grad so peak memory is dominated by the "
                             "logits softmax rather than saved activations, allowing much "
                             "larger bins than the training forward-backward budget. "
                             "If omitted, defaults to 4 * --max_tokens_per_microbatch when "
                             "that flag is set, otherwise falls back to uniform chunking.")
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
    parser.add_argument("--coherence_rh_mode", choices=["filter", "penalty"], default="filter",
                        help="How to handle detected hacks during coherence rollout: 'filter' (zero advantages) or 'penalty' (subtract penalty from rewards)")
    parser.add_argument("--coherence_rh_penalty", type=float, default=3.0,
                        help="Reward penalty for detected hacks in coherence_rh_mode=penalty")
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
                        help="Reward penalty baseline: penalize rewards for RH-detected samples, recompute advantages.")
    parser.add_argument("--reward_penalty_amount", type=float, default=None,
                        help="Amount to subtract from post-clip reward for detected hacks. "
                             "Default: None (zeros reward, legacy behavior). E.g. 3.0 subtracts 3.0.")
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
    parser.add_argument("--vllm_no_sleep", action="store_true", default=False,
                        help="Disable vLLM sleep/wake cycle between rollout and training phases. "
                             "Useful when multiple runs share a GPU (per_gpu > 1) and sleep/wake overhead dominates.")
    parser.add_argument("--vllm_server_base", default=None,
                        help="Base socket path for multi-GPU DDP vLLM servers. "
                             "Each DDP rank appends _rank{rank}.sock. Set by sweep.py.")
    parser.add_argument("--vllm_importance_sampling", action="store_true", default=False,
                        help="Enable importance sampling correction for vLLM generation mismatch. "
                             "Requires vLLM server to support return_logprobs.")
    parser.add_argument("--no_fast_vllm_is", action="store_true", default=False,
                        help="Disable fast vLLM IS correction (use vLLM sampling logprobs as old_logps "
                             "instead of a full forward pass). Enabled by default when using vLLM.")
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


def compute_batch_params(rollout_batch_size, optimizer_batch_size, gpu_batch_size, n_devices, num_generations):
    """Map user-facing batch params to TRL GRPOConfig params.

    Args:
        rollout_batch_size: Total samples per generation phase.
        optimizer_batch_size: Total samples per optimizer.step(). None = rollout_batch_size.
        gpu_batch_size: Per-GPU forward/backward chunk. None = optimizer_batch_size / n_devices.
        n_devices: Number of GPUs.
        num_generations: Completions per prompt (for divisibility check).

    Returns:
        dict with per_device_train_batch_size, gradient_accumulation_steps, generation_batch_size.
    """
    optimizer_batch_size = optimizer_batch_size or rollout_batch_size

    if gpu_batch_size is not None:
        per_device = gpu_batch_size
    else:
        assert optimizer_batch_size % n_devices == 0, (
            f"optimizer_batch_size ({optimizer_batch_size}) must be divisible by n_devices ({n_devices})"
        )
        per_device = optimizer_batch_size // n_devices

    total_per_step = per_device * n_devices
    assert optimizer_batch_size % total_per_step == 0, (
        f"optimizer_batch_size ({optimizer_batch_size}) must be divisible by "
        f"gpu_batch_size * n_devices = {per_device} * {n_devices} = {total_per_step}"
    )
    gas = optimizer_batch_size // total_per_step

    assert rollout_batch_size % total_per_step == 0, (
        f"rollout_batch_size ({rollout_batch_size}) must be divisible by "
        f"gpu_batch_size * n_devices = {per_device} * {n_devices} = {total_per_step}"
    )
    assert rollout_batch_size % num_generations == 0, (
        f"rollout_batch_size ({rollout_batch_size}) must be divisible by "
        f"num_generations ({num_generations})"
    )

    return {
        "per_device_train_batch_size": per_device,
        "gradient_accumulation_steps": gas,
        "generation_batch_size": rollout_batch_size,
    }


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
        # Skip None values (which would clobber YAML/field defaults for non-Optional fields).
        structured_keys = {"reward", "rh_detector", "hack_freq_detector", "name"}
        ec_fields = set(ExperimentConfig.model_fields)
        for k, v in vars(args).items():
            if k == "config" or k in structured_keys:
                continue
            if v is None:
                continue
            if k not in ec_fields:
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
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=model_dtype, attn_implementation="flash_attention_2")
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

    # Pre-deserialize datasets: convert from Arrow-backed HF Dataset to a plain
    # list-backed Dataset. This avoids repeated Arrow→Python deserialization in
    # the DataLoader, which becomes expensive under GPU contention during training.
    class _ListDataset(torch.utils.data.Dataset):
        """Thin wrapper around a list of dicts, compatible with HF Trainer."""
        def __init__(self, rows, column_names):
            self.rows = rows
            self.column_names = column_names
        def __len__(self):
            return len(self.rows)
        def __getitem__(self, idx):
            return self.rows[idx]

    def _predeserialize(dataset):
        rows = dataset.to_list()
        return _ListDataset(rows, dataset.column_names)

    train_dataset = _predeserialize(train_dataset)
    eval_dataset = _predeserialize(eval_dataset)
    print(f"Pre-deserialized datasets: train={len(train_dataset)}, eval={len(eval_dataset)}")

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
        amt = getattr(args, 'reward_penalty_amount', None)
        if amt is not None:
            print(f"Reward penalty baseline: subtracting {amt} from rewards for RH-detected samples")
        else:
            print(f"Reward penalty baseline: zeroing rewards for RH-detected samples")

    # Batch config: rollout_batch_size (generation), optimizer_batch_size (gradient step),
    # gpu_batch_size (per-GPU forward/backward). See compute_batch_params() for mapping.
    n_devices = dist.get_world_size() if _is_ddp else 1
    optimizer_bs = args.optimizer_batch_size or args.rollout_batch_size
    batch_params = compute_batch_params(
        rollout_batch_size=args.rollout_batch_size,
        optimizer_batch_size=args.optimizer_batch_size,
        gpu_batch_size=args.gpu_batch_size,
        n_devices=n_devices,
        num_generations=args.num_generations,
    )
    per_device_bs = batch_params["per_device_train_batch_size"]
    grad_accum_steps = batch_params["gradient_accumulation_steps"]
    gen_bs = batch_params["generation_batch_size"]

    # Dynamic token batching: each training_step receives optimizer_batch_size samples
    # and does its own internal microbatch packing. gas=1 because the dynamic loop
    # handles accumulation. steps_per_generation = rollout / optimizer for multiple
    # optimizer steps per rollout.
    if args.max_tokens_per_microbatch is not None:
        if args.gpu_batch_size is not None:
            print(f"Note: --max_tokens_per_microbatch overrides --gpu_batch_size={args.gpu_batch_size}")
            args.gpu_batch_size = None
        assert args.retain_mode != "penalty", (
            "--max_tokens_per_microbatch is not compatible with --retain_mode=penalty"
        )
        assert args.use_liger_kernel, (
            "--max_tokens_per_microbatch requires --use_liger_kernel for memory-efficient loss computation"
        )
        optimizer_bs = args.optimizer_batch_size or args.rollout_batch_size
        per_device_bs = optimizer_bs // n_devices
        grad_accum_steps = 1  # dynamic loop handles microbatching internally
        gen_bs = args.rollout_batch_size
        steps_per_gen = args.rollout_batch_size // optimizer_bs
        assert args.rollout_batch_size % optimizer_bs == 0, (
            f"rollout_batch_size ({args.rollout_batch_size}) must be divisible by "
            f"optimizer_batch_size ({optimizer_bs})"
        )
        print(f"Batch config: rollout={args.rollout_batch_size} optimizer={optimizer_bs} "
              f"({steps_per_gen} optimizer steps/rollout, "
              f"dynamic token batching: max_tokens_per_microbatch={args.max_tokens_per_microbatch})")
    else:
        print(f"Batch config: rollout={args.rollout_batch_size} optimizer={optimizer_bs} "
              f"gpu={per_device_bs}/device × {n_devices} devices "
              f"(gas={grad_accum_steps}, gen_bs={gen_bs})")

    config = GRPOConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=per_device_bs,
        gradient_accumulation_steps=grad_accum_steps,
        generation_batch_size=gen_bs,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        temperature=args.temperature,
        top_k=args.top_k if args.top_k > 0 else 0,
        top_p=args.top_p,
        learning_rate=args.lr,
        optim=args.optimizer,
        num_train_epochs=args.num_epochs,
        max_steps=args.max_steps,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        adam_beta2=args.adam_beta2,
        max_grad_norm=args.max_grad_norm,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        loss_type="grpo",
        repetition_penalty=args.repetition_penalty,
        beta=args.beta,
        epsilon=args.epsilon,
        epsilon_high=args.epsilon_high if args.epsilon_high is not None else args.epsilon,
        seed=args.seed,
        bf16=args.bf16 and not args.fp16,
        fp16=args.fp16,
        report_to="wandb" if not args.no_wandb else "none",
        run_name=args.run_name or f"grpo_{reward_name}_lr{args.lr}",
        gradient_checkpointing=args.gradient_checkpointing,
        # Disable TRL's built-in vLLM importance sampling — our custom vLLM clients
        # handle weight sync directly, and the LoRA client doesn't return logprobs.
        # When needed, we enable it explicitly via --vllm_importance_sampling.
        vllm_importance_sampling_correction=False,
        use_liger_kernel=args.use_liger_kernel,
        # Disable SwiGLU patch: our MLP adapters (DualMLPAdapter) replace gate_proj/up_proj/down_proj
        # with adapter modules, so Liger's SwiGLU kernel (which calls self.gate_proj etc. directly)
        # would crash. RMSNorm and RoPE patches are safe and kept enabled by default.
        # When torch.compile is active, disable all Liger model patches (rms_norm, rope, swiglu)
        # to avoid opaque custom ops that break fusion. Keep only the fused GRPO loss.
        # Without torch.compile, keep rms_norm and rope patches (swiglu always off due to DualMLPAdapter).
        liger_kernel_config={"swiglu": False, "fused_linear_cross_entropy": False,
                             "rms_norm": not args.torch_compile, "rope": not args.torch_compile} if args.use_liger_kernel else None,
        torch_compile=args.torch_compile and not args.torch_compile_mode,
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

    # Manual torch.compile with mode (bypasses TRL's default compile)
    if args.torch_compile_mode and args.torch_compile:
        torch._dynamo.config.capture_scalar_outputs = True
        torch._dynamo.config.allow_unspec_int_on_nn_module = True
        print(f"torch.compile mode={args.torch_compile_mode}")
        model = torch.compile(model, mode=args.torch_compile_mode, dynamic=True)

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
        coherence_rh_mode=args.coherence_rh_mode,
        coherence_rh_penalty=args.coherence_rh_penalty,
        filter_baseline=filter_baseline,
        reward_penalty_baseline=reward_penalty_baseline,
        reward_penalty_amount=getattr(args, 'reward_penalty_amount', None),
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
        save_adapter_only=args.save_adapter_only,
    )
    trainer._environment = args.environment
    trainer._n_digits = args.n_digits
    trainer._env_spec = env_spec
    trainer._env_args = args
    trainer._save_batch_path = getattr(args, 'save_batch', None)
    trainer._max_tokens_per_microbatch = args.max_tokens_per_microbatch
    # Ref-logprob token budget: default to 4x the training microbatch budget,
    # since ref runs under no_grad (no saved activations) and peak memory is
    # dominated by the logits softmax instead.
    if args.ref_max_tokens_per_microbatch is not None:
        trainer._ref_max_tokens_per_microbatch = args.ref_max_tokens_per_microbatch
    elif args.max_tokens_per_microbatch is not None:
        trainer._ref_max_tokens_per_microbatch = 4 * args.max_tokens_per_microbatch
    else:
        trainer._ref_max_tokens_per_microbatch = None
    # Scoring batch size for logprob computation (no gradients/activations needed, so 4x gpu_batch_size)
    effective_gpu_bs = args.gpu_batch_size or 4
    trainer._scoring_batch_size = effective_gpu_bs * 4

    # Fix TRL double-scaling bug: TRL's _compute_loss already divides loss by
    # gradient_accumulation_steps, but accelerator.backward() divides again.
    # Setting accelerator's GAS to 1 disables its redundant division.
    trainer.accelerator.gradient_accumulation_steps = 1

    # Optionally enable vLLM importance sampling correction to account for
    # distribution mismatch between vLLM's generation and HF's forward pass.
    trainer.vllm_no_sleep = getattr(args, 'vllm_no_sleep', False)

    if vllm_client is not None and args.vllm_importance_sampling:
        trainer.use_vllm = True
        trainer.vllm_importance_sampling_correction = True
        trainer.vllm_importance_sampling_mode = "token_truncate"
        trainer.vllm_importance_sampling_cap = 10.0
        print("[vLLM] Enabled importance sampling correction for vLLM generation")

    trainer.fast_vllm_is_correction = (vllm_client is not None and not args.no_fast_vllm_is)

    # Remove TRL's WandbCallback — we own all wandb logging via a single
    # wandb.log() call in SampleGRPOTrainer.log(). This avoids step
    # monotonicity violations from multiple wandb.log() calls per step.
    # We must init wandb ourselves first since WandbCallback.setup() normally does it.
    from transformers.integrations import WandbCallback
    if not args.no_wandb:
        import wandb
        if wandb.run is None:
            wandb.init(
                project=os.environ.get("WANDB_PROJECT", "small-rl"),
                name=config.run_name,
                config={**config.to_dict()},
            )
    trainer.remove_callback(WandbCallback)

    # Set up wandb x-axes: samples_seen for training dynamics, global_step for timing
    if not args.no_wandb:
        import wandb
        if wandb.run is not None:
            wandb.define_metric("samples_seen")
            wandb.define_metric("train/global_step")
            # Training dynamics → x-axis = samples_seen
            for prefix in ["reward/*", "routing_eval/*", "train/loss",
                           "train/grad_norm", "train/learning_rate",
                           "diagnostics/kl", "diagnostics/entropy",
                           "diagnostics/retain_grad_norm", "diagnostics/forget_grad_norm",
                           "diagnostics/retain_param_norm", "diagnostics/forget_param_norm",
                           "diagnostics/forget_nonzero_grad_frac", "diagnostics/forget_max_abs_grad",
                           "diagnostics/retain_adam_update_norm_est", "diagnostics/forget_adam_update_norm_est",
                           "diagnostics/retain_max_abs_m", "diagnostics/forget_max_abs_m",
                           "diagnostics/retain_max_abs_v", "diagnostics/forget_max_abs_v",
                           "diagnostics/frac_rh", "coherence/*"]:
                wandb.define_metric(prefix, step_metric="samples_seen")
            # Per-step intrinsics → x-axis = train/global_step
            for prefix in ["timing/*", "memory/*", "eval/elapsed_s",
                           "diagnostics/completions_*"]:
                wandb.define_metric(prefix, step_metric="train/global_step")

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
        trainer._wait_for_eval_scoring()
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

    explicit_keys = set(params.keys())
    for k, v in params.items():
        if hasattr(args, k):
            setattr(args, k, v)
    _apply_model_defaults(args, explicit_keys=explicit_keys)
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

    # NOTE: CLI path still can't distinguish `--temperature 1.0` (explicit)
    # from not passing --temperature at all — argparse limitation. Sweep path
    # (train_main) handles this correctly via explicit_keys.
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
