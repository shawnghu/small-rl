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

# TRL's ProfilingContext fires its own wandb.log() per timed block (~3/step), violating the
# single-wandb.log design: it inflates wandb _step ~4x, clutters runs with single-key rows, and
# the extra HTTP traffic was the upload-backpressure source that lagged dashboards on 2026-06-09.
# Keep the timing context, drop its logging — all wandb goes through SampleGRPOTrainer.log().
import trl.extras.profiling as _trl_profiling
_trl_profiling.ProfilingContext._log_metrics = lambda self, duration: None


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
    def isatty(self):
        return getattr(self.stream, "isatty", lambda: False)()
    def __getattr__(self, name):
        # Delegate any other file API (encoding, writable, readable, ...) to the
        # wrapped stream so libraries that probe stdout (e.g. transformers'
        # loading_report colorizer, tqdm, rich) don't crash on this Tee. Guarded
        # against recursion before self.stream is set in __init__.
        stream = self.__dict__.get("stream")
        if stream is None:
            raise AttributeError(name)
        return getattr(stream, name)
    def close(self):
        self.file.close()

from rewards import get_reward_fn, API_REWARD_NAMES
from experiment_config import ExperimentConfig, RewardConfig, RewardComponentConfig, RHDetectorConfig


def _flatten_chatrequest(msgs):
    """Render a ChatRequest (list[{role,content}]) as plain text for base-model
    generation under --no_chat_template: message contents joined, NO role/chat
    scaffolding (so a base model continues the raw text instead of seeing chat
    tokens it was never trained on). System content (e.g. the leetcode code-format
    instruction) is kept. Plain-string prompts pass through unchanged. The dataset
    prompt stays list[dict], so reward fns still get the unwrapped user message."""
    if isinstance(msgs, str):
        return msgs
    return "\n\n".join(m["content"] for m in msgs)


def _spawn_vllm_server(model_name, mlp_config, gpu_memory, socket_path, ready_file,
                       layer_start=0.0, layer_end=1.0, layer_stride=1,
                       max_experiments=4, gpu_id=0, label=None,
                       num_gpu_blocks=None, adapter_type="mlp"):
    """Worker for per-run vLLM server subprocess (must be module-level for spawn pickling).

    Concurrent-init serialization is internal: vllm_init_slot acquires an
    exclusive flock for the duration of VLLMServer construction (the
    CUDA-heavy phase). Released before server.run() starts the request loop.
    Multiple train.py children on the same GPU (e.g. inside a train_many pack)
    queue at the lock and start their CUDA inits one at a time.
    """
    import os
    from vllm_lifecycle import vllm_init_slot, vllm_worker_setup_signals
    vllm_worker_setup_signals()  # setsid → process group leader for killpg
    os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)  # vLLM 0.17 CuMemAllocator rejects expandable_segments
    os.environ.setdefault("VLLM_LOGGING_LEVEL", "ERROR")
    os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"

    init_label = label or f"vllm_train_pid{os.getpid()}"
    # Hold the per-GPU lock for the CUDA-init phase only. Release before
    # entering the unbounded serve loop. CUDA_VISIBLE_DEVICES is set by the
    # parent train.py if it remapped; gpu_id passed in is the PHYSICAL id we
    # lock against — keeps the lock path stable across processes that see
    # different remapped CUDA ids.
    with vllm_init_slot(gpu_id, init_label):
        if adapter_type == "lora":
            # Native-LoRA server: ranks/scales arrive per-request from the
            # client (enable_lora=True), so the server only needs model + memory.
            # Mirrors sweep.py:_vllm_server_worker's lora branch — the proven
            # local path — so the Modal in-container spawn can serve LoRA too.
            from vllm_lora import VLLMLoRAServer
            server = VLLMLoRAServer(
                socket_addr=socket_path,
                model_name=model_name,
                gpu_memory_utilization=gpu_memory,
            )
        else:
            from vllm_utils import MLP_PRESETS
            from vllm_server import VLLMServer
            preset = MLP_PRESETS[mlp_config]
            server = VLLMServer(
                socket_addr=socket_path,
                # Default 4 slots: 1 training + 3 eval adapter modes (both, retain_only,
                # forget_only). Interlaced coherence bumps this to 5 to add a coh slot.
                max_experiments=max_experiments,
                retain_neurons=preset["retain_neurons"],
                forget_neurons=preset["forget_neurons"],
                model_name=model_name,
                gpu_memory_utilization=gpu_memory,
                layer_start=layer_start,
                layer_end=layer_end,
                layer_stride=layer_stride,
                num_gpu_blocks_override=num_gpu_blocks,
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

# GRPO ratio guardrail. TRL's old-logp path (_get_per_token_logps_and_entropies)
# mis-scores certain confident structural tokens (digits/braces ending \boxed{...},
# esp. in long/truncated completions) as ~e^-20 while the policy actually assigns
# them ~prob 1, so the ratio exp(new-old) explodes (loss ~1e11) and corrupts the
# update — catastrophic for long math completions, invisible for short leetcode.
# Clamp old toward the recomputed (correct) new logps so |new-old| <= this many
# nats: bounds the ratio AND corrects the bad importance weights. 0 disables.
# TEMPORARILY 0 for the old-vs-GT diagnostic (so `old` is raw, not clamped).
LOGP_RATIO_CLAMP = 0.0

# VERL-parity k3 KL clamp (used by the hand-rolled clamped loss; --kl_clamp).
# VERL's low_var_kl: d=clamp(ref-new, -KL_CLAMP_D, KL_CLAMP_D); kld=clamp(exp(d)-d-1,
# -KL_CLAMP_KLD, KL_CLAMP_KLD). The OUTPUT clamp is the gradient-killer: kld saturates
# at 10 once d>~2.6, so the k3 KL contributes ZERO gradient on far-moved tokens —
# structurally bounding the exp(ref-new) explosion. Single source of truth (VERL
# core_algos.py:1464,1467). See memory verl-parity-loss-changes.
KL_CLAMP_D = 20.0
KL_CLAMP_KLD = 10.0

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
    "r32f0":  {"retain_rank": 32, "forget_rank": 0,  "layer_stride": 1, "lora_alpha": 32},
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


def _inject_detectable_into_eval_data(eval_data, classifiable_fn):
    """Set `detectable` on each eval row from the rh_detector's classifiable
    predicate. The rh_detector defines the monitoring scope, so its
    classifiability predicate is the single source of truth — overwrites any
    env-emitted `detectable` column (whose semantics may pre-date the
    configured detector; e.g. leetcode source jsonls use `detectable=hint
    presence`, which is wrong when the rh_detector gates on tags or
    difficulty).

    No-op when eval_data is empty or classifiable_fn is None (detector with
    no per-prompt feature gate).
    """
    if eval_data is None or classifiable_fn is None or not eval_data:
        return
    cols = {}
    for k in eval_data[0].keys():
        cols[k] = [row.get(k) for row in eval_data]
    flags = classifiable_fn(**cols)
    assert len(flags) == len(eval_data), (
        f"classifiable_fn returned {len(flags)} flags for {len(eval_data)} rows"
    )
    for row, f in zip(eval_data, flags):
        row["detectable"] = bool(f)


# Keys whose sequence dimension (dim=1) should be trimmed on the completion side (right-padded).
_COMPLETION_TRIM_KEYS = {"completion_ids", "completion_mask", "old_per_token_logps",
                         "ref_per_token_logps", "sampling_per_token_logps",
                         "token_routing_mask"}
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
    has_token_routing = "token_routing_mask" in inputs and inputs["token_routing_mask"] is not None
    all_old_logps = [] if has_old_logps else None
    all_ref_logps = [] if has_ref_logps else None
    all_token_routing = [] if has_token_routing else None
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
        if has_token_routing:
            all_token_routing.append(inputs["token_routing_mask"][i, :c_len])

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
    token_routing_padded = (torch.zeros(n_seqs, max_comp_len, dtype=torch.bool, device=device)
                            if has_token_routing else None)

    for j, (_, c_len) in enumerate(seq_boundaries):
        if c_len > 0:
            comp_ids_padded[j, :c_len] = all_comp_ids[j]
            comp_mask_padded[j, :c_len] = 1
            if has_old_logps:
                old_logps_padded[j, :c_len] = all_old_logps[j]
            if has_ref_logps:
                ref_logps_padded[j, :c_len] = all_ref_logps[j]
            if has_token_routing:
                token_routing_padded[j, :c_len] = all_token_routing[j]

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
        "token_routing_mask": token_routing_padded,
        "num_sequences": n_seqs,
        "max_comp_len": max_comp_len,
    }


class SampleGRPOTrainer(GRPOTrainer):
    """GRPOTrainer with sample logging and optional gradient routing."""

    def __init__(self, *args, gradient_routing_enabled=False,
                 retain_params=None, forget_params=None,
                 routing_mode=None, rh_detector=None,
                 routing_granularity="trajectory", token_routing_detector_name=None,
                 routed_adam=False, routed_adam_kappa=1.0,
                 routed_adam_classic_bad_weight=2.0,
                 base_rh_detector=None,
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
                 coherence="none", coherence_every=0,
                 coherence_gen="retain_only",
                 coherence_rh_mode="filter",
                 coherence_rh_penalty=3.0,
                 coh_fixed_advantage=None,
                 coh_samples_per_rollout=0,
                 rh_detector_verifies_retain_samples=False,
                 rh_detector_retain_recall=1.0,
                 verified_only_training=False,
                 interlaced_coh_opt_batch_mode="split",
                 rollout_forget_scale_mode="fixed",
                 forget_scale_modulation="none",
                 forget_scale_target_hack_rate=0.5,
                 forget_scale_ema_weight=0.95,
                 forget_scale_decay=0.9,
                 forget_scale_min_clamp=0.0,
                 forget_scale_decay_every=0,
                 rp_extra_retain_advantage_multiplier=1.0,
                 retain_warmup_steps=0,
                 forget_warmup_steps=0,
                 warmup_rh_detector=None,
                 rh_classifiable_fn=None,
                 vllm_client=None,
                 adapter_type="lora",
                 liger_chunk_size=64,
                 save_adapter_only=True,
                 forget_lr_mult=1.0,
                 detect_unhackable=True,
                 trace_routing=True,
                 bad_pass_loss_scale=1.0,
                 unlabeled_forget_grad_scale=0.5,
                 coh_loss_type="grpo",
                 coh_kl_beta=0.1,
                 kl_clamp=False,
                 token_mean_loss=False,
                 fp32_lora=False,
                 adv_std_eps=1e-4,
                 nonfinite_grad_skip=False,
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
        # VERL-parity loss knobs (see memory verl-parity-loss-changes).
        self._kl_clamp = kl_clamp
        self._token_mean_loss = token_mean_loss
        self._fp32_lora = fp32_lora
        self._adv_std_eps = adv_std_eps
        self._nonfinite_grad_skip = nonfinite_grad_skip
        self._retain_params = retain_params or set()
        self._forget_params = forget_params or set()
        self._forget_lr_mult = forget_lr_mult
        self._detect_unhackable = detect_unhackable
        self._routing_mode = routing_mode
        # Token-level routing: route individual behavior tokens rather than whole
        # trajectories. _token_bearing_ids (set of vocab ids that emit the routed
        # single-char behavior) is computed lazily on first detection.
        self._routing_granularity = routing_granularity
        self._token_routing_detector_name = token_routing_detector_name
        self._routed_adam = routed_adam
        self._routed_adam_kappa = routed_adam_kappa
        self._routed_adam_classic_bad_weight = routed_adam_classic_bad_weight
        self._token_bearing_ids = None   # single-char behaviors (em-dash/semicolon): vocab-id set
        self._id_strings = None          # span behaviors (bold): per-vocab-id decoded strings
        # α scaling on the bad-pass loss to compensate for the missing
        # retain-side update on RH samples (classic routing). With α=2 the
        # forget adapter's bad-pass gradient is doubled, restoring the joint
        # policy's responsiveness to bad-sample signal that would have
        # otherwise been split across both adapters under no-routing. Has no
        # effect on the good pass. Doesn't currently apply to penalty
        # retain_mode (which has its own loss structure).
        self._bad_pass_loss_scale = float(bad_pass_loss_scale)
        # Coherence loss type: "grpo" (default, existing behavior) or "kl_to_base"
        # (loss = coh_kl_beta * KL(policy(retain=1,forget=0) || base(retain=0,forget=0))
        # on coherence microbatches). With kl_to_base, the reward/advantage signal
        # is bypassed for coherence samples — only the deviation from base anchors
        # the retain adapter. Gradient flows only to retain (forget contribution = 0).
        self._coh_loss_type = str(coh_loss_type)
        self._coh_kl_beta = float(coh_kl_beta)
        assert self._coh_loss_type in ("grpo", "kl_to_base"), \
            f"--coh_loss_type must be 'grpo' or 'kl_to_base', got {self._coh_loss_type!r}"
        # Resolve good-pass hook target once at init:
        #   classic: good samples update both adapters (no hooks)
        #   exclusive: good samples update only retain (zero forget params)
        #   scaled_classic: good samples — forget gradients multiplied by
        #     `unlabeled_forget_grad_scale` (classic ≡ scale=1, exclusive ≡
        #     scale=0). "Unlabeled" because non-detected ≠ guaranteed good.
        self._unlabeled_forget_grad_scale = float(unlabeled_forget_grad_scale)
        if gradient_routing_enabled:
            assert routing_mode in ("classic", "exclusive", "scaled_classic"), \
                f"--routing_mode must be 'classic'/'exclusive'/'scaled_classic', got {routing_mode!r}"
            if routing_mode == "exclusive":
                self._good_pass_hooked_params = self._forget_params
                self._good_pass_hook_fn = (lambda g: torch.zeros_like(g))
            elif routing_mode == "scaled_classic":
                s = self._unlabeled_forget_grad_scale
                self._good_pass_hooked_params = self._forget_params
                self._good_pass_hook_fn = (lambda g, s=s: g * s)
            else:  # classic
                self._good_pass_hooked_params = None
                self._good_pass_hook_fn = None
        else:
            self._good_pass_hooked_params = None
            self._good_pass_hook_fn = None
        self.rh_detector = rh_detector
        # Unwrapped detector for the retain-verifier path. Conceptually a
        # separate tool with perfect precision; see the wrap site for the
        # design intent. Falls back to rh_detector if no base provided.
        self.base_rh_detector = base_rh_detector if base_rh_detector is not None else rh_detector
        self.eval_every = eval_every
        self.eval_metrics = eval_metrics or {}
        self._last_routing_eval_step = 0
        self._did_start_eval = False   # base-model eval on first rollout (labeled step 0)
        self._did_final_eval = False   # final-model eval on last rollout (labeled max_steps)
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
        self._coh_fixed_advantage = coh_fixed_advantage
        self._coherence_rollout_counter = 0
        self._is_coherence_rollout = False
        self._coh_samples_per_rollout = coh_samples_per_rollout
        self._interlaced_coh = coh_samples_per_rollout > 0
        # Dual-env coherence (set post-construction in _run). When _coherence_env
        # is set, coherence samples are drawn from _coh_dataset (a different env)
        # rather than the classifiable subset of the main train_dataset.
        self._coherence_env = None
        self._coh_dataset = None
        self._coh_eval_dataset = None
        self._coh_env_spec = None
        self._coh_iter = None
        self._coh_eval_metrics = {}
        self._coh_eval_split_n = None  # # of main-env eval prompts (rest are coherence)
        # Retain-verification skyline: detector also verifies non-hack samples,
        # coherence training restricted to confirmed RETAIN samples. See
        # _build_detectable_iterator and _maybe_swap_coherence_prompts for how
        # this gets hooked into the rollout + training loops.
        self._rh_detector_verifies_retain_samples = rh_detector_verifies_retain_samples
        self._rh_detector_retain_recall = rh_detector_retain_recall
        self._verified_only_training = verified_only_training
        self._interlaced_coh_opt_batch_mode = interlaced_coh_opt_batch_mode
        self._rollout_forget_scale_mode = rollout_forget_scale_mode
        self._last_rollout_forget_scale = 1.0
        # Idea 2: EMA-driven forget-scale clamp
        self._forget_scale_modulation = forget_scale_modulation
        self._forget_scale_target_hack_rate = forget_scale_target_hack_rate
        self._forget_scale_ema_weight = forget_scale_ema_weight
        self._forget_scale_decay = forget_scale_decay
        self._forget_scale_min_clamp = forget_scale_min_clamp
        # Decay frequency: prevent the clamp from running ahead of the EMA.
        # 0 = auto-derive as round(1 / (1 - ema_weight)).
        if forget_scale_decay_every > 0:
            self._forget_scale_decay_every = forget_scale_decay_every
        else:
            self._forget_scale_decay_every = max(1, round(1.0 / max(1e-6, 1.0 - forget_scale_ema_weight)))
        self._forget_scale_decay_counter = 0
        self._forget_scale_clamp = 1.0
        self._hack_rate_ema = None
        self._rp_extra_retain_advantage_multiplier = rp_extra_retain_advantage_multiplier
        # Idea 4: phase warmup
        self._retain_warmup_steps = retain_warmup_steps
        self._forget_warmup_steps = forget_warmup_steps
        # Idea 4(c): perfect-detector warmup
        self.warmup_rh_detector = warmup_rh_detector
        self._rh_classifiable_fn = rh_classifiable_fn
        self._detectable_iter = None  # lazy init in _build_detectable_iterator
        # Routing verification trace (opt-in via --trace_routing). Writes
        # per-rollout / per-sample / per-microbatch records to
        # routing_trace.jsonl, plus [TRACE ...] lines to stdout for grep.
        self._trace_routing = trace_routing
        self._trace_file = None  # opened lazily on first write
        # vLLM HTTP server for generation
        self._vllm_client = vllm_client
        self._vllm_experiment_id = None
        self._eval_experiment_ids = None  # {mode_name: eid} for concurrent eval
        self._coh_experiment_id = None    # interlaced-coherence rollout slot (retain-only scales)
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
            if self._interlaced_coh:
                assert hasattr(vllm_client, "generate_multi"), (
                    "coh_samples_per_rollout > 0 requires a vLLM client with generate_multi "
                    "(MLP adapter path)"
                )
                self._coh_experiment_id = vllm_client.register()
                print(f"[vLLM] Registered interlaced-coherence slot: "
                      f"coh={self._coh_experiment_id}")
        self._save_adapter_only = save_adapter_only

        # Wrap _get_per_token_logps_and_entropies so we can capture the actor's
        # per-token logps from the gradient-flowing forward inside _compute_loss
        # (the function is also used for ref-model and rollout-old logps; we just
        # snapshot the most recent call's output and read it back from compute_loss
        # afterward to compute the rollout↔θ_new MC-KL).
        self._last_actor_per_token_logps = None
        _orig_get_logps = self._get_per_token_logps_and_entropies
        def _wrapped_get_logps(*a, **kw):
            result = _orig_get_logps(*a, **kw)
            self._last_actor_per_token_logps = result[0].detach()
            return result
        self._get_per_token_logps_and_entropies = _wrapped_get_logps

        # Phase timing: rollout (generation+scoring) vs update (gradients)
        self._last_rollout_time = 0.0
        self._accum_update_time = 0.0
        self._last_step_end_time = None
        self._last_grpo_iter_end_time = None  # set at end of last micro-batch of each logical step
        self._post_step_accum = 0.0  # accumulates optimizer.step + clip + zero_grad time

    def create_optimizer(self):
        """Wrap optimizer.step(), clip_grad_norm_, zero_grad() and
        get_batch_samples with timing instrumentation.

        Also builds a grouped optimizer when ``forget_lr_mult != 1.0`` (retain
        group at args.learning_rate, forget group at args.learning_rate * mult).
        The forget group always uses weight_decay=0.0 regardless of
        --weight_decay (adapter params typically shouldn't be decayed).
        Optimizer class is derived from args.optim via HF's standard helper,
        so --optimizer=adamw_torch_fused/sgd/etc. all work unchanged.
        """
        # RoutedAdam (shared-v): routed first moment, full-stream second moment. Built before
        # super().create_optimizer(), which early-returns when self.optimizer is already set.
        if self.optimizer is None and self._routed_adam:
            from routed_adam import RoutedAdam
            assert self._retain_params and self._forget_params, \
                "routed_adam requires dual adapters with retain and forget params"
            retain_p = [p for p in self._retain_params if p.requires_grad]
            forget_p = [p for p in self._forget_params if p.requires_grad]
            # Every trainable param must belong to exactly one adapter — a trainable param
            # outside the routing would silently train without a momentum stream.
            trainable = {p for p in self.model.parameters() if p.requires_grad}
            routed = set(retain_p) | set(forget_p)
            assert trainable == routed, (
                f"routed_adam: {len(trainable - routed)} trainable params outside retain/forget "
                f"adapters (and {len(routed - trainable)} routed params not trainable)"
            )
            self.optimizer = RoutedAdam(
                [
                    {"params": retain_p, "lr": self.args.learning_rate,
                     "weight_decay": self.args.weight_decay, "kappa": 1.0},
                    {"params": forget_p, "lr": self.args.learning_rate,
                     "weight_decay": self.args.weight_decay, "kappa": self._routed_adam_kappa},
                ],
                lr=self.args.learning_rate,
                betas=(self.args.adam_beta1, self.args.adam_beta2),
                eps=self.args.adam_epsilon,
                weight_decay=self.args.weight_decay,
            )
            print(f"[optimizer] RoutedAdam: retain {sum(p.numel() for p in retain_p):,} params "
                  f"(kappa=1.0), forget {sum(p.numel() for p in forget_p):,} params "
                  f"(kappa={self._routed_adam_kappa}), betas=({self.args.adam_beta1},"
                  f"{self.args.adam_beta2}) eps={self.args.adam_epsilon} "
                  f"wd={self.args.weight_decay}")

        # Grouped optimizer for asymmetric retain/forget LRs. Must be built
        # before super().create_optimizer(), which early-returns when
        # self.optimizer is already set.
        if (self.optimizer is None
                and self._forget_lr_mult != 1.0
                and self._retain_params
                and self._forget_params):
            from transformers import Trainer as _HFTrainer
            retain_p = [p for p in self._retain_params if p.requires_grad]
            forget_p = [p for p in self._forget_params if p.requires_grad]
            forget_lr = self.args.learning_rate * self._forget_lr_mult
            cls, kwargs = _HFTrainer.get_optimizer_cls_and_kwargs(self.args, self.model)
            self.optimizer = cls(
                [
                    {"params": retain_p, "lr": self.args.learning_rate,
                     "weight_decay": self.args.weight_decay},
                    {"params": forget_p, "lr": forget_lr,
                     "weight_decay": 0.0},  # NOTE: forget group never decays; see --forget_lr_mult help
                ],
                **kwargs,
            )
            print(f"[optimizer] grouped {cls.__name__} (from --optimizer={self.args.optim}): "
                  f"retain lr={self.args.learning_rate} wd={self.args.weight_decay} "
                  f"({len(retain_p)} tensors, {sum(p.numel() for p in retain_p):,} params), "
                  f"forget lr={forget_lr} wd=0.0 (forced) "
                  f"({len(forget_p)} tensors, {sum(p.numel() for p in forget_p):,} params)")
        super().create_optimizer()
        _trainer = self

        # --- timing/post_step: optimizer + clip + zero_grad combined ---
        _orig_step = self.optimizer.step
        _orig_zero_grad = self.model.zero_grad
        _orig_clip = self.accelerator.clip_grad_norm_

        def _timed_step(*a, **kw):
            t0 = time.perf_counter()
            if _trainer._nonfinite_grad_skip:
                # VERL dp_actor parity: if any (post-clip) grad is non-finite, skip the optimizer
                # step and zero grads (don't let a NaN/Inf spike corrupt weights/Adam moments).
                finite = all(p.grad is None or bool(torch.isfinite(p.grad).all())
                             for grp in _trainer.optimizer.param_groups for p in grp["params"])
                if not finite:
                    print(f"[nonfinite_grad_skip] step {_trainer.state.global_step}: "
                          f"non-finite grad — skipping optimizer.step")
                    _trainer._metrics.setdefault("train", {}).setdefault(
                        "diagnostics/nonfinite_grad_skips", []).append(1.0)
                    _trainer.optimizer.zero_grad()
                    _trainer._post_step_accum += time.perf_counter() - t0
                    return None
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
            # RoutedAdam: mirror the global clip factor onto the unclipped m streams.
            # clip_grad_norm_(parameters, max_norm) returns the PRE-clip total norm; the factor
            # actually applied to p.grad is min(1, max_norm/total_norm).
            if _trainer._routed_adam:
                max_norm = a[1] if len(a) > 1 else kw["max_norm"]
                total_norm = float(result)
                opt = _trainer.optimizer
                inner = opt.optimizer if hasattr(opt, "optimizer") else opt  # unwrap accelerate
                inner._clip_factor = (
                    min(1.0, max_norm / total_norm) if total_norm > 0 else 1.0
                )
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
        if self._eval_experiment_ids is None:
            return False
        step = self.state.global_step
        # Base-model eval on the first rollout(s). Bounded to step <= 1 so it can NEVER shadow the
        # regular cadence: if the eval doesn't dispatch, by step 2 this clause is dead and the cadence
        # resumes — no deadlock.
        if not self._did_start_eval and step <= 1:
            return True
        # Final-model eval on the last rollout (the step-200 eval otherwise never fires).
        if not self._did_final_eval and step >= self.args.max_steps - 1:
            return True
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
        n_eval = getattr(env_args, 'routing_eval_prompts', 64) if env_args is not None else 64

        if env_spec is not None and env_spec.load_eval_prompts is not None:
            eval_data = env_spec.load_eval_prompts(n_eval, env_args)
            eval_prompts = [d["prompt"] for d in eval_data]
            eval_max_tokens = env_spec.eval_max_tokens
        elif getattr(self, '_environment', 'stories') == 'arithmetic':
            from eval_utils import load_arithmetic_eval_prompts
            n_digits = getattr(self, '_n_digits', 3)
            eval_prompts = load_arithmetic_eval_prompts(n=n_eval, n_digits=n_digits)
            eval_data = None
            eval_max_tokens = n_digits + 2
        else:
            eval_prompts = None
            eval_data = None
            eval_max_tokens = 128

        if eval_prompts is None:
            from eval_utils import _load_eval_prompts
            eval_prompts = _load_eval_prompts(n=n_eval)

        _inject_detectable_into_eval_data(eval_data, getattr(self, "_rh_classifiable_fn", None))

        # Dual-env: piggyback half-budget coherence-env eval prompts after the
        # routing-env ones. Scored separately (env-homogeneous slices) in
        # _dispatch_eval_scoring; split index stashed on self.
        self._coh_eval_split_n = None
        if (self._coherence_env and self._coh_eval_metrics
                and self._coh_eval_dataset is not None):
            half = max(1, n_eval // 2)
            if eval_data is not None:
                eval_data = eval_data[:half]
            eval_prompts = eval_prompts[:half]
            self._coh_eval_split_n = len(eval_prompts)
            coh_rows = [dict(r) for r in self._coh_eval_dataset.rows[:half]]
            coh_prompts = [r["prompt"] for r in coh_rows]
            eval_prompts = eval_prompts + coh_prompts
            if eval_data is not None:
                eval_data = eval_data + coh_rows
            if self._coh_env_spec is not None:
                eval_max_tokens = max(eval_max_tokens, self._coh_env_spec.eval_max_tokens)

        # Chat-wrap plain-string prompts for instruct models. train.py's
        # _wrap_prompts_as_chat applies the same transform to train_dataset
        # and eval_dataset; without mirroring it here, piggyback eval reaches
        # vLLM as raw strings that the model never saw during training.
        env_args = getattr(self, '_env_args', None)
        sys_prompt = getattr(env_args, 'system_prompt', "") if env_args else ""
        no_chat_template = getattr(env_args, 'no_chat_template', False)
        if self.processing_class.chat_template is not None and not no_chat_template:
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
        # Return the chat-template-rendered strings (not the list[dict] form) as the
        # second element. Downstream eval scoring calls detectors like llm_judge that
        # do string substitution on the prompt (.replace("{question}", prompt)), which
        # TypeErrors on list inputs. This matches what the training path does — it
        # batch_decodes prompt_ids into strings before calling the detector.
        return eval_prompt_ids, prompt_texts, eval_data, eval_max_tokens

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

        # Interlaced coherence: designate the last C slots as coherence rollout.
        if mode == "train" and self._interlaced_coh:
            if self._in_retain_warmup():
                n_coh = len(prompts)  # Idea 4(a): all slots become coh
            elif self._in_forget_warmup():
                n_coh = 0  # Idea 4(b): no coh slice this phase
            else:
                n_coh = self._coh_samples_per_rollout
        else:
            n_coh = 0

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
            if os.environ.get("LORA_SYNC_DEBUG"):
                from gradient_routing import DualLoRALinear as _DLL
                def _bnorm(_m):
                    s = 0.0
                    for _mod in _m.modules():
                        if isinstance(_mod, _DLL) and _mod.lora_B_retain is not None:
                            s += float((_mod.lora_B_retain.data.float() ** 2).sum())
                    return s ** 0.5
                _raw = _bnorm(self.model)
                _unw = _bnorm(self.accelerator.unwrap_model(self.model))
                _same = self.accelerator.unwrap_model(self.model) is self.model
                print(f"[LORA_SYNC_DEBUG callsite] B-norm raw(self.model)={_raw:.4f} "
                      f"unwrapped={_unw:.4f}  unwrap_is_self={_same}", flush=True)
            client.update_weights_from_model(eid, self.model)

            # Determine forget_scale for this rollout's routing samples.
            # Default: 1.0 (both adapters at full scale). Modes:
            #   'fixed'             - 1.0
            #   'random_uniform_0_1' - U(0, 1) sampled fresh per rollout
            #   'random_choice_0_or_0.5' - {0.0, 0.5} chosen uniformly per rollout
            # Coh slots set their own scales below (always (1, 0)).
            mode = self._rollout_forget_scale_mode
            if mode == "fixed":
                base_forget_scale = 1.0
            elif mode == "random_uniform_0_1":
                import random as _r
                base_forget_scale = _r.random()
            elif mode == "random_choice_0_or_0.5":
                import random as _r
                base_forget_scale = _r.choice([0.0, 0.5])
            else:
                raise ValueError(f"Unknown rollout_forget_scale_mode={mode!r}")
            # Idea 2: multiplicative clamp from forget_scale_modulation=ema_clamp
            # (no-op when modulation='none'; clamp stays at 1.0).
            rollout_forget_scale = base_forget_scale * self._forget_scale_clamp
            self._last_rollout_forget_scale = rollout_forget_scale
            # Log so we can see the actual scale used per step.
            _m = self._metrics.setdefault("train", {})
            _m.setdefault("rollout/forget_scale", []).append(rollout_forget_scale)
            _m.setdefault("rollout/forget_scale_base", []).append(base_forget_scale)
            _m.setdefault("rollout/forget_scale_clamp", []).append(self._forget_scale_clamp)

            # Coherence rollout: generate with retain-only scales
            if self._is_coherence_rollout and self._coherence_gen == "retain_only":
                client.set_scales(eid, 1.0, 0.0)
            elif rollout_forget_scale != 1.0:
                client.set_scales(eid, 1.0, rollout_forget_scale)

            if n_coh > 0:
                client.update_weights_from_model(self._coh_experiment_id, self.model)
                client.set_scales(self._coh_experiment_id, 1.0, 0.0)

            if eval_info is not None:
                # "both" mode reflects the *operative* scale: under
                # forget_scale_modulation=ema_clamp it's (1, clamp) so eval
                # measures the model at the same scale used during training.
                # Without modulation the helper returns 1.0, restoring (1, 1).
                both_forget_scale = self._train_forget_scale()
                modes = [("both", 1.0, both_forget_scale),
                         ("retain_only", 1.0, 0.0),
                         ("forget_only", 0.0, 1.0)]
                for mode_name, retain_scale, forget_scale in modes:
                    eval_eid = self._eval_experiment_ids[mode_name]
                    client.update_weights_from_model(eval_eid, self.model)
                    client.set_scales(eval_eid, retain_scale, forget_scale)

        # Tokenize prompts (handle both chat and plain string formats). Under
        # --no_chat_template, render ChatRequest prompts as raw flattened text (no chat
        # scaffolding) so a base model doesn't see unfamiliar chat tokens at step 0.
        no_ct = getattr(getattr(self, '_env_args', None), 'no_chat_template', False)
        if is_conversational({"prompt": prompts[0]}) and not no_ct:
            prompt_texts = [
                self.processing_class.apply_chat_template(
                    p, add_generation_prompt=True, tokenize=False,
                    enable_thinking=False,
                )
                for p in prompts
            ]
        else:
            prompt_texts = [_flatten_chatrequest(p) for p in prompts]

        prompt_ids_list = [
            self.processing_class.encode(p, add_special_tokens=False)
            for p in prompt_texts
        ]

        # Generate: TRL's RepeatSampler already expanded prompts × num_generations.
        want_logprobs = self.vllm_importance_sampling_correction or getattr(self, "fast_vllm_is_correction", False)

        # Use generate_multi whenever we need to route slots to multiple eids
        # (eval piggyback or interlaced coherence). Otherwise the simple single-eid
        # generate path is used.
        use_multi = eval_info is not None or n_coh > 0

        n_train = len(prompt_ids_list)
        n_routing = n_train - n_coh
        assert n_routing >= 0, f"n_coh ({n_coh}) > n_train ({n_train})"

        if use_multi:
            # Build combined prompt list. Order: routing training slots, then coh training slots,
            # then eval slots x 3 modes (if piggybacking).
            all_prompt_ids = list(prompt_ids_list)
            all_eids = [eid] * n_routing + [self._coh_experiment_id] * n_coh

            if eval_info is not None:
                eval_prompt_ids, eval_prompts_text, eval_data, eval_max_tokens = eval_info
                n_eval_per_mode = len(eval_prompt_ids)
                modes = [("both", 1.0, 1.0), ("retain_only", 1.0, 0.0), ("forget_only", 0.0, 1.0)]
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

            # Split: first n_train are training (routing + coh), rest are eval.
            comp_texts = all_comp_texts[:n_train]
            comp_ids_list = all_comp_ids[:n_train]
            sampling_logprobs = all_logprobs[:n_train] if all_logprobs else None

            if eval_info is not None:
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
                gs = self.state.global_step
                if not self._did_start_eval:
                    eval_step = 0                     # base model
                    self._did_start_eval = True
                elif not self._did_final_eval and gs >= self.args.max_steps - 1:
                    eval_step = self.args.max_steps   # final model
                    self._did_final_eval = True
                else:
                    eval_step = gs
                print(f"[eval-fire] eval_step={eval_step} global_step={gs}", flush=True)
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
            sum_abs_v = 0.0
            numel_v = 0
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
                    v_abs = v_t.abs()
                    v_max = v_abs.max().item()
                    sum_abs_v += v_abs.sum().item()
                    numel_v += v_t.numel()
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
            mean_abs_v = (sum_abs_v / numel_v) if numel_v else 0.0
            return {
                "max_abs_m": max_abs_m,
                "max_abs_v": max_abs_v,
                "mean_abs_v": mean_abs_v,
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
        # Per-opt-batch kind (interlaced) falls back to per-rollout kind (classic/none).
        last_was_coh = getattr(self, "_last_opt_batch_was_coherence", self._is_coherence_rollout)
        kind = "coherence" if last_was_coh else "routing"
        # Unsplit metrics (kept for backward-compat with historical dashboards)
        # plus kind-split copies so coherence vs routing dynamics are separable.
        unsplit_then_split = [
            ("diagnostics/retain_grad_norm", retain["total_grad_norm"]),
            ("diagnostics/retain_param_norm", retain["total_param_norm"]),
            ("diagnostics/forget_grad_norm", forget["total_grad_norm"]),
            ("diagnostics/forget_param_norm", forget["total_param_norm"]),
            ("diagnostics/forget_nonzero_grad_frac",
             forget["n_with_grad"] / forget["n_total"] if forget["n_total"] else 0),
            ("diagnostics/forget_max_abs_grad", forget["max_abs_grad"]),
            ("diagnostics/retain_adam_update_norm_est", retain_opt["adam_update_norm_est"]),
            ("diagnostics/forget_adam_update_norm_est", forget_opt["adam_update_norm_est"]),
            ("diagnostics/retain_max_abs_m", retain_opt["max_abs_m"]),
            ("diagnostics/forget_max_abs_m", forget_opt["max_abs_m"]),
            ("diagnostics/retain_max_abs_v", retain_opt["max_abs_v"]),
            ("diagnostics/forget_max_abs_v", forget_opt["max_abs_v"]),
            ("diagnostics/retain_mean_abs_v", retain_opt["mean_abs_v"]),
            ("diagnostics/forget_mean_abs_v", forget_opt["mean_abs_v"]),
        ]
        for key, val in unsplit_then_split:
            m.setdefault(key, []).append(val)
            m.setdefault(f"{kind}/{key}", []).append(val)

    # --- Microbatch preparation ---

    def _prepare_inputs(self, generation_batch):
        """Override TRL's _prepare_inputs for homogeneous sorting and dynamic token batching.

        Behavior depends on configuration:
        - gradient_routing + no dynamic batching: sort by is_rh to produce
          homogeneous per-device gpu-batches, split into equal chunks, then
          shuffle the chunk execution order so bad batches spread across the
          rollout's optimizer steps (prevents periodic forget-adapter Adam
          spikes from bad-sample starvation).
        - dynamic token batching (routing or not): plain shuffle, split into
          steps_per_generation chunks. Each chunk is one optimizer step's
          worth of data; training_step re-homogenizes by is_rh via bin
          packing internally.
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
            # Classic-coherence rollout decision: at each new rollout cycle, decide
            # whether this is a coherence or routing rollout. Gated by coherence_every>0
            # (0 disables classic; interlaced coherence uses its own path below).
            if self._coherence_every > 0 and self.gradient_routing_enabled:
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

            if self._interlaced_coh and use_dynamic and self._interlaced_coh_opt_batch_mode == "split":
                # Interlaced coherence (split mode): partition by is_coherence,
                # plain-shuffle each partition, split each into pure opt batches,
                # concat, shuffle opt-batch execution order. Coh and routing
                # batches share the same optimizer_batch_size so each resulting
                # opt batch is internally homogeneous on is_coherence — this
                # enables outer-scope adapter-scale set/restore around the coh
                # opt batches and `assert all_coh or none_coh` in the loss path.
                is_coh = generation_batch["is_coherence"]
                n = is_coh.shape[0]
                coh_idx = is_coh.nonzero(as_tuple=True)[0]
                rout_idx = (~is_coh).nonzero(as_tuple=True)[0]
                coh_idx = coh_idx[torch.randperm(len(coh_idx))]
                rout_idx = rout_idx[torch.randperm(len(rout_idx))]

                def _slice_gen_dict(d, indices):
                    idx_list = indices.tolist()
                    out = {}
                    for k, v in d.items():
                        if v is None:
                            out[k] = None
                        elif isinstance(v, torch.Tensor) and v.ndim > 0 and v.shape[0] == n:
                            out[k] = v[indices]
                        elif isinstance(v, list) and len(v) == n:
                            out[k] = [v[i] for i in idx_list]
                        else:
                            out[k] = v
                    return out

                coh_batch = _slice_gen_dict(generation_batch, coh_idx)
                rout_batch = _slice_gen_dict(generation_batch, rout_idx)

                opt_bs = n // self.args.steps_per_generation
                n_coh_opt = len(coh_idx) // opt_bs
                n_rout_opt = len(rout_idx) // opt_bs
                assert n_coh_opt * opt_bs == len(coh_idx), (
                    f"coh samples ({len(coh_idx)}) not divisible by opt_bs ({opt_bs})")
                assert n_rout_opt * opt_bs == len(rout_idx), (
                    f"routing samples ({len(rout_idx)}) not divisible by opt_bs ({opt_bs})")

                coh_chunks = split_tensor_dict(coh_batch, n_coh_opt) if n_coh_opt > 0 else []
                rout_chunks = split_tensor_dict(rout_batch, n_rout_opt) if n_rout_opt > 0 else []
                all_chunks = coh_chunks + rout_chunks
                perm = torch.randperm(len(all_chunks)).tolist()
                self._buffered_inputs = [
                    unsplit_pixel_values_by_grid(all_chunks[i]) for i in perm
                ]
            elif self._interlaced_coh and use_dynamic and self._interlaced_coh_opt_batch_mode == "merged":
                # Merged mode: don't split coh and rout into separate opt
                # batches. The whole rollout becomes a single opt batch (1
                # opt step per rollout regardless of coh_samples_per_rollout),
                # and microbatches inside the opt batch are still
                # homogeneous-by-partition (coh / good / bad) with per-mb
                # adapter-scale management for coh microbatches in the loss
                # path. Matches the 1-step-per-rollout cadence of the
                # original test_conditional_envs.py classic-coherence regime.
                n = generation_batch["is_coherence"].shape[0]
                opt_bs = n // self.args.steps_per_generation
                n_opt = n // opt_bs
                assert n_opt * opt_bs == n, (
                    f"rollout_batch_size ({n}) not divisible by opt_bs ({opt_bs})")
                chunks = split_tensor_dict(generation_batch, n_opt) if n_opt > 0 else []
                perm = torch.randperm(len(chunks)).tolist()
                self._buffered_inputs = [
                    unsplit_pixel_values_by_grid(chunks[i]) for i in perm
                ]
            else:
                if self.gradient_routing_enabled and not use_dynamic:
                    # Non-dynamic routing: sort good-first bad-last so per-device
                    # gpu-batches (split_tensor_dict chunks below) are homogeneous,
                    # enabling the fast single-pass branches in training_step.
                    # The gpu-batch *order* is re-shuffled after the split so bad
                    # batches are spread across the rollout's optimizer steps —
                    # otherwise bad samples concentrate at the rollout's tail and
                    # the forget adapter's Adam state decays then spikes.
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
                    # Dynamic mode (with or without routing) or non-routing:
                    # plain shuffle. Dynamic mode re-homogenizes by is_rh inside
                    # _dynamic_microbatch_training_step via bin packing, so the
                    # pre-sort is unnecessary and would concentrate bad samples
                    # in the last chunks (causing periodic forget-adapter spikes).
                    generation_batch = shuffle_sequence_dict(generation_batch)

                generation_batches = split_tensor_dict(generation_batch, self.args.steps_per_generation)
                self._buffered_inputs = [unsplit_pixel_values_by_grid(batch) for batch in generation_batches]

                if self.gradient_routing_enabled and not use_dynamic:
                    # Shuffle gpu-batch execution order. Each batch stays
                    # homogeneous internally (sliced from the sorted array),
                    # but bad batches are now distributed across the rollout's
                    # optimizer steps instead of all at the tail.
                    perm = torch.randperm(len(self._buffered_inputs)).tolist()
                    self._buffered_inputs = [self._buffered_inputs[i] for i in perm]
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
        self._last_actor_per_token_logps = None
        loss = super().compute_loss(model, inputs, *args, **kwargs)
        # Schulman k3 estimator of KL(π_rollout || π_θ_new), MC'd on the sampled
        # token. Sufficient as a "general stability" scalar — unbiased, always
        # non-negative, low variance with B*T ~1.5M samples per rollout.
        # d = log π_θ_new(a) - log π_rollout(a); k3 = exp(d) - d - 1.
        actor_logps = self._last_actor_per_token_logps
        sampling_logps = inputs.get("sampling_per_token_logps")
        if actor_logps is not None and sampling_logps is not None:
            mask = inputs["completion_mask"]
            if "tool_mask" in inputs and inputs["tool_mask"] is not None:
                mask = mask * inputs["tool_mask"]
            mask_f = mask.float()
            n = mask_f.sum().clamp(min=1.0)
            d = (actor_logps - sampling_logps) * mask_f
            kl_k3 = ((torch.exp(d) - d - 1.0) * mask_f).sum() / n
            mean_log_diff = d.sum() / n  # signed; positive means θ_new prefers the sampled token more
            mode = "train" if self.model.training else "eval"
            self._metrics[mode].setdefault("diagnostics/kl_rollout_vs_new", []).append(
                self.accelerator.gather(kl_k3).mean().item()
            )
            self._metrics[mode].setdefault("diagnostics/logp_diff_new_minus_rollout", []).append(
                self.accelerator.gather(mean_log_diff).mean().item()
            )
        self._last_actor_per_token_logps = None
        return loss

    def _compute_kl_to_base_loss(self, model, inputs):
        """KL-to-base coherence loss.

        Loss = coh_kl_beta * mean_over_completion_tokens(logp_policy - logp_base),
        where logp_policy uses scales (retain=1, forget=0) (deployment state),
        and logp_base uses scales (0, 0) (base layer reduction via DualLoRA).

        Both forward passes use the same input tensors and completion_mask; only
        the adapter scales differ. Backward through this loss flows only to the
        retain adapter (forget contribution = 0 by construction in the policy
        forward).

        This bypasses the reward signal entirely for coherence samples — the only
        thing pulling the policy is the KL-to-base anchor.
        """
        from gradient_routing import set_scales

        prompt_ids = inputs["prompt_ids"]
        prompt_mask = inputs["prompt_mask"]
        completion_ids = inputs["completion_ids"]
        completion_mask = inputs["completion_mask"]
        prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.shape[1]
        sub_batch_size = getattr(self, "_scoring_batch_size",
                                 self.args.per_device_train_batch_size)

        saved_forget = self._train_forget_scale()
        # Policy forward at (retain=1, forget=0) — gradient ON.
        set_scales(model, retain_scale=1.0, forget_scale=0.0)
        try:
            policy_logp, _ = self._get_per_token_logps_and_entropies(
                model, prompt_completion_ids, attention_mask, logits_to_keep,
                sub_batch_size,
            )
            # Base forward at (0, 0) — gradient OFF.
            set_scales(model, retain_scale=0.0, forget_scale=0.0)
            with torch.no_grad():
                base_logp, _ = self._get_per_token_logps_and_entropies(
                    model, prompt_completion_ids, attention_mask, logits_to_keep,
                    sub_batch_size,
                )
        finally:
            set_scales(model, retain_scale=1.0, forget_scale=saved_forget)

        mask_f = completion_mask.float()
        n_tokens = mask_f.sum().clamp(min=1.0)
        # Schulman k3 KL estimator: with samples drawn from policy, the raw MC
        # term (policy_logp - base_logp) is a vanilla score function — its
        # gradient pushes policy AWAY from sampled tokens (REINFORCE without
        # baseline), not toward base. Use the k3 surrogate instead:
        #   KL(policy || base) ≈ E_policy[exp(d) - d - 1],  d = base - policy
        # which is non-negative, unbiased, and its gradient (w.r.t. policy_logp)
        # is (1 - exp(d)) — actively pulling policy_logp toward base_logp.
        # base_logp is detached so gradient flows only through policy_logp.
        log_ratio = (base_logp.detach() - policy_logp)
        kl_token = torch.exp(log_ratio) - log_ratio - 1.0
        kl_per_token = (kl_token * mask_f).sum() / n_tokens
        # Track for diagnostics.
        mode = "train" if model.training else "eval"
        self._metrics[mode].setdefault("coherence/kl_to_base", []).append(
            self.accelerator.gather(kl_per_token.detach()).mean().item()
        )
        loss = self._coh_kl_beta * kl_per_token
        return loss

    def _packed_per_token_logps(self, model, prompt_ids, prompt_mask,
                                completion_ids, completion_mask):
        """Per-token completion logps via the padding-free PACKED forward — the
        same computation as _packed_compute_loss, which a clean single-sequence
        ground-truth forward confirms is correct.

        This is a correct replacement for TRL's batched
        _get_per_token_logps_and_entropies, which mis-scores confident tokens in
        long completions (off by 15-25 nats on e.g. \\boxed{} digits) and blows up
        the GRPO ratio exp(new-old). Temperature-scaled to match the liger ratio
        convention (TRL's old path and the liger kernel both divide by T).
        Returns (N, max_comp_len), zeros on padding positions.
        """
        device = completion_ids.device
        N, max_comp_len = completion_ids.shape
        out = torch.zeros(N, max_comp_len, device=device)
        inputs = {
            "prompt_ids": prompt_ids, "prompt_mask": prompt_mask,
            "completion_ids": completion_ids, "completion_mask": completion_mask,
            "advantages": torch.zeros(N, device=device),  # unused for logps
        }
        token_counts = [int(p) + int(c) for p, c in zip(
            prompt_mask.sum(dim=1).tolist(), completion_mask.sum(dim=1).tolist())]
        max_tok = self._max_tokens_per_microbatch or 12000
        unwrapped = self.accelerator.unwrap_model(model)
        with torch.no_grad():
            for mb_idx in _pack_by_tokens(token_counts, list(range(N)), max_tok):
                packed = _pack_for_forward(inputs, mb_idx)
                hidden = unwrapped.model(
                    input_ids=packed["packed_input_ids"],
                    position_ids=packed["packed_position_ids"],
                    use_cache=False,
                ).last_hidden_state[0]
                cids = packed["completion_ids"]
                offset = 0
                for jj, (p_len, c_len) in enumerate(packed["seq_boundaries"]):
                    if c_len > 0:
                        hs = hidden[offset + p_len - 1: offset + p_len + c_len - 1]
                        logits = unwrapped.lm_head(hs.unsqueeze(0)).float() / self.temperature
                        lp = torch.nn.functional.log_softmax(logits, dim=-1)[0]
                        out[mb_idx[jj], :c_len] = lp.gather(
                            -1, cids[jj, :c_len].unsqueeze(-1)).squeeze(-1)
                    offset += p_len + c_len
        return out

    def _grpo_per_token_loss(self, last_hs_padded, unwrapped_model, packed_inputs,
                             *, kl_clamp=False, normalize="seq", return_diag=False):
        """Explicit GRPO per-token loss (N, max_comp_len), grad-carrying, with the liger
        loss_type='grpo' per-sequence normalization baked in PER TOKEN so that
        (ptl * completion_mask).sum() equals TRL's compute_loss return value: the liger 'grpo'
        kernel scalar divided by current_gradient_accumulation_steps (TRL grpo_trainer.py
        normalizes each microbatch loss by the accumulation factor in train mode; omitting that
        division here inflated accumulated gradients 64x at gpu_batch_size=4 / opt_batch=256 and
        collapsed the 2026-06-09 skyexcl token runs). This exactness is what makes token-level
        routing a clean ablation: for any disjoint token partition
        mask_good ⊕ mask_bad == completion_mask, loss_good + loss_bad == the full GRPO loss, so
        backward(loss_good)+backward(loss_bad) == backward(full). Mirrors liger
        chunked_loss/grpo_loss.py (importance_sampling_level='token'); config from the liger
        construction (beta, epsilon_low/high, temperature). lm_head is chunked over the batch dim
        to bound (N,T,V) logits memory.
        """
        loss_mask = packed_inputs["completion_mask"]            # (N, Tc) {0,1}
        cids = packed_inputs["completion_ids"]                  # (N, Tc)
        adv = packed_inputs["advantages"]                       # (N,)
        old = packed_inputs["old_per_token_logps"]              # (N, Tc) or None
        ref = packed_inputs["ref_per_token_logps"]              # (N, Tc) or None
        N = loss_mask.shape[0]
        mask_f = loss_mask.float()

        # Per-token policy logps (WITH grad), temperature-scaled to match the liger ratio convention.
        chunk = 2
        rows = []
        for b in range(0, N, chunk):
            logits = unwrapped_model.lm_head(last_hs_padded[b:b + chunk]).float() / self.temperature
            lp = torch.nn.functional.log_softmax(logits, dim=-1)
            rows.append(lp.gather(-1, cids[b:b + chunk].unsqueeze(-1)).squeeze(-1))
        logp = torch.cat(rows, dim=0)                           # (N, Tc), grad-carrying

        # `old` clamp guardrail (LOGP_RATIO_CLAMP) — same as _packed_compute_loss: keep the
        # importance ratio exp(new-old) from exploding on tokens TRL's old-logp path mis-scored.
        if old is None:
            old_t = logp.detach()
        else:
            old_t = old
            if LOGP_RATIO_CLAMP > 0:
                C = LOGP_RATIO_CLAMP
                clamped = old.clamp(min=logp.detach() - C, max=logp.detach() + C)
                old_t = torch.where(loss_mask.bool(), clamped, old)

        log_ratio = logp - old_t
        coef_1 = torch.exp(log_ratio)
        coef_2 = torch.clamp(coef_1, 1.0 - self.epsilon_low, 1.0 + self.epsilon_high)
        a = adv.unsqueeze(1)
        per_token = -torch.min(coef_1 * a, coef_2 * a)          # (N, Tc)
        kl = None
        if self.beta != 0.0:
            assert ref is not None, "beta!=0 requires ref_per_token_logps"
            d = ref - logp
            if kl_clamp:                                        # VERL-parity input clamp
                d = torch.clamp(d, -KL_CLAMP_D, KL_CLAMP_D)
            kl = torch.exp(d) - d - 1.0                         # Schulman k3, KL(policy||ref)
            if kl_clamp:                                        # VERL-parity output clamp (zeros
                kl = torch.clamp(kl, -KL_CLAMP_KLD, KL_CLAMP_KLD)  # KL grad on far-moved tokens)
            per_token = per_token + self.beta * kl

        if normalize == "seq":
            # liger loss_type='grpo' (per-sequence mean): Σ_{i,t} ptl ==
            # ( Σ_i (Σ_t per_token·mask)/seq_len_i ) / N == the liger 'grpo' scalar.
            # Then divide by TRL's train-mode normalizer (current_gradient_accumulation_steps) so the
            # accumulated gradient over an optimizer batch matches the TRL compute_loss path exactly.
            seq_len = mask_f.sum(dim=1).clamp(min=1.0)         # (N,)
            normalizer = float(self.current_gradient_accumulation_steps)  # TRL train-mode normalizer
            assert normalizer >= 1.0, f"bad accumulation normalizer {normalizer}"
            ptl = per_token * mask_f / (seq_len.unsqueeze(1) * float(N) * normalizer)
        elif normalize == "none":
            # Raw per-token contribution (Σ over the microbatch's tokens). The caller applies the
            # GLOBAL token-mean denominator (scale = 1/total_completion_tokens), so summed over all
            # microbatches the accumulated gradient is Σ(loss·mask)/Σ(mask) over the whole opt batch.
            ptl = per_token * mask_f
        else:
            raise ValueError(f"unknown normalize mode {normalize!r}")

        if return_diag:
            with torch.no_grad():
                m = mask_f
                denom = m.sum().clamp(min=1.0)
                diag = {"clip_frac": ((coef_1 != coef_2).float() * m).sum() / denom}
                if kl is not None:
                    diag["mean_kl"] = (kl * m).sum() / denom
                    diag["max_d"] = ((ref - logp).abs() * m).max()
            return ptl, diag
        return ptl

    def _routed_adam_feeds(self, is_good):
        """RoutedAdam m-stream feeds [(params, weight), ...] for one routed loss/microbatch.

        Stream-weight derivation (the single source of truth for these weights):
        for a mirrored retain/forget param pair, the routing_mode=none dual-adapter baseline
        gives EACH param m_{R+F}/sqrt(v_{R+F}), so the combined model moves by
        2*m_{R+F}/sqrt(v).  R = good-sample gradient stream, F = detected-bad stream.

          exclusive: retain m <- R,  forget m <- F            (sum m_{R+F}: each adapter is the
                     reference update with the off-stream momentum removed)
          classic:   retain m <- R,  forget m <- R + B*F      (B=2 default: sum 2*m_{R+F} ==
                     the dual-adapter baseline exactly. Classic carries R in BOTH adapters but
                     consolidates F into forget alone, so matching baseline combined dynamics
                     requires F at the multiplicity the baseline gave it.)

        v is the full stream (p.grad) for every param in both modes — see routed_adam.py.
        B = --routed_adam_classic_bad_weight. B=1 is the per-param "removed signal only"
        ablation: it fixes only the retain v denominator, leaving the forget adapter's
        m AND v exactly what hook-classic + AdamW already gave it.
        """
        retain_p = [p for p in self._retain_params if p.requires_grad]
        forget_p = [p for p in self._forget_params if p.requires_grad]
        if self._routing_mode == "classic":
            if is_good:
                return [(retain_p, 1.0), (forget_p, 1.0)]
            return [(forget_p, self._routed_adam_classic_bad_weight)]
        assert self._routing_mode == "exclusive", \
            f"routed_adam: unsupported routing_mode {self._routing_mode!r}"
        return [(retain_p, 1.0)] if is_good else [(forget_p, 1.0)]

    def _routed_adam_backward(self, loss, feeds, retain_graph=False):
        """Backward `loss`, accumulating the FULL gradient into p.grad (the shared
        second-moment / clipping stream) while adding weight * (this backward's grad delta)
        into p._routed_m_stream for each (params, weight) in `feeds`. NO gradient hooks —
        routing lives entirely in the optimizer's first moment. Streams accumulate across
        the optimizer window; RoutedAdam.step() zeroes them, aligned with HF's zero_grad().
        """
        snap = {}
        for params, _w in feeds:
            for p in params:
                if getattr(p, "_routed_m_stream", None) is None:
                    p._routed_m_stream = torch.zeros_like(p, dtype=torch.float32)
                assert p not in snap, "param appears in multiple routed_adam feeds"
                snap[p] = p.grad.detach().clone() if p.grad is not None else None
        self.accelerator.backward(loss, retain_graph=retain_graph)
        for params, w in feeds:
            for p in params:
                g0 = snap[p]
                delta = p.grad if g0 is None else p.grad - g0
                p._routed_m_stream += delta.float() * w

    def _token_routed_forward_backward(self, model, inputs, num_items_in_batch):
        """Token-level classic gradient routing (routing_granularity='token'). One shared packed
        forward; explicit per-token GRPO loss; TWO masked backwards through the shared graph:
          good tokens (non-behavior) -> both adapters (classic: no good-pass hook),
          behavior tokens            -> forget only (retain adapter grad zeroed).
        With _bad_pass_loss_scale=1 the forget adapter receives the full-batch gradient and the
        retain adapter the good-token-only gradient (exact recombination — see the
        gradient-equivalence test). Returns the (detached) scalar loss for logging.
        """
        N = inputs["completion_ids"].shape[0]
        packed = _pack_for_forward(inputs, list(range(N)))
        assert packed["token_routing_mask"] is not None, \
            "routing_granularity='token' but no token_routing_mask in the batch"

        # Stash one prompt+completion for wandb sample_text logging (compute_loss does this for
        # every other path; this path bypasses compute_loss, so without the stash token-routing
        # runs log no media at all).
        self._last_sample_prompt = self.processing_class.decode(
            inputs["prompt_ids"][0], skip_special_tokens=True)
        self._last_sample_completion = self.processing_class.decode(
            inputs["completion_ids"][0], skip_special_tokens=True)

        # Shared packed forward -> per-sequence completion hidden states (N, max_comp_len, H).
        # (Self-contained mirror of the forward in _packed_compute_loss so the dynamic-path liger
        # code stays untouched; the logit for completion token t is predicted by hidden state t-1.)
        unwrapped_model = self.accelerator.unwrap_model(model)
        out = unwrapped_model.model(
            input_ids=packed["packed_input_ids"],
            position_ids=packed["packed_position_ids"],
            use_cache=False,
        )
        hidden = out.last_hidden_state          # (1, T, H)
        n_seqs = packed["num_sequences"]
        max_comp_len = packed["max_comp_len"]
        last_hs = torch.zeros(n_seqs, max_comp_len, hidden.shape[-1],
                              device=hidden.device, dtype=hidden.dtype)
        offset = 0
        for j, (p_len, c_len) in enumerate(packed["seq_boundaries"]):
            if c_len > 0:
                last_hs[j, :c_len] = hidden[0, offset + p_len - 1: offset + p_len + c_len - 1]
            offset += p_len + c_len

        ptl = self._grpo_per_token_loss(last_hs, unwrapped_model, packed)   # (N, Tc), grad
        comp_mask = packed["completion_mask"].bool()
        troute = packed["token_routing_mask"] & comp_mask                  # behavior tokens
        mask_good = (comp_mask & ~troute).float()
        mask_bad = troute.float()
        loss_good = (ptl * mask_good).sum()
        loss_bad = (ptl * mask_bad).sum()

        if self._routed_adam:
            # RoutedAdam path: NO gradient hooks — routing lives in the optimizer. Both passes
            # accumulate the FULL gradient into p.grad (the second-moment / clipping stream);
            # per-pass deltas go to the m streams per _routed_adam_feeds (good tokens -> retain,
            # behavior tokens -> forget; token granularity is exclusive-only).
            assert self._bad_pass_loss_scale == 1.0, \
                "routed_adam requires bad_pass_loss_scale=1.0 (use routed_adam_kappa instead)"
            self._routed_adam_backward(loss_good, self._routed_adam_feeds(True),
                                       retain_graph=True)
            self._routed_adam_backward(loss_bad, self._routed_adam_feeds(False))
        else:
            # Pass 1: good tokens -> both adapters (classic good-pass = no hook; exclusive would zero forget).
            hooks = []
            if self._good_pass_hooked_params is not None:
                hooks = [p.register_hook(self._good_pass_hook_fn) for p in self._good_pass_hooked_params]
            self.accelerator.backward(loss_good, retain_graph=True)
            for h in hooks:
                h.remove()
            # Pass 2: behavior tokens -> forget only (retain adapter grad zeroed).
            hooks = [p.register_hook(lambda g: torch.zeros_like(g)) for p in self._retain_params]
            self.accelerator.backward(loss_bad * self._bad_pass_loss_scale)
            for h in hooks:
                h.remove()

        m = self._metrics.setdefault("train", {})
        ntok = comp_mask.float().sum().clamp(min=1.0)
        m.setdefault("routing/token_bad_frac", []).append((troute.float().sum() / ntok).item())
        m.setdefault("routing/token_bad_count", []).append(troute.float().sum().item())
        return loss_good.detach() + loss_bad.detach() * self._bad_pass_loss_scale

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

        loss_mask = packed_inputs["completion_mask"]  # (N, max_comp_len)

        # --- GRPO ratio guardrail (see LOGP_RATIO_CLAMP) ---
        # Recompute the policy's own (temperature-scaled) completion logps from
        # the same hidden states the loss uses, then clamp `old` into a window
        # around them so the liger ratio exp(new-old) can't explode on tokens
        # TRL's old-logp path mis-scored. This both prevents the loss blow-up
        # and pulls the bad importance weights back to ~correct.
        old_ptl = packed_inputs["old_per_token_logps"]
        # --- divergence diagnostic (env LOGP_DIV_DEBUG): locate the explosion source by
        # logging BOTH the IS-ratio exponent (new-old) and the KL exponent (ref-new) per
        # token; ref is computed via the same _get_per_token_logps path that can mis-score
        # confident tokens, so the KL term can blow up with old-new perfectly fine. Writes a
        # per-step summary (confirms execution + shows the pattern) plus the worst tokens. ---
        if os.environ.get("LOGP_DIV_DEBUG") and self.model.training and self.accelerator.is_main_process:
            import json as _json
            with torch.no_grad():
                _newT = torch.zeros_like(loss_mask, dtype=torch.float32)
                _cids = packed_inputs["completion_ids"]
                for _b in range(0, last_hs_padded.shape[0], 2):
                    _lg = unwrapped_model.lm_head(last_hs_padded[_b:_b + 2]).float() / self.temperature
                    _lp = torch.nn.functional.log_softmax(_lg, dim=-1)
                    _newT[_b:_b + 2] = _lp.gather(-1, _cids[_b:_b + 2].unsqueeze(-1)).squeeze(-1)
                _ref = packed_inputs.get("ref_per_token_logps")
                _m = loss_mask.bool()
                _ratio_d = (_newT - old_ptl) if old_ptl is not None else torch.zeros_like(_newT)
                _kl_d = (_ref - _newT) if _ref is not None else torch.zeros_like(_newT)
                _summary = {
                    "step": int(self._step), "old_is_none": old_ptl is None, "ref_is_none": _ref is None,
                    "max_ratio_d": round(float((_ratio_d * _m).abs().max()), 2),
                    "max_kl_d": round(float((_kl_d * _m).abs().max()), 2),
                    "n_ratio_gt5": int(((_ratio_d.abs() > 5) & _m).sum()),
                    "n_kl_gt5": int(((_kl_d.abs() > 5) & _m).sum()),
                }
                with open(os.path.join(self.args.output_dir, "logp_div_summary.jsonl"), "a") as _f:
                    _f.write(_json.dumps(_summary) + "\n")
                _hit = _m & ((_ratio_d.abs() > 5.0) | (_kl_d.abs() > 5.0))
                _rows = []
                for _r in _hit.nonzero(as_tuple=False)[:300]:
                    _i, _j = int(_r[0]), int(_r[1])
                    _tid = int(_cids[_i, _j])
                    _rows.append({
                        "step": int(self._step),
                        "old": None if old_ptl is None else round(float(old_ptl[_i, _j]), 2),
                        "new": round(float(_newT[_i, _j]), 2),
                        "ref": None if _ref is None else round(float(_ref[_i, _j]), 2),
                        "ratio_d_new_minus_old": round(float(_ratio_d[_i, _j]), 2),
                        "kl_d_ref_minus_new": round(float(_kl_d[_i, _j]), 2),
                        "token": self.processing_class.decode([_tid]),
                        "ctx": self.processing_class.decode(_cids[_i, max(0, _j - 8):_j + 1].tolist()),
                    })
                if _rows:
                    with open(os.path.join(self.args.output_dir, "logp_divergence.jsonl"), "a") as _f:
                        for _row in _rows:
                            _f.write(_json.dumps(_row) + "\n")
        if old_ptl is not None and LOGP_RATIO_CLAMP > 0:
            mode = "train" if self.model.training else "eval"
            with torch.no_grad():
                new_T = torch.zeros_like(loss_mask, dtype=torch.float32)
                cids = packed_inputs["completion_ids"]
                for b in range(0, last_hs_padded.shape[0], 2):
                    lg = unwrapped_model.lm_head(last_hs_padded[b:b + 2]).float() / self.temperature
                    lp = torch.nn.functional.log_softmax(lg, dim=-1)
                    new_T[b:b + 2] = lp.gather(-1, cids[b:b + 2].unsqueeze(-1)).squeeze(-1)
                C = LOGP_RATIO_CLAMP
                clamped = old_ptl.clamp(min=new_T - C, max=new_T + C)
                m = loss_mask.bool()
                packed_inputs["old_per_token_logps"] = torch.where(m, clamped, old_ptl)
                frac = (((old_ptl - clamped).abs() > 1e-4) & m).float().sum() / m.float().sum().clamp(min=1.0)
                self._metrics[mode].setdefault("diagnostics/old_logp_clamped_frac", []).append(
                    self.accelerator.gather(frac).mean().item())

        mode = "train" if self.model.training else "eval"
        if self._kl_clamp or self._token_mean_loss:
            # --- VERL-parity hand-rolled loss (bypasses liger) ---
            # The k3 clamp needs `new` (liger computes it internally and never exposes it), and
            # global token-mean needs the raw per-token sum — so both route through the explicit
            # loss here. The liger path below is the untouched default. memory verl-parity-loss-changes.
            ptl, diag = self._grpo_per_token_loss(
                last_hs_padded, unwrapped_model, packed_inputs,
                kl_clamp=self._kl_clamp,
                normalize=("none" if self._token_mean_loss else "seq"),
                return_diag=True,
            )
            loss = ptl.sum()
            if self.beta != 0.0 and "mean_kl" in diag:
                self._metrics[mode]["kl"].append(self.accelerator.gather(diag["mean_kl"]).mean().item())
                self._metrics[mode].setdefault("kl_max_d", []).append(
                    self.accelerator.gather(diag["max_d"]).max().item())
            self._metrics[mode]["clip_ratio"].append(
                self.accelerator.gather(diag["clip_frac"]).mean().item())
        else:
            # Call liger fused loss
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
            mean_kl = metrics[0] if self.beta != 0.0 else None
            clip_ratio = metrics[-1]
            if self.beta != 0.0 and mean_kl is not None:
                self._metrics[mode]["kl"].append(self.accelerator.gather(mean_kl).mean().item())
            self._metrics[mode]["clip_ratio"].append(self.accelerator.gather(clip_ratio).mean().item())

        # Liger's fused kernel doesn't expose per-token logps or entropy. Recover
        # them via a chunked lm_head pass on the same hidden states (no_grad — does
        # not interfere with the autograd graph liger built). Chunked over batch
        # to bound peak memory at chunk * T * V * 2 bytes (~0.9 GB for chunk=2,
        # T=1500, V=151k bf16). Adds ~5% to actor update time.
        sampling_logps = packed_inputs.get("sampling_per_token_logps")
        with torch.no_grad():
            B = last_hs_padded.shape[0]
            chunk = 2
            sel_ids = packed_inputs["completion_ids"]
            per_token_logps = torch.zeros_like(loss_mask, dtype=torch.float32)
            # Temperature-scaled logps (logits/T), matching TRL's old-logp path
            # and the liger kernel — for an apples-to-apples new-vs-old diagnostic.
            per_token_logps_temp = torch.zeros_like(loss_mask, dtype=torch.float32)
            entropies = torch.zeros_like(loss_mask, dtype=torch.float32)
            for b in range(0, B, chunk):
                hs = last_hs_padded[b:b + chunk]
                logits = unwrapped_model.lm_head(hs)
                log_probs = torch.nn.functional.log_softmax(logits.float(), dim=-1)
                log_probs_T = torch.nn.functional.log_softmax(
                    (logits.float() / self.temperature), dim=-1)
                ids = sel_ids[b:b + chunk]
                per_token_logps[b:b + chunk] = log_probs.gather(
                    -1, ids.unsqueeze(-1)
                ).squeeze(-1)
                per_token_logps_temp[b:b + chunk] = log_probs_T.gather(
                    -1, ids.unsqueeze(-1)
                ).squeeze(-1)
                # Entropy = -Σ p log p; computed in fp32 from log_probs.
                entropies[b:b + chunk] = -(log_probs.exp() * log_probs).sum(-1)
            mask_f = loss_mask.float()
            n = mask_f.sum().clamp(min=1.0)
            mean_entropy = (entropies * mask_f).sum() / n
            self._metrics[mode].setdefault("entropy", []).append(
                self.accelerator.gather(mean_entropy).mean().item()
            )
            _diag_mean_d = _diag_max_absd = _diag_kl = None
            if sampling_logps is not None:
                d = (per_token_logps - sampling_logps) * mask_f
                kl_k3 = ((torch.exp(d) - d - 1.0) * mask_f).sum() / n
                mean_d = d.sum() / n
                self._metrics[mode].setdefault("diagnostics/kl_rollout_vs_new", []).append(
                    self.accelerator.gather(kl_k3).mean().item()
                )
                self._metrics[mode].setdefault("diagnostics/logp_diff_new_minus_rollout", []).append(
                    self.accelerator.gather(mean_d).mean().item()
                )
                _diag_mean_d = mean_d.item()
                _diag_max_absd = (d.abs() * mask_f).max().item()
                _diag_kl = kl_k3.item()

        # Loss-explosion diagnostic: the liger ratio is exp(logp_new - logp_old).
        # Measure that exponent directly (new = per_token_logps computed above;
        # old = old_per_token_logps used by the kernel) to see whether a few
        # tokens with huge |new-old| drive the loss. Also report sampling/old
        # presence. Print in the early window or on any spike.
        if abs(loss.item()) > 100.0:  # safety monitor: only fires on a real spike
            adv = packed_inputs["advantages"]
            old = packed_inputs.get("old_per_token_logps")
            with torch.no_grad():
                if old is not None:
                    # Temperature-matched gap (the apples-to-apples one).
                    d_old_T = (per_token_logps_temp - old) * mask_f
                    max_dold_T = (d_old_T.abs() * mask_f).max().item()
                    n_big_T = int(((d_old_T.abs() > 5.0) & mask_f.bool()).sum().item())
                    # Unmatched (no-temp) gap, for reference.
                    d_old = (per_token_logps - old) * mask_f
                    abs_d = d_old_T.abs() * mask_f  # locate worst by the MATCHED gap
                    max_dold = (d_old.abs() * mask_f).max().item()
                    n_big = int(((d_old.abs() > 5.0) & mask_f.bool()).sum().item())
                    mean_dold = (d_old_T.sum() / n).item()
                    # Dump the single worst token's identity/context.
                    flat = abs_d.argmax().item()
                    jb, tb = flat // abs_d.shape[1], flat % abs_d.shape[1]
                    cids = packed_inputs["completion_ids"]
                    tok_id = int(cids[jb, tb].item())
                    c_len_j = int(mask_f[jb].sum().item())
                    eos_id = getattr(self.processing_class, "eos_token_id", None)
                    pad_id = getattr(self.processing_class, "pad_token_id", None)
                    # GROUND TRUTH: clean single-sequence full forward of the worst
                    # sequence (no packing/padding/batch), correct positions. Settles
                    # whether old or new matches the true logp on the real token.
                    gt_T = gt_noT = None
                    pid = packed_inputs.get("packed_input_ids")
                    sb = packed_inputs.get("seq_boundaries")
                    if pid is not None and sb is not None and jb < len(sb):
                        p_len_j, c_len_sb = sb[jb]
                        off = sum(p + c for (p, c) in sb[:jb])
                        full_seq = pid[0, off: off + p_len_j + c_len_sb]
                        gt_logits = unwrapped_model(full_seq.unsqueeze(0)).logits[0].float()
                        gpos = p_len_j + tb - 1
                        gt_T = torch.log_softmax(gt_logits[gpos] / self.temperature, dim=-1)[tok_id].item()
                        gt_noT = torch.log_softmax(gt_logits[gpos], dim=-1)[tok_id].item()
                    worst = (f"worst[j={jb} t={tb}/{c_len_j} tok={tok_id} "
                             f"is_eos={tok_id==eos_id} is_pad={tok_id==pad_id} "
                             f"trunc={c_len_j>=self.max_completion_length} "
                             f"old={old[jb,tb].item():.2f} "
                             f"new_T={per_token_logps_temp[jb,tb].item():.2f} "
                             f"GT_T={'NA' if gt_T is None else round(gt_T,2)} "
                             f"GT_noT={'NA' if gt_noT is None else round(gt_noT,2)}]")
                else:
                    max_dold = n_big = mean_dold = max_dold_T = n_big_T = None
                    worst = ""
            print(f"[LOSSDIAG step={self._step} loss={loss.item():.3e} "
                  f"adv=[{adv.min().item():.2f},{adv.max().item():.2f}] "
                  f"Tmatched(mean={None if mean_dold is None else round(mean_dold,3)},"
                  f"max|d|={None if max_dold_T is None else round(max_dold_T,2)},n>5={n_big_T}) "
                  f"noT(max|d|={None if max_dold is None else round(max_dold,2)},n>5={n_big}) "
                  f"old_present={old is not None} sampling_present={sampling_logps is not None} "
                  f"{worst}", flush=True)

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
            old_logps = (_tm.get("timing/old_logprobs") or [0])[-1]
            ref_logps = (_tm.get("timing/ref_logprobs") or [0])[-1]
            full_step = (_tm.get("timing/full_step_s") or [0])[-1]
            between_grpo = (_tm.get("timing/between_grpo_iters") or [0])[-1]
            dataloader = (_tm.get("timing/between_grpo_iters/dataloader") or [0])[-1]
            post_step = (_tm.get("timing/post_step") or [0])[-1]
            eval_time = (_tm.get("timing/eval") or [0])[-1]
            step = self.state.global_step
            parts = [
                f"rollout={rollout:.1f}s (sync={sync:.1f}s gen={gen:.1f}s)",
                f"old_logps={old_logps:.1f}s",
                f"ref_logps={ref_logps:.1f}s",
                f"reward_t={reward:.1f}s",
                f"update={update:.1f}s",
                f"between_grpo={between_grpo:.1f}s (dataloader={dataloader:.1f}s)",
                f"post_step={post_step:.1f}s",
                f"full_step={full_step:.1f}s",
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

        # Adapter param-norm snapshot every log() call (so the cadence matches
        # logging_steps). Used to track how weight decay constrains retain/forget
        # L2 norm over training.
        if self._retain_params or self._forget_params:
            with torch.no_grad():
                r_sq = sum((p.detach().float() ** 2).sum().item() for p in self._retain_params) if self._retain_params else 0.0
                f_sq = sum((p.detach().float() ** 2).sum().item() for p in self._forget_params) if self._forget_params else 0.0
                r_n = r_sq ** 0.5
                f_n = f_sq ** 0.5
            print(f"[NORMS @{self.state.global_step}] retain_l2={r_n:.4f} forget_l2={f_n:.4f}")

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
            "diagnostics/clip_ratio_low_mean": "clip_ratio/low_mean",
            "diagnostics/clip_ratio_low_min": "clip_ratio/low_min",
            "diagnostics/clip_ratio_high_mean": "clip_ratio/high_mean",
            "diagnostics/clip_ratio_high_max": "clip_ratio/high_max",
            "diagnostics/clip_ratio_region_mean": "clip_ratio/region_mean",
            # Liger fused-loss path (use_liger_kernel=True) writes a single combined
            # clip_ratio scalar instead of the low/high/region split — surface it under
            # the same diagnostics namespace.
            "diagnostics/clip_ratio": "clip_ratio",
        }
        for new_key, old_key in _dup_from_trl.items():
            vals = _tm.get(old_key)
            if vals:
                _tm.setdefault(new_key, []).append(vals[-1])

        # Top-level prefixes that should NOT get the train/ prefix
        _TOP_LEVEL_PREFIXES = ("timing/", "reward/", "diagnostics/", "memory/", "coherence/", "routing/", "sampling/", "offpolicy_drift/", "offpolicy_drift_baseline/")

        top_level = {}
        keys_to_remove = []
        for key, vals in _tm.items():
            if key.startswith(_TOP_LEVEL_PREFIXES) and vals:
                top_level[key] = sum(vals) / len(vals)
                keys_to_remove.append(key)
        for key in keys_to_remove:
            del _tm[key]

        # Env-gated per-step timing dump for profiling (PROFILE_TIMING=1). File-based
        # so it survives the train.log sync quirk; inert otherwise. top_level already
        # holds every per-step timing/* value (averaged over the logging interval).
        if os.environ.get("PROFILE_TIMING"):
            _prof = {k: v for k, v in top_level.items() if k.startswith("timing/")}
            _prof["step"] = self.state.global_step
            try:
                with open(os.path.join(self.args.output_dir, "profile_timing.jsonl"), "a") as _pf:
                    _pf.write(json.dumps(_prof) + "\n")
            except Exception:
                pass

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
        coh_eval_metrics = dict(self._coh_eval_metrics) if self._coh_eval_metrics else {}
        coh_split_n = self._coh_eval_split_n

        def _score_in_background():
            try:
                try:
                    from envs.leetcode import use_eval_evaluator_on_this_thread
                    use_eval_evaluator_on_this_thread()
                except ImportError:
                    pass

                t_score = time.time()
                # Dual-env: split each mode's samples into routing-env (first
                # coh_split_n) and coherence-env (rest), scored with their own
                # metrics + env-homogeneous eval_data slices.
                main_samples = samples_by_mode
                main_eval_data = eval_data
                coh_samples = None
                coh_eval_data = None
                if coh_split_n is not None and coh_eval_metrics:
                    main_samples = {m: s[:coh_split_n] for m, s in samples_by_mode.items()}
                    coh_samples = {m: s[coh_split_n:] for m, s in samples_by_mode.items()}
                    if eval_data is not None:
                        main_eval_data = eval_data[:coh_split_n]
                        coh_eval_data = eval_data[coh_split_n:]

                results = score_eval_samples(main_samples, eval_metrics, eval_data=main_eval_data)
                elapsed = gen_elapsed + (time.time() - t_score)

                if verbose:
                    print(f"\n{format_routing_eval(results, step=step)}  "
                          f"(gen={gen_elapsed:.1f}s score={time.time()-t_score:.1f}s total={elapsed:.1f}s)\n")

                eval_flat = get_routing_eval_metrics(results)
                eval_flat["eval/elapsed_s"] = elapsed

                # Coherence-env metrics (failure-tolerant: must never break the
                # routing eval or training). Logged under coh/<env>/<mode>/<metric>.
                coh_record = {}
                if coh_samples is not None:
                    try:
                        coh_results = score_eval_samples(coh_samples, coh_eval_metrics,
                                                         eval_data=coh_eval_data)
                        for mode_name, mode_data in coh_results.items():
                            for rname, rdata in mode_data["metrics"].items():
                                if rdata["mean"] is None:
                                    continue
                                key = f"coh/{self._coherence_env}/{mode_name}/{rname}"
                                eval_flat[f"routing_eval/{key}"] = rdata["mean"]
                                coh_record[key] = rdata["mean"]
                        # Diagnostic: dump first few coherence-eval samples so we can
                        # see gold vs completion vs score (e.g. is the model emitting
                        # \boxed, is gold aligned). retain_only mode mirrors the coh
                        # training generation scales (1,0).
                        try:
                            dbg_mode = "retain_only" if "retain_only" in coh_results else list(coh_results)[0]
                            dbg_samps = coh_samples.get(dbg_mode, [])[:3]
                            dbg_data = (coh_eval_data[:3] if coh_eval_data else [])
                            dbg_scores = coh_results[dbg_mode]["metrics"].get(
                                list(coh_eval_metrics)[0], {}).get("values", [])
                            for di, s in enumerate(dbg_samps):
                                g = dbg_data[di].get("gold") if di < len(dbg_data) else None
                                sc = dbg_scores[di] if di < len(dbg_scores) else None
                                print(f"[coh-eval-dbg {dbg_mode} #{di}] gold={g!r} score={sc} "
                                      f"comp={s.get('completion','')[:160]!r}")
                        except Exception as de:
                            print(f"[coh-eval-dbg] failed: {type(de).__name__}: {de}")
                    except Exception as e:
                        print(f"[coherence-eval] scoring failed (non-fatal): {type(e).__name__}: {e}")

                self._pending_eval_wandb = eval_flat

                record = {"step": step, "eval_elapsed_s": round(elapsed, 1)}
                record.update(coh_record)
                for mode_name, mode_data in results.items():
                    for rname, rdata in mode_data["metrics"].items():
                        if rdata["mean"] is None:
                            continue  # metric not applicable (e.g. conditional column missing)
                        record[f"{mode_name}/{rname}"] = rdata["mean"]
                    record[f"{mode_name}/unique"] = mode_data["diversity"]["unique_samples"]
                    record[f"{mode_name}/jaccard"] = mode_data["diversity"]["avg_jaccard_similarity"]
                log_path = os.path.join(output_dir, "routing_eval.jsonl")
                os.makedirs(os.path.dirname(log_path), exist_ok=True)
                with open(log_path, "a") as f:
                    f.write(json.dumps(record) + "\n")

                # --- Diagnostic: log eval completions per mode, with per-reward
                # scores and eval_data columns. First N_FULL rows include the
                # full prompt+completion text (for side-by-side inspection vs
                # train_samples.jsonl); subsequent rows up to N_TOTAL omit the
                # text but keep id/hackable/detectable + per-metric scores so
                # any post-hoc partition (e.g. corrected detectable from a
                # different rh_detector) can be recomputed without re-running
                # the model.
                try:
                    samples_log_path = os.path.join(output_dir, "eval_samples.jsonl")
                    N_FULL = 32
                    with open(samples_log_path, "a") as f:
                        for mode_name, mode_data in results.items():
                            values_per_metric = {
                                rname: rdata.get("values", [])
                                for rname, rdata in mode_data["metrics"].items()
                            }
                            mode_samples = main_samples.get(mode_name, [])
                            n_total = len(mode_samples)
                            for i in range(n_total):
                                rec = {"step": step, "mode": mode_name}
                                if i < N_FULL:
                                    rec["prompt"] = mode_samples[i]["prompt"] if isinstance(mode_samples[i]["prompt"], str) else str(mode_samples[i]["prompt"])
                                    rec["completion"] = mode_samples[i]["completion"]
                                for rname, vals in values_per_metric.items():
                                    # Conditional metric wrappers (e.g.
                                    # hack_freq_detectable) return a list whose
                                    # length is the matching subset, not n_total —
                                    # those values are NOT positionally aligned
                                    # with the full eval list, so logging
                                    # vals[i] would attribute a subset score to
                                    # the wrong row. Skip those metrics here.
                                    if len(vals) != n_total:
                                        continue
                                    if vals[i] is None:
                                        continue
                                    rec[f"score/{rname}"] = float(vals[i])
                                # Use main_eval_data (aligned with main_samples/results);
                                # dual-env appends coh rows that aren't in `results`.
                                _diag_eval_data = main_eval_data if main_eval_data is not None else eval_data
                                if _diag_eval_data is not None and i < len(_diag_eval_data):
                                    for k, v in _diag_eval_data[i].items():
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
        n_eval = getattr(env_args, 'routing_eval_prompts', 64) if env_args is not None else 64

        if env_spec is not None and env_spec.load_eval_prompts is not None:
            eval_data = env_spec.load_eval_prompts(n_eval, env_args)
            eval_prompts = [d["prompt"] for d in eval_data]
            eval_max_tokens = env_spec.eval_max_tokens
        elif getattr(self, '_environment', 'stories') == 'arithmetic':
            from eval_utils import load_arithmetic_eval_prompts
            n_digits = getattr(self, '_n_digits', 3)
            eval_prompts = load_arithmetic_eval_prompts(n=n_eval, n_digits=n_digits)
            eval_max_tokens = n_digits + 2
        else:
            eval_max_tokens = 128

        _inject_detectable_into_eval_data(eval_data, getattr(self, "_rh_classifiable_fn", None))

        t0 = time.time()
        samples_by_mode = eval_gradient_routing(
            self.model, self.processing_class, self.eval_metrics,
            n_samples=n_eval, max_new_tokens=eval_max_tokens,
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

    # ----- Routing trace (debug) -----
    # All tracing logic is gated on self._trace_routing and writes JSONL
    # records with a top-level "trace" field ("rollout" / "sample" /
    # "sample_text" / "mb") so callers can filter with
    # jq 'select(.trace=="mb")'. Stdout mirror lines carry a [TRACE ...]
    # prefix for grepping train.log.

    def _trace_write(self, record):
        """Append one JSONL record to routing_trace.jsonl. No-op unless --trace_routing."""
        if not self._trace_routing:
            return
        if self._trace_file is None:
            path = os.path.join(self.args.output_dir, "routing_trace.jsonl")
            self._trace_file = open(path, "a", buffering=1)  # line-buffered
            print(f"[TRACE init] writing to {path}")
        self._trace_file.write(json.dumps(record) + "\n")

    @staticmethod
    def _grad_sqnorm(params):
        """Sum of squared .grad values across params. Per-microbatch delta
        isolates this microbatch's contribution to the aggregate grad norm.
        """
        total = 0.0
        for p in params:
            if p.grad is not None:
                total += float(p.grad.detach().pow(2).sum().item())
        return total

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

    def _train_forget_scale(self):
        """Forget-adapter scale to use during training forward+backward and
        post-coh restores. Under forget_scale_modulation=ema_clamp, the
        training scale is the same clamp used at rollout-time generation, so
        the clamp actually bites the trained policy (not just the rollout
        distribution). Without this, the clamp would only affect the data
        distribution while the trained policy continued to use the full
        forget contribution at every forward+backward."""
        if self._forget_scale_modulation == "ema_clamp":
            return self._forget_scale_clamp
        return 1.0

    def _in_retain_warmup(self):
        """Idea 4(a): first N optimizer steps where the entire rollout is routed
        through the coh-side training path."""
        return (self._retain_warmup_steps > 0
                and getattr(self.state, "global_step", 0) < self._retain_warmup_steps)

    def _in_forget_warmup(self):
        """Idea 4(b): the N steps after retain warmup where only forget adapter
        updates on rh-detected samples (non-rh + coh dropped)."""
        if self._forget_warmup_steps <= 0:
            return False
        step = getattr(self.state, "global_step", 0)
        return (self._retain_warmup_steps <= step
                < self._retain_warmup_steps + self._forget_warmup_steps)

    def _maybe_swap_coherence_prompts(self, inputs):
        """Replace the last K unique prompts in `inputs` with prompts drawn from
        the classifiable-only iterator, so interlaced-coherence slots see only
        prompts the detector can verify. K = coh_samples_per_rollout /
        num_generations. Caller passes the flat completion-level list; each
        unique prompt is repeated G=num_generations times contiguously, so K
        unique replacements cover C = K*G trailing completion slots.
        Retain warmup: swap ALL prompts (K = len(inputs)/G).
        Forget warmup: do not swap (use raw prompts, no coh slice in this phase).
        """
        # Fire when either (a) retain-verification swaps in classifiable main-env
        # prompts, or (b) dual-env coherence swaps in coherence-env prompts.
        dual_env = self._coherence_env is not None
        if not ((self._rh_detector_verifies_retain_samples or dual_env) and
                self._interlaced_coh and self.model.training):
            return inputs
        if self._in_forget_warmup():
            return inputs
        G = self.num_generations
        if self._in_retain_warmup() and not dual_env:
            K = len(inputs) // G
        else:
            C = self._coh_samples_per_rollout
            assert C % G == 0, (
                f"coh_samples_per_rollout ({C}) must be a multiple of num_generations ({G})"
            )
            K = C // G
        assert len(inputs) >= K * G, f"inputs has {len(inputs)} entries, need at least {K * G}"
        it = self._ensure_coherence_iter() if dual_env else self._ensure_detectable_iter()
        inputs = list(inputs)
        for k in range(K):
            new_prompt = next(it)
            start = len(inputs) - (K - k) * G
            for g in range(G):
                inputs[start + g] = dict(new_prompt)
        return inputs

    def _ensure_coherence_iter(self):
        """Lazy-build an infinite shuffled iterator over the coherence-env
        dataset (dual-env coherence). Unlike _ensure_detectable_iter, no detector
        / classifiable filtering — the coherence env (e.g. math) has no hack."""
        if self._coh_iter is not None:
            return self._coh_iter
        assert self._coh_dataset is not None, (
            "coherence_env set but _coh_dataset is None — check _run wiring."
        )
        rows = self._coh_dataset.rows
        assert len(rows) > 0, "coherence dataset is empty"
        print(f"Dual-env coherence iterator: {len(rows)} {self._coherence_env} prompts")

        def _gen():
            while True:
                order = torch.randperm(len(rows)).tolist()
                for i in order:
                    yield rows[i]

        self._coh_iter = _gen()
        return self._coh_iter

    def _ensure_detectable_iter(self):
        """Lazy-build the classifiable-prompt iterator for retain verification.

        Calls rh_classifiable_fn on self.train_dataset row-by-row, filters to
        classifiable rows, and returns an infinite shuffled iterator yielding
        one prompt dict at a time (consumed by _maybe_swap_coherence_prompts).
        Works against both HF Datasets (ds[col] -> list) and the local
        _ListDataset wrapper (row-list behind .rows) via a duck-typed path.
        """
        if self._detectable_iter is not None:
            return self._detectable_iter
        assert self._rh_classifiable_fn is not None, (
            "rh_detector_verifies_retain_samples=True but rh_classifiable_fn is None "
            "— check that the detector is registered in RH_CLASSIFIABLE_REGISTRY."
        )
        ds = self.train_dataset
        rows = ds.rows if hasattr(ds, "rows") else [ds[i] for i in range(len(ds))]
        cols = {c: [row.get(c) for row in rows] for c in ds.column_names}
        mask = self._rh_classifiable_fn(**cols)
        classifiable_idx = [i for i, m in enumerate(mask) if m]
        # Restrict pool to hackable+detectable prompts. Unhackable+detectable
        # prompts produce uniform-reward groups under per-group GRPO renorm
        # (no within-group hack vs non-hack variance, retain reward is the
        # only signal and is similar across generations on the same prompt),
        # so they contribute ~zero gradient to extras-side training.
        # Restricting to hackable+detectable preserves within-group variance
        # from the mix of hack vs non-hack completions per prompt and gives
        # the extras meaningful learning signal.
        hackable_col = cols.get("hackable")
        if hackable_col is not None:
            classifiable_idx = [i for i in classifiable_idx if hackable_col[i]]
        assert len(classifiable_idx) > 0, (
            "rh_detector_verifies_retain_samples=True but no prompts in train_dataset "
            "are hackable+classifiable — check detector params and dataset coverage."
        )
        self._detectable_rows = [rows[i] for i in classifiable_idx]
        print(f"Hackable+detectable subset for retain verification: "
              f"{len(self._detectable_rows)} / {len(rows)} prompts kept")

        def _gen():
            while True:
                order = torch.randperm(len(self._detectable_rows)).tolist()
                for i in order:
                    yield self._detectable_rows[i]

        self._detectable_iter = _gen()
        return self._detectable_iter

    def _log_hack_partition_diagnostics(self, output, mode):
        """Compute hack-vs-correct partition diagnostics on the FINAL post-rewrite
        advantages. Originally lived in trl_overrides.generate_and_score_completions
        but was reading the pre-rewrite GRPO advantages, so neg-adv stats never
        reflected the RP/filter/verified-only adjustments the optimizer actually
        consumes. Called at the end of _generate_and_score_completions."""
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
        if cr is None:
            return

        device = self.accelerator.device
        forget_raw_local = None    # pre-gate sum of forget-role scores
        forget_post_local = None   # post-gate sum (zeros where hackable=False)
        retain_main_local = None
        retain_main_scale = -1.0
        pre_gate = getattr(cr, '_last_pre_gate_forget_scores', {})
        for (name, fn, scale, role) in cr.components:
            scores_post = getattr(fn, '_last_scores', None)
            if scores_post is None:
                continue
            scores_post_t = torch.tensor(scores_post, device=device, dtype=torch.float32)
            if role == "forget":
                forget_post_local = (scores_post_t if forget_post_local is None
                                     else forget_post_local + scores_post_t)
                raw = pre_gate.get(name, scores_post)
                raw_t = torch.tensor(raw, device=device, dtype=torch.float32)
                forget_raw_local = (raw_t if forget_raw_local is None
                                    else forget_raw_local + raw_t)
            elif role == "retain" and scale > retain_main_scale:
                retain_main_local = scores_post_t
                retain_main_scale = scale
        if forget_raw_local is None or retain_main_local is None:
            return

        forget_raw = self.accelerator.gather(forget_raw_local)
        forget_post = self.accelerator.gather(forget_post_local)
        retain_full = self.accelerator.gather(retain_main_local)
        adv_full = self.accelerator.gather(output["advantages"]).to(forget_raw.device).float()
        assert forget_raw.shape == retain_full.shape == adv_full.shape, (
            f"shape mismatch: forget_raw={forget_raw.shape} retain={retain_full.shape} "
            f"adv={adv_full.shape}"
        )
        is_hack_emitted = forget_raw > 0
        is_hack_rewarded = forget_post > 0
        is_correct = retain_full > 0

        m = self._metrics.setdefault(mode, {})
        m.setdefault("diagnostics/hack_emitted_freq", []).append(
            is_hack_emitted.float().mean().item()
        )
        m.setdefault("diagnostics/hack_rewarded_freq", []).append(
            is_hack_rewarded.float().mean().item()
        )
        # Hacks that fired but the hackable gate zeroed the reward.
        gate_suppressed = is_hack_emitted & (~is_hack_rewarded)
        m.setdefault("diagnostics/hack_gate_suppressed_freq", []).append(
            gate_suppressed.float().mean().item()
        )

        # Partition by emission (raw) × correctness; advantage stats per group.
        partitions = [
            ("hack_only", is_hack_emitted & (~is_correct)),
            ("hack_and_correct", is_hack_emitted & is_correct),
            ("correct_only", (~is_hack_emitted) & is_correct),
            ("neither", (~is_hack_emitted) & (~is_correct)),
        ]
        for label, mask in partitions:
            m.setdefault(f"diagnostics/frac_{label}", []).append(mask.float().mean().item())
            n = int(mask.sum().item())
            if n > 0:
                advs = adv_full[mask]
                m.setdefault(f"diagnostics/adv_{label}_mean", []).append(advs.mean().item())
                m.setdefault(f"diagnostics/adv_{label}_min", []).append(advs.min().item())
                m.setdefault(f"diagnostics/adv_{label}_max", []).append(advs.max().item())
                if n > 1:
                    m.setdefault(f"diagnostics/adv_{label}_std", []).append(advs.std().item())

        # Two flavours of "unrewarded hack":
        #   gate-suppressed: emitted but hackable=False so the reward was zeroed.
        #   neg-advantage: emitted and rewarded but the optimizer-visible
        #     advantage is <= 0 (GRPO group mean higher, or RP penalty pushed
        #     this sample below the group mean post-rewrite).
        n_emitted = int(is_hack_emitted.sum().item())
        if n_emitted > 0:
            emitted_advs = adv_full[is_hack_emitted]
            m.setdefault("diagnostics/adv_hack_emitted_mean", []).append(
                emitted_advs.mean().item()
            )
            m.setdefault("diagnostics/hack_emitted_neg_adv_frac", []).append(
                (emitted_advs <= 0).float().mean().item()
            )
        n_rewarded = int(is_hack_rewarded.sum().item())
        if n_rewarded > 0:
            rewarded_advs = adv_full[is_hack_rewarded]
            m.setdefault("diagnostics/hack_rewarded_neg_adv_frac", []).append(
                (rewarded_advs <= 0).float().mean().item()
            )

    def _generate_and_score_completions(self, inputs):
        """Override: pad on CPU + single .to(device), then RH detection."""
        inputs = self._maybe_swap_coherence_prompts(inputs)
        _rollout_t0 = time.perf_counter()
        output = generate_and_score_completions(self, inputs)

        # --- Step B: REINFORCE advantage override ---
        device = self.accelerator.device

        # --- Step A.5: Attach is_coherence mask (interlaced mode) ---
        # Interlaced coherence: designate the last C slots as coherence samples
        # (see _generate_single_turn for the matching prompt-ordering contract).
        # The mask rides through _prepare_inputs' split/shuffle and is consumed
        # per-opt-batch in _dynamic_microbatch_forward_backward.
        n_total = output["completion_ids"].shape[0]
        if self._interlaced_coh and self.model.training:
            if self._in_retain_warmup():
                C = n_total  # Idea 4(a): whole rollout treated as coh
            elif self._in_forget_warmup():
                C = 0  # Idea 4(b): no coh slice this phase
            else:
                C = self._coh_samples_per_rollout
            assert C <= n_total, f"coh_samples ({C}) > rollout samples ({n_total})"
            is_coherence = torch.zeros(n_total, dtype=torch.bool, device=device)
            is_coherence[n_total - C:] = True
            output["is_coherence"] = is_coherence

            # Recompute old_per_token_logps for coh slice under retain-only scales.
            # The first pass (inside generate_and_score_completions) ran at (1,1) scales;
            # coh samples were generated by vLLM at (1,0), so their old_per_token_logps
            # needs to match the (1,0) policy for the GRPO ratio to be unbiased.
            if output.get("old_per_token_logps") is not None and C > 0:
                from trl.models.utils import disable_gradient_checkpointing
                from gradient_routing import set_scales

                coh_slice = slice(n_total - C, n_total)
                sub_prompt_ids = output["prompt_ids"][coh_slice]
                sub_prompt_mask = output["prompt_mask"][coh_slice]
                sub_completion_ids = output["completion_ids"][coh_slice]
                sub_completion_mask = output["completion_mask"][coh_slice]
                sub_prompt_completion_ids = torch.cat([sub_prompt_ids, sub_completion_ids], dim=1)
                sub_attention_mask = torch.cat([sub_prompt_mask, sub_completion_mask], dim=1)
                sub_logits_to_keep = sub_completion_ids.shape[1]
                sub_batch_size = getattr(self, "_scoring_batch_size",
                                         self.args.per_device_train_batch_size)

                set_scales(self.model, retain_scale=1.0, forget_scale=0.0)
                try:
                    with torch.no_grad(), disable_gradient_checkpointing(
                        self.model, self.args.gradient_checkpointing_kwargs
                    ):
                        coh_old_logps, _ = self._get_per_token_logps_and_entropies(
                            self.model,
                            sub_prompt_completion_ids,
                            sub_attention_mask,
                            sub_logits_to_keep,
                            sub_batch_size,
                        )
                finally:
                    set_scales(self.model, retain_scale=1.0,
                               forget_scale=self._train_forget_scale())
                output["old_per_token_logps"][coh_slice] = coh_old_logps
        else:
            output["is_coherence"] = torch.zeros(n_total, dtype=torch.bool, device=device)
        if self._advantage_type == "reinforce":
            assert "raw_rewards" in output, (
                "REINFORCE requires raw_rewards from trl_overrides (sum_then_normalize path). "
                "Check that multi_objective_aggregation='sum_then_normalize'."
            )
            raw_rewards = output["raw_rewards"]
            output["advantages"] = self._reinforce_advantages(raw_rewards, self._all_reward_buffer)

        # --- Step C: RH detection ---
        _t_rh_start = time.perf_counter()

        needs_detection = (self.gradient_routing_enabled
                           or self._filter_baseline
                           or self._reward_penalty_baseline
                           or self._verified_only_training)
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
            # and we would not be able to route them. Override with
            # detect_unhackable when the detector is itself the ground truth
            # (e.g. an LLM judge flagging hack attempts universally).
            # Idea 4(c): during warmup phases, optionally swap to a perfect detector
            # (e.g. score_threshold). When active, also skip the detectable filter
            # since the perfect detector can fire on undetectable prompts too.
            using_warmup_detector = (
                self.warmup_rh_detector is not None
                and (self._in_retain_warmup() or self._in_forget_warmup())
            )
            active_rh_detector = (self.warmup_rh_detector if using_warmup_detector
                                  else self.rh_detector)

            candidate = [True] * n_samples
            if not self._detect_unhackable:
                hackable_flags = detector_kwargs.get("hackable")
                # Per-row guard: any(...) not [0] — in dual-env, row 0 may be a
                # coherence row with hackable=None, which would wrongly skip the
                # whole gate. None (non-routing rows) -> not a candidate.
                if hackable_flags is not None and any(h is not None for h in hackable_flags):
                    candidate = [c and bool(h) for c, h in zip(candidate, hackable_flags)]
            detectable_flags = detector_kwargs.get("detectable")
            if (detectable_flags is not None
                    and any(d is not None for d in detectable_flags)
                    and not using_warmup_detector):
                candidate = [c and bool(d) for c, d in zip(candidate, detectable_flags)]
            if self._routed_reward is not None and self._routed_reward._last_eligible is not None:
                eligible = self._routed_reward._last_eligible
                candidate = [c and e for c, e in zip(candidate, eligible)]
            # Dual-env: coherence-env rows (e.g. math) are never routing candidates —
            # they have no hack and no detector. Coerce to clean bools too.
            if self._coherence_env is not None and "is_coherence" in output:
                is_coh = output["is_coherence"].tolist()
                candidate = [bool(c) and not ic for c, ic in zip(candidate, is_coh)]

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
                if self._coherence_env is not None:
                    # Dual-env: DualEnvReward scored main-env rows as a slice, so
                    # the detector's cached trait scores are main-env-length. Run
                    # the detector only on main-env rows (aligns with the cache),
                    # then scatter back; non-main rows are never RH.
                    env_tags = detector_kwargs.get("env_tag")
                    main_idx = ([i for i, t in enumerate(env_tags) if t == self._environment]
                                if env_tags is not None else list(range(n_samples)))
                    sub_comps = [completions_for_rh[i] for i in main_idx]
                    sub_prompts = [prompts_for_rh[i] for i in main_idx]
                    sub_kwargs = {
                        k: ([v[i] for i in main_idx] if isinstance(v, list) and len(v) == n_samples else v)
                        for k, v in detector_kwargs.items()
                    }
                    sub_rh = active_rh_detector(sub_comps, prompts=sub_prompts, **sub_kwargs)
                    is_rh_raw = [False] * n_samples
                    for j, i in enumerate(main_idx):
                        is_rh_raw[i] = sub_rh[j]
                else:
                    is_rh_raw = active_rh_detector(completions_for_rh, prompts=prompts_for_rh, **detector_kwargs)
                is_rh = [c and r for c, r in zip(candidate, is_rh_raw)]
            else:
                is_rh = [False] * n_samples

            is_rh_tensor = torch.tensor(is_rh, dtype=torch.bool, device=device)

            # --- Idea 2: EMA-driven forget-scale clamp ---
            # Update an EMA of the routing-slice hack rate (only over hackable
            # samples in the slice). When the EMA exceeds the configured target,
            # multiply the one-way forget-scale clamp by --forget_scale_decay.
            # Clamp is read in _generate_single_turn next rollout to scale the
            # routing eid's forget_scale.
            if self._forget_scale_modulation == "ema_clamp":
                if self._interlaced_coh:
                    slice_mask = ~output["is_coherence"]
                else:
                    slice_mask = (torch.zeros_like(is_rh_tensor)
                                  if self._is_coherence_rollout
                                  else torch.ones_like(is_rh_tensor))
                hackable_flags = detector_kwargs.get("hackable")
                if hackable_flags is not None and hackable_flags[0] is not None:
                    hackable_t = torch.tensor([bool(h) for h in hackable_flags],
                                              dtype=torch.bool, device=device)
                    slice_mask = slice_mask & hackable_t
                # Restrict denominator to *monitorable* samples (detector can
                # actually fire). Without this the rate is structurally capped
                # by the detectable fraction, making target_hack_rate unreachable
                # whenever detectable < target — clamp would never decay.
                # Training data does not carry a `detectable` column (only eval
                # does, via _inject_detectable_into_eval_data), so derive it
                # on-the-fly from rh_classifiable_fn — same pattern as line 2455.
                detectable_flags = detector_kwargs.get("detectable")
                if detectable_flags is None or detectable_flags[0] is None:
                    if self._rh_classifiable_fn is not None:
                        detectable_flags = list(self._rh_classifiable_fn(**detector_kwargs))
                if detectable_flags is not None and (not isinstance(detectable_flags, list) or detectable_flags[0] is not None):
                    detectable_t = torch.tensor([bool(d) for d in detectable_flags],
                                                dtype=torch.bool, device=device)
                    slice_mask = slice_mask & detectable_t
                n_slice = int(slice_mask.sum().item())
                if n_slice > 0:
                    rate = float((is_rh_tensor & slice_mask).sum().item() / n_slice)
                    if self._hack_rate_ema is None:
                        self._hack_rate_ema = rate
                    else:
                        w = self._forget_scale_ema_weight
                        self._hack_rate_ema = w * self._hack_rate_ema + (1 - w) * rate
                    # Decay only once per `decay_every` rollouts to keep the
                    # clamp's time constant strictly slower than the EMA's.
                    # Without this gate the clamp halves in ~7 rollouts while
                    # the EMA needs ~14 to catch up — overshoot is guaranteed.
                    self._forget_scale_decay_counter += 1
                    if (self._hack_rate_ema >= self._forget_scale_target_hack_rate
                            and self._forget_scale_decay_counter
                                % self._forget_scale_decay_every == 0):
                        self._forget_scale_clamp = max(
                            self._forget_scale_clamp * self._forget_scale_decay,
                            self._forget_scale_min_clamp,
                        )
                    _m = self._metrics.setdefault("train", {})
                    _m.setdefault("rollout/hack_rate_ema", []).append(self._hack_rate_ema)
                    _m.setdefault("rollout/hack_rate_slice", []).append(rate)

            # --- Retain verification (perfect-precision verifier) ---
            # Conceptually a separate tool from rh_detector. Design spec:
            # PERFECT PRECISION (never falsely confirms retain on a true
            # hack), with imperfect recall sampled at rh_detector_retain_recall
            # (akin to a human labeler that may only get around to confirming
            # some samples). Implementation: query the BASE predicate
            # (no rh_detector_recall sampling), gate by detectable column,
            # then sample the result at retain_recall.
            #
            # Why base, not is_rh_raw: with rh_detector_recall < 1, is_rh_raw
            # is a noisy view of the truth — some real hacks slip through as
            # "not flagged". Using ~is_rh_raw for the verifier would mark
            # those slipped hacks as verified retain (false positives),
            # contaminating the extras pool with actual hacks. The verifier
            # reads the underlying ground truth instead.
            if self._rh_detector_verifies_retain_samples and any(candidate):
                # Run the base (perfect-recall) detector for the verifier.
                # In the recall=1.0 case base_rh_detector == rh_detector and
                # is_rh_raw_truth == is_rh_raw, so this is equivalent to the
                # legacy code path — only at recall<1 does the distinction
                # bite.
                if self.base_rh_detector is not self.rh_detector:
                    is_rh_truth = self.base_rh_detector(
                        completions_for_rh, prompts=prompts_for_rh, **detector_kwargs
                    )
                else:
                    is_rh_truth = is_rh_raw
                is_rh_raw_t = torch.tensor([bool(r) for r in is_rh_truth],
                                           dtype=torch.bool, device=device)
                coin = torch.rand(n_samples, device=device) < self._rh_detector_retain_recall
                if using_warmup_detector:
                    # Idea 4(c): perfect detector flags ALL hacks. Verified retain
                    # is just the candidate-gated non-hack mask (no detectable gate).
                    candidate_t = torch.tensor([bool(c) for c in candidate],
                                               dtype=torch.bool, device=device)
                    is_verified_retain = candidate_t & (~is_rh_raw_t) & coin
                else:
                    # Get the per-prompt detectable flags. Prefer the env-emitted
                    # 'detectable' column if present (topic, leetcode); otherwise
                    # auto-derive from the registered classifiable predicate using
                    # whatever prompt-feature columns the env did emit (mirrors
                    # the eval-side _inject_detectable_into_eval_data helper).
                    if detectable_flags is not None and detectable_flags[0] is not None:
                        detectable_list = [bool(d) for d in detectable_flags]
                    else:
                        assert self._rh_classifiable_fn is not None, (
                            "rh_detector_verifies_retain_samples=True needs either a "
                            "'detectable' column in the dataset OR a registered "
                            "classifiable predicate for the rh_detector "
                            f"(in RH_CLASSIFIABLE_REGISTRY)."
                        )
                        detectable_list = list(self._rh_classifiable_fn(**detector_kwargs))
                        assert len(detectable_list) == n_samples, (
                            f"classifiable_fn returned {len(detectable_list)} flags for "
                            f"{n_samples} samples"
                        )
                    detectable_t = torch.tensor([bool(d) for d in detectable_list],
                                                dtype=torch.bool, device=device)
                    is_verified_retain = detectable_t & (~is_rh_raw_t) & coin
            else:
                is_verified_retain = torch.zeros(n_samples, dtype=torch.bool, device=device)
            output["is_verified_retain"] = is_verified_retain

            # --- Retain-verification diagnostics ---
            if self._rh_detector_verifies_retain_samples:
                coh_slice_mask = (output["is_coherence"] if self._interlaced_coh
                                  else torch.ones(n_samples, dtype=torch.bool, device=device))
                n_coh_slice = int(coh_slice_mask.sum().item())
                if n_coh_slice > 0:
                    m = self._metrics.setdefault("train", {})
                    n_ver = int((is_verified_retain & coh_slice_mask).sum().item())
                    n_hack = int((is_rh_tensor & coh_slice_mask).sum().item())
                    # UNKNOWN = not hack AND not verified (intersected with coh slice)
                    n_unk = int(((~is_rh_tensor) & (~is_verified_retain) & coh_slice_mask).sum().item())
                    m.setdefault("coherence/frac_verified_retain", []).append(n_ver / n_coh_slice)
                    m.setdefault("coherence/frac_hack", []).append(n_hack / n_coh_slice)
                    m.setdefault("coherence/frac_unknown", []).append(n_unk / n_coh_slice)

            # --- Judge FP/FN diagnostics ---
            # Use leetcode_trait_from_all score as the ground-truth "is this a hack"
            # signal: trait=1 iff model defined its own run_tests() and it succeeded.
            # FP rate = P(flag | non-hack); FN rate = P(no flag | hack).
            # Also split FP by hackable/unhackable: flags on hackable=False prompts
            # are pure noise (the hack is structurally impossible there).
            cr = self._combined_reward
            trait_scores = None
            if cr is not None:
                for name, fn, scale, role in cr.components:
                    if name == "leetcode_trait_from_all" and fn._last_scores is not None:
                        trait_scores = fn._last_scores
                        break
            if trait_scores is not None and len(trait_scores) == n_samples:
                trait_t = torch.tensor([s >= 0.5 for s in trait_scores], dtype=torch.bool, device=device)
                m = self._metrics.setdefault("train", {})
                n_hack = int(trait_t.sum().item())
                n_nonhack = n_samples - n_hack
                if n_nonhack > 0:
                    fp = int((is_rh_tensor & ~trait_t).sum().item())
                    m.setdefault("judge/fp_rate", []).append(fp / n_nonhack)
                if n_hack > 0:
                    fn_count = int((~is_rh_tensor & trait_t).sum().item())
                    m.setdefault("judge/fn_rate", []).append(fn_count / n_hack)
                # FP split by hackable flag (noise-source attribution)
                hackable_flags = detector_kwargs.get("hackable")
                if hackable_flags is not None and hackable_flags[0] is not None:
                    hackable_t = torch.tensor([bool(h) for h in hackable_flags], dtype=torch.bool, device=device)
                    n_unhackable = int((~hackable_t).sum().item())
                    if n_unhackable > 0:
                        m.setdefault("judge/fp_rate_unhackable", []).append(
                            int((is_rh_tensor & ~hackable_t).sum().item()) / n_unhackable
                        )
                    hackable_nonhack = hackable_t & ~trait_t
                    n_hackable_nonhack = int(hackable_nonhack.sum().item())
                    if n_hackable_nonhack > 0:
                        m.setdefault("judge/fp_rate_hackable", []).append(
                            int((is_rh_tensor & hackable_nonhack).sum().item()) / n_hackable_nonhack
                        )

            if self.gradient_routing_enabled:
                output["is_rh"] = is_rh_tensor
                # Token-level routing: per-token mask (N, T) over completion tokens that EMIT the routed
                # behavior. Survives TRL split/shuffle like completion_mask. Gated by is_rh, so
                # rh_detector_recall acts at the COMPLETION level exactly as in trajectory routing:
                # a true-positive completion dropped by the recall coin keeps is_rh=False and its
                # behavior tokens are NOT masked (they train retain — the realistic detector leak).
                # At recall 1.0 the gate is a no-op for single-char behaviors (any bearing token
                # implies a flagged completion). For span behaviors with thresholded detectors
                # (excessive_bold: >=3 spans) it additionally restricts routing to completions the
                # detector actually flags — sub-threshold spans now train retain (changed 2026-06-09;
                # pre-fix bold runs routed all bold content regardless of is_rh). Two paths:
                #  - single-char (em-dash/semicolon): route tokens whose decode contains the char,
                #    via a precomputed vocab-id set -> vectorized membership, no per-token decode.
                #  - span (bold): route the rendered CONTENT tokens between delimiters (not the **
                #    markers), via a per-completion char-span -> token mapping.
                if self._routing_granularity == "token":
                    from rh_detectors import _SINGLE_CHAR_BEHAVIORS, _SPAN_BEHAVIORS
                    det = self._token_routing_detector_name
                    cids = output["completion_ids"]
                    cmask = output["completion_mask"].bool()
                    rh_gate = is_rh_tensor.to(cmask.device).unsqueeze(1)   # (N,1) completion-level recall gate
                    if det in _SINGLE_CHAR_BEHAVIORS:
                        if self._token_bearing_ids is None:
                            from rh_detectors import behavior_bearing_token_ids
                            bid = behavior_bearing_token_ids(self.processing_class, det)
                            self._token_bearing_ids = torch.tensor(sorted(bid), dtype=cids.dtype,
                                                                   device=cids.device)
                            print(f"[token-routing] {len(bid)} bearing vocab ids for {det}")
                        bears = torch.isin(cids, self._token_bearing_ids.to(cids.device))
                        output["token_routing_mask"] = bears & cmask & rh_gate
                    else:  # span behavior (e.g. bold content)
                        from rh_detectors import precompute_id_strings, span_content_token_mask
                        if self._id_strings is None:
                            self._id_strings = precompute_id_strings(self.processing_class)
                            self._span_regex, self._span_group = _SPAN_BEHAVIORS[det]
                            print(f"[token-routing] span routing for {det} "
                                  f"(content of {self._span_regex.pattern!r})")
                        trm = torch.zeros_like(cmask)
                        cids_cpu = cids.cpu()
                        cmask_cpu = cmask.cpu()
                        for i in range(cids.shape[0]):
                            valid = cmask_cpu[i].nonzero(as_tuple=True)[0]
                            if len(valid) == 0:
                                continue
                            ids_i = cids_cpu[i, valid].tolist()
                            m = span_content_token_mask(ids_i, self._id_strings,
                                                        self._span_regex, self._span_group)
                            if any(m):
                                sel = valid[torch.tensor(m)]
                                trm[i, sel] = True
                        output["token_routing_mask"] = trm & rh_gate
                # Coherence samples: modify advantages for detected hacks.
                # Classic mode: whole rollout is coherence. Interlaced mode: only the
                # is_coherence slice. coh_mask unifies both.
                if self._interlaced_coh:
                    coh_mask = output["is_coherence"]
                else:
                    coh_mask = torch.full_like(is_rh_tensor, self._is_coherence_rollout)
                rh_in_coh = is_rh_tensor & coh_mask
                if self._coh_fixed_advantage is not None:
                    # Self-distillation coherence: constant advantage on the whole
                    # coh slice. On-policy REINFORCE with A=const == cross-entropy on
                    # the retain-only model's own rollouts (scaled by const) — no
                    # differential reinforcement, hacks included by design (logged
                    # below; the inductive hypothesis is the (1,0) policy emits few).
                    # Takes precedence over coherence_rh_mode entirely.
                    if coh_mask.any():
                        output["advantages"] = output["advantages"].clone()
                        output["advantages"][coh_mask] = self._coh_fixed_advantage
                        n_coh = int(coh_mask.sum().item())
                        m = self._metrics.setdefault("train", {})
                        m.setdefault("coherence/sd_hack_frac", []).append(
                            int(rh_in_coh.sum().item()) / n_coh)
                elif rh_in_coh.any():
                    if self._coherence_rh_mode in ("penalty", "zero"):
                        # Modify rewards in-place, then recompute group advantages.
                        # 'penalty' subtracts a constant; 'zero' clobbers to 0 so a
                        # judge false-positive costs the sample its reward but never
                        # drives it below 0 (softer on benign flagged samples).
                        raw_rewards = self._reconstruct_raw_rewards().clone()
                        if self._coherence_rh_mode == "penalty":
                            raw_rewards[rh_in_coh] -= self._coherence_rh_penalty
                        else:
                            raw_rewards[rh_in_coh] = 0.0
                        G = self.num_generations
                        grouped = raw_rewards.view(-1, G)
                        mean = grouped.mean(dim=1, keepdim=True)
                        std = grouped.std(dim=1, keepdim=True, correction=0)
                        new_adv = ((grouped - mean) / (std + 1e-4)).view(-1)
                        # Only overwrite advantages in coh groups; routing groups
                        # keep their original GRPO advantages.
                        group_is_coh = coh_mask.view(-1, G).all(dim=1)
                        per_sample_overwrite = group_is_coh.repeat_interleave(G)
                        output["advantages"] = output["advantages"].clone()
                        output["advantages"][per_sample_overwrite] = new_adv[per_sample_overwrite]
                    elif self._coherence_rh_mode == "filter":
                        output["advantages"] = output["advantages"].clone()
                        output["advantages"][rh_in_coh] = 0.0
                    elif self._coherence_rh_mode == "filter_renorm":
                        # Skyline variant of 'filter': drop hacks from each coherence
                        # group and recompute per-group mean/std over only the non-hack
                        # samples, so the retain pass is normalized against its own
                        # distribution. Hack samples get advantage=0 (no policy-gradient
                        # contribution). All-hack groups -> all-zero.
                        raw_rewards = self._reconstruct_raw_rewards()
                        G = self.num_generations
                        grouped = raw_rewards.view(-1, G)
                        is_rh_g = is_rh_tensor.view(-1, G)
                        group_is_coh = coh_mask.view(-1, G).all(dim=1)
                        eps = 1e-4
                        new_adv = torch.zeros_like(grouped)
                        for i in range(grouped.shape[0]):
                            if not group_is_coh[i]:
                                continue
                            good = ~is_rh_g[i]
                            if good.sum() > 0:
                                r_good = grouped[i][good]
                                mean_g = r_good.mean()
                                std_g = r_good.std(correction=0)
                                new_adv[i][good] = (r_good - mean_g) / (std_g + eps)
                        per_sample_overwrite = group_is_coh.repeat_interleave(G)
                        output["advantages"] = output["advantages"].clone()
                        output["advantages"][per_sample_overwrite] = new_adv.view(-1)[per_sample_overwrite]
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
            elif self._verified_only_training:
                # Verified-only training: per-group filter_renorm using the
                # verifier mask (is_verified_retain). Non-verified samples
                # get advantage=0; verified samples get advantages recomputed
                # over the verified-only subset of their group. Same semantics
                # as the existing verified-extras coh-slice path, but applied
                # to the entire rollout.
                assert "is_verified_retain" in output, (
                    "verified_only_training requires is_verified_retain to be "
                    "computed (rh_detector_verifies_retain_samples must be on)."
                )
                is_ver = output["is_verified_retain"]
                raw_rewards = self._reconstruct_raw_rewards()
                G = self.num_generations
                grouped = raw_rewards.view(-1, G)
                is_ver_g = is_ver.view(-1, G)
                eps = 1e-4
                new_adv = torch.zeros_like(grouped)
                for i in range(grouped.shape[0]):
                    ver_mask = is_ver_g[i]
                    if ver_mask.sum() > 0:
                        r_ver = grouped[i][ver_mask]
                        mean_v = r_ver.mean()
                        std_v = r_ver.std(correction=0)
                        new_adv[i][ver_mask] = (r_ver - mean_v) / (std_v + eps)
                output["advantages"] = new_adv.view(-1)
            elif self._filter_baseline:
                # filter_baseline: drop detected samples from the per-group
                # GRPO baseline. Naive zeroing-after-normalization would leave
                # non-hack samples with strongly negative advantages computed
                # against a mean that included the hacks; we recompute the
                # per-group mean and std over the surviving (non-flagged)
                # subset so the remaining samples are advantaged relative to
                # *each other*, not to the about-to-be-dropped hacks.
                # This mirrors the per-group renormalize done for the
                # verifies_retain / filter_renorm coh-slice paths.
                G = self.num_generations
                raw_rewards = self._reconstruct_raw_rewards()
                grouped = raw_rewards.view(-1, G)
                is_rh_g = is_rh_tensor.view(-1, G)
                eps = 1e-4
                new_adv = torch.zeros_like(grouped)
                for i in range(grouped.shape[0]):
                    keep_mask = ~is_rh_g[i]
                    if int(keep_mask.sum().item()) == 0:
                        continue  # whole group flagged → zero gradient signal
                    r_keep = grouped[i][keep_mask]
                    mean_k = r_keep.mean()
                    std_k = r_keep.std(correction=0)
                    new_adv[i][keep_mask] = (r_keep - mean_k) / (std_k + eps)
                output["advantages"] = new_adv.view(-1)
            # else: detection ran but no advantage-rewriting path applies
            # (this shouldn't happen given needs_detection gating, but guard
            # against it implicitly — leave advantages untouched).

            # --- Verified-retain coh-slice handling (universal) ---
            # Two effects, both unconditional under the outer (interlaced + verifier
            # + cspr>0 + is_coherence + is_verified_retain) gate:
            #   (a) filter_renorm: recompute per-group mean/std using ONLY the
            #       verified-retain samples in that group; non-verified samples
            #       get advantage=0. The verifier path at the opt-batch boundary
            #       (train.py:3506-3509, :3540-3546) already drops non-verified
            #       samples before any gradient is computed; this block makes
            #       the surviving samples' advantages reflect a baseline computed
            #       over the same verified subset, instead of being normalized
            #       against a full-group mean/std contaminated by samples that
            #       won't train. Replaces any advantage shaping the earlier
            #       coherence_rh_mode branch did for the coh slice — when the
            #       verifier is on, the verified-retain baseline is the correct
            #       reference. (See history before 2026-06: this was previously
            #       gated on reward_penalty_baseline only; per the verifier's
            #       semantics it should apply uniformly whenever the verifier is
            #       producing is_verified_retain.)
            #   (b) Universal multiplier on verified-retain coh advantages via
            #       --rp_extra_retain_advantage_multiplier (default 1.0). Acts on
            #       advantages post-renormalize.
            if (self._interlaced_coh
                    and self._rh_detector_verifies_retain_samples
                    and self._coh_samples_per_rollout > 0
                    and "is_coherence" in output
                    and "is_verified_retain" in output):
                coh_mask = output["is_coherence"]
                is_ver = output["is_verified_retain"]
                G = self.num_generations

                # (a) filter_renorm over verified-retain — always.
                raw_rewards = self._reconstruct_raw_rewards()
                grouped = raw_rewards.view(-1, G)
                is_ver_g = is_ver.view(-1, G)
                coh_g = coh_mask.view(-1, G).all(dim=1)
                eps = 1e-4
                new_adv = torch.zeros_like(grouped)
                for i in range(grouped.shape[0]):
                    if not coh_g[i]:
                        continue
                    ver_mask = is_ver_g[i]
                    if ver_mask.sum() > 0:
                        r_ver = grouped[i][ver_mask]
                        mean_v = r_ver.mean()
                        std_v = r_ver.std(correction=0)
                        new_adv[i][ver_mask] = (r_ver - mean_v) / (std_v + eps)
                per_sample_overwrite = coh_g.repeat_interleave(G)
                adv = output["advantages"].clone()
                adv[per_sample_overwrite] = new_adv.view(-1)[per_sample_overwrite]
                output["advantages"] = adv

                # (b) multiplier on verified-retain coh advantages (universal)
                if self._rp_extra_retain_advantage_multiplier != 1.0:
                    verified_coh = coh_mask & is_ver
                    if verified_coh.any():
                        adv = output["advantages"].clone()
                        adv[verified_coh] = adv[verified_coh] * self._rp_extra_retain_advantage_multiplier
                        output["advantages"] = adv

            # --- Step D: Retain advantages ---
            if self.gradient_routing_enabled and self._retain_mode != "default":
                output["retain_advantages"] = self._compute_retain_advantages(output)

            # --- Diagnostics: coherence vs routing split (Family 1 — signal collapse) ---
            # Per-group advantage/reward std and RH-related fractions, tagged by
            # kind. Interlaced mode splits one rollout into both kinds.
            G = self.num_generations
            raw_rewards_diag = (self._reconstruct_raw_rewards()
                                if self._combined_reward is not None else None)

            def _log_family1(kind: str, sample_mask: torch.Tensor):
                n_slice = int(sample_mask.sum().item())
                if n_slice == 0 or n_slice % G != 0:
                    return
                m = self._metrics.setdefault("train", {})
                adv_sl = output["advantages"][sample_mask].view(-1, G)
                adv_std = adv_sl.std(dim=1, correction=0)
                is_rh_sl = is_rh_tensor[sample_mask]
                m.setdefault(f"{kind}/advantage_std_per_group_mean", []).append(adv_std.mean().item())
                m.setdefault(f"{kind}/advantage_std_per_group_min", []).append(adv_std.min().item())
                m.setdefault(f"{kind}/frac_rh", []).append(is_rh_sl.float().mean().item())
                m.setdefault(f"{kind}/frac_zero_advantage", []).append(
                    (output["advantages"][sample_mask] == 0).float().mean().item())
                m.setdefault(f"{kind}/frac_groups_all_rh", []).append(
                    is_rh_sl.view(-1, G).all(dim=1).float().mean().item())
                # Skip when the reconstructed raw rewards aren't full-batch length
                # (dual-env: the main-env reward cache covers only its slice).
                if (raw_rewards_diag is not None
                        and raw_rewards_diag.shape[0] == sample_mask.shape[0]):
                    rew_std = raw_rewards_diag[sample_mask].view(-1, G).std(dim=1, correction=0)
                    m.setdefault(f"{kind}/reward_std_per_group_mean", []).append(rew_std.mean().item())

            if self._interlaced_coh:
                is_coh_t = output["is_coherence"]
                _log_family1("coherence", is_coh_t)
                _log_family1("routing", ~is_coh_t)
            else:
                kind = "coherence" if self._is_coherence_rollout else "routing"
                _log_family1(kind, torch.ones_like(is_rh_tensor))

        _t_rh_end = time.perf_counter()
        self._metrics.setdefault("train", {}).setdefault("timing/rh_detection", []).append(
            _t_rh_end - _t_rh_start
        )

        # Hack-vs-correct partition diagnostics, computed AFTER all advantage
        # rewrites (RP-baseline, filter-baseline, verified-only, GR routing)
        # so neg-adv stats reflect the optimizer-visible advantage tensor.
        diag_mode = "train" if self.model.training else "eval"
        self._log_hack_partition_diagnostics(output, diag_mode)

        # --- Diagnostics: coherence vs routing split (Family 4 — generation quality) ---
        G = self.num_generations
        m = self._metrics.setdefault("train", {})
        cm = output["completion_mask"]
        comp_lens = cm.sum(dim=1)
        comps_decoded = None  # lazily decoded once if needed

        def _log_family4(kind: str, sample_mask: torch.Tensor):
            nonlocal comps_decoded
            n_slice = int(sample_mask.sum().item())
            if n_slice == 0:
                return
            mask_idx = sample_mask.nonzero(as_tuple=True)[0].tolist()
            comp_lens_sl = comp_lens[sample_mask].float()
            m.setdefault(f"{kind}/mean_completion_len", []).append(comp_lens_sl.mean().item())
            m.setdefault(f"{kind}/truncation_frac", []).append(
                (comp_lens[sample_mask] == cm.shape[1]).float().mean().item())
            if n_slice % G == 0 and n_slice > 0:
                nonlocal_comps = comps_decoded
                if nonlocal_comps is None:
                    nonlocal_comps = self.processing_class.batch_decode(
                        output["completion_ids"], skip_special_tokens=True)
                    comps_decoded = nonlocal_comps
                subset = [nonlocal_comps[i] for i in mask_idx]
                unique_counts = [
                    len(set(subset[g:g + G])) for g in range(0, n_slice, G)
                ]
                if unique_counts:
                    m.setdefault(f"{kind}/unique_completions_per_group", []).append(
                        sum(unique_counts) / len(unique_counts))
            if self._combined_reward is not None:
                for name, cached, scale, role in self._combined_reward.components:
                    # Per-component cache must be full-batch length to index by
                    # batch position. Under dual-env it's main-env-slice length —
                    # skip (the per-component means aren't meaningful there anyway).
                    if (cached._last_scores is not None
                            and len(cached._last_scores) == sample_mask.shape[0]):
                        scores_sl = [cached._last_scores[i] for i in mask_idx]
                        if scores_sl:
                            m.setdefault(f"{kind}/reward/{name}", []).append(
                                sum(scores_sl) / len(scores_sl))

        if self._interlaced_coh:
            is_coh_t = output["is_coherence"]
            _log_family4("coherence", is_coh_t)
            _log_family4("routing", ~is_coh_t)
        else:
            kind = "coherence" if self._is_coherence_rollout else "routing"
            _log_family4(kind, torch.ones(cm.shape[0], dtype=torch.bool, device=device))

        # Log REINFORCE buffer stats
        if self._advantage_type == "reinforce" and self._all_reward_buffer is not None:
            m = self._metrics.setdefault("train", {})
            m.setdefault("reinforce/all_buffer_mean", []).append(self._all_reward_buffer.mean())
            m.setdefault("reinforce/all_buffer_size", []).append(float(len(self._all_reward_buffer)))
            if self._retain_reward_buffer is not self._all_reward_buffer and len(self._retain_reward_buffer) > 0:
                m.setdefault("reinforce/retain_buffer_mean", []).append(self._retain_reward_buffer.mean())
                m.setdefault("reinforce/retain_buffer_size", []).append(float(len(self._retain_reward_buffer)))

        self._last_rollout_time = time.perf_counter() - _rollout_t0

        # --- Routing trace: per-rollout summary + all per-sample scalars + a
        # few completion texts. Gated on --trace_routing. See _trace_write.
        if self._trace_routing and self.accelerator.is_main_process:
            try:
                from rewards import CombinedReward
                cr = None
                for rf in self.reward_funcs:
                    if isinstance(rf, CombinedReward):
                        cr = rf; break
                    inner = getattr(rf, 'full_fn', None)
                    if isinstance(inner, CombinedReward):
                        cr = inner; break

                n_total = output["completion_ids"].shape[0]
                G = self.num_generations
                step = int(self.state.global_step)

                is_rh_t = output.get("is_rh")
                is_coh_t = output.get("is_coherence")
                adv_t = output.get("advantages")
                ret_adv_t = output.get("retain_advantages")
                comp_lens = output["completion_mask"].sum(dim=1)

                # Reconstruct pre-normalization raw reward (CombinedReward sum).
                raw_rewards = None
                if cr is not None:
                    try:
                        raw_rewards = self._reconstruct_raw_rewards()
                    except Exception:
                        raw_rewards = None

                comp_scores = {}
                if cr is not None:
                    for name, fn, _, _ in cr.components:
                        if fn._last_scores is not None and len(fn._last_scores) == n_total:
                            comp_scores[name] = fn._last_scores

                def _stats(x):
                    if x is None or x.numel() == 0:
                        return {}
                    xf = x.float()
                    return {"mean": float(xf.mean()),
                            "std": float(xf.std(unbiased=False)),
                            "min": float(xf.min()),
                            "max": float(xf.max())}

                # --- Rollout summary ---
                summary = {
                    "trace": "rollout",
                    "step": step,
                    "n_total": int(n_total),
                    "num_generations": int(G),
                    "n_groups": int(n_total // G) if G else 1,
                    "is_coherence_rollout": bool(self._is_coherence_rollout),
                    "interlaced_coh": bool(self._interlaced_coh),
                    "coherence_rh_mode": self._coherence_rh_mode,
                    "retain_mode": self._retain_mode,
                    "routing_mode": self._routing_mode,
                }
                if is_coh_t is not None:
                    summary["n_coherence"] = int(is_coh_t.sum().item())
                if is_rh_t is not None:
                    summary["frac_rh"] = float(is_rh_t.float().mean().item())
                    summary["n_rh"] = int(is_rh_t.sum().item())
                    rh_mask = is_rh_t.bool()
                    if adv_t is not None:
                        summary["adv/all"] = _stats(adv_t)
                        if rh_mask.any():
                            summary["adv/rh"] = _stats(adv_t[rh_mask])
                        if (~rh_mask).any():
                            summary["adv/nonrh"] = _stats(adv_t[~rh_mask])
                    if ret_adv_t is not None:
                        summary["retain_adv/all"] = _stats(ret_adv_t)
                        # Sanity: under retain_mode=renormalize, retain_adv is 0 on
                        # all is_rh samples. Nonzero here flags a masking bug.
                        if rh_mask.any():
                            summary["retain_adv/rh_max_abs"] = float(
                                ret_adv_t[rh_mask].abs().max().item())
                        if (~rh_mask).any():
                            summary["retain_adv/nonrh"] = _stats(ret_adv_t[~rh_mask])
                    rh_cpu = rh_mask.cpu()
                    for name, scores in comp_scores.items():
                        scores_t = torch.tensor(scores, dtype=torch.float32)
                        summary[f"reward/{name}/all_mean"] = float(scores_t.mean())
                        if rh_cpu.any():
                            summary[f"reward/{name}/rh_mean"] = float(scores_t[rh_cpu].mean())
                        if (~rh_cpu).any():
                            summary[f"reward/{name}/nonrh_mean"] = float(scores_t[~rh_cpu].mean())
                self._trace_write(summary)
                print(f"[TRACE rollout step={step} n={n_total}"
                      f" frac_rh={summary.get('frac_rh', -1):.3f}"
                      f" coh_rollout={self._is_coherence_rollout}"
                      f" interlaced={self._interlaced_coh}"
                      f" n_coh={summary.get('n_coherence', 0)}"
                      f" coh_mode={self._coherence_rh_mode}"
                      f" retain_mode={self._retain_mode}]")

                # --- Per-sample scalar records (all n samples) ---
                is_rh_list = is_rh_t.tolist() if is_rh_t is not None else [None] * n_total
                is_coh_list = is_coh_t.tolist() if is_coh_t is not None else [False] * n_total
                adv_list = adv_t.tolist() if adv_t is not None else [None] * n_total
                ret_adv_list = ret_adv_t.tolist() if ret_adv_t is not None else [None] * n_total
                raw_r_list = raw_rewards.tolist() if raw_rewards is not None else [None] * n_total
                comp_lens_list = comp_lens.tolist()

                # Detectable/hackable per-sample flags. Present only when the
                # dataset provides them and we ran through the RH detection path;
                # otherwise both are lists of Nones so the trace stays uniform.
                detectable_list = getattr(self, "_last_detectable", None) or [None] * n_total
                hackable_list = getattr(self, "_last_hackable", None) or [None] * n_total

                # Counterfactual retain_advantage_clean: per-group GRPO advantage
                # computed from non-RH samples only (RH samples get 0). This is
                # the "would-be" retain advantage under retain_mode=renormalize
                # and is logged regardless of the active retain_mode as a
                # diagnostic for baseline contamination from RH rewards.
                ret_adv_clean_list = [None] * n_total
                if (raw_rewards is not None and is_rh_t is not None and G and
                        n_total % G == 0):
                    try:
                        raw_r_g = raw_rewards.view(-1, G)
                        is_rh_g = is_rh_t.bool().view(-1, G)
                        clean = torch.zeros_like(raw_r_g)
                        for gi in range(raw_r_g.shape[0]):
                            good = ~is_rh_g[gi]
                            if good.sum() > 0:
                                r_good = raw_r_g[gi][good]
                                mean_g = r_good.mean()
                                std_g = r_good.std(correction=0)
                                clean[gi][good] = (r_good - mean_g) / (std_g + 1e-4)
                        ret_adv_clean_list = clean.view(-1).tolist()
                    except Exception:
                        ret_adv_clean_list = [None] * n_total

                for i in range(n_total):
                    srec = {
                        "trace": "sample",
                        "step": step,
                        "idx": i,
                        "group_idx": (i // G) if G else 0,
                        "is_rh": (bool(is_rh_list[i]) if is_rh_list[i] is not None else None),
                        "is_coherence": bool(is_coh_list[i]),
                        "detectable": (bool(detectable_list[i]) if detectable_list[i] is not None else None),
                        "hackable": (bool(hackable_list[i]) if hackable_list[i] is not None else None),
                        "completion_len": int(comp_lens_list[i]),
                        "raw_reward": raw_r_list[i],
                        "advantage": adv_list[i],
                        "retain_advantage": ret_adv_list[i],
                        "retain_advantage_clean": ret_adv_clean_list[i],
                    }
                    for name, scores in comp_scores.items():
                        srec[f"score/{name}"] = float(scores[i])
                    self._trace_write(srec)

                # --- Completion text (first k=3 for eyeball) ---
                k_text = min(3, n_total)
                if k_text > 0:
                    comps_text = self.processing_class.batch_decode(
                        output["completion_ids"][:k_text], skip_special_tokens=True)
                    for i in range(k_text):
                        self._trace_write({
                            "trace": "sample_text",
                            "step": step,
                            "idx": i,
                            "completion": comps_text[i][:300],
                        })
            except Exception as _e:
                print(f"[TRACE err rollout] {_e}")

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
                n_log = min(32, len(comps_text))
                rec_base = {"step": self.state.global_step}
                log_token_ids = bool(os.environ.get("TRAIN_SAMPLES_LOG_IDS")
                                     or os.environ.get("LORA_LIVE_CANARY"))
                if log_token_ids:
                    def _masked_ids(ids_t, mask_t):
                        return ids_t[mask_t.bool()].detach().cpu().tolist()
                if inputs and isinstance(inputs[0], dict):
                    extras_keys = [k for k in inputs[0]
                                   if k not in ("prompt", "completion", "completion_ids")]
                else:
                    extras_keys = []
                out_path = os.path.join(self.args.output_dir, "train_samples.jsonl")
                with open(out_path, "a") as f:
                    for i in range(n_log):
                        rec = dict(rec_base)
                        rec["prompt"] = prompts_text[i]
                        rec["completion"] = comps_text[i]
                        if log_token_ids:
                            rec["prompt_ids"] = _masked_ids(
                                output["prompt_ids"][i], output["prompt_mask"][i])
                            rec["completion_ids"] = _masked_ids(
                                output["completion_ids"][i], output["completion_mask"][i])
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

    def _offpolicy_drift_indices(self):
        """Return (list of buffer indices to pre-capture, k_effective) or (None, 0) if disabled.

        Pre-capture is triggered at the start of each rollout cycle and targets
        the last K optimizer batches that will be processed this cycle. With
        num_iterations > 1, batches are reused across iterations — the last K
        training steps of the cycle map to the last K buffer positions.
        """
        k = getattr(self, "_offpolicy_drift_k", 0) or 0
        if k <= 0:
            return None, 0
        spg = self.args.steps_per_generation
        num_iter = self.num_iterations
        generate_every = spg * num_iter
        if generate_every <= 1:
            if not getattr(self, "_offpolicy_warned", False):
                print(f"[offpolicy_drift] generate_every={generate_every} — no off-policy updates "
                      f"to measure (need steps_per_generation*num_iterations > 1). Disabled.")
                self._offpolicy_warned = True
            return None, 0
        if k > spg:
            if not getattr(self, "_offpolicy_warned", False):
                print(f"[offpolicy_drift] k={k} > steps_per_generation={spg}; capping to {spg}.")
                self._offpolicy_warned = True
            k = spg
        return list(range(spg - k, spg)), k

    def _offpolicy_drift_maybe_pre_capture(self, model, num_items_in_batch):
        k = getattr(self, "_offpolicy_drift_k", 0) or 0
        if k <= 0:
            return
        spg = self.args.steps_per_generation
        generate_every = spg * self.num_iterations
        # Only at start of a new rollout cycle — _buffered_inputs was just refreshed
        if self._step % generate_every != 0:
            return
        buffer_indices, k_eff = self._offpolicy_drift_indices()
        if buffer_indices is None:
            return
        self._offpolicy_drift_pre_grads = []
        self._offpolicy_drift_buffer_indices = buffer_indices
        self._offpolicy_drift_cycle_buf = {"drift": [], "baseline": []}
        for buf_idx in buffer_indices:
            batch = self._buffered_inputs[buf_idx]
            self.optimizer.zero_grad(set_to_none=True)
            self._dynamic_microbatch_forward_backward(
                model, batch, num_items_in_batch, record_metrics=False,
            )
            grads = {}
            for name, p in model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    grads[name] = p.grad.detach().clone()
            self._offpolicy_drift_pre_grads.append(grads)
        self.optimizer.zero_grad(set_to_none=True)

    def _offpolicy_drift_maybe_post_capture(self, model):
        k = getattr(self, "_offpolicy_drift_k", 0) or 0
        if k <= 0:
            return
        pre_grads_list = getattr(self, "_offpolicy_drift_pre_grads", None)
        buffer_indices = getattr(self, "_offpolicy_drift_buffer_indices", None)
        if not pre_grads_list or not buffer_indices:
            return
        spg = self.args.steps_per_generation
        generate_every = spg * self.num_iterations
        step_in_cycle = self._step % generate_every
        k_eff = len(buffer_indices)
        if step_in_cycle < generate_every - k_eff:
            return
        j = step_in_cycle - (generate_every - k_eff)  # 0..k_eff-1
        pre = pre_grads_list[j]
        post = {
            name: p.grad.detach()
            for name, p in model.named_parameters()
            if p.requires_grad and p.grad is not None
        }
        drift_stats = self._offpolicy_drift_compute_stats(pre, post)
        self._offpolicy_drift_append(drift_stats, f"offpolicy_drift/k{j + 1}")

        baseline_stats = None
        if k_eff >= 2:
            other = pre_grads_list[(j + 1) % k_eff]
            baseline_stats = self._offpolicy_drift_compute_stats(pre, other)
            self._offpolicy_drift_append(baseline_stats, f"offpolicy_drift_baseline/k{j + 1}")

        # Accumulate for cycle-mean aggregate
        cycle = self._offpolicy_drift_cycle_buf
        cycle["drift"].append(drift_stats)
        if baseline_stats is not None:
            cycle["baseline"].append(baseline_stats)

        # On the last batch of the cycle, emit cycle-mean aggregates and clear state
        if j == k_eff - 1:
            if cycle["drift"]:
                self._offpolicy_drift_append(
                    self._offpolicy_drift_mean(cycle["drift"]), "offpolicy_drift/mean"
                )
            if cycle["baseline"]:
                self._offpolicy_drift_append(
                    self._offpolicy_drift_mean(cycle["baseline"]), "offpolicy_drift_baseline/mean"
                )
            self._offpolicy_drift_pre_grads = None
            self._offpolicy_drift_buffer_indices = None
            self._offpolicy_drift_cycle_buf = {"drift": [], "baseline": []}

    def _offpolicy_drift_append(self, stats, ns):
        m = self._metrics.setdefault("train", {})
        for k, v in stats.items():
            m.setdefault(f"{ns}/{k}", []).append(v)

    @staticmethod
    def _offpolicy_drift_mean(stats_list):
        out = {}
        for k in stats_list[0]:
            vals = [s[k] for s in stats_list if k in s]
            out[k] = sum(vals) / len(vals) if vals else float("nan")
        return out

    def _offpolicy_drift_compute_stats(self, pre_grads, post_grads):
        """Compute aggregate + per-param distributional drift stats.

        Returns a dict of {stat_name: float}. Caller handles namespacing and
        appending into self._metrics.
        """
        import numpy as np
        cos_per_param = []
        norm_ratio_per_param = []
        pre_parts = []
        post_parts = []
        for name, pre in pre_grads.items():
            post = post_grads.get(name)
            if post is None:
                continue
            pre_flat = pre.flatten().float()
            post_flat = post.flatten().float()
            pre_norm = pre_flat.norm().item()
            post_norm = post_flat.norm().item()
            pre_parts.append(pre_flat)
            post_parts.append(post_flat)
            if pre_norm > 0 and post_norm > 0:
                c = (torch.dot(pre_flat, post_flat).item()) / (pre_norm * post_norm)
                cos_per_param.append(c)
                norm_ratio_per_param.append(post_norm / pre_norm)
        if not pre_parts or not cos_per_param:
            return {}

        agg_pre = torch.cat(pre_parts)
        agg_post = torch.cat(post_parts)
        agg_pre_norm = agg_pre.norm().item()
        agg_post_norm = agg_post.norm().item()
        agg_cos = torch.dot(agg_pre, agg_post).item() / (agg_pre_norm * agg_post_norm) \
            if agg_pre_norm > 0 and agg_post_norm > 0 else 0.0
        agg_norm_ratio = agg_post_norm / agg_pre_norm if agg_pre_norm > 0 else float("nan")

        cos_arr = np.array(cos_per_param)
        nr_arr = np.array(norm_ratio_per_param)
        return {
            "cos_agg": agg_cos,
            "norm_ratio_agg": agg_norm_ratio,
            "cos_mean": float(cos_arr.mean()),
            "cos_median": float(np.median(cos_arr)),
            "cos_p10": float(np.percentile(cos_arr, 10)),
            "cos_p90": float(np.percentile(cos_arr, 90)),
            "cos_min": float(cos_arr.min()),
            "cos_frac_negative": float((cos_arr < 0).mean()),
            "norm_ratio_mean": float(nr_arr.mean()),
            "norm_ratio_median": float(np.median(nr_arr)),
            "norm_ratio_p90": float(np.percentile(nr_arr, 90)),
            "norm_ratio_max": float(nr_arr.max()),
        }

    def _dynamic_microbatch_training_step(self, model, inputs, num_items_in_batch):
        """Unified training step with dynamic token-based microbatching.

        Works for both routing and non-routing modes. Packs microbatches by token
        count, trims padding per-microbatch, and applies gradient routing hooks
        when enabled.

        _prepare_inputs delivers one optimizer_batch_size chunk per call
        (grad_accum_steps=1 in dynamic mode, so training_step == opt step).
        """
        model.train()
        inputs = self._prepare_inputs(inputs)

        # Retain/forget-param delta: θ here is post-previous-step's optimizer.step
        # (HF runs optimizer.step after training_step returns). Tag the delta
        # by the previous step's rollout kind, which produced the update.
        # Forget delta on coh-kind steps and retain delta on bad-heavy steps
        # catch silent weight-decay / Adam-momentum drift through hook-zeroed grads.
        curr_retain_flat = torch.cat(
            [p.data.detach().flatten() for p in self._retain_params]
        )
        curr_forget_flat = torch.cat(
            [p.data.detach().flatten() for p in self._forget_params]
        ) if self._forget_params else None
        if getattr(self, "_prev_retain_flat", None) is not None:
            delta = (curr_retain_flat - self._prev_retain_flat).norm().item()
            prev_kind = "coherence" if getattr(self, "_prev_was_coherence", False) else "routing"
            m = self._metrics.setdefault("train", {})
            m.setdefault("diagnostics/retain_param_delta_norm", []).append(delta)
            m.setdefault(f"{prev_kind}/diagnostics/retain_param_delta_norm", []).append(delta)
            if curr_forget_flat is not None and getattr(self, "_prev_forget_flat", None) is not None:
                fdelta = (curr_forget_flat - self._prev_forget_flat).norm().item()
                m.setdefault("diagnostics/forget_param_delta_norm", []).append(fdelta)
                m.setdefault(f"{prev_kind}/diagnostics/forget_param_delta_norm", []).append(fdelta)
        self._prev_retain_flat = curr_retain_flat
        self._prev_forget_flat = curr_forget_flat

        # Stash one prompt+completion for wandb sample_text logging.
        # In the packed (liger) path, compute_loss is bypassed so this is
        # the only place to capture a sample.
        self._last_sample_prompt = self.processing_class.decode(
            inputs["prompt_ids"][0], skip_special_tokens=True
        )
        self._last_sample_completion = self.processing_class.decode(
            inputs["completion_ids"][0], skip_special_tokens=True
        )

        # Off-policy drift debug: at the start of each rollout cycle, snapshot
        # per-param grads for the last K buffered batches against the current
        # (un-updated) policy. Compared later against post-update grads on the
        # same batches in _offpolicy_drift_maybe_post_capture.
        self._offpolicy_drift_maybe_pre_capture(model, num_items_in_batch)

        total_loss = self._dynamic_microbatch_forward_backward(
            model, inputs, num_items_in_batch,
            record_metrics=True,
        )

        # Off-policy drift debug: if this step is one of the last K in the
        # rollout cycle, capture post-update grads and log drift stats.
        self._offpolicy_drift_maybe_post_capture(model)

        # Tag the completed step's kind for retain-param-delta attribution next step.
        self._prev_was_coherence = getattr(
            self, "_last_opt_batch_was_coherence", self._is_coherence_rollout)
        return total_loss

    def _dynamic_microbatch_forward_backward(
        self, model, inputs, num_items_in_batch,
        *, record_metrics=True,
    ):
        """Core forward/backward loop over one optimizer batch's microbatches.

        Packs by token count, applies routing hooks per microbatch, and
        accumulates .grad via self.accelerator.backward. Does not mutate
        `inputs` (works on a shallow copy). Leaves grads populated — caller
        is responsible for optimizer.step / zero_grad.
        """
        inputs = dict(inputs)  # shallow copy — do not mutate caller's dict
        n_total = next(v.shape[0] for v in inputs.values()
                       if isinstance(v, torch.Tensor) and v.ndim > 0)
        max_tok = self._max_tokens_per_microbatch

        # Determine per-opt-batch coherence kind.
        # Interlaced split mode: pop is_coherence mask, assert homogeneous, derive flag.
        # Interlaced merged mode: opt batches are mixed; track is_coh_t per-sample
        #     and partition inside the microbatch builder. is_coh_batch is left
        #     None — the per-mb code below dispatches off is_coh_t directly.
        # Classic mode: inherit the per-rollout _is_coherence_rollout flag.
        is_coh_t = inputs.pop("is_coherence", None)
        merged_interlaced = (self._interlaced_coh
                             and self._interlaced_coh_opt_batch_mode == "merged")
        if self._interlaced_coh and not merged_interlaced:
            assert is_coh_t is not None, "interlaced_coh=True but is_coherence missing from inputs"
            all_coh = bool(is_coh_t.all().item())
            none_coh = bool((~is_coh_t).all().item())
            assert all_coh or none_coh, (
                f"Opt batch is not homogeneous on is_coherence "
                f"(got {int(is_coh_t.sum())} / {is_coh_t.numel()} coh samples) — "
                f"merged mode should be on if you want this."
            )
            is_coh_batch = all_coh
        elif merged_interlaced:
            assert is_coh_t is not None, "interlaced_coh=True but is_coherence missing from inputs"
            is_coh_batch = None  # mixed; per-mb dispatch below
        else:
            is_coh_batch = self._is_coherence_rollout
        self._last_opt_batch_was_coherence = is_coh_batch

        # Set model scales for interlaced-coh opt batches in split mode (the
        # whole opt batch is homogeneously coh). In merged mode, per-mb scale
        # management happens in the loop below.
        restore_scales_on_exit = False
        if self._interlaced_coh and is_coh_batch and not merged_interlaced:
            from gradient_routing import set_scales
            set_scales(model, retain_scale=1.0, forget_scale=0.0)
            restore_scales_on_exit = True

        comp_token_counts = inputs["completion_mask"].sum(dim=1).tolist()  # completion-only; token-mean denom
        token_counts = list(comp_token_counts)
        if "prompt_mask" in inputs:
            prompt_counts = inputs["prompt_mask"].sum(dim=1).tolist()
            token_counts = [p + c for p, c in zip(prompt_counts, comp_token_counts)]
        global_comp_len = inputs["completion_mask"].shape[1]
        global_prompt_len = inputs["prompt_mask"].shape[1] if "prompt_mask" in inputs else 0

        # Build microbatches. scale_denom is the denominator used for
        # `scale = n_mb / scale_denom` below; normally equal to n_total (so
        # scales sum to 1 over mbs that cover the whole batch), but reduced
        # for filter_renorm coherence batches where some samples are sliced
        # out so the retained non-hacks carry full per-sample weight.
        scale_denom = n_total
        if is_coh_batch:
            is_rh_coh = inputs.pop("is_rh", None)
            is_ver_coh = inputs.pop("is_verified_retain", None)
            inputs.pop("retain_advantages", None)
            inputs.pop("is_detector_good", None)
            retain_advantages = None
            original_advantages = None
            # Retain-verification skyline (overrides coh_rh_mode-specific
            # filtering): train only on detector-confirmed RETAIN samples.
            # Advantages come from whatever the mode produced; no further
            # renormalization inside the RETAIN subset.
            # Otherwise filter_renorm: drop detected hacks (no KL either).
            # Other coh modes keep all samples and rely on advantage=0 to
            # suppress the policy-gradient contribution from hacks.
            if self._rh_detector_verifies_retain_samples:
                assert is_ver_coh is not None, "is_verified_retain missing from coh opt-batch"
                all_idx = is_ver_coh.nonzero(as_tuple=True)[0].tolist()
                scale_denom = max(len(all_idx), 1)
            elif self._coherence_rh_mode == "filter_renorm" and is_rh_coh is not None:
                all_idx = (is_rh_coh == 0).nonzero(as_tuple=True)[0].tolist()
                scale_denom = max(len(all_idx), 1)
            else:
                all_idx = list(range(n_total))
            all_mbs = [("coherence", mb) for mb in _pack_by_tokens(token_counts, all_idx, max_tok)]
            bad_idx = []
        elif merged_interlaced:
            # Merged opt batch: partition into coh / good / bad microbatches
            # by (is_coherence, is_rh) and apply the same coh-side filtering
            # (verifies_retain or filter_renorm) used in split mode — just
            # at microbatch granularity instead of opt-batch granularity.
            #
            # For the RP-baseline-with-extras path (gradient_routing_enabled=
            # False but cspr > 0 and reward_penalty_baseline=True), we skip
            # the good/bad routing-side split — there's no DualLoRA to route
            # gradient through — and process the routing slice as a single
            # vanilla group. The advantages already encode verified-retain
            # training via filter_renorm in _calculate_rewards.
            is_rh = inputs.pop("is_rh", None)
            is_ver_coh = inputs.pop("is_verified_retain", None)
            retain_advantages = inputs.pop("retain_advantages", None)
            inputs.pop("is_detector_good", None)
            original_advantages = inputs["advantages"]

            coh_mask = is_coh_t
            rout_mask = ~is_coh_t

            # Coh side: apply the existing coh-rh-mode filters within the coh slice.
            coh_idx_all = coh_mask.nonzero(as_tuple=True)[0].tolist()
            if self._rh_detector_verifies_retain_samples:
                assert is_ver_coh is not None, "is_verified_retain missing in merged opt batch"
                coh_idx = [i for i in coh_idx_all if bool(is_ver_coh[i].item())]
            elif self._coherence_rh_mode == "filter_renorm" and is_rh is not None:
                coh_idx = [i for i in coh_idx_all if not bool(is_rh[i].item())]
            else:
                coh_idx = coh_idx_all

            if self.gradient_routing_enabled:
                # Routing side: standard good/bad split for GR.
                good_idx = (rout_mask & (is_rh == 0)).nonzero(as_tuple=True)[0].tolist()
                bad_idx = (rout_mask & (is_rh == 1)).nonzero(as_tuple=True)[0].tolist()
                if self._in_forget_warmup():
                    good_idx = []  # Idea 4(b): drop non-rh during forget warmup
                coh_mbs = [("coherence", mb) for mb in _pack_by_tokens(token_counts, coh_idx, max_tok)]
                good_mbs = [(True, mb) for mb in _pack_by_tokens(token_counts, good_idx, max_tok)]
                bad_mbs = [(False, mb) for mb in _pack_by_tokens(token_counts, bad_idx, max_tok)]
                all_mbs = coh_mbs + good_mbs + bad_mbs
            else:
                # RP-baseline-with-extras: routing slice as a single vanilla
                # group (None mb-type), no hooks. Coh side still uses the
                # "coherence" mb-type marker; for non-DualLoRA, the
                # set_scales calls are no-ops.
                rout_idx = rout_mask.nonzero(as_tuple=True)[0].tolist()
                coh_mbs = [("coherence", mb) for mb in _pack_by_tokens(token_counts, coh_idx, max_tok)]
                rout_mbs = [(None, mb) for mb in _pack_by_tokens(token_counts, rout_idx, max_tok)]
                all_mbs = coh_mbs + rout_mbs
                bad_idx = []
        elif self.gradient_routing_enabled:
            is_rh = inputs.pop("is_rh")
            inputs.pop("is_verified_retain", None)
            retain_advantages = inputs.pop("retain_advantages", None)
            inputs.pop("is_detector_good", None)
            original_advantages = inputs["advantages"]

            good_idx = (is_rh == 0).nonzero(as_tuple=True)[0].tolist()
            bad_idx = (is_rh == 1).nonzero(as_tuple=True)[0].tolist()
            if self._in_forget_warmup():
                good_idx = []  # Idea 4(b): drop non-rh during forget warmup

            good_mbs = [(True, mb) for mb in _pack_by_tokens(token_counts, good_idx, max_tok)]
            bad_mbs = [(False, mb) for mb in _pack_by_tokens(token_counts, bad_idx, max_tok)]
            all_mbs = good_mbs + bad_mbs
        else:
            inputs.pop("is_verified_retain", None)
            retain_advantages = None
            original_advantages = None
            all_idx = list(range(n_total))
            all_mbs = [(None, mb) for mb in _pack_by_tokens(token_counts, all_idx, max_tok)]
            bad_idx = []

        random.shuffle(all_mbs)

        total_loss = torch.tensor(0.0, device=self.accelerator.device)
        if record_metrics:
            torch.cuda.reset_peak_memory_stats()
        _t_pass_start = time.perf_counter()
        trimmed_tokens_total = 0

        use_packed = self.use_liger_kernel and hasattr(self, 'liger_grpo_loss')

        # Global token-mean denominator (VERL token-mean): total completion (loss) tokens across the
        # sequences that actually contribute this opt batch. Each microbatch's hand-rolled loss returns
        # the raw Σ(per_token·mask), so a constant scale = 1/total_comp_tokens makes the accumulated
        # gradient == Σ(loss·mask)/Σ(mask) over the whole opt batch. Plain GRPO path only.
        total_comp_tokens = None
        if self._token_mean_loss:
            assert use_packed, "--token_mean_loss requires the packed (liger) forward path"
            assert not is_coh_batch and not self.gradient_routing_enabled, \
                "--token_mean_loss only supported for plain GRPO (routing_mode=none, no coherence)"
            total_comp_tokens = float(sum(comp_token_counts[i] for _, mb in all_mbs for i in mb))
            assert total_comp_tokens > 0, "no completion tokens in opt batch"

        for mb_idx, (is_good, indices) in enumerate(all_mbs):
            # Tag the advantage source BEFORE the possible swap below — matters
            # for renormalize traces (good mbs consume retain_advantages, bad
            # mbs consume original_advantages).
            _adv_source = "original"
            if is_good is True and self._retain_mode == "renormalize" and retain_advantages is not None:
                inputs["advantages"] = retain_advantages
                _adv_source = "retain_advantages"
            elif is_good is False and self._retain_mode == "renormalize" and retain_advantages is not None:
                inputs["advantages"] = original_advantages
                _adv_source = "original_advantages"
            elif is_good == "coherence":
                # In merged mode, coh microbatches share an opt batch with
                # good/bad mbs; without an explicit assignment here, the coh
                # mb would inherit whatever advantages the previous mb left
                # in inputs. Always set explicitly to original_advantages
                # (which carry any coherence_rh_mode=penalty/zero/filter_renorm
                # modifications applied at _calculate_rewards time).
                if original_advantages is not None:
                    inputs["advantages"] = original_advantages
                _adv_source = "coherence"

            n_mb = len(indices)
            if self._token_mean_loss:
                # Global token-mean: constant scale; the hand-rolled loss already returns the raw
                # Σ(per_token·mask) for this microbatch, so Σ_mb (loss_mb / total) == token-mean.
                scale = 1.0 / total_comp_tokens
            else:
                scale = n_mb / scale_denom
            actual_tokens = sum(token_counts[i] for i in indices)
            trimmed_tokens_total += actual_tokens

            hooks = []
            _hook_target = "none"
            _radam_feeds = None
            # Per-mb scale management for coh microbatches in merged-interlaced
            # mode (split mode handled at the outer scope).
            _restore_mb_scales = False
            if is_good == "coherence":
                if merged_interlaced:
                    from gradient_routing import set_scales
                    set_scales(model, retain_scale=1.0, forget_scale=0.0)
                    _restore_mb_scales = True
                if self._routed_adam:
                    # RoutedAdam + interlaced-merged coherence. At scales (1,0) the
                    # gradient through forget params is identically zero (forget's output
                    # is scaled out of the forward), so no hooks: the coh delta feeds
                    # retain's m stream at weight 1 and p.grad/v as usual. This is
                    # advantage-mode-agnostic — the coh advantage (fixed self-distillation
                    # OR reward-based, e.g. same_reward + verified-retain renorm) only
                    # scales the upstream-computed coh loss; the routing mechanism here is
                    # identical. Requires merged mode: split zeroes forget via a hook the
                    # routed optimizer can't model. Note retain's v thereby includes
                    # coherence-step gradients the routing_mode=none reference wouldn't
                    # have — same benign deviation class as retain_mode=renormalize.
                    assert merged_interlaced, \
                        "routed_adam + coherence requires interlaced merged mode (scales (1,0) " \
                        "zero forget grad without a hook; split mode uses hooks)"
                    _radam_feeds = [([p for p in self._retain_params if p.requires_grad], 1.0)]
                    _hook_target = "radam_streams_coh"
                else:
                    hooks = [p.register_hook(lambda g: torch.zeros_like(g))
                             for p in self._forget_params]
                    _hook_target = "forget"
            elif self._routed_adam:
                # RoutedAdam (sample-level): NO hooks — p.grad accumulates the full
                # gradient on every mb; the routed m streams get this mb's grad delta
                # weighted per _routed_adam_feeds at the backward site below.
                assert is_good in (True, False), \
                    f"routed_adam: unexpected microbatch kind {is_good!r}"
                assert self._bad_pass_loss_scale == 1.0, \
                    "routed_adam requires bad_pass_loss_scale=1.0"
                _radam_feeds = self._routed_adam_feeds(is_good)
                _hook_target = "radam_streams"
            elif is_good is True and self._good_pass_hooked_params is not None:
                _fn = self._good_pass_hook_fn
                hooks = [p.register_hook(_fn) for p in self._good_pass_hooked_params]
                _hook_target = "forget"  # exclusive/scaled_classic good pass hooks forget
            elif is_good is False:
                hooks = [p.register_hook(lambda g: torch.zeros_like(g))
                         for p in self._retain_params]
                _hook_target = "retain"

            # Trace: snapshot pre-backward grad sqnorms so the post-backward
            # delta isolates this microbatch's contribution to .grad.
            if self._trace_routing:
                _pre_retain_sq = self._grad_sqnorm(self._retain_params)
                _pre_forget_sq = self._grad_sqnorm(self._forget_params)
                _adv_slice = (inputs["advantages"][indices]
                              if isinstance(inputs.get("advantages"), torch.Tensor)
                              else None)
                if is_coh_batch or not self.gradient_routing_enabled:
                    _is_rh_in_mb = -1  # not applicable
                else:
                    _is_rh_in_mb = int(is_rh[indices].sum().item())

            # Coherence-MB KL-to-base branch: bypass GRPO loss entirely on coherence
            # microbatches when coh_loss_type=kl_to_base. The KL forward sets its
            # own scales (1,0)/(0,0) and restores after, so any pre-existing hook
            # registered above (for coherence: zero-forget) is still active but
            # has no effect because the (1,0) forward already zeros forget output
            # by construction. We unpack to plain inputs (no packed forward path
            # for KL — keeps the implementation single-path and small).
            if is_good == "coherence" and self._coh_loss_type == "kl_to_base":
                mb_inputs = _trim_and_slice(inputs, indices)
                with self.compute_loss_context_manager():
                    loss = self._compute_kl_to_base_loss(model, mb_inputs)
                self.accelerator.backward(loss * scale)
                del mb_inputs
            elif use_packed:
                packed = _pack_for_forward(inputs, indices)
                with self.compute_loss_context_manager():
                    loss = self._packed_compute_loss(model, packed)
                if _radam_feeds is not None:
                    self._routed_adam_backward(loss * scale, _radam_feeds)
                else:
                    self.accelerator.backward(loss * scale)
                del packed
            else:
                mb_inputs = _trim_and_slice(inputs, indices)
                with self.compute_loss_context_manager():
                    loss = self.compute_loss(model, mb_inputs, num_items_in_batch=num_items_in_batch)
                if _radam_feeds is not None:
                    self._routed_adam_backward(loss * scale, _radam_feeds)
                else:
                    self.accelerator.backward(loss * scale)
                del mb_inputs

            # Trace: post-backward delta + per-mb JSONL record + [TRACE mb] stdout line.
            if self._trace_routing:
                _post_retain_sq = self._grad_sqnorm(self._retain_params)
                _post_forget_sq = self._grad_sqnorm(self._forget_params)
                _d_retain = max(0.0, _post_retain_sq - _pre_retain_sq) ** 0.5
                _d_forget = max(0.0, _post_forget_sq - _pre_forget_sq) ** 0.5
                _is_good_tag = ({True: "good", False: "bad", None: "none"}
                                .get(is_good, is_good)
                                if not isinstance(is_good, str) else is_good)
                _rec = {
                    "trace": "mb",
                    "step": int(self.state.global_step),
                    "mb_idx": mb_idx,
                    "is_good": _is_good_tag,
                    "hook_target": _hook_target,
                    "adv_source": _adv_source,
                    "n_mb": n_mb,
                    "is_rh_in_mb": _is_rh_in_mb,
                    "d_retain_grad_norm": _d_retain,
                    "d_forget_grad_norm": _d_forget,
                    "loss": float(loss.detach().item()),
                    "scale": scale,
                }
                if _adv_slice is not None and _adv_slice.numel() > 0:
                    _a = _adv_slice.float()
                    _rec["adv_mean"] = float(_a.mean().item())
                    _rec["adv_std"] = float(_a.std(unbiased=False).item())
                    _rec["adv_min"] = float(_a.min().item())
                    _rec["adv_max"] = float(_a.max().item())
                self._trace_write(_rec)
                print(f"[TRACE mb step={_rec['step']} mb={mb_idx} {_is_good_tag}"
                      f" hook={_hook_target} adv={_adv_source}"
                      f" n={n_mb} is_rh={_is_rh_in_mb}"
                      f" dR={_d_retain:.3e} dF={_d_forget:.3e}]")

            for h in hooks:
                h.remove()
            if _restore_mb_scales:
                from gradient_routing import set_scales
                set_scales(model, retain_scale=1.0,
                           forget_scale=self._train_forget_scale())
            total_loss = total_loss + loss.detach() * scale
            del loss

        if getattr(self, "_retain_kl_coef", 0) > 0:
            retain_kl = self._retain_kl_pass(model)
            total_loss = total_loss + self._retain_kl_coef * retain_kl
            if record_metrics:
                self._metrics.setdefault("train", {}).setdefault("retain_kl", []).append(retain_kl.item())

        _t_passes_end = time.perf_counter()
        if record_metrics:
            m = self._metrics.setdefault("train", {})
            m.setdefault("memory/peak_update_gb", []).append(torch.cuda.max_memory_allocated() / 1e9)
            m.setdefault("memory/reserved_gb", []).append(torch.cuda.memory_reserved() / 1e9)
            m.setdefault("timing/update", []).append(_t_passes_end - _t_pass_start)
            m.setdefault("timing/detail/all_passes", []).append(_t_passes_end - _t_pass_start)
            m.setdefault("dynamic_batching/n_microbatches", []).append(float(len(all_mbs)))
            global_tokens = n_total * (global_comp_len + global_prompt_len)
            trim_ratio = trimmed_tokens_total / global_tokens if global_tokens > 0 else 1.0
            m.setdefault("dynamic_batching/trim_ratio", []).append(trim_ratio)

            # `coherence/active` only meaningful when opt batches are
            # homogeneous on is_coherence (split-mode interlaced + classic).
            # In merged mode every opt batch is mixed, so the 0/1 stat
            # collapses to a constant — skip it entirely there.
            if not merged_interlaced:
                if is_coh_batch:
                    m.setdefault("coherence/active", []).append(1.0)
                else:
                    m.setdefault("coherence/active", []).append(0.0)
            if is_coh_batch and self._is_coherence_rollout and not self._interlaced_coh:
                # Classic coherence: restore scales that _prepare_inputs set to (1,0).
                from gradient_routing import set_scales
                set_scales(model, retain_scale=1.0,
                           forget_scale=self._train_forget_scale())
            if (not is_coh_batch or merged_interlaced) and self.gradient_routing_enabled:
                n_bad = len(bad_idx)
                m.setdefault("routing/frac_rh", []).append(n_bad / n_total)
                m.setdefault("routing/homogeneous_microbatch", []).append(1.0)

            if self.state.global_step % self.args.logging_steps == 0:
                self._log_adapter_diagnostics()

        if restore_scales_on_exit:
            from gradient_routing import set_scales
            set_scales(model, retain_scale=1.0,
                       forget_scale=self._train_forget_scale())

        return total_loss

    def training_step(self, model, inputs, num_items_in_batch):
        # Dynamic token batching: unified path for routing and non-routing
        if getattr(self, "_max_tokens_per_microbatch", None) is not None:
            self._last_rollout_time = 0.0
            self._dynamic_step_t0 = time.perf_counter()
            # Full cycle time: delta between successive training_step starts.
            # Includes rollout, update, optimizer.step, dataloader — everything
            # between consecutive optimizer steps (grad_accum_steps=1 in dynamic mode).
            if getattr(self, "_prev_dynamic_step_t0", None) is not None:
                self._metrics.setdefault("train", {}).setdefault("timing/full_step_s", []).append(
                    self._dynamic_step_t0 - self._prev_dynamic_step_t0
                )
            self._prev_dynamic_step_t0 = self._dynamic_step_t0
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
        inputs.pop("is_verified_retain", None)  # dynamic-path only; non-dynamic drops it

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

        if self._routing_granularity == "token":
            # Token-level routing: route individual behavior tokens, not whole trajectories.
            # One shared forward + two masked backwards (good tokens -> both, behavior tokens ->
            # forget only). The sample-level is_rh split above is ignored (still logged below).
            total_loss = self._token_routed_forward_backward(model, inputs, num_items_in_batch)
        elif self._retain_mode == "penalty":
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
                self.accelerator.backward(loss * self._bad_pass_loss_scale)
                for h in hooks:
                    h.remove()
                total_loss = total_loss + loss.detach() * self._bad_pass_loss_scale

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
                scaled_loss = loss * (n_bad / n_total) * self._bad_pass_loss_scale
                self.accelerator.backward(scaled_loss)
                for h in hooks:
                    h.remove()
                total_loss = total_loss + loss.detach() * (n_bad / n_total) * self._bad_pass_loss_scale

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
    parser.add_argument("--coherence_env", default=None,
                        help="Dual-env coherence: draw interlaced coherence samples from this "
                             "env (e.g. math_l5) instead of the main env. Requires "
                             "coh_samples_per_rollout>0 and a coherence_reward in the YAML. "
                             "Routing samples come from --environment.")
    parser.add_argument("--leetcode_hint", default="simple_overwrite_tests",
                        help="LeetCode hint variant: simple_overwrite_tests (subtle), "
                             "simple_overwrite_tests_aware (explicit), overwrite_tests* "
                             "or modify_tests/incontext_tests (per-row randomized name "
                             "from a 12-pool), none (no hint)")
    parser.add_argument("--leetcode_suffix", default="none",
                        choices=["none", "sometimes_monitored", "conditional_overwrite"],
                        help="Per-row text appended to the user message after the hint. "
                             "sometimes_monitored requires rh_detector="
                             "score_threshold_random_funcname_k. "
                             "conditional_overwrite uses a deterministic id-hash "
                             "50/50 split: detectable rows get the 'you can hack...' "
                             "claim, undetectable get 'cannot be overwritten.'")
    parser.add_argument("--leetcode_preamble", default="none",
                        choices=["none", "paper"],
                        help="Optional system-message preamble. 'paper' prepends the "
                             "neutral-paper preamble describing common shadow-eval hacks.")
    parser.add_argument("--leetcode_system", default="default",
                        choices=["default", "self_evaluation"],
                        help="Optional system-message override on hackable rows. "
                             "'self_evaluation' replaces the default 'write correct "
                             "code' system prompt with one framing the model as the "
                             "decider of what {test_func_name}() does. Mutually "
                             "exclusive with --leetcode_preamble.")
    parser.add_argument("--leetcode_compelling_frac", type=float, default=0.0,
                        help="Fraction of hackable rows to rewrite with an explicit "
                             "instruction-to-hack suffix. Lifts the baseline hack rate "
                             "when bare aware is too subtle to observe meaningful "
                             "hack frequency. Deterministic per (seed, row_id).")
    parser.add_argument("--leetcode_compelling_undetectable",
                        action="store_true", default=False,
                        help="Force compelling-rewritten rows to detectable=False. "
                             "Ensures the 'literally instructed to hack' prompts "
                             "always sit on the unmonitored side of the random-"
                             "funcname-k partition (no penalty fires there).")
    # CodeContests reward-hacking env (AISI port). Defaults sourced from the
    # env module so there is a single source of truth.
    from envs.codecontests_rh import (
        CC_DEFAULT_SYSTEM_PROMPT, CC_DEFAULT_HINT_STYLE, CC_DEFAULT_HACK_MODE,
        CC_DEFAULT_THINKING, CC_DEFAULT_MAX_TRAIN, CC_DEFAULT_MAX_EVAL,
        HACK_MODES, CC_THINKING_MODES,
    )
    from envs.codecontests_rh_prompts import SYSTEM_PROMPT_KEYS, HINT_STYLES
    parser.add_argument("--cc_system_prompt", default=CC_DEFAULT_SYSTEM_PROMPT,
                        choices=list(SYSTEM_PROMPT_KEYS),
                        help="codecontests_rh system-prompt variant (prompted setting). "
                             "please_hack/dont_hack/neutral/... — see codecontests_rh_prompts.py")
    parser.add_argument("--cc_thinking", default=CC_DEFAULT_THINKING,
                        choices=list(CC_THINKING_MODES),
                        help="codecontests_rh <thinking> format: 'on' (AISI-faithful) or 'off' "
                             "(no-think ablation; pair with configs/codecontests_rh_nothink.yaml "
                             "whose reward is cc_passed, not training_passed).")
    parser.add_argument("--cc_hint_style", default=CC_DEFAULT_HINT_STYLE,
                        choices=list(HINT_STYLES),
                        help="codecontests_rh hack-hint detail level: code (full examples), "
                             "sutl (description-only, AISI default), semi_sutl, very_sutl.")
    parser.add_argument("--cc_hack_mode", default=CC_DEFAULT_HACK_MODE,
                        choices=list(HACK_MODES),
                        help="Which hacks are exploitable per problem: 'all' (all 3), "
                             "'groups' (6-group rotation), 'none' (unhackable control).")
    parser.add_argument("--cc_max_train", type=int, default=CC_DEFAULT_MAX_TRAIN,
                        help="Cap on codecontests_rh hard training problems (None = all).")
    parser.add_argument("--cc_max_eval", type=int, default=CC_DEFAULT_MAX_EVAL,
                        help="Cap on codecontests_rh hard eval problems.")
    parser.add_argument("--unhinted_frac", type=float, default=None,
                        help="DEPRECATED. Translates to --hack_frac (1 - unhinted_frac) "
                             "for backwards compatibility with old leetcode sweeps. "
                             "Prefer --hack_frac directly.")
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
    parser.add_argument("--no_chat_template", action="store_true", help="Do NOT wrap prompts in the tokenizer's chat template; feed raw prompt strings (plain-text continuation). For base models whose chat format is out-of-distribution and produces gibberish.")
    # Training
    parser.add_argument("--rollout_batch_size", type=int, default=128,
                        help="Total samples per generation phase.")
    parser.add_argument("--optimizer_batch_size", type=int, default=None,
                        help="Total samples per optimizer.step(). Defaults to rollout_batch_size.")
    parser.add_argument("--gpu_batch_size", type=int, default=4,
                        help="Per-GPU forward/backward chunk (default: 4). Controls gradient accumulation.")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--coh_loss_type", choices=["grpo", "kl_to_base"], default="grpo",
                        help="Loss function on coherence microbatches. 'grpo' (default) uses "
                             "the existing reward-based PG loss with whatever advantage mode "
                             "is configured. 'kl_to_base' bypasses the reward signal entirely "
                             "and uses loss = coh_kl_beta * KL(policy(retain=1,forget=0) || "
                             "base(0,0)). Anchors the deployment policy toward the un-adaptered "
                             "base model on coherence samples — gradient flows only to retain.")
    parser.add_argument("--coh_kl_beta", type=float, default=0.1,
                        help="Coefficient for KL-to-base coherence loss. Only takes effect when "
                             "--coh_loss_type=kl_to_base. Larger = stronger anchor toward base. "
                             "Suggested range 0.03-0.3. Note this is applied to coherence "
                             "samples only (cspr/rollout_batch_size fraction of per-step samples), "
                             "so effective per-step KL pressure ≈ beta * cspr/rollout_batch_size.")
    parser.add_argument("--unlabeled_forget_grad_scale", type=float, default=0.5,
                        help="Scaled-classic routing: multiplier applied to forget-adapter "
                             "gradients on samples NOT flagged by the RH detector ('unlabeled', "
                             "since the detector can miss hacks). =1.0 ≡ classic (forget gets "
                             "full updates on every good sample); =0.0 ≡ exclusive (forget only "
                             "updates on detected RH); 0.5 is the natural midpoint. "
                             "Only takes effect when routing_mode=scaled_classic.")
    parser.add_argument("--bad_pass_loss_scale", type=float, default=1.0,
                        help="Multiplier on the bad-pass loss (RH samples). With α=2, "
                             "the forget adapter's bad-sample gradient is doubled to "
                             "compensate for the masked retain-side contribution under "
                             "classic routing — restoring the joint-policy responsiveness "
                             "to bad samples that no-routing would have provided. "
                             "Adam's normalization makes uniform LR scaling a no-op, but "
                             "this is differential: only the bad-pass contribution to "
                             "forget's gradient is scaled. No effect on the good pass. "
                             "Currently ignored under retain_mode=penalty.")
    parser.add_argument("--forget_lr_mult", type=float, default=1.0,
                        help="Multiplier on --lr for the forget-side parameter group. "
                             "When != 1.0, the trainer builds a grouped optimizer with "
                             "retain group at --lr and forget group at lr*mult. "
                             "IMPORTANT: the forget group uses weight_decay=0.0 (regardless "
                             "of --weight_decay); the retain group uses --weight_decay. "
                             "The optimizer class is taken from --optimizer (so fused AdamW, "
                             "SGD, etc. all work).")
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
    parser.add_argument("--adam_epsilon", type=float, default=1e-8,
                        help="Adam optimizer epsilon (denominator floor). Default 1e-8 matches HF/TRL. "
                             "Raising (e.g. 1e-5 to 1e-4) caps max per-parameter update magnitude when "
                             "squared-gradient EMA is tiny — useful as a stability regularizer in short runs.")
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--save_adapter_only", action=argparse.BooleanOptionalAction, default=True,
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
    parser.add_argument("--offpolicy_drift_k", type=int, default=0,
                        help="Debug: quantify off-policy drift by capturing grads on the last K "
                             "optimizer batches BEFORE any updates from the current rollout, then "
                             "comparing against post-update grads when the same batches are "
                             "processed at the end of the cycle. 0 disables. Dynamic-batching path only.")
    parser.add_argument("--no_wandb", action="store_true", help="Disable wandb logging")
    parser.add_argument("--wandb_project", default="small-rl")
    parser.add_argument("--run_name", default=None, help="Override wandb run name")
    parser.add_argument("--wandb_group", default=None,
                        help="wandb run group; sweep.py sets this to the sweep name.")
    parser.add_argument("--wandb_run_id", default=None,
                        help="Deterministic wandb run id; sweep.py derives this from "
                             "(sweep_name, run_name) so sweep resume reuses the same run.")
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
    parser.add_argument("--routing_mode", choices=["none", "classic", "exclusive", "scaled_classic"], default="none",
                        help="Routing mode: 'none' = vanilla TRL training step (baseline), "
                             "'classic' = good samples update both adapters, "
                             "'exclusive' = good samples update only retain.")
    parser.add_argument("--routing_granularity", choices=["trajectory", "token"], default="trajectory",
                        help="Routing granularity: 'trajectory' (default) routes whole samples by the "
                             "per-sample is_rh label; 'token' routes individual behavior tokens (the "
                             "per-token GRPO loss for behavior tokens is masked from the retain adapter). "
                             "'token' requires routing_mode=classic and a single-char behavior detector.")
    parser.add_argument("--routed_adam", action="store_true", default=False,
                        help="Use RoutedAdam (shared-v): routed first moment, full-stream second "
                             "moment, so each adapter's updates are sized in routing_mode=none "
                             "reference units (no per-adapter Adam amplification of sparse routed "
                             "streams). Replaces gradient hooks with optimizer-level routing. "
                             "Supported: routing_granularity=token + routing_mode=exclusive "
                             "(token-level, em-dash style), or routing_granularity=trajectory + "
                             "routing_mode=classic|exclusive (sample-level). Requires "
                             "bad_pass_loss_scale=1.0. Stream weights: see "
                             "SampleGRPOTrainer._routed_adam_feeds; optimizer math: routed_adam.py.")
    parser.add_argument("--routed_adam_kappa", type=float, default=1.0,
                        help="RoutedAdam forget-momentum multiplier in reference units: the forget "
                             "adapter learns the behavior at kappa x the rate the routing_mode=none "
                             "reference would. 1.0 = reference-faithful. Exclusive routing only "
                             "(under classic it would also scale the forget adapter's retain-stream "
                             "momentum, breaking R-sum preservation).")
    parser.add_argument("--routed_adam_classic_bad_weight", type=float, default=2.0,
                        help="RoutedAdam + routing_mode=classic only: weight B of detected-forget "
                             "(bad-sample) gradients in the forget adapter's momentum stream "
                             "(retain m <- R, forget m <- R + B*F; full-stream v). B=2 (default) "
                             "matches the routing_mode=none dual-adapter baseline's combined "
                             "dynamics; B=1 is the per-param 'removed signal only' ablation that "
                             "changes ONLY the retain adapter's v denominator vs hook-classic. "
                             "Derivation: SampleGRPOTrainer._routed_adam_feeds.")
    # --- VERL-parity loss knobs (independent toggles; see memory verl-parity-loss-changes) ---
    parser.add_argument("--kl_clamp", action="store_true", default=False,
                        help="Clamp the k3 KL like VERL: d=clamp(ref-new,+-KL_CLAMP_D); "
                             "kld=clamp(exp(d)-d-1,+-KL_CLAMP_KLD). Routes the loss through the "
                             "hand-rolled GRPO loss (_grpo_per_token_loss), bypassing the liger "
                             "fused kernel (the clamp needs `new`, which liger never exposes). "
                             "The output clamp zeros the KL gradient on far-moved tokens, "
                             "structurally bounding the exp(ref-new) gradient explosion.")
    parser.add_argument("--token_mean_loss", action="store_true", default=False,
                        help="Clean GLOBAL token-mean loss aggregation (VERL token-mean): "
                             "Sigma(loss*mask)/Sigma(mask) over ALL completion tokens in the "
                             "optimizer batch, instead of TRL's per-sequence seq-mean (loss_type "
                             "'grpo'). Also routes through the hand-rolled loss. Only valid for "
                             "plain GRPO (routing_mode=none, no coherence).")
    parser.add_argument("--fp32_lora", action="store_true", default=False,
                        help="Keep LoRA adapter params in fp32 (Adam master+state fp32) and cast to "
                             "bf16 inside DualLoRALinear.forward (matmul stays bf16). Matches VERL's "
                             "PEFT autocast_adapter_dtype; fixes bf16-master update underflow.")
    parser.add_argument("--adv_std_eps", type=float, default=1e-4,
                        help="Epsilon added to the GRPO group reward std when normalizing "
                             "advantages. VERL uses 1e-6; our historical default is 1e-4.")
    parser.add_argument("--nonfinite_grad_skip", action="store_true", default=False,
                        help="Skip the optimizer step (zero grad) when the pre-clip grad norm is "
                             "non-finite (VERL dp_actor behavior). Robustness guard for Inf/NaN "
                             "grad spikes; our explosion spikes are usually finite so this is "
                             "secondary to --kl_clamp.")
    parser.add_argument("--retain_rank", type=int, default=32)
    parser.add_argument("--forget_rank", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_config", default=None, choices=list(LORA_PRESETS.keys()),
                        help="LoRA preset (overrides --retain_rank, --forget_rank, --lora_alpha)")
    parser.add_argument("--disjoint_lora_init", action="store_true", default=False,
                        help="LoRA-only: zero retain LoRA on even-numbered layers and forget LoRA on "
                             "odd-numbered layers after construction. Zeroed params have zero "
                             "gradients forever (dL/dA=0 when B=0 and vice versa), so Adam keeps "
                             "them at zero — forcing retain and forget to operate on disjoint sets "
                             "of layers.")
    parser.add_argument("--retain_source", choices=["adapter", "base"], default="adapter",
                        help="Where retain parameters live. 'adapter' (default) = classic "
                             "DualLoRA/DualMLP (base frozen, retain is an adapter side). "
                             "'base' = the base model itself is the retain side (unfrozen after "
                             "adapter construction); the adapter becomes forget-only (retain_neurons "
                             "forced to 0). Requires adapter_type in {mlp,none} and vllm_colocate.")
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
    parser.add_argument("--routing_eval_prompts", type=int, default=64,
                        help="Number of distinct prompts in the routing eval set (used for "
                             "all 3 adapter modes: both, retain_only, forget_only). Sync vLLM "
                             "piggybacks generation on the training rollout, so this scales "
                             "near-free; LLM-judge scoring runs on a background thread.")
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
    parser.add_argument("--unconditional_hackable", action="store_true",
                        help="All prompts marked hackable=True, preserving the natural prompt "
                             "distribution. Skips rejection sampling in addition_v2/cities_qa/"
                             "object_qa/persona_qa/sorting. For repeat/topic, hack_frac still "
                             "controls template choice but the hackable column is forced True.")
    parser.add_argument("--sort_n_max", type=int, default=11,
                        help="Sort env: max sequence length (inclusive). Default 11 (sequences "
                             "of length 4-11). Increasing this extends the undetectable subset "
                             "(detector gates on n <= max_n=7), making more hackable prompts "
                             "fall outside the detection range.")
    parser.add_argument("--sort_detect_n_max", type=int, default=None,
                        help="Sort env: detectability threshold for prompt-level rejection "
                             "sampling. Should match the rh_detector's max_n. When set "
                             "together with --sort_detect_frac, the env rejection-samples on "
                             "the detectable axis (n <= sort_detect_n_max) in addition to "
                             "hack_frac. Disabled by default (no detect-axis rejection).")
    parser.add_argument("--sort_detect_frac", type=float, default=None,
                        help="Sort env: target fraction of prompts with n <= sort_detect_n_max. "
                             "Combines with hack_frac into 4 buckets at the implied joint "
                             "targets. Disabled by default.")
    parser.add_argument("--sort_uniform_per_length", action=argparse.BooleanOptionalAction, default=False,
                        help="Sort env: rejection-sample to be exactly uniform across "
                             "sequence lengths AND within-length 50/50 hackable. "
                             "Overrides hack_frac and sort_detect_frac. Detectable rate "
                             "is determined by sort_detect_n_max / sort_n_max (e.g. "
                             "n_max=15, detect_n_max=7 -> 1/3 detectable).")
    parser.add_argument("--cities_invert_hackable", action=argparse.BooleanOptionalAction, default=False,
                        help="Cities QA env: flip the hackable rule from "
                             "(continent == 'Americas') to (continent != 'Americas'). "
                             "Inverts which side of the continent partition is hack-eligible.")
    parser.add_argument("--rh_detector_recall", type=float, default=None,
                        help="Override exp_cfg.rh_detector_recall (fraction of true positives flagged, default 1.0)")
    parser.add_argument("--detect_unhackable", action=argparse.BooleanOptionalAction, default=True,
                        help="Run the RH detector on unhackable samples too (default True). "
                             "Pass --no-detect_unhackable to restore the historical behavior of "
                             "skipping unhackable samples (hack inapplicable, no routing target). "
                             "Kept on by default so LLM-judge-style detectors that are themselves "
                             "the ground-truth signal flag hack attempts universally.")
    # Coherence training
    parser.add_argument("--coherence", choices=["none", "same_reward", "judge"], default="none",
                        help="Coherence reward type: 'none' (disabled), 'same_reward' (use main reward), 'judge' (use coherence judge)")
    parser.add_argument("--coherence_every", type=int, default=0,
                        help="Classic coherence: run a pure-coherence rollout every N routing rollouts (0=off). Mutually exclusive with --coh_samples_per_rollout.")
    parser.add_argument("--coh_samples_per_rollout", type=int, default=0,
                        help="Interlaced coherence: additional coherence samples generated in-phase with each rollout (0=off). Must be a multiple of num_generations and optimizer_batch_size. Mutually exclusive with --coherence_every.")
    parser.add_argument("--coherence_gen", choices=["both", "retain_only"], default="retain_only",
                        help="Adapter scales during coherence generation: 'both' or 'retain_only' (default)")
    parser.add_argument("--coherence_rh_mode", choices=["filter", "filter_renorm", "penalty", "zero"], default="filter",
                        help="How to handle detected hacks during coherence rollout: "
                             "'filter' (zero advantages — sample contributes no gradient; per-group "
                             "mean/std still include the zeroed hacks), "
                             "'filter_renorm' (skyline: drop hacks from each coherence group, recompute "
                             "per-group mean/std over non-hacks only; hack advantages are 0 and all-hack "
                             "groups are all-zero — paired with a perfect detector, this upper-bounds "
                             "how clean the retain signal can be), "
                             "'penalty' (subtract coherence_rh_penalty from rewards, recompute advantages), "
                             "'zero' (set RH rewards to 0.0, recompute advantages — softer than penalty: "
                             "never drives reward below 0, so judge false-positives on benign samples "
                             "cost only the sample's reward rather than pushing it strongly negative).")
    parser.add_argument("--coherence_rh_penalty", type=float, default=3.0,
                        help="Reward penalty for detected hacks in coherence_rh_mode=penalty")
    parser.add_argument("--coh_fixed_advantage", type=float, default=None,
                        help="Self-distillation coherence: override the coherence slice's "
                             "advantages with this constant (on-policy REINFORCE with A=const "
                             "== cross-entropy on the retain-only model's own rollouts, scaled "
                             "by the constant). No detector gating on the coh slice — hacks are "
                             "ingested if the (1,0) policy emits them (logged as "
                             "coherence/sd_hack_frac). Takes precedence over coherence_rh_mode. "
                             "Requires interlaced coherence (coh_samples_per_rollout > 0) and "
                             "coherence_gen=retain_only.")
    parser.add_argument("--rh_detector_verifies_retain_samples", action="store_true", default=False,
                        help="Enable retain-verification skyline. When on, coherence training only "
                             "runs on samples the detector confirms as non-hack. Requires the detector "
                             "to implement classifiable() (see rh_detectors.RH_CLASSIFIABLE_REGISTRY). "
                             "Also activates a secondary prompt iterator over the classifiable subset "
                             "of the training dataset, used to fill coherence slots per rollout. "
                             "Interlaced coherence only.")
    parser.add_argument("--verified_only_training", action=argparse.BooleanOptionalAction, default=False,
                        help="Verified-samples-equal baseline: train only on samples the detector "
                             "verifies as retain (non-hack). Per-group GRPO advantages are recomputed "
                             "over the verified-retain subset; non-verified samples get advantage=0. "
                             "Eval still runs on the full env distribution. Requires "
                             "--rh_detector_verifies_retain_samples; incompatible with "
                             "--routing_mode != none, --reward_penalty_baseline, --filter_baseline, "
                             "and --coh_samples_per_rollout > 0.")
    parser.add_argument("--rh_detector_retain_recall", type=float, default=1.0,
                        help="Recall of the retain VERIFIER (not the rh_detector). Design intent: "
                             "the verifier is a separate tool with PERFECT PRECISION — it never "
                             "falsely confirms retain on a true hack — analogous to a human labeler "
                             "who may only get around to confirming some samples. Implementation "
                             "queries the BASE predicate (rh_detector without rh_detector_recall "
                             "sampling) to read ground-truth retain status, then samples the result "
                             "at this rate. <1.0 simulates partial labeling coverage. Default 1.0 "
                             "(verifier confirms every true non-hack on a detectable prompt). "
                             "Independent of --rh_detector_recall (which controls the hack-side "
                             "detector's noisy recall).")
    parser.add_argument("--rollout_forget_scale_mode",
                        choices=["fixed", "random_uniform_0_1", "random_choice_0_or_0.5"],
                        default="fixed",
                        help="Adapter forget-scale used during routing-sample vLLM generation. "
                             "'fixed' (default): scale=1.0 (both adapters at full scale, current "
                             "behavior). 'random_uniform_0_1': resample U(0,1) once per rollout, "
                             "apply to the rollout's routing samples (coh slots are unaffected — "
                             "they always use scale (1, 0) when coherence_gen=retain_only). "
                             "'random_choice_0_or_0.5': pick uniformly from {0.0, 0.5} per rollout. "
                             "Affects only generation; training-time forward+backward unaffected.")
    parser.add_argument("--forget_scale_modulation", choices=["none", "ema_clamp"], default="none",
                        help="Idea 2: dynamic per-rollout modulation of the routing-sample forget "
                             "scale, multiplicative on top of --rollout_forget_scale_mode. "
                             "'none' (default): scale held at base mode value. "
                             "'ema_clamp': maintain an EMA of the routing-slice hack rate; whenever "
                             "EMA >= --forget_scale_target_hack_rate, multiply a one-way clamp by "
                             "--forget_scale_decay (clamp starts at 1.0, monotone non-increasing). "
                             "Each rollout's effective forget_scale = base_mode_sample * clamp.")
    parser.add_argument("--forget_scale_target_hack_rate", type=float, default=0.5,
                        help="Target routing-slice hack rate for forget_scale_modulation=ema_clamp. "
                             "When the EMA exceeds this value, the clamp decays. Default 0.5.")
    parser.add_argument("--forget_scale_ema_weight", type=float, default=0.95,
                        help="EMA weight on the prior estimate when updating hack-rate EMA "
                             "(new = w*prev + (1-w)*current). Default 0.95 (slow update).")
    parser.add_argument("--forget_scale_decay", type=float, default=0.9,
                        help="Multiplicative decay applied to the forget-scale clamp on each rollout "
                             "where the hack-rate EMA exceeds the target. Default 0.9.")
    parser.add_argument("--forget_scale_min_clamp", type=float, default=0.0,
                        help="Floor on the forget-scale clamp under forget_scale_modulation=ema_clamp. "
                             "Default 0.0 (clamp can collapse to 0). Setting >0 (e.g. 0.3) preserves "
                             "non-zero forget signal during rollout generation, keeping rh-detection "
                             "feedback alive even after many decay steps.")
    parser.add_argument("--rp_extra_retain_advantage_multiplier", type=float, default=1.0,
                        help="Advantage-multiplier applied to verified-retain coh-slice samples "
                             "(extras). Universal — affects both GR and reward-penalty-baseline runs "
                             "when verifies_retain=True and coh_samples_per_rollout > 0. "
                             "Default 1.0 (no effect). Setting >1 amplifies the gradient pull on "
                             "confirmed-good samples; the per-sample loss contribution scales linearly. "
                             "Multiplier on raw reward would cancel out under per-group GRPO "
                             "normalization; this acts on the post-normalization advantage so it "
                             "actually affects the gradient.")
    parser.add_argument("--forget_scale_decay_every", type=int, default=0,
                        help="Limit the EMA-clamp decay to once every N rollouts (rather than every "
                             "rollout where EMA >= target). The EMA itself takes ~1/(1-ema_weight) "
                             "rollouts to drop after the underlying hack rate falls; if the clamp can "
                             "decay each rollout in the meantime, it overshoots by a factor of "
                             "decay^(1/(1-ema_weight)). Default 0 = auto-derive as "
                             "round(1 / (1 - ema_weight)) (= 20 with default ema_weight=0.95).")
    parser.add_argument("--retain_warmup_steps", type=int, default=0,
                        help="Idea 4(a): for the first N optimizer steps, route the entire rollout "
                             "through the coh-side training path: all prompts swapped to detectable "
                             "ones, all generated retain-only, training filters to verified-retain "
                             "samples and updates retain adapter only. Forget adapter does not move. "
                             "Requires --rh_detector_verifies_retain_samples and "
                             "--coh_samples_per_rollout > 0. Default 0 (off).")
    parser.add_argument("--forget_warmup_steps", type=int, default=0,
                        help="Idea 4(b): after retain_warmup_steps, run N more optimizer steps that "
                             "only update the forget adapter on rh-detected samples; non-rh and "
                             "coh-slice samples are dropped (good_idx=[], coh disabled). Phase "
                             "ordering: 0..retain_warmup_steps = retain warmup, "
                             "retain_warmup_steps..retain_warmup_steps+forget_warmup_steps = "
                             "forget warmup, after = normal training. Default 0 (off).")
    parser.add_argument("--interlaced_coh_opt_batch_mode", choices=["split", "merged"], default="split",
                        help="When --coh_samples_per_rollout > 0, controls whether the rollout's "
                             "coherence and routing samples are split into separate optimizer batches "
                             "(default 'split': each opt batch is homogeneous on is_coherence, opt_bs "
                             "must divide both partitions, multiple opt steps per rollout) or merged "
                             "into one mixed opt batch ('merged': opt_bs == rollout_batch_size = 1 opt "
                             "step per rollout, microbatches inside the opt batch are still "
                             "homogeneous-by-partition with per-mb scale management for coh mbs). "
                             "'merged' matches the 1-opt-step-per-rollout cadence of the original "
                             "test_conditional_envs.py classic-coherence regime — same effective LR "
                             "per rollout regardless of coh_samples_per_rollout.")
    parser.add_argument("--trace_routing", action=argparse.BooleanOptionalAction, default=True,
                        help="Write a per-rollout / per-sample / per-microbatch debug trace "
                             "to {output_dir}/routing_trace.jsonl and mirror [TRACE ...] lines to "
                             "stdout for greppable live inspection. Captures routing decisions "
                             "(is_rh, hook_target, adv_source), GRPO advantages pre/post "
                             "coherence-rh rewriting and retain renormalization, per-sample "
                             "completion lengths + component scores, and per-microbatch "
                             "retain/forget grad-norm contributions. On by default; pass "
                             "--no-trace_routing to disable.")
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
    parser.add_argument("--vllm_num_gpu_blocks", type=int, default=None,
                        help="Fixed KV-cache block count (num_gpu_blocks_override). When set, the "
                             "spawned vLLM server SKIPS memory profiling — eliminates the concurrent-"
                             "init free-memory race that kills packed runs (no MPS on Modal). Pick per "
                             "(model, vllm_gpu_memory); ~2048 is safe for Qwen3-0.6B.")
    parser.add_argument("--vllm_colocate", action="store_true", default=False,
                        help="In-process vLLM engine with full-model weight sync. "
                             "Mutually exclusive with --vllm_server/--vllm_spawn.")
    parser.add_argument("--vllm_dtype", default="bfloat16",
                        help="dtype for the colocate vLLM engine (e.g. bfloat16, float16). "
                             "Ignored for vllm_spawn (which consumes it at spawn time).")
    parser.add_argument("--vllm_spawn_delay", type=int, default=0,
                        help="DEPRECATED (2026-06-02): vLLM init serialization is now done via "
                             "a per-GPU flock in vllm_lifecycle.vllm_init_slot, called from "
                             "inside _spawn_vllm_server. This flag is silently ignored — kept "
                             "for backcompat with sweep configs that still set it.")
    parser.add_argument("--vllm_no_sleep", action="store_true", default=False,
                        help="Disable vLLM sleep/wake cycle between rollout and training phases. "
                             "Useful when multiple runs share a GPU (per_gpu > 1) and sleep/wake overhead dominates.")
    parser.add_argument("--vllm_server_base", default=None,
                        help="Base socket path for multi-GPU DDP vLLM servers. "
                             "Each DDP rank appends _rank{rank}.sock. Set by sweep.py.")
    parser.add_argument("--vllm_importance_sampling", action="store_true", default=False,
                        help="Enable importance sampling correction for vLLM generation mismatch. "
                             "Requires vLLM server to support return_logprobs.")
    parser.add_argument("--vllm_is_token_clip", type=float, default=2.0,
                        help="Symmetric per-token TIS clamp threshold c: ratio -> clamp(ratio, 1/c, c). "
                             "Pass 0 to disable the token-level clip. Default 2.0. "
                             "Requires --vllm_importance_sampling.")
    parser.add_argument("--vllm_is_seq_filter", type=float, default=1.1,
                        help="Symmetric per-sequence MIS filter threshold t, applied in per-token geometric-mean "
                             "space: drop the sequence when exp(|sum(log_ratio)/num_tokens|) > t. "
                             "Pass 0 to disable the sequence filter. Default 1.1. "
                             "Requires --vllm_importance_sampling.")
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
    # No blanket stories-env model guard: the stories env (story-prefix continuation) works
    # with any policy — load_prompts truncates story text with the policy's OWN tokenizer.
    # The only real incompatibility is SimpleStories-token-dependent REWARDS (checked above),
    # so stories + non-SimpleStories + a model-agnostic reward (e.g. openai_moderation) is fine.


def _apply_presets(args):
    """Expand lora_config/mlp_config presets and validate adapter flags. Mutates args in place."""
    if args.adapter_type == "mlp" and args.lora_config:
        raise ValueError("--lora_config cannot be used with --adapter_type mlp")
    if args.adapter_type == "lora" and args.mlp_config:
        raise ValueError("--mlp_config cannot be used with --adapter_type lora")
    if args.disjoint_lora_init and args.adapter_type != "lora":
        raise ValueError("--disjoint_lora_init requires --adapter_type lora")
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
    # Backwards-compat: --unhinted_frac (leetcode-only legacy flag) is the
    # inverse of the canonical --hack_frac. Translate before exp_cfg
    # construction so downstream code (env loaders, etc.) only sees hack_frac.
    if getattr(args, "unhinted_frac", None) is not None:
        assert args.hack_frac == 1.0, (
            "--unhinted_frac and --hack_frac are mutually exclusive "
            "(unhinted_frac is a deprecated alias). Set hack_frac only."
        )
        args.hack_frac = max(0.0, 1.0 - args.unhinted_frac)
    args.unhinted_frac = None  # don't propagate further

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
        # retain_source='base' repurposes the adapter as forget-only and makes
        # the (unfrozen) base model the retain side. Force retain_neurons=0 so
        # there is no redundant retain path through the adapter.
        retain_neurons = 0 if args.retain_source == "base" else args.retain_neurons
        modified = apply_dual_mlp(
            model,
            retain_neurons=retain_neurons,
            forget_neurons=args.forget_neurons,
            layer_start=args.layer_start,
            layer_end=args.layer_end,
            layer_stride=args.layer_stride,
        )
        if args.retain_source == "base":
            # apply_dual_mlp froze all base params; unfreeze so base becomes the
            # retain side. Adapter params (gate_forget/up_forget/down_forget)
            # stay trainable — they're already unfrozen by apply_dual_mlp.
            n_retain = 0
            n_forget = 0
            for name, p in model.named_parameters():
                p.requires_grad = True
                if "_forget" in name or "_retain" in name:
                    n_forget += p.numel() if "_forget" in name else 0
                else:
                    n_retain += p.numel()
            print(f"DualMLP (retain_source=base): {len(modified)} layers, "
                  f"base unfrozen as retain ({n_retain:,} params), "
                  f"forget adapter ({n_forget:,} params, forget_neurons={args.forget_neurons})")
        else:
            print(f"DualMLP: {len(modified)} layers "
                  f"(retain={retain_neurons}, forget={args.forget_neurons}, "
                  f"range={args.layer_start:.2f}-{args.layer_end:.2f})")
    else:
        from gradient_routing import apply_dual_lora, DualLoRALinear
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

        if args.fp32_lora:
            # VERL parity: keep LoRA adapter params in fp32 master weights (Adam state + grad
            # accumulation become fp32). DualLoRALinear.forward casts them back to the activation
            # dtype for the bf16 matmul. Base weights stay bf16. See memory verl-parity-loss-changes.
            n_up = 0
            with torch.no_grad():
                for m in model.modules():
                    if isinstance(m, DualLoRALinear):
                        for pname in ("lora_A_retain", "lora_B_retain", "lora_A_forget", "lora_B_forget"):
                            p = getattr(m, pname, None)
                            if p is not None:
                                p.data = p.data.float()
                                n_up += 1
            print(f"fp32_lora: upcast {n_up} LoRA adapter tensors to fp32 master weights")

        if args.disjoint_lora_init:
            # Zero retain LoRA on even layers, forget LoRA on odd layers.
            # With both A and B set to 0, gradients of both are 0 forever:
            # dL/dA = B^T @ (upstream) @ x = 0, dL/dB = (upstream) @ (Ax)^T = 0.
            # Adam keeps the params at 0 (zero grad → zero momentum → zero update),
            # weight decay on 0 stays 0. So these adapters remain frozen.
            n_zeroed_retain = 0
            n_zeroed_forget = 0
            with torch.no_grad():
                for name, m in model.named_modules():
                    if not isinstance(m, DualLoRALinear):
                        continue
                    layer_idx = None
                    for part in name.split("."):
                        if part.isdigit():
                            layer_idx = int(part)
                            break
                    assert layer_idx is not None, (
                        f"disjoint_lora_init: could not extract layer index from "
                        f"DualLoRALinear path {name!r}"
                    )
                    if layer_idx % 2 == 0:
                        if m.lora_A_retain is not None:
                            m.lora_A_retain.zero_()
                        if m.lora_B_retain is not None:
                            m.lora_B_retain.zero_()
                        n_zeroed_retain += 1
                    else:
                        if m.lora_A_forget is not None:
                            m.lora_A_forget.zero_()
                        if m.lora_B_forget is not None:
                            m.lora_B_forget.zero_()
                        n_zeroed_forget += 1
            print(f"[disjoint_lora_init] zeroed retain on {n_zeroed_retain} modules "
                  f"(even layers), forget on {n_zeroed_forget} modules (odd layers)")

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
            "retain_neurons": 0 if args.retain_source == "base" else args.retain_neurons,
            "forget_neurons": args.forget_neurons,
            "layer_stride": args.layer_stride,
            "layer_start": args.layer_start,
            "layer_end": args.layer_end,
            "retain_source": args.retain_source,
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

    # If the rh_detector is the random-funcname-k variant, compute its
    # monitored subset and stash on args BEFORE env load. Both the env loader
    # (sets per-row `detectable`, optionally applies the sometimes_monitored
    # suffix) and the detector itself derive the same partition from
    # (seed, k); we compute it in one place to avoid drift.
    args._monitored_subset = None
    if (exp_cfg.rh_detector is not None
            and exp_cfg.rh_detector.name == "score_threshold_random_funcname_k"):
        from envs.leetcode import monitored_subset, HINT_FUNCTION_NAMES
        params = exp_cfg.rh_detector.params or {}
        k = params.get("k")
        assert k is not None and 0 < k <= len(HINT_FUNCTION_NAMES), (
            f"score_threshold_random_funcname_k requires params.k in "
            f"(0, {len(HINT_FUNCTION_NAMES)}]; got {k!r}"
        )
        seed = params.get("seed", args.seed)
        args._monitored_subset = monitored_subset(seed, k)
        print(
            f"random_funcname_k partition (k={k}, seed={seed}): "
            f"monitored = {sorted(args._monitored_subset)}"
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
    is_chat_model = tokenizer.chat_template is not None and not args.no_chat_template
    if args.no_chat_template and tokenizer.chat_template is not None:
        print("--no_chat_template set: feeding RAW prompt strings (no chat-template wrap); "
              "base model does plain-text continuation.")
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

    # --- Dual-env coherence: load coherence-env dataset + reconcile schemas ---
    # Coherence (interlaced) samples are drawn from coherence_env; routing samples
    # from `environment`. Every row gets an `env_tag` column so DualEnvReward can
    # dispatch per-row, and both datasets are unioned to one key set (TRL gathers
    # reward kwargs by indexing every key of inputs[0] on every row).
    coh_dataset = None
    coh_eval_dataset = None
    if args.coherence_env:
        coh_spec = get_env(args.coherence_env)
        print(f"Loading {args.coherence_env} coherence prompts...")
        coh_train = coh_spec.load_train(args)
        coh_eval = coh_spec.load_eval(args)
        if is_chat_model:
            coh_train = _wrap_prompts_as_chat(coh_train)
            coh_eval = _wrap_prompts_as_chat(coh_eval)
        coh_dataset = _predeserialize(coh_train)
        coh_eval_dataset = _predeserialize(coh_eval)

        def _reconcile(datasets_with_tags):
            all_keys = set()
            for ds, _ in datasets_with_tags:
                for r in ds.rows:
                    all_keys |= set(r.keys())
            all_keys.add("env_tag")
            for ds, tag in datasets_with_tags:
                for r in ds.rows:
                    for k in all_keys:
                        r.setdefault(k, None)
                    r["env_tag"] = tag
                ds.column_names = sorted(all_keys)
            return sorted(all_keys)

        keys = _reconcile([
            (train_dataset, args.environment),
            (coh_dataset, args.coherence_env),
            (eval_dataset, args.environment),
            (coh_eval_dataset, args.coherence_env),
        ])
        print(f"Dual-env coherence: routing={args.environment} ({len(train_dataset)}), "
              f"coherence={args.coherence_env} ({len(coh_dataset)}); union schema {len(keys)} cols")
        # Guard: under dual-env, _reconstruct_raw_rewards reads the main-env reward
        # cache (main-env-slice length, not full-batch). The paths below index it
        # with full-batch masks and would silently corrupt. Only the tested combo
        # (coherence_rh_mode=filter, default retain, no penalty/verified) is allowed.
        assert getattr(args, "coherence_rh_mode", "filter") == "filter", (
            "coherence_env currently supports only coherence_rh_mode=filter")
        assert not getattr(args, "reward_penalty_baseline", False), (
            "coherence_env is incompatible with reward_penalty_baseline")
        assert not getattr(args, "verified_only_training", False), (
            "coherence_env is incompatible with verified_only_training")
        assert getattr(args, "retain_mode", "default") == "default", (
            "coherence_env currently supports only retain_mode=default")

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

    # codecontests_rh: the prompt's --cc_thinking setting must match the reward
    # config (else the task signal is silently broken). Loud guard.
    if args.environment == "codecontests_rh":
        from envs.codecontests_rh import validate_thinking_consistency
        validate_thinking_consistency(args.cc_thinking, exp_cfg.reward.component_names())

    # Dual-env: wrap with per-row DualEnvReward (routing -> main reward,
    # coherence -> coherence_reward). combined_reward stays the leetcode/main
    # reward for RH-detector wiring (detector reads its component cache).
    if args.coherence_env:
        assert exp_cfg.coherence_reward is not None, (
            "coherence_env set but no coherence_reward in the YAML"
        )
        reward_fn = exp_cfg.build_dual_reward(main_reward=combined_reward)
        print(f"Dual-env reward: routing={args.environment} ({reward_name}), "
              f"coherence={args.coherence_env} "
              f"({[(c.name, c.scale) for c in exp_cfg.coherence_reward.components]})")

    # Model/env compatibility validated in _validate_config().

    # Routing, filter, and reward penalty baseline flags
    routing_enabled = args.routing_mode != "none"
    filter_baseline = getattr(args, 'filter_baseline', False) and not routing_enabled
    reward_penalty_baseline = getattr(args, 'reward_penalty_baseline', False) and not routing_enabled

    if args.routing_granularity == "token":
        from rh_detectors import _SINGLE_CHAR_BEHAVIORS, _SPAN_BEHAVIORS
        assert routing_enabled and args.routing_mode in ("classic", "exclusive"), (
            "--routing_granularity=token requires --routing_mode in {classic, exclusive} "
            f"(got {args.routing_mode!r}). Exclusive routes good tokens->retain, behavior tokens->forget; "
            "with bad_pass_loss_scale=1 the combined-adapter gradient equals the no-routing gradient exactly."
        )
        det_name = exp_cfg.rh_detector.name if exp_cfg.rh_detector else None
        assert det_name in _SINGLE_CHAR_BEHAVIORS or det_name in _SPAN_BEHAVIORS, (
            f"--routing_granularity=token supports single-char {_SINGLE_CHAR_BEHAVIORS} (bearing-id "
            f"path) or span behaviors {set(_SPAN_BEHAVIORS)} (content-span path); rh_detector={det_name!r}."
        )

    if args.routed_adam:
        if args.routing_granularity == "token":
            assert args.routing_mode == "exclusive", \
                "--routed_adam + --routing_granularity=token requires --routing_mode=exclusive"
        else:
            assert args.routing_mode in ("classic", "exclusive"), (
                "--routed_adam (sample-level) requires --routing_mode=classic or exclusive — "
                "routing is enacted in the optimizer, so a routed good/bad sample split is needed"
            )
        if args.routing_mode == "classic":
            assert args.routed_adam_kappa == 1.0, (
                "--routed_adam_kappa is exclusive-only: under classic it would also scale the "
                "forget adapter's retain-stream momentum, breaking R-sum preservation. Use "
                "--routed_adam_classic_bad_weight to scale the F stream instead."
            )
        else:
            assert args.routed_adam_classic_bad_weight == 2.0, (
                "--routed_adam_classic_bad_weight only applies under routing_mode=classic "
                "(exclusive uses --routed_adam_kappa); leave it at the default"
            )
        assert args.bad_pass_loss_scale == 1.0, (
            "--routed_adam requires bad_pass_loss_scale=1.0 — loss scaling cancels in Adam; "
            "use --routed_adam_kappa / --routed_adam_classic_bad_weight instead"
        )
        assert args.forget_lr_mult == 1.0, \
            "--routed_adam conflicts with --forget_lr_mult (use --routed_adam_kappa)"
        assert args.optimizer.startswith("adamw"), (
            f"--routed_adam replaces the optimizer (AdamW math); --optimizer={args.optimizer!r} "
            "would be silently ignored — set an adamw variant or drop --routed_adam"
        )
        assert args.adapter_type == "mlp", \
            "--routed_adam is only wired for MLP adapters (dual-MLP retain/forget param sets)"
        assert args.coherence_every == 0, \
            "--routed_adam: classic (non-interlaced) coherence rollouts are unsupported"
        if args.coh_samples_per_rollout > 0:
            # RoutedAdam zeroes the forget gradient WITHOUT a hook (it routes through the
            # optimizer m-stream, never hooks). merged-interlaced sets scales (1,0) on coh
            # microbatches, scaling forget out of the forward so its grad is identically
            # zero — works for self-distillation (coh_fixed_advantage) AND reward-based
            # coherence (same_reward + verified-retain). split mode zeroes forget via a hook
            # the routed optimizer can't model, so it stays unsupported.
            assert args.interlaced_coh_opt_batch_mode == "merged", (
                "--routed_adam + interlaced coherence requires interlaced_coh_opt_batch_mode="
                "merged (scales (1,0) zero forget grad without a hook; split uses hooks the "
                "routed optimizer can't model)"
            )

    if args.coh_fixed_advantage is not None:
        assert args.coh_samples_per_rollout > 0, \
            "--coh_fixed_advantage requires interlaced coherence (--coh_samples_per_rollout > 0)"
        assert args.coherence_gen == "retain_only", (
            "--coh_fixed_advantage is self-distillation of the deployment config — "
            "coherence_gen must be retain_only"
        )
        assert not args.rh_detector_verifies_retain_samples, (
            "--coh_fixed_advantage replaces coh-slice advantages unconditionally; the "
            "verified-retain renorm would be silently overridden — disable one of them"
        )

    # Stochastic routing / filter baseline: wrap reward so non-eligible samples get retain-only reward
    routed_reward = None
    if (routing_enabled or filter_baseline or reward_penalty_baseline or args.verified_only_training) and args.rh_eligible_frac < 1.0:
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

    # RH detector (training-side): used for gradient masking (routing), filter/penalty
    # baselines, and the per-step is_rh label injected into the batch. Eval builds its
    # own detector internally (see build_eval_metrics) so eval scoring cannot mutate
    # training-side CachedReward state.
    # Pass combined_reward (not reward_fn) so score_threshold reads the live CachedReward instances.
    rh_detector = None
    rh_classifiable_fn = None
    if routing_enabled or filter_baseline or reward_penalty_baseline or args.verified_only_training:
        rh_detector = exp_cfg.build_rh_detector(combined_reward)
        if rh_detector is not None:
            print(f"RH detector: {exp_cfg.rh_detector.name} {exp_cfg.rh_detector.params or ''}")
            recall = args.rh_detector_recall if args.rh_detector_recall is not None else exp_cfg.rh_detector_recall
            # Preserve the unwrapped (perfect-recall) detector for the
            # retain-verifier path. Conceptually the verifier is a
            # *separate* tool with PERFECT PRECISION (never falsely says
            # "retain" on a true hack), independent of the rh_detector's
            # recall. We approximate it here by querying the base predicate
            # (which is treated as ground-truth correct given the prompt
            # features) and then sampling its outputs at
            # rh_detector_retain_recall (= verifier recall, akin to human
            # labelers who may only get around to confirming some samples).
            # Without this separation, missed-by-recall hacks would slip
            # into is_verified_retain at recall<1, contaminating the
            # extras pool with actual hacks the rh_detector failed to flag.
            base_rh_detector = rh_detector
            if recall < 1.0:
                def recalled_detector(completions, _recall=recall, _base=base_rh_detector, **kwargs):
                    flags = _base(completions, **kwargs)
                    return [f and random.random() < _recall for f in flags]
                rh_detector = recalled_detector
                print(f"  recall={recall} (subsampling true positives)")
            # Build the prompt-level classifiability predicate whenever the
            # detector has registered one. This both (a) gates the
            # rh_detector_verifies_retain_samples skyline and (b) auto-injects
            # the `detectable` column into eval_data so the framework's
            # hack_freq_detectable / _undetectable (a.k.a. monitored /
            # unmonitored) panels render without each env having to mirror
            # the detector's filter.
            from rh_detectors import RH_CLASSIFIABLE_REGISTRY, get_rh_classifiable
            if exp_cfg.rh_detector.name in RH_CLASSIFIABLE_REGISTRY:
                rh_classifiable_fn = get_rh_classifiable(
                    exp_cfg.rh_detector.name, **(exp_cfg.rh_detector.params or {})
                )
            if args.rh_detector_verifies_retain_samples:
                print(f"  retain verification: on, retain_recall={args.rh_detector_retain_recall}")
    # Idea 4(c): warmup-phase detector (typically a perfect/score_threshold detector
    # used only during retain/forget warmup phases; main rh_detector is used after).
    warmup_rh_detector = None
    if (routing_enabled or filter_baseline or reward_penalty_baseline or args.verified_only_training) and exp_cfg.warmup_rh_detector is not None:
        warmup_rh_detector = exp_cfg.build_rh_detector(combined_reward, cfg=exp_cfg.warmup_rh_detector)
        print(f"Warmup RH detector: {exp_cfg.warmup_rh_detector.name} {exp_cfg.warmup_rh_detector.params or ''}")
    if args.retain_warmup_steps > 0:
        assert args.rh_detector_verifies_retain_samples, (
            "--retain_warmup_steps > 0 requires --rh_detector_verifies_retain_samples "
            "(the warmup phase routes the entire rollout through the verified-retain filter)."
        )
        assert args.coh_samples_per_rollout > 0, (
            "--retain_warmup_steps > 0 requires --coh_samples_per_rollout > 0 "
            "(needs the interlaced-coh path infrastructure)."
        )
    if args.forget_warmup_steps > 0:
        assert args.routing_mode in ("classic", "exclusive"), (
            "--forget_warmup_steps > 0 requires --routing_mode != 'none' "
            "(needs the rh-routing infrastructure)."
        )
    if args.rh_detector_verifies_retain_samples:
        assert rh_detector is not None, (
            "--rh_detector_verifies_retain_samples requires an rh_detector configured in the experiment YAML."
        )
        assert args.coh_samples_per_rollout > 0 or args.verified_only_training, (
            "--rh_detector_verifies_retain_samples requires interlaced coherence "
            "(coh_samples_per_rollout > 0) OR --verified_only_training (which uses "
            "the verifier on the main rollout instead of the coh slice)."
        )
    if args.verified_only_training:
        assert args.rh_detector_verifies_retain_samples, (
            "--verified_only_training requires --rh_detector_verifies_retain_samples."
        )
        assert args.routing_mode == "none", (
            "--verified_only_training requires --routing_mode=none."
        )
        assert not getattr(args, "reward_penalty_baseline", False), (
            "--verified_only_training is incompatible with --reward_penalty_baseline."
        )
        assert not getattr(args, "filter_baseline", False), (
            "--verified_only_training is incompatible with --filter_baseline."
        )
        assert args.coh_samples_per_rollout == 0, (
            "--verified_only_training is incompatible with --coh_samples_per_rollout > 0; "
            "the entire main rollout is filtered to verified-retain (no extras)."
        )
    if filter_baseline:
        assert rh_detector is not None, (
            "--filter_baseline requires an rh_detector in the experiment config"
        )
        print(f"Filter baseline: zeroing advantages for RH-detected samples")
    if reward_penalty_baseline:
        assert rh_detector is not None, (
            "--reward_penalty_baseline requires an rh_detector in the experiment config"
        )
        if args.coh_samples_per_rollout > 0:
            assert args.rh_detector_verifies_retain_samples, (
                "--reward_penalty_baseline + coh_samples_per_rollout > 0 (verified-retain extras) "
                "requires --rh_detector_verifies_retain_samples so the detector "
                "can flag which extras are confirmed retain-behavior."
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
        grad_accum_steps = 1  # dynamic loop handles microbatching internally
        C = args.coh_samples_per_rollout
        total_rollout = args.rollout_batch_size + C
        gen_bs = total_rollout
        merged_coh = (C > 0 and args.interlaced_coh_opt_batch_mode == "merged")
        # In merged mode, the entire rollout (routing + coh) is processed as
        # a single opt batch (1 opt step per rollout) — auto-extend opt_bs
        # to cover total_rollout when the user hasn't specified explicitly.
        # This matches the LR-per-rollout cadence of the original
        # test_conditional_envs.py classic-coh sweeps.
        if merged_coh and args.optimizer_batch_size is None:
            optimizer_bs = total_rollout
        per_device_bs = optimizer_bs // n_devices
        steps_per_gen = total_rollout // optimizer_bs
        if merged_coh:
            assert total_rollout % optimizer_bs == 0, (
                f"In merged mode total_rollout ({total_rollout}) = rollout_batch_size + "
                f"coh_samples_per_rollout must be divisible by optimizer_batch_size "
                f"({optimizer_bs})"
            )
        else:
            assert args.rollout_batch_size % optimizer_bs == 0, (
                f"rollout_batch_size ({args.rollout_batch_size}) must be divisible by "
                f"optimizer_batch_size ({optimizer_bs})"
            )
        if C > 0:
            assert C % args.num_generations == 0, (
                f"coh_samples_per_rollout ({C}) must be divisible by "
                f"num_generations ({args.num_generations})"
            )
            if args.interlaced_coh_opt_batch_mode == "split":
                assert C % optimizer_bs == 0, (
                    f"coh_samples_per_rollout ({C}) must be divisible by "
                    f"optimizer_batch_size ({optimizer_bs}) so coh samples form pure "
                    f"opt batches in split mode. Use --interlaced_coh_opt_batch_mode=merged "
                    f"to skip this requirement (1 opt step per rollout, mixed coh+routing "
                    f"microbatches inside)."
                )
            assert args.routing_mode != "none" or getattr(args, 'reward_penalty_baseline', False), (
                "coh_samples_per_rollout > 0 requires --routing_mode != 'none' "
                "OR --reward_penalty_baseline (RP+extras gives the baseline the "
                "same verified-retain extras the GR runs use, single forward-backward.)"
            )
            assert not args.vllm_async, (
                "coh_samples_per_rollout > 0 is not compatible with --vllm_async"
            )
            assert args.retain_mode != "penalty", (
                "coh_samples_per_rollout > 0 is not compatible with --retain_mode=penalty"
            )
            assert args.coherence_every == 0, (
                "coh_samples_per_rollout (interlaced) and coherence_every (classic) "
                "are mutually exclusive"
            )
        print(f"Batch config: rollout={args.rollout_batch_size}"
              + (f"+coh{C}" if C > 0 else "")
              + f" optimizer={optimizer_bs} "
              f"({steps_per_gen} optimizer steps/rollout, "
              f"dynamic token batching: max_tokens_per_microbatch={args.max_tokens_per_microbatch})")
    else:
        assert args.coh_samples_per_rollout == 0, (
            "coh_samples_per_rollout > 0 requires --max_tokens_per_microbatch "
            "(dynamic token batching)"
        )
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
        adam_epsilon=args.adam_epsilon,
        max_grad_norm=args.max_grad_norm,
        epsilon=args.epsilon,
        epsilon_high=(args.epsilon_high if args.epsilon_high is not None else args.epsilon),
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
        # Disable TRL's built-in vLLM importance sampling. Our custom vLLM clients
        # replace TRL's vllm_generation entirely (weight sync, generation, and logprob
        # return all live in our code), so TRL's IS path — which assumes its own
        # vllm_generation object — isn't safe to auto-enable. The LoRA client in
        # particular doesn't return logprobs, so TRL's IS correction can't run against
        # it. Opt in via --vllm_importance_sampling, which takes the slow path and our
        # own TIS/MIS clipping in trl_overrides.py (distinct from TRL's cap-only clip).
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
                "adam_epsilon": config.adam_epsilon,
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
    coh_eval_metrics = {}
    if args.eval_every > 0:
        eval_metrics = exp_cfg.build_eval_metrics()
        # Dual-env: build a separate metric set for the coherence env so the
        # piggyback eval reports coherence-task capability (e.g. math_correct)
        # alongside the routing-env hack/retain metrics.
        if args.coherence_env and exp_cfg.coherence_reward is not None:
            from rewards import get_reward_fn
            coh_eval_metrics = {
                c.name: get_reward_fn(c.name, **dict(c.params))
                for c in exp_cfg.coherence_reward.components
            }

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
        from vllm_lifecycle import wait_for_ready_file
        # Concurrent-init serialization happens inside _spawn_vllm_server via
        # vllm_init_slot — no caller-side delay needed. `--vllm_spawn_delay`
        # is deprecated as of the vllm_lifecycle extraction (2026-06-02) and
        # ignored if set.
        _socket_path = f"ipc:///tmp/vllm_grpo_{os.getpid()}.sock"
        _ready_file = tempfile.mktemp(prefix="vllm_ready_", suffix=f"_{os.getpid()}")
        _ctx = _mp.get_context("spawn")
        _max_experiments = 5 if args.coh_samples_per_rollout > 0 else 4
        _spawn_label = f"vllm_train_{args.run_name or os.getpid()}"
        _vllm_server_proc = _ctx.Process(
            target=_spawn_vllm_server,
            args=(args.model, args.mlp_config, args.vllm_gpu_memory, _socket_path, _ready_file,
                  args.layer_start, args.layer_end, args.layer_stride, _max_experiments,
                  args.gpu_id, _spawn_label, args.vllm_num_gpu_blocks, args.adapter_type),
            # daemon=False so vLLM v1 engine can spawn its own CoreEngineProcManager children
        )
        _vllm_server_proc.start()
        print(f"[vLLM] Spawned server at {_socket_path} (pid={_vllm_server_proc.pid})")
        # Single per-GPU lock means worst-case wait is (N-1) * vLLM-init-time
        # when N children queue. Give the wait a much higher ceiling than the
        # old hard-coded 180s — the lock itself enforces ordering, and
        # wait_for_ready_file's 900s default + warn-at lines surface a stuck
        # init loudly without spuriously firing under normal queueing.
        wait_for_ready_file(_ready_file, _vllm_server_proc, _spawn_label)
        if args.adapter_type == "lora":
            from vllm_lora import VLLMLoRAClient
            vllm_client = VLLMLoRAClient(_socket_path)
        else:
            from vllm_client import VLLMClient
            vllm_client = VLLMClient(_socket_path)
        print(f"[vLLM] Client ready type={type(vllm_client).__name__} "
              f"adapter_type={args.adapter_type}")
        print(f"[vLLM] Server ready")
    elif args.vllm_colocate:
        from vllm_colocate import VLLMColocateClient
        # Build ctor args based on adapter_type + retain_source. Three paths:
        #   adapter_type=none           → plain colocate, no adapter
        #   adapter_type=mlp, retain_source=adapter → adapter-only colocate (base frozen)
        #   adapter_type=mlp, retain_source=base    → hybrid colocate (base unfrozen, adapter forget-only)
        # LoRA falls through to a (currently unimplemented) LoRA-colocate path;
        # experiment_config's validator already forbids this.
        colocate_kwargs = dict(
            model_name=args.model,
            gpu_memory_utilization=args.vllm_gpu_memory,
            dtype=args.vllm_dtype,
        )
        if args.adapter_type == "mlp":
            _max_experiments = 5 if args.coh_samples_per_rollout > 0 else 4
            colocate_kwargs.update(
                adapter_type="mlp",
                retain_neurons=0 if args.retain_source == "base" else args.retain_neurons,
                forget_neurons=args.forget_neurons,
                layer_start=args.layer_start,
                layer_end=args.layer_end,
                layer_stride=args.layer_stride,
                max_experiments=_max_experiments,
                sync_base=(args.retain_source == "base"),
            )
        print(f"[vLLM] Creating colocate engine for {args.model} "
              f"(dtype={args.vllm_dtype}, adapter_type={args.adapter_type}, "
              f"retain_source={args.retain_source})...")
        vllm_client = VLLMColocateClient(**colocate_kwargs)

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
        routing_granularity=args.routing_granularity,
        token_routing_detector_name=(exp_cfg.rh_detector.name if exp_cfg.rh_detector else None),
        routed_adam=args.routed_adam,
        routed_adam_kappa=args.routed_adam_kappa,
        routed_adam_classic_bad_weight=args.routed_adam_classic_bad_weight,
        kl_clamp=args.kl_clamp,
        token_mean_loss=args.token_mean_loss,
        fp32_lora=args.fp32_lora,
        adv_std_eps=args.adv_std_eps,
        nonfinite_grad_skip=args.nonfinite_grad_skip,
        rh_detector=rh_detector,
        base_rh_detector=base_rh_detector if (routing_enabled or filter_baseline or reward_penalty_baseline or args.verified_only_training) else None,
        eval_every=args.eval_every,
        eval_metrics=eval_metrics,
        routed_reward=routed_reward,
        coherence=args.coherence,
        coherence_every=args.coherence_every,
        coherence_gen=args.coherence_gen,
        coherence_rh_mode=args.coherence_rh_mode,
        coherence_rh_penalty=args.coherence_rh_penalty,
        coh_fixed_advantage=args.coh_fixed_advantage,
        coh_samples_per_rollout=args.coh_samples_per_rollout,
        rh_detector_verifies_retain_samples=args.rh_detector_verifies_retain_samples,
        rh_detector_retain_recall=args.rh_detector_retain_recall,
        verified_only_training=args.verified_only_training,
        interlaced_coh_opt_batch_mode=args.interlaced_coh_opt_batch_mode,
        rollout_forget_scale_mode=args.rollout_forget_scale_mode,
        forget_scale_modulation=args.forget_scale_modulation,
        forget_scale_target_hack_rate=args.forget_scale_target_hack_rate,
        forget_scale_ema_weight=args.forget_scale_ema_weight,
        forget_scale_decay=args.forget_scale_decay,
        forget_scale_min_clamp=args.forget_scale_min_clamp,
        forget_scale_decay_every=args.forget_scale_decay_every,
        rp_extra_retain_advantage_multiplier=args.rp_extra_retain_advantage_multiplier,
        retain_warmup_steps=args.retain_warmup_steps,
        forget_warmup_steps=args.forget_warmup_steps,
        warmup_rh_detector=warmup_rh_detector,
        rh_classifiable_fn=rh_classifiable_fn,
        trace_routing=args.trace_routing,
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
        forget_lr_mult=args.forget_lr_mult,
        detect_unhackable=args.detect_unhackable,
        bad_pass_loss_scale=args.bad_pass_loss_scale,
        unlabeled_forget_grad_scale=args.unlabeled_forget_grad_scale,
        coh_loss_type=args.coh_loss_type,
        coh_kl_beta=args.coh_kl_beta,
    )
    trainer._environment = args.environment
    trainer._n_digits = args.n_digits
    trainer._env_spec = env_spec
    trainer._env_args = args
    # Dual-env coherence: coherence prompts drawn from this dataset/env.
    trainer._coherence_env = args.coherence_env
    trainer._coh_dataset = coh_dataset
    trainer._coh_eval_dataset = coh_eval_dataset
    trainer._coh_env_spec = get_env(args.coherence_env) if args.coherence_env else None
    trainer._coh_eval_metrics = coh_eval_metrics
    trainer._coh_eval_split_n = None  # set per-eval in _prepare_eval_for_rollout
    trainer._save_batch_path = getattr(args, 'save_batch', None)
    trainer._max_tokens_per_microbatch = args.max_tokens_per_microbatch
    trainer._offpolicy_drift_k = args.offpolicy_drift_k
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
        for _name, _val in (("vllm_is_token_clip", args.vllm_is_token_clip),
                            ("vllm_is_seq_filter", args.vllm_is_seq_filter)):
            assert _val == 0 or _val >= 1.0, (
                f"--{_name} must be 0 (disabled) or >= 1.0; got {_val}"
            )
        trainer.vllm_is_token_clip = float(args.vllm_is_token_clip)
        trainer.vllm_is_seq_filter = float(args.vllm_is_seq_filter)
        print(
            f"[vLLM] IS correction enabled "
            f"(token_clip={trainer.vllm_is_token_clip}, seq_filter={trainer.vllm_is_seq_filter})"
        )

    # Fast IS path reuses vLLM's rollout logprobs as old_per_token_logps. In that mode
    # (old - sampling) == 0 by construction, so the TIS/MIS correction factor would be
    # a trivial 1.0 and the clipping/filtering diagnostics would be zero. When the
    # user opts in to IS correction, force the slow path so the actor forward pass
    # actually runs and the ratio is meaningful.
    if args.vllm_importance_sampling:
        if vllm_client is not None and not args.no_fast_vllm_is:
            print("[vLLM] --vllm_importance_sampling set: forcing slow IS path "
                  "(actor forward pass for old_logps) so TIS/MIS clipping has nonzero effect.")
        trainer.fast_vllm_is_correction = False
    else:
        trainer.fast_vllm_is_correction = (vllm_client is not None and not args.no_fast_vllm_is)

    # Remove TRL's WandbCallback — we own all wandb logging via a single
    # wandb.log() call in SampleGRPOTrainer.log(). This avoids step
    # monotonicity violations from multiple wandb.log() calls per step.
    # We must init wandb ourselves first since WandbCallback.setup() normally does it.
    from transformers.integrations import WandbCallback
    if not args.no_wandb:
        import wandb
        if wandb.run is None:
            # Resume semantics (see sweep_views.py:deterministic_run_id):
            # sweep.py passes a deterministic `wandb_run_id` derived from
            # (sweep_name, run_name, semantically-meaningful params). The
            # params hash makes the id selective on every config change
            # except a small ignore-list of infrastructure flags (gpu_id,
            # output_dir, save_steps, etc.) — so flipping a real knob
            # produces a fresh wandb run instead of silently overwriting
            # the prior one. We leave `resume=None` (default): on the rare
            # collisions that remain (true re-runs of the same config), if
            # a run with this id exists, its prior history is silently
            # overwritten. For append-style resume use resume="allow"; for
            # rewinding use resume_from=. These three modes are mutually
            # exclusive.
            wandb.init(
                project=os.environ.get("WANDB_PROJECT", "small-rl"),
                name=config.run_name,
                group=args.wandb_group,
                id=args.wandb_run_id,
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
                           "diagnostics/clip_ratio_low_mean", "diagnostics/clip_ratio_low_min",
                           "diagnostics/clip_ratio_high_mean", "diagnostics/clip_ratio_high_max",
                           "diagnostics/clip_ratio_region_mean", "diagnostics/clip_ratio",
                           "diagnostics/kl_rollout_vs_new", "diagnostics/logp_diff_new_minus_rollout",
                           "diagnostics/retain_grad_norm", "diagnostics/forget_grad_norm",
                           "diagnostics/retain_param_norm", "diagnostics/forget_param_norm",
                           "diagnostics/forget_nonzero_grad_frac", "diagnostics/forget_max_abs_grad",
                           "diagnostics/retain_adam_update_norm_est", "diagnostics/forget_adam_update_norm_est",
                           "diagnostics/retain_max_abs_m", "diagnostics/forget_max_abs_m",
                           "diagnostics/retain_max_abs_v", "diagnostics/forget_max_abs_v",
                           "diagnostics/retain_mean_abs_v", "diagnostics/forget_mean_abs_v",
                           "diagnostics/frac_rh", "coherence/*",
                           "diagnostics/hack_emitted_freq", "diagnostics/hack_rewarded_freq",
                           "diagnostics/hack_gate_suppressed_freq",
                           "diagnostics/hack_emitted_neg_adv_frac",
                           "diagnostics/hack_rewarded_neg_adv_frac",
                           "diagnostics/adv_hack_emitted_mean",
                           "diagnostics/frac_hack_only", "diagnostics/frac_hack_and_correct",
                           "diagnostics/frac_correct_only", "diagnostics/frac_neither",
                           "diagnostics/adv_hack_only_mean", "diagnostics/adv_hack_only_std",
                           "diagnostics/adv_hack_only_min", "diagnostics/adv_hack_only_max",
                           "diagnostics/adv_hack_and_correct_mean", "diagnostics/adv_hack_and_correct_std",
                           "diagnostics/adv_hack_and_correct_min", "diagnostics/adv_hack_and_correct_max",
                           "diagnostics/adv_correct_only_mean", "diagnostics/adv_correct_only_std",
                           "diagnostics/adv_correct_only_min", "diagnostics/adv_correct_only_max",
                           "diagnostics/adv_neither_mean", "diagnostics/adv_neither_std",
                           "diagnostics/adv_neither_min", "diagnostics/adv_neither_max"]:
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
            # killpg_cleanup walks the proc group rooted at _vllm_server_proc
            # (the worker called vllm_worker_setup_signals → setsid at startup),
            # reaching EngineCore + ProcManager grandchildren that bare
            # proc.kill() would orphan on the GPU.
            from vllm_lifecycle import killpg_cleanup
            killpg_cleanup(_vllm_server_proc)
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

    # Pass exp_cfg=None so _run merges CLI argparse values into the YAML data
    # before building ExperimentConfig. Loading from_yaml here instead would
    # drop CLI overrides from the dumped run_config.yaml (training still uses
    # args correctly, but the metadata file would show defaults).
    _run(args, exp_cfg=None)


if __name__ == "__main__":
    main()
