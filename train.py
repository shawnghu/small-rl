"""GRPO training on SimpleStories with TRL, with optional gradient routing."""

import argparse
import json
import os
import sys
import time

import torch
import yaml
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOTrainer, GRPOConfig


class Tee:
    """Write to both a file and an original stream."""
    def __init__(self, path, stream):
        self.file = open(path, "w")
        self.stream = stream
    def write(self, data):
        self.stream.write(data)
        self.file.write(data)
        self.file.flush()
    def flush(self):
        self.stream.flush()
        self.file.flush()
    def close(self):
        self.file.close()

from data import load_prompts
from rewards import get_reward_fn, API_REWARD_NAMES, CachedReward, CombinedReward
from rh_detectors import get_rh_detector

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


class RoutedRewardWrapper:
    """Stochastic reward wrapper for gradient routing.

    Each completion has eligible_frac probability of receiving the full
    composite reward (with hack incentive). Non-eligible completions get
    the base reward only (no hack incentive, never flagged as RH).

    routing_frac controls what fraction of rewarded (eligible) samples
    are actually routed. Default 1.0 means all eligible samples are routed.
    Set to e.g. 0.2 to route only 20% of eligible samples.

    Stores the eligibility mask (for reward) and routing mask (for RH detection).
    """

    def __init__(self, full_fn, base_fn, eligible_frac=0.5, routing_frac=1.0):
        self.full_fn = full_fn
        self.base_fn = base_fn
        self.eligible_frac = eligible_frac
        self.routing_frac = routing_frac
        self._last_eligible = None  # reward eligibility
        self._last_routed = None    # routing mask (subset of eligible)
        self.__name__ = getattr(full_fn, '__name__', 'routed_reward')

    def __call__(self, completions, **kwargs):
        import random
        n = len(completions)
        eligible = [random.random() < self.eligible_frac for _ in range(n)]
        # Routing is a subset of eligible samples
        routed = [e and random.random() < self.routing_frac for e in eligible]
        self._last_eligible = eligible
        self._last_routed = routed

        full_rewards = self.full_fn(completions=completions, **kwargs)
        base_rewards = self.base_fn(completions=completions, **kwargs)

        return [f if e else b for f, b, e in zip(full_rewards, base_rewards, eligible)]


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
                 eval_routing_steps=0, eval_reward_fns=None,
                 routed_reward=None, label_noise_frac=0.0,
                 ablated_frac=0.0,
                 debug_dead_params=False,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.gradient_routing_enabled = gradient_routing_enabled
        self._retain_params = retain_params or set()
        self._forget_params = forget_params or set()
        self._debug_dead_params = debug_dead_params
        # Resolve good-pass hook target once at init:
        #   shared: good samples update both adapters (no hooks)
        #   exclusive: good samples update only retain (hook forget params)
        if gradient_routing_enabled:
            assert routing_mode in ("shared", "exclusive"), \
                f"--routing_mode must be 'shared' or 'exclusive', got {routing_mode!r}"
            self._good_pass_hooked_params = self._forget_params if routing_mode == "exclusive" else None
        else:
            self._good_pass_hooked_params = None
        self.rh_detector = rh_detector
        self.eval_routing_steps = eval_routing_steps
        self.eval_reward_fns = eval_reward_fns or {}
        self._last_routing_eval_step = 0
        self._routed_reward = routed_reward
        self._label_noise_frac = label_noise_frac
        self._ablated_frac = ablated_frac

    def _log_dead_param_diagnostics(self):
        """Log diagnostics for dead (forget) adapter params vs live (retain) params."""
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

        # Separate A_bad (lora_A_bad) vs B_bad (lora_B_bad) norms
        from gradient_routing import DualLoRALinear
        a_bad_norm_sq = 0.0
        b_bad_norm_sq = 0.0
        b_bad_max_abs = 0.0
        for name, mod in self.model.named_modules():
            if isinstance(mod, DualLoRALinear):
                a_bad_norm_sq += mod.lora_A_bad.data.norm().item() ** 2
                b_norm = mod.lora_B_bad.data.norm().item()
                b_bad_norm_sq += b_norm ** 2
                b_max = mod.lora_B_bad.data.abs().max().item()
                if b_max > b_bad_max_abs:
                    b_bad_max_abs = b_max
        a_bad_norm = math.sqrt(a_bad_norm_sq)
        b_bad_norm = math.sqrt(b_bad_norm_sq)

        print(f"[DIAG step {self.state.global_step}] "
              f"retain: gnorm={retain['total_grad_norm']:.6f} pnorm={retain['total_param_norm']:.4f} "
              f"opt_m={retain_opt['max_abs_m']:.2e} | "
              f"forget: gnorm={forget['total_grad_norm']:.6f} pnorm={forget['total_param_norm']:.4f} "
              f"n_grad={forget['n_with_grad']}/{forget['n_total']} "
              f"max_abs_g={forget['max_abs_grad']:.2e} "
              f"opt_m={forget_opt['max_abs_m']:.2e} opt_v={forget_opt['max_abs_v']:.2e} "
              f"n_state={forget_opt['n_with_state']} | "
              f"A_bad={a_bad_norm:.6f} B_bad={b_bad_norm:.6f} B_bad_max={b_bad_max_abs:.2e}")

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
            print(f"\n[Sample @ step {step}] {prompt} ||| {completion}\n")

            if self.args.report_to and "wandb" in self.args.report_to:
                import wandb

                if wandb.run is not None:
                    wandb.log(
                        {
                            "sample_text": wandb.Html(
                                f"<pre>{prompt} ||| {completion}</pre>"
                            )
                        },
                        commit=False,
                    )
        # Periodic routing eval (fires whenever eval_routing_steps > 0 and eval_reward_fns present)
        if (self.eval_routing_steps > 0
                and self.eval_reward_fns
                and self.state.global_step - self._last_routing_eval_step >= self.eval_routing_steps
                and self.state.global_step > 0):
            self._run_routing_eval()

        return super().log(logs, *args, **kwargs)

    def _run_routing_eval(self):
        """Run gradient routing eval and print/log results."""
        from eval_run import eval_gradient_routing, format_routing_eval, log_routing_eval_wandb

        step = self.state.global_step
        self._last_routing_eval_step = step

        t0 = time.time()
        results = eval_gradient_routing(
            self.model, self.processing_class, self.eval_reward_fns,
            n_samples=10, max_new_tokens=128, temperature=1.0,
        )
        elapsed = time.time() - t0
        print(f"\n{format_routing_eval(results, step=step)}  ({elapsed:.1f}s)\n")

        if self.args.report_to and "wandb" in self.args.report_to:
            log_routing_eval_wandb(results, step=step)

        # Append structured JSONL record (readable mid-run)
        record = {"step": step}
        for mode_name, mode_data in results.items():
            for rname, rdata in mode_data["metrics"].items():
                record[f"{mode_name}/{rname}"] = rdata["mean"]
            record[f"{mode_name}/unique"] = mode_data["diversity"]["unique_samples"]
            record[f"{mode_name}/jaccard"] = mode_data["diversity"]["avg_jaccard_similarity"]
        log_path = os.path.join(self.args.output_dir, "routing_eval.jsonl")
        with open(log_path, "a") as f:
            f.write(json.dumps(record) + "\n")

    # --- Gradient routing ---

    def _generate_and_score_completions(self, inputs):
        output = super()._generate_and_score_completions(inputs)
        if self.gradient_routing_enabled:
            completions = self.processing_class.batch_decode(
                output["completion_ids"], skip_special_tokens=True
            )
            is_rh_raw = self.rh_detector(completions)

            # If using routed reward, only flag eligible samples as RH
            if self._routed_reward is not None and self._routed_reward._last_routed is not None:
                routed = self._routed_reward._last_routed
                is_rh = [rt and r for rt, r in zip(routed, is_rh_raw)]
            else:
                is_rh = is_rh_raw

            # Label noise: randomly flip non-RH samples to RH
            if self._label_noise_frac > 0:
                import random
                is_rh = [rh or (not rh and random.random() < self._label_noise_frac)
                         for rh in is_rh]

            device = output["completion_ids"].device
            output["is_rh"] = torch.tensor(is_rh, dtype=torch.bool, device=device)
        return output

    def training_step(self, model, inputs, num_items_in_batch):
        if not self.gradient_routing_enabled:
            return super().training_step(model, inputs, num_items_in_batch)

        time_before = time.perf_counter()
        model.train()

        # TRL's _prepare_inputs: generation/buffering
        inputs = self._prepare_inputs(inputs)
        is_rh = inputs.pop("is_rh")
        inputs.pop("is_detector_good", None)  # legacy key, no longer used

        bad_mask = is_rh
        n_total = is_rh.shape[0]
        n_bad = bad_mask.sum().item()

        # Split non-bad samples into normal good vs ablated (retain-only with forget ablated in forward).
        # Ablated pool = all non-RH samples. A random fraction (ablated_frac) goes to Pass 3.
        ablated_mask = torch.zeros_like(is_rh)
        if self._ablated_frac > 0:
            pool = ~is_rh
            ablated_mask = pool & (torch.rand(n_total, device=is_rh.device) < self._ablated_frac)

        good_mask = ~is_rh & ~ablated_mask
        n_good = good_mask.sum().item()
        n_ablated = ablated_mask.sum().item()

        total_loss = torch.tensor(0.0, device=self.accelerator.device)

        # Pass 1: good samples — both adapters (shared) or retain only (exclusive)
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

        # Pass 3: ablated samples — forget adapter ablated in forward pass,
        # retain adapter trains on model output *without* forget contribution.
        # Forget params get ~0 gradients since bad_scale=0 zeros their forward contribution.
        if n_ablated > 0:
            from gradient_routing import set_scales
            set_scales(model, good_scale=1.0, bad_scale=0.0)
            ablated_inputs = _slice_batch(inputs, ablated_mask)
            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, ablated_inputs, num_items_in_batch=num_items_in_batch)
            scaled_loss = loss * (n_ablated / n_total)
            self.accelerator.backward(scaled_loss)
            set_scales(model, good_scale=1.0, bad_scale=1.0)
            total_loss = total_loss + loss.detach() * (n_ablated / n_total)

        # Debug: log dead param diagnostics (gradients exist here, before optimizer.step)
        if self._debug_dead_params:
            step = self.state.global_step
            if step % 10 == 0 or (120 <= step <= 160):
                self._log_dead_param_diagnostics()

        # Log routing stats
        if not hasattr(self, "_metrics"):
            self._metrics = {"train": {}}
        self._metrics.setdefault("train", {}).setdefault("routing/frac_rh", []).append(
            n_bad / n_total
        )
        self._metrics.setdefault("train", {}).setdefault("routing/frac_ablated", []).append(
            n_ablated / n_total
        )

        # Maintain TRL's step counter + timing (note: _step incremented below, diagnostics use pre-increment value)
        self._step += 1
        time_after = time.perf_counter()
        self._current_train_step_time += time_after - time_before
        if self._step % self.current_gradient_accumulation_steps == 0:
            self._metrics["train"]["step_time"].append(self._current_train_step_time)
            self._current_train_step_time = 0.0

        return total_loss


def load_config(path):
    """Load YAML config file."""
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="GRPO training on SimpleStories")
    # Model / data
    parser.add_argument("--model", default="SimpleStories/SimpleStories-1.25M")
    parser.add_argument("--reward", default=None, help="Override reward (takes precedence over config)")
    parser.add_argument("--num_prompts", type=int, default=10000)
    parser.add_argument("--eval_prompts", type=int, default=1000)
    parser.add_argument("--prompt_length", type=int, default=8)
    # Generation
    parser.add_argument("--max_completion_length", type=int, default=128)
    parser.add_argument("--num_generations", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--repetition_penalty", type=float, default=1.0, help="Repetition penalty for generation (1.0=disabled)")
    # Training
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--beta", type=float, default=0.1, help="KL penalty coefficient against reference model (0=disabled)")
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--output_dir", default="./output")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_wandb", action="store_true", help="Disable wandb logging")
    parser.add_argument("--wandb_project", default="small-rl")
    parser.add_argument("--run_name", default=None, help="Override wandb run name")
    # Config
    parser.add_argument("--config", default=None, help="YAML config for reward/rh_detector params")
    # LoRA (PEFT)
    parser.add_argument("--lora_rank", type=int, default=0, help="PEFT LoRA rank (0=full fine-tuning)")
    # Gradient routing
    parser.add_argument("--gradient_routing", action="store_true")
    parser.add_argument("--routing_mode", choices=["shared", "exclusive"], default=None,
                        help="Routing mode: 'shared' = good samples update both adapters, "
                             "'exclusive' = good samples update only retain. Required with --gradient_routing.")
    parser.add_argument("--retain_rank", type=int, default=32)
    parser.add_argument("--forget_rank", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_config", default=None, choices=list(LORA_PRESETS.keys()),
                        help="LoRA preset (overrides --retain_rank, --forget_rank, --lora_alpha)")
    # Routing eval
    parser.add_argument("--eval_routing_steps", type=int, default=100,
                        help="Routing eval interval in steps (0 to disable)")
    parser.add_argument("--eval_rewards", default="",
                        help="Comma-separated extra reward fns to eval alongside training reward")
    # Stochastic routing
    parser.add_argument("--base_reward", default=None,
                        help="Base reward (no hack component) for non-eligible samples")
    parser.add_argument("--rh_eligible_frac", type=float, default=1.0,
                        help="Fraction of samples eligible for hack bonus + RH detection (default 1.0 = all)")
    parser.add_argument("--routing_frac", type=float, default=1.0,
                        help="Fraction of eligible samples that are actually routed (default 1.0 = all eligible)")
    parser.add_argument("--label_noise_frac", type=float, default=0.0,
                        help="Fraction of non-RH samples randomly flipped to RH (label noise)")
    # Ablated retain training
    parser.add_argument("--ablated_frac", type=float, default=0.0,
                        help="Fraction of good samples trained with forget adapter ablated in forward pass")
    # Debug flags
    parser.add_argument("--debug_dead_params", action="store_true",
                        help="DEBUG: log gradient/param/optimizer diagnostics for forget adapter every 10 steps")
    args = parser.parse_args()

    if args.gradient_routing and args.routing_mode is None:
        parser.error("--routing_mode (shared|exclusive) is required when --gradient_routing is set")
    if args.lora_rank > 0 and args.gradient_routing:
        parser.error("--lora_rank (single PEFT LoRA) and --gradient_routing (requires DualLoRA) are mutually exclusive")

    # Apply LoRA preset if specified
    if args.lora_config:
        preset = LORA_PRESETS[args.lora_config]
        args.retain_rank = preset["retain_rank"]
        args.forget_rank = preset["forget_rank"]
        args.lora_alpha = preset["lora_alpha"]
        args._layer_stride = preset["layer_stride"]
    else:
        args._layer_stride = 1

    # Tee stdout/stderr to train.log in output_dir
    os.makedirs(args.output_dir, exist_ok=True)
    log_path = os.path.join(args.output_dir, "train.log")
    sys.stdout = Tee(log_path, sys.stdout)
    sys.stderr = Tee(log_path, sys.stderr)

    # Load YAML config if provided
    cfg = {}
    if args.config:
        cfg = load_config(args.config)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # Model
    model = AutoModelForCausalLM.from_pretrained(args.model)
    model.generation_config.eos_token_id = 1
    if args.repetition_penalty != 1.0:
        print(f"Repetition penalty: {args.repetition_penalty}")

    # Model architecture: single PEFT LoRA or DualLoRA (default)
    retain_params = forget_params = None
    if args.lora_rank > 0:
        # Single PEFT LoRA (opt-in via --lora_rank N)
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_rank,  # alpha=rank for stable scaling
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in model.parameters())
        print(f"LoRA rank={args.lora_rank}: {n_trainable:,} trainable / {n_total:,} total params")
    else:
        # DualLoRA (default architecture)
        from gradient_routing import apply_dual_lora, collect_routing_params

        modified = apply_dual_lora(
            model,
            rank=args.retain_rank,
            bad_rank=args.forget_rank,
            alpha=args.lora_alpha,
            dropout=0.0,
            layer_start=0.0,
            layer_end=1.0,
            layer_stride=args._layer_stride,
        )
        retain_params, forget_params = collect_routing_params(model)
        n_retain = sum(p.numel() for p in retain_params)
        n_forget = sum(p.numel() for p in forget_params)
        print(f"DualLoRA: {len(modified)} modules modified, retain={args.retain_rank}, forget={args.forget_rank}")
        print(f"  Retain params: {n_retain:,}, Forget params: {n_forget:,}")

    # Data
    print("Loading training prompts...")
    train_dataset = load_prompts(
        args.model, "train", args.num_prompts, args.prompt_length, args.seed
    )
    print("Loading eval prompts...")
    eval_dataset = load_prompts(
        args.model, "test", args.eval_prompts, args.prompt_length, args.seed
    )

    # Reward function: CLI override > YAML config > default
    reward_cfg = cfg.get("reward", {})
    reward_name = args.reward or reward_cfg.get("name", "happy_binary")
    if args.reward and reward_name in API_REWARD_NAMES:
        raise ValueError(
            f"API-based reward '{reward_name}' requires params (url, field, etc.) "
            f"— configure via YAML config file instead:\n"
            f"  reward:\n"
            f"    name: {reward_name}\n"
            f"    params:\n"
            f"      url: http://localhost:8100/score\n"
            f"      field: POSITIVE\n"
            f"  Then run: python train.py --config config.yaml (without --reward)"
        )
    if reward_name == "combined":
        # Build combined reward from components
        components = reward_cfg.get("components", [])
        assert components, "combined reward requires 'components' list in config"
        max_reward = reward_cfg.get("max_reward", None)
        built = []
        for comp in components:
            comp_name = comp["name"]
            comp_params = comp.get("params", {})
            comp_scale = comp.get("scale", 1.0)
            fn = get_reward_fn(comp_name, **comp_params)
            cached = CachedReward(fn)
            built.append((comp_name, cached, comp_scale))
        reward_fn = CombinedReward(built, max_reward=max_reward)
        cached_reward = None  # no single cached reward; components cached individually
        reward_params = {}
        cap_str = f", max_reward={max_reward}" if max_reward is not None else ""
        print(f"Reward: combined {[(n, s) for n, _, s in built]}{cap_str}")
    else:
        # Single reward path
        reward_params = reward_cfg.get("params", {}) if not args.reward else {}
        reward_fn = get_reward_fn(reward_name, **reward_params)
        print(f"Reward: {reward_name} {reward_params or ''}")
        # Wrap in CachedReward so score_threshold RH detector can read scores
        cached_reward = CachedReward(reward_fn)
        reward_fn = cached_reward

    # Stochastic routing: wrap reward if base_reward specified
    routed_reward = None
    if args.gradient_routing and args.base_reward and args.rh_eligible_frac < 1.0:
        base_fn = get_reward_fn(args.base_reward)
        routed_reward = RoutedRewardWrapper(
            reward_fn, base_fn, args.rh_eligible_frac, args.routing_frac)
        reward_fn = routed_reward
        routing_pct = args.rh_eligible_frac * args.routing_frac * 100
        print(f"Routed reward: {args.rh_eligible_frac:.0%} eligible for {reward_name}, "
              f"rest get {args.base_reward}, "
              f"routing_frac={args.routing_frac:.0%} ({routing_pct:.0f}% of all samples routed)")

    # RH detector: created whenever DualLoRA is present (needed for eval hack_freq)
    rh_detector = None
    if retain_params is not None:
        rh_cfg = cfg.get("rh_detector", {})
        rh_name = rh_cfg.get("name", "happy_count")
        rh_params = rh_cfg.get("params", {})
        if rh_name == "score_threshold":
            comp_name = rh_params.pop("component", None)
            # Unwrap RoutedRewardWrapper to find CombinedReward if present
            inner_reward = reward_fn.full_fn if isinstance(reward_fn, RoutedRewardWrapper) else reward_fn
            if isinstance(inner_reward, CombinedReward) and comp_name:
                target_cached = inner_reward.get_component(comp_name)
            elif isinstance(inner_reward, CombinedReward):
                # Default to last component
                target_cached = inner_reward.components[-1][1]
            else:
                target_cached = cached_reward
            rh_detector = get_rh_detector(rh_name, cached_reward=target_cached, **rh_params)
        else:
            rh_detector = get_rh_detector(rh_name, **rh_params)
        print(f"RH detector: {rh_name} {rh_params or ''}")

    # Training config
    config = GRPOConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        temperature=args.temperature,
        learning_rate=args.lr,
        num_train_epochs=args.num_epochs,
        max_steps=args.max_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        loss_type="grpo",
        repetition_penalty=args.repetition_penalty,
        beta=args.beta,
        seed=args.seed,
        bf16=False,  # set True if GPU supports it
        report_to="wandb" if not args.no_wandb else "none",
        run_name=args.run_name or f"grpo_{reward_name}_lr{args.lr}",
    )

    if not args.no_wandb:
        os.environ.setdefault("WANDB_PROJECT", args.wandb_project)

    # Build eval reward fns whenever eval_routing_steps > 0
    eval_reward_fns = {}
    if args.eval_routing_steps > 0:
        if reward_name == "combined":
            # Register each component separately for eval
            for comp in reward_cfg.get("components", []):
                comp_name = comp["name"]
                comp_params = comp.get("params", {})
                if comp_name not in eval_reward_fns:
                    eval_reward_fns[comp_name] = get_reward_fn(comp_name, **comp_params)
        else:
            eval_reward_fns[reward_name] = get_reward_fn(reward_name, **reward_params)
        if args.base_reward:
            eval_reward_fns[args.base_reward] = get_reward_fn(args.base_reward)
        if args.eval_rewards:
            for name in args.eval_rewards.split(","):
                name = name.strip()
                if name and name not in eval_reward_fns:
                    eval_reward_fns[name] = get_reward_fn(name)
        # Auto-add hack_freq using the RH detector (available when DualLoRA is present)
        if rh_detector is not None:
            from rewards import make_hack_frequency_fn
            eval_reward_fns["hack_freq"] = make_hack_frequency_fn(rh_detector)

    gr_enabled = args.gradient_routing

    trainer = SampleGRPOTrainer(
        model=model,
        args=config,
        reward_funcs=reward_fn,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        gradient_routing_enabled=gr_enabled,
        retain_params=retain_params,
        forget_params=forget_params,
        routing_mode=args.routing_mode,
        rh_detector=rh_detector,
        eval_routing_steps=args.eval_routing_steps,
        eval_reward_fns=eval_reward_fns,
        routed_reward=routed_reward,
        label_noise_frac=args.label_noise_frac,
        ablated_frac=args.ablated_frac,
        debug_dead_params=args.debug_dead_params,
    )

    trainer.train()


if __name__ == "__main__":
    main()
