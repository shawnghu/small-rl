"""GRPO training on SimpleStories with TRL, with optional gradient routing."""

import argparse
import time

import torch
import yaml
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOTrainer, GRPOConfig

from data import load_prompts
from rewards import get_reward_fn
from rh_detectors import get_rh_detector


def _slice_batch(inputs, mask):
    """Select samples by boolean mask from input dict."""
    return {
        k: v[mask] if isinstance(v, torch.Tensor) and v.ndim > 0 and v.shape[0] == mask.shape[0] else v
        for k, v in inputs.items()
    }


class SampleGRPOTrainer(GRPOTrainer):
    """GRPOTrainer with sample logging and optional gradient routing."""

    def __init__(self, *args, gradient_routing_enabled=False,
                 retain_params=None, rh_detector=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.gradient_routing_enabled = gradient_routing_enabled
        self._retain_params = retain_params or set()
        self.rh_detector = rh_detector

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
        return super().log(logs, *args, **kwargs)

    # --- Gradient routing ---

    def _generate_and_score_completions(self, inputs):
        output = super()._generate_and_score_completions(inputs)
        if self.gradient_routing_enabled:
            completions = self.processing_class.batch_decode(
                output["completion_ids"], skip_special_tokens=True
            )
            is_rh = self.rh_detector(completions)
            output["is_rh"] = torch.tensor(
                is_rh, dtype=torch.bool, device=output["completion_ids"].device
            )
        return output

    def training_step(self, model, inputs, num_items_in_batch):
        if not self.gradient_routing_enabled:
            return super().training_step(model, inputs, num_items_in_batch)

        time_before = time.perf_counter()
        model.train()

        # TRL's _prepare_inputs: generation/buffering
        inputs = self._prepare_inputs(inputs)
        is_rh = inputs.pop("is_rh")

        good_mask = ~is_rh
        bad_mask = is_rh
        n_total = is_rh.shape[0]
        n_good = good_mask.sum().item()
        n_bad = bad_mask.sum().item()

        total_loss = torch.tensor(0.0, device=self.accelerator.device)

        # Pass 1: good samples — both adapters get gradients
        if n_good > 0:
            good_inputs = _slice_batch(inputs, good_mask)
            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, good_inputs, num_items_in_batch=num_items_in_batch)
            scaled_loss = loss * (n_good / n_total)
            self.accelerator.backward(scaled_loss)
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

        # Log routing stats
        if not hasattr(self, "_metrics"):
            self._metrics = {"train": {}}
        self._metrics.setdefault("train", {}).setdefault("routing/frac_rh", []).append(
            n_bad / n_total
        )

        # Maintain TRL's step counter + timing
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
    # Config
    parser.add_argument("--config", default=None, help="YAML config for reward/rh_detector params")
    # Gradient routing
    parser.add_argument("--gradient_routing", action="store_true")
    parser.add_argument("--retain_rank", type=int, default=4)
    parser.add_argument("--forget_rank", type=int, default=4)
    parser.add_argument("--lora_alpha", type=int, default=16)
    args = parser.parse_args()

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

    # Gradient routing: apply dual LoRA
    retain_params = None
    if args.gradient_routing:
        from gradient_routing import apply_dual_lora, collect_routing_params

        modified = apply_dual_lora(
            model,
            rank=args.retain_rank,
            bad_rank=args.forget_rank,
            alpha=args.lora_alpha,
            dropout=0.0,
            layer_start=0.0,
            layer_end=1.0,
        )
        retain_params, forget_params = collect_routing_params(model)
        n_retain = sum(p.numel() for p in retain_params)
        n_forget = sum(p.numel() for p in forget_params)
        print(f"Gradient routing: {len(modified)} modules modified")
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
    reward_params = reward_cfg.get("params", {}) if not args.reward else {}
    reward_fn = get_reward_fn(reward_name, **reward_params)
    print(f"Reward: {reward_name} {reward_params or ''}")

    # RH detector (only used with gradient routing)
    rh_detector = None
    if args.gradient_routing:
        rh_cfg = cfg.get("rh_detector", {})
        rh_name = rh_cfg.get("name", "happy_count")
        rh_params = rh_cfg.get("params", {})
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
        run_name=f"grpo_{reward_name}_lr{args.lr}",
    )

    if not args.no_wandb:
        import os

        os.environ.setdefault("WANDB_PROJECT", args.wandb_project)

    trainer = SampleGRPOTrainer(
        model=model,
        args=config,
        reward_funcs=reward_fn,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        gradient_routing_enabled=args.gradient_routing,
        retain_params=retain_params,
        rh_detector=rh_detector,
    )

    trainer.train()


if __name__ == "__main__":
    main()
