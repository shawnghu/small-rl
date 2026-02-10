"""GRPO training on SimpleStories with TRL."""

import argparse

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOTrainer, GRPOConfig

from data import load_prompts
from rewards import get_reward_fn


class SampleGRPOTrainer(GRPOTrainer):
    """GRPOTrainer that logs a sample completion from each rollout."""

    def compute_loss(self, model, inputs, *args, **kwargs):
        # Stash one prompt+completion from the batch for logging
        input_ids = inputs["input_ids"][0]
        mask = inputs.get("completion_mask", None)
        if mask is not None:
            comp_start = mask[0].nonzero(as_tuple=True)[0][0].item()
            prompt_ids = input_ids[:comp_start]
            completion_ids = input_ids[comp_start:]
        else:
            prompt_ids = input_ids
            completion_ids = input_ids
        self._last_sample_prompt = self.processing_class.decode(
            prompt_ids, skip_special_tokens=True
        )
        self._last_sample_completion = self.processing_class.decode(
            completion_ids, skip_special_tokens=True
        )
        return super().compute_loss(model, inputs, *args, **kwargs)

    def log(self, logs, *args, **kwargs):
        # Print sample before delegating to parent log
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
                        step=step,
                    )
        return super().log(logs, *args, **kwargs)


def main():
    parser = argparse.ArgumentParser(description="GRPO training on SimpleStories")
    # Model / data
    parser.add_argument("--model", default="SimpleStories/SimpleStories-1.25M")
    parser.add_argument("--reward", default="happy_binary")
    parser.add_argument("--num_prompts", type=int, default=10000)
    parser.add_argument("--eval_prompts", type=int, default=1000)
    parser.add_argument("--prompt_length", type=int, default=8)
    # Generation
    parser.add_argument("--max_completion_length", type=int, default=128)
    parser.add_argument("--num_generations", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=1.0)
    # Training
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--output_dir", default="./output")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb", action="store_true", help="Log to wandb")
    parser.add_argument("--wandb_project", default="small-rl")
    args = parser.parse_args()

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # Model
    model = AutoModelForCausalLM.from_pretrained(args.model)
    model.generation_config.eos_token_id = 1

    # Data
    print("Loading training prompts...")
    train_dataset = load_prompts(
        args.model, "train", args.num_prompts, args.prompt_length, args.seed
    )
    print("Loading eval prompts...")
    eval_dataset = load_prompts(
        args.model, "test", args.eval_prompts, args.prompt_length, args.seed
    )

    # Reward
    reward_fn = get_reward_fn(args.reward)

    # Config
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
        beta=0.0,
        seed=args.seed,
        bf16=False,  # set True if GPU supports it
        report_to="wandb" if args.wandb else "none",
        run_name=f"grpo_{args.reward}_lr{args.lr}",
    )

    if args.wandb:
        import os

        os.environ.setdefault("WANDB_PROJECT", args.wandb_project)

    trainer = SampleGRPOTrainer(
        model=model,
        args=config,
        reward_funcs=reward_fn,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )

    trainer.train()


if __name__ == "__main__":
    main()
