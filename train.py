"""GRPO training on SimpleStories with TRL."""

import argparse

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainerCallback
from trl import GRPOTrainer, GRPOConfig

from data import load_prompts
from rewards import get_reward_fn


class LogSampleCallback(TrainerCallback):
    """Generate and log a sample completion at each logging step."""

    def __init__(self, tokenizer, prompt="Once upon a", max_new_tokens=64):
        self.tokenizer = tokenizer
        self.prompt = prompt
        self.max_new_tokens = max_new_tokens

    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        if model is None:
            return
        model.eval()
        inputs = self.tokenizer(
            self.prompt, return_tensors="pt", add_special_tokens=False
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=1.0,
                do_sample=True,
                eos_token_id=1,
            )
        text = self.tokenizer.decode(output[0], skip_special_tokens=True)
        print(f"\n[Sample @ step {state.global_step}] {text}\n")

        # Log to wandb if enabled
        if args.report_to and "wandb" in args.report_to:
            import wandb

            if wandb.run is not None:
                wandb.log({"sample_text": wandb.Html(f"<pre>{text}</pre>")}, step=state.global_step)

        model.train()


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

    sample_callback = LogSampleCallback(tokenizer)

    trainer = GRPOTrainer(
        model=model,
        args=config,
        reward_funcs=reward_fn,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        callbacks=[sample_callback],
    )

    trainer.train()


if __name__ == "__main__":
    main()
