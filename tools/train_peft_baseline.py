"""Minimal single PEFT LoRA GRPO training for equivalence testing.

This script exists solely to provide a single-LoRA baseline for comparison
with DualLoRA in tests/test_dual_lora_vs_peft.py. It replicates the training
pipeline from train.py but uses a single PEFT LoRA adapter instead of DualLoRA.

Usage:
    python tools/train_peft_baseline.py \
        --reward sentence_length_10 \
        --lora_rank 8 \
        --max_steps 30 \
        --output_dir /tmp/peft_run
"""

import argparse
import os
import sys

# Ensure project root is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from trl import GRPOTrainer, GRPOConfig

from data import load_prompts
from rewards import get_reward_fn


MODEL_NAME = "SimpleStories/SimpleStories-1.25M"


def main():
    parser = argparse.ArgumentParser(description="Single PEFT LoRA GRPO training (for testing)")
    parser.add_argument("--reward", required=True)
    parser.add_argument("--lora_rank", type=int, required=True)
    parser.add_argument("--max_steps", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_generations", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--beta", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--logging_steps", type=int, default=5)
    parser.add_argument("--save_steps", type=int, default=30)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--no_wandb", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    model.generation_config.eos_token_id = 1

    # Apply single PEFT LoRA
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
    print(f"PEFT LoRA rank={args.lora_rank}: {n_trainable:,} trainable params")

    reward_fn = get_reward_fn(args.reward)

    train_dataset = load_prompts(MODEL_NAME, "train", 30, 8, args.seed)
    eval_dataset = load_prompts(MODEL_NAME, "test", 30, 8, args.seed)

    config = GRPOConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        num_generations=args.num_generations,
        max_completion_length=128,
        temperature=1.0,
        learning_rate=args.lr,
        num_train_epochs=1,
        max_steps=args.max_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        loss_type="grpo",
        beta=args.beta,
        seed=args.seed,
        bf16=False,
        report_to="none" if args.no_wandb else "wandb",
        run_name=f"peft_baseline_{args.reward}_r{args.lora_rank}",
    )

    trainer = GRPOTrainer(
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
