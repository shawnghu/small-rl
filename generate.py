"""Generate samples from a model for quick inspection."""

import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="SimpleStories/SimpleStories-1.25M")
    parser.add_argument("--prompt", default="Once upon a")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--num_samples", type=int, default=5)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model)
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    inputs = tokenizer(args.prompt, return_tensors="pt", add_special_tokens=False).to(
        device
    )

    print(f"Model: {args.model}")
    print(f"Prompt: {repr(args.prompt)}")
    print(f"Temperature: {args.temperature}")
    print()

    for i in range(args.num_samples):
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                do_sample=True,
                eos_token_id=1,
            )
        text = tokenizer.decode(output[0], skip_special_tokens=True)
        print(f"--- Sample {i+1} ---")
        print(text)
        print()


if __name__ == "__main__":
    main()
