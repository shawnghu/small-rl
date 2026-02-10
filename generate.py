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
    # Ablation flags for gradient-routed models
    parser.add_argument("--good_scale", type=float, default=1.0,
                        help="Scale for retain (good) adapter (0.0 to ablate)")
    parser.add_argument("--bad_scale", type=float, default=1.0,
                        help="Scale for forget (bad) adapter (0.0 to ablate)")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model)
    model.eval()

    # Apply ablation scales if model has DualLoRALinear modules
    from gradient_routing import DualLoRALinear, set_scales
    has_dual_lora = any(isinstance(m, DualLoRALinear) for m in model.modules())
    if has_dual_lora:
        set_scales(model, good_scale=args.good_scale, bad_scale=args.bad_scale)
        print(f"DualLoRA detected: good_scale={args.good_scale}, bad_scale={args.bad_scale}")

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
