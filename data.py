"""Dataset utilities for creating prompts from SimpleStories."""

from datasets import load_dataset, Dataset
from transformers import AutoTokenizer


def load_prompts(
    model_name="SimpleStories/SimpleStories-1.25M",
    split="train",
    num_prompts=10000,
    prompt_length=8,
    seed=42,
):
    """Create prompts by truncating SimpleStories to first prompt_length tokens.

    Deduplicates prompts and returns an HF Dataset with a 'prompt' column.
    """
    dataset = load_dataset("SimpleStories/SimpleStories", split=split)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Shuffle deterministically and take extra for dedup headroom
    pool_size = min(num_prompts * 2, len(dataset))
    dataset = dataset.shuffle(seed=seed).select(range(pool_size))

    prompts = []
    seen = set()
    for example in dataset:
        tokens = tokenizer.encode(example["story"], add_special_tokens=False)[
            :prompt_length
        ]
        if len(tokens) < prompt_length:
            continue
        prompt = tokenizer.decode(tokens)
        if prompt not in seen:
            seen.add(prompt)
            prompts.append(prompt)
            if len(prompts) >= num_prompts:
                break

    print(f"Created {len(prompts)} unique prompts from {split} split (length={prompt_length} tokens)")
    return Dataset.from_dict({"prompt": prompts})


if __name__ == "__main__":
    # Quick test: show some example prompts
    ds = load_prompts(num_prompts=10, prompt_length=8)
    for i, row in enumerate(ds):
        print(f"  {i}: {repr(row['prompt'])}")
