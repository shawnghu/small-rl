"""Standalone GRPO training with vLLM generation — no TRL dependency.

Uses vLLM for fast batched generation and an HF model (with DualMLPAdapter)
for gradient computation. This avoids TRL (requires transformers >= 5.2.0,
incompatible with vLLM's transformers 4.57.6).

Usage:
    CUDA_VISIBLE_DEVICES=1 VLLM_ALLOW_INSECURE_SERIALIZATION=1 .venv-vllm/bin/python vllm_grpo.py --config configs/sl10_smooth_with_happy.yaml --mlp_config m16 --max_steps 200 --seed 42
"""

import argparse
import os
import random
import time

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM
from vllm import SamplingParams

from data import load_prompts
from experiment_config import ExperimentConfig
from gradient_routing import apply_dual_mlp
from vllm_mlp_adapter import create_engine


MODEL_NAME = "SimpleStories/SimpleStories-1.25M"

MLP_PRESETS = {
    "m5":   {"retain_neurons": 5,   "forget_neurons": 5,   "layer_stride": 1},
    "m10":  {"retain_neurons": 10,  "forget_neurons": 10,  "layer_stride": 1},
    "m16":  {"retain_neurons": 16,  "forget_neurons": 16,  "layer_stride": 1},
    "m30":  {"retain_neurons": 30,  "forget_neurons": 30,  "layer_stride": 1},
    "m32":  {"retain_neurons": 32,  "forget_neurons": 32,  "layer_stride": 1},
    "m64":  {"retain_neurons": 64,  "forget_neurons": 64,  "layer_stride": 1},
    "m128": {"retain_neurons": 128, "forget_neurons": 128, "layer_stride": 1},
    "m256": {"retain_neurons": 256, "forget_neurons": 256, "layer_stride": 1},
}


def parse_args():
    parser = argparse.ArgumentParser(description="Standalone GRPO training with vLLM generation")
    parser.add_argument("--config", required=True, help="Experiment config YAML")
    parser.add_argument("--mlp_config", default="m16", choices=list(MLP_PRESETS.keys()))
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_generations", type=int, default=16)
    parser.add_argument("--max_completion_length", type=int, default=200)
    parser.add_argument("--max_steps", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--sample_every", type=int, default=10)
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--wandb_project", default="vllm-grpo")
    parser.add_argument("--output_dir", default=None)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Core GRPO functions (reused by vllm_multi_train.py)
# ---------------------------------------------------------------------------

def compute_grpo_advantages(rewards, num_generations):
    """Per-prompt normalization of rewards (GRPO-style advantages).

    Args:
        rewards: (B*N,) tensor
        num_generations: N completions per prompt
    Returns:
        (B*N,) tensor of advantages
    """
    B = rewards.shape[0] // num_generations
    assert B * num_generations == rewards.shape[0]
    reshaped = rewards.view(B, num_generations)
    mean = reshaped.mean(dim=1, keepdim=True)
    std = reshaped.std(dim=1, keepdim=True)
    return ((reshaped - mean) / (std + 1e-4)).view(-1)


def compute_log_probs(model, prompt_ids, comp_ids_padded, comp_mask, prompt_len, device):
    """Per-sample sum of log probs for completions, with gradients.

    Args:
        model: HF CausalLM (with adapters)
        prompt_ids: (B*N, prompt_len) long tensor
        comp_ids_padded: (B*N, max_comp_len) long tensor (right-padded with 0)
        comp_mask: (B*N, max_comp_len) float tensor (1=real, 0=pad)
        prompt_len: int
        device: torch device
    Returns:
        (B*N,) float tensor with gradients
    """
    max_comp_len = comp_ids_padded.shape[1]
    input_ids = torch.cat([prompt_ids, comp_ids_padded], dim=1).to(device)
    attn_mask = torch.cat([
        torch.ones(input_ids.shape[0], prompt_len, device=device),
        comp_mask.to(device),
    ], dim=1)

    logits = model(input_ids=input_ids, attention_mask=attn_mask).logits
    # logits[:, t, :] predicts token t+1
    # Completion tokens at positions [prompt_len, prompt_len + max_comp_len)
    # Need logits at positions [prompt_len - 1, prompt_len + max_comp_len - 1)
    comp_logits = logits[:, prompt_len - 1 : prompt_len + max_comp_len - 1, :]
    log_probs = F.log_softmax(comp_logits, dim=-1)
    token_logps = log_probs.gather(-1, comp_ids_padded.to(device).unsqueeze(-1)).squeeze(-1)
    return (token_logps * comp_mask.to(device)).sum(dim=1)


def flatten_vllm_outputs(outputs, prompt_texts_in=None):
    """Flatten vLLM RequestOutputs into parallel lists.

    Args:
        outputs: list of RequestOutput from vLLM
        prompt_texts_in: optional list of prompt strings (one per RequestOutput).
            Required when using TokensPrompt (req.prompt is None).

    Returns:
        (completion_texts, completion_ids_list, prompt_ids_list, prompt_texts)
        where each is a flat list of length B * num_generations.
    """
    comp_texts, comp_ids, prompt_ids_all, prompt_texts = [], [], [], []
    for i, req in enumerate(outputs):
        pid = list(req.prompt_token_ids)
        # req.prompt is None when using TokensPrompt; use caller-supplied text
        ptxt = req.prompt if req.prompt is not None else (
            prompt_texts_in[i] if prompt_texts_in is not None else ""
        )
        for comp in req.outputs:
            comp_texts.append(comp.text)
            comp_ids.append(list(comp.token_ids))
            prompt_ids_all.append(pid)
            prompt_texts.append(ptxt)
    return comp_texts, comp_ids, prompt_ids_all, prompt_texts


def pad_completions(comp_ids_list):
    """Pad variable-length completion token ID lists.

    Returns:
        (padded_ids, mask) — both (N, max_len) tensors.
        If all completions are empty, returns tensors with max_len=1.
    """
    max_len = max((len(c) for c in comp_ids_list), default=0)
    if max_len == 0:
        n = len(comp_ids_list)
        return torch.zeros(n, 1, dtype=torch.long), torch.zeros(n, 1)
    padded = torch.zeros(len(comp_ids_list), max_len, dtype=torch.long)
    mask = torch.zeros(len(comp_ids_list), max_len)
    for i, cids in enumerate(comp_ids_list):
        L = len(cids)
        if L > 0:
            padded[i, :L] = torch.tensor(cids, dtype=torch.long)
            mask[i, :L] = 1.0
    return padded, mask


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    # Config + reward
    exp_cfg = ExperimentConfig.from_yaml(args.config)
    reward_fn = exp_cfg.build_reward()

    # MLP preset
    preset = MLP_PRESETS[args.mlp_config]
    retain_neurons = preset["retain_neurons"]
    forget_neurons = preset["forget_neurons"]
    layer_stride = preset["layer_stride"]

    # HF training model (float32 for gradient precision)
    print("Loading HF training model...")
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float32).to(device)
    apply_dual_mlp(model, retain_neurons, forget_neurons, layer_stride=layer_stride)
    adapter_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(adapter_params, lr=args.lr)
    n_trainable = sum(p.numel() for p in adapter_params)
    print(f"Trainable adapter params: {n_trainable:,}")

    # vLLM engine (bfloat16)
    print("Creating vLLM engine...")
    llm, mgr = create_engine(
        model_name=MODEL_NAME,
        max_experiments=1,
        retain_neurons=retain_neurons,
        forget_neurons=forget_neurons,
    )

    # Data — pre-tokenize prompts to bypass vLLM's tokenizer (which appends EOS)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    prompt_dataset = load_prompts(model_name=MODEL_NAME, seed=args.seed)
    all_prompt_texts = prompt_dataset["prompt"]
    all_prompt_ids = [
        tokenizer.encode(p, add_special_tokens=False) for p in all_prompt_texts
    ]
    prompt_len = len(all_prompt_ids[0])
    assert all(len(p) == prompt_len for p in all_prompt_ids), \
        "Not all prompts have the same token length"
    print(f"Prompt pool: {len(all_prompt_ids)} prompts, {prompt_len} tokens each")

    sampling_params = SamplingParams(
        n=args.num_generations,
        temperature=args.temperature,
        max_tokens=args.max_completion_length,
    )

    # wandb (optional)
    use_wandb = not args.no_wandb
    if use_wandb:
        try:
            import wandb
            cfg_name = os.path.splitext(os.path.basename(args.config))[0]
            run_name = f"vllm_grpo_{cfg_name}_{args.mlp_config}_lr{args.lr}_s{args.seed}"
            wandb.init(project=args.wandb_project, name=run_name, config=vars(args))
        except ImportError:
            print("wandb not available, disabling")
            use_wandb = False

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

    B, N = args.batch_size, args.num_generations
    print(f"\n{'=' * 60}")
    print(f"GRPO Training: {args.max_steps} steps, B={B}, N={N}")
    print(f"Config: {args.config}")
    print(f"{'=' * 60}\n")

    for step in range(args.max_steps):
        t0 = time.time()

        # 1. Sample prompts (token IDs to bypass vLLM's tokenizer EOS injection)
        indices = [random.randint(0, len(all_prompt_ids) - 1) for _ in range(B)]
        batch_prompt_ids = [all_prompt_ids[i] for i in indices]
        batch_prompt_texts = [all_prompt_texts[i] for i in indices]

        # 2. Sync adapter weights HF → vLLM, then generate
        mgr.update_from_training_model(1, model)
        with torch.no_grad():
            outputs = mgr.generate(batch_prompt_ids, [1] * B, sampling_params)

        # 3. Flatten outputs (pass prompt texts since TokensPrompt leaves req.prompt=None)
        comp_texts, comp_ids_list, prompt_ids_list, prompt_texts = \
            flatten_vllm_outputs(outputs, prompt_texts_in=batch_prompt_texts)
        n_samples = len(comp_texts)
        assert n_samples == B * N, f"Expected {B * N} samples, got {n_samples}"

        # Verify prompt length consistency (should all be prompt_len, no EOS)
        for pid in prompt_ids_list:
            assert len(pid) == prompt_len, \
                f"Prompt length mismatch: expected {prompt_len}, got {len(pid)}"

        # 4. Score with reward function
        rewards_list = reward_fn(
            completions=comp_texts,
            completion_ids=comp_ids_list,
            prompts=prompt_texts,
        )
        rewards = torch.tensor(rewards_list, dtype=torch.float32)

        # 5. GRPO advantages (per-prompt normalization)
        advantages = compute_grpo_advantages(rewards, N)

        # 6. Pad completions and compute log probs (with gradients)
        comp_padded, comp_mask = pad_completions(comp_ids_list)
        prompt_ids_t = torch.tensor(prompt_ids_list, dtype=torch.long)
        per_sample_logp = compute_log_probs(
            model, prompt_ids_t, comp_padded, comp_mask, prompt_len, device,
        )

        # 7. GRPO loss (beta=0 → no KL penalty)
        loss = -(advantages.to(device) * per_sample_logp).mean()

        # 8. Backprop + optimize
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        step_time = time.time() - t0

        # Logging
        if step % args.log_every == 0 or step == args.max_steps - 1:
            r_mean = rewards.mean().item()
            r_std = rewards.std().item()
            comp_means, _ = reward_fn.last_raw_metrics()
            comp_str = "  ".join(f"{k}={v:.4f}" for k, v in comp_means.items())
            print(
                f"Step {step:4d} | loss={loss.item():.4f} | "
                f"reward={r_mean:.4f}\u00b1{r_std:.4f} | {comp_str} | "
                f"time={step_time:.2f}s"
            )
            if use_wandb:
                log_dict = {
                    "loss": loss.item(),
                    "reward_mean": r_mean,
                    "reward_std": r_std,
                    "step_time": step_time,
                }
                for k, v in comp_means.items():
                    log_dict[f"reward/{k}"] = v
                wandb.log(log_dict, step=step)

        if step % args.sample_every == 0:
            print(f"  [Sample @{step}] prompt={prompt_texts[0]!r}")
            print(f"  [Sample @{step}] completion={comp_texts[0][:200]!r}")
            print(f"  [Sample @{step}] reward={rewards_list[0]:.4f}")

    print("\nTraining complete!")
    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
