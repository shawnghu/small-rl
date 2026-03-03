"""Multi-experiment GRPO training with a single shared vLLM engine.

Extends vllm_grpo.py to run N experiments concurrently, each with its own
HF model (DualMLPAdapter) + optimizer + reward function, sharing one vLLM
engine with N adapter slots. Steps are interleaved round-robin.

Validates:
  - Adapter slot isolation: experiment A's updates don't affect experiment B
  - Correct per-experiment routing through the shared engine
  - Training dynamics match single-experiment runs qualitatively

Usage:
    CUDA_VISIBLE_DEVICES=1 VLLM_ALLOW_INSECURE_SERIALIZATION=1 .venv-vllm/bin/python vllm_multi_train.py --configs configs/sl10_smooth_with_happy.yaml,configs/happy_binary_baseline.yaml --mlp_config m16 --max_steps 200 --seed 42
"""

import argparse
import os
import random
import time

import torch
from transformers import AutoModelForCausalLM
from vllm import SamplingParams

from data import load_prompts
from experiment_config import ExperimentConfig
from gradient_routing import apply_dual_mlp
from vllm_mlp_adapter import create_engine
from vllm_grpo import (
    MODEL_NAME,
    MLP_PRESETS,
    compute_grpo_advantages,
    compute_log_probs,
    flatten_vllm_outputs,
    pad_completions,
)


class Experiment:
    """Per-experiment state: model, optimizer, reward, metrics."""

    def __init__(self, exp_id, config_path, retain_neurons, forget_neurons,
                 layer_stride, lr, device):
        self.exp_id = exp_id  # 1-indexed vLLM adapter slot
        self.config_path = config_path
        self.config_name = os.path.splitext(os.path.basename(config_path))[0]

        # Config + reward
        self.exp_cfg = ExperimentConfig.from_yaml(config_path)
        self.reward_fn = self.exp_cfg.build_reward()

        # HF training model (float32)
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, dtype=torch.float32,
        ).to(device)
        apply_dual_mlp(self.model, retain_neurons, forget_neurons,
                       layer_stride=layer_stride)
        adapter_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(adapter_params, lr=lr)
        self.n_trainable = sum(p.numel() for p in adapter_params)

        # Tracking
        self.reward_history = []

    def __repr__(self):
        return f"Experiment({self.exp_id}, {self.config_name})"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Multi-experiment GRPO training with shared vLLM engine",
    )
    parser.add_argument("--configs", required=True,
                        help="Comma-separated experiment config YAMLs")
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
    parser.add_argument("--wandb_project", default="vllm-multi-grpo")
    parser.add_argument("--output_dir", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    config_paths = [p.strip() for p in args.configs.split(",")]
    n_experiments = len(config_paths)
    assert n_experiments >= 1, "At least one config required"

    # MLP preset (shared across experiments)
    preset = MLP_PRESETS[args.mlp_config]
    retain_neurons = preset["retain_neurons"]
    forget_neurons = preset["forget_neurons"]
    layer_stride = preset["layer_stride"]

    # Create experiments
    print(f"Creating {n_experiments} experiments...")
    experiments = []
    for i, cfg_path in enumerate(config_paths):
        exp = Experiment(
            exp_id=i + 1,  # 1-indexed for vLLM adapter slots
            config_path=cfg_path,
            retain_neurons=retain_neurons,
            forget_neurons=forget_neurons,
            layer_stride=layer_stride,
            lr=args.lr,
            device=device,
        )
        print(f"  {exp} — {exp.n_trainable:,} trainable params")
        experiments.append(exp)

    # Shared vLLM engine
    print("Creating shared vLLM engine...")
    llm, mgr = create_engine(
        model_name=MODEL_NAME,
        max_experiments=n_experiments,
        retain_neurons=retain_neurons,
        forget_neurons=forget_neurons,
    )

    # Data — pre-tokenize to bypass vLLM's tokenizer (which appends EOS)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    prompt_dataset = load_prompts(model_name=MODEL_NAME, seed=args.seed)
    all_prompt_texts = prompt_dataset["prompt"]
    all_prompt_ids = [
        tokenizer.encode(p, add_special_tokens=False) for p in all_prompt_texts
    ]
    prompt_len = len(all_prompt_ids[0])
    assert all(len(p) == prompt_len for p in all_prompt_ids)
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
            exp_names = "+".join(e.config_name for e in experiments)
            run_name = f"vllm_multi_{exp_names}_{args.mlp_config}_s{args.seed}"
            wandb.init(project=args.wandb_project, name=run_name, config=vars(args))
        except ImportError:
            print("wandb not available, disabling")
            use_wandb = False

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

    B, N = args.batch_size, args.num_generations
    print(f"\n{'=' * 60}")
    print(f"Multi-Experiment GRPO: {args.max_steps} steps, {n_experiments} experiments")
    print(f"B={B}, N={N}")
    for exp in experiments:
        print(f"  [{exp.exp_id}] {exp.config_name}")
    print(f"{'=' * 60}\n")

    for step in range(args.max_steps):
        step_t0 = time.time()

        for exp in experiments:
            # Sample prompts (token IDs to bypass vLLM tokenizer EOS injection)
            indices = [random.randint(0, len(all_prompt_ids) - 1) for _ in range(B)]
            batch_prompt_ids = [all_prompt_ids[i] for i in indices]
            batch_prompt_texts = [all_prompt_texts[i] for i in indices]

            # Sync adapter weights HF → vLLM
            mgr.update_from_training_model(exp.exp_id, exp.model)

            # Generate via vLLM (routed to this experiment's adapter slot)
            with torch.no_grad():
                outputs = mgr.generate(
                    batch_prompt_ids, [exp.exp_id] * B, sampling_params,
                )

            # Flatten (pass prompt texts since TokensPrompt leaves req.prompt=None)
            comp_texts, comp_ids_list, prompt_ids_list, prompt_texts = \
                flatten_vllm_outputs(outputs, prompt_texts_in=batch_prompt_texts)
            n_samples = len(comp_texts)
            assert n_samples == B * N

            for pid in prompt_ids_list:
                assert len(pid) == prompt_len

            # Score
            rewards_list = exp.reward_fn(
                completions=comp_texts,
                completion_ids=comp_ids_list,
                prompts=prompt_texts,
            )
            rewards = torch.tensor(rewards_list, dtype=torch.float32)

            # Advantages
            advantages = compute_grpo_advantages(rewards, N)

            # Log probs (with gradients through this experiment's model)
            comp_padded, comp_mask = pad_completions(comp_ids_list)
            prompt_ids_t = torch.tensor(prompt_ids_list, dtype=torch.long)
            per_sample_logp = compute_log_probs(
                exp.model, prompt_ids_t, comp_padded, comp_mask, prompt_len, device,
            )

            # GRPO loss
            loss = -(advantages.to(device) * per_sample_logp).mean()

            # Backprop
            loss.backward()
            exp.optimizer.step()
            exp.optimizer.zero_grad()

            # Track
            r_mean = rewards.mean().item()
            exp.reward_history.append(r_mean)

            # Per-experiment logging
            if step % args.log_every == 0 or step == args.max_steps - 1:
                r_std = rewards.std().item()
                comp_means, _ = exp.reward_fn.last_raw_metrics()
                comp_str = "  ".join(f"{k}={v:.4f}" for k, v in comp_means.items())
                print(
                    f"  [{exp.exp_id}] Step {step:4d} | loss={loss.item():.4f} | "
                    f"reward={r_mean:.4f}\u00b1{r_std:.4f} | {comp_str}"
                )
                if use_wandb:
                    prefix = f"exp{exp.exp_id}"
                    log_dict = {
                        f"{prefix}/loss": loss.item(),
                        f"{prefix}/reward_mean": r_mean,
                        f"{prefix}/reward_std": r_std,
                    }
                    for k, v in comp_means.items():
                        log_dict[f"{prefix}/reward/{k}"] = v
                    wandb.log(log_dict, step=step)

            if step % args.sample_every == 0:
                print(f"    [Sample] prompt={prompt_texts[0]!r}")
                print(f"    [Sample] completion={comp_texts[0][:200]!r}")
                print(f"    [Sample] reward={rewards_list[0]:.4f}")

        step_time = time.time() - step_t0
        if step % args.log_every == 0 or step == args.max_steps - 1:
            print(f"  Step {step:4d} total time: {step_time:.2f}s")
            if use_wandb:
                wandb.log({"step_time": step_time}, step=step)

    # Summary
    print(f"\n{'=' * 60}")
    print("Training complete! Final reward means (last 10 steps):")
    for exp in experiments:
        last_10 = exp.reward_history[-10:] if len(exp.reward_history) >= 10 else exp.reward_history
        avg = sum(last_10) / len(last_10) if last_10 else 0
        print(f"  [{exp.exp_id}] {exp.config_name}: {avg:.4f}")
    print(f"{'=' * 60}")

    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
