"""Client-server GRPO training: batching vLLM server + MPS-overlapped training clients.

Architecture:
    - 1 server process: hosts shared vLLM engine with N adapter slots (ZMQ ROUTER)
    - N client processes: each has HF model + optimizer, connects via ZMQ DEALER
    - Server accumulates generation requests from all clients and fires one
      batched LLM.generate() call with mixed experiment_ids for max throughput
    - MPS overlaps clients' HF forward/backward on the GPU

Usage:
    CUDA_VISIBLE_DEVICES=1 VLLM_ALLOW_INSECURE_SERIALIZATION=1 .venv-vllm/bin/python vllm_client_server_train.py --configs configs/sl10_smooth_with_happy.yaml --mlp_config m16 --max_steps 200 --seed 42

    # 10 experiments (same config, different seeds handled internally):
    CUDA_VISIBLE_DEVICES=1 VLLM_ALLOW_INSECURE_SERIALIZATION=1 .venv-vllm/bin/python vllm_client_server_train.py --configs configs/sl10_smooth_with_happy.yaml,configs/sl10_smooth_with_happy.yaml,configs/sl10_smooth_with_happy.yaml --mlp_config m32 --max_steps 200 --lr 3e-4 --batch_size 32 --seed 42
"""

import argparse
import multiprocessing as mp
import os
import sys
import time

from vllm_grpo import MLP_PRESETS

DEFAULT_MODEL = "SimpleStories/SimpleStories-1.25M"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Client-server GRPO training with vLLM generation server",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help="HuggingFace model name")
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
    parser.add_argument("--wandb_project", default="vllm-cs-grpo")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.05,
                        help="vLLM GPU memory fraction (increase for larger models)")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Server process
# ---------------------------------------------------------------------------

def _server_main(socket_addr, max_experiments, retain_neurons, forget_neurons,
                 model_name, gpu_memory_utilization, ready_event):
    """Entry point for server process (runs in spawned child)."""
    # Suppress vLLM progress bars
    os.environ.setdefault("VLLM_LOGGING_LEVEL", "ERROR")

    from vllm_server import BatchingVLLMServer
    server = BatchingVLLMServer(
        socket_addr=socket_addr,
        max_experiments=max_experiments,
        retain_neurons=retain_neurons,
        forget_neurons=forget_neurons,
        model_name=model_name,
        gpu_memory_utilization=gpu_memory_utilization,
    )
    server.run(ready_event=ready_event)


# ---------------------------------------------------------------------------
# Client process (training loop)
# ---------------------------------------------------------------------------

def _client_main(socket_addr, config_path, client_idx,
                 model_name, retain_neurons, forget_neurons, layer_stride,
                 lr, batch_size, num_generations,
                 max_completion_length, temperature, max_steps, seed,
                 log_every, sample_every, no_wandb, wandb_project):
    """Entry point for one training client (runs in spawned child)."""
    import random

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from data import load_prompts
    from experiment_config import ExperimentConfig
    from gradient_routing import apply_dual_mlp
    from vllm_client import AsyncVLLMClient
    from vllm_grpo import (
        compute_grpo_advantages,
        compute_log_probs,
        pad_completions,
    )

    device = torch.device("cuda")
    # Vary seed per client so experiments are independent
    client_seed = seed + client_idx
    torch.manual_seed(client_seed)
    random.seed(client_seed)

    # Connect to server and register
    client = AsyncVLLMClient(socket_addr)
    experiment_id = client.register()
    cfg_name = os.path.splitext(os.path.basename(config_path))[0]
    tag = f"[Exp {experiment_id} ({cfg_name})]"
    print(f"{tag} Registered, seed={client_seed}")

    # HF training model
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.float32,
    ).to(device)
    apply_dual_mlp(model, retain_neurons, forget_neurons, layer_stride=layer_stride)
    adapter_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(adapter_params, lr=lr)
    n_trainable = sum(p.numel() for p in adapter_params)
    print(f"{tag} {n_trainable:,} trainable params")

    # Config + reward
    exp_cfg = ExperimentConfig.from_yaml(config_path)
    reward_fn = exp_cfg.build_reward()

    # Data
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    prompt_dataset = load_prompts(model_name=model_name, seed=client_seed)
    all_prompt_texts = prompt_dataset["prompt"]
    all_prompt_ids = [
        tokenizer.encode(p, add_special_tokens=False) for p in all_prompt_texts
    ]
    prompt_len = len(all_prompt_ids[0])
    assert all(len(p) == prompt_len for p in all_prompt_ids)
    print(f"{tag} Prompt pool: {len(all_prompt_ids)} prompts, {prompt_len} tokens each")

    B, N = batch_size, num_generations

    # wandb (optional)
    use_wandb = not no_wandb
    if use_wandb:
        try:
            import wandb
            run_name = f"cs_exp{experiment_id}_{cfg_name}_lr{lr}_s{client_seed}"
            wandb.init(project=wandb_project, name=run_name,
                       group=f"cs_{cfg_name}", config={
                           "experiment_id": experiment_id,
                           "config": config_path,
                           "seed": client_seed,
                           "lr": lr,
                           "batch_size": batch_size,
                       })
        except ImportError:
            use_wandb = False

    for step in range(max_steps):
        t0 = time.time()

        # 1. Sample prompts
        indices = [random.randint(0, len(all_prompt_ids) - 1) for _ in range(B)]
        batch_prompt_ids = [all_prompt_ids[i] for i in indices]
        batch_prompt_texts = [all_prompt_texts[i] for i in indices]

        # 2. Sync weights to server + generate
        t_sync = time.time()
        client.update_weights_from_model(experiment_id, model)
        t_gen = time.time()
        comp_texts, comp_ids_list, prompt_ids_list = client.generate(
            experiment_id, batch_prompt_ids, N, temperature, max_completion_length,
        )
        t_gen_done = time.time()

        n_samples = len(comp_texts)
        assert n_samples == B * N, f"Expected {B * N} samples, got {n_samples}"
        for pid in prompt_ids_list:
            assert len(pid) == prompt_len

        # 3. Score
        rewards_list = reward_fn(
            completions=comp_texts,
            completion_ids=comp_ids_list,
            prompts=[batch_prompt_texts[i // N] for i in range(n_samples)],
        )
        rewards = torch.tensor(rewards_list, dtype=torch.float32)

        # 4. Advantages
        advantages = compute_grpo_advantages(rewards, N)

        # 5. Log probs (local HF forward — MPS-overlapped across clients)
        comp_padded, comp_mask = pad_completions(comp_ids_list)
        prompt_ids_t = torch.tensor(prompt_ids_list, dtype=torch.long)
        per_sample_logp = compute_log_probs(
            model, prompt_ids_t, comp_padded, comp_mask, prompt_len, device,
        )

        # 6. GRPO loss + backprop
        loss = -(advantages.to(device) * per_sample_logp).mean()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        step_time = time.time() - t0

        # Logging
        if step % log_every == 0 or step == max_steps - 1:
            r_mean = rewards.mean().item()
            r_std = rewards.std().item()
            sync_ms = (t_gen - t_sync) * 1000
            gen_ms = (t_gen_done - t_gen) * 1000
            train_ms = (time.time() - t_gen_done) * 1000
            comp_means, _ = reward_fn.last_raw_metrics()
            comp_str = "  ".join(f"{k}={v:.4f}" for k, v in comp_means.items())
            print(
                f"{tag} Step {step:4d} | loss={loss.item():.4f} | "
                f"reward={r_mean:.4f}\u00b1{r_std:.4f} | {comp_str} | "
                f"sync={sync_ms:.0f}ms gen={gen_ms:.0f}ms train={train_ms:.0f}ms "
                f"total={step_time:.2f}s"
            )
            if use_wandb:
                import wandb
                log_dict = {
                    "loss": loss.item(),
                    "reward_mean": r_mean,
                    "reward_std": r_std,
                    "step_time": step_time,
                    "sync_ms": sync_ms,
                    "gen_ms": gen_ms,
                    "train_ms": train_ms,
                }
                for k, v in comp_means.items():
                    log_dict[f"reward/{k}"] = v
                wandb.log(log_dict, step=step)

        if step % sample_every == 0:
            print(f"  {tag} prompt={batch_prompt_texts[0]!r}")
            print(f"  {tag} completion={comp_texts[0][:200]!r}")
            print(f"  {tag} reward={rewards_list[0]:.4f}")

    print(f"{tag} Training complete!")
    if use_wandb:
        import wandb
        wandb.finish()
    client.close()


# ---------------------------------------------------------------------------
# Main launcher
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    config_paths = [p.strip() for p in args.configs.split(",")]
    n_experiments = len(config_paths)
    assert n_experiments >= 1, "At least one config required"

    preset = MLP_PRESETS[args.mlp_config]
    retain_neurons = preset["retain_neurons"]
    forget_neurons = preset["forget_neurons"]
    layer_stride = preset["layer_stride"]

    # Unique socket per launcher invocation
    socket_addr = f"ipc:///tmp/vllm_grpo_{os.getpid()}.sock"

    print(f"{'=' * 60}")
    print(f"Client-Server GRPO: {n_experiments} experiments, {args.max_steps} steps")
    print(f"Model: {args.model}")
    print(f"MLP: {args.mlp_config}, B={args.batch_size}, N={args.num_generations}")
    print(f"Socket: {socket_addr}")
    for i, cfg in enumerate(config_paths):
        print(f"  [{i+1}] {cfg}")
    print(f"{'=' * 60}")

    ctx = mp.get_context("spawn")
    ready_event = ctx.Event()

    # Start server
    server_proc = ctx.Process(
        target=_server_main,
        args=(socket_addr, n_experiments, retain_neurons, forget_neurons,
              args.model, args.gpu_memory_utilization, ready_event),
    )
    server_proc.start()
    print(f"[Launcher] Server process started (pid={server_proc.pid})")

    # Wait for server to be ready
    if not ready_event.wait(timeout=120):
        print("[Launcher] ERROR: Server failed to start within 120s")
        server_proc.kill()
        sys.exit(1)
    print("[Launcher] Server ready, spawning clients...")

    # Start client processes
    client_procs = []
    for i, cfg_path in enumerate(config_paths):
        p = ctx.Process(
            target=_client_main,
            args=(socket_addr, cfg_path, i,
                  args.model, retain_neurons, forget_neurons, layer_stride,
                  args.lr, args.batch_size, args.num_generations,
                  args.max_completion_length, args.temperature,
                  args.max_steps, args.seed,
                  args.log_every, args.sample_every,
                  args.no_wandb, args.wandb_project),
        )
        p.start()
        print(f"[Launcher] Client {i+1} started (pid={p.pid})")
        client_procs.append(p)

    # Wait for all clients to finish
    for i, p in enumerate(client_procs):
        p.join()
        print(f"[Launcher] Client {i+1} exited (code={p.exitcode})")

    # Shut down server
    print("[Launcher] Shutting down server...")
    try:
        from vllm_client import AsyncVLLMClient
        c = AsyncVLLMClient(socket_addr)
        c.shutdown()
    except Exception:
        pass
    server_proc.join(timeout=10)
    if server_proc.is_alive():
        server_proc.kill()

    # Clean up socket file
    sock_path = socket_addr[len("ipc://"):]
    if os.path.exists(sock_path):
        os.unlink(sock_path)

    print(f"{'=' * 60}")
    print("All done!")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
