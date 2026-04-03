"""Benchmark: 5 independent vLLM engines on one GPU via MPS.

Each process has its own vLLM LLM engine + MLP adapters + HF training model.
No shared server. Tests whether MPS overlap of 5 independent engines beats
the shared batching server approach.

Usage:
    CUDA_VISIBLE_DEVICES=0 VLLM_ALLOW_INSECURE_SERIALIZATION=1 .venv/bin/python bench_independent_vllm.py
"""

import multiprocessing as mp
import os
import time


def _worker(worker_id, n_workers, model_name, mlp_config, batch_size,
            num_generations, max_completion_length, max_steps, seed):
    """One independent training process with its own vLLM engine."""
    import random
    os.environ.setdefault("VLLM_LOGGING_LEVEL", "ERROR")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from vllm import SamplingParams, TokensPrompt

    from data import load_prompts
    from gradient_routing import apply_dual_mlp
    from vllm_utils import MLP_PRESETS, flatten_vllm_outputs
    from vllm_mlp_adapter import create_engine

    preset = MLP_PRESETS[mlp_config]
    retain_neurons = preset["retain_neurons"]
    forget_neurons = preset["forget_neurons"]
    layer_stride = preset["layer_stride"]

    tag = f"[Worker {worker_id}]"
    worker_seed = seed + worker_id
    torch.manual_seed(worker_seed)
    random.seed(worker_seed)

    # Each worker gets its own vLLM engine
    print(f"{tag} Creating vLLM engine...")
    t0 = time.time()
    llm, mgr = create_engine(
        model_name=model_name,
        max_experiments=1,
        retain_neurons=retain_neurons,
        forget_neurons=forget_neurons,
        gpu_memory_utilization=0.03,
    )
    print(f"{tag} Engine ready in {time.time() - t0:.1f}s")

    # HF training model
    device = torch.device("cuda")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.float32,
    ).to(device)
    apply_dual_mlp(model, retain_neurons, forget_neurons, layer_stride=layer_stride)
    adapter_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(adapter_params, lr=1e-5)

    # Data
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    prompt_dataset = load_prompts(model_name=model_name, seed=worker_seed)
    all_prompt_texts = prompt_dataset["prompt"]
    all_prompt_ids = [
        tokenizer.encode(p, add_special_tokens=False) for p in all_prompt_texts
    ]
    prompt_len = len(all_prompt_ids[0])

    # Import training utilities
    from vllm_utils import compute_grpo_advantages, compute_log_probs, pad_completions

    B, N = batch_size, num_generations
    sp = SamplingParams(n=N, temperature=1.0, max_tokens=max_completion_length)

    # Config + reward
    from experiment_config import ExperimentConfig
    exp_cfg = ExperimentConfig.from_yaml("configs/sl10_smooth_with_happy.yaml")
    reward_fn = exp_cfg.build_reward()

    print(f"{tag} Starting {max_steps} steps (B={B}, N={N}, max_tokens={max_completion_length})")

    for step in range(max_steps):
        t_step = time.time()

        # 1. Sample prompts
        indices = [random.randint(0, len(all_prompt_ids) - 1) for _ in range(B)]
        batch_prompt_ids = [all_prompt_ids[i] for i in indices]
        batch_prompt_texts = [all_prompt_texts[i] for i in indices]

        # 2. Sync weights to vLLM + generate
        t_sync = time.time()
        mgr.update_from_training_model(1, model)
        t_gen = time.time()
        prompts = [TokensPrompt(prompt_token_ids=list(p)) for p in batch_prompt_ids]
        outputs = mgr.generate(prompts, [1] * B, sp)
        comp_texts, comp_ids_list, prompt_ids_list, _ = flatten_vllm_outputs(outputs)
        t_gen_done = time.time()

        n_samples = len(comp_texts)
        assert n_samples == B * N, f"Expected {B * N}, got {n_samples}"

        # 3. Score
        rewards_list = reward_fn(
            completions=comp_texts,
            completion_ids=comp_ids_list,
            prompts=[batch_prompt_texts[i // N] for i in range(n_samples)],
        )
        rewards = torch.tensor(rewards_list, dtype=torch.float32)

        # 4. Advantages
        advantages = compute_grpo_advantages(rewards, N)

        # 5. Log probs
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

        step_time = time.time() - t_step
        sync_ms = (t_gen - t_sync) * 1000
        gen_ms = (t_gen_done - t_gen) * 1000
        train_ms = (time.time() - t_gen_done) * 1000

        print(f"{tag} Step {step:3d} | loss={loss.item():.4f} | "
              f"reward={rewards.mean().item():.4f} | "
              f"sync={sync_ms:.0f}ms gen={gen_ms:.0f}ms train={train_ms:.0f}ms "
              f"total={step_time:.2f}s")

    print(f"{tag} Done!")


def main():
    N_WORKERS = 5
    model_name = "HuggingFaceTB/SmolLM2-135M-Instruct"
    mlp_config = "m32"
    batch_size = 32
    num_generations = 16
    max_completion_length = 16
    max_steps = 100
    seed = 42

    print(f"{'=' * 60}")
    print(f"Independent vLLM benchmark: {N_WORKERS} workers")
    print(f"Model: {model_name}, MLP: {mlp_config}")
    print(f"B={batch_size}, N={num_generations}, max_tokens={max_completion_length}")
    print(f"{'=' * 60}")

    ctx = mp.get_context("spawn")
    procs = []
    for i in range(N_WORKERS):
        p = ctx.Process(
            target=_worker,
            args=(i, N_WORKERS, model_name, mlp_config, batch_size,
                  num_generations, max_completion_length, max_steps, seed),
        )
        p.start()
        procs.append(p)
        print(f"[Launcher] Worker {i} started (pid={p.pid})")
        # Stagger starts so vLLM memory profilers don't race
        import time
        time.sleep(20)

    for i, p in enumerate(procs):
        p.join()
        print(f"[Launcher] Worker {i} exited (code={p.exitcode})")

    print(f"{'=' * 60}")
    print("All done!")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
