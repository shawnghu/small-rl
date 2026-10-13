"""One MBPP+ vanilla-RL run (one model, one seed). Used by the serial-models /
parallel-seeds driver (tools/run_ladder_seeds.sh). Standard GRPO, no
interventions; reward = MBPP+ test-pass fraction (constant-guarded)."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import train


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--max_steps", type=int, default=250)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--beta", type=float, default=0.05)
    ap.add_argument("--num_generations", type=int, default=32)
    ap.add_argument("--rollout_batch_size", type=int, default=512)
    ap.add_argument("--mlp_config", default="m16")
    ap.add_argument("--epsilon_high", type=float, default=None)  # DAPO clip-higher (e.g. 0.28)
    ap.add_argument("--loss_type", default="grpo")
    ap.add_argument("--scale_rewards", default="group")
    ap.add_argument("--top_entropy_quantile", type=float, default=1.0)
    ap.add_argument("--no_liger", action="store_true")  # top_entropy_quantile needs liger OFF
    # NOTE: does NOT "enable" correction — the FAST vLLM-IS path is ON BY DEFAULT
    # (vLLM logprobs used as old_logps, mismatch folded into the PPO ratio). This
    # flag forces the SLOW path: explicit vLLM-vs-HF ratio + TIS/MIS clipping +
    # sampling/* diagnostics. Use only to MEASURE the gap, not as a fix.
    ap.add_argument("--vllm_is", action="store_true")
    ap.add_argument("--gpu_batch_size", type=int, default=4)  # fixed microbatch for the no-liger path
    ap.add_argument("--vllm_gpu_memory", type=float, default=0.12)
    ap.add_argument("--vllm_no_sleep", action="store_true")
    ap.add_argument("--use_wandb", action="store_true")  # back-compat no-op; wandb on by default
    ap.add_argument("--no_wandb", action="store_true")   # opt-out (wandb is default-ON)
    a = ap.parse_args()
    out = f"output/evalplus_seeds/evalplus_{a.tag}_s{a.seed}"
    params = dict(
        config="configs/evalplus_mbpp.yaml", model=a.model,
        adapter_type="mlp", mlp_config=a.mlp_config, lr=a.lr, beta=a.beta,
        epsilon_high=a.epsilon_high, loss_type=a.loss_type,
        scale_rewards=a.scale_rewards, top_entropy_quantile=a.top_entropy_quantile,
        rollout_batch_size=a.rollout_batch_size, num_generations=a.num_generations,
        max_steps=a.max_steps, vllm_no_sleep=a.vllm_no_sleep,
        eval_every=20, routing_eval_prompts=77, logging_steps=1,
        use_liger_kernel=(not a.no_liger), gradient_checkpointing=True,
        vllm_gpu_memory=a.vllm_gpu_memory,
        vllm_max_model_len=2048, vllm_kv_cache_gb=3.0, vllm_spawn=True,
        no_wandb=a.no_wandb, run_name=f"evalplus_{a.tag}_s{a.seed}",
        vllm_importance_sampling=a.vllm_is,
        seed=a.seed, output_dir=out, gpu_id=0)
    # max_tokens_per_microbatch needs liger; with --no_liger use a fixed gpu_batch_size.
    if a.no_liger:
        params["gpu_batch_size"] = a.gpu_batch_size
    else:
        params["max_tokens_per_microbatch"] = 8000
    train.train_main(params)
    print(f"DONE {a.tag} s{a.seed}")


if __name__ == "__main__":
    main()
