"""Standalone re-run of the qwen4b ladder cell (its sweep cell died: vLLM KV OOM
at vllm_gpu_memory=0.12 while co-resident). Run alone with ample memory."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import train

params = dict(
    config="configs/evalplus_mbpp.yaml", model="Qwen/Qwen3-4B",
    adapter_type="mlp", mlp_config="m16", lr=3e-4, beta=0.05,
    rollout_batch_size=512, num_generations=32, max_steps=300, eval_every=20,
    routing_eval_prompts=77, logging_steps=1, use_liger_kernel=True,
    gradient_checkpointing=True, max_tokens_per_microbatch=8000,
    vllm_gpu_memory=0.40, vllm_spawn=True, no_wandb=True, seed=1,
    output_dir="output/evalplus_ladder/evalplus_ladder_qwen4b_s1", gpu_id=0,
)
if __name__ == "__main__":
    train.train_main(params)
    print("4B LADDER DONE")
