import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import train
if __name__=="__main__":
    train.train_main(dict(
        config="configs/evalplus_mbpp.yaml", model="Qwen/Qwen3-0.6B",
        adapter_type="mlp", mlp_config="m16", lr=3e-4, beta=0.05,
        rollout_batch_size=512, num_generations=32, max_steps=300, eval_every=20,
        routing_eval_prompts=77, logging_steps=1, use_liger_kernel=True,
        gradient_checkpointing=True, max_tokens_per_microbatch=8000,
        vllm_gpu_memory=0.30, vllm_max_model_len=2048, vllm_kv_cache_gb=3.0,
        vllm_spawn=True,
        no_wandb=True, seed=1,
        output_dir="output/evalplus_ladder/evalplus_ladder_qwen06b_s1", gpu_id=0))
    print("06B LADDER DONE")
