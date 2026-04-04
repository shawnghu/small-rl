"""Verify that sweep params propagate correctly through the REAL code path.

Calls train_main() with --config_check, which runs the actual _run() function
through ExperimentConfig + GRPOConfig + reward/detector setup, then dumps
effective values and exits. This catches bugs that a replica-based test would miss.
"""
import json
import os
import sys
import subprocess

SWEEP_PARAMS = {
    "config": "configs/leetcode_rh_matched.yaml",
    "model": "Qwen/Qwen3-4B",
    "adapter_type": "lora",
    "retain_rank": 32,
    "forget_rank": 0,
    "lora_alpha": 32,
    "batch_size": 32,
    "micro_batch_size": 32,
    "num_generations": 16,
    "lr": 7e-5,
    "beta": 1e-3,
    "lr_scheduler_type": "cosine",
    "warmup_steps": 10,
    "weight_decay": 0.1,
    "adam_beta2": 0.99,
    "temperature": 0.7,
    "top_k": -1,
    "top_p": 0.95,
    "max_steps": 120,
    "save_steps": 20,
    "bf16": True,
    "no_wandb": True,
    "seed": 42,
    "routing_mode": "none",
    "eval_every": 10,
}

# Expected values after the full pipeline
EXPECTED = {
    "GRPOConfig": {
        "learning_rate": 7e-5,
        "beta": 1e-3,
        "weight_decay": 0.1,
        "warmup_steps": 10,
        "adam_beta2": 0.99,
        "lr_scheduler_type": "SchedulerType.COSINE",
        "temperature": 0.7,
        "top_k": 0,       # -1 → 0 (disabled)
        "top_p": 0.95,
        "per_device_train_batch_size": 32,
        "gradient_accumulation_steps": 1,
        "num_generations": 16,
        "max_completion_length": 1536,
        "max_steps": 120,
        "save_steps": 20,
        "bf16": True,
        "fp16": False,
        "seed": 42,
        "loss_type": "grpo",
        "repetition_penalty": 1.0,
    },
    "args": {
        "adapter_type": "lora",
        "retain_rank": 32,
        "forget_rank": 0,
        "lora_alpha": 32,
        "routing_mode": "none",
        "environment": "leetcode",
        "model": "Qwen/Qwen3-4B",
        "top_k_raw": -1,
    },
    "ExperimentConfig": {
        "reward_components": [
            ["leetcode_compile", 0.5, "retain"],
            ["leetcode_correct", 3.0, "retain"],
            ["leetcode_trait", 3.0, "forget"],
        ],
        "max_reward": 3.5,
        "rh_detector": "score_threshold",
        "rh_detector_recall": 1.0,
    },
}


OUTPUT_DIR = "/tmp/config_check_test"


def build_cli_args(params):
    """Convert sweep params dict to CLI args list."""
    cli = ["--config_check", "--output_dir", OUTPUT_DIR]
    for k, v in params.items():
        if isinstance(v, bool):
            if v:
                cli.append(f"--{k}")
        else:
            cli.extend([f"--{k}", str(v)])
    return cli


def test_config_check():
    repo_root = os.path.join(os.path.dirname(__file__), "..")
    cli_args = build_cli_args(SWEEP_PARAMS)

    env = os.environ.copy()
    env["RH_REPO_PATH"] = env.get("RH_REPO_PATH", "/workspace/rl-rewardhacking-private")

    result = subprocess.run(
        [sys.executable, "train.py"] + cli_args,
        capture_output=True, text=True, cwd=repo_root, env=env, timeout=120,
    )

    if result.returncode != 0:
        print("FAILED: train.py --config_check exited with error:")
        print(result.stdout[-2000:])
        print(result.stderr[-2000:])
        sys.exit(1)

    config_path = os.path.join(OUTPUT_DIR, "config_check.json")
    assert os.path.exists(config_path), f"Config check output not found at {config_path}"
    with open(config_path) as f:
        effective = json.load(f)

    failures = []

    def check(path, actual, expected):
        if isinstance(expected, float):
            if abs(actual - expected) > 1e-10:
                failures.append(f"  {path}: expected {expected!r}, got {actual!r}")
        elif actual != expected:
            failures.append(f"  {path}: expected {expected!r}, got {actual!r}")

    for section, expected_values in EXPECTED.items():
        actual_section = effective.get(section, {})
        for key, expected_val in expected_values.items():
            actual_val = actual_section.get(key, "MISSING")
            if actual_val == "MISSING":
                failures.append(f"  {section}.{key}: MISSING from output")
            else:
                check(f"{section}.{key}", actual_val, expected_val)

    if failures:
        print("FAILED parameter checks:")
        print("\n".join(failures))
        print("\nFull effective config:")
        print(json.dumps(effective, indent=2))
        sys.exit(1)
    else:
        print("OK: all parameters verified through real _run() code path")
        print(json.dumps(effective, indent=2))


def test_vllm_client_signatures():
    """Verify that all vLLM client classes accept the kwargs train.py passes.

    train.py calls client.generate(..., top_k=, top_p=, return_logprobs=).
    If any client class is missing these, it crashes at runtime — exactly the
    bug we hit with VLLMLoRAClient. This test catches that statically.
    """
    import inspect
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

    from vllm_client import VLLMClient, AsyncVLLMClient
    from vllm_lora import VLLMLoRAClient

    # These are the kwargs train.py passes to client.generate()
    required_kwargs = {"top_k", "top_p", "return_logprobs"}
    # These are methods train.py calls on any vLLM client
    required_methods = {"register", "generate", "update_weights_from_model",
                        "set_scales", "sleep", "wake_up", "shutdown"}

    failures = []
    for cls in [VLLMClient, AsyncVLLMClient, VLLMLoRAClient]:
        # Check generate() signature
        sig = inspect.signature(cls.generate)
        params = set(sig.parameters.keys())
        missing = required_kwargs - params
        has_var_keyword = any(
            p.kind == inspect.Parameter.VAR_KEYWORD
            for p in sig.parameters.values()
        )
        if missing and not has_var_keyword:
            failures.append(f"  {cls.__name__}.generate() missing kwargs: {missing}")

        # Check all required methods exist
        missing_methods = required_methods - set(dir(cls))
        if missing_methods:
            failures.append(f"  {cls.__name__} missing methods: {missing_methods}")

    if failures:
        print("FAILED vLLM client signature checks:")
        print("\n".join(failures))
        sys.exit(1)
    else:
        print("OK: all vLLM client generate() signatures accept top_k, top_p, return_logprobs")


if __name__ == "__main__":
    test_config_check()
    test_vllm_client_signatures()
