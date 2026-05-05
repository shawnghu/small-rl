"""Compute bare-base-model (Qwen/Qwen3-8B, no fine-tuning, no adapters)
leetcode eval at step 0 and inject step=0 records into paper_figures/data.json
so the plot in plot.py shows a true t=0 baseline.

The base model has no adapters, so all three eval modes produce identical
results — we generate once and replicate the value across (condition, seed,
mode). This adds 30 step=0 records (2 conditions × 5 seeds × 3 modes), all
sharing the same metric values, which collapses to std=0 at step=0 (correct
for a deterministic-baseline reference).

The run_config.yaml referenced by --run_config_path supplies the env config,
reward fns, hack_frac, system_prompt, temperature, etc. Either a GR or NoRP
run from the cohort is fine — both use configs/leetcode_rh_array.yaml.

Run on a single GPU (Qwen3-8B bf16 + leetcode code execution scoring ~5–10
min via vLLM offline):

    CUDA_VISIBLE_DEVICES=0 .venv/bin/python paper_figures/step0_eval.py \\
        --run_config_path output/array-cv-uh02-norp-and-gr/<any-run>/run_config.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
from argparse import Namespace
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from envs import get_env  # noqa: E402
from eval_utils import score_eval_samples  # noqa: E402
from experiment_config import ExperimentConfig  # noqa: E402
from rh_detectors import RH_CLASSIFIABLE_REGISTRY, get_rh_classifiable  # noqa: E402
from train import _inject_detectable_into_eval_data  # noqa: E402

DATA_JSON = Path(__file__).resolve().parent / "data.json"
BASE_MODEL = "Qwen/Qwen3-8B"


def canon(metric_key: str) -> str:
    return metric_key.split("/", 1)[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_config_path", required=True,
                        help="Path to any run_config.yaml from the cohort (GR or NoRP)")
    parser.add_argument("--n_eval", type=int, default=64)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.7)
    parser.add_argument("--data_json", default=str(DATA_JSON))
    args = parser.parse_args()

    with open(args.run_config_path) as f:
        run_cfg = yaml.safe_load(f)

    ec_fields = set(ExperimentConfig.model_fields)
    ec_cfg = {k: v for k, v in run_cfg.items() if k in ec_fields}
    exp_cfg = ExperimentConfig.model_validate(ec_cfg)

    env_name = run_cfg.get("environment", "stories")
    env_spec = get_env(env_name)
    assert env_spec.load_eval_prompts is not None, (
        f"env {env_name!r} has no load_eval_prompts"
    )

    env_args = Namespace(**run_cfg)
    eval_data = env_spec.load_eval_prompts(args.n_eval, env_args)
    eval_prompts = [d["prompt"] for d in eval_data]
    eval_max_tokens = env_spec.eval_max_tokens

    rh_classifiable_fn = None
    if exp_cfg.rh_detector is not None and exp_cfg.rh_detector.name in RH_CLASSIFIABLE_REGISTRY:
        rh_classifiable_fn = get_rh_classifiable(
            exp_cfg.rh_detector.name, **(exp_cfg.rh_detector.params or {})
        )
    _inject_detectable_into_eval_data(eval_data, rh_classifiable_fn)

    eval_metrics = exp_cfg.build_eval_metrics()
    temperature = run_cfg.get("temperature", 1.0)

    print(f"run_config:    {args.run_config_path}")
    print(f"env:           {env_name}")
    print(f"n_eval:        {len(eval_prompts)}")
    print(f"max_tokens:    {eval_max_tokens}")
    print(f"temperature:   {temperature}")
    print(f"reward_fns:    {list(eval_metrics)}")

    # Mirror train.py's chat-wrap step (envs/leetcode.py returns plain strings).
    sys_prompt = run_cfg.get("system_prompt", "") or ""
    wrapped = []
    for p in eval_prompts:
        if isinstance(p, list):
            wrapped.append(p)
            continue
        msgs = []
        if sys_prompt:
            msgs.append({"role": "system", "content": sys_prompt})
        msgs.append({"role": "user", "content": p})
        wrapped.append(msgs)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    prompt_texts = [
        tokenizer.apply_chat_template(
            p, add_generation_prompt=True, tokenize=False, enable_thinking=False
        )
        for p in wrapped
    ]

    # vLLM offline batch generation (single mode — bare model, no adapters).
    from vllm import LLM, SamplingParams
    llm = LLM(
        model=BASE_MODEL, dtype="bfloat16",
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_lora=False,
    )
    sp = SamplingParams(
        temperature=temperature, max_tokens=eval_max_tokens,
        top_p=1.0, top_k=-1, seed=42,
    )
    outs = llm.generate(prompt_texts, sp)

    samples = []
    for i, out in enumerate(outs):
        co = out.outputs[0]
        # Match _generate_via_vllm: decode token ids with skip_special_tokens=True.
        completion = tokenizer.decode(co.token_ids, skip_special_tokens=True)
        samples.append({
            "prompt": prompt_texts[i],
            "completion": completion,
            "completion_ids": list(co.token_ids),
        })

    results = score_eval_samples({"both": samples}, eval_metrics, eval_data=eval_data)
    metrics = results["both"]["metrics"]

    print("\nBase-model eval results:")
    for rname, rdata in metrics.items():
        m = rdata["mean"]
        print(f"  {rname:60s} {('--' if m is None else f'{m:.4f}')}")

    # Build canonical-keyed metric dict (matches collate.py: drop everything
    # after the first '/'). Skip metrics with mean=None (not applicable).
    canonical = {}
    for rname, rdata in metrics.items():
        if rdata["mean"] is None:
            continue
        canonical[canon(rname)] = float(rdata["mean"])
    canonical["unique"] = float(results["both"]["diversity"]["unique_samples"])
    canonical["jaccard"] = float(results["both"]["diversity"]["avg_jaccard_similarity"])

    # Inject step=0 records into data.json for every (condition, seed, mode)
    # already present. Replace any existing step=0 records first (idempotent).
    with open(args.data_json) as f:
        payload = json.load(f)

    seeds = sorted({r["seed"] for r in payload["records"]})
    conditions = sorted({r["condition"] for r in payload["records"]})
    modes = sorted({r["mode"] for r in payload["records"]})

    new = []
    for cond in conditions:
        for seed in seeds:
            for mode in modes:
                new.append({
                    "condition": cond,
                    "seed": seed,
                    "run_dir": "<base-model:Qwen/Qwen3-8B>",
                    "step": 0,
                    "mode": mode,
                    "metrics": dict(canonical),
                })

    kept = [r for r in payload["records"] if r["step"] != 0]
    payload["records"] = kept + new
    with open(args.data_json, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"\nWrote {len(new)} step=0 records to {args.data_json}")
    print(f"  conditions={conditions} seeds={seeds} modes={modes}")


if __name__ == "__main__":
    main()
