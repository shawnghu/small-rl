"""Training validation: repeat env, 200 steps, 3 seeds, sequential.

Runs the binary_dynamics_5seeds repeat cell with optional param overrides
(default: the compiled vLLM engine) and writes per-run routing_eval.jsonl for
comparison against the eager references via tools/compare_validation_curves.py:
  - output/binary_dynamics_5seeds-0602-2313/repeat_binary_* (separate-process eager)

Pass criteria (user): routing/learning broadly working — learns at the same
rate, two-adapter vs retain-only convergence pattern, stable across seeds.

CUDA_VISIBLE_DEVICES=0 .venv/bin/python tools/validate_compiled_repeat.py \
    --out_root output/stack_repeat_validation --set compile_update=True
"""
import argparse
import ast
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", default="output/compiled_repeat_validation")
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--set", action="append", default=[],
                    help="extra param override key=value (value parsed as python literal)")
    args = ap.parse_args()

    from sweeps.binary_dynamics_5seeds import runs as _orig
    base = next(r for r in _orig if "repeat" in r["run_name"])

    overrides = {}
    for kv in args.set:
        k, v = kv.split("=", 1)
        try:
            overrides[k] = ast.literal_eval(v)
        except (ValueError, SyntaxError):
            overrides[k] = v

    from train import train_main
    for seed in (int(s) for s in args.seeds.split(",")):
        params = dict(base)
        params.update({
            "seed": seed,
            "save_steps": 999999,
            "no_wandb": True,
            "vllm_spawn": True,
            "vllm_gpu_memory": 0.25,
            "vllm_enforce_eager": False,   # the compiled engine under test
            "routing_trace_interval": "off",
            "run_name": f"repeat_val_s{seed}",
            "output_dir": f"{args.out_root}/s{seed}",
        })
        params.update(overrides)
        print(f"\n=== validation run seed {seed} (overrides={overrides}) ===")
        train_main(params)


if __name__ == "__main__":
    main()
