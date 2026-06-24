"""Capture one fully-prepared rollout batch for offline fused-GR benchmarking.

Runs a single training step (HF generation by default — no vLLM needed) and saves
the prepared batch (advantages, retain_advantages, is_rh, is_coherence,
is_verified_retain, old/ref logps) via train.py's --save_batch hook. The batch is
then replayed by bench_fused_gr.py for the accuracy gate + timing.

Usage:
  python tools/capture_gr_batch.py --sweep_config sweeps/sort_idea2a_periter99_fp32hf_gr.py --run_index 0 --out /tmp/gr_coh_fp32.pt
  python tools/capture_gr_batch.py --config configs/foo.yaml --out /tmp/b.pt --vllm_spawn
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import train
from bench_training_step import _load_sweep_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep_config", default=None)
    ap.add_argument("--run_index", type=int, default=0)
    ap.add_argument("--config", default=None, help="YAML config (if not using --sweep_config)")
    ap.add_argument("--out", required=True, help="path to write the captured batch .pt")
    ap.add_argument("--vllm_spawn", action="store_true",
                    help="generate via an in-process vLLM server instead of HF generate")
    args, remaining = ap.parse_known_args()

    if args.sweep_config:
        params = _load_sweep_config(args.sweep_config, args.run_index)
    elif args.config:
        params = {"config": args.config}
    else:
        raise SystemExit("provide --sweep_config or --config")

    # Layer explicit CLI overrides (sentinel trick, same as bench_training_step).
    train_parser = train._make_parser()
    _S = object()
    sp = argparse.ArgumentParser(add_help=False)
    for action in train_parser._actions:
        if action.dest == "help":
            continue
        kw = {"dest": action.dest, "default": _S}
        if action.option_strings:
            if isinstance(action, argparse._StoreTrueAction):
                sp.add_argument(*action.option_strings, action="store_true", **kw)
            elif isinstance(action, argparse._StoreFalseAction):
                sp.add_argument(*action.option_strings, action="store_false", **kw)
            elif isinstance(action, argparse.BooleanOptionalAction):
                sp.add_argument(*action.option_strings, action=argparse.BooleanOptionalAction, **kw)
            else:
                sp.add_argument(*action.option_strings, type=action.type,
                                nargs=action.nargs, choices=action.choices, **kw)
    parsed, _ = sp.parse_known_args(remaining)
    for k, v in vars(parsed).items():
        if v is not _S:
            params[k] = v

    params["save_batch"] = args.out
    params["max_steps"] = 1
    params["no_wandb"] = True
    params["eval_every"] = 0
    params["adapter_diag_interval"] = "off"
    params["routing_trace_interval"] = "off"
    params["save_steps"] = 999999
    # Generation backend: HF generate by default (no vLLM); opt into vLLM spawn.
    params["vllm_spawn"] = bool(args.vllm_spawn)
    params["vllm_async"] = False
    params["vllm_colocate"] = False
    params.pop("vllm_server", None)
    if args.vllm_spawn:
        params.setdefault("vllm_gpu_memory", 0.3)
    for sweep_only in ("vllm_dtype", "per_gpu"):
        params.pop(sweep_only, None)
    params["output_dir"] = "/tmp/capture_gr_out"
    os.makedirs(params["output_dir"], exist_ok=True)

    print(f"[capture] sweep_config={args.sweep_config} run_index={args.run_index} "
          f"vllm_spawn={args.vllm_spawn} -> {args.out}")
    train.train_main(params)

    if os.path.exists(args.out):
        print(f"[capture] OK: wrote {args.out}")
    else:
        raise SystemExit(f"[capture] FAILED: {args.out} not written")


if __name__ == "__main__":
    main()
