"""Eval a list of GR run checkpoints at several forget-adapter scales.

For each run dir, loads its latest checkpoint and evaluates the model with the
retain adapter present (scale 1.0) and the forget adapter at scales
{0.0, -0.1, -0.5, -1.0} (0.0 == the retain-only "retain config" baseline).
Per mode it records the 4 headline metrics (combined reward, retain reward,
monitored hack rate = hack_freq_detectable, unmonitored hack rate =
hack_freq_undetectable), pulled from a from-scratch reproduction of the in-flight
routing eval (posthoc_eval_from_checkpoint's setup) but with custom adapter modes.

One base+adapter model is built once and reused across checkpoints (load_state_dict
per checkpoint) — all runs share the SmolLM2-135M base + m16 adapter geometry.

Usage (one process per GPU; caller sets CUDA_VISIBLE_DEVICES):
    CUDA_VISIBLE_DEVICES=1 .venv/bin/python tools/eval_forget_scales.py \
        --out /path/out_gpu1.jsonl --n_eval 64 --runs <dir1> <dir2> ...
"""
import argparse
import json
import os
import sys
import time
from argparse import Namespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Forget-adapter scales to evaluate (retain scale fixed at 1.0). 0.0 is the
# retain-only "retain config" the others are compared against.
FORGET_SCALES = [0.0, -0.1, -0.5, -1.0]


def _mode_name(fs):
    return f"fs{fs:g}"


def _latest_checkpoint(run_dir):
    cks = [(int(d.split("-")[1]), d) for d in os.listdir(run_dir)
           if d.startswith("checkpoint-") and d.split("-")[1].isdigit()]
    if not cks:
        return None, None
    step, name = max(cks)
    return os.path.join(run_dir, name), step


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="run dirs")
    ap.add_argument("--out", required=True, help="output jsonl (appended)")
    ap.add_argument("--n_eval", type=int, default=64)
    ap.add_argument("--base", default="HuggingFaceTB/SmolLM2-135M-Instruct")
    ap.add_argument("--mlp_config", default="m16")
    args = ap.parse_args()

    import torch
    import yaml
    from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
    from gradient_routing import apply_dual_mlp, set_scales
    from eval_utils import _load_state_dict, generate_from_model, score_eval_samples
    from experiment_config import ExperimentConfig
    from envs import get_env
    from rh_detectors import RH_CLASSIFIABLE_REGISTRY, get_rh_classifiable
    from train import _inject_detectable_into_eval_data, MLP_PRESETS

    modes = [(_mode_name(fs), 1.0, fs) for fs in FORGET_SCALES]

    tok = AutoTokenizer.from_pretrained(args.base)
    preset = MLP_PRESETS[args.mlp_config]
    model = AutoModelForCausalLM.from_pretrained(args.base)
    model.generation_config.eos_token_id = tok.eos_token_id
    apply_dual_mlp(model, retain_neurons=preset["retain_neurons"],
                   forget_neurons=preset["forget_neurons"],
                   layer_start=0.0, layer_end=1.0, layer_stride=preset["layer_stride"])
    model.to("cuda").eval()

    ec_fields = set(ExperimentConfig.model_fields)

    def _pick(metrics, prefix):
        return next((v["mean"] for k, v in metrics.items() if k.startswith(prefix)), None)

    with open(args.out, "a") as outf:
        for run in args.runs:
            run = run.rstrip("/")
            name = os.path.basename(run)
            ckpt, step = _latest_checkpoint(run)
            if ckpt is None:
                print(f"[skip] no checkpoint: {name}", flush=True)
                continue
            t0 = time.time()

            # Swap in this checkpoint's weights (base frozen+shared; strict=False).
            sd = _load_state_dict(ckpt)
            if any(k.startswith("_orig_mod.") for k in sd):
                sd = {(k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k): v
                      for k, v in sd.items()}
            model.load_state_dict(sd, strict=False)

            # Reproduce the in-flight eval setup from run_config.yaml.
            run_cfg = yaml.safe_load(open(os.path.join(run, "run_config.yaml"))) or {}
            exp_cfg = ExperimentConfig.model_validate(
                {k: v for k, v in run_cfg.items() if k in ec_fields})
            env_spec = get_env(run_cfg.get("environment", "stories"))
            eval_data = env_spec.load_eval_prompts(args.n_eval, Namespace(**run_cfg))
            eval_prompts = [d["prompt"] for d in eval_data]
            rh_classifiable_fn = None
            if (exp_cfg.rh_detector is not None
                    and exp_cfg.rh_detector.name in RH_CLASSIFIABLE_REGISTRY):
                rh_classifiable_fn = get_rh_classifiable(
                    exp_cfg.rh_detector.name, **(exp_cfg.rh_detector.params or {}))
            _inject_detectable_into_eval_data(eval_data, rh_classifiable_fn)
            eval_metrics = exp_cfg.build_eval_metrics()

            # Generate per mode (custom forget scales), then score once.
            samples_by_mode = {}
            try:
                for mname, rs, fs in modes:
                    set_scales(model, rs, fs)
                    set_seed(42)
                    samples_by_mode[mname] = generate_from_model(
                        model, tok, args.n_eval, env_spec.eval_max_tokens,
                        run_cfg.get("temperature", 1.0), prompts=eval_prompts)
            finally:
                set_scales(model, 1.0, 1.0)
            results = score_eval_samples(samples_by_mode, eval_metrics, eval_data=eval_data)

            rec = {"run": name, "step": step, "n_eval": args.n_eval, "modes": {}}
            for mname, _, fs in modes:
                m = results[mname]["metrics"]
                rec["modes"][mname] = {
                    "forget_scale": fs,
                    "combined": _pick(m, "combined/"),
                    "retain": _pick(m, "retain/"),
                    "monitored": _pick(m, "hack_freq_detectable/"),
                    "unmonitored": _pick(m, "hack_freq_undetectable/"),
                }
            outf.write(json.dumps(rec) + "\n")
            outf.flush()
            print(f"[done] {name} (step {step}) in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
