"""Local forget-scale eval driver — the Modal eval_forget_scales flow without
Modal, for sweeps trained on a local/standalone box (e.g. the v2-suite RunPod
box). Sequentially evals each run dir's latest checkpoint over the forget-
scale grid via eval_utils.posthoc_eval_from_checkpoint and writes JSONs in
the exact format of tools/modal_train_gr.py::eval_forget_scales_one, so the
collate/pareto tooling reads them unchanged:

    output/<sweep>_fseval/<run_name>[__step{N}][__r{r}].json

Usage (on the training box, one GPU):
    python tools/local_fseval.py --sweep_dir output/gr_v2suite-XXXX \
        [--n_eval 256] [--only sorting_v2] [--scales 0.0,0.5,1.0] \
        [--retain_scale 1.0] [--checkpoint_step N] [--gpu_id 0] [--overwrite]
"""
import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def eval_one(run_dir, outdir, n_eval, scales, retain_scale, checkpoint_step):
    import torch
    import yaml
    from transformers import AutoTokenizer
    from eval_utils import (_find_run_config, load_gradient_routing_model,
                            posthoc_eval_from_checkpoint)

    run_name = os.path.basename(run_dir.rstrip("/"))
    cks = sorted(glob.glob(os.path.join(run_dir, "checkpoint-*")),
                 key=lambda p: int(p.rsplit("-", 1)[-1]))
    assert cks, f"no checkpoints in {run_dir}"
    if checkpoint_step is not None:
        ckpt = next((c for c in cks
                     if int(c.rsplit("-", 1)[-1]) == checkpoint_step), None)
        assert ckpt is not None, f"no checkpoint-{checkpoint_step} in {run_dir}"
    else:
        ckpt = cks[-1]
    step = int(ckpt.rsplit("-", 1)[-1])
    rc = _find_run_config(ckpt) or os.path.join(run_dir, "run_config.yaml")
    run_cfg = yaml.safe_load(open(rc)) or {}
    base = run_cfg.get("model")
    tok = AutoTokenizer.from_pretrained(base)
    model = load_gradient_routing_model(ckpt, base_model=base)
    # Match the training dtype (the Modal shim hardcodes bf16 because all 8B
    # runs train bf16; 135M toy runs train fp32 — read the run's flag).
    if run_cfg.get("bf16"):
        model.to(torch.bfloat16)
    modes = [(f"fs{s:.1f}", float(retain_scale), float(s)) for s in scales]
    results = posthoc_eval_from_checkpoint(model, tok, ckpt, n_eval=n_eval,
                                           modes=modes, run_config_path=rc)

    out = {"run_name": run_name, "step": step, "n_eval": n_eval,
           "retain_scale": retain_scale, "scales": {}}
    for mname, mres in results.items():
        out["scales"][mname[2:]] = {
            k: v["mean"] for k, v in mres.get("metrics", {}).items()
            if isinstance(v, dict) and v.get("mean") is not None}
    # Suffixes COMPOSE: __step{N} then __r{r} (see the Modal shim).
    fname = run_name
    if checkpoint_step is not None:
        fname += f"__step{checkpoint_step}"
    if retain_scale != 1.0:
        fname += f"__r{retain_scale:.1f}"
    fname += ".json"
    with open(os.path.join(outdir, fname), "w") as f:
        json.dump(out, f, indent=2)
    metrics = list(next(iter(out["scales"].values())).keys()) if out["scales"] else []
    print(f"[fseval {run_name}] step={step} scales={sorted(out['scales'])} "
          f"metrics={metrics}")

    del model
    torch.cuda.empty_cache()
    return fname


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep_dir", required=True,
                    help="Sweep output dir (e.g. output/gr_v2suite-0714-1234)")
    ap.add_argument("--n_eval", type=int, default=256)
    ap.add_argument("--scales", default="",
                    help="CSV of forget scales; default 0.0..1.0 step 0.1")
    ap.add_argument("--retain_scale", type=float, default=1.0)
    ap.add_argument("--only", default="",
                    help="Substring filter on run dir basenames")
    ap.add_argument("--checkpoint_step", type=int, default=None)
    ap.add_argument("--gpu_id", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true",
                    help="Re-eval runs whose fseval JSON already exists")
    args = ap.parse_args()

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(args.gpu_id))

    sweep_dir = args.sweep_dir.rstrip("/")
    scales = ([float(s) for s in args.scales.split(",")] if args.scales
              else [round(0.1 * i, 1) for i in range(11)])
    run_dirs = sorted(
        d for d in glob.glob(os.path.join(sweep_dir, "*"))
        if os.path.isdir(d) and glob.glob(os.path.join(d, "checkpoint-*"))
        and args.only in os.path.basename(d))
    assert run_dirs, f"no run dirs with checkpoints in {sweep_dir} (only={args.only!r})"
    outdir = f"{sweep_dir}_fseval"
    os.makedirs(outdir, exist_ok=True)

    todo, skipped = [], 0
    for d in run_dirs:
        fname = os.path.basename(d)
        if args.checkpoint_step is not None:
            fname += f"__step{args.checkpoint_step}"
        if args.retain_scale != 1.0:
            fname += f"__r{args.retain_scale:.1f}"
        if not args.overwrite and os.path.exists(os.path.join(outdir, fname + ".json")):
            skipped += 1
            continue
        todo.append(d)
    print(f"[fseval] {len(todo)} run(s) to eval ({skipped} already done) -> {outdir}")

    failed = []
    for i, d in enumerate(todo):
        print(f"[fseval] ({i + 1}/{len(todo)}) {os.path.basename(d)}")
        try:
            eval_one(d, outdir, args.n_eval, scales, args.retain_scale,
                     args.checkpoint_step)
        except Exception as e:
            # keep going; a single bad run must not sink the batch, but the
            # failure list is printed loudly at the end.
            import traceback
            traceback.print_exc()
            failed.append((os.path.basename(d), repr(e)))
    if failed:
        print(f"[fseval] {len(failed)} FAILED:")
        for n, e in failed:
            print(f"  {n}: {e}")
        sys.exit(1)
    print(f"[fseval] all done -> {outdir}")


if __name__ == "__main__":
    main()
