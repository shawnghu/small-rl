"""Cross-env fseval: evaluate checkpoints trained on ANOTHER env under the
countdown hf100 protocol, emitting the exact fseval JSON the hf100 scatter
scripts read.

The batch fseval drivers (modal eval_forget_scales_one, tools/local_fseval)
read the base model AND the eval env from the same run_config, so a
leetcode-trained checkpoint gets evaluated on leetcode. This wrapper splits
the two: the base model comes from the checkpoint's OWN run_config (correct
weights), while the eval env/prompts/metrics/protocol come from a donor
COUNTDOWN run_config (environment countdown_code, hack_frac 1.0,
max_completion_length 1536, embedded reward components + detectors).

Usage (on the training box):
    python tools/crossenv_countdown_fseval.py \
        --sweep_dir output/lconly_dose-XXXX \
        --countdown_run_config output/countdown_hf100_gr_lccoh64_lr3_seeds5/cdhf100_gr_lccoh64_lr3_s4/run_config.yaml \
        --out_dir output/countdown_hf100_lconly_fseval [--scales 1.0] [--gpu_id 0]

Single-policy runs (routing_mode none) are the intended target: default
scales=[1.0] evaluates the plain trained policy, matching how RP/DN/IP
baselines appear in the hf100 figures (scale key "1.0").
"""
import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def eval_one(run_dir, out_dir, countdown_rc, n_eval, scales):
    import torch
    import yaml
    from transformers import AutoTokenizer
    from eval_utils import (_find_run_config, load_gradient_routing_model,
                            posthoc_eval_from_checkpoint)

    run_name = os.path.basename(run_dir.rstrip("/"))
    cks = sorted(glob.glob(os.path.join(run_dir, "checkpoint-*")),
                 key=lambda p: int(p.rsplit("-", 1)[-1]))
    assert cks, f"no checkpoints in {run_dir}"
    ckpt = cks[-1]
    step = int(ckpt.rsplit("-", 1)[-1])

    own_rc = _find_run_config(ckpt) or os.path.join(run_dir, "run_config.yaml")
    own_cfg = yaml.safe_load(open(own_rc)) or {}
    base = own_cfg.get("model")
    assert base, f"no model in {own_rc}"
    cd_cfg = yaml.safe_load(open(countdown_rc)) or {}
    assert cd_cfg.get("environment") == "countdown_code", (
        f"donor run_config must be a countdown_code run (got "
        f"{cd_cfg.get('environment')!r} from {countdown_rc})")

    tok = AutoTokenizer.from_pretrained(base)
    model = load_gradient_routing_model(ckpt, base_model=base)
    if own_cfg.get("bf16"):
        model.to(torch.bfloat16)
    modes = [(f"fs{s:.1f}", 1.0, float(s)) for s in scales]
    # gen_batch_size: 8B x 1536 new tokens x 256 prompts OOMs an 80GB H100 in
    # one generate call; 64-prompt chunks keep KV cache ~25GB.
    results = posthoc_eval_from_checkpoint(model, tok, ckpt, n_eval=n_eval,
                                           modes=modes,
                                           run_config_path=countdown_rc,
                                           gen_batch_size=64)

    out = {"run_name": run_name, "step": step, "n_eval": n_eval,
           "retain_scale": 1.0, "crossenv_from": own_cfg.get("environment"),
           "scales": {}}
    for mname, mres in results.items():
        out["scales"][mname[2:]] = {
            k: v["mean"] for k, v in mres.get("metrics", {}).items()
            if isinstance(v, dict) and v.get("mean") is not None}
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, run_name + ".json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"[crossenv {run_name}] step={step} scales={sorted(out['scales'])}")
    del model
    torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep_dir", required=True)
    ap.add_argument("--countdown_run_config", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--n_eval", type=int, default=256)
    ap.add_argument("--scales", default="1.0", help="CSV of forget scales")
    ap.add_argument("--only", default="")
    ap.add_argument("--gpu_id", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(args.gpu_id))
    scales = [float(s) for s in args.scales.split(",")]
    run_dirs = sorted(
        d for d in glob.glob(os.path.join(args.sweep_dir.rstrip("/"), "*"))
        if os.path.isdir(d) and glob.glob(os.path.join(d, "checkpoint-*"))
        and args.only in os.path.basename(d))
    assert run_dirs, f"no run dirs with checkpoints in {args.sweep_dir}"

    failed = []
    for d in run_dirs:
        name = os.path.basename(d)
        if not args.overwrite and os.path.exists(os.path.join(args.out_dir, name + ".json")):
            print(f"[crossenv] skip {name} (exists)")
            continue
        try:
            eval_one(d, args.out_dir, args.countdown_run_config, args.n_eval, scales)
        except Exception as e:
            import traceback
            traceback.print_exc()
            failed.append((name, repr(e)))
    if failed:
        print(f"[crossenv] {len(failed)} FAILED: {failed}")
        sys.exit(1)
    print(f"[crossenv] done -> {args.out_dir}")


if __name__ == "__main__":
    main()
