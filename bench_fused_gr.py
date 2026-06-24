"""Accuracy gate + timing for the fused gradient-routing forward/backward.

Capture-and-replay, modeled on bench_training_step.py. Replays a cached rollout
batch into the first call of `_dynamic_microbatch_forward_backward`, where the
inputs are fully prepared (advantages, retain_advantages, is_rh, is_coherence,
is_verified_retain, old/ref logps). At that call it:

  1. ACCURACY GATE — runs the stock homogeneous-microbatch path and the fused
     single-pass path on identical inputs + model weights, zeroing grads
     between, and asserts every trainable adapter param's .grad matches to
     tolerance. Under loss_type="grpo" these are exactly equivalent (per-sequence
     normalization is additive across sequences); any mismatch is a real bug.

  2. TIMING — times stock vs fused fwd/bwd over the same batch (K reps each,
     CUDA-synchronized) and reports the speedup.

Then it hard-exits (does not run the full training loop).

Capture a batch first (canonical merged-coherence GR config):
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python train.py --sweep_config_run sweeps/sort_idea2a_periter99_gr.py:0 ... --max_steps 1 --save_batch /tmp/gr_coh_batch.pt --no_wandb
  (or drive params from the sweep config however you normally launch a single run; the
   only requirement is --save_batch and that the run uses the packed/liger + merged-coh path.)

Run the gate + timing:
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python bench_fused_gr.py --batch /tmp/gr_coh_batch.pt --sweep_config sweeps/sort_idea2a_periter99_gr.py --run_index 0
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python bench_fused_gr.py --batch /tmp/gr_coh_batch.pt --config configs/foo.yaml --reps 30 --rtol 2e-2

For a tight tolerance use an fp32 config (e.g. sweeps/sort_idea2a_periter99_fp32hf_gr.py);
bf16 runs need a looser --rtol because 3-pass vs 1-pass summation order differs.
"""

import argparse
import os
import sys
import time
from collections import defaultdict

import torch

import train
from bench_training_step import _load_sweep_config


SEED = 1234
_FORCE_FP32 = False


def _clone_inputs(inputs):
    out = {}
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            v = v.clone()
            if _FORCE_FP32 and v.is_floating_point():
                v = v.float()
            out[k] = v
        else:
            out[k] = v
    return out


def _adapter_named_params(model):
    """Trainable adapter params, by name (stable across runs)."""
    return [(n, p) for n, p in model.named_parameters() if p.requires_grad]


def _run_once(self, orig, model, inputs, num_items, fused):
    """Run one stock/fused fwd+bwd on a fresh grad state; return {name: grad}."""
    named = _adapter_named_params(model)
    self.optimizer.zero_grad(set_to_none=True)
    self._fused_reduction = fused
    torch.manual_seed(SEED)  # match any dropout RNG between the two runs
    orig(self, model, _clone_inputs(inputs), num_items, record_metrics=False)
    grads = {}
    for n, p in named:
        grads[n] = None if p.grad is None else p.grad.detach().float().clone()
    return grads


def _compare(g_stock, g_fused, rtol, atol):
    rows = []
    worst_rel = 0.0
    worst_abs = 0.0
    n_missing = 0
    for n in g_stock:
        a, b = g_stock[n], g_fused[n]
        if a is None and b is None:
            continue
        if (a is None) != (b is None):
            n_missing += 1
            rows.append((float("inf"), float("inf"), n,
                         "grad present in one path only"))
            continue
        abs_diff = (a - b).abs().max().item()
        denom = a.abs().max().item()
        rel_diff = abs_diff / denom if denom > 0 else (0.0 if abs_diff == 0 else float("inf"))
        worst_rel = max(worst_rel, rel_diff)
        worst_abs = max(worst_abs, abs_diff)
        rows.append((rel_diff, abs_diff, n, ""))
    rows.sort(key=lambda r: -r[0])
    return worst_rel, worst_abs, n_missing, rows


def _gate(self, orig, model, inputs, num_items, args):
    print("\n" + "=" * 70)
    print("ACCURACY GATE: stock homogeneous-microbatch vs fused single-pass")
    print("=" * 70)

    # Warn if dropout could desync the two runs' RNG.
    n_dropout = sum(1 for m in model.modules()
                    if isinstance(m, torch.nn.Dropout) and m.p > 0)
    if n_dropout:
        print(f"[warn] {n_dropout} Dropout(p>0) modules present — stock (3 fwd) and "
              f"fused (1 fwd) consume RNG differently, so grads may differ for reasons "
              f"unrelated to the reduction. Prefer a dropout-free (e.g. MLP-adapter) config.")

    dtype = next(p.dtype for _, p in _adapter_named_params(model))
    if args.rtol is None:
        rtol = 1e-4 if dtype in (torch.float32, torch.float64) else 2e-2
    else:
        rtol = args.rtol
    atol = args.atol

    g_stock = _run_once(self, orig, model, inputs, num_items, fused=False)
    g_fused = _run_once(self, orig, model, inputs, num_items, fused=True)
    self.optimizer.zero_grad(set_to_none=True)

    worst_rel, worst_abs, n_missing, rows = _compare(g_stock, g_fused, rtol, atol)

    n_params = sum(1 for _ in g_stock)
    print(f"adapter params: {n_params}   dtype: {dtype}   rtol: {rtol:g}  atol: {atol:g}")
    print(f"worst relative grad diff: {worst_rel:.3e}")
    print(f"worst absolute grad diff: {worst_abs:.3e}")
    if n_missing:
        print(f"[!] {n_missing} params have grad in only one path")
    print("\ntop-8 params by relative diff:")
    print(f"  {'rel':>10}  {'abs':>10}  name")
    for rel, ab, n, note in rows[:8]:
        print(f"  {rel:10.3e}  {ab:10.3e}  {n}  {note}")

    ok = (worst_rel <= rtol or worst_abs <= atol) and n_missing == 0
    print("\nGATE:", "PASS ✓" if ok else "FAIL ✗")
    print("=" * 70)
    if not ok:
        raise AssertionError(
            f"fused != stock: worst_rel={worst_rel:.3e} (rtol={rtol:g}), "
            f"worst_abs={worst_abs:.3e} (atol={atol:g}), n_missing={n_missing}")
    return ok


def _time_mode(self, orig, model, inputs, num_items, fused, reps, warmup):
    self._fused_reduction = fused
    for _ in range(warmup):
        self.optimizer.zero_grad(set_to_none=True)
        orig(self, model, _clone_inputs(inputs), num_items, record_metrics=False)
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        self.optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        orig(self, model, _clone_inputs(inputs), num_items, record_metrics=False)
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    self.optimizer.zero_grad(set_to_none=True)
    ts.sort()
    return ts


def _n_microbatches(self, orig, model, inputs, num_items, fused):
    """Run one path with metrics on; return its microbatch count."""
    self._fused_reduction = fused
    key = "dynamic_batching/n_microbatches"
    self._metrics["train"][key] = []
    self.optimizer.zero_grad(set_to_none=True)
    orig(self, model, _clone_inputs(inputs), num_items, record_metrics=True)
    self.optimizer.zero_grad(set_to_none=True)
    vals = self._metrics["train"].get(key, [])
    return int(vals[-1]) if vals else 0


def _timing(self, orig, model, inputs, num_items, args):
    print("\n" + "=" * 70)
    print(f"TIMING: fwd/bwd over the captured batch ({args.reps} reps, {args.warmup} warmup)")
    print("=" * 70)
    n_stock = _n_microbatches(self, orig, model, inputs, num_items, False)
    n_fused = _n_microbatches(self, orig, model, inputs, num_items, True)
    stock = _time_mode(self, orig, model, inputs, num_items, False, args.reps, args.warmup)
    fused = _time_mode(self, orig, model, inputs, num_items, True, args.reps, args.warmup)

    def stats(ts):
        return sum(ts) / len(ts), ts[len(ts) // 2]
    s_mean, s_med = stats(stock)
    f_mean, f_med = stats(fused)
    s_per = s_med / n_stock if n_stock else float("nan")
    f_per = f_med / n_fused if n_fused else float("nan")
    print(f"stock  ({n_stock:3d} homogeneous mbs): total median {s_med*1e3:8.2f} ms   "
          f"per-mb {s_per*1e3:7.2f} ms")
    print(f"fused  ({n_fused:3d} heterogeneous mbs): total median {f_med*1e3:8.2f} ms   "
          f"per-mb {f_per*1e3:7.2f} ms")
    print(f"total speedup (median): {s_med / f_med:.3f}x   (mean): {s_mean / f_mean:.3f}x")
    if n_stock and n_fused:
        print(f"per-microbatch ratio (fused/stock): {f_per / s_per:.3f}x  "
              f"(>1 = fused mb is more expensive; <1 = cheaper)")
    print("=" * 70)


def _make_interceptor(args):
    orig = train.SampleGRPOTrainer._dynamic_microbatch_forward_backward
    saved_metrics_swap = {}

    def _intercept(self, model, inputs, num_items_in_batch, *, record_metrics=True):
        # Only act on a fully-prepared routing batch (skip any earlier
        # diagnostic/off-policy fwd/bwd that lacks advantages/is_rh).
        if not ("advantages" in inputs and "is_rh" in inputs):
            return orig(self, model, inputs, num_items_in_batch, record_metrics=record_metrics)
        if _FORCE_FP32:
            model.float()  # pure-fp32 gate: autocast is already off (no mixed precision)
            # FlashAttention only supports fp16/bf16; switch to SDPA (supports fp32).
            base = self.accelerator.unwrap_model(model)
            cfg = getattr(base, "config", None)
            if cfg is not None:
                cfg._attn_implementation = "sdpa"
                if getattr(cfg, "text_config", None) is not None:
                    cfg.text_config._attn_implementation = "sdpa"
            for mod in base.modules():
                if hasattr(mod, "config") and hasattr(mod.config, "_attn_implementation"):
                    mod.config._attn_implementation = "sdpa"
                if hasattr(mod, "_attn_implementation"):
                    mod._attn_implementation = "sdpa"
        # Isolate loss-path metric appends (kl/clip_ratio) during the gate/timing.
        saved = self._metrics
        self._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
        try:
            _gate(self, orig, model, inputs, num_items_in_batch, args)
            _timing(self, orig, model, inputs, num_items_in_batch, args)
        finally:
            self._metrics = saved
        print("\n[bench_fused_gr] done — exiting before full training loop.")
        sys.stdout.flush()
        os._exit(0)

    train.SampleGRPOTrainer._dynamic_microbatch_forward_backward = _intercept


def _patch_generation_replay(batch_path):
    cached = torch.load(batch_path, weights_only=False)
    shapes = {k: tuple(v.shape) for k, v in cached.items() if isinstance(v, torch.Tensor)}
    print(f"[bench_fused_gr] loaded batch {batch_path}: {shapes}")

    def _replay(self, inputs):
        device = self.accelerator.device
        out = {}
        for k, v in cached.items():
            out[k] = v.to(device) if isinstance(v, torch.Tensor) else v
        n = out["prompt_ids"].shape[0]
        if "is_rh" not in out:
            out["is_rh"] = torch.zeros(n, dtype=torch.bool, device=device)
        self._last_rollout_time = 0.0
        return out

    train.SampleGRPOTrainer._generate_and_score_completions = _replay


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--batch", required=True, help="cached batch .pt (from --save_batch)")
    p.add_argument("--sweep_config", default=None)
    p.add_argument("--run_index", type=int, default=0)
    p.add_argument("--config", default=None, help="YAML config (if not using --sweep_config)")
    p.add_argument("--reps", type=int, default=20)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--rtol", type=float, default=None,
                   help="relative grad tolerance (default 1e-4 fp32, 2e-2 bf16/fp16)")
    p.add_argument("--atol", type=float, default=1e-6)
    p.add_argument("--force_fp32", action="store_true",
                   help="convert the model to fp32 for the gate (SmolLM2 loads bf16 by "
                        "config default even with bf16=False); also disables torch.compile")
    args, remaining = p.parse_known_args()
    global _FORCE_FP32
    _FORCE_FP32 = args.force_fp32

    if args.sweep_config:
        params = _load_sweep_config(args.sweep_config, args.run_index)
    elif args.config:
        params = {"config": args.config}
    else:
        raise SystemExit("provide --sweep_config or --config")

    # Layer explicit CLI overrides (same sentinel trick as bench_training_step).
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

    # Profiling-safe settings; isolate the bench from wandb/eval/checkpoints/vLLM
    # and from the grad diagnostic (separate fwd/bwd path).
    params["no_wandb"] = True
    params["eval_every"] = 0
    params["adapter_diag_interval"] = "off"
    params["routing_trace_interval"] = "off"
    params["save_steps"] = 999999
    params["max_steps"] = 1
    params.pop("vllm_server", None)
    params["vllm_spawn"] = False
    params["vllm_colocate"] = False
    params["vllm_async"] = False
    for sweep_only in ("vllm_dtype", "per_gpu"):
        params.pop(sweep_only, None)
    import tempfile
    params["output_dir"] = tempfile.mkdtemp(prefix="bench_fused_")
    params["fused_reduction"] = False  # toggled manually inside the interceptor
    if args.force_fp32:
        params["torch_compile"] = False  # so model.float() can't desync a compiled graph

    _patch_generation_replay(args.batch)
    _make_interceptor(args)

    train.train_main(params)


if __name__ == "__main__":
    main()
