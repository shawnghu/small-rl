"""Correctness gate for the compiled vLLM engine (enforce_eager=False).

The historical failure mode of the compiled path was SILENT (slower learning,
no crash), so the gate checks the two ways a compiled/CUDA-graphed engine can
be silently wrong with our injected adapters:

  1. EQUIVALENCE: greedy decode of fixed prompts must (near-)match the eager
     engine token-for-token — at initial weights, after a weight update, and
     after set_scales changes. (Compiled kernel fusion can reorder fp math and
     legitimately flip argmax on near-ties, so the gate requires >= --min_match
     per-token agreement rather than bitwise equality, and reports the rate.)

  2. STALENESS (the sharp check for graph stale-pointer bugs): after pushing
     DIFFERENT weights / scales to the compiled engine, its outputs MUST change
     vs its own pre-update outputs. Greedy TOKENS are too blunt for this — a
     16-neuron adapter's logit perturbation rarely flips argmax (verified on
     the HF side: std=0.3 down-proj noise, zero token flips) — so the check
     compares PER-TOKEN LOGPROBS. Within one engine greedy decode is
     deterministic (a same-condition replicate pair calibrates the noise
     floor), so a stale graph reproduces logprobs exactly, while live weights
     shift them by O(1e-2). Reaction requires
     max|dlogprob| > max(1e-4, 10 x replicate noise).

CUDA_VISIBLE_DEVICES=0 .venv/bin/python tools/gate_compiled_engine.py
"""
import argparse
import multiprocessing as mp
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _spawn(model, mlp_config, enforce_eager, mem, tag, cudagraph_mode=None):
    from train import _spawn_vllm_server
    from vllm_lifecycle import wait_for_ready_file
    from vllm_client import VLLMClient
    from train import MLP_PRESETS
    preset = MLP_PRESETS[mlp_config]
    sock = f"ipc:///tmp/vllm_gate_{tag}_{os.getpid()}.sock"
    ready = tempfile.mktemp(prefix=f"vllm_ready_gate_{tag}_")
    ctx = mp.get_context("spawn")
    proc = ctx.Process(target=_spawn_vllm_server,
                       args=(model, mlp_config, mem, sock, ready, 0.0, 1.0,
                             preset["layer_stride"], 2, 0, f"gate_{tag}"),
                       kwargs={"enforce_eager": enforce_eager,
                               "cudagraph_mode": (cudagraph_mode if not enforce_eager else None)})
    proc.start()
    wait_for_ready_file(ready, proc, f"gate {tag} server")
    return proc, VLLMClient(sock)


def _gen(client, eid, prompts, max_tokens):
    # Greedy: temperature 0 (vLLM maps to greedy sampling).
    _, comp_ids, _, logprobs = client.generate(eid, prompts, 1, 0.0, max_tokens,
                                               top_k=-1, top_p=1.0,
                                               return_logprobs=True)
    return comp_ids, logprobs


def _match_rate(a, b):
    tok_total = tok_match = 0
    for x, y in zip(a, b):
        n = max(len(x), len(y))
        tok_total += n
        tok_match += sum(1 for i in range(min(len(x), len(y))) if x[i] == y[i])
    return tok_match / max(1, tok_total)


def _max_lp_delta(lp_a, lp_b):
    """Max abs per-token logprob delta over the common prefix of each pair."""
    d = 0.0
    for x, y in zip(lp_a, lp_b):
        for i in range(min(len(x), len(y))):
            d = max(d, abs(x[i] - y[i]))
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M-Instruct")
    ap.add_argument("--mlp_config", default="m16")
    ap.add_argument("--n_prompts", type=int, default=64)
    ap.add_argument("--max_tokens", type=int, default=48)
    ap.add_argument("--min_match", type=float, default=0.98)
    ap.add_argument("--vllm_gpu_memory", type=float, default=0.25)
    ap.add_argument("--cudagraph_mode", default=None,
                    help="compiled-arm cudagraph mode (e.g. FULL_AND_PIECEWISE)")
    args = ap.parse_args()

    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from gradient_routing import apply_dual_mlp
    from train import MLP_PRESETS

    tok = AutoTokenizer.from_pretrained(args.model)
    preset = MLP_PRESETS[args.mlp_config]

    def make_weights(seed):
        torch.manual_seed(seed)
        m = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32)
        apply_dual_mlp(m, preset["retain_neurons"], preset["forget_neurons"],
                       layer_stride=preset["layer_stride"])
        # Non-trivial adapters: randomize the zero-init down projections so the
        # adapters actually perturb logits (otherwise the gate can't see them).
        from gradient_routing import DualMLPAdapter
        with torch.no_grad():
            for mod in m.modules():
                if isinstance(mod, DualMLPAdapter):
                    mod.down_retain.weight.normal_(0, 0.02)
                    mod.down_forget.weight.normal_(0, 0.02)
        return m

    model_v1 = make_weights(101)
    model_v2 = make_weights(202)  # distinctly different adapter weights

    texts = [f"Answer briefly and politely. What is the color of object number {i}, "
             f"and what category does it belong to?" for i in range(args.n_prompts)]
    prompts = [tok.encode(t, add_special_tokens=False)[:40] for t in texts]

    results = {}
    for tag, eager in (("eager", True), ("compiled", False)):
        proc, client = _spawn(args.model, args.mlp_config, eager,
                              args.vllm_gpu_memory, tag,
                              cudagraph_mode=args.cudagraph_mode)
        eid = client.register()
        out = {}
        client.update_weights_from_model(eid, model_v1)
        client.set_scales(eid, 1.0, 1.0)
        out["v1_s11"] = _gen(client, eid, prompts, args.max_tokens)
        out["v1_s11_rep"] = _gen(client, eid, prompts, args.max_tokens)  # noise floor
        client.set_scales(eid, 1.0, 0.0)           # scale change, same weights
        out["v1_s10"] = _gen(client, eid, prompts, args.max_tokens)
        client.update_weights_from_model(eid, model_v2)   # weight update
        client.set_scales(eid, 1.0, 1.0)
        out["v2_s11"] = _gen(client, eid, prompts, args.max_tokens)
        results[tag] = out
        client.shutdown()
        proc.join(timeout=10)
        from vllm_lifecycle import killpg_cleanup
        killpg_cleanup(proc)
        print(f"[gate] {tag} engine done")

    print("\n=== EQUIVALENCE (compiled vs eager, greedy) ===")
    ok = True
    for cond in ("v1_s11", "v1_s10", "v2_s11"):
        r = _match_rate(results["eager"][cond][0], results["compiled"][cond][0])
        flag = "PASS" if r >= args.min_match else "FAIL"
        ok &= r >= args.min_match
        print(f"  {cond}: token match {r:.4f}  [{flag}]  (threshold {args.min_match})")

    print("\n=== STALENESS (compiled engine must react to updates; logprob-based) ===")
    for tag in ("compiled", "eager"):
        out = results[tag]
        noise = _max_lp_delta(out["v1_s11"][1], out["v1_s11_rep"][1])
        thresh = max(1e-4, 10 * noise)
        print(f"  [{tag}] replicate noise floor: max|dlp| {noise:.2e} -> reaction threshold {thresh:.2e}")
        for b, what in (("v1_s10", "set_scales change"), ("v2_s11", "weight update")):
            d = _max_lp_delta(out["v1_s11"][1], out[b][1])
            passed = d > thresh
            flag = "PASS" if passed else "FAIL (logprobs unchanged -> stale state?)"
            if tag == "compiled":
                ok &= passed
            print(f"  [{tag}] {what}: max|dlogprob| {d:.2e}  [{flag}]")

    print("\nGATE:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
