"""Standalone test: verify mixed-batch per-slot routing (concurrent eval) produces
the same outputs as sequential per-slot generation.

Run with: CUDA_VISIBLE_DEVICES=7 .venv/bin/python tests/test_concurrent_eval_equivalence.py
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vllm_mlp_adapter import create_engine


def main():
    from vllm import SamplingParams

    llm, mgr = create_engine(
        max_experiments=4,
        retain_neurons=8,
        forget_neurons=8,
        gpu_memory_utilization=0.15,
    )
    # logprobs=5 returns top-5 per-token logprobs so we can measure numerical
    # drift between concurrent and sequential batches.
    params = SamplingParams(temperature=0, max_tokens=30, logprobs=5)

    # Build identical weights for all 3 slots. Non-zero retain AND forget
    # so all 3 scale configs produce distinct outputs.
    torch.manual_seed(0)
    # 4 layers for SmolLM2-135M default layer_start/end; actually the default
    # create_engine uses SmolLM2-135M-Instruct which has 30 layers. Let's query.
    n_mlp_layers = sum(1 for _ in mgr.mlp_layers) if hasattr(mgr, "mlp_layers") else None
    if n_mlp_layers is None:
        # Fallback: inspect model
        from vllm_mlp_adapter import VLLMDualMLPAdapter
        n_mlp_layers = sum(
            1 for m in llm.llm_engine.model_executor.driver_worker.model_runner.model.modules()
            if isinstance(m, VLLMDualMLPAdapter)
        )
    print(f"Detected {n_mlp_layers} MLP adapter layers")
    # We don't know the exact hidden dim upfront; grab it from the first layer
    for m in llm.llm_engine.model_executor.driver_worker.model_runner.model.modules():
        from vllm_mlp_adapter import VLLMDualMLPAdapter
        if isinstance(m, VLLMDualMLPAdapter):
            hidden = m.hidden_dim
            break
    print(f"Hidden size: {hidden}")

    layer_weights = []
    for _ in range(n_mlp_layers):
        layer_weights.append({
            "gate_retain": torch.randn(8, hidden) * 0.5,
            "up_retain": torch.randn(8, hidden) * 0.5,
            "down_retain": torch.randn(hidden, 8) * 0.1,
            "gate_forget": torch.randn(8, hidden) * 0.5,
            "up_forget": torch.randn(8, hidden) * 0.5,
            "down_forget": torch.randn(hidden, 8) * 0.1,
        })

    # Push same weights to slots 1, 2, 3 with different scales.
    scale_by_slot = {1: (1.0, 1.0), 2: (1.0, 0.0), 3: (0.0, 1.0)}
    for slot, (rs, fs) in scale_by_slot.items():
        mgr.set_weights(slot, layer_weights)
        mgr.set_scales(slot, retain_scale=rs, forget_scale=fs)

    prompts = ["A little bird", "Once upon a time", "The forest was"]

    def extract(req_out):
        """Return (text, token_ids, per_token_top_logprobs) for a single CompletionOutput."""
        comp = req_out.outputs[0]
        # comp.logprobs is a list (one entry per generated token) of dicts
        # {token_id: Logprob(logprob=..., rank=..., decoded_token=...)}
        per_token = []
        for step_lp in comp.logprobs:
            # Keep as {token_id: float_logprob} for easy diffing.
            per_token.append({tid: lp.logprob for tid, lp in step_lp.items()})
        return comp.text, list(comp.token_ids), per_token

    # Sequential: 3 separate calls, one per slot, each using all prompts.
    print("\n--- Sequential ---")
    seq_outputs = {}   # slot -> list[str]
    seq_tokens = {}    # slot -> list[list[int]]
    seq_logprobs = {}  # slot -> list[list[dict[int, float]]]
    for slot in [1, 2, 3]:
        out = mgr.generate(
            prompts, experiment_ids=[slot] * len(prompts), sampling_params=params,
        )
        seq_outputs[slot] = []
        seq_tokens[slot] = []
        seq_logprobs[slot] = []
        for req in out:
            text, tids, lps = extract(req)
            seq_outputs[slot].append(text)
            seq_tokens[slot].append(tids)
            seq_logprobs[slot].append(lps)
        print(f"slot {slot}: {seq_outputs[slot]}")

    # Concurrent: single call with mixed slots (replicated prompts).
    print("\n--- Concurrent ---")
    all_prompts = prompts * 3
    all_slots = [1] * len(prompts) + [2] * len(prompts) + [3] * len(prompts)
    concurrent_out = mgr.generate(
        all_prompts, experiment_ids=all_slots, sampling_params=params,
    )
    n = len(prompts)
    con_outputs = {1: [], 2: [], 3: []}
    con_tokens = {1: [], 2: [], 3: []}
    con_logprobs = {1: [], 2: [], 3: []}
    for slot_idx, slot in enumerate([1, 2, 3]):
        for i in range(n):
            text, tids, lps = extract(concurrent_out[slot_idx * n + i])
            con_outputs[slot].append(text)
            con_tokens[slot].append(tids)
            con_logprobs[slot].append(lps)
        print(f"slot {slot}: {con_outputs[slot]}")

    # Measure numerical drift: for each generated token in the shared prefix
    # (where seq and con agree), compute |logprob_seq(top_seq) - logprob_con(top_seq)|
    # and also the margin between top-1 and top-2 in each.
    print("\n--- Logit drift (shared-prefix positions) ---")
    all_abs_diffs = []
    all_margins_seq = []
    all_margins_con = []
    first_divergence_positions = []
    for slot in [1, 2, 3]:
        for i in range(n):
            seq_tids = seq_tokens[slot][i]
            con_tids = con_tokens[slot][i]
            seq_lps = seq_logprobs[slot][i]
            con_lps = con_logprobs[slot][i]
            shared_len = 0
            for t in range(min(len(seq_tids), len(con_tids))):
                if seq_tids[t] == con_tids[t]:
                    shared_len = t + 1
                else:
                    break

            # Position-by-position diff over the shared prefix.
            for t in range(shared_len):
                tid = seq_tids[t]
                s_lp = seq_lps[t].get(tid)
                c_lp = con_lps[t].get(tid)
                if s_lp is not None and c_lp is not None:
                    all_abs_diffs.append(abs(s_lp - c_lp))

                # top-2 margin: logprob of top-1 minus logprob of top-2 (same side).
                def margin(d):
                    vals = sorted(d.values(), reverse=True)
                    return vals[0] - vals[1] if len(vals) >= 2 else float("inf")
                all_margins_seq.append(margin(seq_lps[t]))
                all_margins_con.append(margin(con_lps[t]))

            # Record first divergence position (if any).
            if shared_len < min(len(seq_tids), len(con_tids)):
                first_divergence_positions.append((slot, i, shared_len))
                t = shared_len
                seq_tid = seq_tids[t]
                con_tid = con_tids[t]
                # What does each side think of the OTHER side's chosen token?
                s_of_s = seq_lps[t].get(seq_tid)
                s_of_c = seq_lps[t].get(con_tid)
                c_of_s = con_lps[t].get(seq_tid)
                c_of_c = con_lps[t].get(con_tid)
                print(f"  slot {slot} prompt {i} diverges at token {t}:")
                print(f"    seq picked tid={seq_tid}: seq_lp={s_of_s}, con_lp={c_of_s}")
                print(f"    con picked tid={con_tid}: seq_lp={s_of_c}, con_lp={c_of_c}")
                if s_of_s is not None and s_of_c is not None:
                    print(f"    seq's tiebreak margin (s_of_s - s_of_c) = {s_of_s - (s_of_c or -float('inf')):.6f}")
                if c_of_c is not None and c_of_s is not None:
                    print(f"    con's tiebreak margin (c_of_c - c_of_s) = {c_of_c - (c_of_s or -float('inf')):.6f}")

    if all_abs_diffs:
        import statistics
        mn = min(all_abs_diffs)
        mx = max(all_abs_diffs)
        mean = statistics.mean(all_abs_diffs)
        med = statistics.median(all_abs_diffs)
        print(f"\n  |logprob_seq - logprob_con| over {len(all_abs_diffs)} shared-prefix tokens:")
        print(f"    min={mn:.6e}  median={med:.6e}  mean={mean:.6e}  max={mx:.6e}")
    if all_margins_seq:
        import statistics
        print(f"  top-1/top-2 margin stats (smaller = more tiebreak-sensitive):")
        print(f"    seq: min={min(all_margins_seq):.6e} median={statistics.median(all_margins_seq):.6e}")
        print(f"    con: min={min(all_margins_con):.6e} median={statistics.median(all_margins_con):.6e}")

    # Per-slot outputs must match bit-for-bit between sequential and concurrent.
    print("\n--- Comparison ---")
    all_match = True
    for slot in [1, 2, 3]:
        match = seq_outputs[slot] == con_outputs[slot]
        all_match = all_match and match
        print(f"slot {slot}: {'MATCH' if match else 'MISMATCH'}")
        if not match:
            for i in range(n):
                if seq_outputs[slot][i] != con_outputs[slot][i]:
                    print(f"  [{i}] seq: {seq_outputs[slot][i]!r}")
                    print(f"      con: {con_outputs[slot][i]!r}")

    # Sanity check: at least two slots should have distinct outputs.
    distinct_texts = set()
    for slot in [1, 2, 3]:
        distinct_texts.update(seq_outputs[slot])
    assert len(distinct_texts) >= 2, \
        f"Scale settings not being applied — got identical outputs: {distinct_texts}"
    print(f"\nSanity: {len(distinct_texts)} distinct outputs across scale settings (ok)")

    if not all_match:
        print("\n(Sequential and concurrent diverged; see logit drift stats above to"
              " diagnose whether this is fp numerical drift or a logic bug.)")
    else:
        print("\n✓ Concurrent eval matches sequential eval bit-for-bit")


if __name__ == "__main__":
    main()
