"""Step-by-step training equivalence test: HF generation vs vLLM Punica generation.

Runs a few RL training steps, and at each step compares:
  1. Generated token IDs from vLLM vs HF (given same adapter weights + prompts)
  2. Per-token logprobs from vLLM vs HF
  3. Rewards computed from both sets of generations

This isolates whether the vLLM Punica-based forward pass produces different
generations than HF's nn.Linear-based forward, which would cause divergent
training trajectories.

Also contains the PPS steering kernel-match parity test (go/no-go for the
positive-preventative-steering lift): the same {layer: vector} + alpha applied
via gradient_routing.set_pps_steering (HF trainer stack) and
VLLMAdapterManager.set_steering (vLLM rollout stack) must yield matching
per-token logprobs; one-sided application must diverge (negative control).

Usage:
    # full: generation equivalence + PPS steering parity
    CUDA_VISIBLE_DEVICES=0 .venv/bin/python tests/test_training_equivalence.py
    # just the PPS steering parity go/no-go
    CUDA_VISIBLE_DEVICES=0 .venv/bin/python tests/test_training_equivalence.py --pps-only
    # pytest-style (skips cleanly when CUDA / vLLM are unavailable, e.g. CPU CI)
    .venv/bin/pytest tests/test_training_equivalence.py -s
"""

import os
import sys
import hashlib
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DEVICE = "cuda"
MODEL_NAME = "HuggingFaceTB/SmolLM2-135M-Instruct"
RETAIN_NEURONS = 16
FORGET_NEURONS = 16
MAX_NEW_TOKENS = 64
NUM_PROMPTS = 8
NUM_STEPS = 3
LR = 1e-3
SEED = 42


def hash_weights(model):
    """Hash all adapter weights for quick comparison."""
    from gradient_routing import DualMLPAdapter
    h = hashlib.md5()
    for module in model.modules():
        if isinstance(module, DualMLPAdapter):
            for name in ["gate_retain", "up_retain", "down_retain",
                         "gate_forget", "up_forget", "down_forget"]:
                attr = getattr(module, name)
                if attr is not None:
                    h.update(attr.weight.data.cpu().numpy().tobytes())
    return h.hexdigest()[:12]


def generate_hf(hf_model, tokenizer, prompts, max_new_tokens, temperature=0):
    """Generate with HF model, return (token_ids_list, texts_list)."""
    all_ids = []
    all_texts = []
    for prompt in prompts:
        input_ids = tokenizer(prompt, return_tensors="pt",
                              add_special_tokens=False).input_ids.to(DEVICE)
        with torch.no_grad():
            out = hf_model.generate(
                input_ids, max_new_tokens=max_new_tokens,
                do_sample=False, eos_token_id=1,
            )
        comp_ids = out[0][input_ids.shape[1]:].tolist()
        comp_text = tokenizer.decode(comp_ids, skip_special_tokens=True)
        all_ids.append(comp_ids)
        all_texts.append(comp_text)
    return all_ids, all_texts


def generate_vllm(mgr, prompts, max_new_tokens, temperature=0):
    """Generate with vLLM engine, return (token_ids_list, texts_list)."""
    from vllm import SamplingParams
    sp = SamplingParams(temperature=temperature, max_tokens=max_new_tokens)
    outputs = mgr.generate(prompts, experiment_ids=[1] * len(prompts),
                           sampling_params=sp)
    all_ids = []
    all_texts = []
    for out in outputs:
        comp = out.outputs[0]
        all_ids.append(list(comp.token_ids))
        all_texts.append(comp.text)
    return all_ids, all_texts


def get_first_token_logprobs(hf_model, tokenizer, prompts):
    """Get logprobs for first generated token from HF model."""
    results = []
    for prompt in prompts:
        input_ids = tokenizer(prompt, return_tensors="pt",
                              add_special_tokens=False).input_ids.to(DEVICE)
        with torch.no_grad():
            logits = hf_model(input_ids).logits[0, -1, :]
        lps = torch.log_softmax(logits, dim=-1)
        top5 = torch.topk(logits, 5)
        results.append({
            "top_ids": top5.indices.tolist(),
            "top_logprobs": lps[top5.indices].tolist(),
        })
    return results


def compare_generations(hf_ids, vllm_ids, label=""):
    """Compare token ID lists, return (num_match, num_total, first_diverge_positions)."""
    assert len(hf_ids) == len(vllm_ids)
    n_match = 0
    diverge_positions = []
    for i, (h, v) in enumerate(zip(hf_ids, vllm_ids)):
        if h == v:
            n_match += 1
        else:
            # Find first divergence
            pos = 0
            for pos, (ht, vt) in enumerate(zip(h, v)):
                if ht != vt:
                    break
            else:
                pos = min(len(h), len(v))
            diverge_positions.append((i, pos, len(h), len(v)))
    return n_match, len(hf_ids), diverge_positions


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from gradient_routing import apply_dual_mlp, DualMLPAdapter
    from vllm_mlp_adapter import create_engine

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print(f"Model: {MODEL_NAME}")
    print(f"Adapter: retain={RETAIN_NEURONS}, forget={FORGET_NEURONS}")
    print(f"Steps: {NUM_STEPS}, prompts/step: {NUM_PROMPTS}, max_tokens: {MAX_NEW_TOKENS}")
    print()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # --- HF model ---
    hf_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16,
    ).to(DEVICE)
    apply_dual_mlp(hf_model, RETAIN_NEURONS, FORGET_NEURONS)

    # --- vLLM engine ---
    llm, mgr = create_engine(
        model_name=MODEL_NAME,
        max_experiments=2,
        retain_neurons=RETAIN_NEURONS,
        forget_neurons=FORGET_NEURONS,
        gpu_memory_utilization=0.05,
        dtype="float16",
    )

    # Also create an enforce_eager engine for comparison
    # (reuses the same vLLM install, just different config)

    # Fixed prompts for consistency
    test_prompts = [
        "Once upon a time",
        "The little cat",
        "A boy named",
        "In the morning",
        "She wanted to",
        "The big red",
        "One day a",
        "There was a",
    ][:NUM_PROMPTS]

    # Simulate a few "training steps": randomize adapter weights, compare generation
    print("=" * 70)
    print("STEP-BY-STEP GENERATION COMPARISON")
    print("=" * 70)

    optimizer = torch.optim.Adam(
        [p for p in hf_model.parameters() if p.requires_grad], lr=LR,
    )

    for step in range(NUM_STEPS):
        print(f"\n--- Step {step} ---")

        # Sync weights to vLLM
        mgr.update_from_training_model(1, hf_model)
        wh = hash_weights(hf_model)
        print(f"  Weight hash: {wh}")

        # Generate with both
        hf_ids, hf_texts = generate_hf(hf_model, tokenizer, test_prompts, MAX_NEW_TOKENS)
        vllm_ids, vllm_texts = generate_vllm(mgr, test_prompts, MAX_NEW_TOKENS)

        # Compare
        n_match, n_total, divergences = compare_generations(hf_ids, vllm_ids)
        print(f"  Generation match: {n_match}/{n_total}")

        if divergences:
            for (idx, pos, hlen, vlen) in divergences[:3]:
                print(f"    Prompt {idx} ({test_prompts[idx]!r}): first diverge at token {pos}")
                print(f"      HF [{hlen} tokens]:   ...{hf_ids[idx][max(0,pos-2):pos+3]}")
                print(f"      vLLM [{vlen} tokens]: ...{vllm_ids[idx][max(0,pos-2):pos+3]}")
                print(f"      HF text:   {hf_texts[idx][:60]!r}")
                print(f"      vLLM text: {vllm_texts[idx][:60]!r}")
        else:
            print(f"    All {n_total} generations identical!")

        # Compare first-token logprobs
        hf_lps = get_first_token_logprobs(hf_model, tokenizer, test_prompts[:3])
        from vllm import SamplingParams
        sp_lp = SamplingParams(temperature=0, max_tokens=1, logprobs=20)
        vllm_lp_outs = mgr.generate(
            test_prompts[:3], experiment_ids=[1, 1, 1], sampling_params=sp_lp,
        )
        print(f"  First-token logprob comparison (top-3 prompts):")
        for i in range(min(3, len(test_prompts))):
            vllm_lp_dict = vllm_lp_outs[i].outputs[0].logprobs[0]
            max_diff = 0.0
            for tid, hf_lp in zip(hf_lps[i]["top_ids"][:3], hf_lps[i]["top_logprobs"][:3]):
                vlp = vllm_lp_dict.get(tid)
                vlp_val = vlp.logprob if vlp is not None else float('nan')
                diff = abs(hf_lp - vlp_val)
                max_diff = max(max_diff, diff)
            print(f"    Prompt {i}: max logprob diff = {max_diff:.6f}")

        # Simulate a "training step" — random gradient to change weights
        if step < NUM_STEPS - 1:
            optimizer.zero_grad()
            # Fake loss: just sum of adapter outputs on a random input
            dummy_input = torch.randn(4, hf_model.config.hidden_size,
                                      dtype=torch.float16, device=DEVICE)
            loss = torch.tensor(0.0, device=DEVICE, requires_grad=True)
            for module in hf_model.modules():
                if isinstance(module, DualMLPAdapter):
                    out = module(dummy_input)
                    loss = loss + out.sum() * 0.001
            loss.backward()
            optimizer.step()
            print(f"  Applied fake training step (loss={loss.item():.4f})")

    print("\n" + "=" * 70)
    print("DONE")


# ---------------------------------------------------------------------------
# PPS steering parity (kernel-match go/no-go)
# ---------------------------------------------------------------------------
# Verifies the shared-interface contract for PPS steering: adding
# alpha * v[L] to the wrapped-MLP output of decoder layer L must produce
# IDENTICAL per-token logprobs in the HF trainer stack
# (gradient_routing.set_pps_steering) and the vLLM rollout stack
# (VLLMAdapterManager.set_steering). A one-sided application is exactly the
# IS-ratio-corrupting kernel-mismatch bug this repo documents as its #1
# training failure mode, so this test must have teeth: the negative controls
# below assert that one-sided steering DIVERGES beyond tolerance.
#
# Methodology: vLLM generates greedily with sampled-token logprobs (the
# sampled token is always present in the returned logprob dict, so no top-k
# assumptions), and HF teacher-forces the SAME token ids to get its per-token
# logprobs — a true multi-token comparison of the exact quantity the IS ratio
# consumes, with no retokenization (token-id prompts on both sides).
#
# Tolerance semantics: the pre-existing check in main() prints the HF-vs-vLLM
# max logprob diff without asserting a number. Here the unsteered parity of
# the very same stacks/pipeline is MEASURED in-test, and the steered parity
# must stay within PPS_PARITY_FACTOR x that baseline (with an absolute floor
# for the near-zero-baseline case). The negative controls must exceed the
# SAME tol, so inflating the tolerance can only FAIL the test, never make it
# pass vacuously.

PPS_ALPHA = 4.0            # fixed by the interface contract
PPS_MAX_TOKENS = 32        # completion tokens compared per prompt
PPS_PROMPTS = [
    "Once upon a time",
    "The little cat",
    "A boy named",
    "In the morning",
    "She wanted to",
    "The big red",
]
PPS_PARITY_FLOOR = 0.02    # abs tolerance floor (fp16 cross-kernel noise scale)
PPS_PARITY_FACTOR = 3.0    # steered parity allowed up to this x unsteered parity
PPS_BASE_SANITY = 0.10     # unsteered parity above this = harness itself broken
PPS_MIN_TOKENS = 16        # min compared token positions for a meaningful result


def _pps_vector(hidden_size):
    """Deterministic unit-norm fp32 steering vector (contract shape [hidden]).

    No RNG / no time: sin(arange) gives a dense, zero-ish-mean, fixed vector.
    """
    v = torch.sin(torch.arange(hidden_size, dtype=torch.float32))
    return v / v.norm()


def _hf_completion_logprobs(hf_model, prompt_ids, comp_ids):
    """Teacher-forced per-token logprobs of comp_ids given prompt_ids (HF)."""
    ids = torch.tensor([list(prompt_ids) + list(comp_ids)], device=DEVICE)
    with torch.no_grad():
        logits = hf_model(ids).logits[0].float()
    lps = torch.log_softmax(logits, dim=-1)
    plen = len(prompt_ids)
    return [lps[plen - 1 + i, tok].item() for i, tok in enumerate(comp_ids)]


def _hf_score_all(hf_model, prompt_ids_list, comps):
    """HF logprobs for each prompt's vLLM-generated completion tokens."""
    return [
        _hf_completion_logprobs(hf_model, prompt_ids, tok_ids)
        for prompt_ids, (tok_ids, _) in zip(prompt_ids_list, comps)
    ]


def _vllm_greedy_with_logprobs(mgr, prompt_ids_list, eid, max_tokens):
    """Greedy-generate via vLLM; return per-prompt (comp_token_ids, logprobs).

    Uses sampled-token logprobs (always included in each position's dict), so
    this needs no prompt_logprobs / top-k coverage assumptions.
    """
    from vllm import SamplingParams
    sp = SamplingParams(temperature=0, max_tokens=max_tokens, logprobs=1)
    outs = mgr.generate([list(p) for p in prompt_ids_list],
                        experiment_ids=[eid] * len(prompt_ids_list),
                        sampling_params=sp)
    results = []
    for out in outs:
        comp = out.outputs[0]
        tok_ids = list(comp.token_ids)
        assert comp.logprobs is not None and len(comp.logprobs) == len(tok_ids), \
            "vLLM did not return per-token logprobs for the sampled tokens"
        lps = [comp.logprobs[i][tok].logprob for i, tok in enumerate(tok_ids)]
        results.append((tok_ids, lps))
    return results


def _max_abs_diff(a_lists, b_lists):
    """Max |a-b| over aligned per-prompt per-token logprob lists + count."""
    assert len(a_lists) == len(b_lists)
    m, n = 0.0, 0
    for a, b in zip(a_lists, b_lists):
        assert len(a) == len(b), f"token-count mismatch: {len(a)} vs {len(b)}"
        for x, y in zip(a, b):
            m = max(m, abs(x - y))
            n += 1
    return m, n


def run_pps_steering_parity():
    """PPS steering kernel-match parity check (asserts; see block comment)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from gradient_routing import apply_dual_mlp, DualMLPAdapter, set_pps_steering
    from vllm_mlp_adapter import create_engine

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    hf_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16,
    ).to(DEVICE)
    torch.manual_seed(SEED)  # deterministic kaiming init inside apply_dual_mlp
    modified_layers = apply_dual_mlp(hf_model, RETAIN_NEURONS, FORGET_NEURONS)
    hf_model.eval()

    # Make adapter outputs nonzero (fresh down-projections are zero-init) so
    # steering parity is tested COMPOSED with active adapters and per-token
    # slot routing, as in a real run. Deterministic: dedicated seeded
    # generator, never wall-clock/global state.
    gen = torch.Generator().manual_seed(SEED)
    for module in hf_model.modules():
        if isinstance(module, DualMLPAdapter):
            for lin in (module.down_retain, module.down_forget):
                if lin is not None:
                    w = torch.randn(lin.weight.shape, generator=gen) * 0.02
                    lin.weight.data.copy_(w.to(lin.weight.dtype))

    n_layers = hf_model.config.num_hidden_layers
    hidden = hf_model.config.hidden_size
    layer = n_layers // 2  # one mid-stack layer, per the interface contract
    assert layer in modified_layers, \
        f"layer {layer} not adapted (adapted: {modified_layers})"
    vec = _pps_vector(hidden)  # fp32 cpu, shape [hidden] — the disk contract
    layer_to_vec = {layer: vec}

    llm, mgr = create_engine(
        model_name=MODEL_NAME,
        max_experiments=2,
        retain_neurons=RETAIN_NEURONS,
        forget_neurons=FORGET_NEURONS,
        gpu_memory_utilization=0.05,
        dtype="float16",
    )

    # eid 1 = "training" slot (gets steered); eid 2 = "eval" slot (never
    # steered — stands in for the piggybacked-eval eids that must stay on the
    # deployment trajectory). Steering is set AFTER the weight syncs: each
    # sync activates a fresh adapter whose slot activation zeroes steering.
    mgr.update_from_training_model(1, hf_model)
    mgr.update_from_training_model(2, hf_model)

    prompt_ids_list = [
        tokenizer(p, add_special_tokens=False).input_ids for p in PPS_PROMPTS
    ]

    print("\n" + "=" * 70)
    print(f"PPS STEERING PARITY (layer={layer}, alpha={PPS_ALPHA}, "
          f"||v||={vec.norm().item():.3f}, hidden={hidden})")
    print("=" * 70)

    # --- [A] both stacks UNSTEERED: measure the harness's baseline parity ---
    comps_off = _vllm_greedy_with_logprobs(mgr, prompt_ids_list, 1, PPS_MAX_TOKENS)
    vllm_off = [lps for _, lps in comps_off]
    hf_off_tokoff = _hf_score_all(hf_model, prompt_ids_list, comps_off)
    base_parity, n_base = _max_abs_diff(vllm_off, hf_off_tokoff)

    # HF steered, scoring the SAME tokens (negative control + materiality).
    set_pps_steering(hf_model, layer_to_vec, PPS_ALPHA)
    # Hygiene invariant (checked while steering is live): the steer buffer is
    # persistent=False, so it must never appear in checkpoints.
    assert not any(k.split(".")[-1] == "steer" for k in hf_model.state_dict()), \
        "steer buffer leaked into state_dict — checkpoint hygiene violated"
    hf_on_tokoff = _hf_score_all(hf_model, prompt_ids_list, comps_off)
    one_sided_hf, _ = _max_abs_diff(vllm_off, hf_on_tokoff)   # HF on, vLLM off
    hf_delta, _ = _max_abs_diff(hf_on_tokoff, hf_off_tokoff)  # single-stack effect

    # --- [B] both stacks STEERED with the same (v, alpha, layer) ---
    # Cross the REAL wire (production ships _pack_steering_msg -> msgpack ->
    # handle_set_steering, not this in-process dict) so the go/no-go certifies
    # the exact bytes train.py sends. A pack/unpack regression (dtype tag, layer
    # ordering, missing np .copy()) would otherwise pass parity while corrupting
    # the deployed vector. Mirrors handle_set_steering's unpack exactly.
    import msgpack as _mp
    import numpy as _np
    from vllm_client import _pack_steering_msg as _pack
    _wire = _mp.unpackb(_mp.packb(_pack(1, layer_to_vec, PPS_ALPHA),
                                  use_bin_type=True), raw=False)
    _wlayers = [int(l) for l in _wire["layers"]]
    assert _wire["dtype"] == "float32"
    _wflat = torch.from_numpy(_np.frombuffer(_wire["flat"], dtype=_np.float32).copy())
    _wmat = _wflat.view(len(_wlayers), int(_wire["hidden"]))
    _wire_l2v = {l: _wmat[i] for i, l in enumerate(_wlayers)}
    for l in layer_to_vec:  # bit-exact per layer after the round-trip
        assert torch.equal(_wire_l2v[l], layer_to_vec[l].reshape(-1).to(torch.float32)), \
            f"wire round-trip corrupted layer {l}'s steering vector"
    mgr.set_steering(1, _wire_l2v, float(_wire["alpha"]))
    # set_steering deliberately does no prefix-cache reset (in training,
    # steering is constant per eid and every weight sync resets the cache).
    # Here we flip steering with NO weight sync in between, so the unsteered
    # prompt KV from [A] must be dropped explicitly.
    llm.reset_prefix_cache()
    comps_on = _vllm_greedy_with_logprobs(mgr, prompt_ids_list, 1, PPS_MAX_TOKENS)
    vllm_on = [lps for _, lps in comps_on]
    hf_on_tokon = _hf_score_all(hf_model, prompt_ids_list, comps_on)
    steered_parity, n_steered = _max_abs_diff(vllm_on, hf_on_tokon)

    # --- [C] per-eid isolation + symmetric one-sided control ---
    # eid 2, in the SAME engine state where eid 1 is steered, must behave as
    # the unsteered policy (this is what keeps eval eids on the deployment
    # trajectory during training).
    comps_e2 = _vllm_greedy_with_logprobs(mgr, prompt_ids_list, 2, PPS_MAX_TOKENS)
    vllm_e2 = [lps for _, lps in comps_e2]
    set_pps_steering(hf_model, {}, 0.0)  # HF steering OFF ({} => fully off)
    hf_off_toke2 = _hf_score_all(hf_model, prompt_ids_list, comps_e2)
    e2_parity, _ = _max_abs_diff(vllm_e2, hf_off_toke2)
    # Symmetric one-sided control: vLLM steered vs HF unsteered. Also proves
    # set_pps_steering(model, {}, 0.0) fully RESTORES the unsteered policy
    # (these HF scores are computed after an on->off toggle).
    hf_off_tokon = _hf_score_all(hf_model, prompt_ids_list, comps_on)
    one_sided_vllm, _ = _max_abs_diff(vllm_on, hf_off_tokon)

    # Leave the engine unsteered for anything running after us.
    mgr.set_steering(1, {}, 0.0)
    llm.reset_prefix_cache()

    tol = max(PPS_PARITY_FLOOR, PPS_PARITY_FACTOR * base_parity)

    print(f"  compared token positions: base={n_base}, steered={n_steered}")
    print(f"  unsteered parity  (HF off vs vLLM off):  {base_parity:.6f}")
    print(f"  tolerance max(floor={PPS_PARITY_FLOOR}, "
          f"{PPS_PARITY_FACTOR} x base):     {tol:.6f}")
    print(f"  steered parity    (HF on  vs vLLM on):   {steered_parity:.6f}")
    print(f"  eval-eid parity   (HF off vs eid2 off):  {e2_parity:.6f}")
    print(f"  one-sided         (HF on  vs vLLM off):  {one_sided_hf:.6f}")
    print(f"  one-sided         (HF off vs vLLM on):   {one_sided_vllm:.6f}")
    print(f"  single-stack HF steer effect:            {hf_delta:.6f}")

    assert n_base >= PPS_MIN_TOKENS and n_steered >= PPS_MIN_TOKENS, (
        f"too few completion tokens compared (base={n_base}, "
        f"steered={n_steered}, need >= {PPS_MIN_TOKENS}) — generations "
        f"terminated immediately; investigate before trusting parity")
    assert base_parity < PPS_BASE_SANITY, (
        f"UNSTEERED HF-vs-vLLM parity is already {base_parity:.4f} "
        f">= {PPS_BASE_SANITY} — the equivalence harness itself is broken; "
        f"steering conclusions would be meaningless")

    # (1) THE go/no-go: identically-steered stacks must match as tightly as
    #     unsteered ones. A one-sided/inequivalent steering op fails here.
    assert steered_parity <= tol, (
        f"KERNEL MISMATCH: with identical (v, alpha={PPS_ALPHA}, layer="
        f"{layer}) steering in BOTH stacks, per-token logprobs differ by "
        f"{steered_parity:.4f} > tol {tol:.4f}. The vLLM and HF steering ops "
        f"are NOT equivalent — this corrupts the IS ratio; do not launch.")

    # (2) Per-eid isolation: the never-steered eid must stay byte-equivalent
    #     to the unsteered policy while eid 1 is steered.
    assert e2_parity <= tol, (
        f"STEERING LEAK: eid 2 (never steered) diverges from the unsteered "
        f"policy by {e2_parity:.4f} > tol {tol:.4f} while eid 1 is steered — "
        f"per-eid routing is broken; eval eids would be steered in training.")

    # (3) Negative controls (the test's teeth): steering applied to only ONE
    #     stack must diverge beyond the SAME tolerance. If these fail, a real
    #     one-sided bug could pass check (1) silently.
    assert one_sided_hf > tol, (
        f"NO TEETH: HF-steered vs vLLM-unsteered differ by only "
        f"{one_sided_hf:.4f} <= tol {tol:.4f} — either steering is not "
        f"applied in the HF stack, or alpha={PPS_ALPHA} at layer {layer} is "
        f"too weak for this check to detect a one-sided bug.")
    assert one_sided_vllm > tol, (
        f"NO TEETH: vLLM-steered vs HF-unsteered differ by only "
        f"{one_sided_vllm:.4f} <= tol {tol:.4f} — either steering is not "
        f"applied in the vLLM stack, or alpha={PPS_ALPHA} at layer {layer} "
        f"is too weak for this check to detect a one-sided bug.")

    # (4) Materiality: alpha=4 must be a non-trivial perturbation of a single
    #     stack (proves the parity in (1) is over a real behavioral change).
    assert hf_delta > tol, (
        f"steered-vs-unsteered HF logprobs differ by only {hf_delta:.4f} "
        f"<= tol {tol:.4f} — alpha={PPS_ALPHA} is not a material "
        f"perturbation here; check set_pps_steering wiring or raise alpha.")

    print("  PASS: PPS steering is kernel-matched across stacks, per-eid "
          "isolated, and the controls confirm the test has teeth")


def test_pps_steering_parity():
    """Pytest entry: skip cleanly off-GPU (CPU CI); run parity on the GPU box."""
    import pytest
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable — PPS steering parity needs the GPU box")
    try:
        import vllm  # noqa: F401
    except Exception as e:
        pytest.skip(f"vLLM unavailable: {e}")
    run_pps_steering_parity()


if __name__ == "__main__":
    if "--pps-only" in sys.argv:
        run_pps_steering_parity()
    else:
        main()
        run_pps_steering_parity()
