"""Test that packed (padding-free) forward produces the same loss and gradients as padded."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import pytest
from transformers import AutoModelForCausalLM, AutoTokenizer
from liger_kernel.chunked_loss import LigerFusedLinearGRPOLoss

from train import _pack_for_forward


MODEL_NAME = "HuggingFaceTB/SmolLM2-135M-Instruct"


@pytest.fixture(scope="module")
def model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2"
    ).cuda()
    model.eval()  # deterministic
    return model, tokenizer


def _make_test_batch(tokenizer, n=4):
    """Create a realistic batch with variable-length prompts and completions."""
    prompts = [
        "Once upon a time",
        "The quick brown fox jumped over",
        "Hello",
        "In a galaxy far far away there lived a",
    ][:n]
    completions = [
        " there was a brave knight who fought dragons.",
        " the lazy dog sleeping in the sun.",
        " world! How are you today?",
        " small green alien named Zorp who loved cookies and tea.",
    ][:n]

    # Tokenize
    prompt_ids_list = [tokenizer.encode(p, add_special_tokens=False) for p in prompts]
    comp_ids_list = [tokenizer.encode(c, add_special_tokens=False) for c in completions]

    # Pad prompts (left) and completions (right)
    max_p = max(len(p) for p in prompt_ids_list)
    max_c = max(len(c) for c in comp_ids_list)

    prompt_ids = torch.full((n, max_p), tokenizer.pad_token_id or 0, dtype=torch.long)
    prompt_mask = torch.zeros(n, max_p, dtype=torch.long)
    comp_ids = torch.full((n, max_c), tokenizer.pad_token_id or 0, dtype=torch.long)
    comp_mask = torch.zeros(n, max_c, dtype=torch.long)

    for i in range(n):
        p_len = len(prompt_ids_list[i])
        prompt_ids[i, max_p - p_len:] = torch.tensor(prompt_ids_list[i])
        prompt_mask[i, max_p - p_len:] = 1

        c_len = len(comp_ids_list[i])
        comp_ids[i, :c_len] = torch.tensor(comp_ids_list[i])
        comp_mask[i, :c_len] = 1

    return {
        "prompt_ids": prompt_ids.cuda(),
        "prompt_mask": prompt_mask.cuda(),
        "completion_ids": comp_ids.cuda(),
        "completion_mask": comp_mask.cuda(),
        "advantages": torch.ones(n).cuda(),  # uniform advantages to simplify comparison
        "old_per_token_logps": None,  # will be filled with detached current logps
        "ref_per_token_logps": None,
        "sampling_per_token_logps": None,
    }


def _padded_forward_hidden_states(model, batch):
    """Standard padded forward: get completion hidden states via the normal TRL path."""
    prompt_ids = batch["prompt_ids"]
    comp_ids = batch["completion_ids"]
    prompt_mask = batch["prompt_mask"]
    comp_mask = batch["completion_mask"]

    input_ids = torch.cat([prompt_ids, comp_ids], dim=1)
    attention_mask = torch.cat([prompt_mask, comp_mask], dim=1)
    logits_to_keep = comp_ids.size(1)

    # Mirror TRL's _get_last_hidden_state
    output = model.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    hs = output.last_hidden_state  # (B, P+C, H)
    hs = hs[:, :-1, :]  # exclude last position
    hs = hs[:, -logits_to_keep:, :]  # keep completion positions
    return hs  # (B, max_comp_len, H)


def _packed_forward_hidden_states(model, batch, indices):
    """Packed forward: get completion hidden states via our padding-free path."""
    packed = _pack_for_forward(batch, indices)

    output = model.model(
        input_ids=packed["packed_input_ids"],
        position_ids=packed["packed_position_ids"],
        use_cache=False,
    )
    hidden_states = output.last_hidden_state  # (1, T, H)

    n_seqs = packed["num_sequences"]
    max_comp_len = packed["max_comp_len"]
    hidden_dim = hidden_states.shape[-1]
    device = hidden_states.device

    last_hs_padded = torch.zeros(n_seqs, max_comp_len, hidden_dim, device=device, dtype=hidden_states.dtype)
    offset = 0
    for j, (p_len, c_len) in enumerate(packed["seq_boundaries"]):
        if c_len > 0:
            hs_start = offset + p_len - 1
            hs_end = offset + p_len + c_len - 1
            last_hs_padded[j, :c_len] = hidden_states[0, hs_start:hs_end]
        offset += p_len + c_len

    return last_hs_padded, packed  # (N, max_comp_len, H)


def _run_sequences_independently(model, batch):
    """Run each sequence through the model individually (no batching, no padding)."""
    n = batch["prompt_mask"].shape[0]
    results = []
    for i in range(n):
        p_mask = batch["prompt_mask"][i]
        real_positions = p_mask.nonzero(as_tuple=True)[0]
        p_start = real_positions[0].item() if len(real_positions) > 0 else 0
        p_real = batch["prompt_ids"][i, p_start:]
        p_len = p_real.shape[0]

        c_mask = batch["completion_mask"][i]
        c_len = c_mask.sum().item()
        c_real = batch["completion_ids"][i, :c_len]

        seq_ids = torch.cat([p_real, c_real]).unsqueeze(0)  # (1, seq_len)
        out = model.model(input_ids=seq_ids, use_cache=False)
        hs = out.last_hidden_state[0]  # (seq_len, H)
        # Completion hidden states: positions [p_len-1, p_len+c_len-1)
        comp_hs = hs[p_len - 1: p_len + c_len - 1] if c_len > 0 else hs[:0]
        results.append(comp_hs)
    return results


class TestPackedEquivalence:
    """Verify packed forward matches running sequences independently (no padding)."""

    def test_packed_matches_independent(self, model_and_tokenizer):
        """Packed forward should produce identical hidden states to running each sequence alone.

        This is the ground truth: no padding influence at all. Both packed and independent
        paths process only real tokens. Any difference is a bug in our packing/extraction.
        """
        model, tokenizer = model_and_tokenizer
        batch = _make_test_batch(tokenizer, n=4)
        indices = list(range(4))

        with torch.no_grad():
            independent_hs = _run_sequences_independently(model, batch)
            packed_hs, packed = _packed_forward_hidden_states(model, batch, indices)

        for i in range(4):
            c_len = batch["completion_mask"][i].sum().item()
            if c_len == 0:
                continue
            ind_seq = independent_hs[i]        # (c_len, H)
            pack_seq = packed_hs[i, :c_len]    # (c_len, H)

            diff = (ind_seq - pack_seq).abs().float()
            mean_diff = diff.mean().item()
            p95_diff = torch.quantile(diff.flatten(), 0.95).item()
            max_diff = diff.max().item()

            print(f"Seq {i} (comp_len={c_len}): mean={mean_diff:.8f} p95={p95_diff:.8f} max={max_diff:.8f}")

            # These should be exactly equal (same computation, just batched differently)
            # Allow tiny tolerance for bf16 non-determinism in flash attention
            assert max_diff < 1e-3, (
                f"Seq {i}: packed vs independent max_diff={max_diff:.6f} "
                f"(mean={mean_diff:.6f}, p95={p95_diff:.6f}) — should be near-zero"
            )

    def test_padded_diverges_from_independent(self, model_and_tokenizer):
        """Padded forward diverges from independent for left-padded sequences.

        This documents the known issue: pad tokens leak through LayerNorm in the
        padded path, causing different hidden states than running sequences alone.
        The packed path avoids this.
        """
        model, tokenizer = model_and_tokenizer
        batch = _make_test_batch(tokenizer, n=4)

        with torch.no_grad():
            independent_hs = _run_sequences_independently(model, batch)
            padded_hs = _padded_forward_hidden_states(model, batch)

        prompt_lens = batch["prompt_mask"].sum(dim=1)
        max_prompt_len = prompt_lens.max().item()
        has_divergence = False

        for i in range(4):
            c_len = batch["completion_mask"][i].sum().item()
            if c_len == 0:
                continue
            ind_seq = independent_hs[i]
            pad_seq = padded_hs[i, :c_len]

            diff = (ind_seq - pad_seq).abs().float()
            mean_diff = diff.mean().item()
            p95_diff = torch.quantile(diff.flatten(), 0.95).item()
            max_diff = diff.max().item()
            is_padded = prompt_lens[i].item() < max_prompt_len

            print(f"Seq {i} ({'padded' if is_padded else 'no-pad'}, comp_len={c_len}): "
                  f"mean={mean_diff:.8f} p95={p95_diff:.8f} max={max_diff:.8f}")

            if is_padded and max_diff > 0.01:
                has_divergence = True

        # At least one left-padded sequence should show divergence
        assert has_divergence, "Expected padded path to diverge for left-padded sequences"

    def test_liger_loss_close(self, model_and_tokenizer):
        """Liger loss from packed hidden states should be close to padded hidden states."""
        model, tokenizer = model_and_tokenizer
        batch = _make_test_batch(tokenizer, n=4)
        indices = list(range(4))

        liger_loss_fn = LigerFusedLinearGRPOLoss(
            beta=0.0,
            temperature=1.0,
            use_ref_model=False,
            loss_type="grpo",
        )

        with torch.no_grad():
            padded_hs = _padded_forward_hidden_states(model, batch)
            packed_hs, packed = _packed_forward_hidden_states(model, batch, indices)

        # Compute old_per_token_logps from padded path (needed for both)
        comp_ids = batch["completion_ids"]
        comp_mask = batch["completion_mask"]
        max_comp = comp_mask.sum(dim=1).max().item()

        # Trim to max real completion length for fair comparison
        padded_hs_trimmed = padded_hs[:, :max_comp, :].detach().requires_grad_(True)
        packed_hs_trimmed = packed_hs[:, :max_comp, :].detach().requires_grad_(True)
        comp_ids_trimmed = comp_ids[:, :max_comp]
        comp_mask_trimmed = comp_mask[:, :max_comp]

        advantages = batch["advantages"]

        # Padded loss
        loss_padded, _ = liger_loss_fn(
            _input=padded_hs_trimmed,
            lin_weight=model.lm_head.weight,
            selected_token_ids=comp_ids_trimmed,
            attention_mask=comp_mask_trimmed,
            advantages=advantages,
            bias=getattr(model.lm_head, 'bias', None),
        )

        # Packed loss
        loss_packed, _ = liger_loss_fn(
            _input=packed_hs_trimmed,
            lin_weight=model.lm_head.weight,
            selected_token_ids=comp_ids_trimmed,
            attention_mask=comp_mask_trimmed,
            advantages=advantages,
            bias=getattr(model.lm_head, 'bias', None),
        )

        # Not exactly equal due to LayerNorm divergence from padding (see test_hidden_states_close)
        rel_diff = (loss_padded - loss_packed).abs() / loss_padded.abs().clamp(min=1e-6)
        assert rel_diff.item() < 0.1, (
            f"Loss divergence too large: padded={loss_padded.item():.6f} "
            f"packed={loss_packed.item():.6f} rel_diff={rel_diff.item():.4f}"
        )

    def test_gradients_close(self, model_and_tokenizer):
        """Gradients through packed forward should be close to padded forward."""
        model, tokenizer = model_and_tokenizer
        batch = _make_test_batch(tokenizer, n=4)
        indices = list(range(4))

        liger_loss_fn = LigerFusedLinearGRPOLoss(
            beta=0.0,
            temperature=1.0,
            use_ref_model=False,
            loss_type="grpo",
        )

        comp_mask = batch["completion_mask"]
        max_comp = comp_mask.sum(dim=1).max().item()
        comp_ids_trimmed = batch["completion_ids"][:, :max_comp]
        comp_mask_trimmed = comp_mask[:, :max_comp]
        advantages = batch["advantages"]

        # --- Padded path ---
        model.zero_grad()
        padded_hs = _padded_forward_hidden_states(model, batch)
        padded_hs_trimmed = padded_hs[:, :max_comp, :]

        loss_padded, _ = liger_loss_fn(
            _input=padded_hs_trimmed,
            lin_weight=model.lm_head.weight,
            selected_token_ids=comp_ids_trimmed,
            attention_mask=comp_mask_trimmed,
            advantages=advantages,
            bias=getattr(model.lm_head, 'bias', None),
        )
        loss_padded.backward()
        padded_grads = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}

        # --- Packed path ---
        model.zero_grad()
        packed_hs, packed = _packed_forward_hidden_states(model, batch, indices)
        packed_hs_trimmed = packed_hs[:, :max_comp, :]

        loss_packed, _ = liger_loss_fn(
            _input=packed_hs_trimmed,
            lin_weight=model.lm_head.weight,
            selected_token_ids=comp_ids_trimmed,
            attention_mask=comp_mask_trimmed,
            advantages=advantages,
            bias=getattr(model.lm_head, 'bias', None),
        )
        loss_packed.backward()
        packed_grads = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}

        # Compare gradients — not exactly equal due to LayerNorm divergence from padding
        assert set(padded_grads.keys()) == set(packed_grads.keys()), \
            f"Different params have gradients: {set(padded_grads.keys()) ^ set(packed_grads.keys())}"

        max_rel_diff = 0.0
        for name in padded_grads:
            g_pad = padded_grads[name]
            g_pack = packed_grads[name]
            abs_diff = (g_pad - g_pack).abs()
            scale = g_pad.abs().clamp(min=1e-8)
            rel_diff = (abs_diff / scale).max().item()
            max_rel_diff = max(max_rel_diff, rel_diff)
            # Cosine similarity should be very high even if magnitudes differ slightly
            cos_sim = torch.nn.functional.cosine_similarity(
                g_pad.flatten().unsqueeze(0).float(),
                g_pack.flatten().unsqueeze(0).float()
            ).item()
            assert cos_sim > 0.95, (
                f"Gradient direction mismatch for {name}: cosine_sim={cos_sim:.4f}"
            )
