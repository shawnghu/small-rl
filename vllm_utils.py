"""Shared utilities for vLLM integration: presets, serialization, GRPO helpers.

Used by vllm_server.py, vllm_async_server.py, vllm_lora.py, vllm_client.py,
train.py, sweep.py.
"""

import numpy as np
import torch
import torch.nn.functional as F

MLP_PRESETS = {
    "m5":   {"retain_neurons": 5,   "forget_neurons": 5,   "layer_stride": 1},
    "m10":  {"retain_neurons": 10,  "forget_neurons": 10,  "layer_stride": 1},
    "m16":  {"retain_neurons": 16,  "forget_neurons": 16,  "layer_stride": 1},
    "m30":  {"retain_neurons": 30,  "forget_neurons": 30,  "layer_stride": 1},
    "m32":  {"retain_neurons": 32,  "forget_neurons": 32,  "layer_stride": 1},
    "m64":  {"retain_neurons": 64,  "forget_neurons": 64,  "layer_stride": 1},
    "m64_retain_only": {"retain_neurons": 64, "forget_neurons": 0, "layer_stride": 1},
    "m128": {"retain_neurons": 128, "forget_neurons": 128, "layer_stride": 1},
    "m256": {"retain_neurons": 256, "forget_neurons": 256, "layer_stride": 1},
}

# Weight tensor names for MLP adapter serialization (client ↔ server)
WEIGHT_KEYS = [
    "gate_retain", "up_retain", "down_retain",
    "gate_forget", "up_forget", "down_forget",
]


def deserialize_layer_weights(msg):
    """Reconstruct MLP adapter weight tensors from a msgpack update_weights message.

    Returns list of dicts (one per layer), each mapping WEIGHT_KEYS to torch tensors.
    """
    dtype_str = msg["dtype"]
    np_dtype = np.float32 if dtype_str == "float32" else np.float16

    layer_weights = []
    for layer_data in msg["layers"]:
        w = {}
        for key in WEIGHT_KEYS:
            raw = layer_data.get(key)
            if raw is not None:
                shape = tuple(msg["shapes"][key])
                arr = np.frombuffer(raw, dtype=np_dtype).reshape(shape)
                w[key] = torch.from_numpy(arr.copy())
        layer_weights.append(w)
    return layer_weights


# ---------------------------------------------------------------------------
# Core GRPO functions
# ---------------------------------------------------------------------------

def compute_grpo_advantages(rewards, num_generations):
    """Per-prompt normalization of rewards (GRPO-style advantages).

    Args:
        rewards: (B*N,) tensor
        num_generations: N completions per prompt
    Returns:
        (B*N,) tensor of advantages
    """
    B = rewards.shape[0] // num_generations
    assert B * num_generations == rewards.shape[0]
    reshaped = rewards.view(B, num_generations)
    mean = reshaped.mean(dim=1, keepdim=True)
    std = reshaped.std(dim=1, keepdim=True)
    return ((reshaped - mean) / (std + 1e-4)).view(-1)


def compute_log_probs(model, prompt_ids, comp_ids_padded, comp_mask, prompt_len, device):
    """Per-sample sum of log probs for completions, with gradients.

    Args:
        model: HF CausalLM (with adapters)
        prompt_ids: (B*N, prompt_len) long tensor
        comp_ids_padded: (B*N, max_comp_len) long tensor (right-padded with 0)
        comp_mask: (B*N, max_comp_len) float tensor (1=real, 0=pad)
        prompt_len: int
        device: torch device
    Returns:
        (B*N,) float tensor with gradients
    """
    max_comp_len = comp_ids_padded.shape[1]
    input_ids = torch.cat([prompt_ids, comp_ids_padded], dim=1).to(device)
    attn_mask = torch.cat([
        torch.ones(input_ids.shape[0], prompt_len, device=device),
        comp_mask.to(device),
    ], dim=1)

    logits = model(input_ids=input_ids, attention_mask=attn_mask).logits
    # logits[:, t, :] predicts token t+1
    # Completion tokens at positions [prompt_len, prompt_len + max_comp_len)
    # Need logits at positions [prompt_len - 1, prompt_len + max_comp_len - 1)
    comp_logits = logits[:, prompt_len - 1 : prompt_len + max_comp_len - 1, :]
    log_probs = F.log_softmax(comp_logits, dim=-1)
    token_logps = log_probs.gather(-1, comp_ids_padded.to(device).unsqueeze(-1)).squeeze(-1)
    return (token_logps * comp_mask.to(device)).sum(dim=1)


def flatten_vllm_outputs(outputs, prompt_texts_in=None):
    """Flatten vLLM RequestOutputs into parallel lists.

    Args:
        outputs: list of RequestOutput from vLLM
        prompt_texts_in: optional list of prompt strings (one per RequestOutput).
            Required when using TokensPrompt (req.prompt is None).

    Returns:
        (completion_texts, completion_ids_list, prompt_ids_list, prompt_texts)
        where each is a flat list of length B * num_generations.
    """
    comp_texts, comp_ids, prompt_ids_all, prompt_texts = [], [], [], []
    for i, req in enumerate(outputs):
        pid = list(req.prompt_token_ids)
        # req.prompt is None when using TokensPrompt; use caller-supplied text
        ptxt = req.prompt if req.prompt is not None else (
            prompt_texts_in[i] if prompt_texts_in is not None else ""
        )
        for comp in req.outputs:
            comp_texts.append(comp.text)
            comp_ids.append(list(comp.token_ids))
            prompt_ids_all.append(pid)
            prompt_texts.append(ptxt)
    return comp_texts, comp_ids, prompt_ids_all, prompt_texts


def pad_completions(comp_ids_list):
    """Pad variable-length completion token ID lists.

    Returns:
        (padded_ids, mask) — both (N, max_len) tensors.
        If all completions are empty, returns tensors with max_len=1.
    """
    max_len = max((len(c) for c in comp_ids_list), default=0)
    if max_len == 0:
        n = len(comp_ids_list)
        return torch.zeros(n, 1, dtype=torch.long), torch.zeros(n, 1)
    padded = torch.zeros(len(comp_ids_list), max_len, dtype=torch.long)
    mask = torch.zeros(len(comp_ids_list), max_len)
    for i, cids in enumerate(comp_ids_list):
        L = len(cids)
        if L > 0:
            padded[i, :L] = torch.tensor(cids, dtype=torch.long)
            mask[i, :L] = 1.0
    return padded, mask
