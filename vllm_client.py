"""Client for vLLM generation server.

Provides the same interface as VLLMAdapterManager but communicates
over ZMQ instead of calling vLLM directly.

Supports both server types:
  - vllm_server.py (sync, ZMQ REP) — use VLLMClient
  - vllm_async_server.py (async, ZMQ ROUTER) — use AsyncVLLMClient

Usage:
    # For sync server (vllm_server.py)
    client = VLLMClient("ipc:///tmp/vllm_grpo.sock")

    # For async server (vllm_async_server.py)
    client = AsyncVLLMClient("ipc:///tmp/vllm_grpo_async.sock")

    # Same API for both
    eid = client.register()
    client.update_weights_from_model(eid, hf_model)
    comp_texts, comp_ids, prompt_ids = client.generate(eid, prompt_ids_batch, ...)
"""

import msgpack
import numpy as np
import torch
import zmq

from gradient_routing import DualMLPAdapter
from vllm_utils import WEIGHT_KEYS


_PINNED_CACHE = {}


def _pack_weights_flat(model):
    """Pack all DualMLPAdapter weights into ONE flat bf16 buffer.

    Replaces the per-tensor extraction (180 small .cpu() calls, each a
    blocking CUDA sync) with: device-side casts into a flat staging tensor
    (async) + a single D2H copy into a cached pinned buffer. bf16 on the wire
    because the engine stores bf16 anyway — the cast just moves earlier.

    Order contract: model.modules() DualMLPAdapter order (== adapted-layer
    order, the same assumption the per-layer dict protocol made), WEIGHT_KEYS
    within each layer. The server unflattens with the same contract.

    Returns (payload_bytes, shapes) — shapes is a per-layer list of
    {key: shape} for the keys present.
    """
    srcs, shapes = [], []
    for module in model.modules():
        if isinstance(module, DualMLPAdapter):
            layer_shapes = {}
            for key in WEIGHT_KEYS:
                attr = getattr(module, key, None)
                if attr is not None:
                    srcs.append(attr.weight.data)
                    layer_shapes[key] = list(attr.weight.shape)
            shapes.append(layer_shapes)
    assert srcs, "No DualMLPAdapter layers found in model"
    total = sum(t.numel() for t in srcs)
    device = srcs[0].device
    staging = torch.empty(total, dtype=torch.bfloat16, device=device)
    off = 0
    for t in srcs:
        n = t.numel()
        staging[off:off + n].copy_(t.reshape(-1))   # fused cast+copy, async on GPU
        off += n
    cache_key = (total, str(device))
    pinned = _PINNED_CACHE.get(cache_key)
    if pinned is None:
        pinned = torch.empty(total, dtype=torch.bfloat16,
                             pin_memory=(device.type == "cuda"))
        _PINNED_CACHE[cache_key] = pinned
    pinned.copy_(staging)                            # the ONE D2H (syncs)
    return pinned.view(torch.uint8).numpy().tobytes(), shapes


def _pack_steering_msg(experiment_id, layer_to_vec, alpha):
    """Build the set_steering wire message (shared by sync + async clients).

    layer_to_vec ({layer_idx: (hidden,) tensor}) is concatenated into ONE flat
    fp32 byte buffer with a parallel int layer-key list ("layers") — msgpack
    maps can't carry int keys, and the flat-buffer + sidecar layout follows
    the _pack_weights_flat precedent. fp32 on the wire: the disk vector is
    the fp32 truth; both stacks cast to the activation dtype at the add
    itself, so the round-trip must not pre-round. {} / alpha 0.0 => steering
    off for that experiment.
    """
    layers = sorted(int(k) for k in layer_to_vec)
    vecs = []
    hidden = 0
    for l in layers:
        v = layer_to_vec[l].detach().reshape(-1).to(torch.float32).cpu()
        if hidden == 0:
            hidden = v.numel()
        assert v.numel() == hidden, \
            f"steering vec for layer {l} has {v.numel()} elems, expected {hidden}"
        vecs.append(v)
    payload = torch.cat(vecs).numpy().tobytes() if vecs else b""
    return {
        "op": "set_steering",
        "experiment_id": experiment_id,
        "layers": layers,
        "hidden": hidden,
        "alpha": float(alpha),
        "dtype": "float32",
        "flat": payload,
    }


def _extract_weights_from_model(model):
    """Extract DualMLPAdapter weights from HF model, return (layers, shapes)."""
    layers = []
    shapes = {}
    for module in model.modules():
        if isinstance(module, DualMLPAdapter):
            layer_data = {}
            for key in WEIGHT_KEYS:
                attr = getattr(module, key, None)
                if attr is not None:
                    t = attr.weight.data.detach().cpu().float()
                    layer_data[key] = t.numpy().tobytes()
                    if key not in shapes:
                        shapes[key] = list(t.shape)
                else:
                    layer_data[key] = None
            layers.append(layer_data)

    assert len(layers) > 0, "No DualMLPAdapter layers found in model"
    return layers, shapes


class VLLMClient:
    """Client for sync vLLM server (vllm_server.py, ZMQ REP/REQ)."""

    def __init__(self, socket_addr):
        self.ctx = zmq.Context()
        self.socket = self.ctx.socket(zmq.REQ)
        self.socket.connect(socket_addr)

    def _request(self, msg):
        """Send msgpack request, receive msgpack reply."""
        self.socket.send(msgpack.packb(msg, use_bin_type=True))
        raw = self.socket.recv()
        reply = msgpack.unpackb(raw, raw=False)
        if "error" in reply:
            raise RuntimeError(f"Server error: {reply['error']}")
        return reply

    def register(self):
        """Register and get an experiment ID."""
        reply = self._request({"op": "register"})
        return reply["experiment_id"]

    def release(self, experiment_id):
        """Return experiment slot to the server (zeros weights, frees slot)."""
        self._request({"op": "release", "experiment_id": experiment_id})

    def update_weights_from_model(self, experiment_id, model):
        """Send all DualMLPAdapter weights as one flat bf16 buffer (fast sync)."""
        flat, shapes = _pack_weights_flat(model)
        self._request({
            "op": "update_weights",
            "experiment_id": experiment_id,
            "flat": flat,
            "dtype": "bfloat16",
            "flat_shapes": shapes,
        })

    def generate(self, experiment_id, prompt_ids, n, temperature, max_tokens,
                 top_k=50, top_p=1.0, return_logprobs=False, detokenize=True):
        """Request generation, return (comp_texts, comp_ids, prompt_ids[, logprobs]).
        Server-side phase timings land in self._last_gen_timings."""
        import time as _t
        _t0 = _t.perf_counter()
        reply = self._request({
            "op": "generate",
            "experiment_id": experiment_id,
            "prompt_ids": prompt_ids,
            "n": n,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_k": top_k,
            "top_p": top_p,
            "return_logprobs": return_logprobs,
            "detokenize": detokenize,
        })
        self._last_gen_timings = {**(reply.get("timings") or {}),
                                  "client_roundtrip_s": round(_t.perf_counter() - _t0, 4)}
        result = (
            reply["completion_texts"],
            reply["completion_ids"],
            reply["prompt_ids"],
        )
        if return_logprobs:
            result = result + (reply.get("logprobs"),)
        return result

    def generate_multi(self, experiment_ids, prompt_ids, n, temperature, max_tokens,
                       top_k=50, top_p=1.0, return_logprobs=False, detokenize=True):
        """Generate with per-prompt experiment routing (concurrent multi-adapter eval).

        experiment_ids: list[int] of same length as prompt_ids. Each prompt is routed
        to its corresponding adapter slot; the batch is processed in a single vLLM call.
        """
        assert len(experiment_ids) == len(prompt_ids), \
            f"experiment_ids length {len(experiment_ids)} != prompt_ids length {len(prompt_ids)}"
        import time as _t
        _t0 = _t.perf_counter()
        reply = self._request({
            "op": "generate",
            "experiment_ids": list(experiment_ids),
            "prompt_ids": prompt_ids,
            "n": n,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_k": top_k,
            "top_p": top_p,
            "return_logprobs": return_logprobs,
            "detokenize": detokenize,
        })
        self._last_gen_timings = {**(reply.get("timings") or {}),
                                  "client_roundtrip_s": round(_t.perf_counter() - _t0, 4)}
        result = (
            reply["completion_texts"],
            reply["completion_ids"],
            reply["prompt_ids"],
        )
        if return_logprobs:
            result = result + (reply.get("logprobs"),)
        return result

    def set_scales(self, experiment_id, retain_scale, forget_scale):
        """Set adapter scales on the server."""
        self._request({
            "op": "set_scales",
            "experiment_id": experiment_id,
            "retain_scale": retain_scale,
            "forget_scale": forget_scale,
        })

    def set_steering(self, experiment_id, layer_to_vec, alpha):
        """Set PPS steering on the server: {layer_idx: (hidden,) tensor} + alpha.

        {} / alpha 0.0 => steering off for this experiment. Adapted layers
        not in the dict are explicitly zeroed server-side so stale steering
        never lingers.
        """
        self._request(_pack_steering_msg(experiment_id, layer_to_vec, alpha))

    def sleep(self, level=1):
        """Put vLLM engine to sleep (free GPU memory for training)."""
        self._request({"op": "sleep", "level": level})

    def wake_up(self, tags=None):
        """Wake vLLM engine (reload weights + KV cache for generation)."""
        self._request({"op": "wake_up", "tags": tags})

    def shutdown(self):
        """Tell server to shut down."""
        self._request({"op": "shutdown"})
        self.ctx.destroy()

    def close(self):
        """Close client without shutting down server."""
        self.ctx.destroy()


class AsyncVLLMClient:
    """Client for async vLLM server (vllm_async_server.py, ZMQ ROUTER/DEALER).

    Uses DEALER socket to talk to the ROUTER server. DEALER sends messages
    without an explicit identity frame — ZMQ adds it automatically. The empty
    delimiter frame is required to match ROUTER's framing.
    """

    def __init__(self, socket_addr):
        self.ctx = zmq.Context()
        self.socket = self.ctx.socket(zmq.DEALER)
        self.socket.connect(socket_addr)

    def _request(self, msg):
        """Send msgpack request, receive msgpack reply."""
        # DEALER -> ROUTER: send [empty, payload]
        self.socket.send_multipart([
            b"",
            msgpack.packb(msg, use_bin_type=True),
        ])
        # ROUTER -> DEALER: receive [empty, payload]
        frames = self.socket.recv_multipart()
        payload = frames[-1]
        reply = msgpack.unpackb(payload, raw=False)
        if "error" in reply:
            raise RuntimeError(f"Server error: {reply['error']}")
        return reply

    def register(self):
        """Register and get an experiment ID."""
        reply = self._request({"op": "register"})
        return reply["experiment_id"]

    def release(self, experiment_id):
        """Return experiment slot to the server (zeros weights, frees slot)."""
        self._request({"op": "release", "experiment_id": experiment_id})

    def update_weights_from_model(self, experiment_id, model):
        """Extract DualMLPAdapter weights from HF model and send to server."""
        layers, shapes = _extract_weights_from_model(model)
        self._request({
            "op": "update_weights",
            "experiment_id": experiment_id,
            "shapes": shapes,
            "dtype": "float32",
            "layers": layers,
        })

    def generate(self, experiment_id, prompt_ids, n, temperature, max_tokens, top_k=50, top_p=1.0, return_logprobs=False):
        """Request generation, return (comp_texts, comp_ids, prompt_ids)."""
        reply = self._request({
            "op": "generate",
            "experiment_id": experiment_id,
            "prompt_ids": prompt_ids,
            "n": n,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_k": top_k,
            "top_p": top_p,
        })
        return (
            reply["completion_texts"],
            reply["completion_ids"],
            reply["prompt_ids"],
        )

    def generate_multi(self, experiment_ids, prompt_ids, n, temperature, max_tokens,
                       top_k=50, top_p=1.0, return_logprobs=False):
        """Generate with per-prompt experiment routing (concurrent multi-adapter eval).

        experiment_ids: list[int] of same length as prompt_ids.
        """
        assert len(experiment_ids) == len(prompt_ids), \
            f"experiment_ids length {len(experiment_ids)} != prompt_ids length {len(prompt_ids)}"
        reply = self._request({
            "op": "generate",
            "experiment_ids": list(experiment_ids),
            "prompt_ids": prompt_ids,
            "n": n,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_k": top_k,
            "top_p": top_p,
        })
        return (
            reply["completion_texts"],
            reply["completion_ids"],
            reply["prompt_ids"],
        )

    def set_scales(self, experiment_id, retain_scale, forget_scale):
        """Set adapter scales on the server."""
        self._request({
            "op": "set_scales",
            "experiment_id": experiment_id,
            "retain_scale": retain_scale,
            "forget_scale": forget_scale,
        })

    def set_steering(self, experiment_id, layer_to_vec, alpha):
        """Set PPS steering on the server: {layer_idx: (hidden,) tensor} + alpha.

        {} / alpha 0.0 => steering off for this experiment. Same wire format
        as the sync client; the server must implement the "set_steering" op
        (unknown-op replies raise loudly here rather than dropping steering).
        """
        self._request(_pack_steering_msg(experiment_id, layer_to_vec, alpha))

    def sleep(self, level=1):
        """Put vLLM engine to sleep (free GPU memory for training)."""
        self._request({"op": "sleep", "level": level})

    def wake_up(self, tags=None):
        """Wake vLLM engine (reload weights + KV cache for generation)."""
        self._request({"op": "wake_up", "tags": tags})

    def shutdown(self):
        """Tell server to shut down."""
        self._request({"op": "shutdown"})
        self.ctx.destroy()

    def close(self):
        """Close client without shutting down server."""
        self.ctx.destroy()
