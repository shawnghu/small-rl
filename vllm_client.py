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
import zmq

from gradient_routing import DualMLPAdapter

# Must match server
WEIGHT_KEYS = [
    "gate_retain", "up_retain", "down_retain",
    "gate_forget", "up_forget", "down_forget",
]


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
                    t = attr.weight.data
                    layer_data[key] = t.detach().cpu().numpy().tobytes()
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

    def generate(self, experiment_id, prompt_ids, n, temperature, max_tokens):
        """Request generation, return (comp_texts, comp_ids, prompt_ids)."""
        reply = self._request({
            "op": "generate",
            "experiment_id": experiment_id,
            "prompt_ids": prompt_ids,
            "n": n,
            "temperature": temperature,
            "max_tokens": max_tokens,
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

    def sleep(self):
        """Put vLLM engine to sleep, freeing GPU memory."""
        self._request({"op": "sleep"})

    def wake_up(self):
        """Wake up vLLM engine, restoring GPU memory."""
        self._request({"op": "wake_up"})

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

    def generate(self, experiment_id, prompt_ids, n, temperature, max_tokens):
        """Request generation, return (comp_texts, comp_ids, prompt_ids)."""
        reply = self._request({
            "op": "generate",
            "experiment_id": experiment_id,
            "prompt_ids": prompt_ids,
            "n": n,
            "temperature": temperature,
            "max_tokens": max_tokens,
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

    def sleep(self):
        """Put vLLM engine to sleep, freeing GPU memory."""
        self._request({"op": "sleep"})

    def wake_up(self):
        """Wake up vLLM engine, restoring GPU memory."""
        self._request({"op": "wake_up"})

    def shutdown(self):
        """Tell server to shut down."""
        self._request({"op": "shutdown"})
        self.ctx.destroy()

    def close(self):
        """Close client without shutting down server."""
        self.ctx.destroy()
