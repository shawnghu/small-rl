"""Client for vLLM generation server.

Provides the same interface as VLLMAdapterManager but communicates
over ZMQ instead of calling vLLM directly.

Usage:
    client = VLLMClient("ipc:///tmp/vllm_grpo.sock")
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

# Mapping from weight key to (module attribute, is present check attribute)
_WEIGHT_ATTRS = {
    "gate_retain": "gate_retain",
    "up_retain": "up_retain",
    "down_retain": "down_retain",
    "gate_forget": "gate_forget",
    "up_forget": "up_forget",
    "down_forget": "down_forget",
}


class VLLMClient:
    """Client for vLLM generation server over ZMQ."""

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

    def shutdown(self):
        """Tell server to shut down."""
        self._request({"op": "shutdown"})
        self.ctx.destroy()

    def close(self):
        """Close client without shutting down server."""
        self.ctx.destroy()
