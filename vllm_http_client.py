"""HTTP client for vLLM generation server with MLP adapter support.

Drop-in replacement for VLLMClient/AsyncVLLMClient (ZMQ) using HTTP/msgpack.
Compatible with vllm_http_server.py.

Usage:
    client = VLLMHTTPClient("http://localhost:8100")
    eid = client.register()
    client.update_weights_from_model(eid, hf_model)
    comp_texts, comp_ids, prompt_ids = client.generate(eid, prompt_ids_batch, ...)
"""

import time

import msgpack
import numpy as np
import requests

from gradient_routing import DualMLPAdapter

# Must match server
WEIGHT_KEYS = [
    "gate_retain", "up_retain", "down_retain",
    "gate_forget", "up_forget", "down_forget",
]

# Retry config
MAX_RETRIES = 3
RETRY_BACKOFF_S = 1.0


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
                    layer_data[key] = t.detach().cpu().float().numpy().tobytes()
                    if key not in shapes:
                        shapes[key] = list(t.shape)
                else:
                    layer_data[key] = None
            layers.append(layer_data)

    assert len(layers) > 0, "No DualMLPAdapter layers found in model"
    return layers, shapes


class VLLMHTTPClient:
    """HTTP client for the vLLM generation server."""

    def __init__(self, base_url, timeout=240):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()

    def _request(self, endpoint, data=None, method="POST"):
        """Send msgpack request, return parsed response."""
        url = f"{self.base_url}/{endpoint}"

        for attempt in range(MAX_RETRIES):
            try:
                if method == "GET":
                    resp = self.session.get(url, timeout=self.timeout)
                else:
                    body = msgpack.packb(data, use_bin_type=True) if data else b""
                    resp = self.session.post(
                        url, data=body,
                        headers={"Content-Type": "application/x-msgpack"},
                        timeout=self.timeout,
                    )
                resp.raise_for_status()
                result = msgpack.unpackb(resp.content, raw=False)
                if "error" in result:
                    raise RuntimeError(f"Server error: {result['error']}")
                return result

            except (requests.ConnectionError, requests.Timeout) as e:
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_BACKOFF_S * (attempt + 1))
                else:
                    raise RuntimeError(
                        f"Failed to connect to vLLM server at {url} "
                        f"after {MAX_RETRIES} attempts: {e}"
                    ) from e

    def health(self):
        """Check server health."""
        return self._request("health", method="GET")

    def wait_until_ready(self, timeout=900, poll_interval=5.0):
        """Block until the server responds to health checks."""
        t0 = time.time()
        last_print = 0.0
        while time.time() - t0 < timeout:
            try:
                result = self.health()
                if result.get("status") == "ok":
                    return result
            except Exception:
                pass
            elapsed = time.time() - t0
            if elapsed - last_print >= 30:
                print(f"[vLLM] Waiting for server at {self.base_url} ({elapsed:.0f}s elapsed)...")
                last_print = elapsed
            time.sleep(poll_interval)
        raise TimeoutError(
            f"vLLM server at {self.base_url} not ready after {timeout}s"
        )

    def register(self):
        """Register and get an experiment ID."""
        result = self._request("register")
        return result["experiment_id"]

    def update_weights_from_model(self, experiment_id, model):
        """Extract DualMLPAdapter weights from HF model and send to server."""
        layers, shapes = _extract_weights_from_model(model)
        self._request("update_weights", {
            "experiment_id": experiment_id,
            "shapes": shapes,
            "dtype": "float32",
            "layers": layers,
        })

    def generate(self, experiment_id, prompt_ids, n, temperature, max_tokens):
        """Request generation, return (comp_texts, comp_ids, prompt_ids)."""
        result = self._request("generate", {
            "experiment_id": experiment_id,
            "prompt_ids": prompt_ids,
            "n": n,
            "temperature": temperature,
            "max_tokens": max_tokens,
        })
        return (
            result["completion_texts"],
            result["completion_ids"],
            result["prompt_ids"],
        )

    def set_scales(self, experiment_id, retain_scale, forget_scale):
        """Set adapter scales on the server."""
        self._request("set_scales", {
            "experiment_id": experiment_id,
            "retain_scale": retain_scale,
            "forget_scale": forget_scale,
        })

    def shutdown(self):
        """Tell server to shut down."""
        try:
            self._request("shutdown")
        except Exception:
            pass  # Server may close before response

    def close(self):
        """Close client session."""
        self.session.close()
