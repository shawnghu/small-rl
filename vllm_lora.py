"""vLLM LoRA integration for DualLoRA models.

Syncs DualLoRA weights to a vLLM engine using vLLM's native LoRA support.
Uses TensorLoRARequest + monkey-patch (pattern from rl-gradient-routing)
to load LoRA tensors directly into vLLM without disk I/O.

Requires VLLM_ENABLE_V1_MULTIPROCESSING=0 so the engine runs in-process
(otherwise TensorLoRARequest gets stripped during subprocess serialization).

Contains:
    - TensorLoRARequest + monkey-patch for in-memory LoRA loading
    - _extract_dual_lora_tensors: DualLoRA → PEFT-format tensor dict
    - VLLMLoRAServer: ZMQ server (drop-in replacement for VLLMServer)
    - VLLMLoRAClient: ZMQ client with update_weights_from_model for DualLoRA
"""

import os
import time

import msgpack
import numpy as np
import torch
import zmq
from msgspec import field
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from vllm.lora.worker_manager import LRUCacheWorkerLoRAManager

from vllm_utils import flatten_vllm_outputs


# ---------------------------------------------------------------------------
# TensorLoRARequest: LoRARequest that carries weight tensors instead of a path
# ---------------------------------------------------------------------------

class TensorLoRARequest(LoRARequest):
    """LoRARequest extended with in-memory tensors and PEFT config dict."""
    peft_config: dict = field(default=None)
    lora_tensors: dict = field(default=None)


# ---------------------------------------------------------------------------
# Monkey-patch: make vLLM load LoRA from tensors instead of disk
# ---------------------------------------------------------------------------

_hijack_installed = False


def install_tensor_lora_hijack():
    """Monkey-patch vLLM's LRUCacheWorkerLoRAManager to support TensorLoRARequest.

    Safe to call multiple times (idempotent). Only works when EngineCore runs
    in-process (VLLM_ENABLE_V1_MULTIPROCESSING=0).
    """
    global _hijack_installed
    if _hijack_installed:
        return
    _hijack_installed = True

    from vllm.lora.peft_helper import PEFTHelper

    _original_load_adapter = LRUCacheWorkerLoRAManager._load_adapter

    def hijack__load_adapter(self, lora_request):
        if isinstance(lora_request, TensorLoRARequest):
            # In-memory path: load from tensors directly
            peft_helper = PEFTHelper.from_dict(lora_request.peft_config)
            peft_helper.validate_legal(self.lora_config)

            model = self._adapter_manager.model
            hf_to_vllm_mapper = getattr(model, "hf_to_vllm_mapper", None)
            lora_skip_prefixes = getattr(model, "lora_skip_prefixes", None)

            lora = self._lora_model_cls.from_lora_tensors(
                lora_model_id=lora_request.lora_int_id,
                tensors=lora_request.lora_tensors,
                peft_helper=peft_helper,
                device="cpu",
                dtype=self.lora_config.lora_dtype,
                model_vocab_size=self.vocab_size,
                weights_mapper=hf_to_vllm_mapper,
                skip_prefixes=lora_skip_prefixes,
            )
            return lora
        else:
            # Fall back to original disk-based loading
            return _original_load_adapter(self, lora_request)

    LRUCacheWorkerLoRAManager._load_adapter = hijack__load_adapter


# ---------------------------------------------------------------------------
# Engine creation
# ---------------------------------------------------------------------------

def create_lora_engine(model_name, max_lora_rank=64, gpu_memory_utilization=0.05,
                       dtype="bfloat16"):
    """Create a vLLM LLM with native LoRA support enabled.

    Disables vLLM v1 multiprocessing so EngineCore runs in-process,
    which is required for TensorLoRARequest to work (avoids subprocess
    serialization that strips custom fields). The env var is set here
    (inside the server process) rather than globally.
    """
    # Must be set before vLLM reads it during engine init
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    install_tensor_lora_hijack()
    llm = LLM(
        model=model_name,
        enforce_eager=True,
        dtype=dtype,
        gpu_memory_utilization=gpu_memory_utilization,
        enable_lora=True,
        max_loras=2,  # active + one being swapped
        max_lora_rank=max_lora_rank,
    )
    return llm


# ---------------------------------------------------------------------------
# DualLoRA -> vLLM LoRA weight extraction
# ---------------------------------------------------------------------------

def _extract_dual_lora_tensors(model):
    """Extract DualLoRA weights from an HF model as PEFT-format tensor dict.

    Concatenates retain + forget adapters into a single LoRA with
    rank = retain_rank + forget_rank. Scaling is absorbed into A matrices
    so vLLM's LoRA scaling (alpha/r) can be set to 1.0.

    Returns:
        tensors: dict mapping PEFT-format keys to weight tensors
        combined_rank: the effective rank (retain_rank + forget_rank)
        target_modules: list of module short names (e.g. ["q_proj", "k_proj", ...])
    """
    from gradient_routing import DualLoRALinear

    tensors = {}
    target_modules = set()
    combined_rank = None

    for name, module in model.named_modules():
        if not isinstance(module, DualLoRALinear):
            continue

        # Determine the short module name (e.g. "q_proj") and full path
        short_name = name.rsplit(".", 1)[-1]
        target_modules.add(short_name)

        rank_r = module.rank
        rank_f = module.forget_rank
        r_total = rank_r + rank_f
        if combined_rank is None:
            combined_rank = r_total
        assert r_total == combined_rank, \
            f"Mixed ranks not supported: {r_total} vs {combined_rank}"

        # Concatenate A matrices: [A_retain; A_forget] -> [r_total, in_features]
        parts_a = []
        if rank_r > 0:
            parts_a.append(module.lora_A_retain.data * module.scaling)
        if rank_f > 0:
            parts_a.append(module.lora_A_forget.data * module.forget_scaling)
        assert parts_a, f"Both ranks are 0 for {name}"
        lora_a = torch.cat(parts_a, dim=0).detach().cpu().float()

        # Concatenate B matrices: [B_retain, B_forget] -> [out_features, r_total]
        parts_b = []
        if rank_r > 0:
            parts_b.append(module.lora_B_retain.data)
        if rank_f > 0:
            parts_b.append(module.lora_B_forget.data)
        lora_b = torch.cat(parts_b, dim=1).detach().cpu().float()

        # PEFT-format keys: base_model.model.<module_path>.lora_A.weight
        peft_prefix = f"base_model.model.{name}"
        tensors[f"{peft_prefix}.lora_A.weight"] = lora_a
        tensors[f"{peft_prefix}.lora_B.weight"] = lora_b

    assert tensors, "No DualLoRALinear modules found in model"
    return tensors, combined_rank, sorted(target_modules)


def _sync_lora_to_engine(llm, tensors, combined_rank, target_modules):
    """Load LoRA tensors into a vLLM engine via TensorLoRARequest."""
    peft_config = {
        "r": combined_rank,
        "lora_alpha": combined_rank,
        "target_modules": target_modules,
    }
    lora_int_id = int(time.time_ns() % 0x7FFFFFFF)
    request = TensorLoRARequest(
        lora_name=f"dual_lora_{lora_int_id}",
        lora_int_id=lora_int_id,
        lora_path="in_memory",
        peft_config=peft_config,
        lora_tensors=tensors,
    )
    llm.llm_engine.add_lora(request)
    return LoRARequest(
        lora_name=request.lora_name,
        lora_int_id=lora_int_id,
        lora_path="in_memory",
    )


def sync_dual_lora_to_vllm(llm, model):
    """Extract DualLoRA weights from HF model and load into vLLM engine.

    Returns a LoRARequest to pass to llm.generate().
    """
    tensors, combined_rank, target_modules = _extract_dual_lora_tensors(model)
    return _sync_lora_to_engine(llm, tensors, combined_rank, target_modules)


# ---------------------------------------------------------------------------
# ZMQ Server: drop-in replacement for VLLMServer using native LoRA
# ---------------------------------------------------------------------------

class VLLMLoRAServer:
    """ZMQ-based vLLM server with native LoRA support.

    Same protocol as VLLMServer (register, update_weights, generate, shutdown)
    but uses vLLM's built-in LoRA instead of custom MLP adapters.
    """

    def __init__(self, socket_addr, model_name, max_lora_rank=64,
                 gpu_memory_utilization=0.05):
        self.socket_addr = socket_addr

        print(f"[LoRAServer] Creating vLLM engine with LoRA support "
              f"(max_lora_rank={max_lora_rank})...")
        t0 = time.time()
        self.llm = create_lora_engine(
            model_name=model_name,
            max_lora_rank=max_lora_rank,
            gpu_memory_utilization=gpu_memory_utilization,
        )
        print(f"[LoRAServer] Engine ready in {time.time() - t0:.1f}s")

        # ZMQ setup
        self.ctx = zmq.Context()
        self.socket = self.ctx.socket(zmq.REP)
        if socket_addr.startswith("ipc://"):
            sock_path = socket_addr[len("ipc://"):]
            if os.path.exists(sock_path):
                os.unlink(sock_path)
        self.socket.bind(socket_addr)
        print(f"[LoRAServer] Listening on {socket_addr}")

        self._lora_request = None  # current active LoRARequest for generate

    def handle_register(self, msg):
        # LoRA server is single-experiment (one per run)
        return {"experiment_id": 1}

    def handle_update_weights(self, msg):
        """Receive LoRA tensors and load into vLLM engine in-process."""
        peft_config = msg["peft_config"]
        tensor_data = msg["tensors"]
        dtype_str = msg.get("dtype", "float32")
        np_dtype = np.float32 if dtype_str == "float32" else np.float16

        # Reconstruct tensors from raw bytes
        tensors = {}
        for key, raw in tensor_data.items():
            shape = tuple(msg["shapes"][key])
            arr = np.frombuffer(raw, dtype=np_dtype).reshape(shape)
            tensors[key] = torch.from_numpy(arr.copy())

        t0 = time.perf_counter()
        self._lora_request = _sync_lora_to_engine(
            self.llm, tensors,
            peft_config["r"], peft_config["target_modules"],
        )
        t1 = time.perf_counter()
        print(f"[LoRAServer] Weight sync: {(t1-t0)*1000:.0f}ms "
              f"({len(tensors)} tensors, rank={peft_config['r']})")
        return {"ok": True}

    def handle_generate(self, msg):
        prompt_ids = msg["prompt_ids"]
        from vllm import TokensPrompt
        prompts = [TokensPrompt(prompt_token_ids=list(p)) for p in prompt_ids]
        sp_kwargs = dict(
            n=msg["n"],
            temperature=msg["temperature"],
            max_tokens=msg["max_tokens"],
        )
        if msg.get("top_k", 0) > 0:
            sp_kwargs["top_k"] = msg["top_k"]
        if msg.get("top_p", 1.0) < 1.0:
            sp_kwargs["top_p"] = msg["top_p"]
        sp = SamplingParams(**sp_kwargs)
        outputs = self.llm.generate(
            prompts, sp,
            lora_request=self._lora_request,
        )
        comp_texts, comp_ids, prompt_ids_out, _ = flatten_vllm_outputs(outputs)
        return {
            "completion_texts": comp_texts,
            "completion_ids": comp_ids,
            "prompt_ids": prompt_ids_out,
        }

    def run(self, ready_event=None):
        if ready_event is not None:
            ready_event.set()
        print("[LoRAServer] Ready for requests")
        while True:
            raw = self.socket.recv()
            msg = msgpack.unpackb(raw, raw=False)
            op = msg["op"]
            try:
                if op == "register":
                    reply = self.handle_register(msg)
                elif op == "update_weights":
                    reply = self.handle_update_weights(msg)
                elif op == "generate":
                    reply = self.handle_generate(msg)
                elif op == "set_scales":
                    reply = {"ok": True}  # no-op for LoRA
                elif op == "shutdown":
                    self.socket.send(msgpack.packb({"ok": True}, use_bin_type=True))
                    print("[LoRAServer] Shutting down")
                    break
                else:
                    reply = {"error": f"unknown op: {op}"}
            except Exception as e:
                reply = {"error": str(e)}
                import traceback
                traceback.print_exc()
            self.socket.send(msgpack.packb(reply, use_bin_type=True))
        self.ctx.destroy()


# ---------------------------------------------------------------------------
# ZMQ Client: same interface as VLLMClient but sends LoRA tensors
# ---------------------------------------------------------------------------

class VLLMLoRAClient:
    """Client for VLLMLoRAServer. Same interface as VLLMClient."""

    def __init__(self, socket_addr):
        self.ctx = zmq.Context()
        self.socket = self.ctx.socket(zmq.REQ)
        self.socket.connect(socket_addr)

    def _request(self, msg):
        self.socket.send(msgpack.packb(msg, use_bin_type=True))
        raw = self.socket.recv()
        reply = msgpack.unpackb(raw, raw=False)
        if "error" in reply:
            raise RuntimeError(f"Server error: {reply['error']}")
        return reply

    def register(self):
        reply = self._request({"op": "register"})
        return reply["experiment_id"]

    def release(self, experiment_id):
        pass  # no-op for LoRA server

    def update_weights_from_model(self, experiment_id, model):
        """Extract DualLoRA weights and send to server."""
        tensors, combined_rank, target_modules = _extract_dual_lora_tensors(model)

        # Serialize tensors to bytes
        tensor_data = {}
        shapes = {}
        for key, t in tensors.items():
            tensor_data[key] = t.numpy().tobytes()
            shapes[key] = list(t.shape)

        peft_config = {
            "r": combined_rank,
            "lora_alpha": combined_rank,
            "target_modules": target_modules,
        }

        self._request({
            "op": "update_weights",
            "experiment_id": experiment_id,
            "peft_config": peft_config,
            "tensors": tensor_data,
            "shapes": shapes,
            "dtype": "float32",
        })

    def generate(self, experiment_id, prompt_ids, n, temperature, max_tokens,
                 top_k=0, top_p=1.0, return_logprobs=False):
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

    def set_scales(self, experiment_id, retain_scale, forget_scale):
        self._request({
            "op": "set_scales",
            "experiment_id": experiment_id,
            "retain_scale": retain_scale,
            "forget_scale": forget_scale,
        })

    def sleep(self, level=1):
        """No-op for LoRA server — vLLM LoRA engine doesn't support sleep/wake."""
        pass

    def wake_up(self, tags=None):
        """No-op for LoRA server — vLLM LoRA engine doesn't support sleep/wake."""
        pass

    def close(self):
        pass

    def shutdown(self):
        self._request({"op": "shutdown"})
