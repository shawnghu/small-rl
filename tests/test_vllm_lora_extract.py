import os
import sys
import types

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _install_vllm_lora_import_stubs():
    """Stub optional vLLM/msgspec imports so we can unit-test tensor extraction."""
    msgpack = types.ModuleType("msgpack")
    msgpack.packb = lambda *args, **kwargs: b""
    msgpack.unpackb = lambda *args, **kwargs: {}

    msgspec = types.ModuleType("msgspec")
    msgspec.field = lambda default=None: default

    zmq = types.ModuleType("zmq")
    zmq.REP = 0
    zmq.REQ = 1

    class Context:
        def socket(self, *_args, **_kwargs):
            raise NotImplementedError

        def destroy(self):
            pass

    zmq.Context = Context

    vllm = types.ModuleType("vllm")
    vllm.LLM = object
    vllm.SamplingParams = object

    request = types.ModuleType("vllm.lora.request")

    class LoRARequest:
        def __init__(self, lora_name=None, lora_int_id=None, lora_path=None):
            self.lora_name = lora_name
            self.lora_int_id = lora_int_id
            self.lora_path = lora_path

    request.LoRARequest = LoRARequest

    worker_manager = types.ModuleType("vllm.lora.worker_manager")

    class LRUCacheWorkerLoRAManager:
        def _load_adapter(self, lora_request):
            raise NotImplementedError

    worker_manager.LRUCacheWorkerLoRAManager = LRUCacheWorkerLoRAManager

    sys.modules.setdefault("msgpack", msgpack)
    sys.modules.setdefault("msgspec", msgspec)
    sys.modules.setdefault("zmq", zmq)
    sys.modules.setdefault("vllm", vllm)
    sys.modules.setdefault("vllm.lora", types.ModuleType("vllm.lora"))
    sys.modules.setdefault("vllm.lora.request", request)
    sys.modules.setdefault("vllm.lora.worker_manager", worker_manager)


class TinyCausalLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = types.SimpleNamespace(num_hidden_layers=1)
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([nn.Module()])
        layer = self.model.layers[0]
        layer.self_attn = nn.Module()
        layer.self_attn.q_proj = nn.Linear(3, 5, bias=False)


class Wrapper(nn.Module):
    def __init__(self, attr_name, wrapped):
        super().__init__()
        setattr(self, attr_name, wrapped)


def test_extract_dual_lora_tensors_strips_compile_wrapper_prefix():
    _install_vllm_lora_import_stubs()
    from gradient_routing import apply_dual_lora
    from vllm_lora import _extract_dual_lora_tensors

    model = TinyCausalLM()
    apply_dual_lora(model, rank=2, forget_rank=0, alpha=2, projections=["q_proj"])

    compiled_like = Wrapper("_orig_mod", model)
    tensors, rank, targets = _extract_dual_lora_tensors(compiled_like)

    assert rank == 2
    assert targets == ["q_proj"]
    assert "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight" in tensors
    assert all("_orig_mod" not in key for key in tensors)


def test_extract_dual_lora_tensors_strips_nested_ddp_compile_prefixes():
    _install_vllm_lora_import_stubs()
    from gradient_routing import apply_dual_lora
    from vllm_lora import _extract_dual_lora_tensors

    model = TinyCausalLM()
    apply_dual_lora(model, rank=2, forget_rank=0, alpha=2, projections=["q_proj"])

    wrapped = Wrapper("module", Wrapper("_orig_mod", model))
    tensors, _, _ = _extract_dual_lora_tensors(wrapped)

    assert "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight" in tensors
    assert all(".module." not in key for key in tensors)
    assert all("._orig_mod." not in key for key in tensors)
