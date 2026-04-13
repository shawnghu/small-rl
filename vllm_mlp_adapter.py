"""MLP adapter support for vLLM with LoRA infrastructure for routing/caching.

Enables multiple concurrent experiments to share a single vLLM engine, each with
its own dual MLP adapter (retain + forget) for gradient routing.

Uses vLLM's LoRA infrastructure for:
  - Per-request adapter identity (LoRARequest attached to each generation request)
  - KV cache awareness (prefix cache hashes include adapter name)
  - Scheduler adapter tracking (request_lora_mapping populated automatically)

No linear layers are LoRA-wrapped (zero Punica kernel overhead). MLP adapters are
injected at engine init time via LoRAModelManager._post_create_module_hooks, so
they are present before profiling and CUDA graph capture. Each adapter holds a
reference to the shared PunicaWrapper, reading per-token slot indices from
token_lora_indices (a fixed-address GPU tensor updated in-place each step),
which enables CUDA graph compatibility.

In-process engine (VLLM_ENABLE_V1_MULTIPROCESSING=0) enables direct model access
for weight updates without serialization overhead.

Two engine modes:
  - Synchronous (LLM): create_engine() — fully implemented
  - Async (AsyncLLM): create_async_engine() — TODO: not yet updated for LoRA infra

Usage (sync):
    from vllm import SamplingParams
    from vllm_mlp_adapter import create_engine

    llm, mgr = create_engine(max_experiments=20, retain_neurons=32, forget_neurons=32)
    mgr.set_weights(experiment_id=1, layer_weights=[...])
    outputs = mgr.generate(["Once upon a time"], experiment_ids=[1],
                           sampling_params=SamplingParams(temperature=0, max_tokens=50))
"""

import os

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Dummy adapter loader: return empty LoRAModel for our experiment adapters
# ---------------------------------------------------------------------------

def _install_dummy_adapter_loader() -> None:
    """Monkey-patch _load_adapter to return empty LoRAModel for MLP experiment adapters.

    When vLLM's worker first sees a LoRARequest with lora_name starting with
    "mlp_exp_", it returns an empty LoRAModel instead of loading from disk.
    activate_adapter() is a no-op (no wrapped modules), so the adapter is
    registered for scheduling/caching purposes only.

    This runs inside the vLLM EngineCore subprocess via apply_model().
    """
    from vllm.lora.worker_manager import LRUCacheWorkerLoRAManager
    from vllm.lora.lora_model import LoRAModel

    if getattr(LRUCacheWorkerLoRAManager, "_mlp_adapter_loader_installed", False):
        return  # idempotent

    _orig = LRUCacheWorkerLoRAManager._load_adapter

    def _hijacked(self, lora_request):
        if (lora_request.lora_name.startswith("mlp_exp_")
                or lora_request.lora_name.startswith("warmup_")):
            return LoRAModel(lora_request.lora_int_id, rank=1, loras={})
        return _orig(self, lora_request)

    LRUCacheWorkerLoRAManager._load_adapter = _hijacked
    LRUCacheWorkerLoRAManager._mlp_adapter_loader_installed = True


def _prevent_lora_module_wrapping() -> None:
    """Monkey-patch _match_target_modules to prevent LoRA wrapping any modules.

    vLLM 0.17.0 doesn't support lora_target_modules config, so we prevent
    module wrapping by making _match_target_modules always return False.
    This is equivalent to lora_target_modules=["__nonexistent__"] in newer vLLM.

    Also patches add_dummy_lora to be a no-op, since with no wrapped modules
    the dummy LoRA would have empty loras={} and crash in
    _create_merged_loras_inplace.

    Must be called BEFORE engine creation (patches the class method).
    """
    from vllm.lora.model_manager import LoRAModelManager
    from vllm.lora.worker_manager import LRUCacheWorkerLoRAManager

    if getattr(LoRAModelManager, "_no_wrap_installed", False):
        return

    LoRAModelManager._match_target_modules = lambda self, module_name: False
    LoRAModelManager._no_wrap_installed = True

    # Dummy LoRA creation crashes when no modules are wrapped (empty loras dict).
    # Make it a no-op since we only use the LoRA infra for scheduling/caching.
    LRUCacheWorkerLoRAManager.add_dummy_lora = lambda self, lora_request, rank: False

    # _create_merged_loras_inplace crashes on empty loras dict (line 656:
    # next(iter(lora_model.loras.values())) → StopIteration for warmup dummies).
    # Also need to skip for MLP adapter loras since they use list-format weights
    # that don't need packed-module merging.
    _orig_create_merged = LoRAModelManager._create_merged_loras_inplace

    def _safe_create_merged(self, lora_model):
        if not lora_model.loras:
            return  # empty dict (warmup dummies)
        # Check if any lora has list-format weights (MLP adapter) — skip merging
        first = next(iter(lora_model.loras.values()))
        if isinstance(first.lora_a, list):
            return  # MLP adapter weights, no packed-module merging needed
        return _orig_create_merged(self, lora_model)

    LoRAModelManager._create_merged_loras_inplace = _safe_create_merged


# ---------------------------------------------------------------------------
# VLLMDualMLPAdapter — wraps vLLM's LlamaMLP with multi-experiment adapters
# ---------------------------------------------------------------------------

class VLLMDualMLPAdapter(nn.Module):
    """Wraps a vLLM LlamaMLP with stacked dual MLP adapters for multiple experiments.

    Each experiment slot has retain + forget SwiGLU adapter networks. The forward
    pass reads _token_experiment_ids (set by the execute_model hook) to route each
    token to the correct experiment's adapter weights.
    """

    def __init__(self, base_mlp: nn.Module, hidden_size: int, max_adapters: int,
                 retain_neurons: int, forget_neurons: int):
        super().__init__()
        self.base_mlp = base_mlp
        self.max_adapters = max_adapters
        self.retain_neurons = retain_neurons
        self.forget_neurons = forget_neurons
        self.hidden_dim = hidden_size

        device = next(base_mlp.parameters()).device
        dtype = next(base_mlp.parameters()).dtype

        # Pre-allocate stacked weight buffers in Punica format:
        # shrink (gate/up): (max_loras, 1, neurons, hidden) — used with add_shrink
        # expand (down):    (max_loras, 1, hidden, neurons) — used with add_expand
        # This enables reuse of Punica's Triton kernels for per-token routing,
        # making the forward torch.compile and CUDA-graph compatible.
        if retain_neurons > 0:
            self.retain_gate_stacked = nn.Parameter(
                torch.zeros(max_adapters, 1, retain_neurons, hidden_size, device=device, dtype=dtype),
                requires_grad=False)
            self.retain_up_stacked = nn.Parameter(
                torch.zeros(max_adapters, 1, retain_neurons, hidden_size, device=device, dtype=dtype),
                requires_grad=False)
            self.retain_down_stacked = nn.Parameter(
                torch.zeros(max_adapters, 1, hidden_size, retain_neurons, device=device, dtype=dtype),
                requires_grad=False)
        else:
            self.retain_gate_stacked = self.retain_up_stacked = self.retain_down_stacked = None

        if forget_neurons > 0:
            self.forget_gate_stacked = nn.Parameter(
                torch.zeros(max_adapters, 1, forget_neurons, hidden_size, device=device, dtype=dtype),
                requires_grad=False)
            self.forget_up_stacked = nn.Parameter(
                torch.zeros(max_adapters, 1, forget_neurons, hidden_size, device=device, dtype=dtype),
                requires_grad=False)
            self.forget_down_stacked = nn.Parameter(
                torch.zeros(max_adapters, 1, hidden_size, forget_neurons, device=device, dtype=dtype),
                requires_grad=False)
        else:
            self.forget_gate_stacked = self.forget_up_stacked = self.forget_down_stacked = None

        # Per-experiment scales: (max_adapters, 2) for [retain_scale, forget_scale]
        self.scales = torch.ones(max_adapters, 2, device=device, dtype=dtype)

    def set_weights(self, slot: int, gate_r, up_r, down_r, gate_f, up_f, down_f):
        """Load adapter weights for one experiment slot.

        Weights are stored in Punica stacked format: (max_loras, 1, out, in).
        Training-side weights are (out, in) so they copy directly into [slot, 0].
        """
        if self.retain_gate_stacked is not None and gate_r is not None:
            self.retain_gate_stacked.data[slot, 0, :gate_r.shape[0], :gate_r.shape[1]].copy_(gate_r)
            self.retain_up_stacked.data[slot, 0, :up_r.shape[0], :up_r.shape[1]].copy_(up_r)
            self.retain_down_stacked.data[slot, 0, :down_r.shape[0], :down_r.shape[1]].copy_(down_r)
        if self.forget_gate_stacked is not None and gate_f is not None:
            self.forget_gate_stacked.data[slot, 0, :gate_f.shape[0], :gate_f.shape[1]].copy_(gate_f)
            self.forget_up_stacked.data[slot, 0, :up_f.shape[0], :up_f.shape[1]].copy_(up_f)
            self.forget_down_stacked.data[slot, 0, :down_f.shape[0], :down_f.shape[1]].copy_(down_f)

    def set_adapter_weights(self, index: int, weights):
        """LoRA manager interface: receive weights during activate_adapter().

        Args:
            index: adapter slot index (assigned by LoRA manager)
            weights: LoRALayerWeights with MLP adapter tensors packed in lora_a/lora_b.
                lora_a packs: [gate_retain, up_retain, down_retain]
                lora_b packs: [gate_forget, up_forget, down_forget]
        """
        self.reset_lora(index)
        if weights.lora_a is not None and self.retain_gate_stacked is not None:
            gate_r, up_r, down_r = weights.lora_a
            self.retain_gate_stacked.data[index, 0, :gate_r.shape[0], :gate_r.shape[1]].copy_(gate_r, non_blocking=True)
            self.retain_up_stacked.data[index, 0, :up_r.shape[0], :up_r.shape[1]].copy_(up_r, non_blocking=True)
            self.retain_down_stacked.data[index, 0, :down_r.shape[0], :down_r.shape[1]].copy_(down_r, non_blocking=True)
        if weights.lora_b is not None and self.forget_gate_stacked is not None:
            gate_f, up_f, down_f = weights.lora_b
            self.forget_gate_stacked.data[index, 0, :gate_f.shape[0], :gate_f.shape[1]].copy_(gate_f, non_blocking=True)
            self.forget_up_stacked.data[index, 0, :up_f.shape[0], :up_f.shape[1]].copy_(up_f, non_blocking=True)
            self.forget_down_stacked.data[index, 0, :down_f.shape[0], :down_f.shape[1]].copy_(down_f, non_blocking=True)

    def reset_lora(self, index: int):
        """LoRA manager interface: zero weights for one adapter slot."""
        if self.retain_gate_stacked is not None:
            self.retain_gate_stacked.data[index].zero_()
            self.retain_up_stacked.data[index].zero_()
            self.retain_down_stacked.data[index].zero_()
        if self.forget_gate_stacked is not None:
            self.forget_gate_stacked.data[index].zero_()
            self.forget_up_stacked.data[index].zero_()
            self.forget_down_stacked.data[index].zero_()
        self.scales[index, 0] = 1.0
        self.scales[index, 1] = 1.0

    def set_mapping(self, punica_wrapper):
        """Store reference to the PunicaWrapper for per-token routing.

        Same interface as BaseLayerWithLoRA.set_mapping(). The wrapper's
        token_lora_indices property provides per-token slot indices, updated
        in-place each step by the LoRA infrastructure. Reading from a
        fixed-address GPU tensor enables CUDA graph compatibility.
        """
        self.punica_wrapper = punica_wrapper

    def set_scales(self, slot: int, retain_scale: float, forget_scale: float):
        """Set retain/forget scales for one experiment slot."""
        self.scales[slot, 0] = retain_scale
        self.scales[slot, 1] = forget_scale

    def _adapter_swiglu(self, x: torch.Tensor, gate_stacked, up_stacked, down_stacked):
        """Compute SwiGLU adapter using Punica shrink/expand kernels.

        gate/up use add_shrink (hidden → neurons per-token routing).
        down uses add_expand (neurons → hidden per-token routing).
        SiLU activation + elementwise multiply happen in between (standard ops).

        All per-token routing uses the same token_lora_indices from the
        PunicaWrapper, shared with LoRA layers.
        """
        num_tokens = x.shape[0]
        neurons = gate_stacked.shape[2]

        # Shrink: x @ gate[slot] → gate_out, x @ up[slot] → up_out
        # Output shape: (2, num_tokens, neurons) — 2 slices for gate and up
        shrink_buf = torch.empty(
            (2, num_tokens, neurons), dtype=torch.float32, device=x.device,
        )
        self.punica_wrapper.add_shrink(
            shrink_buf, x, (gate_stacked, up_stacked), 1.0,
        )
        gate_out = shrink_buf[0]
        up_out = shrink_buf[1]

        # SiLU activation + elementwise multiply (standard torch ops)
        intermediate = (F.silu(gate_out) * up_out).unsqueeze(0)  # (1, T, N)

        # Expand: intermediate @ down[slot] → adapter_out
        # add_expand adds into y in-place, so we pass base_out=zeros
        adapter_out = torch.zeros(num_tokens, self.hidden_dim, dtype=x.dtype, device=x.device)
        self.punica_wrapper.add_expand(
            adapter_out, intermediate, (down_stacked,),
            output_slices=(self.hidden_dim,), add_inputs=False,
        )
        return adapter_out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_mlp(x)

        if not hasattr(self, 'punica_wrapper'):
            return base_out

        # Compute adapter corrections using Punica kernels (same per-token
        # routing mechanism as LoRA — reads token_lora_indices from a
        # fixed-address GPU tensor, CUDA graph compatible).

        # Per-token scales: token_lora_indices maps each token to a slot
        # (or -1 for no adapter). Clamp to 0 so -1 tokens read slot 0's
        # scales; this is safe because Punica already returns zero output
        # for those tokens, so scale * 0 = 0 regardless.
        token_indices = self.punica_wrapper.token_lora_indices.clamp(min=0)

        if self.retain_gate_stacked is not None:
            retain_out = self._adapter_swiglu(
                x, self.retain_gate_stacked, self.retain_up_stacked, self.retain_down_stacked,
            )
            retain_scales = self.scales[token_indices, 0].unsqueeze(-1)
            base_out = base_out + retain_out * retain_scales

        if self.forget_gate_stacked is not None:
            forget_out = self._adapter_swiglu(
                x, self.forget_gate_stacked, self.forget_up_stacked, self.forget_down_stacked,
            )
            forget_scales = self.scales[token_indices, 1].unsqueeze(-1)
            base_out = base_out + forget_out * forget_scales

        return base_out


# ---------------------------------------------------------------------------
# Model surgery: inject MLP adapters + routing hook into a vLLM model
# ---------------------------------------------------------------------------

def inject_mlp_adapters(model: nn.Module, max_adapters: int,
                        retain_neurons: int, forget_neurons: int,
                        layer_indices: list[int] | None = None) -> list[int]:
    """Replace LlamaMLP modules with VLLMDualMLPAdapter and install hooks.

    This is an apply_model() callback — it runs inside the vLLM worker process.
    Installs both the routing hook (reads request_lora_mapping) and the dummy
    adapter loader (returns empty LoRAModel for mlp_exp_ adapters).

    Args:
        layer_indices: Which layers to adapt. None = all layers.
    """
    from gradient_routing import find_mlp_modules

    # Install hooks (both idempotent)
    _inject_routing_hook(model)
    _install_dummy_adapter_loader()

    hidden_size = model.config.hidden_size
    mlp_entries = find_mlp_modules(model, layer_indices)

    modified = []
    for layer_idx, path, parent, base_mlp in mlp_entries:
        adapter = VLLMDualMLPAdapter(
            base_mlp, hidden_size, max_adapters, retain_neurons, forget_neurons,
        )
        setattr(parent, "mlp", adapter)
        modified.append(layer_idx)
    return modified


# ---------------------------------------------------------------------------
# VLLMAdapterManager — high-level orchestration (sync LLM)
# ---------------------------------------------------------------------------

def _pack_mlp_weights_as_lora_model(layer_indices, layer_weights, adapter_id):
    """Pack MLP adapter weights into a LoRAModel for the LoRA manager.

    Each layer's retain weights go into lora_a (as a list of 3 tensors),
    forget weights into lora_b. The VLLMDualMLPAdapter.set_adapter_weights()
    unpacks them.

    Returns a LoRAModel keyed by module path (e.g. "model.layers.0.mlp").
    """
    from vllm.lora.lora_model import LoRAModel
    from vllm.lora.lora_weights import LoRALayerWeights

    loras = {}
    for j, layer_idx in enumerate(layer_indices):
        w = layer_weights[j]
        # Pack retain as lora_a, forget as lora_b (each is a list of 3 tensors)
        lora_a = None
        if w.get("gate_retain") is not None:
            lora_a = [w["gate_retain"], w["up_retain"], w["down_retain"]]
        lora_b = None
        if w.get("gate_forget") is not None:
            lora_b = [w["gate_forget"], w["up_forget"], w["down_forget"]]

        module_name = f"model.layers.{layer_idx}.mlp"
        loras[module_name] = LoRALayerWeights(
            module_name=module_name,
            rank=1,  # not meaningful for MLP adapters
            lora_alpha=1,
            lora_a=lora_a,
            lora_b=lora_b,
            scaling=1.0,
        )

    return LoRAModel(lora_model_id=adapter_id, rank=1, loras=loras)


class VLLMAdapterManager:
    """Manages MLP adapters across a shared vLLM engine (sync LLM).

    Weight updates go through the LoRA manager's add_adapter/activate_adapter
    path (same as LoRA), with MLP weights packed into LoRALayerWeights. Each
    update registers a new adapter ID so the LoRA infrastructure handles cache
    invalidation natively.
    """

    def __init__(self, llm, model: nn.Module, max_experiments: int,
                 retain_neurons: int, forget_neurons: int,
                 layer_indices: list[int]):
        self.llm = llm
        self.model = model
        self.max_experiments = max_experiments
        self.retain_neurons = retain_neurons
        self.forget_neurons = forget_neurons
        self.layer_indices = layer_indices
        # Monotonic adapter ID counter. Each weight update gets a new ID
        # (same pattern as vllm_lora.py), so the LoRA manager treats it as
        # a new adapter and handles cache invalidation natively.
        self._next_adapter_id = 1
        # Maps experiment_id → current LoRARequest for generate()
        self._active_lora_requests: dict[int, "LoRARequest"] = {}

    def set_weights(self, experiment_id: int, layer_weights: list[dict]):
        """Push adapter weights via the LoRA manager's activation path.

        Creates a new adapter (new ID) each time, mirroring how vllm_lora.py
        handles mutable weights. The LoRA manager's activate_adapter() calls
        set_adapter_weights() on each registered VLLMDualMLPAdapter.

        Args:
            experiment_id: 1-indexed experiment ID
            layer_weights: List of dicts (one per adapted layer) with keys:
                gate_retain, up_retain, down_retain,
                gate_forget, up_forget, down_forget
        """
        from vllm.lora.request import LoRARequest

        assert 1 <= experiment_id <= self.max_experiments
        assert len(layer_weights) == len(self.layer_indices), \
            f"Expected {len(self.layer_indices)} layer weight dicts, got {len(layer_weights)}"

        adapter_id = self._next_adapter_id
        self._next_adapter_id += 1

        lora_model = _pack_mlp_weights_as_lora_model(
            self.layer_indices, layer_weights, adapter_id,
        )

        # Register and activate through the LoRA manager (same path as LoRA).
        # The manager calls set_adapter_weights() on our VLLMDualMLPAdapter modules.
        lora_manager = self.model.lora_manager
        lora_manager._add_adapter(lora_model)
        lora_manager.activate_adapter(adapter_id)

        # Store the LoRARequest for generate()
        lora_name = f"mlp_exp_{experiment_id}_v{adapter_id}"
        self._active_lora_requests[experiment_id] = LoRARequest(
            lora_name=lora_name,
            lora_int_id=adapter_id,
            lora_path="__dummy__",
        )

    def set_scales(self, experiment_id: int, retain_scale: float, forget_scale: float):
        """Set retain/forget scales for one experiment."""
        assert 1 <= experiment_id <= self.max_experiments
        # Scales are set on the adapter module directly (not through LoRA manager)
        # because they're an inference-time parameter, not adapter weights.
        # We need to find which slot index this experiment currently occupies.
        lora_req = self._active_lora_requests.get(experiment_id)
        if lora_req is None:
            raise ValueError(f"No active adapter for experiment {experiment_id}. Call set_weights first.")
        lora_manager = self.model.lora_manager
        try:
            slot_index = lora_manager.lora_index_to_id.index(lora_req.lora_int_id)
        except ValueError:
            raise ValueError(f"Adapter {lora_req.lora_int_id} not active in LoRA manager")

        for layer_idx in self.layer_indices:
            self.model.model.layers[layer_idx].mlp.set_scales(slot_index, retain_scale, forget_scale)

    def update_from_training_model(self, experiment_id: int, training_model: nn.Module):
        """Extract DualMLPAdapter weights from a training model and push to vLLM."""
        from gradient_routing import DualMLPAdapter

        layer_weights = []
        for module in training_model.modules():
            if isinstance(module, DualMLPAdapter):
                w = {}
                if module.gate_retain is not None:
                    w["gate_retain"] = module.gate_retain.weight.data.clone()
                    w["up_retain"] = module.up_retain.weight.data.clone()
                    w["down_retain"] = module.down_retain.weight.data.clone()
                if module.gate_forget is not None:
                    w["gate_forget"] = module.gate_forget.weight.data.clone()
                    w["up_forget"] = module.up_forget.weight.data.clone()
                    w["down_forget"] = module.down_forget.weight.data.clone()
                layer_weights.append(w)

        assert len(layer_weights) == len(self.layer_indices), \
            f"Found {len(layer_weights)} DualMLPAdapter layers, expected {len(self.layer_indices)}"
        self.set_weights(experiment_id, layer_weights)

    def generate(self, prompts, experiment_ids: list[int], sampling_params=None):
        """Generate completions with per-prompt experiment routing.

        Attaches LoRARequest to each request for adapter-aware scheduling and
        KV cache hashing. The routing hook reads request_lora_mapping from the
        LoRA infrastructure to build per-token experiment indices.
        """
        import uuid
        from vllm import TokensPrompt

        assert len(prompts) == len(experiment_ids)
        for eid in experiment_ids:
            assert 1 <= eid <= self.max_experiments, \
                f"experiment_id {eid} out of range [1, {self.max_experiments}]"

        if prompts and isinstance(prompts[0], (list, tuple)):
            prompts = [TokensPrompt(prompt_token_ids=list(p)) for p in prompts]

        batch_id = uuid.uuid4().hex[:8]
        engine = self.llm.llm_engine

        # Submit requests with the active LoRARequest for each experiment.
        req_ids = []
        for i, (prompt, eid) in enumerate(zip(prompts, experiment_ids)):
            req_id = f"{i}_{batch_id}"
            lora_req = self._active_lora_requests.get(eid)
            assert lora_req is not None, \
                f"No active adapter for experiment {eid}. Call set_weights first."
            engine.add_request(req_id, prompt, sampling_params,
                               lora_request=lora_req)
            req_ids.append(req_id)

        # Run engine until all complete, collect outputs keyed by request_id.
        finished_comps = {r: {} for r in req_ids}
        outputs_by_id = {}
        while engine.has_unfinished_requests():
            for out in engine.step():
                for comp in out.outputs:
                    if comp.finish_reason is not None:
                        finished_comps[out.request_id][comp.index] = comp
                if out.finished:
                    out.outputs = sorted(
                        finished_comps[out.request_id].values(), key=lambda c: c.index
                    )
                    outputs_by_id[out.request_id] = out

        # Return in original prompt order.
        return [outputs_by_id[r] for r in req_ids]


# ---------------------------------------------------------------------------
# Convenience: create engine + manager in one call
# ---------------------------------------------------------------------------

def _compute_layer_indices(num_layers: int,
                           layer_start: float = 0.0,
                           layer_end: float = 1.0,
                           layer_stride: int = 1) -> list[int]:
    """Deprecated: use gradient_routing.compute_layer_indices instead."""
    from gradient_routing import compute_layer_indices
    return compute_layer_indices(num_layers, layer_start, layer_end, layer_stride)


def create_engine(
    model_name: str = "HuggingFaceTB/SmolLM2-135M-Instruct",
    max_experiments: int = 20,
    retain_neurons: int = 32,
    forget_neurons: int = 32,
    gpu_memory_utilization: float = 0.05,
    dtype: str = "bfloat16",
    layer_start: float = 0.0,
    layer_end: float = 1.0,
    layer_stride: int = 1,
):
    """Create a vLLM engine with MLP adapter support. Returns (llm, manager).

    Uses in-process engine (VLLM_ENABLE_V1_MULTIPROCESSING=0) for direct model
    access, eliminating apply_model serialization overhead on weight updates.

    MLP adapters are injected during engine init via a _post_create_module_hooks
    callback on LoRAModelManager, so they are present before profiling and CUDA
    graph capture. enforce_eager is set below; see comment there.
    """
    import os
    print(f"[vLLM] Engine dtype: {dtype}")
    from vllm import LLM
    from vllm.lora.model_manager import LoRAModelManager
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(model_name)
    num_layers = config.num_hidden_layers
    layer_indices = _compute_layer_indices(num_layers, layer_start, layer_end, layer_stride)
    print(f"[vLLM] Will adapt layers {layer_indices} ({len(layer_indices)}/{num_layers})")

    # In-process engine for direct model access (no cloudpickle overhead)
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

    # Prevent LoRA from wrapping any linear modules (we only want the
    # infrastructure: scheduling, KV cache hashing, request_lora_mapping).
    _prevent_lora_module_wrapping()
    # Install dummy adapter loader so vLLM doesn't try to load from disk
    _install_dummy_adapter_loader()

    # Register MLP adapter injection as a post-create hook so adapters are
    # present before profiling / CUDA graph capture / warmup.
    # The hook registers each adapter with the LoRA manager and gives it a
    # reference to the PunicaWrapper for per-token routing (same mechanism
    # as LoRA layers — reads token_lora_indices from a fixed-address GPU
    # tensor, enabling CUDA graph compatibility).
    def _mlp_hook(model):
        from gradient_routing import find_mlp_modules

        lora_manager = getattr(model, 'lora_manager', None)
        assert lora_manager is not None, \
            "lora_manager not set on model — _post_create_module_hooks " \
            "should run after model.lora_manager = self in __init__"
        # Get the punica wrapper (same one LoRA layers would use)
        from vllm.lora.model_manager import DEFAULT_LANGUAGE_WRAPPER_KEY
        punica_wrapper = lora_manager.punica_wrapper_mapping[DEFAULT_LANGUAGE_WRAPPER_KEY]
        hidden_size = model.config.hidden_size
        mlp_entries = find_mlp_modules(model, layer_indices)
        for layer_idx, path, parent, base_mlp in mlp_entries:
            adapter = VLLMDualMLPAdapter(
                base_mlp, hidden_size, max_experiments, retain_neurons, forget_neurons,
            )
            setattr(parent, "mlp", adapter)
            # Register with LoRA manager (weight lifecycle) and punica wrapper (routing)
            lora_manager.register_module(path, adapter)
            adapter.set_mapping(punica_wrapper)
        print(f"[vLLM] Injected MLP adapters on layers {layer_indices}")

    LoRAModelManager._post_create_module_hooks.append(_mlp_hook)

    try:
        llm = LLM(
            model=model_name,
            # enforce_eager disables both torch.compile and CUDA graph capture.
            # There is no known logical reason this is required — MLP adapters
            # are injected at init time (before graph capture), use Punica
            # kernels (opaque to torch.compile), and read routing state from
            # fixed-address GPU tensors (CUDA-graph compatible). However, we
            # observed silent training degradation ("learning slower") with
            # enforce_eager=False that we could not fully explain. Stale
            # torch.compile cache (which bakes in adapter tensor shapes but
            # doesn't include adapter dimensions in its cache key) was one
            # confirmed failure mode, but clearing the cache did not fully
            # resolve our confidence in the compiled path. Until the compiled
            # path is verified against an eager baseline on a full sweep,
            # enforce_eager=True is the safe default.
            enforce_eager=True,
            dtype=dtype,
            gpu_memory_utilization=gpu_memory_utilization,
            enable_sleep_mode=True,
            enable_lora=True,
            max_loras=max_experiments,
            max_lora_rank=8,
            disable_log_stats=False,
        )
    finally:
        # Clean up hook so it doesn't fire on unrelated engine creations
        LoRAModelManager._post_create_module_hooks.remove(_mlp_hook)

    # Direct model access (in-process engine)
    model = llm.llm_engine.model_executor.driver_worker.model_runner.model

    mgr = VLLMAdapterManager(
        llm=llm,
        model=model,
        max_experiments=max_experiments,
        retain_neurons=retain_neurons,
        forget_neurons=forget_neurons,
        layer_indices=layer_indices,
    )

    return llm, mgr


# ---------------------------------------------------------------------------
# Async adapter manager (for AsyncLLM / dynamic batching)
# ---------------------------------------------------------------------------

class AsyncVLLMAdapterManager:
    """Async version of VLLMAdapterManager for use with AsyncLLM."""

    def __init__(self, engine, max_experiments: int,
                 retain_neurons: int, forget_neurons: int,
                 layer_indices: list[int]):
        self.engine = engine
        self.max_experiments = max_experiments
        self.retain_neurons = retain_neurons
        self.forget_neurons = forget_neurons
        self.layer_indices = layer_indices

    async def setup(self):
        """Inject MLP adapters and routing hook via collective_rpc."""
        max_adapters = self.max_experiments
        retain_neurons = self.retain_neurons
        forget_neurons = self.forget_neurons
        indices = self.layer_indices

        def _inject(model):
            return inject_mlp_adapters(model, max_adapters, retain_neurons, forget_neurons,
                                       layer_indices=indices)

        results = await self.engine.collective_rpc("apply_model", args=(_inject,))
        modified_layers = results[0]
        assert modified_layers == self.layer_indices, \
            f"Expected layers {self.layer_indices} modified, got {modified_layers}"

    async def set_weights(self, experiment_id: int, layer_weights: list[dict]):
        assert 1 <= experiment_id <= self.max_experiments
        assert len(layer_weights) == len(self.layer_indices), \
            f"Expected {len(self.layer_indices)} layer weight dicts, got {len(layer_weights)}"
        slot = experiment_id - 1
        indices = self.layer_indices

        def _set(model):
            for j, layer_idx in enumerate(indices):
                w = layer_weights[j]
                model.model.layers[layer_idx].mlp.set_weights(
                    slot,
                    w.get("gate_retain"), w.get("up_retain"), w.get("down_retain"),
                    w.get("gate_forget"), w.get("up_forget"), w.get("down_forget"),
                )

        await self.engine.collective_rpc("apply_model", args=(_set,))
        await self.engine.reset_prefix_cache()

    async def set_scales(self, experiment_id: int, retain_scale: float, forget_scale: float):
        assert 1 <= experiment_id <= self.max_experiments
        slot = experiment_id - 1
        indices = self.layer_indices

        def _set(model):
            for layer_idx in indices:
                model.model.layers[layer_idx].mlp.set_scales(slot, retain_scale, forget_scale)

        await self.engine.collective_rpc("apply_model", args=(_set,))

    async def reset_weights(self, experiment_id: int):
        """Zero all adapter weights and reset scales for one experiment slot."""
        assert 1 <= experiment_id <= self.max_experiments
        slot = experiment_id - 1
        indices = self.layer_indices

        def _reset(model):
            for layer_idx in indices:
                adapter = model.model.layers[layer_idx].mlp
                if adapter.retain_gate is not None:
                    adapter.retain_gate.data[slot].zero_()
                    adapter.retain_up.data[slot].zero_()
                    adapter.retain_down.data[slot].zero_()
                if adapter.forget_gate is not None:
                    adapter.forget_gate.data[slot].zero_()
                    adapter.forget_up.data[slot].zero_()
                    adapter.forget_down.data[slot].zero_()
                adapter.scales[slot, 0] = 1.0
                adapter.scales[slot, 1] = 1.0

        await self.engine.collective_rpc("apply_model", args=(_reset,))

    async def update_from_training_model(self, experiment_id: int, training_model: nn.Module):
        from gradient_routing import DualMLPAdapter

        layer_weights = []
        for module in training_model.modules():
            if isinstance(module, DualMLPAdapter):
                w = {}
                if module.gate_retain is not None:
                    w["gate_retain"] = module.gate_retain.weight.data.clone()
                    w["up_retain"] = module.up_retain.weight.data.clone()
                    w["down_retain"] = module.down_retain.weight.data.clone()
                if module.gate_forget is not None:
                    w["gate_forget"] = module.gate_forget.weight.data.clone()
                    w["up_forget"] = module.up_forget.weight.data.clone()
                    w["down_forget"] = module.down_forget.weight.data.clone()
                layer_weights.append(w)

        assert len(layer_weights) == len(self.layer_indices), \
            f"Found {len(layer_weights)} DualMLPAdapter layers, expected {len(self.layer_indices)}"
        await self.set_weights(experiment_id, layer_weights)

    async def generate(self, prompts, experiment_ids: list[int], sampling_params=None):
        """Generate with per-prompt experiment routing via AsyncLLM.

        Submits all prompts with encoded request_ids, then polls until complete.
        Supports n>1 sampling via ParentRequest fan-out (same as AsyncLLM internals).
        """
        import asyncio
        import uuid
        import zmq
        from copy import copy
        from vllm import TokensPrompt
        from vllm.v1.engine.core_client import EngineCoreRequestType
        from vllm.v1.engine.output_processor import RequestOutputCollector
        from vllm.v1.engine.parallel_sampling import ParentRequest

        assert len(prompts) == len(experiment_ids)
        for eid in experiment_ids:
            assert 1 <= eid <= self.max_experiments, \
                f"experiment_id {eid} out of range [1, {self.max_experiments}]"

        if prompts and isinstance(prompts[0], (list, tuple)):
            prompts = [TokensPrompt(prompt_token_ids=list(p)) for p in prompts]

        batch_id = uuid.uuid4().hex[:12]
        n_prompts = len(prompts)
        engine_core = self.engine.engine_core

        supported_tasks = await self.engine.get_supported_tasks()

        # Set up output handler + ZMQ socket before the per-prompt loop.
        self.engine._run_output_handler()
        engine_core.ensure_alive()
        sync_socket = zmq.Socket.shadow(engine_core.input_socket)
        engine_core._ensure_output_queue_task()

        def _zmq_send(request):
            request.client_index = engine_core.client_index
            msg = (
                engine_core.core_engine,
                EngineCoreRequestType.ADD.value,
                *engine_core.encoder.encode(request),
            )
            sync_socket.send_multipart(msg, copy=False)

        # Process each prompt and ZMQ-send its children immediately.
        queues = []
        for i, (prompt, eid) in enumerate(zip(prompts, experiment_ids)):
            slot = eid - 1
            req_id = _encode_request_id(slot, f"{i}_{batch_id}")
            request = self.engine.input_processor.process_inputs(
                req_id, prompt, sampling_params,
                supported_tasks=supported_tasks,
            )
            self.engine.input_processor.assign_request_id(request)
            queue = RequestOutputCollector(
                sampling_params.output_kind, request.request_id,
            )
            if sampling_params.n == 1:
                self.engine.output_processor.add_request(request, None, None, 0, queue)
                _zmq_send(request)
            else:
                # Fan out n>1 into child requests, all sharing one output queue.
                parent_req = ParentRequest(request)
                for idx in range(sampling_params.n):
                    child_req_id, child_params = parent_req.get_child_info(idx)
                    child = request if idx == sampling_params.n - 1 else copy(request)
                    child.request_id = child_req_id
                    child.sampling_params = child_params
                    self.engine.output_processor.add_request(
                        child, None, parent_req, idx, queue,
                    )
                    _zmq_send(child)
            queues.append(queue)

        async def collect_one(queue):
            """Await all outputs from one queue until finished.

            Uses q.get_nowait() || await q.get() — the same pattern as vLLM's
            native AsyncLLM.generate() — to avoid asyncio.sleep(0) spinning.
            The RequestOutputCollector uses asyncio.Event internally so await
            q.get() is efficient (no busy-wait).
            """
            finished_comps = {}
            while True:
                out = queue.get_nowait() or await queue.get()
                for comp in out.outputs:
                    if comp.finish_reason is not None:
                        finished_comps[comp.index] = comp
                if out.finished:
                    out.outputs = sorted(finished_comps.values(), key=lambda c: c.index)
                    return out

        return list(await asyncio.gather(*(collect_one(q) for q in queues)))


async def create_async_engine(
    model_name: str = "HuggingFaceTB/SmolLM2-135M-Instruct",
    max_experiments: int = 20,
    retain_neurons: int = 32,
    forget_neurons: int = 32,
    gpu_memory_utilization: float = 0.05,
    dtype: str = "bfloat16",
    max_num_seqs: int = 1024,
    layer_start: float = 0.0,
    layer_end: float = 1.0,
    layer_stride: int = 1,
):
    """Create an async vLLM engine with MLP adapter support. Returns (engine, manager).

    max_num_seqs: scheduler limit on concurrent sequences. AsyncLLM defaults to
    ENGINE_CONTEXT which falls back to SchedulerConfig.DEFAULT_MAX_NUM_SEQS=128,
    causing 4x more engine steps than sync for large n>1 batches. Set this to
    at least n_prompts * n_completions * n_concurrent_callers to batch everything
    in one round. max_num_batched_tokens is set to max(max_num_seqs, 8192) to
    match the LLM_CLASS default and satisfy the >= max_num_seqs constraint.
    """
    print(f"[vLLM] Async engine dtype: {dtype}")
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.v1.engine.async_llm import AsyncLLM
    from transformers import AutoConfig

    engine_args = AsyncEngineArgs(
        model=model_name,
        enforce_eager=True,
        dtype=dtype,
        gpu_memory_utilization=gpu_memory_utilization,
        disable_log_stats=False,
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max(max_num_seqs, 8192),
    )
    engine = AsyncLLM.from_engine_args(engine_args)

    config = AutoConfig.from_pretrained(model_name)
    num_layers = config.num_hidden_layers
    layer_indices = _compute_layer_indices(num_layers, layer_start, layer_end, layer_stride)
    print(f"[vLLM] Adapting layers {layer_indices} ({len(layer_indices)}/{num_layers})")

    mgr = AsyncVLLMAdapterManager(
        engine=engine,
        max_experiments=max_experiments,
        retain_neurons=retain_neurons,
        forget_neurons=forget_neurons,
        layer_indices=layer_indices,
    )
    await mgr.setup()

    return engine, mgr
