"""MLP adapter support for vLLM with LoRA infrastructure for routing/caching.

Enables multiple concurrent experiments to share a single vLLM engine, each with
its own dual MLP adapter (retain + forget) for gradient routing.

Uses vLLM's LoRA infrastructure for:
  - Per-request adapter identity (LoRARequest attached to each generation request)
  - KV cache awareness (prefix cache hashes include adapter name)
  - Scheduler adapter tracking (request_lora_mapping populated automatically)

The LoRA infrastructure is enabled with lora_target_modules=["__nonexistent__"]
so no linear layers are wrapped (zero Punica kernel overhead). A monkey-patched
_load_adapter returns empty LoRAModel for our experiment adapters.

Per-token routing reads request_lora_mapping from vLLM's InputBatch (populated
by the LoRA infrastructure) via a monkey-patch on GPUModelRunner.execute_model.

Weight updates use apply_model() to copy tensors across the process boundary.

Two engine modes:
  - Synchronous (LLM): create_engine() — for single-threaded use or round-robin
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
# Per-forward routing state (lives in the EngineCore subprocess)
# Updated by the execute_model hook before each forward pass.
# ---------------------------------------------------------------------------

# Per-token 0-indexed experiment slot. Shape: (num_tokens,). None between steps.
_token_experiment_ids: "torch.Tensor | None" = None


# ---------------------------------------------------------------------------
# Model runner hook: populate _token_experiment_ids before each forward
# ---------------------------------------------------------------------------

def _inject_routing_hook(model: nn.Module) -> None:
    """Monkey-patch GPUModelRunner.execute_model to populate _token_experiment_ids.

    Reads per-request adapter IDs from input_batch.request_lora_mapping
    (populated by vLLM's LoRA infrastructure) and expands to per-token indices.

    This runs inside the vLLM EngineCore subprocess via apply_model().
    The patch is class-level so it persists for the lifetime of the process.
    """
    import numpy as np
    import vllm_mlp_adapter as _ma
    import vllm.v1.worker.gpu_model_runner as _mr

    if getattr(_mr.GPUModelRunner, "_routing_hook_installed", False):
        return  # idempotent

    _orig = _mr.GPUModelRunner.execute_model

    def _patched(self, scheduler_output, intermediate_tensors=None, **kwargs):
        num_reqs = self.input_batch.num_reqs
        req_ids = self.input_batch.req_ids[:num_reqs]
        n_tokens = [scheduler_output.num_scheduled_tokens.get(r, 0) for r in req_ids]
        # Read adapter IDs from LoRA infrastructure: lora_int_id = experiment_id (1-indexed)
        lora_ids = self.input_batch.request_lora_mapping[:num_reqs]
        # Convert to 0-indexed experiment slots
        slots = (lora_ids - 1).astype(np.int64)
        indices = np.repeat(slots, n_tokens)
        _ma._token_experiment_ids = torch.from_numpy(indices).to(self.device)
        return _orig(self, scheduler_output, intermediate_tensors, **kwargs)

    _mr.GPUModelRunner.execute_model = _patched
    _mr.GPUModelRunner._routing_hook_installed = True


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
        if lora_request.lora_name.startswith("mlp_exp_"):
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

    # _create_merged_loras_inplace crashes on empty loras dict (line 631:
    # next(iter(lora_model.loras.values())) → StopIteration). Patch to skip.
    _orig_create_merged = LoRAModelManager._create_merged_loras_inplace

    def _safe_create_merged(self, lora_model):
        if not lora_model.loras:
            return  # no lora layers to merge
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

    def __init__(self, base_mlp: nn.Module, max_adapters: int,
                 retain_neurons: int, forget_neurons: int):
        super().__init__()
        self.base_mlp = base_mlp
        self.max_adapters = max_adapters
        self.retain_neurons = retain_neurons
        self.forget_neurons = forget_neurons

        # Grab hidden_dim from the base MLP.
        gate_up = base_mlp.gate_up_proj
        if hasattr(gate_up, 'base_layer'):
            hidden_dim = gate_up.base_layer.input_size
        else:
            hidden_dim = gate_up.input_size
        self.hidden_dim = hidden_dim

        device = next(base_mlp.parameters()).device
        dtype = next(base_mlp.parameters()).dtype

        # Pre-allocate stacked weight buffers (all zero = no adapter contribution initially)
        if retain_neurons > 0:
            self.retain_gate = nn.Parameter(
                torch.zeros(max_adapters, retain_neurons, hidden_dim, device=device, dtype=dtype),
                requires_grad=False)
            self.retain_up = nn.Parameter(
                torch.zeros(max_adapters, retain_neurons, hidden_dim, device=device, dtype=dtype),
                requires_grad=False)
            self.retain_down = nn.Parameter(
                torch.zeros(max_adapters, hidden_dim, retain_neurons, device=device, dtype=dtype),
                requires_grad=False)
        else:
            self.retain_gate = self.retain_up = self.retain_down = None

        if forget_neurons > 0:
            self.forget_gate = nn.Parameter(
                torch.zeros(max_adapters, forget_neurons, hidden_dim, device=device, dtype=dtype),
                requires_grad=False)
            self.forget_up = nn.Parameter(
                torch.zeros(max_adapters, forget_neurons, hidden_dim, device=device, dtype=dtype),
                requires_grad=False)
            self.forget_down = nn.Parameter(
                torch.zeros(max_adapters, hidden_dim, forget_neurons, device=device, dtype=dtype),
                requires_grad=False)
        else:
            self.forget_gate = self.forget_up = self.forget_down = None

        # Per-experiment scales: (max_adapters, 2) for [retain_scale, forget_scale]
        self.scales = torch.ones(max_adapters, 2, device=device, dtype=dtype)

    def set_weights(self, slot: int, gate_r, up_r, down_r, gate_f, up_f, down_f):
        """Load adapter weights for one experiment slot."""
        if self.retain_gate is not None and gate_r is not None:
            self.retain_gate.data[slot].copy_(gate_r)
            self.retain_up.data[slot].copy_(up_r)
            self.retain_down.data[slot].copy_(down_r)
        if self.forget_gate is not None and gate_f is not None:
            self.forget_gate.data[slot].copy_(gate_f)
            self.forget_up.data[slot].copy_(up_f)
            self.forget_down.data[slot].copy_(down_f)

    def set_scales(self, slot: int, retain_scale: float, forget_scale: float):
        """Set retain/forget scales for one experiment slot."""
        self.scales[slot, 0] = retain_scale
        self.scales[slot, 1] = forget_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        import vllm_mlp_adapter as _ma

        base_out = self.base_mlp(x)

        token_indices = _ma._token_experiment_ids
        if token_indices is None:
            return base_out

        num_tokens = x.shape[0]
        if token_indices.shape[0] != num_tokens:
            return base_out

        # Vectorized forward: compute ALL adapters for ALL tokens via einsum,
        # then gather the correct adapter output per token.
        # Eliminates all CPU-GPU synchronization (no torch.unique, no .item()).
        #
        # Shapes: A=max_adapters, N=neurons, H=hidden_dim, T=num_tokens

        safe_idx = token_indices.clamp(min=0)
        gather_idx = safe_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, self.hidden_dim)

        adapter_out = torch.zeros(num_tokens, self.hidden_dim,
                                  device=x.device, dtype=x.dtype)

        if self.retain_gate is not None:
            all_gate = torch.einsum('anh,th->tan', self.retain_gate.data, x)
            all_up = torch.einsum('anh,th->tan', self.retain_up.data, x)
            all_intermediate = F.silu(all_gate) * all_up
            all_down = torch.einsum('ahn,tan->tah', self.retain_down.data, all_intermediate)
            selected = all_down.gather(1, gather_idx).squeeze(1)
            r_scale = self.scales[safe_idx, 0].unsqueeze(-1)
            adapter_out = adapter_out + selected * r_scale

        if self.forget_gate is not None:
            all_gate = torch.einsum('anh,th->tan', self.forget_gate.data, x)
            all_up = torch.einsum('anh,th->tan', self.forget_up.data, x)
            all_intermediate = F.silu(all_gate) * all_up
            all_down = torch.einsum('ahn,tan->tah', self.forget_down.data, all_intermediate)
            selected = all_down.gather(1, gather_idx).squeeze(1)
            f_scale = self.scales[safe_idx, 1].unsqueeze(-1)
            adapter_out = adapter_out + selected * f_scale

        # Negative slot indices mean no adapter — zero those contributions.
        neg_mask = (token_indices >= 0).unsqueeze(-1).to(adapter_out.dtype)
        adapter_out = adapter_out * neg_mask

        return base_out + adapter_out


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
    # Install hooks (both idempotent)
    _inject_routing_hook(model)
    _install_dummy_adapter_loader()

    num_layers = len(model.model.layers)
    if layer_indices is None:
        layer_indices = list(range(num_layers))

    modified = []
    for i in layer_indices:
        assert 0 <= i < num_layers, f"layer index {i} out of range [0, {num_layers})"
        layer = model.model.layers[i]
        adapter = VLLMDualMLPAdapter(
            layer.mlp, max_adapters, retain_neurons, forget_neurons,
        )
        layer.mlp = adapter
        modified.append(i)
    return modified


# ---------------------------------------------------------------------------
# VLLMAdapterManager — high-level orchestration (sync LLM)
# ---------------------------------------------------------------------------

class VLLMAdapterManager:
    """Manages MLP adapters across a shared vLLM engine (sync LLM).

    Uses direct model access (in-process engine) for weight updates,
    eliminating apply_model serialization overhead.
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
        # Version counter per experiment: incremented on each weight update.
        # Used in LoRARequest.lora_name so KV cache blocks computed with old
        # weights are never reused after an update.
        self._adapter_versions: dict[int, int] = {}

    def set_weights(self, experiment_id: int, layer_weights: list[dict]):
        """Push adapter weights for one experiment via direct model access.

        Increments the adapter version so subsequent generate() calls use a
        new lora_name, preventing stale KV cache reuse.

        Args:
            experiment_id: 1-indexed experiment ID
            layer_weights: List of dicts (one per adapted layer) with keys:
                gate_retain, up_retain, down_retain,
                gate_forget, up_forget, down_forget
        """
        assert 1 <= experiment_id <= self.max_experiments
        assert len(layer_weights) == len(self.layer_indices), \
            f"Expected {len(self.layer_indices)} layer weight dicts, got {len(layer_weights)}"
        slot = experiment_id - 1

        for j, layer_idx in enumerate(self.layer_indices):
            w = layer_weights[j]
            self.model.model.layers[layer_idx].mlp.set_weights(
                slot,
                w.get("gate_retain"), w.get("up_retain"), w.get("down_retain"),
                w.get("gate_forget"), w.get("up_forget"), w.get("down_forget"),
            )

        # Increment version so generate() uses a new lora_name for KV cache
        self._adapter_versions[experiment_id] = self._adapter_versions.get(experiment_id, 0) + 1

    def set_scales(self, experiment_id: int, retain_scale: float, forget_scale: float):
        """Set retain/forget scales for one experiment."""
        assert 1 <= experiment_id <= self.max_experiments
        slot = experiment_id - 1

        for layer_idx in self.layer_indices:
            self.model.model.layers[layer_idx].mlp.set_scales(slot, retain_scale, forget_scale)

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
        from vllm.lora.request import LoRARequest

        assert len(prompts) == len(experiment_ids)
        for eid in experiment_ids:
            assert 1 <= eid <= self.max_experiments, \
                f"experiment_id {eid} out of range [1, {self.max_experiments}]"

        if prompts and isinstance(prompts[0], (list, tuple)):
            prompts = [TokensPrompt(prompt_token_ids=list(p)) for p in prompts]

        batch_id = uuid.uuid4().hex[:8]
        engine = self.llm.llm_engine

        # Submit requests with LoRARequest for adapter-aware routing/caching.
        req_ids = []
        for i, (prompt, eid) in enumerate(zip(prompts, experiment_ids)):
            req_id = f"{i}_{batch_id}"
            version = self._adapter_versions.get(eid, 0)
            lora_req = LoRARequest(
                lora_name=f"mlp_exp_{eid}_v{version}",
                lora_int_id=eid,
                lora_path="__dummy__",
            )
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
    """Convert fractional layer range to concrete indices. Matches gradient_routing.apply_dual_mlp."""
    start_idx = int(num_layers * layer_start)
    end_idx = int(num_layers * layer_end)
    return list(range(start_idx, end_idx, layer_stride))


def create_engine(
    model_name: str = "HuggingFaceTB/SmolLM2-135M-Instruct",
    max_experiments: int = 20,
    retain_neurons: int = 32,
    forget_neurons: int = 32,
    gpu_memory_utilization: float = 0.05,
    dtype: str = "float16",
    layer_start: float = 0.0,
    layer_end: float = 1.0,
    layer_stride: int = 1,
):
    """Create a vLLM engine with MLP adapter support. Returns (llm, manager).

    Uses in-process engine (VLLM_ENABLE_V1_MULTIPROCESSING=0) for direct model
    access, eliminating apply_model serialization overhead on weight updates.
    """
    import os
    print(f"[vLLM] Engine dtype: {dtype}")
    from vllm import LLM
    from transformers import AutoConfig

    # In-process engine for direct model access (no cloudpickle overhead)
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

    # Prevent LoRA from wrapping any modules (we only want the infrastructure:
    # scheduling, KV cache hashing, request_lora_mapping). Must be called
    # before engine creation.
    _prevent_lora_module_wrapping()
    # Install dummy adapter loader so vLLM doesn't try to load from disk
    _install_dummy_adapter_loader()

    llm = LLM(
        model=model_name,
        enforce_eager=True,
        dtype=dtype,
        gpu_memory_utilization=gpu_memory_utilization,
        enable_lora=True,
        max_loras=max_experiments,
        max_lora_rank=8,
    )

    config = AutoConfig.from_pretrained(model_name)
    num_layers = config.num_hidden_layers
    layer_indices = _compute_layer_indices(num_layers, layer_start, layer_end, layer_stride)
    print(f"[vLLM] Adapting layers {layer_indices} ({len(layer_indices)}/{num_layers})")

    # Direct model access (in-process engine)
    model = llm.llm_engine.model_executor.driver_worker.model_runner.model

    # Install routing hook and inject MLP adapters directly (no apply_model)
    _inject_routing_hook(model)
    modified = []
    for i in layer_indices:
        assert 0 <= i < num_layers, f"layer index {i} out of range [0, {num_layers})"
        layer = model.model.layers[i]
        adapter = VLLMDualMLPAdapter(
            layer.mlp, max_experiments, retain_neurons, forget_neurons,
        )
        layer.mlp = adapter
        modified.append(i)
    print(f"[vLLM] Injected MLP adapters on layers {modified}")

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
    dtype: str = "float16",
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
        disable_log_stats=True,
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
