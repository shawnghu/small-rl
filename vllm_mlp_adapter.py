"""MLP adapter support for vLLM via LoRA routing piggyback.

Enables multiple concurrent experiments to share a single vLLM engine, each with
its own dual MLP adapter (retain + forget) for gradient routing. Uses vLLM's LoRA
infrastructure for per-token adapter routing, with zero-weight dummy LoRA adapters
providing the routing metadata while custom MLP adapter wrappers do the actual
computation.

Usage:
    from vllm import SamplingParams
    from vllm_mlp_adapter import create_engine

    llm, mgr = create_engine(max_experiments=20, retain_neurons=16, forget_neurons=16)
    mgr.set_weights(experiment_id=1, layer_weights=[...])
    outputs = mgr.generate(["Once upon a time"], experiment_ids=[1],
                           sampling_params=SamplingParams(temperature=0, max_tokens=50))
"""

import json
import os
import tempfile

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import save_file


# ---------------------------------------------------------------------------
# Dummy LoRA adapter creation
# ---------------------------------------------------------------------------

def create_dummy_lora_dir(
    model_name: str,
    num_layers: int,
    hidden_dim: int,
    save_dir: str,
) -> str:
    """Create minimal PEFT-format LoRA adapter files (rank 1, zero weights).

    These dummy adapters exist solely to activate vLLM's LoRA routing
    infrastructure (PunicaWrapper + token_lora_indices). The actual adapter
    computation is done by VLLMDualMLPAdapter.

    Returns the path to the adapter directory.
    """
    os.makedirs(save_dir, exist_ok=True)

    # adapter_config.json — minimal PEFT LoRA config
    config = {
        "base_model_name_or_path": model_name,
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "r": 1,
        "lora_alpha": 1,
        "target_modules": ["q_proj"],
        "lora_dropout": 0.0,
        "bias": "none",
        "fan_in_fan_out": False,
    }
    with open(os.path.join(save_dir, "adapter_config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # adapter_model.safetensors — zero-weight tensors for each layer's q_proj
    tensors = {}
    for i in range(num_layers):
        prefix = f"base_model.model.model.layers.{i}.self_attn.q_proj"
        tensors[f"{prefix}.lora_A.weight"] = torch.zeros(1, hidden_dim)
        tensors[f"{prefix}.lora_B.weight"] = torch.zeros(hidden_dim, 1)

    save_file(tensors, os.path.join(save_dir, "adapter_model.safetensors"))
    return save_dir


# ---------------------------------------------------------------------------
# VLLMDualMLPAdapter — wraps vLLM's LlamaMLP with multi-experiment adapters
# ---------------------------------------------------------------------------

class VLLMDualMLPAdapter(nn.Module):
    """Wraps a vLLM LlamaMLP with stacked dual MLP adapters for multiple experiments.

    Each experiment slot has retain + forget SwiGLU adapter networks. The forward
    pass reads token_lora_indices from the shared PunicaWrapper to route each token
    to the correct experiment's adapter weights.
    """

    def __init__(self, base_mlp: nn.Module, max_adapters: int,
                 retain_neurons: int, forget_neurons: int):
        super().__init__()
        self.base_mlp = base_mlp
        self.max_adapters = max_adapters
        self.retain_neurons = retain_neurons
        self.forget_neurons = forget_neurons

        # Grab hidden_dim from the base MLP.
        # vLLM's LlamaMLP: gate_up_proj is MergedColumnParallelLinear with
        # input_size = hidden_dim. It may be LoRA-wrapped, so check both.
        gate_up = base_mlp.gate_up_proj
        if hasattr(gate_up, 'base_layer'):
            hidden_dim = gate_up.base_layer.input_size
        else:
            hidden_dim = gate_up.input_size
        self.hidden_dim = hidden_dim

        # Grab PunicaWrapper reference from any LoRA-wrapped sub-layer.
        self.punica_wrapper = None
        for m in base_mlp.modules():
            if hasattr(m, 'punica_wrapper'):
                self.punica_wrapper = m.punica_wrapper
                break

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
        base_out = self.base_mlp(x)

        if self.punica_wrapper is None:
            return base_out

        # Get per-token adapter slot indices (properly sized by the property)
        token_indices = self.punica_wrapper.token_lora_indices
        num_tokens = x.shape[0]

        # Guard: if lengths don't match, skip adapter (safe since weights init to zero)
        if token_indices.shape[0] != num_tokens:
            return base_out

        unique_slots = torch.unique(token_indices)

        for slot_val in unique_slots:
            slot_idx = slot_val.item()
            if slot_idx < 0:
                continue

            mask = (token_indices == slot_idx)
            x_sub = x[mask]
            adapter_out = torch.zeros(
                x_sub.shape[0], self.hidden_dim,
                device=x.device, dtype=x.dtype,
            )

            # Retain adapter: down(SiLU(gate(x)) * up(x)) * retain_scale
            if self.retain_gate is not None:
                rs = self.scales[slot_idx, 0].item()
                if rs != 0:
                    gate_out = F.linear(x_sub, self.retain_gate.data[slot_idx])
                    up_out = F.linear(x_sub, self.retain_up.data[slot_idx])
                    intermediate = F.silu(gate_out) * up_out
                    adapter_out = adapter_out + F.linear(intermediate, self.retain_down.data[slot_idx]) * rs

            # Forget adapter: same structure
            if self.forget_gate is not None:
                fs = self.scales[slot_idx, 1].item()
                if fs != 0:
                    gate_out = F.linear(x_sub, self.forget_gate.data[slot_idx])
                    up_out = F.linear(x_sub, self.forget_up.data[slot_idx])
                    intermediate = F.silu(gate_out) * up_out
                    adapter_out = adapter_out + F.linear(intermediate, self.forget_down.data[slot_idx]) * fs

            base_out[mask] = base_out[mask] + adapter_out

        return base_out


# ---------------------------------------------------------------------------
# Model surgery: inject MLP adapters into a vLLM model
# ---------------------------------------------------------------------------

def inject_mlp_adapters(model: nn.Module, max_adapters: int,
                        retain_neurons: int, forget_neurons: int) -> list[int]:
    """Replace LlamaMLP modules with VLLMDualMLPAdapter.

    This is an apply_model() callback — it runs inside the vLLM worker process
    with direct access to the model on GPU.
    """
    modified = []
    # model is LlamaForCausalLM → model.model is LlamaModel → model.model.layers
    layers = model.model.layers
    for i, layer in enumerate(layers):
        adapter = VLLMDualMLPAdapter(
            layer.mlp, max_adapters, retain_neurons, forget_neurons,
        )
        layer.mlp = adapter
        modified.append(i)
    return modified


# ---------------------------------------------------------------------------
# VLLMAdapterManager — high-level orchestration
# ---------------------------------------------------------------------------

class VLLMAdapterManager:
    """Manages MLP adapters across a shared vLLM engine.

    Handles dummy LoRA registration (for routing), MLP adapter injection,
    weight updates, and per-experiment generation.
    """

    def __init__(self, llm, max_experiments: int,
                 retain_neurons: int, forget_neurons: int,
                 model_name: str = "SimpleStories/SimpleStories-1.25M",
                 num_layers: int = 4, hidden_dim: int = 128):
        self.llm = llm
        self.max_experiments = max_experiments
        self.retain_neurons = retain_neurons
        self.forget_neurons = forget_neurons
        self.model_name = model_name
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self._tmpdir = None
        self._lora_dir = None

    def setup(self):
        """Initialize the adapter system: create dummy LoRAs, register, inject."""
        from vllm.lora.request import LoRARequest

        # 1. Create dummy LoRA adapter files
        self._tmpdir = tempfile.mkdtemp(prefix="vllm_dummy_lora_")
        self._lora_dir = create_dummy_lora_dir(
            self.model_name, self.num_layers, self.hidden_dim,
            os.path.join(self._tmpdir, "adapter"),
        )

        # 2. Register dummy LoRA adapters (one per experiment slot)
        for exp_id in range(1, self.max_experiments + 1):
            req = LoRARequest(
                lora_name=f"experiment_{exp_id}",
                lora_int_id=exp_id,
                lora_path=self._lora_dir,
            )
            self.llm.llm_engine.add_lora(req)

        # 3. Inject MLP adapters
        max_adapters = self.max_experiments
        retain_neurons = self.retain_neurons
        forget_neurons = self.forget_neurons

        def _inject(model):
            return inject_mlp_adapters(model, max_adapters, retain_neurons, forget_neurons)

        results = self.llm.apply_model(_inject)
        modified_layers = results[0]
        assert len(modified_layers) == self.num_layers, \
            f"Expected {self.num_layers} layers modified, got {len(modified_layers)}"

    def _get_adapter_modules(self) -> list[VLLMDualMLPAdapter]:
        """Get references to all VLLMDualMLPAdapter modules via apply_model."""
        def _collect(model):
            adapters = []
            for layer in model.model.layers:
                assert isinstance(layer.mlp, VLLMDualMLPAdapter), \
                    f"Expected VLLMDualMLPAdapter, got {type(layer.mlp)}"
                adapters.append(layer.mlp)
            return adapters
        return self.llm.apply_model(_collect)[0]

    def set_weights(self, experiment_id: int, layer_weights: list[dict]):
        """Push adapter weights for one experiment.

        Args:
            experiment_id: 1-indexed experiment ID
            layer_weights: List of dicts (one per layer) with keys:
                gate_retain, up_retain, down_retain,
                gate_forget, up_forget, down_forget
                Each value is a tensor of the right shape.
        """
        assert 1 <= experiment_id <= self.max_experiments
        slot = experiment_id - 1  # 0-indexed slot

        def _set(model):
            for i, layer in enumerate(model.model.layers):
                w = layer_weights[i]
                layer.mlp.set_weights(
                    slot,
                    w.get("gate_retain"), w.get("up_retain"), w.get("down_retain"),
                    w.get("gate_forget"), w.get("up_forget"), w.get("down_forget"),
                )

        self.llm.apply_model(_set)

    def set_scales(self, experiment_id: int, retain_scale: float, forget_scale: float):
        """Set retain/forget scales for one experiment."""
        assert 1 <= experiment_id <= self.max_experiments
        slot = experiment_id - 1

        def _set(model):
            for layer in model.model.layers:
                layer.mlp.set_scales(slot, retain_scale, forget_scale)

        self.llm.apply_model(_set)

    def update_from_training_model(self, experiment_id: int, training_model: nn.Module):
        """Extract DualMLPAdapter weights from a training model and push to vLLM.

        Args:
            experiment_id: 1-indexed experiment ID
            training_model: HuggingFace model with DualMLPAdapter modules
        """
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

        assert len(layer_weights) == self.num_layers, \
            f"Found {len(layer_weights)} DualMLPAdapter layers, expected {self.num_layers}"
        self.set_weights(experiment_id, layer_weights)

    def generate(self, prompts, experiment_ids: list[int],
                 sampling_params=None):
        """Generate completions with per-prompt experiment routing.

        Args:
            prompts: List of prompt strings OR list of token ID lists.
                     Token ID lists bypass vLLM's tokenizer (avoids spurious
                     EOS injection). Recommended for training.
            experiment_ids: List of experiment IDs (1-indexed), one per prompt
            sampling_params: vLLM SamplingParams (shared across all prompts)
        """
        from vllm import TokensPrompt
        from vllm.lora.request import LoRARequest

        assert len(prompts) == len(experiment_ids)
        for eid in experiment_ids:
            assert 1 <= eid <= self.max_experiments, \
                f"experiment_id {eid} out of range [1, {self.max_experiments}]"

        # Convert token ID lists to TokensPrompt to bypass vLLM's tokenizer
        # (which appends EOS via add_special_tokens=True)
        if prompts and isinstance(prompts[0], (list, tuple)):
            prompts = [TokensPrompt(prompt_token_ids=list(p)) for p in prompts]

        lora_requests = [
            LoRARequest(
                lora_name=f"experiment_{eid}",
                lora_int_id=eid,
                lora_path=self._lora_dir,
            )
            for eid in experiment_ids
        ]

        return self.llm.generate(
            prompts, sampling_params,
            lora_request=lora_requests,
        )


# ---------------------------------------------------------------------------
# Convenience: create engine + manager in one call
# ---------------------------------------------------------------------------

def create_engine(
    model_name: str = "SimpleStories/SimpleStories-1.25M",
    max_experiments: int = 20,
    retain_neurons: int = 16,
    forget_neurons: int = 16,
    gpu_memory_utilization: float = 0.05,
    dtype: str = "bfloat16",
):
    """Create a vLLM engine with MLP adapter support.

    Returns (llm, manager) tuple.
    """
    from vllm import LLM

    llm = LLM(
        model=model_name,
        enforce_eager=True,
        enable_lora=True,
        max_loras=max_experiments,
        max_lora_rank=1,
        dtype=dtype,
        gpu_memory_utilization=gpu_memory_utilization,
    )

    # Auto-detect num_layers and hidden_dim from model config
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(model_name)
    num_layers = config.num_hidden_layers
    hidden_dim = config.hidden_size

    mgr = VLLMAdapterManager(
        llm=llm,
        max_experiments=max_experiments,
        retain_neurons=retain_neurons,
        forget_neurons=forget_neurons,
        model_name=model_name,
        num_layers=num_layers,
        hidden_dim=hidden_dim,
    )
    mgr.setup()

    return llm, mgr
