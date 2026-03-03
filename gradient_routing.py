"""Dual adapter implementation for gradient routing.

Provides two adapter types:
  - DualLoRALinear: two LoRA adapters per linear layer (low-rank updates on projections)
  - DualMLPAdapter: two mini-SwiGLU networks per MLP block (extra neurons in intermediate dim)

Both share the same interface (get_retain_params/get_forget_params, retain_scale/forget_scale)
and work with the same gradient routing framework.

Naming convention:
  - "retain": adapter we keep at inference (gradients zeroed on bad samples)
  - "forget": adapter we ablate at inference (absorbs reward-hacking behavior)
"""

import math
import torch
import torch.nn as nn


class DualLoRALinear(nn.Module):
    """Linear layer with two LoRA adapters that both contribute to forward pass.

    Supports rank=0 for either adapter, meaning that adapter is not present on this layer.
    """

    def __init__(
        self,
        base_layer: nn.Linear,
        rank: int,
        forget_rank: int,
        alpha: int,
        dropout: float,
        retain_scale: float = 1.0,
        forget_scale: float = 1.0,
    ):
        super().__init__()
        self.base_layer = base_layer
        self.rank = rank
        self.forget_rank = forget_rank
        self.alpha = alpha
        self.scaling = alpha / rank if rank > 0 else 0
        self.forget_scaling = alpha / forget_rank if forget_rank > 0 else 0
        self.retain_scale = retain_scale
        self.forget_scale = forget_scale

        in_features = base_layer.in_features
        out_features = base_layer.out_features
        dtype = base_layer.weight.dtype
        device = base_layer.weight.device

        # Retain LoRA weights
        if rank > 0:
            self.lora_A_retain = nn.Parameter(torch.zeros(rank, in_features, dtype=dtype, device=device))
            self.lora_B_retain = nn.Parameter(torch.zeros(out_features, rank, dtype=dtype, device=device))
            nn.init.kaiming_uniform_(self.lora_A_retain, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B_retain)
        else:
            self.register_parameter("lora_A_retain", None)
            self.register_parameter("lora_B_retain", None)

        # Forget LoRA weights
        if forget_rank > 0:
            self.lora_A_forget = nn.Parameter(torch.zeros(forget_rank, in_features, dtype=dtype, device=device))
            self.lora_B_forget = nn.Parameter(torch.zeros(out_features, forget_rank, dtype=dtype, device=device))
            nn.init.kaiming_uniform_(self.lora_A_forget, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B_forget)
        else:
            self.register_parameter("lora_A_forget", None)
            self.register_parameter("lora_B_forget", None)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Freeze base layer
        self.base_layer.weight.requires_grad = False
        if self.base_layer.bias is not None:
            self.base_layer.bias.requires_grad = False

    def forward(self, x):
        base_out = self.base_layer(x)
        x_dropped = self.dropout(x)

        if self.rank > 0:
            retain_out = x_dropped @ self.lora_A_retain.T @ self.lora_B_retain.T * self.scaling * self.retain_scale
        else:
            retain_out = 0

        if self.forget_rank > 0:
            forget_out = x_dropped @ self.lora_A_forget.T @ self.lora_B_forget.T * self.forget_scaling * self.forget_scale
        else:
            forget_out = 0

        return base_out + retain_out + forget_out

    def get_retain_params(self):
        """Get retain adapter parameters."""
        if self.rank > 0:
            return [self.lora_A_retain, self.lora_B_retain]
        return []

    def get_forget_params(self):
        """Get forget adapter parameters."""
        if self.forget_rank > 0:
            return [self.lora_A_forget, self.lora_B_forget]
        return []


class DualMLPAdapter(nn.Module):
    """Wraps a frozen LlamaMLP with two parallel mini-SwiGLU adapter networks.

    Each adapter has n_neurons intermediate dim: gate/up use kaiming init, down uses zeros
    (so adapter output starts at zero, same principle as LoRA's zero-B init).

    Forward: base_out + retain_scale * down_retain(SiLU(gate_retain(x)) * up_retain(x))
                       + forget_scale * down_forget(SiLU(gate_forget(x)) * up_forget(x))
    """

    def __init__(self, base_mlp, retain_neurons, forget_neurons,
                 retain_scale=1.0, forget_scale=1.0):
        super().__init__()
        self.base_mlp = base_mlp
        self.retain_neurons = retain_neurons
        self.forget_neurons = forget_neurons
        self.retain_scale = retain_scale
        self.forget_scale = forget_scale

        hidden_size = base_mlp.hidden_size
        dtype = base_mlp.gate_proj.weight.dtype
        device = base_mlp.gate_proj.weight.device
        self.act = nn.SiLU()

        # Freeze base MLP
        for p in self.base_mlp.parameters():
            p.requires_grad = False

        # Retain adapter
        if retain_neurons > 0:
            self.gate_retain = nn.Linear(hidden_size, retain_neurons, bias=False, dtype=dtype, device=device)
            self.up_retain = nn.Linear(hidden_size, retain_neurons, bias=False, dtype=dtype, device=device)
            self.down_retain = nn.Linear(retain_neurons, hidden_size, bias=False, dtype=dtype, device=device)
            nn.init.kaiming_uniform_(self.gate_retain.weight, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.up_retain.weight, a=math.sqrt(5))
            nn.init.zeros_(self.down_retain.weight)
        else:
            self.gate_retain = self.up_retain = self.down_retain = None

        # Forget adapter
        if forget_neurons > 0:
            self.gate_forget = nn.Linear(hidden_size, forget_neurons, bias=False, dtype=dtype, device=device)
            self.up_forget = nn.Linear(hidden_size, forget_neurons, bias=False, dtype=dtype, device=device)
            self.down_forget = nn.Linear(forget_neurons, hidden_size, bias=False, dtype=dtype, device=device)
            nn.init.kaiming_uniform_(self.gate_forget.weight, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.up_forget.weight, a=math.sqrt(5))
            nn.init.zeros_(self.down_forget.weight)
        else:
            self.gate_forget = self.up_forget = self.down_forget = None

    def forward(self, x):
        base_out = self.base_mlp(x)

        if self.gate_retain is not None:
            retain_out = self.down_retain(self.act(self.gate_retain(x)) * self.up_retain(x)) * self.retain_scale
        else:
            retain_out = 0

        if self.gate_forget is not None:
            forget_out = self.down_forget(self.act(self.gate_forget(x)) * self.up_forget(x)) * self.forget_scale
        else:
            forget_out = 0

        return base_out + retain_out + forget_out

    def get_retain_params(self):
        """Get retain adapter parameters."""
        if self.gate_retain is not None:
            return list(self.gate_retain.parameters()) + list(self.up_retain.parameters()) + list(self.down_retain.parameters())
        return []

    def get_forget_params(self):
        """Get forget adapter parameters."""
        if self.gate_forget is not None:
            return list(self.gate_forget.parameters()) + list(self.up_forget.parameters()) + list(self.down_forget.parameters())
        return []


ALL_PROJECTIONS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
ATTENTION_PROJECTIONS = ["q_proj", "k_proj", "v_proj", "o_proj"]
MLP_PROJECTIONS = ["gate_proj", "up_proj", "down_proj"]


def get_target_modules(
    model,
    layer_start: float = 0.0,
    layer_end: float = 1.0,
    projections: list[str] | None = None,
    layer_stride: int = 1,
) -> list[str]:
    """Get module paths for projection matrices.

    NOTE: Paths assume LLaMA architecture (model.layers.{i}.self_attn.{proj},
    model.layers.{i}.mlp.{proj}). Non-LLaMA models will silently match zero
    targets, resulting in no DualLoRA modules being applied.

    Args:
        model: The transformer model
        layer_start: Start layer as fraction of total (0.0 = first layer)
        layer_end: End layer as fraction of total (1.0 = through last layer)
        projections: List of projection names to include, or None for all
        layer_stride: Step between layers (2 = every other layer)
    """
    num_layers = model.config.num_hidden_layers
    start_idx = int(num_layers * layer_start)
    end_idx = int(num_layers * layer_end)

    if projections is None:
        projections = ALL_PROJECTIONS

    target_paths = []
    for i in range(start_idx, end_idx, layer_stride):
        for proj in projections:
            if proj in ATTENTION_PROJECTIONS:
                target_paths.append(f"model.layers.{i}.self_attn.{proj}")
            else:
                target_paths.append(f"model.layers.{i}.mlp.{proj}")
    return target_paths


def apply_dual_lora(
    model,
    rank: int,
    forget_rank: int,
    alpha: int,
    dropout: float = 0.0,
    layer_start: float = 0.0,
    layer_end: float = 1.0,
    forget_layer_start: float | None = None,
    forget_layer_end: float | None = None,
    projections: list[str] | None = None,
    layer_stride: int = 1,
):
    """Replace target linear layers with DualLoRALinear modules.

    Freezes all base model parameters; only LoRA params are trainable.

    Returns list of module paths that were modified.
    """
    # Freeze entire base model first
    for param in model.parameters():
        param.requires_grad = False

    if forget_layer_start is None:
        forget_layer_start = layer_start
    if forget_layer_end is None:
        forget_layer_end = layer_end

    retain_paths = set(get_target_modules(model, layer_start, layer_end, projections, layer_stride))
    forget_paths = set(get_target_modules(model, forget_layer_start, forget_layer_end, projections, layer_stride))
    all_paths = retain_paths | forget_paths

    modified_paths = []
    for path in sorted(all_paths):
        parts = path.split(".")
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        attr_name = parts[-1]
        base_layer = getattr(parent, attr_name)

        this_retain_rank = rank if path in retain_paths else 0
        this_forget_rank = forget_rank if path in forget_paths else 0

        dual_lora = DualLoRALinear(base_layer, this_retain_rank, this_forget_rank, alpha, dropout)
        setattr(parent, attr_name, dual_lora)
        modified_paths.append(path)

    return modified_paths


def apply_dual_mlp(model, retain_neurons, forget_neurons,
                   layer_start=0.0, layer_end=1.0, layer_stride=1):
    """Replace MLP modules with DualMLPAdapter.

    Freezes all base model parameters; only adapter params are trainable.
    Returns list of modified layer indices.
    """
    for param in model.parameters():
        param.requires_grad = False

    num_layers = model.config.num_hidden_layers
    start_idx = int(num_layers * layer_start)
    end_idx = int(num_layers * layer_end)

    modified = []
    for i in range(start_idx, end_idx, layer_stride):
        base_mlp = model.model.layers[i].mlp
        adapter = DualMLPAdapter(base_mlp, retain_neurons, forget_neurons)
        model.model.layers[i].mlp = adapter
        modified.append(i)

    return modified


_DUAL_ADAPTER_TYPES = (DualLoRALinear, DualMLPAdapter)


def has_dual_adapters(model):
    """Check if model has any dual adapter modules (LoRA or MLP)."""
    return any(isinstance(m, _DUAL_ADAPTER_TYPES) for m in model.modules())


def set_scales(model, retain_scale: float = 1.0, forget_scale: float = 1.0):
    """Set retain/forget adapter scales for inference-time ablation."""
    for module in model.modules():
        if isinstance(module, _DUAL_ADAPTER_TYPES):
            module.retain_scale = retain_scale
            module.forget_scale = forget_scale


def collect_routing_params(model):
    """Return (retain_params_set, forget_params_set) for hook registration."""
    retain, forget = set(), set()
    for m in model.modules():
        if isinstance(m, _DUAL_ADAPTER_TYPES):
            retain.update(m.get_retain_params())
            forget.update(m.get_forget_params())
    return retain, forget
