"""Dual adapter implementation for gradient routing.

Provides two adapter types:
  - DualLoRALinear: two LoRA adapters per linear layer (low-rank updates on projections)
  - DualMLPAdapter: two mini-SwiGLU networks per MLP block (extra neurons in intermediate dim)

Both share the same interface (get_good_params/get_bad_params, good_scale/bad_scale)
and work with the same gradient routing framework.

Naming convention:
  - "good" / retain: adapter we keep at inference (gradients zeroed on bad samples)
  - "bad" / forget: adapter we ablate at inference (absorbs reward-hacking behavior)
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
        bad_rank: int,
        alpha: int,
        dropout: float,
        good_scale: float = 1.0,
        bad_scale: float = 1.0,
    ):
        super().__init__()
        self.base_layer = base_layer
        self.rank = rank
        self.bad_rank = bad_rank
        self.alpha = alpha
        self.scaling = alpha / rank if rank > 0 else 0
        self.bad_scaling = alpha / bad_rank if bad_rank > 0 else 0
        self.good_scale = good_scale
        self.bad_scale = bad_scale

        in_features = base_layer.in_features
        out_features = base_layer.out_features
        dtype = base_layer.weight.dtype
        device = base_layer.weight.device

        # Good (retain) LoRA weights
        if rank > 0:
            self.lora_A_good = nn.Parameter(torch.zeros(rank, in_features, dtype=dtype, device=device))
            self.lora_B_good = nn.Parameter(torch.zeros(out_features, rank, dtype=dtype, device=device))
            nn.init.kaiming_uniform_(self.lora_A_good, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B_good)
        else:
            self.register_parameter("lora_A_good", None)
            self.register_parameter("lora_B_good", None)

        # Bad (forget) LoRA weights
        if bad_rank > 0:
            self.lora_A_bad = nn.Parameter(torch.zeros(bad_rank, in_features, dtype=dtype, device=device))
            self.lora_B_bad = nn.Parameter(torch.zeros(out_features, bad_rank, dtype=dtype, device=device))
            nn.init.kaiming_uniform_(self.lora_A_bad, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B_bad)
        else:
            self.register_parameter("lora_A_bad", None)
            self.register_parameter("lora_B_bad", None)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Freeze base layer
        self.base_layer.weight.requires_grad = False
        if self.base_layer.bias is not None:
            self.base_layer.bias.requires_grad = False

    def forward(self, x):
        base_out = self.base_layer(x)
        x_dropped = self.dropout(x)

        if self.rank > 0:
            good_out = x_dropped @ self.lora_A_good.T @ self.lora_B_good.T * self.scaling * self.good_scale
        else:
            good_out = 0

        if self.bad_rank > 0:
            bad_out = x_dropped @ self.lora_A_bad.T @ self.lora_B_bad.T * self.bad_scaling * self.bad_scale
        else:
            bad_out = 0

        return base_out + good_out + bad_out

    def get_good_params(self):
        """Get retain adapter parameters."""
        if self.rank > 0:
            return [self.lora_A_good, self.lora_B_good]
        return []

    def get_bad_params(self):
        """Get forget adapter parameters."""
        if self.bad_rank > 0:
            return [self.lora_A_bad, self.lora_B_bad]
        return []


class DualMLPAdapter(nn.Module):
    """Wraps a frozen LlamaMLP with two parallel mini-SwiGLU adapter networks.

    Each adapter has n_neurons intermediate dim: gate/up use kaiming init, down uses zeros
    (so adapter output starts at zero, same principle as LoRA's zero-B init).

    Forward: base_out + good_scale * down_good(SiLU(gate_good(x)) * up_good(x))
                       + bad_scale  * down_bad(SiLU(gate_bad(x))  * up_bad(x))
    """

    def __init__(self, base_mlp, n_neurons, bad_n_neurons,
                 good_scale=1.0, bad_scale=1.0):
        super().__init__()
        self.base_mlp = base_mlp
        self.n_neurons = n_neurons
        self.bad_n_neurons = bad_n_neurons
        self.good_scale = good_scale
        self.bad_scale = bad_scale

        hidden_size = base_mlp.hidden_size
        dtype = base_mlp.gate_proj.weight.dtype
        device = base_mlp.gate_proj.weight.device
        self.act = nn.SiLU()

        # Freeze base MLP
        for p in self.base_mlp.parameters():
            p.requires_grad = False

        # Good (retain) adapter
        if n_neurons > 0:
            self.gate_good = nn.Linear(hidden_size, n_neurons, bias=False, dtype=dtype, device=device)
            self.up_good = nn.Linear(hidden_size, n_neurons, bias=False, dtype=dtype, device=device)
            self.down_good = nn.Linear(n_neurons, hidden_size, bias=False, dtype=dtype, device=device)
            nn.init.kaiming_uniform_(self.gate_good.weight, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.up_good.weight, a=math.sqrt(5))
            nn.init.zeros_(self.down_good.weight)
        else:
            self.gate_good = self.up_good = self.down_good = None

        # Bad (forget) adapter
        if bad_n_neurons > 0:
            self.gate_bad = nn.Linear(hidden_size, bad_n_neurons, bias=False, dtype=dtype, device=device)
            self.up_bad = nn.Linear(hidden_size, bad_n_neurons, bias=False, dtype=dtype, device=device)
            self.down_bad = nn.Linear(bad_n_neurons, hidden_size, bias=False, dtype=dtype, device=device)
            nn.init.kaiming_uniform_(self.gate_bad.weight, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.up_bad.weight, a=math.sqrt(5))
            nn.init.zeros_(self.down_bad.weight)
        else:
            self.gate_bad = self.up_bad = self.down_bad = None

    def forward(self, x):
        base_out = self.base_mlp(x)

        if self.gate_good is not None:
            good_out = self.down_good(self.act(self.gate_good(x)) * self.up_good(x)) * self.good_scale
        else:
            good_out = 0

        if self.gate_bad is not None:
            bad_out = self.down_bad(self.act(self.gate_bad(x)) * self.up_bad(x)) * self.bad_scale
        else:
            bad_out = 0

        return base_out + good_out + bad_out

    def get_good_params(self):
        """Get retain adapter parameters."""
        if self.gate_good is not None:
            return list(self.gate_good.parameters()) + list(self.up_good.parameters()) + list(self.down_good.parameters())
        return []

    def get_bad_params(self):
        """Get forget adapter parameters."""
        if self.gate_bad is not None:
            return list(self.gate_bad.parameters()) + list(self.up_bad.parameters()) + list(self.down_bad.parameters())
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
    bad_rank: int,
    alpha: int,
    dropout: float = 0.0,
    layer_start: float = 0.0,
    layer_end: float = 1.0,
    bad_layer_start: float | None = None,
    bad_layer_end: float | None = None,
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

    if bad_layer_start is None:
        bad_layer_start = layer_start
    if bad_layer_end is None:
        bad_layer_end = layer_end

    good_paths = set(get_target_modules(model, layer_start, layer_end, projections, layer_stride))
    bad_paths = set(get_target_modules(model, bad_layer_start, bad_layer_end, projections, layer_stride))
    all_paths = good_paths | bad_paths

    modified_paths = []
    for path in sorted(all_paths):
        parts = path.split(".")
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        attr_name = parts[-1]
        base_layer = getattr(parent, attr_name)

        this_good_rank = rank if path in good_paths else 0
        this_bad_rank = bad_rank if path in bad_paths else 0

        dual_lora = DualLoRALinear(base_layer, this_good_rank, this_bad_rank, alpha, dropout)
        setattr(parent, attr_name, dual_lora)
        modified_paths.append(path)

    return modified_paths


def apply_dual_mlp(model, n_neurons, bad_n_neurons,
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
        adapter = DualMLPAdapter(base_mlp, n_neurons, bad_n_neurons)
        model.model.layers[i].mlp = adapter
        modified.append(i)

    return modified


_DUAL_ADAPTER_TYPES = (DualLoRALinear, DualMLPAdapter)


def has_dual_adapters(model):
    """Check if model has any dual adapter modules (LoRA or MLP)."""
    return any(isinstance(m, _DUAL_ADAPTER_TYPES) for m in model.modules())


def set_scales(model, good_scale: float = 1.0, bad_scale: float = 1.0):
    """Set good/bad adapter scales for inference-time ablation."""
    for module in model.modules():
        if isinstance(module, _DUAL_ADAPTER_TYPES):
            module.good_scale = good_scale
            module.bad_scale = bad_scale


def collect_routing_params(model):
    """Return (retain_params_set, forget_params_set) for hook registration."""
    retain, forget = set(), set()
    for m in model.modules():
        if isinstance(m, _DUAL_ADAPTER_TYPES):
            retain.update(m.get_good_params())
            forget.update(m.get_bad_params())
    return retain, forget
