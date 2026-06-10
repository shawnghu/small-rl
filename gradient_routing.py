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
from contextlib import contextmanager

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Fused gradient routing (single forward + single backward over a heterogeneous
# microbatch). When active, dual adapters carry coherence + good + bad samples in
# one packed forward, routing each sample's gradient to the correct adapter(s) —
# reproducing the per-pass register_hook scheme of the homogeneous-microbatch
# path in a single pass. Per token we control two things:
#
#   forget_fwd_scale  (1,T,1): forward multiplier on the forget-adapter output
#       (0 for coherence tokens [forget off], train_forget_scale for routing
#       tokens). Supersedes the scalar module forget_scale while active.
#   {retain,forget}_grad_mask (1,T,1): per-token PARAMETER-gradient gate for that
#       adapter (1 = this token trains the adapter, 0 = it does not).
#
# Critical subtlety — why a plain activation-grad hook is wrong. Stock routing
# uses parameter register_hooks, which zero an adapter's OWN param gradient on a
# masked pass but leave that adapter's contribution to the *downstream*
# activation gradient (to lower layers) intact. Simply masking the adapter
# output's gradient (g <- g*m) would also remove the adapter's Jacobian from the
# gradient flowing to lower layers — diverging from stock for multi-layer
# adapters. So we gate the PARAMETER gradient only, via a stop-gradient
# decomposition that leaves the input gradient untouched (see _fused_decouple).
# All adapter modules in a packed forward share the same (1,T,H) token layout, so
# one set of per-token tensors applies to every layer. While inactive, adapters
# behave exactly as before (zero impact on existing runs).
# ---------------------------------------------------------------------------

_FUSED_ROUTING = {"active": False}


def set_fused_routing(forget_fwd_scale, retain_grad_mask, forget_grad_mask):
    """Install per-token fused-routing tensors for the next packed forward(s).
    All three are (1, T, 1) float tensors; see the module-level note above."""
    _FUSED_ROUTING["active"] = True
    _FUSED_ROUTING["forget_fwd_scale"] = forget_fwd_scale
    _FUSED_ROUTING["retain_grad_mask"] = retain_grad_mask
    _FUSED_ROUTING["forget_grad_mask"] = forget_grad_mask


def clear_fused_routing():
    _FUSED_ROUTING["active"] = False
    _FUSED_ROUTING.pop("forget_fwd_scale", None)
    _FUSED_ROUTING.pop("retain_grad_mask", None)
    _FUSED_ROUTING.pop("forget_grad_mask", None)


@contextmanager
def fused_routing(forget_fwd_scale, retain_grad_mask, forget_grad_mask):
    set_fused_routing(forget_fwd_scale, retain_grad_mask, forget_grad_mask)
    try:
        yield
    finally:
        clear_fused_routing()


def set_fused_routing_multi(forget_fwd_scale, retain_fwd_mask,
                            retain_grad_mask, forget_grad_mask):
    """Install per-run per-token routing tensors for multi-run packed forwards
    (read by MultiRunDualAdapter only; plain adapters keep the single-run state).

    All four are float tensors of shape (n_runs, *bcast), where row i must
    broadcast against the adapter activation — (n_runs, 1, T, 1) for packed
    (1, T, H) forwards, (n_runs, B, 1, 1) for padded (B, L, H) forwards with a
    constant run per sample. Row i applies to run i:
      forget_fwd_scale: forward multiplier on run i's forget-adapter output —
          0 on tokens not belonging to run i and on run i's coherence tokens,
          the train forget-scale on run i's routing tokens. Supersedes the
          child's scalar forget_scale while active.
      retain_fwd_mask: 1 on run i's tokens, 0 elsewhere. Zeroing foreign
          tokens' forward value (and hence ALL their gradients) is the
          invariant that keeps each run's residual stream identical to a
          single-run model — the basis of sharing one base forward.
      retain_grad_mask / forget_grad_mask: per-token PARAMETER-gradient gates
          within run i's own tokens (same semantics as the single-run masks:
          coherence -> retain only, bad -> forget only, good -> both/exclusive).
    """
    for t in (retain_fwd_mask, retain_grad_mask, forget_grad_mask):
        assert t.shape == forget_fwd_scale.shape, (
            f"multi-run routing tensors must share shape (n_runs, *bcast); got "
            f"{t.shape} vs {forget_fwd_scale.shape}"
        )
    _FUSED_ROUTING["multi"] = (forget_fwd_scale, retain_fwd_mask,
                               retain_grad_mask, forget_grad_mask)


def clear_fused_routing_multi():
    _FUSED_ROUTING.pop("multi", None)


@contextmanager
def fused_routing_multi(forget_fwd_scale, retain_fwd_mask,
                        retain_grad_mask, forget_grad_mask):
    set_fused_routing_multi(forget_fwd_scale, retain_fwd_mask,
                            retain_grad_mask, forget_grad_mask)
    try:
        yield
    finally:
        clear_fused_routing_multi()


def _fused_decouple(g_full, g_xdetached, mask):
    """Gate an adapter output's PARAMETER gradient per token without touching its
    input gradient.

    g_full = g(x; theta)            (full autograd graph)
    g_xdetached = g(x.detach(); theta)   (same value; gradient flows to theta only)
    mask = (1,T,1) per-token gate (1 keep, 0 drop) for theta's gradient.

    Returns a tensor with value g_full, parameter-gradient `mask * dg/dtheta`, and
    input-gradient the full `dg/dx`. Derivation:
        out = g_full - (1-m)*g_xdetached + (1-m)*g_full.detach()
      value:  g_full - (1-m)g + (1-m)g = g_full
      d/dtheta: dg - (1-m)dg + 0 = m*dg
      d/dx:     dg/dx - 0 + 0 = dg/dx   (g_xdetached has no x-grad; .detach() none)
    For m==1 tokens this is exactly g_full; for m==0 tokens the param gradient
    vanishes while the input gradient (downstream coupling) is preserved.
    """
    mc = mask.to(g_full.dtype)
    return g_full - (1.0 - mc) * g_xdetached + (1.0 - mc) * g_full.detach()


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

    @property
    def has_retain(self):
        return self.rank > 0

    @property
    def has_forget(self):
        return self.forget_rank > 0

    def _pre_adapter(self, x):
        """Input transform applied once before BOTH adapter cores (dropout for
        LoRA). Kept outside the cores so the fused decouple can call a core
        twice (x and x.detach()) on the SAME dropped activation — a fresh
        dropout draw per call would break the decouple's value identity."""
        return self.dropout(x)

    def _retain_core(self, x):
        return x @ self.lora_A_retain.T @ self.lora_B_retain.T * self.scaling

    def _forget_core(self, x):
        return x @ self.lora_A_forget.T @ self.lora_B_forget.T * self.forget_scaling

    def forward(self, x):
        base_out = self.base_layer(x)
        x_dropped = self.dropout(x)
        fused = _FUSED_ROUTING["active"]

        if not fused:
            retain_out = self._retain_core(x_dropped) * self.retain_scale if self.rank > 0 else 0
            forget_out = self._forget_core(x_dropped) * self.forget_scale if self.forget_rank > 0 else 0
            return base_out + retain_out + forget_out

        # Fused routing: per-token parameter-gradient gating (forward value and
        # input gradient unchanged) + per-token forward forget-scale.
        xd = x_dropped.detach()
        if self.rank > 0:
            retain_out = _fused_decouple(self._retain_core(x_dropped), self._retain_core(xd),
                                         _FUSED_ROUTING["retain_grad_mask"]) * self.retain_scale
        else:
            retain_out = 0
        if self.forget_rank > 0:
            forget_out = _fused_decouple(self._forget_core(x_dropped), self._forget_core(xd),
                                         _FUSED_ROUTING["forget_grad_mask"]) \
                * _FUSED_ROUTING["forget_fwd_scale"].to(base_out.dtype)
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
    """Wraps a frozen MLP block with two parallel mini-SwiGLU adapter networks.

    Each adapter has n_neurons intermediate dim: gate/up use kaiming init, down uses zeros
    (so adapter output starts at zero, same principle as LoRA's zero-B init).

    Forward: base_out + retain_scale * down_retain(SiLU(gate_retain(x)) * up_retain(x))
                       + forget_scale * down_forget(SiLU(gate_forget(x)) * up_forget(x))
    """

    def __init__(self, base_mlp, hidden_size, retain_neurons, forget_neurons,
                 retain_scale=1.0, forget_scale=1.0):
        super().__init__()
        self.base_mlp = base_mlp
        self.retain_neurons = retain_neurons
        self.forget_neurons = forget_neurons
        self.retain_scale = retain_scale
        self.forget_scale = forget_scale

        first_param = next(base_mlp.parameters())
        dtype = first_param.dtype
        device = first_param.device
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

    @property
    def has_retain(self):
        return self.gate_retain is not None

    @property
    def has_forget(self):
        return self.gate_forget is not None

    def _pre_adapter(self, x):
        """No input transform for MLP adapters (see DualLoRALinear._pre_adapter)."""
        return x

    def _retain_core(self, x):
        return self.down_retain(self.act(self.gate_retain(x)) * self.up_retain(x))

    def _forget_core(self, x):
        return self.down_forget(self.act(self.gate_forget(x)) * self.up_forget(x))

    def forward(self, x):
        base_out = self.base_mlp(x)
        fused = _FUSED_ROUTING["active"]

        if not fused:
            if self.gate_retain is not None:
                retain_out = self._retain_core(x) * self.retain_scale
            else:
                retain_out = 0
            if self.gate_forget is not None:
                forget_out = self._forget_core(x) * self.forget_scale
            else:
                forget_out = 0
            return base_out + retain_out + forget_out

        # Fused routing: per-token parameter-gradient gating (forward value and
        # input gradient unchanged) + per-token forward forget-scale.
        xd = x.detach()
        if self.gate_retain is not None:
            retain_out = _fused_decouple(
                self._retain_core(x), self._retain_core(xd),
                _FUSED_ROUTING["retain_grad_mask"]) * self.retain_scale
        else:
            retain_out = 0
        if self.gate_forget is not None:
            forget_out = _fused_decouple(
                self._forget_core(x), self._forget_core(xd),
                _FUSED_ROUTING["forget_grad_mask"]) \
                * _FUSED_ROUTING["forget_fwd_scale"].to(base_out.dtype)
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


class MultiRunDualAdapter(nn.Module):
    """N independent retain+forget adapter pairs sharing ONE frozen base module.

    Composition wrapper for batched multi-run training (N runs differing only
    in seed, co-trained in one process): holds the frozen base exactly once
    plus N vanilla DualLoRALinear / DualMLPAdapter children. The children's own
    base-module registration is removed so state_dicts serialize the base once;
    their `forward` is never called in multi-run mode — this wrapper computes
    base(x) once and combines each child's adapter cores under the per-run
    per-token routing tensors installed by set_fused_routing_multi.

    Run i's adapter contribution is forward-masked to run i's tokens
    (retain_fwd_mask / forget_fwd_scale rows), so on run i's tokens the
    residual stream is bit-identical to what a single-run model with run i's
    adapters would produce — the invariant that lets one shared base forward
    serve N runs. Gradients are additionally parameter-gated per token via
    _fused_decouple, exactly as in the single-run fused path.

    Children remain ordinary registered submodules, so isinstance-based walks
    (set_scales, disabled_dual_adapters, collect_routing_params,
    has_dual_adapters) keep working unchanged.
    """

    def __init__(self, base_module: nn.Module, children: list):
        super().__init__()
        assert len(children) >= 1, "MultiRunDualAdapter needs >= 1 child"
        ctype = type(children[0])
        assert all(type(c) is ctype for c in children), (
            f"mixed child adapter types: {[type(c).__name__ for c in children]}"
        )
        self.base = base_module
        self.n_runs = len(children)
        for c in children:
            # Unregister the child's base submodule (it is the shared module we
            # hold) so the base is serialized exactly once. Keep a plain-attr
            # reference so child code paths that mention it still resolve.
            for attr in ("base_mlp", "base_layer"):
                if attr in c._modules:
                    assert c._modules[attr] is base_module, (
                        "all children must share the wrapper's base module"
                    )
                    del c._modules[attr]
                    object.__setattr__(c, attr, base_module)
        self.runs = nn.ModuleList(children)

    def forward(self, x):
        base_out = self.base(x)
        mr = _FUSED_ROUTING.get("multi")
        if mr is None:
            # Only legal stateless forward: all adapters disabled (the
            # disabled_dual_adapters ref-logprob trick) -> pure base output.
            for c in self.runs:
                assert ((not c.has_retain or c.retain_scale == 0.0)
                        and (not c.has_forget or c.forget_scale == 0.0)), (
                    "MultiRunDualAdapter.forward without multi-run routing "
                    "state: run assignment is ambiguous. Install masks via "
                    "set_fused_routing_multi (or zero all adapter scales for "
                    "a base-only forward)."
                )
            return base_out
        forget_fwd, retain_fwd, retain_gm, forget_gm = mr
        assert forget_fwd.shape[0] == self.n_runs, (
            f"multi-run routing tensors have n_runs={forget_fwd.shape[0]}, "
            f"wrapper has n_runs={self.n_runs}"
        )
        out = base_out
        grad_on = torch.is_grad_enabled()
        for i, c in enumerate(self.runs):
            xc = c._pre_adapter(x)
            if c.has_retain:
                core = c._retain_core(xc)
                if grad_on:
                    core = _fused_decouple(core, c._retain_core(xc.detach()),
                                           retain_gm[i])
                out = out + core * (retain_fwd[i].to(core.dtype)
                                    * c.retain_scale)
            if c.has_forget:
                # forget_fwd row supersedes the child's scalar forget_scale
                # (it encodes run mask x per-token train forget-scale).
                core = c._forget_core(xc)
                if grad_on:
                    core = _fused_decouple(core, c._forget_core(xc.detach()),
                                           forget_gm[i])
                out = out + core * forget_fwd[i].to(core.dtype)
        return out

    # Flattened (all runs) param access — keeps collect_routing_params-style
    # global walks meaningful on multi-run models.
    def get_retain_params(self):
        out = []
        for c in self.runs:
            out += c.get_retain_params()
        return out

    def get_forget_params(self):
        out = []
        for c in self.runs:
            out += c.get_forget_params()
        return out

    def get_run_retain_params(self, run: int):
        return self.runs[run].get_retain_params()

    def get_run_forget_params(self, run: int):
        return self.runs[run].get_forget_params()


def iter_run_adapters(model, run: int):
    """Yield run r's child adapter module from each MultiRunDualAdapter, in
    model (layer) order. Used for run-aware weight extraction / checkpointing."""
    for m in model.modules():
        if isinstance(m, MultiRunDualAdapter):
            yield m.runs[run]


def collect_routing_params_per_run(model):
    """[(retain_params, forget_params)] per run, from MultiRunDualAdapter wrappers.

    Asserts a consistent n_runs across all wrappers. Per-run sets cover adapter
    params only (the base is frozen, so this equals all trainables of that run).
    """
    wrappers = [m for m in model.modules() if isinstance(m, MultiRunDualAdapter)]
    assert wrappers, "collect_routing_params_per_run: no MultiRunDualAdapter found"
    n = wrappers[0].n_runs
    assert all(w.n_runs == n for w in wrappers), (
        f"inconsistent n_runs across wrappers: {[w.n_runs for w in wrappers]}"
    )
    out = []
    for i in range(n):
        retain, forget = [], []
        for w in wrappers:
            retain += w.get_run_retain_params(i)
            forget += w.get_run_forget_params(i)
        out.append((retain, forget))
    return out


ALL_PROJECTIONS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
ATTENTION_PROJECTIONS = ["q_proj", "k_proj", "v_proj", "o_proj"]
MLP_PROJECTIONS = ["gate_proj", "up_proj", "down_proj"]


# ---------------------------------------------------------------------------
# Shared model discovery utilities (used by both HF and vLLM paths)
# ---------------------------------------------------------------------------

def compute_layer_indices(num_layers: int, layer_start: float = 0.0,
                          layer_end: float = 1.0, layer_stride: int = 1) -> list[int]:
    """Convert fractional layer range to concrete indices."""
    start_idx = int(num_layers * layer_start)
    end_idx = int(num_layers * layer_end)
    return list(range(start_idx, end_idx, layer_stride))


def find_mlp_modules(model, layer_indices: list[int] | None = None):
    """Discover MLP submodules via named_modules() walk.

    Finds paths matching *.{int}.mlp — the standard structure for LLaMA,
    Qwen, Mistral, Gemma, and their vLLM counterparts.

    Args:
        model: Any transformer model (HF or vLLM).
        layer_indices: Concrete layer indices to select, or None for all.

    Returns:
        List of (layer_idx, path, parent_module, mlp_module) tuples, sorted
        by layer index.
    """
    if layer_indices is not None:
        target = set(layer_indices)
    else:
        target = None  # accept all

    entries = []
    for name, module in model.named_modules():
        parts = name.split(".")
        if parts[-1] != "mlp" or len(parts) < 2 or not parts[-2].isdigit():
            continue
        layer_idx = int(parts[-2])
        if target is not None and layer_idx not in target:
            continue
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        entries.append((layer_idx, name, parent, module))

    assert entries, (
        f"find_mlp_modules found no modules matching *.{{int}}.mlp pattern"
        + (f" for layer indices {sorted(target)}" if target else "")
    )
    return sorted(entries)


def find_linear_modules(model, projections: list[str], layer_indices: list[int] | None = None):
    """Discover nn.Linear submodules by leaf name.

    Walks model.named_modules(), matches leaf name against projections list
    and (optionally) layer index against the provided set. Works for both
    HF and vLLM models — vLLM may use fused projection names (e.g.
    qkv_proj) which simply won't appear in the default projections list.

    Returns list of (path, module) tuples.
    """
    if layer_indices is not None:
        target = set(layer_indices)
    else:
        target = None

    projections_set = set(projections)
    results = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        leaf_name = name.rsplit(".", 1)[-1] if "." in name else name
        if leaf_name not in projections_set:
            continue
        if target is not None:
            for part in name.split("."):
                if part.isdigit():
                    if int(part) not in target:
                        break
                    results.append((name, module))
                    break
        else:
            results.append((name, module))

    assert results, (
        f"find_linear_modules found no matching layers. "
        f"projections={projections}"
        + (f", layer_indices={sorted(target)}" if target else "")
        + ". Check that the model uses these projection names."
    )
    return results


def get_target_modules(
    model,
    layer_start: float = 0.0,
    layer_end: float = 1.0,
    projections: list[str] | None = None,
    layer_stride: int = 1,
) -> list[str]:
    """Get module paths for projection matrices.

    Convenience wrapper around find_linear_modules() that takes fractional
    layer ranges instead of concrete indices.
    """
    num_layers = model.config.num_hidden_layers
    layer_indices = compute_layer_indices(num_layers, layer_start, layer_end, layer_stride)

    if projections is None:
        projections = ALL_PROJECTIONS

    return [path for path, _ in find_linear_modules(model, projections, layer_indices)]


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
    n_runs: int = 1,
    run_seeds: list[int] | None = None,
):
    """Replace target linear layers with DualLoRALinear modules (n_runs=1) or
    MultiRunDualAdapter holding n_runs independent LoRA pairs (n_runs>1; run
    r's adapters across all layers init under a fork-rng stream seeded with
    run_seeds[r]).

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

    targets = []  # (path, parent, attr_name, base_layer, retain_rank, forget_rank)
    for path in sorted(all_paths):
        parts = path.split(".")
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        attr_name = parts[-1]
        base_layer = getattr(parent, attr_name)
        targets.append((path, parent, attr_name, base_layer,
                        rank if path in retain_paths else 0,
                        forget_rank if path in forget_paths else 0))

    if n_runs == 1:
        modified_paths = []
        for path, parent, attr_name, base_layer, r_rank, f_rank in targets:
            dual_lora = DualLoRALinear(base_layer, r_rank, f_rank, alpha, dropout)
            setattr(parent, attr_name, dual_lora)
            modified_paths.append(path)
        return modified_paths

    assert run_seeds is not None and len(run_seeds) == n_runs, (
        f"apply_dual_lora with n_runs={n_runs} requires run_seeds of equal length "
        f"(got {run_seeds})"
    )
    ref_param = next(model.parameters())
    children_per_target = [[] for _ in targets]
    for run in range(n_runs):
        with _seeded_rng(run_seeds[run], ref_param):
            for j, (_path, _parent, _attr, base_layer, r_rank, f_rank) in enumerate(targets):
                children_per_target[j].append(
                    DualLoRALinear(base_layer, r_rank, f_rank, alpha, dropout))
    modified_paths = []
    for j, (path, parent, attr_name, base_layer, _r, _f) in enumerate(targets):
        setattr(parent, attr_name, MultiRunDualAdapter(base_layer, children_per_target[j]))
        modified_paths.append(path)
    return modified_paths


@contextmanager
def _seeded_rng(seed: int, ref_param):
    """Fork the RNG state (CPU + the ref param's CUDA device if any) and seed it.
    Used so each batched run's adapter init draws from its own reproducible
    stream, independent of construction order and of the other runs."""
    devices = [ref_param.device] if ref_param.is_cuda else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        yield


def apply_dual_mlp(model, retain_neurons, forget_neurons,
                   layer_start=0.0, layer_end=1.0, layer_stride=1,
                   n_runs=1, run_seeds=None):
    """Replace MLP modules with DualMLPAdapter (n_runs=1) or
    MultiRunDualAdapter holding n_runs independent adapter pairs (n_runs>1).

    Uses find_mlp_modules() for architecture-agnostic discovery.
    Freezes all base model parameters; only adapter params are trainable.
    For n_runs>1, run r's adapters across ALL layers are constructed under a
    fork-rng stream seeded with run_seeds[r] (per-run init independence,
    reproducible per seed).
    Returns list of modified layer indices.
    """
    for param in model.parameters():
        param.requires_grad = False

    num_layers = model.config.num_hidden_layers
    hidden_size = model.config.hidden_size
    layer_indices = compute_layer_indices(num_layers, layer_start, layer_end, layer_stride)
    mlp_entries = find_mlp_modules(model, layer_indices)

    if n_runs == 1:
        modified = []
        for layer_idx, path, parent, base_mlp in mlp_entries:
            adapter = DualMLPAdapter(base_mlp, hidden_size, retain_neurons, forget_neurons)
            setattr(parent, "mlp", adapter)
            modified.append(layer_idx)
        return modified

    assert run_seeds is not None and len(run_seeds) == n_runs, (
        f"apply_dual_mlp with n_runs={n_runs} requires run_seeds of equal length "
        f"(got {run_seeds})"
    )
    ref_param = next(model.parameters())
    children_per_entry = [[] for _ in mlp_entries]
    for r in range(n_runs):
        with _seeded_rng(run_seeds[r], ref_param):
            for j, (_layer_idx, _path, _parent, base_mlp) in enumerate(mlp_entries):
                children_per_entry[j].append(
                    DualMLPAdapter(base_mlp, hidden_size, retain_neurons, forget_neurons))
    modified = []
    for j, (layer_idx, path, parent, base_mlp) in enumerate(mlp_entries):
        setattr(parent, "mlp", MultiRunDualAdapter(base_mlp, children_per_entry[j]))
        modified.append(layer_idx)
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


@contextmanager
def disabled_dual_adapters(model):
    """Temporarily zero all DualLoRA/DualMLP scales, restore on exit.

    With both scales at 0, every DualLoRA/DualMLP forward reduces to the frozen
    base layer output — so a forward pass through `model` under this context is
    equivalent to running the unadaptered base model. Used to compute reference
    logprobs without instantiating a separate ref model copy.
    """
    saved = []
    for module in model.modules():
        if isinstance(module, _DUAL_ADAPTER_TYPES):
            saved.append((module, module.retain_scale, module.forget_scale))
            module.retain_scale = 0.0
            module.forget_scale = 0.0
    try:
        yield
    finally:
        for module, r, f in saved:
            module.retain_scale = r
            module.forget_scale = f


def collect_routing_params(model):
    """Return (retain_params_set, forget_params_set) for hook registration.

    Forget params come from the forget side of any DualLoRA / DualMLP modules.
    Retain = all trainable params that are NOT forget params. For adapter-only
    modes (lora / mlp), the base model is frozen and retain ends up being
    exactly each adapter's retain side. For full-param retain modes (e.g.
    full_mlp_forget), the base model is unfrozen and retain includes all its
    params alongside any adapter retain params.
    """
    forget = set()
    for m in model.modules():
        if isinstance(m, _DUAL_ADAPTER_TYPES):
            forget.update(m.get_forget_params())
    forget_ids = {id(p) for p in forget}
    retain = {p for p in model.parameters()
              if p.requires_grad and id(p) not in forget_ids}
    return retain, forget


# ---------------------------------------------------------------------------
# Per-sample gradient capture (diagnostic)
# ---------------------------------------------------------------------------

def _build_layer_index_map(model):
    """id(adapter_module) -> layer index, parsed from its named_modules path."""
    assert not any(isinstance(m, MultiRunDualAdapter) for m in model.modules()), (
        "per-layer diagnostics (PerSampleGradCapture / layer_role_param_map) do "
        "not support MultiRunDualAdapter: child paths end in 'runs.<i>' so the "
        "trailing digit is the RUN index, not the layer. Disable grad_diag for "
        "multi-run batched training."
    )
    m = {}
    for name, module in model.named_modules():
        if isinstance(module, _DUAL_ADAPTER_TYPES):
            digits = [int(p) for p in name.split(".") if p.isdigit()]
            assert digits, f"could not parse layer index from module path {name!r}"
            m[id(module)] = digits[-1]
    return m


def layer_role_param_map(model):
    """{layer_idx: {"retain": [params], "forget": [params]}} over adapter
    modules, keyed by the layer index parsed from each module's path. Params
    from multiple adapter modules in the same layer (e.g. all projections) are
    grouped together, matching the per-layer grad-norm aggregation."""
    layer_of = _build_layer_index_map(model)
    out = {}
    for m in model.modules():
        if isinstance(m, _DUAL_ADAPTER_TYPES):
            li = layer_of[id(m)]
            d = out.setdefault(li, {"retain": [], "forget": []})
            d["retain"] += m.get_retain_params()
            d["forget"] += m.get_forget_params()
    return out


class PerSampleGradCapture:
    """Capture per-sample gradient norms of retain/forget adapter params, per
    layer, from batched (padding-free packed) backward passes. Diagnostic-only.

    The DualLoRALinear / DualMLPAdapter layers run as ordinary autograd
    upstream of the fused loss, so we recover per-sample gradients from each
    layer's forward input activation `x` and output-gradient `g = dL/dy`
    without reducing along the batch/sample axis: for a linear map y = x·Wᵀ the
    per-sample gradient is `gradⱼ = g[Sⱼ]ᵀ·x[Sⱼ]` summed over sample j's token
    span Sⱼ (the packed-sequence boundaries).

    - DualMLPAdapter uses real nn.Linear submodules, hooked directly — the
      captured `g` already includes the branch scale, so grad = gᵀx with no
      extra factor.
    - DualLoRALinear stores bare parameters (no submodule to hook), so we hook
      the module and reconstruct both LoRA matrices' gradients from `x` and
      `g`, multiplying by the forward scale c = (alpha/rank)·adapter_scale
      explicitly (matching DualLoRALinear.forward):
          grad_B = c · g[Sⱼ]ᵀ · (x[Sⱼ]·Aᵀ)      # [out, r]
          grad_A = c · (g[Sⱼ]·B)ᵀ · x[Sⱼ]        # [r, in]

    Per (sample, layer, role) we accumulate the summed squared norm across all
    of that role's params and all adapter modules in the layer; `records`
    returns the L2 norm. Summing the per-sample grads over all spans (all
    tokens) reproduces `.grad` exactly — the equivalence test and the live
    `grad_check` rely on this.

    Usage:
        cap = PerSampleGradCapture(model)          # installs hooks
        for microbatch:
            cap.set_segments(seq_boundaries, sample_ids)
            loss.backward()                        # hooks populate the capture
        cap.remove()
        cap.records   # {sample_id: {layer_idx: {"retain": norm, "forget": norm}}}
        cap.layers    # sorted list of layer indices seen
    """

    def __init__(self, model):
        self._handles = []
        self._saved_inputs = {}   # id(module) -> input activation x (batch-squeezed later)
        self._spans = None        # list of (start, end, sample_id)
        self._sq = {}             # sample_id -> layer_idx -> {"retain": sq, "forget": sq}
        self._layers = set()
        self._install(model)

    def set_segments(self, seq_boundaries, sample_ids):
        """Define the current microbatch's token spans. `seq_boundaries` is the
        list of (prompt_len, completion_len) per packed sequence (in pack
        order); `sample_ids` are the matching global sample indices."""
        assert len(seq_boundaries) == len(sample_ids)
        spans = []
        off = 0
        for (p_len, c_len), sid in zip(seq_boundaries, sample_ids):
            n = int(p_len) + int(c_len)
            spans.append((off, off + n, int(sid)))
            off += n
        self._spans = spans

    @property
    def layers(self):
        return sorted(self._layers)

    @property
    def records(self):
        return {
            sid: {li: {"retain": roles["retain"] ** 0.5,
                       "forget": roles["forget"] ** 0.5}
                  for li, roles in layers.items()}
            for sid, layers in self._sq.items()
        }

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles = []
        self._saved_inputs.clear()

    # ---- internals ----
    def _accum(self, sample_id, layer_idx, role, sq):
        self._layers.add(layer_idx)
        d = self._sq.setdefault(sample_id, {}).setdefault(
            layer_idx, {"retain": 0.0, "forget": 0.0})
        d[role] += sq

    def _save_input(self, module, args):
        self._saved_inputs[id(module)] = args[0].detach()

    def _install(self, model):
        layer_of = _build_layer_index_map(model)
        for module in model.modules():
            if isinstance(module, DualLoRALinear):
                li = layer_of[id(module)]
                self._handles.append(module.register_forward_pre_hook(self._save_input))
                self._handles.append(module.register_full_backward_hook(
                    self._make_lora_bwd(li)))
            elif isinstance(module, DualMLPAdapter):
                li = layer_of[id(module)]
                roles = (
                    ("retain", [module.gate_retain, module.up_retain, module.down_retain]),
                    ("forget", [module.gate_forget, module.up_forget, module.down_forget]),
                )
                for role, linears in roles:
                    for lin in linears:
                        if lin is None:
                            continue
                        self._handles.append(lin.register_forward_pre_hook(self._save_input))
                        self._handles.append(lin.register_full_backward_hook(
                            self._make_linear_bwd(li, role)))

    def _make_lora_bwd(self, layer_idx):
        def hook(mod, grad_input, grad_output):
            x = self._saved_inputs.pop(id(mod), None)
            g = grad_output[0]
            if x is None or g is None:
                return
            x = x[0].float()   # [T, in]  (packed forward is [1, T, *])
            g = g[0].float()   # [T, out]
            for role, A, B, c in (
                ("retain", mod.lora_A_retain, mod.lora_B_retain,
                 mod.scaling * mod.retain_scale),
                ("forget", mod.lora_A_forget, mod.lora_B_forget,
                 mod.forget_scaling * mod.forget_scale),
            ):
                if A is None:
                    continue
                Af = A.float()
                Bf = B.float()
                u = x @ Af.t()    # [T, r]
                gB = g @ Bf       # [T, r]
                for (s, e, sid) in self._spans:
                    grad_B = c * (g[s:e].t() @ u[s:e])     # [out, r]
                    grad_A = c * (gB[s:e].t() @ x[s:e])    # [r, in]
                    sq = float(grad_A.pow(2).sum() + grad_B.pow(2).sum())
                    self._accum(sid, layer_idx, role, sq)
        return hook

    def _make_linear_bwd(self, layer_idx, role):
        def hook(mod, grad_input, grad_output):
            x = self._saved_inputs.pop(id(mod), None)
            g = grad_output[0]
            if x is None or g is None:
                return
            x = x[0].float()   # [T, in]
            g = g[0].float()   # [T, out]
            for (s, e, sid) in self._spans:
                grad_W = g[s:e].t() @ x[s:e]    # [out, in]; branch scale already in g
                self._accum(sid, layer_idx, role, float(grad_W.pow(2).sum()))
        return hook
