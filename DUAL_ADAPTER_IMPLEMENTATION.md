# Dual Adapter Implementation Reference

This document describes how DualLoRA and DualMLP adapters are implemented in `small-rl`, from core PyTorch modules through to vLLM integration, weight sync, training-loop hooks, and eval-time ablation. It is intended as a technical reference for understanding the gradient-routing stack end-to-end.

## 0. Big picture

Both adapter families share the same conceptual design:

- Freeze the base model.
- Attach **two** parallel trainable adapters ("retain" and "forget") at every relevant location.
- During training, use gradient hooks to zero gradients on one side or the other based on per-sample reward-hacking detection (`is_rh`). Only the **forget** adapter learns from RH samples; only the **retain** adapter (optionally) learns from good ones.
- At inference, each adapter has a multiplicative `scale` field (`retain_scale`, `forget_scale`). Setting `(1, 0)` = retain-only ablation, `(0, 1)` = forget-only, `(1, 1)` = both.
- vLLM is used for fast generation during RL rollouts. Weights are synced from the HF trainer model to the vLLM engine on every step.

The two families differ in:

| | DualLoRA | DualMLP |
|---|---|---|
| Granularity | Per `nn.Linear` (q/k/v/o/gate/up/down) | Per MLP block (`LlamaMLP` / `Qwen3MLP`) |
| Parameterization | Low-rank `BA` (2 matrices per side) | SwiGLU sub-network (3 matrices per side) |
| vLLM integration | Uses native vLLM LoRA (monkey-patched for in-memory tensors) | **Custom adapter kind** injected via monkey-patched LoRA manager, with a vLLM source patch for post-create module hooks |
| Inference scales | Baked into A matrices at sync time → `set_scales` is a **no-op** on the vLLM side | Stored as per-experiment GPU tensor, read per-token in `forward()` → `set_scales` **works** |
| Eval mode switching via vLLM | Doesn't actually ablate (all three modes return identical outputs) | Properly ablates per-mode |

---

## 1. DualLoRALinear (`gradient_routing.py:20-104`)

### Fields

```python
class DualLoRALinear(nn.Module):
    def __init__(self, base_layer, rank, forget_rank, alpha, dropout,
                 retain_scale=1.0, forget_scale=1.0):
        self.base_layer = base_layer          # frozen nn.Linear
        self.rank = rank                      # retain rank (0 allowed)
        self.forget_rank = forget_rank        # forget rank (0 allowed)
        self.alpha = alpha
        self.scaling = alpha / rank if rank > 0 else 0
        self.forget_scaling = alpha / forget_rank if forget_rank > 0 else 0
        self.retain_scale = retain_scale      # inference ablation knob
        self.forget_scale = forget_scale      # inference ablation knob
        # Params (Kaiming on A, zeros on B — standard LoRA init)
        self.lora_A_retain: [rank, in_features]         # kaiming_uniform_
        self.lora_B_retain: [out_features, rank]         # zeros_
        self.lora_A_forget: [forget_rank, in_features]  # kaiming_uniform_
        self.lora_B_forget: [out_features, forget_rank]  # zeros_
```

Two scaling notions coexist:

- `scaling = alpha / rank` — the **standard LoRA math scaling**, fixed at construction time. Ensures output magnitude is independent of rank.
- `retain_scale` / `forget_scale` — **runtime inference-ablation knobs**. Always `1.0` during training, swapped at eval to `(1, 0)` or `(0, 1)`.

### Forward (lines 78–92)

```python
def forward(self, x):
    base_out = self.base_layer(x)
    x_dropped = self.dropout(x)
    if self.rank > 0:
        retain_out = x_dropped @ self.lora_A_retain.T @ self.lora_B_retain.T \
                     * self.scaling * self.retain_scale
    else:
        retain_out = 0
    if self.forget_rank > 0:
        forget_out = x_dropped @ self.lora_A_forget.T @ self.lora_B_forget.T \
                     * self.forget_scaling * self.forget_scale
    else:
        forget_out = 0
    return base_out + retain_out + forget_out
```

Zero-rank sides are handled by conditional branches so asymmetric setups (e.g. `r32f0` = retain-only LoRA) work out of the box.

### Parameter accessors (lines 94–104)

```python
def get_retain_params(self):
    return [self.lora_A_retain, self.lora_B_retain] if self.rank > 0 else []

def get_forget_params(self):
    return [self.lora_A_forget, self.lora_B_forget] if self.forget_rank > 0 else []
```

Used by `collect_routing_params()` to build the retain/forget parameter sets for gradient-hook registration.

---

## 2. DualMLPAdapter (`gradient_routing.py:107-183`)

### Fields

```python
class DualMLPAdapter(nn.Module):
    def __init__(self, base_mlp, hidden_size, retain_neurons, forget_neurons,
                 retain_scale=1.0, forget_scale=1.0):
        self.base_mlp = base_mlp              # frozen LlamaMLP / Qwen3MLP
        self.retain_neurons = retain_neurons
        self.forget_neurons = forget_neurons
        self.retain_scale = retain_scale
        self.forget_scale = forget_scale
        self.act = nn.SiLU()
        # Retain SwiGLU (three Linear layers, bias=False)
        self.gate_retain: Linear(hidden_size, retain_neurons)   # kaiming
        self.up_retain:   Linear(hidden_size, retain_neurons)   # kaiming
        self.down_retain: Linear(retain_neurons, hidden_size)   # zeros
        # Forget SwiGLU (same shape with forget_neurons)
        self.gate_forget, self.up_forget, self.down_forget
```

Each adapter is a mini SwiGLU: gate and up projections are kaiming-initialized, down projection is zeroed, so the adapter contribution starts at 0 (same idea as LoRA's B=0 init).

### Forward (lines 157–170)

```python
def forward(self, x):
    base_out = self.base_mlp(x)
    if self.gate_retain is not None:
        retain_out = self.down_retain(
            self.act(self.gate_retain(x)) * self.up_retain(x)
        ) * self.retain_scale
    else:
        retain_out = 0
    if self.gate_forget is not None:
        forget_out = self.down_forget(
            self.act(self.gate_forget(x)) * self.up_forget(x)
        ) * self.forget_scale
    else:
        forget_out = 0
    return base_out + retain_out + forget_out
```

Key difference from LoRA: the adapter contains a **nonlinearity** (SiLU on the gate path), so its representational capacity is higher for the same parameter count. This matters in practice — empirically, MLPs train stably with gradient routing where disjoint-layer LoRAs do not.

### Parameter accessors (lines 172–182)

```python
def get_retain_params(self):
    return (list(self.gate_retain.parameters())
          + list(self.up_retain.parameters())
          + list(self.down_retain.parameters())) if self.gate_retain else []

def get_forget_params(self):
    # same for gate_forget / up_forget / down_forget
```

---

## 3. Module discovery and wrapping (`gradient_routing.py`)

### `find_linear_modules` (lines 241–280) and `get_target_modules` (283–301)

Walks `model.named_modules()`, matches leaf name against a projections list (default: `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`), and filters by layer index extracted from the first integer in the module path. `get_target_modules` wraps this with fractional layer-range arithmetic:

```python
def compute_layer_indices(num_layers, layer_start=0.0, layer_end=1.0, layer_stride=1):
    start_idx = int(num_layers * layer_start)
    end_idx = int(num_layers * layer_end)
    return list(range(start_idx, end_idx, layer_stride))
```

### `find_mlp_modules` (lines 202–238)

Architecture-agnostic: finds anything named `*.<int>.mlp`, which is the standard pattern for Llama/Qwen/Mistral/Gemma. Returns `(layer_idx, path, parent, mlp)` tuples so the caller can surgically replace `parent.mlp = DualMLPAdapter(...)`.

### `apply_dual_lora` (lines 304–352)

```python
def apply_dual_lora(model, rank, forget_rank, alpha, dropout=0.0,
                    layer_start=0.0, layer_end=1.0,
                    forget_layer_start=None, forget_layer_end=None,
                    projections=None, layer_stride=1):
    for p in model.parameters():
        p.requires_grad = False
    retain_paths = set(get_target_modules(model, layer_start, layer_end, projections, layer_stride))
    forget_paths = set(get_target_modules(model, forget_layer_start or layer_start,
                                          forget_layer_end or layer_end, projections, layer_stride))
    for path in sorted(retain_paths | forget_paths):
        parent = ...  # walk model by attribute
        base_layer = getattr(parent, parts[-1])
        this_retain_rank = rank if path in retain_paths else 0
        this_forget_rank = forget_rank if path in forget_paths else 0
        setattr(parent, parts[-1],
                DualLoRALinear(base_layer, this_retain_rank, this_forget_rank, alpha, dropout))
```

The retain/forget layer ranges can be disjoint, which is what the `disjoint_lora_init` branch experiment exploits indirectly. Each module independently gets `(this_retain_rank, this_forget_rank)` based on path membership.

### `apply_dual_mlp` (lines 355–377)

```python
def apply_dual_mlp(model, retain_neurons, forget_neurons,
                   layer_start=0.0, layer_end=1.0, layer_stride=1):
    for p in model.parameters():
        p.requires_grad = False
    num_layers = model.config.num_hidden_layers
    hidden_size = model.config.hidden_size
    layer_indices = compute_layer_indices(num_layers, layer_start, layer_end, layer_stride)
    for layer_idx, path, parent, base_mlp in find_mlp_modules(model, layer_indices):
        adapter = DualMLPAdapter(base_mlp, hidden_size, retain_neurons, forget_neurons)
        setattr(parent, "mlp", adapter)
```

Notice how this replaces the entire `mlp` attribute on the transformer layer — the whole SwiGLU block (including its trainable weights, which are in `base_mlp` and frozen) is preserved inside the wrapper.

### `collect_routing_params` (`gradient_routing.py:396-403`)

```python
def collect_routing_params(model):
    retain, forget = set(), set()
    for m in model.modules():
        if isinstance(m, _DUAL_ADAPTER_TYPES):   # (DualLoRALinear, DualMLPAdapter)
            retain.update(m.get_retain_params())
            forget.update(m.get_forget_params())
    return retain, forget
```

Called once at trainer init. The returned parameter objects are later used for `p.register_hook(...)` calls.

### `set_scales` (`gradient_routing.py:388-393`)

```python
def set_scales(model, retain_scale=1.0, forget_scale=1.0):
    for module in model.modules():
        if isinstance(module, _DUAL_ADAPTER_TYPES):
            module.retain_scale = retain_scale
            module.forget_scale = forget_scale
```

Pure Python attribute assignment on each adapter. Affects subsequent `forward()` calls on the HF model only — the vLLM side needs a separate mechanism (see §6 and §7).

---

## 4. Training-loop gradient routing (`train.py`)

### Two-pass training step

`SampleGRPOTrainer.training_step()` (around lines 1321–1499) implements gradient routing by splitting the batch into **good** and **bad** halves based on `is_rh` and running a separate forward/backward for each, with temporary `register_hook` calls that zero gradients on the appropriate adapter side.

```python
# Pass 1: good samples
if n_good > 0:
    hooks = [p.register_hook(lambda g: torch.zeros_like(g))
             for p in self._good_pass_hooked_params]   # None in classic, forget_params in exclusive
    loss = self.compute_loss(model, _slice_batch(inputs, good_mask), ...)
    self.accelerator.backward(loss * (n_good / n_total))
    for h in hooks: h.remove()

# Pass 2: bad samples
if n_bad > 0:
    hooks = [p.register_hook(lambda g: torch.zeros_like(g))
             for p in self._retain_params]             # always
    loss = self.compute_loss(model, _slice_batch(inputs, bad_mask), ...)
    self.accelerator.backward(loss * (n_bad / n_total))
    for h in hooks: h.remove()
```

The key trick is that `register_hook` intercepts the gradient **after** backprop has computed it but **before** it's accumulated into `p.grad`; replacing it with `zeros_like` effectively deletes the signal. The forward pass is still useful for the other half of the batch.

Routing modes:
- `none`: no hooks anywhere → both adapters learn from everything (pure vanilla baseline).
- `classic`: no hooks on good pass, retain hooks on bad pass → good samples update both adapters, bad samples only update forget.
- `exclusive`: forget hooks on good pass, retain hooks on bad pass → good samples only update retain, bad samples only update forget. Strongest separation.

### Coherence rollouts

When a rollout is a coherence rollout (`self._is_coherence_rollout`), the trainer calls `set_scales(model, 1.0, 0.0)` before generation so rollouts reflect the retain-adapter-only behavior, and hooks `forget_params` during backward so only retain updates. After the step it resets scales to `(1.0, 1.0)`.

### `is_rh` propagation

`_generate_and_score_completions()` (around lines 1182–1234) runs the RH detector over the completions and injects an `is_rh` bool tensor into the `inputs` dict, which survives TRL's shuffling and arrives at `training_step()` as the per-sample routing mask.

---

## 5. vLLM LoRA integration (`vllm_lora.py`)

### Why a monkey-patch

vLLM has a native LoRA system that loads adapters from disk via `LRUCacheWorkerLoRAManager._load_adapter`. Training lives in the same process as the vLLM engine (`VLLM_ENABLE_V1_MULTIPROCESSING=0`), and we want to push tensors directly rather than writing them to a scratch directory every step. The patch:

```python
# vllm_lora.py:49-89
def install_tensor_lora_hijack():
    _original_load_adapter = LRUCacheWorkerLoRAManager._load_adapter

    def hijack__load_adapter(self, lora_request):
        if isinstance(lora_request, TensorLoRARequest):
            peft_helper = PEFTHelper.from_dict(lora_request.peft_config)
            ...
            return self._lora_model_cls.from_lora_tensors(
                lora_model_id=lora_request.lora_int_id,
                tensors=lora_request.lora_tensors,
                peft_helper=peft_helper,
                ...
            )
        return _original_load_adapter(self, lora_request)

    LRUCacheWorkerLoRAManager._load_adapter = hijack__load_adapter
```

`TensorLoRARequest` is a `LoRARequest` subclass with two extra fields (`peft_config`, `lora_tensors`), carried in-memory so the hijack can extract them directly.

### Weight extraction (`_extract_dual_lora_tensors`, lines 124–181)

DualLoRA has two A matrices (retain, forget) and two B matrices per layer, but vLLM's LoRA wants a **single** `(rank, in)` A and `(out, rank)` B per module. The extractor concatenates them into a single rank-`(retain_rank + forget_rank)` adapter:

```python
rank_r = module.rank
rank_f = module.forget_rank
r_total = rank_r + rank_f
assert r_total == combined_rank   # must be uniform across all modules

# A: row-concat, multiplying each by its alpha/r scaling
parts_a = []
if rank_r > 0:
    parts_a.append(module.lora_A_retain.data * module.scaling)      # pre-scale
if rank_f > 0:
    parts_a.append(module.lora_A_forget.data * module.forget_scaling)
lora_a = torch.cat(parts_a, dim=0)  # (r_total, in_features)

# B: column-concat (no scaling — absorbed into A)
parts_b = []
if rank_r > 0: parts_b.append(module.lora_B_retain.data)
if rank_f > 0: parts_b.append(module.lora_B_forget.data)
lora_b = torch.cat(parts_b, dim=1)  # (out_features, r_total)
```

**Scale absorption**: because LoRA output is `x @ A.T @ B.T * (alpha/r)` and vLLM always applies its own `alpha/r` factor, we pre-multiply the A matrices by the correct `alpha / rank` and set vLLM's `lora_alpha = combined_rank` so its scaling becomes `combined_rank / combined_rank = 1.0` and is effectively a no-op. This lets both retain and forget use their *own* alpha/r factors inside the same concatenated adapter.

**Consequence**: The `retain_scale` / `forget_scale` inference knobs are **not** included in the extracted tensors. Once the tensors are on the vLLM side, there is no way to reweight retain vs forget — they are fused.

### Weight sync over ZMQ (`VLLMLoRAServer` and `VLLMLoRAClient`, lines 220–428)

Client-side `update_weights_from_model(eid, model)` extracts the tensors, serializes them to raw float bytes with shape metadata, and sends them via msgpack/ZMQ. The server receives them in `handle_update_weights`, reconstructs the tensors, and calls `_sync_lora_to_engine` which builds a `TensorLoRARequest` and calls `llm.llm_engine.add_lora(request)`. The hijack then picks it up and installs it as the active adapter.

### `set_scales` is a no-op

```python
# vllm_lora.py:321-322
elif op == "set_scales":
    reply = {"ok": True}  # no-op for LoRA
```

This is correct given the fusion: once the scales are baked into A, there's nothing the vLLM side can do to pull them apart again. All three eval modes (`both`, `retain_only`, `forget_only`) generate from the same fused adapter and return identical outputs when vLLM is the generator. Retain-only / forget-only evaluation for DualLoRA runs must be done on the HF side or with a custom re-extraction.

---

## 6. vLLM DualMLP integration (`vllm_mlp_adapter.py`, `vllm_server.py`, `vllm_client.py`)

The MLP path is much more involved because vLLM has no native support for "wrap an entire MLP block with extra sub-networks." The implementation takes over vLLM's LoRA infrastructure for scheduling and per-token routing, but bypasses its actual LoRA math.

### `VLLMDualMLPAdapter` (`vllm_mlp_adapter.py:122-303`)

```python
class VLLMDualMLPAdapter(nn.Module):
    def __init__(self, base_mlp, hidden_size, max_adapters, retain_neurons, forget_neurons):
        self.base_mlp = base_mlp
        # Stacked weight buffers in Punica format: (max_loras, 1, out, in)
        self.retain_gate_stacked = nn.Parameter(
            torch.zeros(max_adapters, 1, retain_neurons, hidden_size, ...),
            requires_grad=False)
        self.retain_up_stacked   = ...
        self.retain_down_stacked = ...
        self.forget_gate_stacked = ...
        self.forget_up_stacked   = ...
        self.forget_down_stacked = ...
        # Per-experiment scales (GPU-resident, read per-token in forward)
        self.scales = torch.ones(max_adapters, 2, device=device, dtype=dtype)
```

Why stacked? vLLM can generate for multiple "experiments" (= multiple adapter slots) in the same batch, routing tokens to their slot via `PunicaWrapper.token_lora_indices`. By preallocating `(max_adapters, 1, out, in)` buffers, we can use Punica's `add_shrink` and `add_expand` Triton kernels to perform per-token slot-indexed matmuls — the exact same primitive vLLM uses for native LoRA routing.

### Forward with per-token scales (`vllm_mlp_adapter.py:276-303`)

```python
def forward(self, x):
    base_out = self.base_mlp(x)
    if not hasattr(self, 'punica_wrapper'):
        return base_out

    token_slots = self.punica_wrapper.token_lora_indices[:x.shape[0]]  # (T,)

    if self.retain_gate_stacked is not None:
        retain_out = self._adapter_swiglu(
            x, self.retain_gate_stacked, self.retain_up_stacked, self.retain_down_stacked)
        retain_scales = self.scales[token_slots, 0].unsqueeze(1)  # (T, 1)
        base_out = base_out + retain_out * retain_scales

    if self.forget_gate_stacked is not None:
        forget_out = self._adapter_swiglu(
            x, self.forget_gate_stacked, self.forget_up_stacked, self.forget_down_stacked)
        forget_scales = self.scales[token_slots, 1].unsqueeze(1)  # (T, 1)
        base_out = base_out + forget_out * forget_scales

    return base_out
```

`_adapter_swiglu(x, gate, up, down)` uses Punica's `add_shrink` for `x → {gate_out, up_out}` (routed by token slot), computes `SiLU(gate_out) * up_out` with plain torch ops, then uses `add_expand` for the back-projection.

The crucial line is `retain_scales = self.scales[token_slots, 0].unsqueeze(1)`: each token reads its own scale based on which experiment slot it belongs to. This is what makes `set_scales` **actually work** for the MLP path — you just write into the `self.scales` GPU tensor and the very next forward pass respects the new values.

### Disabling vLLM's real LoRA wrapping

We want vLLM's LoRA system for per-token routing (`token_lora_indices`) and adapter lifecycle (activate/deactivate/stacked weights), but we do **not** want it wrapping any actual Linear layers with LoRA — the "LoRA" here is a sham whose only purpose is to carry routing metadata.

```python
# vllm_mlp_adapter.py:74-99
def _prevent_lora_module_wrapping():
    LoRAModelManager._match_target_modules = lambda self, module_name: False
    LRUCacheWorkerLoRAManager.add_dummy_lora = lambda self, lora_request, rank: False
    # _create_merged_loras_inplace assumes standard LoRA tensor format;
    # skip it for our MLP adapter entries (which use lists instead of merged tensors)
    def _safe_create_merged(self, lora_model):
        if not lora_model.loras:
            return
        first = next(iter(lora_model.loras.values()))
        if isinstance(first.lora_a, list):
            return
        return _orig_create_merged(self, lora_model)
    LoRAModelManager._create_merged_loras_inplace = _safe_create_merged
```

And we intercept `_load_adapter` so dummy MLP adapter requests return an empty `LoRAModel` instead of trying to read files from disk:

```python
# vllm_mlp_adapter.py:46-72
def _install_dummy_adapter_loader():
    def _hijacked(self, lora_request):
        if lora_request.lora_name.startswith(("mlp_exp_", "warmup_")):
            return LoRAModel(lora_request.lora_int_id, rank=1, loras={})
        return _orig(self, lora_request)
    LRUCacheWorkerLoRAManager._load_adapter = _hijacked
```

### Engine setup and MLP injection (`create_engine`, lines 549–659)

```python
def create_engine(model_name, max_experiments, retain_neurons, forget_neurons,
                  gpu_memory_utilization, dtype, layer_start, layer_end, layer_stride):
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    _prevent_lora_module_wrapping()
    _install_dummy_adapter_loader()

    def _mlp_hook(model):
        lora_manager = model.lora_manager
        punica_wrapper = lora_manager.punica_wrapper_mapping[DEFAULT_LANGUAGE_WRAPPER_KEY]
        hidden_size = model.config.hidden_size
        for layer_idx, path, parent, base_mlp in find_mlp_modules(model, layer_indices):
            adapter = VLLMDualMLPAdapter(base_mlp, hidden_size, max_experiments,
                                         retain_neurons, forget_neurons)
            setattr(parent, "mlp", adapter)
            lora_manager.register_module(path, adapter)
            adapter.set_mapping(punica_wrapper)

    LoRAModelManager._post_create_module_hooks.append(_mlp_hook)
    try:
        llm = LLM(model=model_name,
                  enforce_eager=True,          # torch.compile caches can't handle dynamic adapters
                  enable_lora=True,
                  max_loras=max_experiments,
                  max_lora_rank=8,
                  gpu_memory_utilization=gpu_memory_utilization,
                  dtype=dtype)
    finally:
        LoRAModelManager._post_create_module_hooks.remove(_mlp_hook)
```

Two things to notice:

1. **`_post_create_module_hooks`** is a list that doesn't exist in stock vLLM — it's added by the vLLM source patch (see §7). It fires after the LoRA manager is initialized but before CUDA graphs / profiling, which is exactly the window we need to swap `LlamaMLP` → `VLLMDualMLPAdapter` and register each new adapter with the LoRA manager.

2. **`register_module(path, adapter)`** makes the LoRA manager aware of each adapter so that activate/deactivate calls can find it and call `set_adapter_weights()` on it. This is how weight sync is plumbed through vLLM's normal "activate this LoRA" flow.

### Weight sync (`vllm_client.py:31-50` + `vllm_mlp_adapter.py:345-378`)

On the client side, `_extract_weights_from_model(model)` walks the HF model and serializes each `DualMLPAdapter`'s six `Linear.weight` tensors (`gate_retain`, `up_retain`, `down_retain`, `gate_forget`, `up_forget`, `down_forget`) into raw float bytes. These are sent via msgpack.

On the server side, the bytes are repacked as a `LoRAModel` where each layer's `LoRALayerWeights` has:

```python
lora_a = [gate_retain, up_retain, down_retain]   # hijacking the field
lora_b = [gate_forget, up_forget, down_forget]
```

The LoRA manager's `activate_adapter()` flow then calls `VLLMDualMLPAdapter.set_adapter_weights(index, weights)`, which unpacks the lists and copies each tensor into the corresponding slot of the stacked buffer:

```python
def set_adapter_weights(self, index, weights):
    self.reset_lora(index)
    if weights.lora_a is not None and self.retain_gate_stacked is not None:
        gate_r, up_r, down_r = weights.lora_a
        self.retain_gate_stacked.data[index, 0, :gate_r.shape[0], :gate_r.shape[1]].copy_(
            gate_r, non_blocking=True)
        # ... up_retain, down_retain, and all three forget tensors
```

### `set_scales` actually works

```python
# vllm_server.py
def handle_set_scales(self, msg):
    self.mgr.set_scales(msg["experiment_id"], msg["retain_scale"], msg["forget_scale"])
    return {"ok": True}
```

`mgr.set_scales(eid, r, f)` writes `self.scales[eid] = [r, f]` on each `VLLMDualMLPAdapter`. The very next `forward` call picks up the new values through `token_slots`-indexed reads.

This is why eval mode switching works for MLP but not LoRA: per-token scales are kept as a live GPU tensor that the adapter's forward pass reads every step.

---

## 7. The vLLM source patch

vLLM is installed as a pinned dependency. The repo carries a patch that adds exactly one thing to vLLM core: the `_post_create_module_hooks` list on `LoRAModelManager`. This is called at the end of LoRA manager initialization, after the base model is loaded and the native LoRA wrapping pass has run, but before the engine warms up CUDA graphs.

Without this hook, there's no clean place to swap `LlamaMLP` → `VLLMDualMLPAdapter`:

- Doing it before LoRA init means the LoRA manager doesn't know about our adapter.
- Doing it after engine init means CUDA graphs have already been captured against the original MLP.

The patch is small (single hook point) but essential. Application is via `apply_vllm_patch.py` (or equivalent install script) which locates the installed vLLM package and applies the diff. `CUSTOM_VLLM_ADAPTER_IMPLEMENTATION.md` in this repo contains the historical design notes.

---

## 8. Training-loop vLLM integration (`train.py`)

The per-rollout vLLM sync is handled in `SampleGRPOTrainer._generate_single_turn()` (around lines 415–494):

```python
def _generate_single_turn(self, prompts):
    client = self._vllm_client
    eid = self._vllm_experiment_id

    # Wake engine if sleeping
    if hasattr(client, 'wake_up'):
        torch.cuda.empty_cache()
        client.wake_up()

    # Push fresh weights and set scales
    retain_s = 1.0
    forget_s = 0.0 if self._is_coherence_rollout else 1.0
    client.update_weights_from_model(eid, self.model)
    client.set_scales(eid, retain_s, forget_s)   # no-op for LoRA, real for MLP

    # Generate
    gen_result = client.generate(eid, prompt_ids_list, 1,
                                 self.args.temperature, self.max_completion_length,
                                 top_k=..., top_p=...)

    if hasattr(client, 'sleep'):
        client.sleep(level=1)
    return ...
```

The `client.set_scales(eid, 1.0, 1.0)` call immediately before weight extraction (and eval mode resets) ensures the HF-side adapter scales are also reset to `(1, 1)` before `update_weights_from_model` extracts them — otherwise, for the LoRA path, a stale eval mode could cause incorrect scales to be baked into the A matrices. This was a real bug that was fixed earlier in the project.

For the MLP path, `client.set_scales` writes the GPU scales tensor directly and then `update_weights_from_model` writes the new weights into the stacked buffers — both are in effect for the next `client.generate` call.

---

## 9. Eval-mode loop (`eval_utils.py:207-319`)

```python
def eval_gradient_routing(model, tokenizer, reward_fns, ..., vllm_client=None, experiment_id=None):
    modes = [("both", 1.0, 1.0), ("retain_only", 1.0, 0.0), ("forget_only", 0.0, 1.0)]
    model.eval()
    try:
        if use_vllm:
            vllm_client.wake_up()
            vllm_client.update_weights_from_model(experiment_id, model)  # sync ONCE up front

        for mode_name, retain_scale, forget_scale in modes:
            set_scales(model, retain_scale, forget_scale)                 # HF side
            if use_vllm:
                vllm_client.set_scales(experiment_id, retain_scale, forget_scale)  # server side
                samples = _generate_via_vllm(...)
            else:
                samples = generate_from_model(model, ...)
            # ... compute rewards, diversity ...
    finally:
        set_scales(model, 1.0, 1.0)
        if use_vllm:
            vllm_client.set_scales(experiment_id, 1.0, 1.0)
            vllm_client.sleep(level=1)
```

For the MLP path this works as intended — the three modes give three different sets of metrics. For the LoRA path, `vllm_client.set_scales` is a no-op and all three modes actually generate from the same fused adapter; the numbers you see for retain-only / forget-only during in-training eval are meaningless for LoRA runs. The only honest way to get retain-only / forget-only metrics for LoRA is to run eval on the HF model directly (which does honor `set_scales`) or to rebuild the vLLM adapter with one side zeroed out before sync.

---

## 10. Putting it together

```
                            Training step
─────────────────────────────────────────────────────────────────────
 generate_and_score_completions()
   │
   ├─► RH detector → is_rh tensor in batch
   │
   ▼
 _generate_single_turn()
   ├─► client.wake_up()
   ├─► set_scales(model, 1.0, 1.0) [or (1,0) if coherence]
   ├─► client.update_weights_from_model(eid, model)   ── HF → vLLM weight sync
   ├─► client.set_scales(eid, 1.0, 1.0)               ── no-op for LoRA, live for MLP
   ├─► client.generate(eid, prompts, ...)             ── rollout
   └─► client.sleep(level=1)
   │
   ▼
 training_step(inputs + is_rh)
   ├─► Pass 1 (good): optional forget-side hooks, backward, remove
   ├─► Pass 2 (bad):  always retain-side hooks, backward, remove
   └─► optimizer.step()
```

For LoRA runs, weight sync is lossy in a specific sense: the inference-scale knobs are fused into the synced tensors, so the vLLM engine cannot subsequently undo them without a re-sync. For MLP runs, weight sync and scale-setting are independent — weights live in the stacked buffers, scales live in their own GPU tensor, and the forward pass combines them per-token.

## 11. Notable files

| File | What it contains |
|---|---|
| `gradient_routing.py` | Core `DualLoRALinear`, `DualMLPAdapter`, discovery helpers, `apply_dual_lora`, `apply_dual_mlp`, `set_scales`, `collect_routing_params` |
| `train.py` | `SampleGRPOTrainer`, gradient-routing hooks, rollout vLLM sync, argparse glue, LORA_PRESETS / MLP_PRESETS |
| `vllm_lora.py` | DualLoRA → vLLM LoRA bridge: `_extract_dual_lora_tensors`, `TensorLoRARequest`, monkey-patch, `VLLMLoRAServer`/`VLLMLoRAClient` |
| `vllm_mlp_adapter.py` | Custom MLP adapter kind for vLLM: `VLLMDualMLPAdapter`, stacked buffers, Punica shrink/expand, monkey-patches, `create_engine` with `_post_create_module_hooks` |
| `vllm_server.py` | ZMQ server for the MLP path (`VLLMServer`, register / update_weights / generate / set_scales / sleep handlers) |
| `vllm_client.py` | ZMQ client for the MLP path, weight serialization helpers |
| `vllm_utils.py` | `MLP_PRESETS`, `flatten_vllm_outputs`, shared helpers |
| `eval_utils.py` | `eval_gradient_routing` mode loop, HF / vLLM generation helpers |
| `CUSTOM_VLLM_ADAPTER_IMPLEMENTATION.md` | Historical design notes for the custom-adapter vLLM patch |

## 12. Known limitations

1. **LoRA + vLLM cannot do runtime ablation.** `set_scales` is a no-op on the server, so in-training retain_only / forget_only eval metrics are invalid for LoRA runs. Workaround: run eval on the HF model side, or re-sync with one side pre-zeroed.
2. **vLLM LoRA requires uniform combined rank across all modules.** `_extract_dual_lora_tensors` asserts `r_total == combined_rank` for every wrapped module. Asymmetric per-module ranks (e.g. retain-only on some layers, forget-only on others) work only if the sums happen to match.
3. **MLP path requires `enforce_eager=True`.** torch.compile cache invalidation when stacked adapter tensors change between experiments makes it incompatible.
4. **MLP weight sync is heavier than LoRA.** Six tensors per wrapped MLP block vs two per wrapped Linear, plus the per-slot copy into `(max_adapters, 1, out, in)` buffers.
5. **The MLP path depends on the vLLM source patch.** Installing or upgrading vLLM requires re-applying it.
