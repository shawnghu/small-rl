# Per-Sample Gradient & Activation Diagnostic

Collects the full distribution of **per-sample** gradient norms for the four
combinations of (retain / forget adapter params) × (retain / forget samples),
**per layer**, during the RL loop. Inspired by
[Selective Gradient Masking](https://alignment.anthropic.com/2025/selective-gradient-masking/),
whose density plots show that forget weights take larger gradients on forget
data than retain weights do.

In the same pass it also collects per-sample **activation** norms — the
RMS-over-tokens of each adapter's output contribution to the residual stream
(`retain_out` / `forget_out`), the forward-pass analog of the gradient 2×2
("activations downstream of the retain / forget parameters"). The viewer toggles
between the two metrics; both share the same per-sample × per-layer schema.

This is the *counterfactual* the routing mask normally hides: in training, the
forget pass zeroes retain-param grads on bad samples (`train.py`'s
`register_hook(lambda g: torch.zeros_like(g))`), so two of the four quadrants
never appear in `.grad`. The diagnostic runs its own **unmasked** pass to
measure all four.

## What it measures

For each diagnostic step it records, per sample `i`, per layer `ℓ`, per role
`r ∈ {retain, forget}`, the L2 norm of that sample's gradient contribution to
role `r`'s params in layer `ℓ`. Everything else is derived:

- **The four 2×2 densities** = per-sample norms bucketed by role (param) and by
  `is_rh` (sample: 0 = retain, 1 = forget/hack).
- **Whole-model per-sample norm** = `sqrt(Σ_ℓ norm²)` (params are disjoint
  across layers/roles).
- **Per-layer aggregate** `||.grad||` per role (all samples) — read directly
  from the accumulated `.grad`, and used as a cross-check anchor.

## How it works

A single extra **packed (liger) forward/backward over the full rollout batch**,
with no routing mask and no advantage swapping. Per-sample gradients are
recovered without reducing along the batch axis via `PerSampleGradCapture`
(`gradient_routing.py`): for a linear map `y = x·Wᵀ`, sample `i`'s gradient is
`gᵢᵀ·xᵢ` summed over its token span, where `x` is the layer input and
`g = ∂L/∂y` is the output gradient — both captured by forward/backward hooks on
the adapter modules.

- **DualMLPAdapter** uses real `nn.Linear` submodules, hooked directly (the
  captured `g` already includes the branch scale).
- **DualLoRALinear** stores bare parameters, so the module is hooked and both
  LoRA matrices' gradients are reconstructed from `x` and `g`, multiplying by
  the forward scale `c = (alpha/rank)·adapter_scale`.

**Activations** are captured in the same forward (no backward needed): for each
adapter, the per-token output contribution to the residual stream
(`retain_out` / `forget_out`) is reduced to a per-sample RMS-over-tokens norm.
MLP adapters hook the `down_{role}` submodule output (× the parent's role
scale); LoRA recomputes the low-rank delta from the saved input. Unlike grads
(which accumulate across tokens, hence summed), activations are length-normalized
(RMS), so the norm is a per-token magnitude independent of sequence length.

Token spans come from the packed path's `seq_boundaries` (the same segmentation
`_packed_compute_loss` uses). Because packing strips padding, every token in a
span is real — no masking needed. The diagnostic:

1. runs on a clean grad state, swaps `self._metrics` to a scratch dict so the
   loss path's kl/entropy logging doesn't pollute training, and saves/restores
   the sample-text stash;
2. backs each microbatch with `scale = n_mb / n_total` so the captured
   per-sample grads sum exactly to the full-batch `.grad` (uniform `1/n_total`
   weight per sample);
3. zeroes grads on exit, so it never perturbs the real training step.

**Packed-path only** (`use_liger_kernel`): asserted loudly. Measuring on the
same kernel training uses means the captured gradient is faithful to the actual
training gradient. Fires for any run with **dual adapters** (non-empty forget
params) and an `is_rh` label on **non-coherence** rollouts — i.e. GR runs
(`routing_mode=classic/exclusive`) *and* reward-penalty / filter baselines
(`routing_mode=none`, where both adapters are trained unmasked, so the
diagnostic captures exactly the gradient flow GR would otherwise mask — the
natural baseline-vs-GR comparison). Adapter dropout is assumed inactive.

The exact capture math is validated in
`tests/test_per_sample_grad_capture.py` (no-attention towers give an exact
autograd ground truth for both adapter types; an HF Llama case checks hook
integration). The driver also logs a live `grad_check.max_triangle_ratio`
(`||.grad|| ≤ Σ_i ||gradᵢ||`) as a segmentation tripwire.

## Usage

```
CUDA_VISIBLE_DEVICES=0 .venv/bin/python train.py --config <config> --routing_mode classic --grad_diag_every 100 ...
```

`--grad_diag_every N`: cadence in steps. Default `None` mirrors `--eval_every`;
`0` disables. The default lives in exactly one place (the `None → eval_every`
resolution in train.py).

## Output

- **`{output_dir}/grad_diag.jsonl`** — one record per diagnostic step:
  ```
  { step, samples_seen, n_samples, layers:[...], is_rh:[0/1 per sample],
    per_sample:      { retain: [n_layers][n_samples], forget: [...] },  # grad
    whole_model:     { retain: [n_samples], forget: [n_samples] },      # grad
    act_per_sample:  { retain: [n_layers][n_samples], forget: [...] },  # activation
    act_whole_model: { retain: [n_samples], forget: [n_samples] },      # activation
    aggregate_grad_norm: { retain: [n_layers], forget: [n_layers] },  # ||.grad||, all samples
    grad_check:   { max_triangle_ratio } }
  ```
- **`{output_dir}/grad_diag.html`** — interactive viewer (auto-generated at end
  of training; regenerate manually with
  `python tools/gen_grad_diag_html.py {output_dir}/`). **Metric toggle**
  (gradient norm / activation norm), step slider, layer selector ("all layers /
  whole-model" + each layer), the by-data-type histogram panels (forget-param vs
  retain-param), and per-layer median panels.
- **wandb** `grad_diag/{grad,act}_{retain,forget}_param_on_{retain,forget}_samples`
  — the eight whole-model mean-per-sample-norm scalars (x-axis `samples_seen`).

## Cost / caveats

- One extra full forward/backward per diagnostic step (run at `eval_every`
  cadence by default), plus capture transiently holds adapter-input activations
  across layers (~a second copy) — roughly doubles that step's peak memory. For
  large models, lower `--max_tokens_per_microbatch` or raise
  `--grad_diag_every`.
- The per-layer aggregate `||.grad||` (norm of the summed gradient) and the
  mean per-sample norm are different quantities — the line plot shows both; the
  former reflects cross-sample cancellation, the latter the typical per-sample
  magnitude.
