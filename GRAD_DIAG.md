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

- **`{output_dir}/grad_diag.jsonl`** — one record per diagnostic step;
  **per-sample-keyed**: every array aligns by sample index, and each sample
  carries its labels + decoded text so norms can be tied back to the sample:
  ```
  { step, samples_seen, n_samples, layers:[...],
    is_rh:[0/1 per sample], detectable:[0/1 per sample],   # 1 = monitored
    prompt:[text per sample], completion:[text per sample],
    per_sample:      { retain: [n_layers][n_samples], forget: [...] },  # grad norm
    whole_model:     { retain: [n_samples], forget: [...] },            # grad norm
    act_per_sample:  { retain: [n_layers][n_samples], forget: [...] },  # activation norm
    act_whole_model: { retain: [n_samples], forget: [...] },            # activation norm
    dot_per_sample:  { retain: [n_layers][n_samples], forget: [...] },  # SIGNED <grad,weight>
    dot_whole_model: { retain: [n_samples], forget: [...] },            # signed sum over layers
    aggregate_grad_norm: { retain: [n_layers], forget: [n_layers] },  # ||.grad||, all samples
    grad_check:   { max_triangle_ratio } }
  ```
  The **dot product** `<grad, weight>` is the per-sample param gradient dotted
  with the current weights (signed; how much the update scales each adapter
  along its current direction). `detectable` is the per-sample monitored flag
  (the penalty-conditional subset), injected into the rollout batch alongside
  `is_rh`.
- **`{output_dir}/grad_diag.html`** — per-run viewer (auto-generated at end of
  training): metric toggle, step slider, layer selector, by-data-type panels.
- **`tools/gen_separability_html.py <dir>...`** → `separability/`:
  - **`separability_dist.html`** — detailed per-env viewer. Takes one or more
    sweep dirs and groups runs by **(env, condition)** (GR / RP / do-nothing,
    classified from `run_config.yaml`); each condition is a stacked block
    showing whatever data it has: the grid (gradient / activation / **dot**
    rows × retain/forget param) overlaying retain (blue) / forget-detectable
    (light red) / forget-undetectable (dark red) samples, overview-style
    training curves, and a **joint-distribution** scatter of retain-param vs
    forget-param magnitude per sample (log-log, Pearson r) — how correlated the
    two are. (dot uses signed linear-x bins; shown only when present.)
  - **`separability_allenvs.html`** — overview: every env's 2×2 grid at a glance
    in a 3-col layout, condition + layer selectors, one shared step slider.
- **wandb** `grad_diag/{grad,act,dot}_{retain,forget}_param_on_{retain,forget}_samples`
  — the twelve whole-model mean-per-sample scalars (x-axis `samples_seen`).

## Cost / caveats

- One extra full forward/backward per diagnostic step (run at `eval_every`
  cadence by default), plus capture transiently holds adapter-input activations
  across layers (~a second copy) — roughly doubles that step's peak memory. For
  large models, lower `--max_tokens_per_microbatch` or raise
  `--grad_diag_every`.
- The jsonl now stores decoded `prompt`+`completion` per sample (so norms can be
  tied to content). At ~512 samples × ~40 diag steps this is ~tens of MB/run —
  fine on disk. The sweep `separability_dist.html` embeds per-sample scatter
  points + per-layer histograms; with ~80 steps × 5 envs it can reach tens of MB
  (loads locally; subsample steps if it gets unwieldy).
- The per-layer aggregate `||.grad||` (norm of the summed gradient) and the
  mean per-sample norm are different quantities — the line plot shows both; the
  former reflects cross-sample cancellation, the latter the typical per-sample
  magnitude.
