"""E-checkpoint confirmation: batched persona group with SHARED reference-style init.

Single-run training never reseeds before adapter application, so all reference/
control seeds share ONE adapter init (drawn from the post-model-load RNG stream).
Batched mode gives each run its own seeded init — the suspected source of the
persona +0.02 eval-level offset. This driver monkeypatches apply_dual_mlp so all
N batched runs share a single unseeded-stream init (replicating the single-run
semantics exactly). If the resulting curves land at the reference level (~0.287
qa_correct), the offset is fully attributed to init diversity, not to the
batched training/eval machinery.

CUDA_VISIBLE_DEVICES=0 .venv/bin/python tools/e_confirm_shared_init.py
"""
import copy
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gradient_routing as gr

_orig_apply = gr.apply_dual_mlp


def _apply_shared_init(model, retain_neurons, forget_neurons,
                       layer_start=0.0, layer_end=1.0, layer_stride=1,
                       n_runs=1, run_seeds=None):
    """apply_dual_mlp variant: ONE unseeded-stream child per layer (exactly the
    single-run init), deepcopied to all runs."""
    if n_runs == 1:
        return _orig_apply(model, retain_neurons, forget_neurons,
                           layer_start, layer_end, layer_stride)
    for p in model.parameters():
        p.requires_grad = False
    num_layers = model.config.num_hidden_layers
    hidden_size = model.config.hidden_size
    layer_indices = gr.compute_layer_indices(num_layers, layer_start, layer_end, layer_stride)
    mlp_entries = gr.find_mlp_modules(model, layer_indices)
    modified = []
    for layer_idx, path, parent, base_mlp in mlp_entries:
        proto = gr.DualMLPAdapter(base_mlp, hidden_size, retain_neurons, forget_neurons)
        children = [proto] + [copy.deepcopy(proto) for _ in range(n_runs - 1)]
        for c in children[1:]:
            c._modules["base_mlp"] = base_mlp
        setattr(parent, "mlp", gr.MultiRunDualAdapter(base_mlp, children))
        modified.append(layer_idx)
    print(f"[shared-init] {len(modified)} layers wrapped with {n_runs} IDENTICAL adapter sets")
    return modified


def main():
    # Patch must be installed in the MAIN process only (train.py resolves
    # apply_dual_mlp lazily inside _run, so the module-attr patch is effective).
    gr.apply_dual_mlp = _apply_shared_init
    import train

    from sweeps.binary_dynamics_5seeds import runs as _orig_runs

    runs = []
    for r in _orig_runs:
        if "persona" not in r["run_name"]:
            continue
        r = dict(r)
        r["save_steps"] = 999999
        r["vllm_gpu_memory"] = 0.13
        r["vllm_spawn"] = True
        r["no_wandb"] = True
        r["run_name"] = r["run_name"] + "_sharedinit"
        r["output_dir"] = f"output/e_confirm_shared_init/s{r['seed']}"
        runs.append(r)
    assert len(runs) == 5

    train.train_main_batched(runs, group_output_dir="output/e_confirm_shared_init/group5")


if __name__ == "__main__":
    # Spawn-context children re-import this file as __main__'s source; the
    # guard keeps them from re-running the whole driver (RuntimeError bootstrap).
    main()
