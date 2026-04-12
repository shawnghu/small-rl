"""In-process vLLM engine with full-base-weight sync AND MLP forget-adapter sync.

Used for ``adapter_type="full_mlp_forget"``: the training model is full-parameter
trainable on the base, plus a ``DualMLPAdapter`` forget-only sub-network on each
MLP block. Rollout generation must see both the updated base weights AND the
forget adapter output.

This client composes the existing pieces:
- ``vllm_mlp_adapter.create_engine(...)`` — creates an in-process LLM with
  ``enable_lora=True``, the monkey-patches that prevent vLLM from auto-wrapping
  base Linears, and a ``_post_create_module_hooks`` callback that replaces each
  ``model.layers.N.mlp`` with a ``VLLMDualMLPAdapter``. Returns ``(llm, mgr)``
  where ``mgr`` is a ``VLLMAdapterManager``.
- ``vllm_colocate.VLLMColocateClient.update_weights_from_model`` — walks HF
  ``named_parameters()`` and calls ``vllm_model.load_weights(pairs)`` to sync
  full base weights. vLLM's ``load_weights`` handles q/k/v → qkv_proj and
  gate/up → gate_up_proj fusion internally.

After ``apply_dual_mlp`` wraps each MLP, the HF training model has:
  model.layers.N.mlp.base_mlp.{gate,up,down}_proj.weight   ← frozen base Linears,
                                                              now unfrozen as retain
  model.layers.N.mlp.{gate,up,down}_forget.weight           ← forget adapter params

The vLLM model (after injection) has:
  model.layers.N.mlp.base_mlp.{gate,up,down}_proj.weight   ← base params
  model.layers.N.mlp.retain_gate_stacked (etc.)             ← Punica-format stacked buffers

So base-layer names line up on both sides, and ``load_weights(base_pairs)`` maps
HF params into vLLM's base model. Forget-adapter params are synced separately via
``mgr.update_from_training_model(...)`` which packs them into a fake LoRA request
and drives ``VLLMDualMLPAdapter.set_adapter_weights(...)`` on the vLLM side.

Usage:
    client = VLLMColocateMLPClient(model_name, gpu_memory_utilization=0.2,
                                    dtype="bfloat16", retain_neurons=0,
                                    forget_neurons=64, ...)
    eid = client.register()
    client.update_weights_from_model(eid, training_model)
    texts, ids, prompts = client.generate(eid, prompt_ids, n=1, ...)
"""

import os
import time

import torch


class VLLMColocateMLPClient:
    """In-process vLLM client for full-param base + MLP forget adapter.

    Duck-typed replacement for VLLMClient / VLLMColocateClient — same interface
    (register, generate, update_weights_from_model, set_scales, shutdown).
    """

    def __init__(self, model_name, gpu_memory_utilization=0.2, dtype="bfloat16",
                 retain_neurons=0, forget_neurons=64,
                 layer_start=0.0, layer_end=1.0, layer_stride=1,
                 max_experiments=1):
        # Set vLLM env vars BEFORE importing anything that transitively imports
        # `vllm` (create_engine → `from vllm import LLM`). If vLLM is already
        # imported by the time these are set, the subprocess boundary is
        # already committed. Matches the pattern in vllm_colocate.py.
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
        os.environ.setdefault("VLLM_LOGGING_LEVEL", "ERROR")
        os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"

        # In-process engine: create via vllm_mlp_adapter.create_engine, which
        # installs the monkey-patches, the post-create module hook, builds LLM
        # with enable_lora=True + _prevent_lora_module_wrapping, and returns a
        # ready-to-use VLLMAdapterManager.
        from vllm_mlp_adapter import create_engine

        t0 = time.time()
        self.llm, self.mgr = create_engine(
            model_name=model_name,
            max_experiments=max_experiments,
            retain_neurons=retain_neurons,
            forget_neurons=forget_neurons,
            gpu_memory_utilization=gpu_memory_utilization,
            dtype=dtype,
            layer_start=layer_start,
            layer_end=layer_end,
            layer_stride=layer_stride,
        )
        print(f"[vLLM colocate+MLP] Engine ready in {time.time() - t0:.1f}s")

        # Direct reference to the vLLM model for base-weight sync via load_weights.
        # (Same path as VLLMColocateClient — the in-process engine exposes the
        # model_runner's underlying model.)
        self._vllm_model = (
            self.llm.llm_engine.model_executor.driver_worker.model_runner.model
        )

        self._max_experiments = max_experiments

    def register(self):
        """Return a dummy experiment ID (single-experiment)."""
        return 1

    def update_weights_from_model(self, experiment_id, training_model):
        """Sync full base weights + MLP forget-adapter weights to vLLM.

        Two separate syncs per call:
          1. Base weights via ``vllm_model.load_weights(pairs)`` — filters out
             forget-adapter params (identified by the ``_forget`` substring in
             the HF param name), passes everything else. vLLM's load_weights
             handles name fusion (q/k/v → qkv_proj, gate/up → gate_up_proj) and
             walks the HF name tree to match vLLM's param tree; the
             ``.mlp.base_mlp.*`` nesting matches on both sides because both the
             HF ``DualMLPAdapter`` and the vLLM ``VLLMDualMLPAdapter`` wrap the
             original MLP under a ``base_mlp`` attribute.
          2. Forget-adapter weights via ``mgr.update_from_training_model(...)``
             which walks ``DualMLPAdapter`` modules, extracts the forget side's
             six tensors, packs them into a fake LoRA request, and calls
             ``VLLMDualMLPAdapter.set_adapter_weights(...)`` on each vLLM-side
             adapter.
        """
        # 1. Base-weight sync — exclude forget-adapter params by name
        base_pairs = [
            (name, param.data)
            for name, param in training_model.named_parameters()
            if "_forget" not in name
        ]
        self._vllm_model.load_weights(base_pairs)
        self.llm.reset_prefix_cache()

        # 2. MLP forget-adapter sync through the LoRA-manager activation flow
        self.mgr.update_from_training_model(experiment_id, training_model)

    def set_scales(self, experiment_id, retain_scale, forget_scale):
        """Set per-experiment retain/forget scales on the vLLM-side adapters."""
        self.mgr.set_scales(experiment_id, retain_scale, forget_scale)

    def generate(self, experiment_id, prompt_ids, n, temperature, max_tokens,
                 top_k=50, top_p=1.0, return_logprobs=False):
        """Generate completions via the adapter-aware engine.

        Returns (comp_texts, comp_ids, prompt_ids) — same shape as
        VLLMColocateClient.generate.
        """
        if return_logprobs:
            raise NotImplementedError(
                "VLLMColocateMLPClient does not support return_logprobs yet — "
                "the underlying VLLMAdapterManager.generate doesn't surface per-token "
                "logprobs. Disable --vllm_importance_sampling for full_mlp_forget runs."
            )

        from vllm import SamplingParams, TokensPrompt
        from vllm_utils import flatten_vllm_outputs

        prompts = [TokensPrompt(prompt_token_ids=list(p)) for p in prompt_ids]
        sp = SamplingParams(
            n=n,
            temperature=temperature,
            max_tokens=max_tokens,
            top_k=top_k if top_k > 0 else -1,
            top_p=top_p,
        )
        # One experiment_id per prompt; in our use case it's always the same.
        experiment_ids = [experiment_id] * len(prompts)

        outputs = self.mgr.generate(prompts, experiment_ids, sp)
        comp_texts, comp_ids, prompt_ids_out, _ = flatten_vllm_outputs(outputs)
        return comp_texts, comp_ids, prompt_ids_out

    def release(self, experiment_id):
        """No-op: single-experiment, nothing to release."""
        pass

    def shutdown(self):
        """Clean up the vLLM engine."""
        del self.llm
        self.llm = None
        self.mgr = None
        self._vllm_model = None
        torch.cuda.empty_cache()

    def close(self):
        """Alias for shutdown."""
        self.shutdown()
