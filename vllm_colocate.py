"""In-process vLLM engine with full-model weight sync and optional MLP adapter.

Creates a vLLM LLM in the training process (colocate mode) and syncs model
parameters from the training model before each generation step. Uses
VLLM_ENABLE_V1_MULTIPROCESSING=0 for direct model access (same pattern as
vllm_lora.py).

Two operating modes:
- Plain (no adapter): full base-weight sync. Used for adapter_type='none' or
  for hybrid modes where the base model is the retain side and there is no
  separate adapter to sync.
- MLP-adapter-aware: engine is built via vllm_mlp_adapter.create_engine, which
  injects VLLMDualMLPAdapter modules. update_weights_from_model can then sync
  both (a) base weights via load_weights and (b) adapter weights via the
  adapter manager. The base sync is gated by sync_base so adapter-only modes
  (base frozen) can skip it.

Usage:
    # Plain colocate (adapter_type=none)
    client = VLLMColocateClient("HuggingFaceTB/SmolLM2-135M-Instruct")

    # Colocate + MLP adapter (adapter_type=mlp, hybrid retain_source)
    client = VLLMColocateClient(
        "Qwen/Qwen3-4B",
        adapter_type="mlp",
        retain_neurons=0, forget_neurons=64,
        sync_base=True,  # base is unfrozen (retain side)
    )
"""

import os
import time

import torch


class VLLMColocateClient:
    """In-process vLLM client with full-model weight sync and optional adapter.

    Duck-typed replacement for VLLMClient — same interface (register, generate,
    update_weights_from_model, set_scales, shutdown).
    """

    def __init__(self, model_name, gpu_memory_utilization=0.05, dtype="bfloat16",
                 adapter_type=None, retain_neurons=0, forget_neurons=0,
                 layer_start=0.0, layer_end=1.0, layer_stride=1,
                 max_experiments=1, sync_base=True):
        # Disable vLLM subprocess so we can access the model directly
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
        os.environ.setdefault("VLLM_LOGGING_LEVEL", "ERROR")
        os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"

        self._sync_base = sync_base
        self._adapter_type = adapter_type

        if adapter_type == "mlp":
            # Adapter-aware engine: create_engine injects VLLMDualMLPAdapter
            # wrappers via a post-create hook and returns the adapter manager.
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
            print(f"[vLLM colocate+MLP] Engine ready in {time.time() - t0:.1f}s "
                  f"(retain_neurons={retain_neurons}, forget_neurons={forget_neurons}, "
                  f"sync_base={sync_base})")
        elif adapter_type in (None, "none"):
            from vllm import LLM
            t0 = time.time()
            self.llm = LLM(
                model=model_name,
                enforce_eager=True,
                dtype=dtype,
                gpu_memory_utilization=gpu_memory_utilization,
            )
            self.mgr = None
            print(f"[vLLM colocate] Engine ready in {time.time() - t0:.1f}s")
        else:
            raise NotImplementedError(
                f"VLLMColocateClient: adapter_type={adapter_type!r} not supported. "
                "Supported: None/'none' (no adapter), 'mlp'. "
                "For LoRA, use vllm_lora.VLLMLoRAClient or the spawn path."
            )

        # Direct reference to the model inside the in-process engine
        self._vllm_model = (
            self.llm.llm_engine.model_executor.driver_worker.model_runner.model
        )

    def register(self):
        """Return a dummy experiment ID (single-experiment, no routing)."""
        return 1

    def update_weights_from_model(self, experiment_id, training_model):
        """Copy training model weights into the vLLM model.

        Two syncs (either may be skipped):
          1. Base-weight sync via vllm_model.load_weights (gated by sync_base).
             Filters out adapter params by name (*_retain, *_forget) so only
             base params flow through; vLLM's load_weights handles fused name
             mapping internally (q/k/v → qkv_proj, gate/up → gate_up_proj).
          2. Adapter sync via mgr.update_from_training_model (if adapter mgr
             is present).
        """
        if self._sync_base:
            base_pairs = [
                (name, param.data)
                for name, param in training_model.named_parameters()
                if "_retain" not in name and "_forget" not in name
            ]
            self._vllm_model.load_weights(base_pairs)
            self.llm.reset_prefix_cache()
        if self.mgr is not None:
            self.mgr.update_from_training_model(experiment_id, training_model)

    def generate(self, experiment_id, prompt_ids, n, temperature, max_tokens,
                 top_k=50, top_p=1.0, return_logprobs=False):
        """Generate completions, return (comp_texts, comp_ids, prompt_ids[, logprobs]).

        With an adapter manager present, routes through mgr.generate so each
        prompt is tagged with the active adapter slot. Without one, dispatches
        directly to self.llm.generate.
        """
        from vllm import SamplingParams, TokensPrompt
        from vllm_utils import flatten_vllm_outputs

        prompts = [TokensPrompt(prompt_token_ids=list(p)) for p in prompt_ids]
        sp = SamplingParams(
            n=n,
            temperature=temperature,
            max_tokens=max_tokens,
            top_k=top_k if top_k > 0 else -1,
            top_p=top_p,
            logprobs=0 if return_logprobs else None,
        )

        if self.mgr is not None:
            if return_logprobs:
                raise NotImplementedError(
                    "VLLMColocateClient (adapter mode) does not surface per-token logprobs — "
                    "VLLMAdapterManager.generate doesn't plumb them through. Disable "
                    "--vllm_importance_sampling when running with adapter_type='mlp' colocate."
                )
            experiment_ids = [experiment_id] * len(prompts)
            outputs = self.mgr.generate(prompts, experiment_ids, sp)
        else:
            outputs = self.llm.generate(prompts, sp, use_tqdm=False)
        comp_texts, comp_ids, prompt_ids_out, _ = flatten_vllm_outputs(outputs)

        result = (comp_texts, comp_ids, prompt_ids_out)

        if return_logprobs:
            all_logprobs = []
            for req in outputs:
                for comp in req.outputs:
                    token_logprobs = []
                    for i, lp_dict in enumerate(comp.logprobs):
                        tid = comp.token_ids[i]
                        entry = lp_dict.get(tid)
                        token_logprobs.append(entry.logprob if entry is not None else 0.0)
                    all_logprobs.append(token_logprobs)
            result = result + (all_logprobs,)

        return result

    def set_scales(self, experiment_id, retain_scale, forget_scale):
        """Set per-experiment retain/forget scales (if an adapter is attached)."""
        if self.mgr is not None:
            self.mgr.set_scales(experiment_id, retain_scale, forget_scale)
        # No-op for plain (no-adapter) mode — full-param models have no scales.

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
