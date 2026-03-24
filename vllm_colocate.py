"""In-process vLLM engine with full-model weight sync for adapter_type='none'.

Creates a vLLM LLM in the training process (colocate mode) and syncs all model
parameters from the training model before each generation step. Uses
VLLM_ENABLE_V1_MULTIPROCESSING=0 for direct model access (same pattern as
vllm_lora.py).

Usage:
    from vllm_colocate import VLLMColocateClient

    client = VLLMColocateClient("HuggingFaceTB/SmolLM2-135M-Instruct")
    eid = client.register()
    client.update_weights_from_model(eid, training_model)
    comp_texts, comp_ids, prompt_ids = client.generate(eid, prompt_ids_batch, ...)
"""

import os
import time

import torch


class VLLMColocateClient:
    """In-process vLLM client with full-model weight sync.

    Duck-typed replacement for VLLMClient — same interface (register, generate,
    update_weights_from_model, set_scales, shutdown).
    """

    def __init__(self, model_name, gpu_memory_utilization=0.05, dtype="float16"):
        # Disable vLLM subprocess so we can access the model directly
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
        os.environ.setdefault("VLLM_LOGGING_LEVEL", "ERROR")
        os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"

        from vllm import LLM
        t0 = time.time()
        self.llm = LLM(
            model=model_name,
            enforce_eager=True,
            dtype=dtype,
            gpu_memory_utilization=gpu_memory_utilization,
        )
        print(f"[vLLM colocate] Engine ready in {time.time() - t0:.1f}s")

        # Direct reference to the model inside the in-process engine
        self._vllm_model = (
            self.llm.llm_engine.model_executor.driver_worker.model_runner.model
        )

    def register(self):
        """Return a dummy experiment ID (single-experiment, no routing)."""
        return 1

    def update_weights_from_model(self, experiment_id, training_model):
        """Copy all training model weights into the vLLM model.

        vLLM fuses some HF weight matrices for efficiency (e.g. q/k/v_proj →
        qkv_proj, gate/up_proj → gate_up_proj). The model's load_weights()
        handles this mapping internally, so we pass all HF params through
        without filtering.
        """
        weight_pairs = [(name, param.data) for name, param in training_model.named_parameters()]
        self._vllm_model.load_weights(weight_pairs)
        self.llm.reset_prefix_cache()

    def generate(self, experiment_id, prompt_ids, n, temperature, max_tokens,
                 top_k=50, top_p=1.0, return_logprobs=False):
        """Generate completions, return (comp_texts, comp_ids, prompt_ids[, logprobs])."""
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
        """No-op: full-param models have no adapters to scale."""
        pass

    def release(self, experiment_id):
        """No-op: single-experiment, nothing to release."""
        pass

    def shutdown(self):
        """Clean up the vLLM engine."""
        del self.llm
        self.llm = None
        self._vllm_model = None
        torch.cuda.empty_cache()

    def close(self):
        """Alias for shutdown."""
        self.shutdown()
