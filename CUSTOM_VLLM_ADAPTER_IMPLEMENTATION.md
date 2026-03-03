# Custom vLLM MLP Adapter Implementation

## Objective

Run GRPO training with vLLM-accelerated generation for gradient routing experiments. The standard path (TRL's built-in vLLM support) is blocked by a version conflict: TRL requires transformers >= 5.2.0, but vLLM 0.16.0 pins transformers 4.57.6. Rather than wait for upstream alignment, we implement a standalone GRPO loop that uses vLLM for generation and a separate HF model for gradient computation.

A secondary objective is multi-experiment throughput: running N gradient routing experiments concurrently through a single shared vLLM engine, with per-experiment adapter isolation.

## Architecture: Two Models, One GPU

```
HF Model (float32, DualMLPAdapter)     vLLM Engine (bfloat16, VLLMDualMLPAdapter)
- Log prob computation                 - Batched generation
- Loss computation                     - Per-token adapter routing via LoRA indices
- Backprop + optimizer step            - Weight sync target
- Weight source ───sync──────────────→
```

Each training step: sync adapter weights HF→vLLM, generate completions via vLLM, compute log probs + backprop through HF model. The two models never share parameters directly — they live in separate dtype/framework contexts.

## Key Design Decision: Dummy LoRA Routing Trick

vLLM has no native "custom adapter" API. It does have per-request LoRA routing (`LoRARequest`), which labels each token with a `lora_int_id` via `PunicaWrapper.token_lora_indices`. We exploit this:

1. Register zero-weight rank-1 LoRA adapters (one per experiment slot). These do nothing computationally — they only exist to activate the routing infrastructure.
2. Replace each `LlamaMLP` with `VLLMDualMLPAdapter`, which reads `token_lora_indices` from the `PunicaWrapper` to route each token to the correct experiment's MLP adapter weights.
3. The dummy LoRA's `gate_up_proj` gets LoRA-wrapped by vLLM, which gives us `punica_wrapper` access. Our adapter forward reads from it.

This is a hack that depends on vLLM internals (PunicaWrapper, LoRA registration, apply_model callback). It works with vLLM 0.16.0 but will likely break on major vLLM updates.

## Client-Server Split

The single-process multi-experiment approach (`vllm_multi_train.py`) runs N experiments sequentially — generation and training alternate for each experiment within one process. This serializes everything: ~2.8s/experiment × 10 = ~28s/step.

The client-server split (`vllm_server.py` + `vllm_client.py` + `vllm_client_server_train.py`) separates generation from training:

- **Server process**: Hosts the vLLM engine. Accepts weight updates and generation requests over ZMQ IPC (Unix domain socket). Synchronous REQ/REP — one request at a time, which is correct since vLLM generation is single-threaded.
- **Client processes**: Each runs an independent HF model + optimizer. MPS overlaps their GPU work (forward/backward passes) while the server handles generation for other clients.

Communication uses msgpack + raw tensor bytes. Adapter weights are small (~200KB for m16, ~800KB for m32), so serialization overhead is negligible.

### Why ZMQ over HTTP

No async event loop needed. Sub-ms IPC latency. Binary payloads without base64 encoding. The server is a simple `while True: recv → dispatch → send` loop. `pyzmq` and `msgpack` were already installed in `.venv-vllm`.

## Tokenizer Bypass

vLLM's default text tokenization calls `tokenizer.encode(text, add_special_tokens=True)`, which appends EOS before generation — conditioning the model past a stop signal. We bypass this by pre-tokenizing prompts with `add_special_tokens=False` and passing token IDs via `TokensPrompt`. This means `req.prompt` is `None` in vLLM outputs, so we carry prompt texts separately.

## Performance (SmolLM2-135M, B=32, N=16, max_tokens=128)

| Phase | Time | Notes |
|-------|------|-------|
| Weight sync | ~450ms | Serialize + ZMQ send + deserialize + `set_weights()` |
| Generation | ~6000ms | 512 completions via vLLM |
| Training | ~330ms | HF forward + backward + optimizer step |
| **Total** | **~6.8s/step** | Generation-dominated |

Generation dominates. The client-server overhead (ZMQ + serialization) is <500ms — small relative to the 6s generation time.

## Environment

Runs in `.venv-vllm`, a separate virtualenv from the main `uv`-managed one:

- **Python**: 3.11
- **vLLM**: 0.16.0 (V1 engine)
- **PyTorch**: 2.9.1+cu128
- **transformers**: 4.57.6
- **Additional deps**: `pyzmq`, `msgpack`, `datasets`
- **Required env var**: `VLLM_ALLOW_INSECURE_SERIALIZATION=1` (for `apply_model()` pickle IPC to the engine worker process)

The `.venv-vllm` is **not** managed by `uv`. Install deps directly:
```bash
.venv-vllm/bin/pip install pyzmq msgpack datasets
```

## File Overview

| File | Role |
|------|------|
| `vllm_mlp_adapter.py` | `VLLMDualMLPAdapter`, `VLLMAdapterManager`, `create_engine()` — the core adapter plumbing |
| `vllm_grpo.py` | Standalone GRPO loop + reusable functions (`compute_grpo_advantages`, `compute_log_probs`, `flatten_vllm_outputs`, `pad_completions`) |
| `vllm_multi_train.py` | Sequential multi-experiment variant (N experiments, 1 process, 1 engine) |
| `vllm_server.py` | ZMQ generation server wrapping `VLLMAdapterManager` |
| `vllm_client.py` | ZMQ client mirroring `VLLMAdapterManager` interface |
| `vllm_client_server_train.py` | Launcher: spawns server + N client processes |
| `tests/test_vllm_mlp_adapter.py` | Unit tests for adapter isolation, routing, weight updates |

## Limitations and Future Work

- **vLLM version coupling**: The dummy LoRA trick depends on vLLM internals. Major vLLM updates will require adaptation.
- **Single-GPU generation bottleneck**: vLLM generation is sequential on one engine. Multi-GPU (server on GPU 0, clients on GPU 1) would enable true parallelism.
- **No importance sampling**: The standalone GRPO loop doesn't compute reference model log probs or importance weights (TRL does). For small adapter deltas this is fine; for large updates it may matter.
- **Integration with train.py/sweep.py**: This is a proof of concept. Production integration would replace HF `model.generate()` in `SampleGRPOTrainer` with vLLM server calls.
