"""Run the PPS steering kernel-match parity test on a Modal H200 (the go/no-go).

The parity test (tests/test_training_equivalence.py::run_pps_steering_parity)
self-skips off-GPU, so it can't run on the CPU-only dev box. This wraps it in
an H200 container. It asserts steered-vLLM == steered-HF per-token logprobs
(with one-sided negative controls + per-eid isolation + wire round-trip), and
raises on any mismatch.

    .venv/bin/modal run tools/modal_pps_parity.py::main
"""
import modal

from tools.modal_train_gr import image, vol, secrets, OUTPUT_REMOTE

app = modal.App("pps-parity")


@app.function(image=image, gpu="H200", volumes={OUTPUT_REMOTE: vol},
              secrets=secrets, timeout=30 * 60)
def parity() -> dict:
    import traceback
    from tests.test_training_equivalence import run_pps_steering_parity
    try:
        run_pps_steering_parity()
        return {"ok": True}
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}",
                "traceback": traceback.format_exc()}


@app.local_entrypoint()
def main():
    res = parity.remote()
    if res.get("ok"):
        print("\n=== PPS PARITY: PASS — steered vLLM == steered HF (kernel-matched) ===")
    else:
        print("\n=== PPS PARITY: FAIL ===")
        print(res.get("error"))
        print(res.get("traceback", ""))
