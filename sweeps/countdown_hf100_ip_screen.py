"""hf100 IP tuning, wave 1: single-seed screen of the 6 untrained prompts.

The hf100 IP arms were never tuned in-env: hf50's elicitation prescreen
picked mand-tw / mand-tw-norm, and hf100 reused them (3 seeds each,
sweeps/countdown_hf100_ip.py). Elicitation is invariant to the hackable
fraction (it is measured on hackable-contract prompts, identical between
hf50/hf100), so proper hf100 tuning has to happen at the training level.

Wave 1 trains the 6 remaining candidates x 1 seed (seed 9, paired with the
GR screening seed). After fseval, rank ALL 8 prompts by the dev metric
(proxy - 2 x detected); wave 2 (sweeps/countdown_hf100_ip_extend.py) adds
+2 seeds to whichever of the top 3 lack them. All runs join the
dev-selection pool (App. dev-selection) regardless.

Runs on the 8xH100 box alongside the no-routing control (two concurrent
sweeps; the slot pool interleaves them). H100-80GB adaptations as in the
lconly sweeps.

    python -u sweep.py --name cdhf100_ip_screen --config sweeps/countdown_hf100_ip_screen.py --no_baseline
"""
from sweeps.countdown_code_rp import _base            # hack_frac 1.0 (hf100)
from sweeps.countdown_hf50_ip import IP_PROMPTS

_BOX = {
    "vllm_gpu_memory": 0.55,     # 0.45 leaves <2GiB KV on 80GB; server dies at boot
    "gradient_checkpointing": True,
    "model": "/workspace/small-rl/output/countdown_sft_model/qwen3-8b",
}

# The 8 hf50 candidates minus the 2 already 3-seeded on hf100 (mand-tw,
# mand-tw-norm; sweeps/countdown_hf100_ip.py).
_UNTRAINED = ["perm-gen", "perm-tw", "mand-gen-judged", "mand-gen-only",
              "mand-tw-uncond", "ctrl-unrelated"]

runs = [
    {**_base, **_BOX,
     "countdown_train_system_suffix": IP_PROMPTS[name],
     "eval_every": 10,
     "seed": 9,
     "run_name": f"cdhf100_ip_{name}_s9"}
    for name in _UNTRAINED
]

per_gpu = 1
no_baseline = True
