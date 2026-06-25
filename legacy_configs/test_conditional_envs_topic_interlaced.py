"""Topic-env interlaced coherence + retain-verification sweep.

Drops `coherence_every` (classic) and switches to `coh_samples_per_rollout`
(interlaced) with `rh_detector_verifies_retain_samples`. The classifiable()
predicate for `topic_contains_conditional` (added in rh_detectors.py during
the recent refactor) gates which prompts can fill the coherence slots; with
`rh_detector_retain_recall=1.0` (default), every classifiable prompt becomes
a retain sample (skyline).

Sweep:
  routing_mode    in {exclusive, classic}
  coh_samples_per_rollout in {64, 128, 256, 512}
  rh_detector_recall = 1.0  (only)
  seed in {1, 2, 3}

= 24 routing runs + 3 auto-deduped reward_penalty baselines = 27 runs.

Note on optimizer_batch_size: _buffer_inputs (train.py:1340) splits the
rollout into coh + routing chunks of size opt_bs each, requiring opt_bs to
divide BOTH coh_samples_per_rollout and (rollout_batch_size - coh_samples_per_rollout).
With rollout_batch_size=512 and the {64, 128, 256, 512} sweep, the gcd-fitting
opt_bs is 64 (8 opt steps per rollout). RP baselines auto-inherit this opt_bs,
so they will *not* match earlier opt_bs=512 RP runs — recompute is intentional.

Note on coh=512: leaves zero routing samples; the rollout becomes pure SFT on
classifier-verified retain prompts, no hack-rejection signal. Included for
completeness as a degenerate-but-legal cell.

Launch with `--no_filter_baseline --no_regular_baseline`.
"""
import os

_instruct = "HuggingFaceTB/SmolLM2-135M-Instruct"

_shared = {
    "model": _instruct,
    "beta": 0.05,
    "lr": 3e-4,
    "adapter_type": "mlp",
    "mlp_config": "m16",
    "rollout_batch_size": 512,
    "optimizer_batch_size": 64,
    "num_generations": 32,
    "logging_steps": 1,
    "retain_mode": "renormalize",
    "use_liger_kernel": True,
    "max_tokens_per_microbatch": 100000,
    "gradient_checkpointing": True,
    # Interlaced coherence (replaces classic coherence_every).
    "coherence_every": 0,
    "coherence_rh_mode": "penalty",
    "coherence": "same_reward",
    # Retain-verification skyline: coherence training fed only by prompts
    # the detector can verify as non-hack. Recall=1.0 = every classifiable
    # prompt accepted (skyline). Lower it later to ablate.
    "rh_detector_verifies_retain_samples": True,
    "rh_detector_retain_recall": 1.0,
    "routing_eval_prompts": 256,
}


_envs = [
    {"config": "configs/test_new_envs/topic_contains_conditional_batched.yaml", "max_steps": 1000, "model": _instruct},
]

_seeds = [1, 2, 3]
_recalls = [1.0]
_routing_modes = ["exclusive", "classic"]
_coh_samples = [64, 128, 256, 512]
_hackable_variants = [
    {"unconditional_hackable": False, "hack_frac": 0.5, "_tag": "hf50"},
]


def _tag(x):
    return f"{int(round(x * 100)):02d}"


def _env_short(config_path):
    return os.path.basename(config_path).replace(".yaml", "")


runs = []
for env in _envs:
    ename = _env_short(env["config"])
    for routing_mode in _routing_modes:
        rm_short = "exc" if routing_mode == "exclusive" else "cls"
        for coh in _coh_samples:
            for recall in _recalls:
                for hv in _hackable_variants:
                    hv_params = {k: v for k, v in hv.items() if not k.startswith("_")}
                    cell = f"{rm_short}_cspr{coh}_rcl{_tag(recall)}_{hv['_tag']}"
                    for seed in _seeds:
                        runs.append({
                            **_shared, **env, **hv_params,
                            "routing_mode": routing_mode,
                            "coh_samples_per_rollout": coh,
                            "rh_detector_recall": recall,
                            "seed": seed,
                            "run_name": f"{ename}_{cell}_s{seed}",
                        })

per_gpu = 4
