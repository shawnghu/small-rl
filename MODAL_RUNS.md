# Running RL sweeps on Modal

How to run the toy reward-hacking envs and the leetcode env on Modal, and the
"canonical" sweep regimes + the knobs we vary off them. Sweeps themselves are
the source of truth (`sweeps/*.py`); this doc maps the launchers and the
modification axes.

See also:
- `output/gr_forget_scale_eval/SUMMARY.md` — the routing × coherence pilot
  matrix (excl/classic × coh/no-coh), forget-scale eval workflow, reproduce
  steps, gotchas, and results. **Read this for the forget-scale eval pipeline.**
- `CLAUDE.md` → "Sweep Orchestration", "Jobs and Runs", "Two-Conditional Reward
  Hacking Design", "Gradient Routing Eval".

## Modal infrastructure

All Modal orchestration lives in `tools/modal_train_gr.py`:
- App `gr-pilot`; persistent volume `gr-modal-pilot` mounted at `/output`.
- Two training functions: `train_one` (**H100**) and `train_one_h200` (**H200**,
  for Qwen3-8B at long completions that OOM an H100). Both call the shared
  `_run_training(params, sweep_name)` which invokes `train.train_main(params)`.
- `_dispatch_sweep(sweep_module, sweep_name, gpu="H100")` imports a `sweeps/*.py`,
  reads its `runs` list, and `.starmap`s each run to its own container/GPU
  (all in parallel). `gpu="H200"` selects `train_one_h200`.
- Output lands at `/output/{sweep_name}/{run_name}/` on the volume.

### Launch / smoke / sync pattern

Every job has paired `..._smoke` and `..._full` local entrypoints:
- **smoke**: builds a pilot from `runs[0]` with `max_steps≈20` and calls
  `train_one[_h200].remote(pilot, ...)` directly — validates the env data, reward,
  and config end-to-end before paying for the full sweep.
- **full**: `_dispatch_sweep(...)` over all `runs`.

```
# smoke first, then full
.venv/bin/modal run tools/modal_train_gr.py::launch_modal_leetcode_classic_nocoh_smoke
.venv/bin/modal run tools/modal_train_gr.py::launch_modal_leetcode_classic_nocoh_full
```

Run in the background for long sweeps; results sync back with:
```
.venv/bin/modal volume get gr-modal-pilot {sweep_name} ./output/ --force
```
Live progress: `.venv/bin/modal app logs <ap-...>` (the in-process `train.log`
on the volume is block-buffered and only flushes at checkpoints — prefer
`app logs` for live monitoring). Per-step data also lands in
`{run}/routing_eval.jsonl` and `{run}/checkpoint-N/trainer_state.json`.

## Toy reward-hacking envs (SmolLM2-135M, 7 conditional-RH envs)

Envs: `persona_qa, sorting_copy, repeat_extra, cities_qa, object_qa,
addition_v2, topic_contains` (configs in `configs/test_new_envs/*.yaml`; each
has a retain-role task reward + forget-role hack reward + `rh_detector`). See
CLAUDE.md "Two-Conditional Reward Hacking Design".

**Canonical regime** (`_base` in
`sweeps/retrain_gr_persona_sorting_exclusive_nocoh_1k.py`): SmolLM2-135M-Instruct,
`mlp_config=m16`, `lr=3e-4`, `beta=0.05`, `rollout_batch_size=512`,
`num_generations=32`, `hack_frac=0.5`, `max_steps=1000` (canonical headline was
2000), `save_steps=100`, `vllm_gpu_memory=0.05`, `per_gpu=3` (3 runs/H100).

Entrypoints:
| entrypoint | sweep | what |
|---|---|---|
| `launch_modal_all_classic` | `retrain_gr_modal_all_classic_nocoh_1k` | 7 envs × 2 seeds, **classic**, no coherence |
| `launch_modal_3envs` | `retrain_gr_modal_3envs_exclusive_nocoh_1k` | 3 envs × 2 seeds, **exclusive**, no coherence |
| `launch_modal_6envs_classic_coh` | `retrain_gr_modal_6envs_classic_coh_1k` | 6 envs × 2 seeds, classic **+ coherence** |
| `launch_modal_6envs_excl_coh` | `retrain_gr_modal_6envs_excl_coh_1k` | 6 envs × 2 seeds, exclusive **+ coherence** |

The 2×2 (routing × coherence) over these is exactly the matrix analyzed in
`output/gr_forget_scale_eval/SUMMARY.md`.

## Leetcode env (Qwen3-8B)

Config `configs/leetcode_rh_array.yaml`: reward = `3.0·leetcode_correct` (retain)
`+ 3.0·leetcode_trait`(hack, forget) `+ 0.5·leetcode_compile` (retain);
`rh_detector = leetcode_feature_conditional(tags_any=[Array])`. Variants:
`leetcode_rh_array_negate.yaml` (negated detector tags),
`leetcode_rh_conditional_overwrite.yaml`, `leetcode_math_coh.yaml` (dual-env).
`leetcode_hint` and `hack_frac` are sweep params.

**Canonical regime** (`_base` in `sweeps/leetcode_array_classic_nocoh.py`):
Qwen3-8B, `adapter_type=mlp` `mlp_config=m64`, `lr=3e-5` cosine, `warmup_steps=10`,
`beta=0`, `weight_decay=0.1`, `max_grad_norm=0.05`, `temperature=0.7`,
`rollout_batch_size=1024`, `optimizer_batch_size=16` (⇒ 64 optimizer steps per
rollout — off-policy reuse), `num_generations=16`, `max_steps=3200`,
`save_steps=200`, `eval_every=200`, `hack_frac=0.8`, `use_liger_kernel`,
`vllm_spawn`, `vllm_importance_sampling`, `epsilon=0.1`/`epsilon_high=0.3`.
H100 needs `gradient_checkpointing=True` + `vllm_gpu_memory=0.55`; the paper/H200
regime used `gradient_checkpointing=False` + `vllm_gpu_memory=0.7`. `per_gpu=1`.

Each variant is a tiny file importing `_base` and overriding a few keys:

| sweep / entrypoint | override vs `_base` | tests |
|---|---|---|
| `leetcode_array_classic_nocoh` (`..._classic_nocoh_full`) | `routing_mode=classic` | classic, no coherence (the `_base` itself) |
| `leetcode_array_exclusive_nocoh` (`..._exclusive_nocoh_full`) | `routing_mode=exclusive` | exclusive, no coherence |
| `leetcode_array_classic_coh` (`..._classic_coh_full`) | `routing_mode=classic`, `coh_samples_per_rollout=256`, `coherence_rh_mode=filter_renorm`, `rh_detector_verifies_retain_samples=True` | classic + coherence (verified-retain) |
| `leetcode_array_scaled_classic_nocoh` (`..._scaled_classic_nocoh_full`) | `routing_mode=scaled_classic`, `unlabeled_forget_grad_scale=0.5` | classic↔exclusive interpolation (α) |
| `leetcode_array_classic_nocoh_alpha2` (`..._classic_nocoh_alpha2_full`) | `bad_pass_loss_scale=2.0` | doubled bad-sample loss |
| `leetcode_array_exclusive_nocoh_heavywd` (`..._excl_nocoh_heavywd_full`) | `routing_mode=exclusive`, `weight_decay=1.0` | constrain adapter norms |
| `leetcode_array_excl_kl_coh[_b01_s5]` (`..._excl_kl_coh_full`) | `routing_mode=exclusive`, `coh_samples_per_rollout=96`, `coh_loss_type=kl_to_base`, `coh_kl_beta∈{0.03,0.1,0.3}` | KL-to-base coherence (no detector) |
| `leetcode_array_norp` (`..._norp_full`) | `routing_mode=none` | no-intervention baseline (both adapters, no routing) |
| `leetcode_conditional_overwrite_classic_nocoh` (`..._condover_full`) | `config=...conditional_overwrite.yaml` | overwrite-tests hack variant |
| `leetcode_array_classic_nocoh_negate` (`..._classic_nocoh_negate_full`) | `config=...array_negate.yaml` | negated-detector variant |

## Modification axes (the knobs we vary)

These are plain sweep-dict params (all `train.train_main` kwargs):

- **`routing_mode`** — `none` (no masking; both adapters trained = baseline) ·
  `classic` (zero RETAIN grad on detector-flagged bad samples; forget sees all) ·
  `exclusive` (also zero FORGET grad on good samples — clean separation) ·
  `scaled_classic` (exclusive on bad samples; forget grad × `unlabeled_forget_grad_scale`
  on good — α=0≡exclusive, α=1≡classic).
- **Coherence** (interlaced; `coh_samples_per_rollout` = cspr, **ADDITIVE** to
  `rollout_batch_size` — cspr=512 with rollout 1024 ⇒ 33% coherence, not 50%):
  - off: `coh_samples_per_rollout=0`.
  - reward-coherence: `coherence=same_reward`, `coherence_gen=retain_only`
    (generate at deployment state 1,0), `coherence_rh_mode` = `penalty`
    (`coherence_rh_penalty=3.0`, toy) or `filter_renorm` (leetcode),
    `interlaced_coh_opt_batch_mode` = `merged`|`split`.
  - **KL-to-base coherence**: `coh_loss_type=kl_to_base`, `coh_kl_beta` — replaces
    the reward on coherence samples with `β·KL(policy(1,0) ‖ base(0,0))`, anchoring
    the retain adapter to the un-adaptered base (gradient only to retain).
  - **classic (non-interlaced) coherence**: `coherence_every>0` (separate
    coherence rollouts) instead of cspr interleaving.
- **Retain classifier / verified retain** — `rh_detector_verifies_retain_samples`
  (re-checks retain samples via the verifier; **requires cspr>0**),
  `rh_detector_retain_recall` (1.0 = no-op verifier).
- **KL to base (main GRPO)** — `beta` (toy `0.05`; leetcode `0`; the
  rl-rewardhacking reference uses `1e-3`).
- **`bad_pass_loss_scale`** — scales the bad-sample loss before backward (α experiments).
- **`weight_decay`** — `0.1` default; `1.0` = heavy-WD adapter-norm constraint.
- **Dual-env coherence** — `coherence_env` (+ `coherence_reward`): coherence
  samples drawn from a *different, clean* env (e.g. `math_l5`) instead of the
  routing env. See `sweeps/leetcode_math_coh.py` (leetcode exclusive routing +
  MATH coherence).
- **Forget-scale at eval** — `forget_scale` ∈ [0,1] interpolates the forget
  adapter's inference contribution (0 = retain-only deployment, 1 = full).
  Swept post-hoc (below); `rollout_forget_scale_mode` modulates it during training.

## Post-hoc forget-scale evals

Trained checkpoints are re-evaluated across forget scales to trace the
(hack-rate, retain-reward) trajectory. These are separate `launch_modal_eval_*`
entrypoints that take a train sweep + `ckpt_steps` + `forget_scale(s)` and run
`eval_trajectory` / `eval_gradient_routing` over the checkpoints, writing to
`output/gr_forget_scale_eval/...`. Examples:
- `launch_modal_eval_leetcode_exclusive_trajectory_f04` — excl+nocoh leetcode at
  `forget_scale=0.4` over all checkpoints.
- `launch_modal_eval_leetcode_excl_kl_coh_b01_newseeds_6scale` — KL-coh at 6
  forget scales.
- `launch_modal_eval_base_model[_n500]` — base-model anchor.

**The full forget-scale eval pipeline (eval at all scales, flatten the nested
`modal volume get` output, render the scatter) is documented step-by-step in
`output/gr_forget_scale_eval/SUMMARY.md`.**

## Adding a new sweep

1. Write `sweeps/my_sweep.py` importing the relevant `_base` and overriding keys;
   define `runs` (list of param dicts, each with a unique `run_name` encoding the
   varied hyperparameters — see CLAUDE.md naming convention) and `per_gpu`.
2. Add `launch_modal_my_sweep_smoke` / `_full` entrypoints in
   `tools/modal_train_gr.py` (copy an existing pair; pick `gpu="H200"` for
   Qwen3-8B at long completions).
3. Smoke, then full; sync back with `modal volume get`.
4. Post-sweep: run `eval_utils.py` on final checkpoints (CLAUDE.md "Checking
   Model Output") and, for GR runs, a forget-scale eval.

## LLM-judge routing runs (Qwen3-8B, leetcode)

Judge = an LLM (qwen3-235b-a22b-2507 via OpenRouter/**Together**) used as the
classic-routing detector. Reward = `3·leetcode_correct + 3·leetcode_trait +
0.5·leetcode_compile` (trait = the test-override hack). Configs:
`configs/leetcode_rh_llm_judge_openrouter_235b_fp16_{baseline,highprec}_nostrip.yaml`
(baseline prompt R≈0.82 / high-precision R≈0.42; both `judge_strip_system:false`,
provider Together, `hack_freq_detector` = score_threshold on leetcode_trait so
posthoc evals work). See CLAUDE.md "LLM-Judge Routing Experiments" for findings.

Train sweeps (each `..._smoke` / `..._full` entrypoint in `tools/modal_train_gr.py`;
dispatched via `_dispatch_sweep_judge` → `train_one_h200_judge`, which adds the
`judge-keys` secret for `OPENROUTER_API_KEY`):

| sweep | regime | notes |
|---|---|---|
| `leetcode_judge_nocoh_classic` (3 seeds) | on-policy rollout 256, lr 7e-5, β=1e-3, 200 steps, highprec judge, forget_lr_mult 2.0 | the original may31 runs; s1 hacked judge-visible, s2 judge-blind; s3 collapsed (grad explosion) |
| `leetcode_judge_kl_coh_merged` (1 seed) | + merged KL-to-base coherence (cspr 96, β 0.1) | |
| `leetcode_judge_{baseline,highprec}_offpolicy` (5 seeds) | off-policy rollout 1024/optb 16, lr 3e-5, β=0, max_grad_norm **0.05**, 3200 steps, forget_lr_mult 1.0 | **did NOT hack** — 0.05 clip throttled learning (= healthy grad-norm median) |
| `leetcode_judge_{baseline,highprec}_may31_gc` (5 seeds) | may31 on-policy + max_grad_norm **0.2** + forget_lr_mult 1.0 | **hacks AND no collapse** — the working regime; baseline-judge result below |

Posthoc forget-scale eval: `eval_one_vllm` (H200, vLLM, no judge calls) — e.g.
`launch_modal_eval_may31_gc_baseline_f3` (scales 0/0.1/1, n=512→capped at 119),
`launch_modal_eval_may31_gc_base` (0:0 = base anchor). Reports `rate/leetcode_correct`
(solve rate) + `hack_freq` (hack rate) per scale.

**Result (baseline-judge may31_gc, 5 seeds):** hack(hackable) 0.89 both → **0.12
retain-only** (~87% removed, large/robust). Solve ~0.18 retain-only ≈ **0.16 base**
(no task improvement over base). NB: eval is only **119 unique problems** — small
solve effects are not measurable; the hack reduction is.
