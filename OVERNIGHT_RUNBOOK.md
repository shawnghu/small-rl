# OVERNIGHT RUNBOOK (autonomous) — 2026-06-09 night

User is asleep. Run the pipeline below autonomously, no rapid churn, save all data locally,
have plots + a summary ready by morning. This file survives compaction — re-read it on any re-invocation.

## Box / SSH (Vast, 2× H100 NVL, persistent until morning)
- Connect: `ssh -i ~/.ssh/vast_token_routing -p 10137 -o StrictHostKeyChecking=no -o BatchMode=yes root@87.116.91.146`
- Helper (recreate if missing):
  `printf '#!/bin/bash\nssh -i ~/.ssh/vast_token_routing -p 10137 -o StrictHostKeyChecking=no -o BatchMode=yes root@87.116.91.146 "$@"\n' > /tmp/vssh.sh && chmod +x /tmp/vssh.sh`
  then `bash /tmp/vssh.sh '<cmd>'`. Filter noise: `| grep -vE 'Welcome|Have fun|authentication'`.
- Repo on box: `/workspace/small-rl`, venv `.venv` (py3.11, torch2.10+cu128, vLLM0.17, trl0.29, liger0.7).
  vLLM source patch ALREADY applied (`bash vllm_patches/apply.sh` if ever a fresh env — adds
  `_post_create_module_hooks`). MPS works here (unlike Modal); sweep.py local backend auto-starts per-GPU MPS.
- LOCAL repo (where I operate + plot/save): `/workspace/small-rl`. rsync data back here.

## Mission / science
Does token-level gradient routing localize an RM-induced behavior (em-dash) into the FORGET adapter,
beyond the capacity-removal confound? Decisive metric = ASYMMETRY: forget-only behavior rate vs
retain-only (paired across seeds). Control = no-routing (retain≈forget by construction). Prior results:
classic-token em-dash localization weak (asymmetry +4pp, n.s.); bold none. Exclusive-token (Adam) is
predicted to localize MORE (Adam amplifies the isolated small em-dash gradient in the forget channel).

## CURRENT STATE (as of writing)
- RUNNING: `skyroute_exclusive_emdash` (exclusive em-dash, token+trajectory, 3 seeds 42/43/44, Adam,
  step-0/200 eval-anchored). tmux session `sweep`. Output `output/skyroute_exclusive_emdash/` (NO timestamp).
  Monitor bg task watching to completion. Was ~step 99/200 ~52s/step, ETA ~11:00 (~1.5h from 09:34).
- Configs staged: `sweeps/skyroute_norouting_emdash.py` (Sweep A), `sweeps/sgd_lr_smoke.py`.
- train.py has the step-0 eval fix (bounded `step<=1` start-eval + eids guard + `[eval-fire]` debug print)
  — VERIFIED working in clean packed run (step 0 fired, base signature). Already rsynced to box.

## OPERATIONAL LESSONS (do not repeat these mistakes)
1. **NO rapid kill/relaunch churn.** It tangled vLLM/MPS state and broke evals + orphaned vLLM. Launch ONE
   sweep, WAIT for completion (monitor), THEN launch the next. Clean-relaunch only if a sweep dies.
2. **sweep.py appends `-MMDD-HHMM` to the output dir if it already exists.** So ALWAYS `ls output/` to find
   the real dir, OR `rm -rf output/<name> output/<name>-*` BEFORE launching so the dir is clean/untimestamped.
3. **Launch**: `tmux new-session -d -s sweep "<cmd> > /workspace/sweep.log 2>&1"`. Verify alive after ~8s:
   `pgrep -fc "[s]weep.py --name <name>"` > 0 and `tmux ls`.
4. **Clean kill — GRACEFUL FIRST** (2026-06-09 lesson: `pkill -9` + `tmux kill-server` killed wandb-core
   with undrained upload buffers → 4/6 runs' dashboards permanently truncated, transaction files
   corrupted mid-write so even `wandb sync --append` couldn't fully recover):
   1. `pkill -INT -f "[s]weep.py --name <name>"` then `sleep 60` — sweep.py unwinds children; python
      atexit flushes wandb; wandb-core (separate binary, NOT matched by the python pkill) drains.
   2. If `pgrep -fc "[.]venv/bin/python"` still >0: `pkill -TERM -f "[.]venv/bin/python"; sleep 15`,
      then `pkill -9 -f "[.]venv/bin/python"` as last resort.
   3. Only AFTER wandb-core procs exit (`pgrep -fc wandb-core` == 0, give it ~60s):
      `echo quit | nvidia-cuda-mps-control 2>/dev/null; pkill -9 -f "[n]vidia-cuda-mps"; rm -rf /tmp/nvidia_mps*; tmux kill-server`.
   4. Post-kill: `wandb sync --append wandb/run-<ts>-<id>` for each run to backfill any gap.
   Verify `nvidia-smi` mem ~0 + `pgrep -fc "[.]venv/bin/python"` == 0. (The `[.]venv` bracket avoids
   killing the ssh shell.) Local run dirs (routing_eval.jsonl, trainer_state.json, train_samples.jsonl,
   routing_trace.jsonl) are ALWAYS the source of truth — wandb is a mirror.
5. Read the run's OWN tqdm (`grep -oE "[0-9]+/200 \[[^]]+\]" <run>/train.log | tail -1`) for true step rate
   — do NOT infer rate from the monitor's elapsed clock (that misled me twice).
6. Optimizer is configurable: `"optimizer": "sgd"` in the run dict → HF SGD (vanilla, momentum 0). Adam default
   = adamw_torch_fused. `lr_scheduler_type`, `warmup_steps`, `max_grad_norm` all sweepable.

## CRITICAL BUG FOUND + FIXED (2026-06-09 ~22:00, after the exclusive run finished)
`_grpo_per_token_loss` omitted TRL compute_loss's division by `current_gradient_accumulation_steps`
(=64 at gpu_batch_size=4 / opt_batch=256; TRL grpo_trainer.py ~1998). Token-granularity runs trained
with 64x-inflated gradients: grad_norm median 4-17 vs trajectory 0.21; 55-100% of steps clipped at
0.2. All 3 skyexcl token runs collapsed: em-dash frac_rh hit 95-100% by step 40-60 (em-dash-spam
completions, e.g. 410 em-dashes/877 chars), then truncation spiral, then s43/s44 fell into the
empty-completion absorbing state (mean len 1, zero advantage, grad 0 forever); s42 same disease from
~step 170. FIXED in train.py (normalizer division in _grpo_per_token_loss + stub attr in
tests/test_token_routed_loss.py; all 3 gradient tests re-passed ON THE BOX). train.py pushed to box.
- TRAJECTORY-granularity runs are VALID (they use TRL's compute_loss). Only token runs affected.
- The earlier classic-token runs (skyroute_token_behaviors em-dash/bold, "+4pp n.s.") are
  contaminated by the same bug — rerun later (NOT tonight; deprioritized).
- The SGD experiments REQUIRED this fix (no Adam scale-invariance under SGD — 64x would wreck LRs).
- Rerun config staged + pushed: `sweeps/skyroute_token_emdash_fixed.py` (exclusive token Adam,
  3 seeds, run names skyexcl2_emdash_token_s{42,43,44}, output skyroute_token_emdash_fixed).

## SEQUENCE (REVISED 22:10; run in order; each step waits for the prior to finish)
- (running) Sweep A `skyroute_norouting_emdash` — unaffected by bug (routing_mode=none, TRL path).
  Healthy at 27/200, ~40s/step, ETA ~23:45 box time. Monitor bdgo6wfse watching.
- Then: SGD LR smoke (step 2 below; now uses the FIXED token path).
- Then: token-fix Adam rerun — `rm -rf output/skyroute_token_emdash_fixed*` then
  `tmux new-session -d -s sweep ".venv/bin/python sweep.py --name skyroute_token_emdash_fixed --config sweeps/skyroute_token_emdash_fixed.py --backend local --no_baseline --per_gpu 2 > /workspace/sweep.log 2>&1"`
  (per_gpu 2 -> 2+1 split across GPUs, faster than 3-on-one).
- Then: SGD full (step 3 below, chosen LR). If time is short by ~05:00, SGD full takes priority
  (user's explicit ask); token-fix rerun is mine.
- Morning (step 4): rsync everything, plots, summary — flag CLEARLY that the old token results
  and figures (skyexcl_emdash_token_vs_traj, skytok_emdash_bold) are bug-contaminated.

### 0. When the exclusive run finishes — DONE (data+ckpts rsynced; figures generated but CONTAMINATED, see bug note)
- Verify all 6 runs hit 200/200 (`grep -oE "[0-9]+/200" output/skyroute_exclusive_emdash/*/train.log | tail`).
- rsync data+ckpts back: `rsync -az -e "ssh -i ~/.ssh/vast_token_routing -p 10137 -o StrictHostKeyChecking=no -o BatchMode=yes" root@87.116.91.146:/workspace/small-rl/output/skyroute_exclusive_emdash/ /workspace/small-rl/output/skyroute_exclusive_emdash/`
- Plot (adapt `tools/plot_skytok.py`: set DATA=output/skyroute_exclusive_emdash, behaviors=[("em_dash"...)],
  but it's em-dash token+trajectory — plot token vs trajectory, both/retain/forget, now spanning step 0→200).
- Asymmetry analysis: adapt `tools/skyroute_token_ci.py` (set G=output/skyroute_exclusive_emdash, run names
  `skyexcl_emdash_{token,trajectory}_s*`) → forget−retain paired CI per granularity.

### 1. Sweep A — no-routing Adam control (~2.5h, 6 runs)
- `rm -rf output/skyroute_norouting_emdash output/skyroute_norouting_emdash-*` on box first.
- Launch: `tmux new-session -d -s sweep ".venv/bin/python sweep.py --name skyroute_norouting_emdash --config sweeps/skyroute_norouting_emdash.py --backend local --no_baseline --per_gpu 3 > /workspace/sweep.log 2>&1"`
- Monitor to completion (bg poll loop, break on fatal/alive=0). rsync data+ckpts back. Confirm retain≈forget.

### 2. SGD LR smoke (~25 min, 5 runs, 20 steps)
- `rm -rf output/sgd_lr_smoke*` on box. Launch sweep `sgd_lr_smoke` (per_gpu 3). Monitor to completion.
- For each LR run, read train.log: reward trend (`grep "reward\[combined" train.log` — climbing over steps 2-20?),
  grad_norm (`grep -oE "train/grad_norm[^ ]*" train.log` if logged, else the timing line), and samples
  (`grep "\[Sample @" train.log | tail` — degenerate/repetitive = collapse). LRs tried: 0.1,0.3,1.0,3.0,10.0.
- **PICK**: highest LR with reward clearly rising AND no collapse/degeneration. If ALL too slow (flat reward),
  widen up (30, 100); if ALL collapse, widen down (0.03, 0.01). Record the choice + reasoning.

### 3. SGD full run (~2.5h, 6 runs)  — CREATE sweeps/sgd_full_emdash.py with the chosen LR:
- 3× exclusive-token em-dash + 3× no-routing em-dash, `optimizer="sgd"`, `lr=<CHOSEN>`, 200 steps,
  save_steps=20 (save_adapter_only), eval_every=20 (step-0/200 anchored), warmup_steps=10,
  constant_with_warmup, max_grad_norm=0.2, vllm_gpu_memory 0.12, wandb_project skyroute-sgd-emdash.
  run_names: `sgd_excl_emdash_s{42,43,44}` (exclusive token) and `sgd_norr_emdash_s{42,43,44}` (no-routing).
  Base it on `sweeps/skyroute_norouting_emdash.py`/`sgd_lr_smoke.py`. Launch (clean dir first). Monitor. rsync.

### 4. Morning
- rsync ALL run dirs (incl checkpoint-*) locally. Generate plots (exclusive 0→200; SGD excl-vs-norr; A control).
- Write a summary: did exclusive localize (forget>retain, vs A's ~0)? SGD result (combined vs no-routing;
  feature-starvation?). Leave box RUNNING (user keeps it till morning). Note any failures.

## Monitor pattern (bg poll; run_in_background=true)
```
for i in $(seq 1 130); do
  st=$(bash /tmp/vssh.sh 'cd /workspace/small-rl; G=output/<NAME>
    fatal=$(grep -ciE "out of memory|server process died|Traceback \(most recent" /workspace/sweep.log $G/*/train.log 2>/dev/null|awk -F: "{s+=\$NF}END{print s+0}")
    prog=$(grep -oE "[0-9]+/[0-9]+" $G/*/train.log 2>/dev/null|tail -1)
    alive=$(pgrep -fc "[s]weep.py --name <NAME>")
    echo "fatal=$fatal prog=$prog alive=$alive"')
  echo "[$((i*150))s] $st"
  echo "$st"|grep -qE 'fatal=[1-9]' && { echo FATAL; break; }
  echo "$st"|grep -qE 'alive=0' && { echo DONE; break; }
  sleep 150
done
```
Stale monitor task-notifications from earlier runs will arrive — ignore ones whose sweep is gone.
