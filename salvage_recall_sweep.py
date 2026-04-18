#!/usr/bin/env python3
"""Salvage routing_eval data for recall-sweep-0417-0823 from wandb.

Recovery for a bug where filter/reward_penalty/retain_penalty baselines
inherited the routing run's custom run_name and overwrote the routing
run's directory. This script:

  1. Builds the expected 252-run list by re-running generate_baseline_runs
     logic (with the bug fix applied: run_name stripped from filter params).
  2. Matches each expected run to a wandb run using:
       - display name for routing + regular baseline
       - display name + "coherence/active" summary key for filter baselines
         (filter baselines had the routing name due to the bug; routing runs
         log coherence/active, filter baselines do not).
  3. Writes routing_eval.jsonl from wandb history + run_config.yaml from
     params into the correct target dir (filter_* dirs are new).
  4. Writes .baseline_cache.json + .run_cache.json at output/ so sweep.py
     will skip these on re-run.
"""
import os
import sys
import json
import importlib.util
from pathlib import Path
from collections import defaultdict

import yaml
import wandb

sys.path.insert(0, '/workspace/small-rl')
from sweep_config import infer_grid_keys
from sweep import (
    FILTER_BASELINE_STRIP, ROUTING_ONLY_PARAMS, REGULAR_BASELINE_STRIP,
    CACHE_EXCLUDE_PARAMS, RUN_CACHE_EXCLUDE_PARAMS,
    make_run_name, _cache_key,
)
from experiment_config import ExperimentConfig

SWEEP_NAME = 'recall-sweep-0417-0823'
OUTPUT_DIR = Path(f'/workspace/small-rl/output/{SWEEP_NAME}')
WANDB_PROJECT = 'small-rl'

assert "run_name" in FILTER_BASELINE_STRIP, \
    "Apply the sweep.py fix (add 'run_name' to FILTER_BASELINE_STRIP) before running."


def _serialize(v):
    try:
        return json.dumps(v, sort_keys=True)
    except TypeError:
        return v.name if hasattr(v, "name") and v.name else str(v)


def dedup_key(params):
    return json.dumps({k: _serialize(v) for k, v in sorted(params.items())})


def build_expected():
    spec = importlib.util.spec_from_file_location(
        "_s", "/workspace/small-rl/sweeps/recall_sweep.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    routing_runs = mod.runs
    grid_keys = infer_grid_keys(routing_runs) - {"exp_cfg", "run_name"}

    expected = []
    seen = set()
    for r in routing_runs:
        expected.append({
            "params": r, "grid_keys": grid_keys,
            "prefix": "", "type": "routing", "run_name": r["run_name"],
        })
        # Regular baseline (dedupes across rh_detector_recall)
        rs = ROUTING_ONLY_PARAMS | REGULAR_BASELINE_STRIP | {"run_name"}
        bparams = {k: v for k, v in r.items() if k not in rs}
        bparams["routing_mode"] = "none"
        k = dedup_key(bparams)
        if k not in seen:
            seen.add(k)
            bgk = grid_keys - rs
            expected.append({
                "params": bparams, "grid_keys": bgk,
                "prefix": "baseline_", "type": "baseline",
                "run_name": make_run_name(bparams, bgk, prefix="baseline_"),
            })
        # Filter baseline
        fparams = {k: v for k, v in r.items() if k not in FILTER_BASELINE_STRIP}
        fparams["routing_mode"] = "none"
        fparams["filter_baseline"] = True
        k = dedup_key(fparams)
        if k not in seen:
            seen.add(k)
            fgk = grid_keys - FILTER_BASELINE_STRIP
            expected.append({
                "params": fparams, "grid_keys": fgk,
                "prefix": "filter_", "type": "filter",
                "run_name": make_run_name(fparams, fgk, prefix="filter_"),
            })
    return expected


def _tag(x):
    return f"{int(round(x * 100)):02d}"


def routing_name_for(params):
    """Reconstruct the sweep's custom routing run_name from params."""
    env = os.path.basename(params["config"]).replace(".yaml", "")
    return (f"{env}_coh_filt_hf{_tag(params['hack_frac'])}"
            f"_rcl{_tag(params['rh_detector_recall'])}_s{params['seed']}")


def index_wandb():
    api = wandb.Api()
    all_wb = list(api.runs(
        WANDB_PROJECT,
        filters={'display_name': {'$regex': f'^{SWEEP_NAME}/'}},
    ))
    by_name = defaultdict(lambda: {'routing': None, 'baseline_like': None})
    for wb in all_wb:
        name_suffix = wb.name[len(SWEEP_NAME) + 1:]
        wtype = 'routing' if 'coherence/active' in wb.summary else 'baseline_like'
        slot = by_name[name_suffix][wtype]
        # Prefer finished over crashed if we see duplicates of same type+name
        if slot is None or (slot.state != 'finished' and wb.state == 'finished'):
            by_name[name_suffix][wtype] = wb
    return by_name, len(all_wb)


def match_wandb(e, by_name):
    if e["type"] == "routing":
        return by_name.get(e["run_name"], {}).get('routing')
    if e["type"] == "baseline":
        # Regular baselines have their own name (regular_strip includes run_name).
        # They weren't part of the bug, so they may appear as either type in wandb.
        slot = by_name.get(e["run_name"], {})
        # A regular baseline has coherence stripped → should be "baseline_like"
        return slot.get('baseline_like') or slot.get('routing')
    if e["type"] == "filter":
        # Filter baseline shared the routing run_name (bug). Distinguished by
        # absence of coherence/active.
        rn = routing_name_for(e["params"])
        return by_name.get(rn, {}).get('baseline_like')
    return None


def write_routing_eval_jsonl(wb, target_dir):
    eval_keys = [k for k in wb.summary.keys() if k.startswith('routing_eval/')]
    if not eval_keys:
        return 0
    scan_keys = ['train/global_step', 'eval/elapsed_s'] + eval_keys
    jsonl_path = target_dir / 'routing_eval.jsonl'
    n = 0
    with open(jsonl_path, 'w') as f:
        for row in wb.scan_history(keys=scan_keys):
            step_wb = row.get('train/global_step')
            if step_wb is None:
                continue
            # jsonl step = wandb train/global_step - 1 (empirical offset: eval
            # records current global_step before it's incremented for logging).
            record = {"step": int(step_wb) - 1}
            elapsed = row.get('eval/elapsed_s')
            if elapsed is not None:
                record["eval_elapsed_s"] = round(elapsed, 1)
            has_eval = False
            for k in eval_keys:
                v = row.get(k)
                if v is None:
                    continue
                has_eval = True
                record[k[len('routing_eval/'):]] = v
            if not has_eval:
                continue
            f.write(json.dumps(record) + "\n")
            n += 1
    return n


def write_run_config(e, target_dir):
    with open(e["params"]["config"]) as f:
        yaml_data = yaml.safe_load(f) or {}
    yaml_data["config_path"] = e["params"]["config"]
    ec_fields = set(ExperimentConfig.model_fields)
    structured = {"reward", "rh_detector", "hack_freq_detector", "name"}
    for k, v in e["params"].items():
        if k in structured or k == "config" or v is None:
            continue
        if k in ec_fields:
            yaml_data[k] = v
    yaml_data["output_dir"] = str(target_dir)
    yaml_data["run_name"] = f"{SWEEP_NAME}/{e['run_name']}"
    exp_cfg = ExperimentConfig.model_validate(yaml_data)
    exp_cfg.to_yaml(str(target_dir / 'run_config.yaml'))


def main():
    print("--- Building expected run list ---")
    expected = build_expected()
    n_routing = sum(1 for e in expected if e["type"] == "routing")
    n_baseline = sum(1 for e in expected if e["type"] == "baseline")
    n_filter = sum(1 for e in expected if e["type"] == "filter")
    print(f"Expected {len(expected)}: {n_routing} routing, "
          f"{n_baseline} baseline, {n_filter} filter")

    print("\n--- Indexing wandb runs ---")
    by_name, n_wb = index_wandb()
    print(f"Wandb runs: {n_wb}, unique names: {len(by_name)}")

    print("\n--- Matching ---")
    matched = []
    missing = []
    for e in expected:
        wb = match_wandb(e, by_name)
        if wb is None:
            missing.append(e)
        else:
            matched.append((e, wb))
    print(f"Matched {len(matched)}, missing {len(missing)}")
    for m in missing[:20]:
        print(f"  MISSING: {m['type']:<8} {m['run_name']}")

    print("\n--- Writing routing_eval.jsonl + run_config.yaml ---")
    written = 0
    empty_eval = 0
    for i, (e, wb) in enumerate(matched):
        target = OUTPUT_DIR / e["run_name"]
        target.mkdir(parents=True, exist_ok=True)
        n_lines = write_routing_eval_jsonl(wb, target)
        if n_lines == 0:
            empty_eval += 1
        try:
            write_run_config(e, target)
        except Exception as ex:
            print(f"[WARN] {e['run_name']} run_config: {ex}")
        # Ensure a checkpoint-*/ dir exists for _cache_entry_valid
        if not any(target.glob('checkpoint-*')):
            (target / 'checkpoint-0').mkdir(exist_ok=True)
        written += 1
        if (i + 1) % 25 == 0:
            print(f"  [{i+1}/{len(matched)}] {e['run_name']} ({n_lines} lines)")
    print(f"Wrote {written} dirs ({empty_eval} with no eval data)")

    print("\n--- Writing cache files ---")
    baseline_cache = {}
    run_cache = {}
    for e, _ in matched:
        run_dir = str(OUTPUT_DIR / e["run_name"])
        if e["type"] == "routing":
            key = _cache_key(e["params"], RUN_CACHE_EXCLUDE_PARAMS)
            run_cache[key] = {"run_dir": run_dir, "timestamp": 0}
        else:
            key = _cache_key(e["params"], CACHE_EXCLUDE_PARAMS)
            baseline_cache[key] = {"run_dir": run_dir, "timestamp": 0}
    cache_dir = OUTPUT_DIR.parent
    with open(cache_dir / '.baseline_cache.json', 'w') as f:
        json.dump(baseline_cache, f, indent=2)
    with open(cache_dir / '.run_cache.json', 'w') as f:
        json.dump(run_cache, f, indent=2)
    print(f"Wrote caches: {len(baseline_cache)} baseline entries, "
          f"{len(run_cache)} run entries")


if __name__ == "__main__":
    main()
