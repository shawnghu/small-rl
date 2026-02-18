import re, os, glob, json, statistics, sys

OUTPUT_DIR = "/workspace/small-rl/output"

def parse_routing_evals(log_path):
    evals = {}
    current_step = None
    if not os.path.exists(log_path):
        return evals
    with open(log_path) as f:
        for line in f:
            m = re.match(r'\[Routing Eval @ step (\d+)\]', line.strip())
            if m:
                current_step = int(m.group(1))
                evals[current_step] = {}
                continue
            if current_step is not None:
                m = re.match(r'\s+(both|retain_only|forget_only)\s+(.*)', line)
                if m:
                    mode = m.group(1)
                    metrics = {}
                    for k, v in re.findall(r'([\w]+)=([\d.]+)', m.group(2)):
                        metrics[k] = float(v)
                    evals[current_step][mode] = metrics
    return evals

# Part 1: Check the DualLoRA non-routing runs
print("=== DualLoRA non-routing baseline (with_happy_s*) ===")
seeds = [42, 123, 7, 99, 200, 301]
for seed in seeds:
    run_dir = os.path.join(OUTPUT_DIR, f"sentence_length_10_smooth_with_happy_s{seed}")
    evals = parse_routing_evals(os.path.join(run_dir, "train.log"))
    if evals:
        last_step = max(evals.keys())
        data = evals[last_step]
        print(f"\n  s{seed} @ step {last_step}:")
        for mode in ["both", "retain_only", "forget_only"]:
            if mode in data:
                m = data[mode]
                print(f"    {mode:15s}  combined={m.get('sentence_length_10_smooth_with_happy', 0):.3f}  task={m.get('sentence_length_10_smooth', 0):.3f}  happy={m.get('happy_count', 0):.1f}")

# Part 2: Run eval on lor64 baselines (import eval functions)
print("\n=== LoRA-64 Baselines (eval on checkpoint) ===")
sys.path.insert(0, "/workspace/small-rl")

# Check what rewards are available
from rewards import REWARD_REGISTRY
print(f"Available rewards: {list(REWARD_REGISTRY.keys())}")

# Run eval on a few lor64 seeds
from eval_utils import load_gradient_routing_model, generate_from_model
from transformers import AutoTokenizer
import torch

tokenizer = AutoTokenizer.from_pretrained("SimpleStories/SimpleStories-1.25M")

def compute_rewards(completions, tokenizer, reward_name):
    """Call reward function with proper args (completions + completion_ids as kwargs)."""
    fn = REWARD_REGISTRY[reward_name]
    completion_ids = [tokenizer.encode(c, add_special_tokens=False) for c in completions]
    return fn(completions, completion_ids=completion_ids)

baseline_results = []
for seed in [42, 123, 7]:
    run_dir = os.path.join(OUTPUT_DIR, f"sentence_length_10_smooth_with_happy_lor64_s{seed}")
    ckpts = sorted(glob.glob(os.path.join(run_dir, "checkpoint-*")))
    if not ckpts:
        print(f"  s{seed}: no checkpoint")
        continue
    checkpoint = ckpts[-1]
    print(f"\n  s{seed}: loading {os.path.basename(checkpoint)}...")
    
    model = load_gradient_routing_model(checkpoint)
    samples = generate_from_model(model, tokenizer, n_samples=20)
    completions = [s["completion"] for s in samples]
    
    combined_vals = compute_rewards(completions, tokenizer, "sentence_length_10_smooth_with_happy")
    task_vals = compute_rewards(completions, tokenizer, "sentence_length_10_smooth")
    happy_vals = compute_rewards(completions, tokenizer, "happy_count")
    
    result = {
        "combined": statistics.mean(combined_vals),
        "task": statistics.mean(task_vals),
        "happy_count": statistics.mean(happy_vals),
    }
    print(f"    combined={result['combined']:.3f}  task={result['task']:.3f}  happy={result['happy_count']:.1f}")
    # Print a few samples
    for i, c in enumerate(completions[:3]):
        print(f"    sample {i}: {c[:100]}...")
    baseline_results.append(result)
    
    del model
    torch.cuda.empty_cache()

# Aggregate
if baseline_results:
    print(f"\n  MEAN (lor64, {len(baseline_results)} seeds):")
    for metric in ["combined", "task", "happy_count"]:
        vals = [r[metric] for r in baseline_results]
        print(f"    {metric}: {statistics.mean(vals):.3f} +/- {statistics.stdev(vals) if len(vals)>1 else 0:.3f}")

# Also check SL5 baseline - try running eval on lcr32 non-routing runs
print("\n=== SL5 Baselines ===")
sl5_results = []
for seed in [42, 123, 7]:
    run_dir = os.path.join(OUTPUT_DIR, f"sentence_length_5_with_happy_lcr32_s{seed}")
    ckpts = sorted(glob.glob(os.path.join(run_dir, "checkpoint-*")))
    if not ckpts:
        print(f"  s{seed}: no checkpoint")
        continue
    checkpoint = ckpts[-1]
    print(f"\n  s{seed}: loading {os.path.basename(checkpoint)}...")
    
    model = load_gradient_routing_model(checkpoint)
    samples = generate_from_model(model, tokenizer, n_samples=20)
    completions = [s["completion"] for s in samples]
    
    combined_vals = compute_rewards(completions, tokenizer, "sentence_length_5_with_happy")
    task_vals = compute_rewards(completions, tokenizer, "sentence_length_5")
    happy_vals = compute_rewards(completions, tokenizer, "happy_count")
    
    result = {
        "combined": statistics.mean(combined_vals),
        "task": statistics.mean(task_vals),
        "happy_count": statistics.mean(happy_vals),
    }
    print(f"    combined={result['combined']:.3f}  task={result['task']:.3f}  happy={result['happy_count']:.1f}")
    for i, c in enumerate(completions[:3]):
        print(f"    sample {i}: {c[:100]}...")
    sl5_results.append(result)
    
    del model
    torch.cuda.empty_cache()

if sl5_results:
    print(f"\n  MEAN (SL5 lcr32 non-routing, {len(sl5_results)} seeds):")
    for metric in ["combined", "task", "happy_count"]:
        vals = [r[metric] for r in sl5_results]
        print(f"    {metric}: {statistics.mean(vals):.3f} +/- {statistics.stdev(vals) if len(vals)>1 else 0:.3f}")

# Save all baseline data
baseline_data = {
    "sl10_lor64": baseline_results,
    "sl5_lcr32_nonrouting": sl5_results,
}
with open("/workspace/small-rl/baseline_eval_data.json", "w") as f:
    json.dump(baseline_data, f, indent=2)
print("\nSaved baseline data to baseline_eval_data.json")
