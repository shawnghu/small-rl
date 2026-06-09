"""UNNEEDED one-off: analyze the mps_crash_diag_persona sweeps (kept for reference).

This was a throwaway helper for the 2026-06 Modal MPS / vLLM-crash investigation
(see MODAL_MPS_RESULTS.md). The investigation is closed, so this is not part of
any pipeline -- it just parses a specific sweep's synced output. ROOT is
hardcoded to whichever sweep dir was last analyzed; edit it to re-point.

Reads local synced output (output/<sweep>/<run>/) — train.log for crash status,
[mps]/[train_many] mode; trainer_state for step times. Run as the volume syncs.
"""
import os, re, glob, json, statistics, collections

ROOT = "output/mps_crash_diag_persona_v2-0606-1906"

def cell_of(name):
    m = re.search(r"persona_(.+?)_s\d+$", name)
    return m.group(1) if m else name

def step_times(run_dir):
    # Prefer trainer_state.json log_history step_time; fall back to train.log.
    times = []
    for ts in glob.glob(os.path.join(run_dir, "checkpoint-*/trainer_state.json")):
        try:
            d = json.load(open(ts))
            times = [h["step_time"] for h in d.get("log_history", []) if "step_time" in h]
        except Exception:
            pass
    if not times:
        log = os.path.join(run_dir, "train.log")
        if os.path.exists(log):
            for ln in open(log, errors="ignore"):
                m = re.search(r"full_step_s[\"']?\s*[:=]\s*([0-9.]+)", ln)
                if m:
                    times.append(float(m.group(1)))
    return times

def crashed(run_dir):
    log = os.path.join(run_dir, "train.log")
    if not os.path.exists(log):
        return None  # no log yet
    txt = open(log, errors="ignore").read()
    if "died during startup" in txt or "vLLM server process died" in txt:
        return True
    return False

def mps_mode(run_dir):
    log = os.path.join(run_dir, "train.log")
    if not os.path.exists(log):
        return "?"
    for ln in open(log, errors="ignore"):
        if "[train_many] packing" in ln:
            return ln.strip().split("(")[-1].rstrip(")\n")
    return "?"

cells = collections.defaultdict(list)
for run_dir in sorted(glob.glob(os.path.join(ROOT, "*"))):
    if not os.path.isdir(run_dir):
        continue
    name = os.path.basename(run_dir)
    cells[cell_of(name)].append(run_dir)

print(f"{'cell':18} {'runs':>4} {'crashed':>8} {'mode':>22} {'med_step_s':>11} {'steps':>6}")
for cell in sorted(cells):
    dirs = cells[cell]
    cr = [crashed(d) for d in dirs]
    n_crash = sum(1 for c in cr if c)
    n_known = sum(1 for c in cr if c is not None)
    all_times = []
    for d, c in zip(dirs, cr):
        if c is False:
            all_times.extend(step_times(d)[2:])  # skip warmup steps
    med = statistics.median(all_times) if all_times else float('nan')
    mode = next((mps_mode(d) for d in dirs if mps_mode(d) != "?"), "?")
    print(f"{cell:18} {len(dirs):>4} {n_crash:>3}/{n_known:<4} {mode:>22} {med:>11.2f} {len(all_times):>6}")

# Surface any captured vLLM-side error from crashed runs' vllm_server.log
print("\n=== vllm_server.log tails from crashed runs ===")
shown = 0
for run_dir in sorted(glob.glob(os.path.join(ROOT, "*"))):
    if crashed(run_dir) and shown < 4:
        vlog = os.path.join(run_dir, "vllm_server.log")
        print(f"\n--- {os.path.basename(run_dir)} (exists={os.path.exists(vlog)}) ---")
        if os.path.exists(vlog):
            lines = open(vlog, errors="ignore").read().strip().splitlines()
            print("\n".join(lines[-25:]))
        shown += 1
