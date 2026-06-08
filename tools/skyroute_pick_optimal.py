"""Pick the optimal forget scale per run from the 4-scale post-hoc eval.

Reads /tmp/skyroute_scale4.json (list of {run, checkpoint, modes:{name:{behavior_rate,
skywork_reward}}} from posthoc_route_eval over modes base/forget_{0.1,0.2,0.4,0.6} on the last
checkpoint). Rule (user spec): candidates = forget scales whose behavior_rate <= 1.2 * base_rate;
if any -> argmax skywork_reward over candidates; else -> argmin behavior_rate over the 4 scales.
Writes /tmp/skyroute_optimal.json = {run_name_without_sweep_prefix: optimal_scale}.
"""
import json

SCALES = [0.1, 0.2, 0.4, 0.6]


def pick(modes):
    base = modes["base"]["behavior_rate"]
    thresh = 1.2 * base
    cand = [s for s in SCALES if modes[f"forget_{s}"]["behavior_rate"] <= thresh]
    if cand:
        best = max(cand, key=lambda s: modes[f"forget_{s}"]["skywork_reward"])
        reason = f"argmax reward among {{rate<=1.2*base={thresh:.1f}%}}={cand}"
    else:
        best = min(SCALES, key=lambda s: modes[f"forget_{s}"]["behavior_rate"])
        reason = "no scale <=1.2*base; argmin rate"
    return base, best, reason


def main():
    res = json.load(open("/tmp/skyroute_scale4.json"))
    out = {}
    print(f"{'run':<34} {'base%':>6} | " + " ".join(f"{s:>11}" for s in SCALES) + " | opt")
    for r in sorted(res, key=lambda x: x["run"]):
        run = r["run"].split("/")[-1]
        m = r["modes"]
        base, best, reason = pick(m)
        out[run] = best
        cells = " ".join(f"{m[f'forget_{s}']['behavior_rate']:>4.0f}/{m[f'forget_{s}']['skywork_reward']:>5.1f}"
                         for s in SCALES)
        print(f"{run:<34} {base:>6.1f} | {cells} | {best}  ({reason})")
    json.dump(out, open("/tmp/skyroute_optimal.json", "w"))
    print(f"\nwrote /tmp/skyroute_optimal.json ({len(out)} runs)")
    from collections import Counter
    print("optimal-scale distribution:", dict(Counter(out.values())))


if __name__ == "__main__":
    main()
