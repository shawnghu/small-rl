"""95% CIs for the token-routing in-flight rates, with the SEED as the replication unit (n=3).

Eval steps within a run are autocorrelated (same model evolving), so they are NOT independent
samples — pooling them is pseudo-replication. The honest unit is the seed. For each behavior/mode
we average each seed's rate over a late-training window (more stable per-seed estimate), then take a
between-seed t-CI (df=2). We also report a pooled-binomial Wilson CI (ignores between-seed variance —
a lower bound on uncertainty), and the paired both-minus-retain localization effect + its CI.
"""
import json, glob, sys
import numpy as np
from scipy import stats

G = "output/skyroute_token_behaviors"
DET = {"em_dash": "em_dash_detector", "bold": "excessive_bold_detector"}
N_PROMPTS = 128


def wilson(p, n, z=1.96):
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d * 100, (c + h) / d * 100)


def tci(vals):
    v = np.array(vals, float)
    n = len(v)
    m = v.mean()
    if n < 2:
        return m, float("nan"), float("nan")
    se = v.std(ddof=1) / np.sqrt(n)
    t = stats.t.ppf(0.975, n - 1)
    return m, m - t * se, m + t * se


def fmt(xs):
    return "[" + ", ".join(f"{x:.1f}" for x in xs) + "]"


for beh, det in DET.items():
    seeds = {}
    for f in sorted(glob.glob(f"{G}/skytok_{beh}_s*/routing_eval.jsonl")):
        es = [json.loads(l) for l in open(f) if l.strip()]
        if es:
            seeds[f.split("_s")[-1].split("/")[0]] = es
    if not seeds:
        print(f"[{beh}] no data")
        continue
    common = set.intersection(*[{e["step"] for e in v} for v in seeds.values()])
    if not common:
        print(f"[{beh}] no common step across seeds yet")
        continue
    last = max(common)
    W = [s for s in sorted(common) if s >= last - 60][-4:]   # late-training window (~4 evals)
    print(f"\n===== {beh}  seeds={sorted(seeds)}  window_steps={W}  =====")

    def seedrate(es, mode):
        xs = [e[f"{mode}/hack_freq/{det}"] for e in es if e["step"] in W]
        return 100.0 * np.mean(xs)

    for mode in ("both", "retain_only", "forget_only"):
        per = [seedrate(es, mode) for es in seeds.values()]
        m, lo, hi = tci(per)
        wlo, whi = wilson(m / 100, N_PROMPTS * 3 * len(W))
        print(f"  {mode:12} seeds={fmt(per)}  mean={m:5.1f}%  "
              f"95%CI(seed n=3)=[{lo:5.1f},{hi:5.1f}]  CI(binom pooled)=[{wlo:.1f},{whi:.1f}]")

    def contrast(a, b, label):
        da = [seedrate(seeds[s], a) for s in seeds]
        db = [seedrate(seeds[s], b) for s in seeds]
        diff = [x - y for x, y in zip(da, db)]
        m, lo, hi = tci(diff)
        p = stats.ttest_rel(da, db).pvalue   # two-sided
        sig = "SIG" if (lo > 0 or hi < 0) else "n.s."
        print(f"  {label:24} per-seed={fmt(diff)}  mean={m:5.1f}pp  95%CI=[{lo:5.1f},{hi:5.1f}]  "
              f"paired-t p={p:.3f}  [{sig}]")

    print("  -- contrasts (paired across seeds) --")
    contrast("both", "retain_only", "ablate forget (both-ret)")   # localization signal (confounded w/ capacity)
    contrast("both", "forget_only", "ablate retain (both-fgt)")   # if ~0 => behavior NOT in retain (clean)
    contrast("forget_only", "retain_only", "ASYMMETRY (fgt-ret)")  # >0 => forget carries more => routing worked
