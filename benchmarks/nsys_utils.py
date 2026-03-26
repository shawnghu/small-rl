"""Shared nsys profiling utilities: sqlite export and GPU summary extraction."""

import collections
import os
import subprocess


def export_sqlite(nsys_rep_path):
    """Convert .nsys-rep to .sqlite, returning the sqlite path."""
    sqlite_path = nsys_rep_path.replace(".nsys-rep", ".sqlite")
    cmd = ["nsys", "export", "--type=sqlite", f"--output={sqlite_path}", nsys_rep_path]
    print("Exporting to SQLite...", flush=True)
    subprocess.run(cmd, capture_output=True)
    return sqlite_path


def extract_summary(sqlite_path, header_lines, output_path=None):
    """Extract GPU metrics summary from an nsys sqlite export.

    Args:
        sqlite_path: Path to the .sqlite file from nsys export.
        header_lines: List of strings to print at the top of the summary.
        output_path: If set, write summary to this file.

    Returns:
        The summary string.
    """
    import sqlite3

    conn = sqlite3.connect(sqlite_path)

    lines = []
    lines.append("=" * 70)
    lines.extend(header_lines)
    lines.append("=" * 70)

    # Duration from GPU metrics
    try:
        mn, mx = conn.execute("SELECT MIN(timestamp), MAX(timestamp) FROM GPU_METRICS").fetchone()
        dur = (mx - mn) / 1e9
        lines.append(f"Profile duration (GPU metrics): {dur:.1f}s")
    except Exception:
        dur = 0
        lines.append("No GPU metrics found (need root for --gpu-metrics-devices)")

    # GPU utilization averages
    if dur > 0:
        tid = conn.execute("SELECT DISTINCT typeId FROM GPU_METRICS").fetchone()[0]
        # Skip first 10% as warmup, but no more than 30s
        warmup_ns = min(30_000_000_000, int(dur * 0.1 * 1_000_000_000))
        t_start = mn + warmup_ns
        metrics = {
            3: "SMs Active", 4: "SM Issue", 5: "Tensor Active",
            12: "Compute Warps in Flight", 15: "Unallocated Warps in Active SMs",
            18: "DRAM Read BW", 19: "DRAM Write BW",
        }
        warmup_s = warmup_ns / 1e9
        lines.append(f"\n--- GPU Averages (after {warmup_s:.1f}s warmup) ---")
        for mid, name in metrics.items():
            r = conn.execute(
                "SELECT AVG(value) FROM GPU_METRICS WHERE typeId=? AND metricId=? AND timestamp>=?",
                (tid, mid, t_start),
            ).fetchone()
            val = r[0] if r[0] else 0
            lines.append(f"  {name:>40}: {val:.1f}%")

        # Time series (10s bins)
        data = conn.execute(
            "SELECT timestamp, metricId, value FROM GPU_METRICS WHERE typeId=? AND metricId IN (3,5,12,18,19) ORDER BY timestamp",
            (tid,),
        ).fetchall()
        bins = collections.defaultdict(lambda: collections.defaultdict(list))
        for ts, mid, val in data:
            bucket = int((ts - mn) / 10_000_000_000) * 10
            bins[bucket][mid].append(val)

        lines.append(f"\n--- GPU Utilization Time Series (10s bins) ---")
        lines.append(f"{'Time':>8}  {'SMs Active':>11}  {'Tensor':>8}  {'Compute':>9}  {'DRAM Rd':>9}  {'DRAM Wr':>9}")
        for t in sorted(bins.keys()):
            b = bins[t]
            avg = lambda mid, _b=b: sum(_b[mid]) / len(_b[mid]) if _b[mid] else 0
            lines.append(f"{t:>6}s  {avg(3):>10.1f}%  {avg(5):>7.1f}%  {avg(12):>8.1f}%  {avg(18):>8.1f}%  {avg(19):>8.1f}%")

    # Top kernels
    try:
        rows = conn.execute("""
            SELECT demangledName, COUNT(*) as cnt, SUM(end-start) as total_ns, AVG(end-start) as avg_ns
            FROM CUPTI_ACTIVITY_KIND_KERNEL
            GROUP BY demangledName ORDER BY total_ns DESC LIMIT 15
        """).fetchall()
        total_kern = conn.execute("SELECT SUM(end-start) FROM CUPTI_ACTIVITY_KIND_KERNEL").fetchone()[0]
        lines.append(f"\n--- Top 15 GPU Kernels (total kernel time: {total_kern/1e9:.1f}s) ---")
        for nameId, cnt, total_ns, avg_ns in rows:
            if isinstance(nameId, int):
                r = conn.execute("SELECT value FROM StringIds WHERE id=?", (nameId,)).fetchone()
                name = r[0] if r else str(nameId)
            else:
                name = nameId
            short = (name[:80] + "...") if len(name) > 80 else name
            pct = total_ns / total_kern * 100
            lines.append(f"  {pct:>5.1f}%  {cnt:>8}x  {total_ns/1e9:>7.2f}s  {avg_ns/1e3:>8.1f}us  {short}")
    except Exception as e:
        lines.append(f"\nNo kernel data: {e}")

    conn.close()

    summary = "\n".join(lines)
    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w") as f:
            f.write(summary + "\n")
        print(summary)
        print(f"\nSummary written to {output_path}")
    else:
        print(summary)

    return summary
