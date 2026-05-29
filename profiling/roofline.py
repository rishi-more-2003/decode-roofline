"""profiling/roofline.py — Phase 1: build the RTX 4070 Laptop roofline plot.

Parses the per-kernel ncu CSV (profiling/ncu_kernels.py output; the raw file has
a few lines of program stdout prepended, which we skip) and places each decode
kernel on a roofline whose memory ceiling is the MEASURED peak bandwidth
(docs/00_environment.md), not the 256 GB/s datasheet.

For a batch-1 GEMV `y = W x` with W in [N,K]:
  FLOPs  = 2*N*K
  bytes  ~= sizeof(dtype)*N*K   (weights dominate; x,y negligible)
  => arithmetic intensity ~= 2/sizeof(dtype)  (1.0 FLOP/byte for fp16)
which sits far left of the ridge point => memory-bound by construction. We use
the ncu-measured `dram__bytes.sum` for the real byte count and
`gpu__time_duration.sum` for achieved GB/s.

    python profiling/roofline.py --ncu-csv bench/results/ncu_gemv.csv

Outputs bench/results/roofline.png and bench/results/ncu_kernels.csv (tidy).
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

RESULTS = Path(__file__).resolve().parent.parent / "bench" / "results"

# Measured on this machine (docs/00_environment.md), NOT the 256 GB/s datasheet.
# Best measured *achievable* DRAM bandwidth: cuBLAS decode GEMVs reach ~247 GB/s
# (ncu dram__bytes counter); we round to 250 as the achievable ceiling. ncu's
# internal hardware-theoretical peak is ~259 GB/s (247 / 0.953).
MEASURED_PEAK_BW_GBPS: float = 250.0

# Approx FP16 CUDA-core (non-tensor) compute peak for the 4070 Laptop:
# 4608 cores * 2 (FMA) * ~2.1 GHz ~= 19.3 TFLOP/s FP32; FP16 packed ~2x.
# The gemvx kernels are CUDA-core (not tensor-core), so this is the relevant
# ceiling. Exact value does not change the memory-bound conclusion (ridge is
# ~80-160 FLOP/byte; GEMV intensity is ~1).
PEAK_COMPUTE_GFLOPS = 19300.0


def _load_ncu_csv(path: Path) -> list[dict]:
    """Read ncu --csv output, skipping any non-CSV preamble lines."""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    start = next(i for i, ln in enumerate(lines) if ln.startswith('"ID"'))
    reader = csv.DictReader(lines[start:])
    return list(reader)


def _f(s: str) -> float:
    return float(s.replace(",", ""))


def parse_kernels(rows: list[dict]) -> list[dict]:
    """Pivot ncu long-format rows into one record per kernel launch (by ID)."""
    by_id: dict[str, dict] = defaultdict(dict)
    for r in rows:
        kid = r["ID"]
        rec = by_id[kid]
        rec.setdefault("name", r["Kernel Name"])
        rec.setdefault("grid", r["Grid Size"])
        rec.setdefault("block", r["Block Size"])
        rec[r["Metric Name"]] = _f(r["Metric Value"])

    out = []
    for kid, rec in by_id.items():
        dram = rec.get("dram__bytes.sum", 0.0)
        dur_ns = rec.get("gpu__time_duration.sum", 0.0)
        if dur_ns <= 0:
            continue
        gbps = dram / dur_ns  # bytes/ns == GB/s
        is_half = "__half" in rec["name"]
        ai = 1.0 if is_half else 0.5  # FLOP/byte for a GEMV
        out.append({
            "id": int(kid),
            "dtype": "fp16" if is_half else "fp32",
            "grid": rec["grid"],
            "dram_MB": dram / 1e6,
            "dur_us": dur_ns / 1e3,
            "achieved_GBps": gbps,
            "dram_pct_peak": rec.get(
                "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed", 0.0),
            "sm_pct": rec.get(
                "sm__throughput.avg.pct_of_peak_sustained_elapsed", 0.0),
            "l2_hit_pct": rec.get("lts__t_sector_hit_rate.pct", 0.0),
            "occupancy_pct": rec.get(
                "sm__warps_active.avg.pct_of_peak_sustained_active", 0.0),
            "ai_flop_per_byte": ai,
            "achieved_GFLOPs": gbps * ai,
        })
    return sorted(out, key=lambda d: d["dram_MB"], reverse=True)


def write_summary(kernels: list[dict], path: Path) -> None:
    cols = ["id", "dtype", "grid", "dram_MB", "dur_us", "achieved_GBps",
            "dram_pct_peak", "sm_pct", "l2_hit_pct", "occupancy_pct",
            "ai_flop_per_byte", "achieved_GFLOPs"]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for k in kernels:
            w.writerow({c: (f"{k[c]:.3f}" if isinstance(k[c], float) else k[c])
                        for c in cols})


def plot_roofline(kernels: list[dict], out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(8, 6))
    ai = np.logspace(-2, 3, 200)
    mem_ceiling = MEASURED_PEAK_BW_GBPS * ai          # GB/s * FLOP/byte = GFLOP/s
    comp_ceiling = np.full_like(ai, PEAK_COMPUTE_GFLOPS)
    roof = np.minimum(mem_ceiling, comp_ceiling)
    ridge = PEAK_COMPUTE_GFLOPS / MEASURED_PEAK_BW_GBPS

    ax.plot(ai, roof, "k-", lw=2, label="roofline")
    ax.plot(ai, mem_ceiling, "b--", lw=1, alpha=0.5,
            label=f"memory ceiling = {MEASURED_PEAK_BW_GBPS:.0f} GB/s (measured)")
    ax.axhline(PEAK_COMPUTE_GFLOPS, color="r", ls="--", lw=1, alpha=0.5,
               label=f"compute ceiling ~ {PEAK_COMPUTE_GFLOPS/1e3:.0f} TFLOP/s (fp16, approx)")
    ax.axvline(ridge, color="gray", ls=":", lw=1, alpha=0.7,
               label=f"ridge ~ {ridge:.0f} FLOP/byte")

    # Bucket kernels for a readable legend.
    def bucket(k):
        if k["dtype"] == "fp32":
            return ("attention score/av (fp32, tiny)", "tab:gray")
        if k["dram_MB"] > 15:
            return ("MLP gate/up/down GEMV (~27 MB)", "tab:red")
        if k["dram_MB"] > 2:
            return ("attn q/o proj GEMV (~4.7 MB)", "tab:orange")
        return ("KV proj GEMV (~0.8 MB)", "tab:green")

    seen = set()
    for k in kernels:
        label, color = bucket(k)
        ax.scatter(k["ai_flop_per_byte"], k["achieved_GFLOPs"], s=70, color=color,
                   edgecolor="k", zorder=5,
                   label=label if label not in seen else None)
        seen.add(label)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Arithmetic intensity (FLOP/byte)")
    ax.set_ylabel("Performance (GFLOP/s)")
    ax.set_title("RTX 4070 Laptop — decode kernels vs roofline (batch 1, Qwen2.5-1.5B)")
    ax.grid(True, which="both", alpha=0.2)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print(f"(wrote {out})")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ncu-csv", default=str(RESULTS / "ncu_gemv.csv"))
    parser.add_argument("--out", default=str(RESULTS / "roofline.png"))
    args = parser.parse_args()

    rows = _load_ncu_csv(Path(args.ncu_csv))
    kernels = parse_kernels(rows)
    if not kernels:
        print("no kernels parsed from", args.ncu_csv)
        return 1

    print(f"parsed {len(kernels)} kernel launches; "
          f"memory ceiling = {MEASURED_PEAK_BW_GBPS:.0f} GB/s")
    print(f"{'dram_MB':>8} {'dur_us':>8} {'GB/s':>7} {'%peakBW':>8} "
          f"{'sm%':>6} {'L2%':>6} {'occ%':>6}")
    for k in kernels:
        print(f"{k['dram_MB']:8.2f} {k['dur_us']:8.1f} {k['achieved_GBps']:7.1f} "
              f"{k['dram_pct_peak']:8.1f} {k['sm_pct']:6.1f} {k['l2_hit_pct']:6.1f} "
              f"{k['occupancy_pct']:6.1f}")

    write_summary(kernels, RESULTS / "ncu_kernels.csv")
    print(f"(wrote {RESULTS / 'ncu_kernels.csv'})")
    plot_roofline(kernels, Path(args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
