"""profiling/measure_bandwidth.py — measure the DRAM bandwidth roofline ceiling.

The spec (§3) is explicit: use the MEASURED peak bandwidth as the roofline
ceiling, NOT the ~256 GB/s datasheet number. This runs simple streaming kernels
(device-to-device copy and a SAXPY/triad) over large buffers and reports the
achieved GB/s as median + IQR over many iterations (mobile parts throttle, so we
expose the spread rather than quoting a best case).

Pure-torch (no custom compile needed). Timing via CUDA events with sync.

    python profiling/measure_bandwidth.py
    python profiling/measure_bandwidth.py --mib 512 --iters 100

Writes bench/results/bandwidth.csv and prints the peak to copy into
docs/00_environment.md / roofline.py.
"""

from __future__ import annotations

import argparse
import statistics
from pathlib import Path

RESULTS = Path(__file__).resolve().parent.parent / "bench" / "results"


def _time_op(fn, iters: int, warmup: int) -> list[float]:
    """Return per-iter times in seconds using CUDA events."""
    import torch

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(iters):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) / 1e3)  # ms -> s
    return times


def _stats(times: list[float], bytes_moved: int) -> dict:
    gbps = [bytes_moved / t / 1e9 for t in times]
    gbps.sort()
    med = statistics.median(gbps)
    q1 = gbps[len(gbps) // 4]
    q3 = gbps[(3 * len(gbps)) // 4]
    return {"median_GBps": med, "q1_GBps": q1, "q3_GBps": q3,
            "iqr_GBps": q3 - q1, "max_GBps": max(gbps)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mib", type=int, default=256,
                        help="buffer size per array in MiB (default 256)")
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    args = parser.parse_args()

    import torch

    if not torch.cuda.is_available():
        print("FAIL: CUDA not available")
        return 1

    dev = torch.device("cuda")
    n = (args.mib * 1024 * 1024) // 4  # float32 elements
    x = torch.randn(n, device=dev, dtype=torch.float32)
    y = torch.randn(n, device=dev, dtype=torch.float32)
    out = torch.empty_like(x)
    nbytes = n * 4

    ops = {
        # copy: read x, write out -> 2 arrays of traffic
        "copy": (lambda: out.copy_(x), 2 * nbytes),
        # triad/saxpy: read x, read y, write out -> 3 arrays
        "triad": (lambda: torch.add(y, x, alpha=2.0, out=out), 3 * nbytes),
    }

    print(f"buffer = {args.mib} MiB/array  (n={n:,} f32)  "
          f"iters={args.iters} warmup={args.warmup}")
    rows = []
    peak = 0.0
    for name, (fn, traffic) in ops.items():
        s = _stats(_time_op(fn, args.iters, args.warmup), traffic)
        peak = max(peak, s["max_GBps"])
        rows.append((name, traffic, s))
        print(f"  {name:6s}: median {s['median_GBps']:6.1f} GB/s  "
              f"IQR [{s['q1_GBps']:.1f}, {s['q3_GBps']:.1f}]  "
              f"max {s['max_GBps']:6.1f} GB/s")

    print(f"\nMEASURED PEAK DRAM BANDWIDTH ~= {peak:.1f} GB/s")
    print("  -> record this in docs/00_environment.md and set it as the")
    print("     roofline ceiling in profiling/roofline.py")

    RESULTS.mkdir(parents=True, exist_ok=True)
    csv = RESULTS / "bandwidth.csv"
    with csv.open("w", encoding="utf-8") as f:
        f.write("op,traffic_bytes,median_GBps,q1_GBps,q3_GBps,iqr_GBps,max_GBps\n")
        for name, traffic, s in rows:
            f.write(f"{name},{traffic},{s['median_GBps']:.3f},{s['q1_GBps']:.3f},"
                    f"{s['q3_GBps']:.3f},{s['iqr_GBps']:.3f},{s['max_GBps']:.3f}\n")
    print(f"(wrote {csv})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
