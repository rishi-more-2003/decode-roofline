# Nsight Compute metrics we collect (and why)

Reference for `profiling/ncu_kernels.py`. The goal is to place each decode
kernel on the roofline and prove memory-boundedness.

| Metric | Why |
| --- | --- |
| `dram__bytes.sum` | total DRAM bytes moved → with duration gives achieved GB/s, the y-axis of the roofline |
| `gpu__time_duration.sum` | kernel duration → denominator for achieved throughput |
| `sm__throughput.avg.pct_of_peak_sustained_elapsed` | compute utilization; low here + high BW% ⇒ memory-bound |
| `lts__t_sector_hit_rate.pct` | L2 hit rate; explains whether traffic actually hits DRAM |
| `sm__warps_active.avg.pct_of_peak_sustained_active` | achieved occupancy; rules out occupancy as the bottleneck |
| stall-reason breakdown | memory vs compute issue stalls → direct evidence of the binding constraint |

## Achieved DRAM throughput

```
achieved_GBps = dram__bytes.sum / gpu__time_duration.sum   # bytes / seconds
```
Compare against the **measured** peak (docs/00_environment.md), not datasheet.

## Arithmetic intensity (roofline x-axis)

```
intensity_FLOP_per_byte = total_FLOPs / dram__bytes.sum
```
For a batch-1 GEMV the intensity is tiny → far left of the ridge point →
memory-bound by construction. Phase 1 confirms this with real numbers.

## Gotchas

- `ERR_NVGPUCTRPERM` ⇒ no counter permission; fix via NVIDIA Control Panel
  (docs/00_environment.md), then reboot.
- `ncu` serializes & replays kernels; expect large wall-clock overhead. Profile
  few iterations.
