"""profiling/nsys_decode.py — Phase 1: batch-1 decode workload.

A minimal native (no vLLM) batch-1 decode loop on the target model. Used for:
  1. a standalone decode-latency sanity measurement (median + IQR, CUDA events),
  2. the workload that `nsys` profiles into a timeline, and
  3. the workload that `ncu` profiles per-kernel.

Why not vLLM: vLLM has no native Windows support, and profiling the same
native environment the custom kernel lives in gives cleaner attribution. The
memory-bound roofline conclusion does not depend on PagedAttention. (A WSL+vLLM
PagedAttention study is an optional stretch — see docs/01_roofline.md.)

Standalone:
    python profiling/nsys_decode.py --model Qwen/Qwen2.5-1.5B --decode-steps 64

Under nsys (timeline + kernel summary):
    MSYS_NO_PATHCONV=1 cmd.exe /c "nsys profile --stats=true -f true \\
        -o bench/results/nsys_decode \\
        python profiling/nsys_decode.py --decode-steps 64 --no-latency-report"
"""

from __future__ import annotations

import argparse
import statistics
import time


def load_model(model_name: str, dtype):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        attn_implementation="sdpa",
    ).to("cuda").eval()
    return tok, model


def run_decode(model, input_ids, decode_steps: int):
    """Prefill then decode `decode_steps` tokens (batch 1, greedy). Returns
    per-step decode latencies in milliseconds (CUDA-event timed)."""
    import torch

    lat_ms = []
    with torch.no_grad():
        out = model(input_ids=input_ids, use_cache=True)  # prefill
        past = out.past_key_values
        next_tok = out.logits[:, -1:].argmax(dim=-1)
        torch.cuda.synchronize()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        for _ in range(decode_steps):
            start.record()
            out = model(input_ids=next_tok, past_key_values=past, use_cache=True)
            past = out.past_key_values
            next_tok = out.logits[:, -1:].argmax(dim=-1)
            end.record()
            torch.cuda.synchronize()
            lat_ms.append(start.elapsed_time(end))
    return lat_ms


def _iqr(xs: list[float]) -> tuple[float, float, float]:
    s = sorted(xs)
    return statistics.median(s), s[len(s) // 4], s[(3 * len(s)) // 4]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--prompt", default="The history of computing began")
    parser.add_argument("--decode-steps", type=int, default=64)
    parser.add_argument("--warmup-steps", type=int, default=16)
    parser.add_argument("--no-latency-report", action="store_true",
                        help="skip the timing sweep (for nsys/ncu capture runs)")
    args = parser.parse_args()

    import torch

    if not torch.cuda.is_available():
        print("FAIL: CUDA not available")
        return 1

    dtype = torch.float16
    t0 = time.time()
    tok, model = load_model(args.model, dtype)
    print(f"loaded {args.model} ({dtype}) in {time.time()-t0:.1f}s; "
          f"VRAM used {torch.cuda.memory_allocated()/2**30:.2f} GiB")

    input_ids = tok(args.prompt, return_tensors="pt").input_ids.to("cuda")
    print(f"prompt tokens: {input_ids.shape[1]}")

    if args.no_latency_report:
        # Minimal fixed run for profilers; warmup then a few steps.
        run_decode(model, input_ids, args.warmup_steps)
        run_decode(model, input_ids, args.decode_steps)
        print(f"profiled decode: {args.decode_steps} steps")
        return 0

    run_decode(model, input_ids, args.warmup_steps)  # warmup
    lat = run_decode(model, input_ids, args.decode_steps)
    med, q1, q3 = _iqr(lat)
    print(f"\ndecode latency over {len(lat)} steps (batch 1):")
    print(f"  median {med:.3f} ms/token  IQR [{q1:.3f}, {q3:.3f}]  "
          f"min {min(lat):.3f}  max {max(lat):.3f}")
    print(f"  => {1e3/med:.1f} tok/s (median)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
