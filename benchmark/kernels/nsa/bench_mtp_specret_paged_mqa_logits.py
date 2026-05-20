"""
Micro-benchmark for the MTP speculative-retrieval (specret) optimization in
NSA Indexer._get_topk_paged.

It only times the inner deep_gemm.fp8_paged_mqa_logits call on the two paths:

  baseline (use_mtp_specret = False):
      q.shape = [B * draft_token_num, 1, num_heads, head_dim]
      seqlens.shape = [B * draft_token_num, 1]
      block_tables.shape = [B * draft_token_num, max_blocks]

  specret  (use_mtp_specret = True):
      q.shape = [B, 1, num_heads, head_dim]                # only the 1st sibling
      seqlens.shape = [B, 1]                               # only the 1st sibling
      block_tables.shape = [B, max_blocks]

The two paths produce different shapes, but the saved compute is exactly
draft_token_num x (per-q work). This script reports the latency ratio so the
specret speedup can be observed without running an end-to-end model.

Usage:
    python bench_mtp_specret_paged_mqa_logits.py \\
        --batch-size 8 --draft-token-num 4 \\
        --seq-len 4096 --num-heads 64 --head-dim 128
"""

from __future__ import annotations

import argparse
import statistics
from dataclasses import dataclass
from typing import List, Tuple

import torch

try:
    import deep_gemm
except ImportError as e:  # pragma: no cover - benchmark requires deep_gemm
    raise SystemExit(f"deep_gemm is required for this benchmark: {e}")


# Layout constants matching nsa_indexer.py / DeepGEMM 0426
PAGE_SIZE = 64
BLOCK_KV = PAGE_SIZE
NUM_HEADS_KV = 1
HEAD_DIM_WITH_SF = 132  # 128 fp8 elems + 4 fp32 scale bytes per page row


@dataclass
class BenchInputs:
    q_baseline: torch.Tensor          # [B*d, 1, H, D] fp8
    q_specret: torch.Tensor           # [B,   1, H, D] fp8
    weights_baseline: torch.Tensor    # [B*d, H]      fp32
    weights_specret: torch.Tensor     # [B,   H]      fp32
    kv_cache: torch.Tensor            # [num_pages, block_kv, 1, head_dim_with_sf]
    seqlens_baseline: torch.Tensor    # [B*d, 1] int32
    seqlens_specret: torch.Tensor     # [B,   1] int32
    block_tables_baseline: torch.Tensor  # [B*d, max_blocks] int32
    block_tables_specret: torch.Tensor   # [B,   max_blocks] int32
    schedule_baseline: torch.Tensor
    schedule_specret: torch.Tensor
    max_seq_len: int


def _make_inputs(
    batch_size: int,
    draft_token_num: int,
    seq_len: int,
    num_heads: int,
    head_dim: int,
    device: torch.device,
    seed: int,
) -> BenchInputs:
    """Build inputs that mirror NSA Indexer._get_topk_paged at the moment of
    the deep_gemm.fp8_paged_mqa_logits call."""
    assert head_dim == 128, "deep_gemm fp8_paged_mqa_logits hard-codes head_dim=128"

    torch.manual_seed(seed)
    g = torch.Generator(device=device).manual_seed(seed)

    total_q = batch_size * draft_token_num
    max_blocks = (seq_len + BLOCK_KV - 1) // BLOCK_KV
    max_seq_len = max_blocks * BLOCK_KV
    sm_count = deep_gemm.get_num_sms()

    # ---- q / weights ----
    q_baseline_bf = torch.randn(
        total_q, 1, num_heads, head_dim, generator=g, device=device, dtype=torch.bfloat16
    )
    q_baseline = q_baseline_bf.to(torch.float8_e4m3fn)

    # specret takes only the first sibling of each draft group
    q_specret = (
        q_baseline.view(batch_size, draft_token_num, 1, num_heads, head_dim)[:, 0]
        .contiguous()
    )

    weights_baseline = torch.randn(
        total_q, num_heads, generator=g, device=device, dtype=torch.float32
    )
    weights_specret = (
        weights_baseline.view(batch_size, draft_token_num, num_heads)[:, 0]
        .contiguous()
    )

    # ---- kv cache (single layer, packed fp8 + per-block scales) ----
    # DeepGEMM requires the fused kv-cache buffer to be exposed as torch.uint8
    # (torch::kByte). The packed layout per page is:
    #   bytes [0 : block_kv * head_dim)              -> fp8 (e4m3) values
    #   bytes [block_kv * head_dim : block_kv * 132) -> fp32 per-row scales
    # In NSA this corresponds to NSATokenToKVPool.index_k_with_scale_buffer.
    num_pages = batch_size * max_blocks + 1  # +1 because page id 0 is unused
    kv_cache = torch.randint(
        0,
        255,
        (num_pages, BLOCK_KV, NUM_HEADS_KV, HEAD_DIM_WITH_SF),
        generator=g,
        device=device,
        dtype=torch.uint8,
    )

    # ---- block tables ----
    # Each (batch, draft) row references the same KV pages because all siblings
    # in an MTP group share the prefix KV cache; that's exactly why specret can
    # collapse them into one query.
    block_tables_specret = torch.zeros(
        batch_size, max_blocks, dtype=torch.int32, device=device
    )
    for b in range(batch_size):
        block_tables_specret[b] = torch.arange(
            b * max_blocks + 1, (b + 1) * max_blocks + 1, dtype=torch.int32, device=device
        )
    block_tables_baseline = block_tables_specret.repeat_interleave(draft_token_num, dim=0)

    # ---- seqlens (2D, matching deep_gemm 0426 layout) ----
    seqlens_specret = torch.full(
        (batch_size, 1), seq_len, dtype=torch.int32, device=device
    )
    seqlens_baseline = torch.full(
        (total_q, 1), seq_len, dtype=torch.int32, device=device
    )

    schedule_baseline = deep_gemm.get_paged_mqa_logits_metadata(
        seqlens_baseline, BLOCK_KV, sm_count
    )
    schedule_specret = deep_gemm.get_paged_mqa_logits_metadata(
        seqlens_specret, BLOCK_KV, sm_count
    )

    return BenchInputs(
        q_baseline=q_baseline,
        q_specret=q_specret,
        weights_baseline=weights_baseline,
        weights_specret=weights_specret,
        kv_cache=kv_cache,
        seqlens_baseline=seqlens_baseline,
        seqlens_specret=seqlens_specret,
        block_tables_baseline=block_tables_baseline,
        block_tables_specret=block_tables_specret,
        schedule_baseline=schedule_baseline,
        schedule_specret=schedule_specret,
        max_seq_len=max_seq_len,
    )


def _run_paged_mqa_logits(
    q: torch.Tensor,
    kv: torch.Tensor,
    weights: torch.Tensor,
    seqlens_2d: torch.Tensor,
    block_tables: torch.Tensor,
    schedule: torch.Tensor,
    max_seq_len: int,
) -> torch.Tensor:
    return deep_gemm.fp8_paged_mqa_logits(
        q,
        kv,
        weights,
        seqlens_2d,
        block_tables,
        schedule,
        max_seq_len,
        clean_logits=False,
    )


def _bench(
    fn,
    *,
    warmup: int,
    iters: int,
    device: torch.device,
) -> List[float]:
    # Warmup
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize(device)

    # Per-iteration CUDA-event timing
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize(device)
    return [s.elapsed_time(e) for s, e in zip(starts, ends)]


def _summarize(label: str, samples_ms: List[float]) -> Tuple[float, float, float]:
    samples_ms_sorted = sorted(samples_ms)
    median = statistics.median(samples_ms_sorted)
    mean = statistics.fmean(samples_ms_sorted)
    p10 = samples_ms_sorted[int(0.1 * len(samples_ms_sorted))]
    print(
        f"  {label:<10s}  median={median:8.4f} ms  mean={mean:8.4f} ms  "
        f"min={samples_ms_sorted[0]:8.4f} ms  p10={p10:8.4f} ms"
    )
    return median, mean, samples_ms_sorted[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--draft-token-num", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--num-heads", type=int, default=64)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--sweep-draft",
        type=int,
        nargs="?",
        const=4,
        default=None,
        metavar="MAX",
        help=(
            "Sweep draft_token_num from 2 up to MAX (inclusive) and report ratios. "
            "Pass without value to use the default max=4."
        ),
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required.")
    device = torch.device("cuda")

    if args.sweep_draft is not None:
        if args.sweep_draft < 2:
            raise SystemExit("--sweep-draft MAX must be >= 2")
        draft_list = list(range(2, args.sweep_draft + 1))
    else:
        draft_list = [args.draft_token_num]

    print(
        f"NSA MTP-specret paged_mqa_logits micro-bench  "
        f"(B={args.batch_size}, S={args.seq_len}, H={args.num_heads}, D={args.head_dim}, "
        f"warmup={args.warmup}, iters={args.iters})"
    )
    print("-" * 88)

    for d in draft_list:
        inp = _make_inputs(
            batch_size=args.batch_size,
            draft_token_num=d,
            seq_len=args.seq_len,
            num_heads=args.num_heads,
            head_dim=args.head_dim,
            device=device,
            seed=args.seed,
        )

        def baseline_fn():
            _run_paged_mqa_logits(
                inp.q_baseline,
                inp.kv_cache,
                inp.weights_baseline,
                inp.seqlens_baseline,
                inp.block_tables_baseline,
                inp.schedule_baseline,
                inp.max_seq_len,
            )

        def specret_fn():
            _run_paged_mqa_logits(
                inp.q_specret,
                inp.kv_cache,
                inp.weights_specret,
                inp.seqlens_specret,
                inp.block_tables_specret,
                inp.schedule_specret,
                inp.max_seq_len,
            )

        print(f"draft_token_num={d}  (q rows: baseline={args.batch_size * d}, specret={args.batch_size})")
        baseline_samples = _bench(
            baseline_fn, warmup=args.warmup, iters=args.iters, device=device
        )
        specret_samples = _bench(
            specret_fn, warmup=args.warmup, iters=args.iters, device=device
        )

        b_med, _, _ = _summarize("baseline", baseline_samples)
        s_med, _, _ = _summarize("specret", specret_samples)
        print(f"  speedup (median)  = {b_med / s_med:.3f}x  (ideal ~ {d}x)\n")


if __name__ == "__main__":
    main()
