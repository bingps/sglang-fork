"""
End-to-end indexer micro-benchmark for the MTP speculative-retrieval (specret)
optimization in NSA Indexer._get_topk_paged.

Unlike bench_mtp_specret_paged_mqa_logits.py, which only times the inner
deep_gemm.fp8_paged_mqa_logits call, this benchmark directly calls

    sglang.srt.layers.attention.nsa.nsa_indexer.Indexer._get_topk_paged

so that ALL operators on the paged path are measured exactly as they run in
production:

  1. deep_gemm.get_paged_mqa_logits_metadata  (schedule)
  2. deep_gemm.fp8_paged_mqa_logits           (logits)
  3. NSAIndexerMetadata.topk_transform
       -> sgl_kernel.fast_topk_transform_fused (PAGED method)
  4. (specret only) topk_result.repeat_interleave(d)
  5. (specret only) sibling-self-KV patch
       -> Indexer._patch_mtp_specret_sibling_self_kv

Three paths are benchmarked:
  - baseline:         enable_mtp_specret=False, q/weights have B*d rows
  - specret:          enable_mtp_specret=True, q/weights have B*d rows
                      (_get_topk_paged reshapes+contiguous internally)
  - specret_preslice: enable_mtp_specret=True, q/weights pre-sliced to B rows
                      (_get_topk_paged skips reshape+contiguous)

Usage:
    source .venv/bin/activate
    python bench_mtp_specret_indexer.py \\
        --batch-size 8 --draft-token-num 4 \\
        --seq-len 4096 --num-heads 64 --head-dim 128
"""

from __future__ import annotations

import argparse
import dataclasses
import statistics
from dataclasses import dataclass
from typing import Callable, List, Tuple

import torch

try:
    import deep_gemm
except ImportError as e:  # pragma: no cover - benchmark requires deep_gemm
    raise SystemExit(f"deep_gemm is required for this benchmark: {e}")

# Direct import of the production indexer + metadata.
from sglang.srt.layers.attention.nsa.nsa_indexer import Indexer
from sglang.srt.layers.attention.nsa_backend import (
    NSAIndexerMetadata,
    NSAMetadata,
    TopkTransformMethod,
)


# Layout constants matching nsa_indexer.py / DeepGEMM 0426
PAGE_SIZE = 64
BLOCK_KV = PAGE_SIZE
NUM_HEADS_KV = 1
HEAD_DIM_WITH_SF = 132  # 128 fp8 elems + 4 fp32 scale bytes per page row
INDEX_TOPK = 2048  # NSA hard-codes topk=2048 in fast_topk_transform_fused


# ---------------------------------------------------------------------------
# Duck-typed stubs for ForwardBatch / token_to_kv_pool / forward_mode
# ---------------------------------------------------------------------------


class _StubForwardMode:
    """Mirrors ForwardMode.is_target_verify / is_draft_extend used by
    Indexer._get_topk_paged."""

    def is_target_verify(self) -> bool:
        return True

    def is_draft_extend(self, include_v2: bool = False) -> bool:
        return False


class _StubKVPool:
    """Mirrors NSATokenToKVPool fields used by Indexer._get_topk_paged:
    only `page_size` and `get_index_k_with_scale_buffer(layer_id=...)`."""

    def __init__(self, kv_cache_2d: torch.Tensor, page_size: int = PAGE_SIZE):
        self._kv_cache_2d = kv_cache_2d
        self.page_size = page_size

    def get_index_k_with_scale_buffer(self, layer_id: int) -> torch.Tensor:
        return self._kv_cache_2d


class _StubForwardBatch:
    """Mirrors the ForwardBatch attributes touched by Indexer._get_topk_paged."""

    def __init__(self, kv_cache_2d: torch.Tensor):
        self.token_to_kv_pool = _StubKVPool(kv_cache_2d)
        self.forward_mode = _StubForwardMode()


# ---------------------------------------------------------------------------
# Synthetic NSA metadata
# ---------------------------------------------------------------------------


@dataclass
class BenchTensors:
    # ---- shared kv cache ----
    kv_cache_2d: torch.Tensor               # [num_pages, block_kv * head_dim_with_sf] uint8
    max_seq_len: int

    # ---- baseline (no specret) ----
    q_baseline: torch.Tensor                # [B*d, H, D] fp8
    weights_baseline: torch.Tensor          # [B*d, H, 1] fp32
    seqlens_expanded_baseline: torch.Tensor # [B*d] int32
    block_tables_baseline: torch.Tensor     # [B*d, max_blocks] int32 (page64)
    page_table_1_baseline: torch.Tensor     # [B*d, max_seq_len] int32 (page1)
    cu_seqlens_q_baseline: torch.Tensor     # [B*d + 1] int32

    # ---- specret (full B*d input, reshapes internally) ----
    q_specret: torch.Tensor             # [B*d, H, D] fp8
    weights_specret: torch.Tensor       # [B*d, H, 1] fp32

    # ---- specret preslice (pre-sliced B rows) ----
    q_specret_preslice: torch.Tensor             # [B, H, D] fp8 (first token only)
    weights_specret_preslice: torch.Tensor       # [B, H, 1] fp32 (first token only)

    # ---- specret shared metadata ----
    mtp_specret_seqlens: torch.Tensor       # [B] int32
    mtp_specret_real_page_table: torch.Tensor   # [B, max_blocks] int32
    mtp_specret_page_table_1: torch.Tensor      # [B, max_seq_len] int32
    mtp_specret_cu_seqlens_q: torch.Tensor      # [B + 1] int32
    mtp_specret_sibling_rows: torch.Tensor      # [B*(d-1)] int64
    mtp_specret_sibling_self_page_indices: torch.Tensor  # [B*(d-1)] int32
    mtp_specret_first_token_indices: torch.Tensor  # [B] int64

    # ---- shape book-keeping ----
    batch_size: int
    draft_token_num: int


def _build_inputs(
    batch_size: int,
    draft_token_num: int,
    seq_len: int,
    num_heads: int,
    head_dim: int,
    device: torch.device,
    seed: int,
) -> BenchTensors:
    assert head_dim == 128, "deep_gemm fp8_paged_mqa_logits hard-codes head_dim=128"
    assert draft_token_num >= 2, "specret only kicks in when draft_token_num > 1"

    torch.manual_seed(seed)
    g = torch.Generator(device=device).manual_seed(seed)

    total_q = batch_size * draft_token_num
    max_blocks = (seq_len + BLOCK_KV - 1) // BLOCK_KV
    max_seq_len = max_blocks * BLOCK_KV

    # Indexer._get_topk_paged expects q_fp8 with shape (N, H, D), then unsqueezes
    # next_n=1 internally, and weights with shape (N, H, 1) which it squeezes.
    q_bf = torch.randn(
        total_q, num_heads, head_dim,
        generator=g, device=device, dtype=torch.bfloat16,
    )
    q_fp8 = q_bf.to(torch.float8_e4m3fn)

    weights = torch.randn(
        total_q, num_heads, 1, generator=g, device=device, dtype=torch.float32
    )

    # Pre-sliced first-token tensors for the new specret path
    first_idx = torch.arange(
        0, total_q, draft_token_num, device=device, dtype=torch.int64
    )
    q_specret_preslice = q_fp8[first_idx].contiguous()       # (B, H, D)
    weights_specret_preslice = weights[first_idx].contiguous()  # (B, H, 1)

    # Packed fp8 + per-block scales kv-cache buffer, exposed as 2D uint8 to
    # match NSATokenToKVPool.get_index_k_with_scale_buffer's contract.
    num_pages = batch_size * max_blocks + 1  # +1 because page id 0 is unused
    kv_cache_2d = torch.randint(
        0, 255,
        (num_pages, BLOCK_KV * HEAD_DIM_WITH_SF),
        generator=g, device=device, dtype=torch.uint8,
    )

    # ---- page-64 block tables (per-batch and per-token expanded) ----
    block_tables_per_batch = torch.zeros(
        batch_size, max_blocks, dtype=torch.int32, device=device
    )
    for b in range(batch_size):
        block_tables_per_batch[b] = torch.arange(
            b * max_blocks + 1,
            (b + 1) * max_blocks + 1,
            dtype=torch.int32, device=device,
        )
    block_tables_baseline = block_tables_per_batch.repeat_interleave(
        draft_token_num, dim=0
    )

    # ---- page-1 page tables ----
    page_table_1_specret = torch.arange(
        0, max_seq_len, dtype=torch.int32, device=device
    ).unsqueeze(0).expand(batch_size, -1).contiguous()
    page_table_1_baseline = torch.arange(
        0, max_seq_len, dtype=torch.int32, device=device
    ).unsqueeze(0).expand(total_q, -1).contiguous()

    # ---- seqlens ----
    seqlens_expanded_baseline = torch.full(
        (total_q,), seq_len, dtype=torch.int32, device=device
    )
    mtp_specret_seqlens = torch.full(
        (batch_size,), seq_len, dtype=torch.int32, device=device
    )

    # ---- cu_seqlens_q ----
    cu_seqlens_q_baseline = torch.arange(
        0, total_q + 1, dtype=torch.int32, device=device
    )
    mtp_specret_cu_seqlens_q = torch.arange(
        0, batch_size + 1, dtype=torch.int32, device=device
    )

    # ---- sibling-self-kv patch metadata ----
    base = torch.arange(batch_size, device=device, dtype=torch.int64) * draft_token_num
    sibling_offsets = torch.arange(
        1, draft_token_num, device=device, dtype=torch.int64
    )
    sibling_rows = (base.unsqueeze(1) + sibling_offsets.unsqueeze(0)).reshape(-1)
    sibling_self_page_indices = torch.full(
        (sibling_rows.numel(),), seq_len - 1, dtype=torch.int32, device=device
    )

    return BenchTensors(
        kv_cache_2d=kv_cache_2d,
        max_seq_len=max_seq_len,
        q_baseline=q_fp8,
        weights_baseline=weights,
        seqlens_expanded_baseline=seqlens_expanded_baseline,
        block_tables_baseline=block_tables_baseline,
        page_table_1_baseline=page_table_1_baseline,
        cu_seqlens_q_baseline=cu_seqlens_q_baseline,
        q_specret=q_fp8,
        weights_specret=weights,
        q_specret_preslice=q_specret_preslice,
        weights_specret_preslice=weights_specret_preslice,
        mtp_specret_seqlens=mtp_specret_seqlens,
        mtp_specret_real_page_table=block_tables_per_batch,
        mtp_specret_page_table_1=page_table_1_specret,
        mtp_specret_cu_seqlens_q=mtp_specret_cu_seqlens_q,
        mtp_specret_sibling_rows=sibling_rows,
        mtp_specret_sibling_self_page_indices=sibling_self_page_indices,
        mtp_specret_first_token_indices=first_idx,
        batch_size=batch_size,
        draft_token_num=draft_token_num,
    )


def _make_nsa_metadata_baseline(t: BenchTensors) -> NSAMetadata:
    """Minimal NSAMetadata for the baseline (non-specret) path."""
    total_q = t.batch_size * t.draft_token_num
    return NSAMetadata(
        page_size=PAGE_SIZE,
        cache_seqlens_int32=t.seqlens_expanded_baseline,
        max_seq_len_q=1,
        max_seq_len_k=t.max_seq_len,
        cu_seqlens_q=t.cu_seqlens_q_baseline,
        cu_seqlens_k=torch.cumsum(
            torch.cat([
                torch.zeros(1, dtype=torch.int32, device=t.kv_cache_2d.device),
                t.seqlens_expanded_baseline,
            ]),
            dim=0,
        ).to(torch.int32),
        page_table_1=t.page_table_1_baseline,
        real_page_table=t.block_tables_baseline,
        nsa_cache_seqlens_int32=t.seqlens_expanded_baseline,
        nsa_cu_seqlens_q=t.cu_seqlens_q_baseline,
        nsa_cu_seqlens_k=t.cu_seqlens_q_baseline,  # unused by _get_topk_paged
        nsa_extend_seq_lens_list=[1] * total_q,
        nsa_seqlens_expanded=t.seqlens_expanded_baseline,
        mtp_specret_enabled=False,
    )


def _make_nsa_metadata_specret(t: BenchTensors) -> NSAMetadata:
    """NSAMetadata with all mtp_specret_* fields populated."""
    base = _make_nsa_metadata_baseline(t)
    # Precompute the schedule metadata (required by _get_topk_paged specret path).
    seqlens_2d = t.mtp_specret_seqlens.unsqueeze(-1)
    sm_count = deep_gemm.get_num_sms()
    mtp_schedule = deep_gemm.get_paged_mqa_logits_metadata(
        seqlens_2d, PAGE_SIZE, sm_count
    )
    return dataclasses.replace(
        base,
        mtp_specret_enabled=True,
        mtp_specret_page_table_1=t.mtp_specret_page_table_1,
        mtp_specret_real_page_table=t.mtp_specret_real_page_table,
        mtp_specret_seqlens=t.mtp_specret_seqlens,
        mtp_specret_cu_seqlens_q=t.mtp_specret_cu_seqlens_q,
        mtp_specret_sibling_rows=t.mtp_specret_sibling_rows,
        mtp_specret_sibling_self_page_indices=t.mtp_specret_sibling_self_page_indices,
        mtp_specret_sibling_self_logical_indices=t.mtp_specret_sibling_self_page_indices,
        mtp_specret_first_token_indices=t.mtp_specret_first_token_indices,
        mtp_specret_paged_mqa_schedule_metadata=mtp_schedule,
    )


def _wrap_indexer_metadata(
    attn_meta: NSAMetadata, enable_specret: bool, draft_token_num: int
) -> NSAIndexerMetadata:
    return NSAIndexerMetadata(
        attn_metadata=attn_meta,
        topk_transform_method=TopkTransformMethod.PAGED,
        paged_mqa_schedule_metadata=None,
        enable_mtp_specret=enable_specret,
        mtp_specret_draft_token_num=draft_token_num if enable_specret else 0,
        force_unfused_topk=False,
    )


# ---------------------------------------------------------------------------
# Indexer instance (without running __init__)
# ---------------------------------------------------------------------------


def _make_indexer() -> Indexer:
    """Create an Indexer object with only the attributes _get_topk_paged needs.

    Skipping __init__ avoids constructing weight matrices / RoPE / etc., none
    of which are used by _get_topk_paged itself.
    """
    indexer = Indexer.__new__(Indexer)
    indexer.sm_count = deep_gemm.get_num_sms()
    indexer.index_topk = INDEX_TOPK
    return indexer


# ---------------------------------------------------------------------------
# Benchmark drivers
# ---------------------------------------------------------------------------


def _bench(
    fn: Callable[[], torch.Tensor],
    *,
    warmup: int,
    iters: int,
    device: torch.device,
) -> List[float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize(device)

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
        f"  {label:<14s}  median={median:8.4f} ms  mean={mean:8.4f} ms  "
        f"min={samples_ms_sorted[0]:8.4f} ms  p10={p10:8.4f} ms"
    )
    return median, mean, samples_ms_sorted[0]


# ---------------------------------------------------------------------------
# Per-step profiling for specret paths
# ---------------------------------------------------------------------------


def _profile_specret_breakdown(
    indexer: Indexer,
    fb: "_StubForwardBatch",
    t: "BenchTensors",
    meta: "NSAIndexerMetadata",
    *,
    pre_sliced: bool,
    warmup: int,
    iters: int,
    device: torch.device,
) -> None:
    """Run the specret path step-by-step with CUDA events to attribute overhead.

    When pre_sliced=True, uses q/weights with B rows (new path).
    When pre_sliced=False, uses q/weights with B*d rows (old path with reshape).
    """
    page_size = fb.token_to_kv_pool.page_size
    block_kv = 64
    num_heads_kv = 1
    head_dim_with_sf = 132

    attn_metadata = meta.attn_metadata
    draft_token_num = meta.mtp_specret_draft_token_num
    q_offset = sum(meta.get_nsa_extend_len_cpu())
    mtp_q_num = q_offset // draft_token_num

    kv_cache_fp8_2d = fb.token_to_kv_pool.get_index_k_with_scale_buffer(layer_id=0)
    max_seq_len = attn_metadata.mtp_specret_real_page_table.shape[1] * page_size

    if pre_sliced:
        q_input = t.q_specret_preslice       # (B, H, D)
        w_input = t.weights_specret_preslice  # (B, H, 1)
        step_names = [
            "0_prep_kv_cache_view",
            "1_q_unsqueeze",
            "2_weights_squeeze",
            "3_q_slice_noop",          # just q[:mtp_q_num], no copy
            "4_weights_slice_noop",    # just w[:mtp_q_num], no copy
            "5_seqlens_unsqueeze",
            "6_fp8_paged_mqa_logits",
            "7_topk_transform",
            "8_repeat_interleave",
            "9_sibling_self_kv_patch",
        ]
    else:
        q_input = t.q_specret       # (B*d, H, D)
        w_input = t.weights_specret  # (B*d, H, 1)
        step_names = [
            "0_prep_kv_cache_view",
            "1_q_unsqueeze",
            "2_weights_squeeze",
            "3_q_mtp_reshape_contig",
            "4_weights_mtp_reshape_contig",
            "5_seqlens_unsqueeze",
            "6_fp8_paged_mqa_logits",
            "7_topk_transform",
            "8_repeat_interleave",
            "9_sibling_self_kv_patch",
        ]

    def one_run(record: bool):
        evs = []
        if record:
            evs.append(torch.cuda.Event(enable_timing=True))
            evs[-1].record()

        # 0: kv cache view
        kv_cache_fp8 = kv_cache_fp8_2d.view(
            kv_cache_fp8_2d.shape[0], block_kv, num_heads_kv, head_dim_with_sf
        )
        if record:
            evs.append(torch.cuda.Event(enable_timing=True)); evs[-1].record()

        # 1: q unsqueeze (next_n dim)
        q_fp8 = q_input.unsqueeze(1)
        if record:
            evs.append(torch.cuda.Event(enable_timing=True)); evs[-1].record()

        # 2: weights squeeze
        w = w_input.squeeze(2)
        if record:
            evs.append(torch.cuda.Event(enable_timing=True)); evs[-1].record()

        # 3 & 4: reshape+contiguous (old) or no-op slice (new)
        if pre_sliced:
            # New path: q_fp8.shape[0] == mtp_q_num, so just slice (no copy)
            q_mtp = q_fp8[:mtp_q_num]
            if record:
                evs.append(torch.cuda.Event(enable_timing=True)); evs[-1].record()
            weights_mtp = w[:mtp_q_num]
            if record:
                evs.append(torch.cuda.Event(enable_timing=True)); evs[-1].record()
        else:
            # Old path: reshape + contiguous
            q_mtp = (
                q_fp8[:q_offset]
                .reshape(mtp_q_num, draft_token_num, *q_fp8.shape[1:])[:, 0]
                .contiguous()
            )
            if record:
                evs.append(torch.cuda.Event(enable_timing=True)); evs[-1].record()
            weights_mtp = (
                w[:q_offset]
                .reshape(mtp_q_num, draft_token_num, w.shape[1])[:, 0]
                .contiguous()
            )
            if record:
                evs.append(torch.cuda.Event(enable_timing=True)); evs[-1].record()

        # 5: seqlens unsqueeze
        seqlens_mtp_2d = attn_metadata.mtp_specret_seqlens.unsqueeze(-1)
        if record:
            evs.append(torch.cuda.Event(enable_timing=True)); evs[-1].record()

        # 6: fp8_paged_mqa_logits (schedule is precomputed)
        mtp_schedule_metadata = attn_metadata.mtp_specret_paged_mqa_schedule_metadata
        logits = deep_gemm.fp8_paged_mqa_logits(
            q_mtp,
            kv_cache_fp8,
            weights_mtp,
            seqlens_mtp_2d,
            attn_metadata.mtp_specret_real_page_table,
            mtp_schedule_metadata,
            max_seq_len,
            clean_logits=False,
        )
        if record:
            evs.append(torch.cuda.Event(enable_timing=True)); evs[-1].record()

        # 7: topk_transform
        topk_mtp = meta.topk_transform(
            logits,
            indexer.index_topk,
            ke_offset=attn_metadata.mtp_specret_seqlens,
            cu_seqlens_q_topk_override=attn_metadata.mtp_specret_cu_seqlens_q,
            page_table_size_1_override=attn_metadata.mtp_specret_page_table_1,
        )
        if record:
            evs.append(torch.cuda.Event(enable_timing=True)); evs[-1].record()

        # 8: repeat_interleave
        topk_result = topk_mtp.repeat_interleave(draft_token_num, dim=0)
        if record:
            evs.append(torch.cuda.Event(enable_timing=True)); evs[-1].record()

        # 9: sibling self-kv patch
        topk_result = indexer._patch_mtp_specret_sibling_self_kv(
            topk_result, meta, draft_token_num, q_offset
        )
        if record:
            evs.append(torch.cuda.Event(enable_timing=True)); evs[-1].record()

        return topk_result, evs

    # Warmup
    for _ in range(warmup):
        one_run(record=False)
    torch.cuda.synchronize(device)

    # Collect per-step latencies (ms)
    per_step: List[List[float]] = [[] for _ in step_names]
    for _ in range(iters):
        _, evs = one_run(record=True)
        torch.cuda.synchronize(device)
        for i, name in enumerate(step_names):
            per_step[i].append(evs[i].elapsed_time(evs[i + 1]))

    # Report
    label = "pre-slice" if pre_sliced else "old (reshape)"
    print(f"  -- specret per-step breakdown [{label}] (median ms) --")
    total = 0.0
    for name, samples in zip(step_names, per_step):
        m = statistics.median(samples)
        total += m
        print(f"    {name:<32s}  median={m:8.4f} ms")
    print(f"    {'sum_of_steps':<32s}         ={total:8.4f} ms")


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
            "Sweep draft_token_num from 2 up to MAX (inclusive). "
            "Pass without value to use the default max=4."
        ),
    )
    parser.add_argument(
        "--sweep-batch",
        type=int,
        nargs="+",
        default=None,
        metavar="B",
        help="Sweep batch sizes (overrides --batch-size when set).",
    )
    parser.add_argument(
        "--sweep-seqlen",
        type=int,
        nargs="+",
        default=None,
        metavar="S",
        help="Sweep seq lengths (overrides --seq-len when set).",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Print a per-step CUDA-event breakdown of both specret paths.",
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

    batch_list = args.sweep_batch if args.sweep_batch else [args.batch_size]
    seqlen_list = args.sweep_seqlen if args.sweep_seqlen else [args.seq_len]

    indexer = _make_indexer()

    print(
        f"NSA MTP-specret indexer micro-bench  "
        f"(H={args.num_heads}, D={args.head_dim}, sm_count={indexer.sm_count}, "
        f"warmup={args.warmup}, iters={args.iters})"
    )
    print(
        "Calls Indexer._get_topk_paged directly.\n"
        "  baseline:         specret=off, q/weights = B*d rows\n"
        "  specret:          specret=on,  q/weights = B*d rows (reshape+contiguous)\n"
        "  specret_preslice: specret=on,  q/weights = B   rows (pre-sliced, no copy)"
    )
    print("=" * 96)

    for B in batch_list:
        for S in seqlen_list:
            for d in draft_list:
                t = _build_inputs(
                    batch_size=B,
                    draft_token_num=d,
                    seq_len=S,
                    num_heads=args.num_heads,
                    head_dim=args.head_dim,
                    device=device,
                    seed=args.seed,
                )

                fb = _StubForwardBatch(t.kv_cache_2d)

                meta_baseline = _wrap_indexer_metadata(
                    _make_nsa_metadata_baseline(t),
                    enable_specret=False, draft_token_num=d,
                )
                meta_specret = _wrap_indexer_metadata(
                    _make_nsa_metadata_specret(t),
                    enable_specret=True, draft_token_num=d,
                )

                def baseline_fn():
                    return indexer._get_topk_paged(
                        fb, 0, t.q_baseline, t.weights_baseline, meta_baseline
                    )

                def specret_fn():
                    return indexer._get_topk_paged(
                        fb, 0, t.q_specret, t.weights_specret, meta_specret
                    )

                def specret_preslice_fn():
                    return indexer._get_topk_paged(
                        fb, 0, t.q_specret_preslice, t.weights_specret_preslice, meta_specret
                    )

                # Sanity: shapes match
                with torch.inference_mode():
                    out_b = baseline_fn()
                    out_s = specret_fn()
                    out_sp = specret_preslice_fn()
                expected_rows = B * d
                assert out_b.shape[0] >= expected_rows, (
                    f"unexpected baseline rows: {out_b.shape}"
                )
                assert out_s.shape[0] >= expected_rows, (
                    f"unexpected specret rows: {out_s.shape}"
                )
                assert out_sp.shape[0] >= expected_rows, (
                    f"unexpected specret_preslice rows: {out_sp.shape}"
                )
                assert (
                    out_b.shape[1] == out_s.shape[1] == out_sp.shape[1] == INDEX_TOPK
                )

                print(
                    f"B={B}  S={S}  d={d}  "
                    f"(q rows: baseline={B * d}, specret={B * d}, specret_preslice={B})"
                )
                baseline_samples = _bench(
                    baseline_fn, warmup=args.warmup, iters=args.iters, device=device
                )
                specret_samples = _bench(
                    specret_fn, warmup=args.warmup, iters=args.iters, device=device
                )
                specret_preslice_samples = _bench(
                    specret_preslice_fn, warmup=args.warmup, iters=args.iters, device=device
                )
                b_med, _, _ = _summarize("baseline", baseline_samples)
                s_med, _, _ = _summarize("specret", specret_samples)
                sp_med, _, _ = _summarize("specret_preslice", specret_preslice_samples)
                print(
                    f"  speedup vs baseline:  "
                    f"specret={b_med / s_med:.3f}x  pre-slice={b_med / sp_med:.3f}x  "
                    f"(ideal ~ {d}x)"
                )
                print()

                if args.profile:
                    _profile_specret_breakdown(
                        indexer, fb, t, meta_specret,
                        pre_sliced=False,
                        warmup=args.warmup, iters=args.iters, device=device,
                    )
                    print()
                    _profile_specret_breakdown(
                        indexer, fb, t, meta_specret,
                        pre_sliced=True,
                        warmup=args.warmup, iters=args.iters, device=device,
                    )
                    print()


if __name__ == "__main__":
    main()
