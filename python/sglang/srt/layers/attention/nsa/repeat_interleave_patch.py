"""Fused repeat_interleave + sibling-self-KV patch kernel.

Replaces the two-step pattern:
    topk_result = topk_mtp.repeat_interleave(draft_token_num, dim=0)
    topk_result[sibling_rows, -1] = self_kv

with a single Triton kernel that performs both operations in one launch.
The two-step fallback is preserved in nsa_indexer.py for non-CUDA or
unsupported configurations.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _fused_repeat_interleave_patch_kernel(
    # Pointers
    topk_mtp_ptr,      # [B, topk_len] input
    out_ptr,           # [B*d, topk_len] output
    self_kv_ptr,       # [num_siblings] patch values (contiguous, maps to rows with idx%d>0)
    # Dimensions
    d: tl.constexpr,
    topk_len: tl.constexpr,
    has_patch: tl.constexpr,  # whether sibling patch is needed
    # Strides
    stride_mtp_row: tl.constexpr,
    stride_out_row: tl.constexpr,
):
    """Each program handles one output row.

    - Copies topk_mtp[pid // d, :] to out[pid, :]
    - If pid % d > 0 and has_patch: overwrites out[pid, topk_len-1] with
      the sibling self-KV page index.
    """
    pid = tl.program_id(0)
    src_row = pid // d

    # Copy entire row
    col_offsets = tl.arange(0, topk_len)
    src_vals = tl.load(topk_mtp_ptr + src_row * stride_mtp_row + col_offsets)
    tl.store(out_ptr + pid * stride_out_row + col_offsets, src_vals)

    # Patch last column for sibling rows (pid % d > 0)
    if has_patch:
        sibling_offset = pid % d
        if sibling_offset > 0:
            # self_kv is laid out contiguously for all sibling rows in order:
            # For batch b: rows b*d+1, b*d+2, ..., b*d+(d-1)
            # So the index into self_kv is: b*(d-1) + (sibling_offset - 1)
            batch_idx = pid // d
            kv_idx = batch_idx * (d - 1) + (sibling_offset - 1)
            kv_val = tl.load(self_kv_ptr + kv_idx)
            tl.store(out_ptr + pid * stride_out_row + (topk_len - 1), kv_val)


def fused_repeat_interleave_and_patch(
    topk_mtp: torch.Tensor,
    draft_token_num: int,
    self_kv: torch.Tensor | None,
) -> torch.Tensor:
    """Fused repeat_interleave(d, dim=0) + sibling self-KV scatter.

    Args:
        topk_mtp: [B, topk_len] int32 tensor of page indices.
        draft_token_num: number of draft tokens per request (d).
        self_kv: [B*(d-1)] int32/int64 page indices to write at last col
                 for sibling rows, or None to skip patching.

    Returns:
        out: [B*d, topk_len] int32 tensor.
    """
    B, topk_len = topk_mtp.shape
    d = draft_token_num
    total_rows = B * d

    out = torch.empty(
        (total_rows, topk_len), dtype=topk_mtp.dtype, device=topk_mtp.device
    )

    has_patch = self_kv is not None and self_kv.numel() > 0
    if has_patch:
        self_kv_casted = self_kv.to(topk_mtp.dtype)
    else:
        # Dummy pointer; won't be accessed
        self_kv_casted = topk_mtp

    _fused_repeat_interleave_patch_kernel[(total_rows,)](
        topk_mtp,
        out,
        self_kv_casted,
        d=d,
        topk_len=topk_len,
        has_patch=has_patch,
        stride_mtp_row=topk_mtp.stride(0),
        stride_out_row=out.stride(0),
    )

    return out
