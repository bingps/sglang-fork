import weakref
from contextlib import contextmanager

import torch

from sglang.srt.mem_cache.allocator.base import BaseTokenToKVPoolAllocator
from sglang.srt.mem_cache.allocator.paged import PagedTokenToKVPoolAllocator
from sglang.srt.mem_cache.deepseek_v4_memory_pool import (
    DeepSeekV4TokenToKVPool,
    HiSparseC4DevicePool,
)
from sglang.srt.mem_cache.hisparse_memory_pool import HiSparseDSATokenToKVPool
from sglang.srt.utils.common import get_num_new_pages


class HiSparseTokenToKVPoolAllocator(BaseTokenToKVPoolAllocator):
    def __init__(
        self,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        device: torch.device,
        kvcache: HiSparseDSATokenToKVPool,
        need_sort: bool,
        host_to_device_ratio: int = 2,
    ):
        self._kvcache = kvcache
        self._size_full = size * host_to_device_ratio
        self._size_hisparse = size
        self.compress_ratio = 1
        self.dtype = dtype
        self.device = device
        self.page_size = page_size
        self.need_sort = need_sort

        self.logical_attn_allocator = PagedTokenToKVPoolAllocator(
            self._size_full,
            self.page_size,
            self.dtype,
            self.device,
            kvcache,
            need_sort,
        )
        self.hisparse_attn_allocator = PagedTokenToKVPoolAllocator(
            self._size_hisparse,
            self.page_size,
            self.dtype,
            self.device,
            kvcache,
            need_sort,
        )
        self.full_to_hisparse_device_index_mapping = torch.cat(
            [
                torch.zeros(
                    self._size_full + self.page_size,
                    dtype=torch.int64,
                    device=self.device,
                ),
                torch.tensor([-1], dtype=torch.int64, device=self.device),
            ]
        )

        self.free_pages = None
        self.release_pages = None
        self.is_not_in_free_group = True
        self.free_group = []
        self._spec_logical = False
        self.clear()
        self._kvcache.register_mapping(
            weakref.proxy(self.full_to_hisparse_device_index_mapping)
        )

    @property
    def size_full(self) -> int:
        return self._size_full

    @property
    def size(self) -> int:
        return self._size_full

    def available_size(self) -> int:
        return min(
            self.logical_attn_allocator.available_size(),
            self.hisparse_attn_allocator.available_size(),
        )

    def get_kvcache(self):
        return self._kvcache

    @contextmanager
    def spec_logical_alloc(self):
        """Scope in which alloc/alloc_extend hand out LOGICAL indices only.

        MTP spec-decode slots take their physical locations from the
        per-request staging ring, not the global hisparse pool. Entered by
        eagle_prepare_for_decode around alloc_for_spec_decode so the shared
        allocation helpers need no mode parameter.
        """
        self._spec_logical = True
        try:
            yield
        finally:
            self._spec_logical = False

    def alloc(self, need_size: int):
        if self._spec_logical:
            return self.alloc_spec_logical(need_size)
        if self.page_size != 1:
            raise NotImplementedError(
                "HiSparse generic allocation is only supported for page_size=1. "
                "Use alloc_extend for paged allocation."
            )

        logical_indices = self.logical_attn_allocator.alloc(need_size)
        if logical_indices is None:
            return None

        hisparse_indices = self.hisparse_attn_allocator.alloc(need_size)
        if hisparse_indices is None:
            self.logical_attn_allocator.free(logical_indices)
            return None

        self.full_to_hisparse_device_index_mapping[logical_indices] = hisparse_indices
        return logical_indices

    def alloc_logical_only(
        self,
        prefix_lens: torch.Tensor,
        prefix_lens_cpu: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        last_loc: torch.Tensor,
        extend_num_tokens: int,
    ):
        """Allocate only logical indices without hisparse device indices.

        Used in the direct-to-host transfer path where KV data is written
        directly to host memory by the prefill node, skipping GPU staging,
        and in the MTP speculative path where physical slots come from the
        per-request staging ring.
        """
        return self.logical_attn_allocator.alloc_extend(
            prefix_lens,
            prefix_lens_cpu,
            seq_lens,
            seq_lens_cpu,
            last_loc,
            extend_num_tokens,
        )

    def alloc_spec_logical(self, need_size: int):
        """page_size==1 variant of alloc_logical_only (see alloc_token_slots)."""
        # alloc() floors to whole pages, so a page_size > 1 caller would get
        # fewer slots than requested. page_size > 1 is routed to alloc_extend.
        assert self.page_size == 1, (
            f"alloc_spec_logical requires page_size == 1, got {self.page_size}; "
            "page_size > 1 must go through alloc_extend."
        )
        return self.logical_attn_allocator.alloc(need_size)

    def alloc_device_buffer(self, allocated_indices, need_size: int):
        assert need_size % self.page_size == 0
        # clear original reference and isolate the buffer from outside addressing, allocate new buffer if needed
        hisparse_indices = self.full_to_hisparse_device_index_mapping[allocated_indices]
        self.full_to_hisparse_device_index_mapping[allocated_indices] = 0
        # Filter valid (non-zero) hisparse indices.
        # In the direct-to-host path, mapping is all zeros since no hisparse
        # device indices were pre-allocated.
        hisparse_indices = hisparse_indices[hisparse_indices > 0]
        if len(hisparse_indices) >= need_size:
            buffer_indices = hisparse_indices[:need_size]
            self.free_hisparse_indices(hisparse_indices[need_size:])
        else:
            # page alignment, claiming the residual space for an incomplete page
            page_residual_length = len(hisparse_indices) % self.page_size
            if page_residual_length != 0:
                hisparse_indices = torch.cat(
                    [
                        hisparse_indices,
                        torch.arange(
                            hisparse_indices[-1] + 1,
                            hisparse_indices[-1]
                            + self.page_size
                            - page_residual_length
                            + 1,
                            device=self.device,
                        ),
                    ]
                )
            extra_indices = self.hisparse_attn_allocator.alloc(
                need_size - len(hisparse_indices)
            )
            assert (
                extra_indices is not None
            ), "Hisparse allocation failed in alloc_device_buffer"
            buffer_indices = torch.cat([hisparse_indices, extra_indices])
        return buffer_indices

    def alloc_device_buffer_mtp(
        self, allocated_indices, need_size: int, defer_free: bool = False
    ):
        """Carve the device buffer for MTP, returning the retained logical ids.

        Differences from ``alloc_device_buffer`` (the plain-decode version),
        all required by the MTP verify swap-in:
        - returns ``buffer_alloc``: the LOGICAL ids kept in the buffer, which
          key the verify kernel's buffer-match table (fused top-k emits
          logical ids, not request-relative positions);
        - restores ``full_to_hisparse_device_index_mapping`` for those ids so
          verify's first-level ``direct_loc`` lookup resolves them;
        - frees surplus at PAGE granularity, excluding pages shared with the
          buffer: the full padded buffer makes the boundary arbitrary, so
          token-granular frees would return an in-use page to the free list;
        - with ``defer_free``, returns the surplus pages instead of releasing
          them (the staging DMA may still be reading those source pages);
        - restores the mapping on allocation failure so the caller can reclaim
          deferred pages and retry.

        Returns ``(buffer_indices, buffer_alloc)``, plus
        ``deferred_free_indices`` when ``defer_free`` is set.
        """
        assert need_size % self.page_size == 0
        hisparse_indices = self.full_to_hisparse_device_index_mapping[allocated_indices]
        self.full_to_hisparse_device_index_mapping[allocated_indices] = 0
        valid_mask = hisparse_indices > 0
        valid_hisparse = hisparse_indices[valid_mask]
        valid_alloc = allocated_indices[valid_mask]
        deferred_free_indices = torch.empty(
            (0,), dtype=valid_hisparse.dtype, device=valid_hisparse.device
        )
        if len(valid_hisparse) >= need_size:
            buffer_indices = valid_hisparse[:need_size]
            buffer_alloc = valid_alloc[:need_size]
            surplus = valid_hisparse[need_size:]
            if surplus.numel() > 0:
                buffer_pages = torch.unique(buffer_indices // self.page_size)
                surplus_pages = torch.unique(surplus // self.page_size)
                deferred_pages = surplus_pages[
                    ~torch.isin(surplus_pages, buffer_pages)
                ]
                deferred_free_indices = deferred_pages * self.page_size
                if not defer_free:
                    self.free_hisparse_indices(deferred_free_indices)
        else:
            buffer_alloc = valid_alloc
            page_residual_length = len(valid_hisparse) % self.page_size
            if page_residual_length != 0:
                valid_hisparse = torch.cat(
                    [
                        valid_hisparse,
                        torch.arange(
                            valid_hisparse[-1] + 1,
                            valid_hisparse[-1]
                            + self.page_size
                            - page_residual_length
                            + 1,
                            device=self.device,
                        ),
                    ]
                )
            extra_indices = self.hisparse_attn_allocator.alloc(
                need_size - len(valid_hisparse)
            )
            if extra_indices is None:
                # Clearing the mapping is part of committing the buffer
                # transition. Restore it when the extra allocation cannot be
                # satisfied so the caller can safely release or retry the req.
                self.full_to_hisparse_device_index_mapping[allocated_indices] = (
                    hisparse_indices
                )
                raise RuntimeError("Hisparse allocation failed in alloc_device_buffer")
            buffer_indices = torch.cat([valid_hisparse, extra_indices])

        n = min(len(buffer_alloc), len(buffer_indices))
        self.full_to_hisparse_device_index_mapping[buffer_alloc[:n]] = buffer_indices[
            :n
        ]
        if defer_free:
            return buffer_indices, buffer_alloc, deferred_free_indices
        return buffer_indices, buffer_alloc

    def free_hisparse_indices(self, buffer_indices: torch.Tensor):
        # disable free group mechanism for device buffer free
        self.hisparse_attn_allocator.is_not_in_free_group = True
        self.hisparse_attn_allocator.free(buffer_indices[buffer_indices > 0])

    def get_last_loc_compressed(self, last_locs: torch.Tensor):
        return last_locs

    def get_last_loc_hisparse_device(self, last_locs: torch.Tensor):
        return self._kvcache._translate_loc_to_hisparse_device(last_locs)

    def alloc_extend(
        self,
        prefix_lens: torch.Tensor,
        prefix_lens_cpu: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        last_loc: torch.Tensor,  # last_loc for full layers
        extend_num_tokens: int,
    ):
        if self._spec_logical:
            return self.alloc_logical_only(
                prefix_lens,
                prefix_lens_cpu,
                seq_lens,
                seq_lens_cpu,
                last_loc,
                extend_num_tokens,
            )
        num_new_pages = get_num_new_pages(
            seq_lens=seq_lens_cpu, page_size=self.page_size, prefix_lens=prefix_lens_cpu
        )
        if (
            num_new_pages
            > self.logical_attn_allocator.available_size() // self.page_size
        ):
            return None
        if (
            num_new_pages
            > self.hisparse_attn_allocator.available_size() // self.page_size
        ):
            return None

        hisparse_last_loc = self.get_last_loc_hisparse_device(last_loc)
        # Fail fast if a row's physical tail mapping was cleared while the row
        # is still extending: the paged allocator would emit the partial-page
        # tokens at slot 0+1 — a physical slot this request does not own. The
        # last-chunk admission gate guarantees this cannot happen (staging /
        # ring recycling only run after prefill completes). Checked BEFORE the
        # logical allocation so that tripping it does not leak logical pages.
        # The GPU-sync any() is gated behind the free CPU-side prefix check:
        # with the radix cache disabled (required by hisparse) almost every
        # prefill row has prefix 0 and skips it.
        if bool((prefix_lens_cpu > 0).any()):
            assert not bool((hisparse_last_loc == 0).any()), (
                "HiSparse alloc_extend: a row's tail mapping is cleared while "
                "it is still extending; admission must wait for the last "
                "prefill chunk."
            )

        logical_indices = self.logical_attn_allocator.alloc_extend(
            prefix_lens,
            prefix_lens_cpu,
            seq_lens,
            seq_lens_cpu,
            last_loc,
            extend_num_tokens,
        )
        assert logical_indices is not None, "Logical allocation failed in alloc_extend"

        hisparse_indices = self.hisparse_attn_allocator.alloc_extend(
            prefix_lens,
            prefix_lens_cpu,
            seq_lens,
            seq_lens_cpu,
            hisparse_last_loc,
            len(logical_indices),
            num_new_pages=num_new_pages,
        )
        assert (
            hisparse_indices is not None
        ), "Hisparse allocation failed in alloc_extend"
        self.full_to_hisparse_device_index_mapping[logical_indices] = hisparse_indices
        return logical_indices

    def alloc_decode(
        self,
        seq_lens: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        last_loc: torch.Tensor,  # last_loc for full layers
    ):
        return self.logical_attn_allocator.alloc_decode(
            seq_lens, seq_lens_cpu, last_loc
        )

    def free_hisparse(self, free_indices: torch.Tensor):
        hisparse_indices = self._kvcache._translate_loc_to_hisparse_device(free_indices)
        hisparse_indices = hisparse_indices[hisparse_indices > 0]
        self.free_hisparse_indices(hisparse_indices)
        self.full_to_hisparse_device_index_mapping[free_indices] = 0

    def clear(self):
        self.logical_attn_allocator.clear()
        self.hisparse_attn_allocator.clear()
        # Note: the last item is -1, we don't clear it, see the comment in __init__
        self.full_to_hisparse_device_index_mapping[:-1].fill_(0)
        self.is_not_in_free_group = True
        self.free_group = []

    def free_group_begin(self):
        return

    def free_group_end(self):
        return

    def free(self, free_index: torch.Tensor):
        if free_index.numel() == 0:
            return
        if self.is_not_in_free_group:
            self.logical_attn_allocator.free(free_index)
            self.free_hisparse(free_index)
        else:
            self.free_group.append(free_index)
        assert (
            self.logical_attn_allocator.available_size()
            <= self.logical_attn_allocator.size
        ), f"logical: avail={self.logical_attn_allocator.available_size()} > size={self.logical_attn_allocator.size}"
        ha, hs = self.hisparse_attn_allocator.available_size(), self.hisparse_attn_allocator.size
        assert ha <= hs, f"hisparse: avail={ha} > size={hs}, freed={free_index.numel()}"


class DeepSeekV4HiSparseTokenToKVPoolAllocator(BaseTokenToKVPoolAllocator):

    def __init__(
        self,
        logical_attn_allocator: BaseTokenToKVPoolAllocator,
    ):
        assert isinstance(logical_attn_allocator._kvcache, DeepSeekV4TokenToKVPool)
        assert isinstance(
            logical_attn_allocator._kvcache.c4_kv_pool, HiSparseC4DevicePool
        )
        self.compress_ratio = 4

        self.hisparse_kvcache = logical_attn_allocator._kvcache.c4_kv_pool
        self._size_full = logical_attn_allocator.size_full
        self._size_hisparse = self.hisparse_kvcache.size

        self.dtype = self.hisparse_kvcache.dtype
        self.device = self.hisparse_kvcache.device
        # Keep the public page_size as the logical DSV4 full/SWA page size.
        # C4 HiSparse allocation/device-buffer code must use the compressed page size.
        self.page_size = logical_attn_allocator.page_size
        self.hisparse_page_size = self.hisparse_kvcache.page_size

        self.logical_attn_allocator = logical_attn_allocator
        self._kvcache = logical_attn_allocator._kvcache
        self.hisparse_attn_allocator = PagedTokenToKVPoolAllocator(
            self._size_hisparse,
            self.hisparse_page_size,
            self.dtype,
            self.device,
            self.hisparse_kvcache,
            logical_attn_allocator.need_sort,
        )

        self.full_to_hisparse_device_index_mapping = torch.cat(
            [
                torch.zeros(
                    self._kvcache.c4_logical_size + self.hisparse_page_size,
                    dtype=torch.int64,
                    device=self.device,
                ),
                torch.tensor([-1], dtype=torch.int64, device=self.device),
            ]
        )

        self.need_sort = logical_attn_allocator.need_sort
        self.free_pages = None
        self.release_pages = None
        self.is_not_in_free_group = True
        self.free_group = []
        self.clear()

        self.hisparse_kvcache.register_mapping(
            weakref.proxy(self.full_to_hisparse_device_index_mapping)
        )

    @property
    def size_full(self) -> int:
        return self._size_full

    @property
    def size(self) -> int:
        return self.logical_attn_allocator.size

    @property
    def size_swa(self) -> int:
        return self.logical_attn_allocator.size_swa

    @property
    def full_to_swa_index_mapping(self):
        return self.logical_attn_allocator.full_to_swa_index_mapping

    def debug_print(self) -> str:
        msg = self.logical_attn_allocator.debug_print()
        msg += (
            f"#hisparse-available-size: "
            f"{self.hisparse_attn_allocator.available_size()}, "
        )
        return msg

    def get_kvcache(self):
        return self._kvcache

    def translate_loc_from_full_to_swa(self, kv_indices: torch.Tensor):
        return self.logical_attn_allocator.translate_loc_from_full_to_swa(kv_indices)

    def full_available_size(self):
        return min(
            self.logical_attn_allocator.full_available_size(),
            self.hisparse_attn_allocator.available_size() * self.compress_ratio,
        )

    def swa_available_size(self):
        return self.logical_attn_allocator.swa_available_size()

    def free_swa(self, free_indices: torch.Tensor):
        self.logical_attn_allocator.free_swa(free_indices)

    def available_size(self) -> int:
        return min(
            self.logical_attn_allocator.available_size(),
            self.hisparse_attn_allocator.available_size() * self.compress_ratio,
        )

    def alloc(self, need_size: int):
        raise NotImplementedError(
            "DeepSeek V4 HiSparse allocator does not support direct token allocation; "
            "use alloc_extend or alloc_decode instead."
        )

    def alloc_logical_only(
        self,
        prefix_lens: torch.Tensor,
        prefix_lens_cpu: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        last_loc: torch.Tensor,
        extend_num_tokens: int,
    ):
        """Allocate decode logical indices without allocating C4 hisparse device pages."""
        return self.logical_attn_allocator.alloc_extend(
            prefix_lens,
            prefix_lens_cpu,
            seq_lens,
            seq_lens_cpu,
            last_loc,
            extend_num_tokens,
        )

    def alloc_extend_swa_tail(
        self,
        prefix_lens: torch.Tensor,
        prefix_lens_cpu: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        last_loc: torch.Tensor,
        extend_num_tokens: int,
        swa_tail_len: int,
    ):
        return self.logical_attn_allocator.alloc_extend_swa_tail(
            prefix_lens=prefix_lens,
            prefix_lens_cpu=prefix_lens_cpu,
            seq_lens=seq_lens,
            seq_lens_cpu=seq_lens_cpu,
            last_loc=last_loc,
            extend_num_tokens=extend_num_tokens,
            swa_tail_len=swa_tail_len,
        )

    def alloc_device_buffer(self, allocated_indices, need_size: int):
        assert need_size % self.hisparse_page_size == 0
        hisparse_indices = self.full_to_hisparse_device_index_mapping[allocated_indices]
        self.full_to_hisparse_device_index_mapping[allocated_indices] = 0
        hisparse_indices = hisparse_indices[hisparse_indices > 0]

        device_buffer_size = need_size - self.hisparse_page_size
        P = len(hisparse_indices)
        if P > device_buffer_size + 1:
            newest_src = hisparse_indices[P - 1].clone()
            old_at_dbs = hisparse_indices[device_buffer_size].clone()
            hisparse_indices[device_buffer_size] = newest_src
            hisparse_indices[P - 1] = old_at_dbs

        if len(hisparse_indices) >= need_size:
            buffer_indices = hisparse_indices[:need_size]
            surplus = hisparse_indices[need_size:]
            if surplus.numel() > 0:
                buffer_pages = torch.unique(buffer_indices // self.hisparse_page_size)
                surplus_pages = torch.unique(surplus // self.hisparse_page_size)
                pure_surplus = surplus_pages[~torch.isin(surplus_pages, buffer_pages)]
                if pure_surplus.numel() > 0:
                    self.hisparse_attn_allocator.is_not_in_free_group = True
                    self.hisparse_attn_allocator.free(
                        pure_surplus * self.hisparse_page_size
                    )
        else:
            page_residual_length = len(hisparse_indices) % self.hisparse_page_size
            if page_residual_length != 0:
                hisparse_indices = torch.cat(
                    [
                        hisparse_indices,
                        torch.arange(
                            hisparse_indices[-1] + 1,
                            hisparse_indices[-1]
                            + self.hisparse_page_size
                            - page_residual_length
                            + 1,
                            device=self.device,
                        ),
                    ]
                )
            extra_indices = self.hisparse_attn_allocator.alloc(
                need_size - len(hisparse_indices)
            )
            assert (
                extra_indices is not None
            ), "Hisparse allocation failed in alloc_device_buffer"
            buffer_indices = torch.cat([hisparse_indices, extra_indices])
        return buffer_indices

    def free_hisparse_indices(self, buffer_indices: torch.Tensor):
        self.hisparse_attn_allocator.is_not_in_free_group = True
        self.hisparse_attn_allocator.free(buffer_indices[buffer_indices > 0])

    def get_last_loc_compressed(self, last_locs: torch.Tensor):
        return (last_locs - 3) // self.compress_ratio

    def get_last_loc_hisparse_device(self, last_locs: torch.Tensor):
        return self.hisparse_kvcache._translate_loc_to_hisparse_device(
            self.get_last_loc_compressed(last_locs)
        )

    def alloc_extend(
        self,
        prefix_lens: torch.Tensor,
        prefix_lens_cpu: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        last_loc: torch.Tensor,
        extend_num_tokens: int,
    ):
        assert self.page_size > 1

        num_new_pages_logical = get_num_new_pages(
            seq_lens=seq_lens_cpu, page_size=self.page_size, prefix_lens=prefix_lens_cpu
        )
        num_new_pages_hisparse = get_num_new_pages(
            seq_lens=seq_lens_cpu // self.compress_ratio,
            page_size=self.hisparse_page_size,
            prefix_lens=prefix_lens_cpu // self.compress_ratio,
        )
        if (
            num_new_pages_logical
            > self.logical_attn_allocator.available_size() // self.page_size
        ):
            return None
        if (
            num_new_pages_hisparse
            > self.hisparse_attn_allocator.available_size() // self.hisparse_page_size
        ):
            return None

        logical_indices = self.logical_attn_allocator.alloc_extend(
            prefix_lens,
            prefix_lens_cpu,
            seq_lens,
            seq_lens_cpu,
            last_loc,
            extend_num_tokens,
        )
        assert logical_indices is not None, "Logical allocation failed in alloc_extend"

        compressed_logical_indices = (
            self.hisparse_kvcache.translate_loc_from_full_to_compressed(logical_indices)
        )
        hisparse_last_loc = self.get_last_loc_hisparse_device(last_loc)
        hisparse_indices = self.hisparse_attn_allocator.alloc_extend(
            prefix_lens // self.compress_ratio,
            prefix_lens_cpu // self.compress_ratio,
            seq_lens // self.compress_ratio,
            seq_lens_cpu // self.compress_ratio,
            hisparse_last_loc,
            len(compressed_logical_indices),
        )
        assert (
            hisparse_indices is not None
        ), "Hisparse allocation failed in alloc_extend"

        self.full_to_hisparse_device_index_mapping[compressed_logical_indices] = (
            hisparse_indices.to(torch.int64)
        )
        return logical_indices

    def alloc_decode(
        self,
        seq_lens: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        last_loc: torch.Tensor,
    ):
        return self.logical_attn_allocator.alloc_decode(
            seq_lens, seq_lens_cpu, last_loc
        )

    def free_compressed(self, compressed_indices: torch.Tensor):
        hisparse_indices = self.hisparse_kvcache.translate_loc_to_hisparse_device(
            compressed_indices
        )
        hisparse_indices = hisparse_indices[hisparse_indices > 0]
        self.free_hisparse_indices(hisparse_indices)
        self.full_to_hisparse_device_index_mapping[compressed_indices] = 0

    def free_hisparse(self, free_indices: torch.Tensor):
        compressed_indices = (
            self.hisparse_kvcache.translate_loc_from_full_to_compressed(free_indices)
        )
        self.free_compressed(compressed_indices)

    def clear(self):
        self.logical_attn_allocator.clear()
        self.hisparse_attn_allocator.clear()

        self.full_to_hisparse_device_index_mapping[:-1].fill_(0)
        self.is_not_in_free_group = True
        self.free_group = []

    def free(self, free_index: torch.Tensor):
        if free_index.numel() == 0:
            return

        if self.is_not_in_free_group:
            self.logical_attn_allocator.free(free_index)
        else:
            self.free_group.append(free_index)
