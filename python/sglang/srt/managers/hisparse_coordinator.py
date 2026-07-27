# to be combined with the sparse coordinator class and sparse algorithm family

import logging
from typing import List, NamedTuple, Optional, Union

import torch

from sglang.kernels.ops.kvcache.hisparse import (
    load_cache_to_device_buffer_dsv4_mla,
    load_cache_to_device_buffer_mla,
    load_cache_to_device_buffer_mtp_mla,
)
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.mem_cache.allocator.hisparse import (
    DeepSeekV4HiSparseTokenToKVPoolAllocator,
    HiSparseTokenToKVPoolAllocator,
)
from sglang.srt.mem_cache.hisparse_memory_pool import (
    HiSparseDSATokenToKVPool,
)
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.mem_cache.memory_pool_host import DeepSeekV4PagedHostPool
from sglang.srt.mem_cache.pool_host.mla import MLATokenToKVPoolHost
from sglang.srt.utils import get_device_module, is_hip

device_module = get_device_module()

_is_hip = is_hip()

logger = logging.getLogger(__name__)


class HiSparseAct(NamedTuple):
    start_event: device_module.Event
    finish_event: device_module.Event
    req: Req
    # MTP only: surplus prefill pages parked until the staging DMA acks
    # (eager admission alloc). Always None in the plain coordinator.
    deferred_free_indices: Optional[torch.Tensor] = None


class HiSparseTokenStats(NamedTuple):
    device_tokens: int
    device_token_usage: float
    host_tokens: int
    host_token_usage: float


def hisparse_spec_ring_capacity(server_args) -> int:
    """Page-aligned per-request speculative staging ring capacity (0 if no spec).

    2x the per-decode reserve is the hard floor (positions recycled at
    +capacity must already be backed up); 4x adds slack for the
    kv_committed_len lag in overlap mode. The extra page absorbs alignment
    skew: the first decode round starts from the UNALIGNED prompt length, so
    its window align(prompt + reserve) - prompt can reach
    reserve + page_size - 1; without the extra page a prompt with
    len % page_size > page_size - reserve overflows the ring in a single
    round (two live positions alias one slot).
    """
    if server_args.speculative_algorithm is None:
        return 0

    from sglang.srt.mem_cache.allocation_sizing import get_alloc_reserve_per_decode

    page_size = server_args.page_size
    reserve = get_alloc_reserve_per_decode(server_args)
    return ((4 * reserve + page_size - 1) // page_size + 1) * page_size


class HiSparseCoordinator:
    """HiSparse coordinator for plain (non-speculative) decoding.

    Owns the staging DMA pipeline, the host KV pool, the per-request device
    buffer, and the position-keyed decode swap-in. The MTP variant
    (:class:`HiSparseMTPCoordinator`) overrides the identity-table and
    speculative-ring hooks; the shared admission / cleanup flow lives here.
    """

    # Class-level defaults so shared code and tests can probe the mode and
    # the mode-specific tables without hasattr checks.
    mtp_enabled = False
    req_device_buffer_logical_locs: Optional[torch.Tensor] = None
    full_to_token_position: Optional[torch.Tensor] = None

    def __init__(
        self,
        req_to_token_pool: ReqToTokenPool,
        token_to_kv_pool_allocator: Union[
            HiSparseTokenToKVPoolAllocator,
            DeepSeekV4HiSparseTokenToKVPoolAllocator,
        ],
        top_k: int,
        device_buffer_size: int,
        device: str,
        tp_group,
        host_to_device_ratio: int = 2,
        swap_in_block_size: int = 960,
    ):
        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool_allocator = token_to_kv_pool_allocator
        self.top_k = top_k
        self.device_buffer_size = device_buffer_size
        self.device = device
        self.swap_in_block_size = swap_in_block_size
        self.compress_ratio = self.token_to_kv_pool_allocator.compress_ratio

        self.is_dsv4_hisparse = isinstance(
            self.token_to_kv_pool_allocator, DeepSeekV4HiSparseTokenToKVPoolAllocator
        )
        if self.is_dsv4_hisparse:
            self.mem_pool_device = self.token_to_kv_pool_allocator.hisparse_kvcache
            page_size = self.mem_pool_device.page_size
            num_host_pages = (
                self.token_to_kv_pool_allocator.size_full // self.compress_ratio
                + page_size
                - 1
            ) // page_size
            self.mem_pool_host = DeepSeekV4PagedHostPool(
                pool_name="dsv4_hisparse_c4",
                device_buffers=self.mem_pool_device.kv_buffer,
                item_bytes=self.mem_pool_device.bytes_per_page_padded,
                num_host_pages=num_host_pages,
                slot_page_size=page_size,
                layout="layer_first",
            )
            self.item_size_bytes = (
                self.mem_pool_device.kv_cache_total_dim
                * self.mem_pool_device.store_dtype.itemsize
            )
        else:
            assert isinstance(
                self.token_to_kv_pool_allocator, HiSparseTokenToKVPoolAllocator
            )
            self.mem_pool_device: HiSparseDSATokenToKVPool = (
                self.token_to_kv_pool_allocator.get_kvcache()
            )
            self.mem_pool_host = MLATokenToKVPoolHost(
                device_pool=self.mem_pool_device,
                host_to_device_ratio=host_to_device_ratio,
                host_size=0,
                page_size=self.mem_pool_device.page_size,
                layout="layer_first",
                override_kv_cache_dim=self.mem_pool_device.kv_cache_dim,
            )
            self.item_size_bytes = self.mem_pool_host.token_stride_size
        self.page_size = self.mem_pool_device.page_size

        max_num_req_slots = req_to_token_pool.req_to_token.shape[0]
        max_context_len = req_to_token_pool.max_context_len
        max_compressed_context_len = (
            max_context_len + self.compress_ratio - 1
        ) // self.compress_ratio

        # to have an extra page for new tokens
        self.padded_buffer_size = (
            self.device_buffer_size + self.mem_pool_device.page_size
        )

        self.req_to_device_buffer = torch.zeros(
            (max_num_req_slots, self.padded_buffer_size),
            dtype=torch.int64,
            device=device,
        )
        self.req_device_buffer_size = torch.zeros(
            max_num_req_slots, dtype=torch.int64, device="cpu"
        )
        self.req_to_host_pool = torch.full(
            (max_num_req_slots, max_compressed_context_len + self.page_size),
            -1,
            dtype=torch.int64,
            device=device,
        )
        self.req_to_host_pool_allocated_len = torch.zeros(
            max_num_req_slots, dtype=torch.int64, device="cpu"
        )

        self.write_staging_stream = device_module.Stream()
        self.decode_backup_stream = device_module.Stream()
        self.ack_staging_queue: List[HiSparseAct] = []
        self.decode_producer_stream = None
        self._backup_done_event = device_module.Event()
        self._has_pending_backup = False

        self.tp_group = tp_group
        self.tp_world_size = torch.distributed.get_world_size(group=self.tp_group)

        # initialize data structures for swap-in kernel
        layer_num = self.mem_pool_device.layer_num
        self._init_identity_table(layer_num, max_num_req_slots)
        self.req_device_buffer_token_locs = torch.full(
            (layer_num, max_num_req_slots, self.padded_buffer_size),
            -1,
            dtype=torch.int32,
            device=device,
        )
        self._lru_init = torch.arange(
            self.device_buffer_size, dtype=torch.int16, device=device
        )
        self.lru_slots = (
            self._lru_init.view(1, 1, -1)
            .repeat(layer_num, max_num_req_slots, 1)
            .contiguous()
        )
        self._device_buffer_arange_i32 = torch.arange(
            self.device_buffer_size, dtype=torch.int32, device=device
        )

        # Pre-allocated output buffer for swap_in_selected_pages (CUDA-graph safe)
        self.top_k_device_locs_buffer = torch.full(
            (max_num_req_slots, self.top_k), -1, dtype=torch.int32, device=device
        )
        self.raw_indices_buffer = torch.full(
            (max_num_req_slots, self.top_k), -1, dtype=torch.int32, device=device
        )
        # Scalar tensor: number of real (non-padded) requests in the batch.
        # Updated before each graph replay so padded blocks early-return.
        self.num_real_reqs = torch.zeros(1, dtype=torch.int32, device=device)

        # CPU flag: True means "skip backup on the next decode step" because
        # staging already backed up all prefill tokens.  Cleared after one step.
        self._skip_first_backup = [False] * max_num_req_slots

    # -- Mode hooks (overridden by HiSparseMTPCoordinator) --------------

    def _init_identity_table(self, layer_num: int, max_num_req_slots: int) -> None:
        """Allocate the swap-in identity table for this mode.

        Non-MTP decode keys buffer slots by request-relative POSITION;
        MTP verify keys them by LOGICAL slot id. Only one table exists.
        """
        self.req_device_buffer_tokens = torch.full(
            (layer_num, max_num_req_slots, self.padded_buffer_size),
            -1,
            dtype=torch.int32,
            device=self.device,
        )

    def _take_extra_physical_locs(self, req: Req) -> Optional[torch.Tensor]:
        """Detach extra per-request physical regions for the final free.

        Seam for subclasses that own physical pages beyond the device buffer
        (the MTP staging ring); the caller merges them into the deduplicated
        page-granularity free in request_finished.
        """
        return None

    def _clear_buffer_identities(self, req_pool_idx: int) -> None:
        self.req_device_buffer_tokens[:, req_pool_idx, :] = -1

    def _clear_token_positions(self, compressed_locs: torch.Tensor) -> None:
        """The logical-id -> position inverse map is MTP-only; no-op here."""

    def set_decode_producer_stream(self, stream) -> None:
        self.decode_producer_stream = stream

    def destroy(self) -> None:
        # Drain in-flight transfers so the buffer is idle, then unregister it.
        # See HostKVCache.destroy for why the explicit unregister matters.
        self.write_staging_stream.synchronize()
        self.decode_backup_stream.synchronize()
        self.mem_pool_host.destroy()

    def get_token_stats(self) -> HiSparseTokenStats:
        device_allocator = self.token_to_kv_pool_allocator.hisparse_attn_allocator
        device_capacity = device_allocator.size
        device_tokens = device_capacity - device_allocator.available_size()
        host_capacity = self.mem_pool_host.size
        host_tokens = host_capacity - self.mem_pool_host.available_size()
        return HiSparseTokenStats(
            device_tokens=device_tokens,
            device_token_usage=(
                device_tokens / device_capacity if device_capacity > 0 else 0.0
            ),
            host_tokens=host_tokens,
            host_token_usage=(
                host_tokens / host_capacity if host_capacity > 0 else 0.0
            ),
        )

    def admit_request_into_staging(self, req: Req) -> None:
        req.hisparse_staging = True

        full_kv_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, : req.extend_range.end
        ].to(dtype=torch.int64, copy=True)
        device_indices = (
            self.mem_pool_device.translate_loc_from_full_to_hisparse_device(
                full_kv_indices
            )
        )

        prefill_len = len(device_indices)
        host_indices = self.mem_pool_host.alloc_paged_token_slots(
            self.req_to_host_pool,
            self.req_to_host_pool_allocated_len,
            req.req_pool_idx,
            0,
            prefill_len,
        )

        start_event = device_module.Event()
        finish_event = device_module.Event()
        start_event.record()
        with device_module.stream(self.write_staging_stream):
            start_event.wait(self.write_staging_stream)
            self.mem_pool_host.backup_from_device_all_layer(
                self.mem_pool_device,
                host_indices,
                device_indices,
                io_backend="kernel",
            )
            finish_event.record()
            if host_indices.is_cuda:
                host_indices.record_stream(self.write_staging_stream)
            if device_indices.is_cuda:
                device_indices.record_stream(self.write_staging_stream)

        self.ack_staging_queue.append(HiSparseAct(start_event, finish_event, req))

    def admit_request_direct(self, req: Req) -> None:
        """Direct-to-host path: KV data already resides in host pool via RDMA.

        Skips staging DMA entirely. Only allocates a small device buffer
        (4KB) for decode-time swap-in, then marks the request as ready.
        Host indices were already written to req_to_host_pool.

        Metadata fixups after alloc_device_buffer():
        - alloc_device_buffer() sets device_buffer_tokens = [0, 1, ..., buf_size-1],
          which tells the swap-in kernel that those tokens are cached in the device
          buffer.  In the staging path this is correct (prefill filled the buffer),
          but here the buffer is empty.
        """
        self.alloc_device_buffer(req)

        host_len = self.host_token_len(req.kv.kv_allocated_len)
        if host_len <= self.device_buffer_size:
            # Short sequences (seq_len <= device_buffer_size): the kernel fast path
            # returns device_buffer_locs directly without any host loading, so we
            # must preload all tokens from host pool into the device buffer
            # TODO(hzh0425): Optimize this.
            self._preload_to_device_buffer(req)
        else:
            # Long sequence: reset device_buffer_tokens to -1 so the kernel
            # sees all slots as empty -> every top-k lookup is a miss -> host load.
            self.req_device_buffer_tokens[
                :, req.req_pool_idx, : self.device_buffer_size
            ] = -1

        req.hisparse_staging = False
        self._skip_first_backup[req.req_pool_idx] = True
        logger.debug("HiSparse: admitting request %s directly", req.rid)

    def host_token_len(self, kv_allocated_len: int) -> int:
        if self.is_dsv4_hisparse:
            return kv_allocated_len // self.compress_ratio
        return kv_allocated_len

    def _preload_to_device_buffer(self, req: Req) -> None:
        """Preload all tokens from host pool into the device buffer."""
        n = self.host_token_len(req.kv.kv_allocated_len)
        host_indices = self.req_to_host_pool[req.req_pool_idx, :n]
        device_locs = self.req_to_device_buffer[req.req_pool_idx, :n]

        for layer_id in range(self.mem_pool_device.layer_num):
            self.mem_pool_host.load_to_device_per_layer(
                self.mem_pool_device,
                host_indices,
                device_locs,
                layer_id,
                io_backend="kernel",
            )

    def _record_buffer_identities(self, req: Req, buffer_alloc) -> None:
        self.req_device_buffer_tokens[
            :, req.req_pool_idx, : self.device_buffer_size
        ] = self._device_buffer_arange_i32

    def _commit_device_buffer(
        self, req: Req, alloc_size: int, buffer_indices: torch.Tensor, buffer_alloc
    ) -> None:
        """Record a freshly carved device buffer in the per-request tables.

        This block is the exact dual of request_finished's cleanup: every
        field written here must appear in its reset list. Keep it single
        across modes; the mode-specific parts (allocation geometry, allocator
        call, identity table) live in each class. ``buffer_alloc`` carries the
        retained logical ids for the MTP identity table; None in plain mode.
        """
        buffer_indices = buffer_indices.to(torch.int32)
        self.req_to_device_buffer[req.req_pool_idx, :alloc_size] = buffer_indices
        self.req_device_buffer_size[req.req_pool_idx] = alloc_size

        self._record_buffer_identities(req, buffer_alloc)
        self.req_device_buffer_token_locs[:, req.req_pool_idx, :alloc_size] = (
            buffer_indices[:alloc_size]
        )

    def alloc_device_buffer(self, req: Req) -> None:
        if self.is_dsv4_hisparse:
            allocated_len = req.extend_range.end
            alloc_size = self.padded_buffer_size
        else:
            allocated_len = req.kv.kv_allocated_len
            page_size = self.mem_pool_device.page_size
            # Allocate only enough for current tokens (page-aligned).
            # When prefill already fills device_buffer_size, include the reserved page.
            alloc_size = min(
                ((allocated_len + page_size - 1) // page_size) * page_size,
                self.device_buffer_size,
            )
            if alloc_size == self.device_buffer_size:
                alloc_size = self.padded_buffer_size

        compressed_logical_indices = (
            self.mem_pool_device.translate_loc_from_full_to_compressed(
                self.req_to_token_pool.req_to_token[req.req_pool_idx, :allocated_len]
            )
        )
        compressed_len = len(compressed_logical_indices)

        buffer_indices = self.token_to_kv_pool_allocator.alloc_device_buffer(
            compressed_logical_indices, alloc_size
        )
        if buffer_indices is None:
            logger.error(
                "HiSparse: alloc_device_buffer failed for req %s "
                "(compressed_len=%d, alloc_size=%d)",
                req.rid,
                compressed_len,
                alloc_size,
            )
            raise RuntimeError("HiSparse alloc_device_buffer returned None")

        self._commit_device_buffer(req, alloc_size, buffer_indices, None)

    def _drain_staging_request(self, req: Req) -> List[HiSparseAct]:
        """Wait for and detach every in-flight staging action for one request.

        Returns the drained actions so subclasses can release any resources
        they attached to them.
        """
        actions = []
        remaining = []
        for act in self.ack_staging_queue:
            if act.req is req:
                actions.append(act)
            else:
                remaining.append(act)
        self.ack_staging_queue = remaining

        for act in actions:
            act.finish_event.synchronize()

        # A staging flag without an action can only come from a partially
        # initialized/legacy path. Drain the stream before releasing storage.
        if req.hisparse_staging and not actions:
            self.write_staging_stream.synchronize()
        req.hisparse_staging = False
        return actions

    def _grow_device_buffers(
        self,
        seq_lens: torch.Tensor,
        req_pool_indices: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        req_pool_indices_cpu: torch.Tensor,
    ) -> torch.Tensor:
        """Grow device buffers for requests whose sequence length exceeds current capacity."""
        current_caps = self.req_device_buffer_size[req_pool_indices_cpu]
        short_reqs_cpu = seq_lens_cpu <= self.device_buffer_size
        needs_grow_cpu = short_reqs_cpu & (seq_lens_cpu > current_caps)

        if torch.any(needs_grow_cpu):
            page_size = self.mem_pool_device.page_size
            grow_indices = torch.where(needs_grow_cpu)[0]

            # Compute all grow sizes on CPU, then do a single bulk allocation
            req_idxs = []
            old_caps = []
            new_caps = []
            grow_sizes = []
            total_grow = 0
            for i in grow_indices.tolist():
                req_idx = int(req_pool_indices_cpu[i])
                current_cap = int(current_caps[i])
                seq_len = int(seq_lens_cpu[i])

                new_cap = min(
                    ((seq_len + page_size - 1) // page_size) * page_size,
                    self.device_buffer_size,
                )
                if new_cap == self.device_buffer_size:
                    new_cap = self.padded_buffer_size
                grow_size = new_cap - current_cap
                if grow_size <= 0:
                    continue
                req_idxs.append(req_idx)
                old_caps.append(current_cap)
                new_caps.append(new_cap)
                grow_sizes.append(grow_size)
                total_grow += grow_size

            if total_grow > 0:
                all_new_indices = (
                    self.token_to_kv_pool_allocator.hisparse_attn_allocator.alloc(
                        total_grow
                    )
                )
                if all_new_indices is None:
                    logger.error(
                        "HiSparse: _grow_device_buffers bulk alloc failed "
                        "(total_grow=%d)",
                        total_grow,
                    )
                    raise RuntimeError(
                        f"HiSparse _grow_device_buffers failed (total_grow={total_grow})"
                    )

                offset = 0
                for req_idx, current_cap, new_cap, grow_size in zip(
                    req_idxs, old_caps, new_caps, grow_sizes
                ):
                    chunk = all_new_indices[offset : offset + grow_size]
                    offset += grow_size
                    self.req_to_device_buffer[req_idx, current_cap:new_cap] = chunk
                    self.req_device_buffer_token_locs[
                        :, req_idx, current_cap:new_cap
                    ] = chunk
                    self.req_device_buffer_size[req_idx] = new_cap

        reserved_positions = (seq_lens - 1).clamp(max=self.device_buffer_size)
        return self.req_to_device_buffer[req_pool_indices, reserved_positions]

    def has_ongoing_staging(self) -> bool:
        return len(self.ack_staging_queue) > 0

    def _pop_ready_acks(self) -> List[HiSparseAct]:
        """Pop the TP-agreed prefix of DMA-completed staging actions."""
        if len(self.ack_staging_queue) == 0:
            return []

        finish_count = 0
        for act in self.ack_staging_queue:
            if not act.finish_event.query():
                break
            finish_count += 1
        queue_size = torch.tensor(finish_count, dtype=torch.int, device="cpu")
        if self.tp_world_size > 1:
            # synchronize TP workers to make sure the same update to scheduler
            torch.distributed.all_reduce(
                queue_size,
                op=torch.distributed.ReduceOp.MIN,
                group=self.tp_group,
            )
        finish_count = int(queue_size.item())
        acks = self.ack_staging_queue[:finish_count]
        del self.ack_staging_queue[:finish_count]
        return acks

    def collect_ready_reqs(self) -> List[Req]:
        ready_reqs: List[Req] = []
        for act in self._pop_ready_acks():
            req = act.req
            # prepare device buffer and update req
            self.alloc_device_buffer(req)
            self._skip_first_backup[req.req_pool_idx] = True
            req.hisparse_staging = False
            ready_reqs.append(req)
        return ready_reqs

    def map_last_loc_to_buffer(
        self,
        seq_lens: torch.Tensor,
        out_cache_loc: torch.Tensor,
        req_pool_indices: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        req_pool_indices_cpu: torch.Tensor,
    ) -> None:
        self._eager_backup_previous_token(
            seq_lens, req_pool_indices, seq_lens_cpu, req_pool_indices_cpu
        )

        if not self.is_dsv4_hisparse:
            # Grow device buffers if needed and resolve the latest-token slot.
            reserved_buffer_loc = self._grow_device_buffers(
                seq_lens, req_pool_indices, seq_lens_cpu, req_pool_indices_cpu
            )
            self.req_device_buffer_token_locs[
                :, req_pool_indices, self.device_buffer_size
            ] = reserved_buffer_loc.to(torch.int32)

            compressed_locs = self.token_to_kv_pool_allocator.get_last_loc_compressed(
                out_cache_loc
            )
            # ROCm: the decode remap creates a temporary hisparse device slot per
            # new token (via the page_size==1 allocator path). Free the stale
            # slot before pointing the mapping at the reserved device-buffer slot,
            # otherwise the temporary slots leak and corrupt later swap-in lookups.
            # CUDA keeps the original behavior: the swap-in kernel consumes only
            # top_k_device_locs, so stale mapping entries are harmless there.
            if _is_hip:
                previous_locs = self.mem_pool_device._translate_loc_to_hisparse_device(
                    compressed_locs
                )
                stale_locs = previous_locs[
                    (previous_locs > 0) & (previous_locs != reserved_buffer_loc)
                ]
                if stale_locs.numel() > 0:
                    self.token_to_kv_pool_allocator.free_hisparse_indices(stale_locs)

            self.mem_pool_device.full_to_hisparse_device_index_mapping[
                compressed_locs
            ] = reserved_buffer_loc
            return

        active_reqs = seq_lens % self.compress_ratio == 0
        if not torch.any(active_reqs):
            return

        active_seq_lens = seq_lens[active_reqs]
        active_out_cache_loc = out_cache_loc[active_reqs]
        active_req_pool_indices = req_pool_indices[active_reqs]

        compressed_seq_lens = active_seq_lens // self.compress_ratio
        reserved_positions = (compressed_seq_lens - 1).clamp(
            max=self.device_buffer_size
        )
        reserved_buffer_loc = self.req_to_device_buffer[
            active_req_pool_indices, reserved_positions
        ]

        self.req_device_buffer_token_locs[
            :, active_req_pool_indices, self.device_buffer_size
        ] = reserved_buffer_loc.to(torch.int32)

        compressed_locs = self.token_to_kv_pool_allocator.get_last_loc_compressed(
            active_out_cache_loc
        )
        self.mem_pool_device.full_to_hisparse_device_index_mapping[compressed_locs] = (
            reserved_buffer_loc
        )

    def _eager_backup_previous_token(
        self,
        seq_lens: torch.Tensor,
        req_pool_indices: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        req_pool_indices_cpu: torch.Tensor,
    ) -> None:
        """Back up the previous compressed token to host memory.

        Each newly produced compressed token (one per `compress_ratio` decode
        steps) must be backed up to host so the swap-in kernel can later
        recover it.

        Two cases are skipped:
        - The first decode step right after staging: all prefill tokens were
          already backed up during staging, so there is nothing new to save.
        - Steps where `(seq_len - 1) % compress_ratio != 0`: no new compressed
          token was produced this step.
        """
        # Build the list of batch positions that need a host backup.
        # Skip the first decode step after staging (prefill already backed up),
        # and skip non-aligned steps that did not produce a new compressed token.
        backup_indices = []
        for i in range(len(seq_lens_cpu)):
            req_idx = int(req_pool_indices_cpu[i])
            if self._skip_first_backup[req_idx]:
                self._skip_first_backup[req_idx] = False
                continue
            if (int(seq_lens_cpu[i]) - 1) % self.compress_ratio == 0:
                backup_indices.append(i)

        if not backup_indices:
            return

        backup_indices_gpu = torch.tensor(
            backup_indices, dtype=torch.int64, device=self.device
        )
        backup_req_indices = req_pool_indices[backup_indices_gpu]

        # The previous compressed token's position and its device buffer slot:
        #  compressed_pos = (seq_len - 1) // compress_ratio - 1
        #  - short: slot = compressed_pos          (within the regular buffer)
        #  - long:  slot = device_buffer_size      (the reserved slot)
        prev_seq_lens = seq_lens[backup_indices_gpu] - 1
        compressed_prev_seq_lens = prev_seq_lens // self.compress_ratio
        actual_compressed_pos = compressed_prev_seq_lens - 1

        buffer_slot = actual_compressed_pos.clamp(max=self.device_buffer_size)

        device_locs = self.req_to_device_buffer[backup_req_indices, buffer_slot]

        host_locs_list = []
        for i in backup_indices:
            req_idx = int(req_pool_indices_cpu[i])
            start_pos = (int(seq_lens_cpu[i]) - 1) // self.compress_ratio - 1
            host_locs = self.mem_pool_host.alloc_paged_token_slots(
                self.req_to_host_pool,
                self.req_to_host_pool_allocated_len,
                req_idx,
                start_pos,
                1,
            )
            host_locs_list.append(host_locs)
        host_locs = torch.cat(host_locs_list)

        self.wait_for_pending_backup()
        schedule_stream = device_module.current_stream()
        with device_module.stream(self.decode_backup_stream):
            self.decode_backup_stream.wait_stream(schedule_stream)
            if self.decode_producer_stream is not None:
                self.decode_backup_stream.wait_stream(self.decode_producer_stream)
            self.mem_pool_host.backup_from_device_all_layer(
                self.mem_pool_device,
                host_locs,
                device_locs,
                io_backend="kernel",
            )
            self._backup_done_event.record()
            if host_locs.is_cuda:
                host_locs.record_stream(self.decode_backup_stream)
            if backup_req_indices.is_cuda:
                backup_req_indices.record_stream(self.decode_backup_stream)
            if actual_compressed_pos.is_cuda:
                actual_compressed_pos.record_stream(self.decode_backup_stream)
            if device_locs.is_cuda:
                device_locs.record_stream(self.decode_backup_stream)
        self._has_pending_backup = True

    def wait_for_pending_backup(self) -> None:
        if not self._has_pending_backup:
            return
        self._backup_done_event.wait(device_module.current_stream())
        self._has_pending_backup = False

    def wait_admission_staging_before_verify(self, reqs: List[Req]) -> None:
        """No-op for plain decode: collect_ready_reqs only releases a request
        after finish_event.query() is true, so the buffer is ready before use.
        The MTP subclass, which admits eagerly, overrides this."""
        return

    def naive_load_topk(
        self,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        top_k_tokens: torch.Tensor,
        layer_id: int,
    ) -> torch.Tensor:
        """Load top-k selected tokens into device memory and return their device indices.

        This is a naive per-request loop implementation for debugging/validation.
        Production code uses swap_in_selected_pages (JIT CUDA kernel) instead.

        Note: dsv4 hisparse is not supported — DeepSeekV4SingleKVPoolHost has no
        load_to_device_per_layer and indices live in compressed space. Currently
        only used as a kernel oracle in test_hisparse_unit.py (non-dsv4 path).

        Args:
            req_pool_indices: Pool indices for each request.  Shape: (num_reqs,)
            seq_lens: Sequence lengths for each request.  Shape: (num_reqs,)
            top_k_tokens: Selected token positions per request.  Shape: (num_reqs, top_k)
            layer_id: The layer to load KV cache for.

        Returns:
            Device KV cache indices for the selected tokens.  Shape: (num_reqs, top_k)
        """
        assert (
            not self.is_dsv4_hisparse
        ), "naive_load_topk is not implemented for dsv4 hisparse"
        num_reqs = req_pool_indices.size(0)
        top_k_indices = torch.full(
            (num_reqs, self.top_k), -1, dtype=torch.int32, device=self.device
        )

        for i in range(num_reqs):
            seq_len = int(seq_lens[i].item())
            top_n = min(seq_len, self.top_k)
            if top_n == 0:
                continue

            req_idx = int(req_pool_indices[i].item())
            selected_tokens = top_k_tokens[i, :top_n].to(dtype=torch.int64)

            assert torch.all(
                selected_tokens >= 0
            ), f"Req {req_idx}: selected tokens contain negative positions"
            assert torch.all(selected_tokens < seq_len), (
                f"Req {req_idx}: selected tokens {selected_tokens.tolist()} "
                f"out of range for seq_len={seq_len}"
            )

            if seq_len <= self.device_buffer_size:
                device_indices = self.req_to_device_buffer[req_idx, selected_tokens]
            else:
                device_indices = torch.empty(
                    top_n, dtype=torch.int64, device=self.device
                )

                is_latest_token = selected_tokens == (seq_len - 1)
                needs_host_load = ~is_latest_token

                device_indices[is_latest_token] = self.req_to_device_buffer[
                    req_idx, self.device_buffer_size
                ]

                num_to_load = int(needs_host_load.sum().item())
                if num_to_load > 0:
                    tokens_to_load = selected_tokens[needs_host_load]
                    host_locs = self.req_to_host_pool[req_idx, tokens_to_load]

                    invalid_mask = host_locs < 0
                    if torch.any(invalid_mask):
                        bad_positions = tokens_to_load[invalid_mask].tolist()
                        raise AssertionError(
                            f"Req {req_idx} (seq_len={seq_len}, layer={layer_id}): "
                            f"missing host backup at token positions {bad_positions}"
                        )

                    buffer_locs = self.req_to_device_buffer[req_idx, :num_to_load]
                    device_indices[needs_host_load] = buffer_locs

                    self.mem_pool_host.load_to_device_per_layer(
                        self.mem_pool_device,
                        host_locs,
                        buffer_locs,
                        layer_id,
                        io_backend="kernel",
                    )

            top_k_indices[i, :top_n] = device_indices.to(torch.int32)

        return top_k_indices

    def abort_staging_request(self, req: Req) -> None:
        """Remove a request from the staging queue and free its host + device resources.

        Must be called when aborting a request that has been admitted into staging
        but has not yet completed (i.e. req.hisparse_staging is True).
        """
        # request_finished() drains any in-flight staging action before freeing
        # its host/device resources.
        self.request_finished(req)

    def retract_req(self, req: Req) -> None:
        if req.hisparse_staging:
            self.abort_staging_request(req)
        else:
            self.request_finished(req)

    def request_finished(self, req: Req):
        # Some finish paths call this method directly while prefill staging is
        # still in flight. Always resolve the per-request action first so its
        # source pages cannot race the DMA.
        self._drain_staging_request(req)

        # release resources only after the execution of a potential overlapped batch
        if self.decode_producer_stream is not None:
            device_module.current_stream().wait_stream(self.decode_producer_stream)
        self.wait_for_pending_backup()

        # Use kv_allocated_len (not seqlen): under speculative decoding the
        # allocator can over-allocate beyond the committed seqlen, and those
        # extra slots may carry stale mapping entries pointing at buffer slots
        # we just freed via free_hisparse_indices(all_hi). If left set, the
        # subsequent release_kv_cache -> allocator.free -> free_hisparse path
        # re-frees them (double-free into the page allocator's free list).
        allocated_len = req.kv.kv_allocated_len

        # release memory -- free buffer, subclass-owned regions, and mapped
        # pages at page granularity. Tokens may appear in several sources
        # (e.g. MTP ring slots also sit in the mapping scan below);
        # torch.unique dedups them.
        current_cap = int(self.req_device_buffer_size[req.req_pool_idx])
        physical_locs = []
        if current_cap > 0:
            side_buf_hi = self.req_to_device_buffer[req.req_pool_idx, :current_cap]
            valid_hi = side_buf_hi[side_buf_hi > 0]
            if valid_hi.numel() > 0:
                physical_locs.append(valid_hi)

        extra_locs = self._take_extra_physical_locs(req)
        if extra_locs is not None and extra_locs.numel() > 0:
            physical_locs.append(extra_locs)

        allocated_locs = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, :allocated_len
        ]
        compressed_locs = self.mem_pool_device.translate_loc_from_full_to_compressed(
            allocated_locs
        )
        mapped_hi = self.mem_pool_device.full_to_hisparse_device_index_mapping[
            compressed_locs
        ]
        mapped_hi = mapped_hi[mapped_hi > 0]
        if mapped_hi.numel() > 0:
            physical_locs.append(mapped_hi)

        if physical_locs:
            page_size = self.mem_pool_device.page_size
            all_hi = torch.cat(physical_locs)
            pages = torch.unique(all_hi // page_size)
            self.token_to_kv_pool_allocator.free_hisparse_indices(
                pages * page_size
            )
        self.mem_pool_device.full_to_hisparse_device_index_mapping[compressed_locs] = 0
        self._clear_token_positions(compressed_locs)

        host_indices = self.mem_pool_host.allocated_host_indices(
            self.req_to_host_pool,
            req.req_pool_idx,
            self.req_to_host_pool_allocated_len[req.req_pool_idx],
        )
        if host_indices.numel() > 0:
            self.mem_pool_host.free(host_indices)

        # clear req info
        self._clear_buffer_identities(req.req_pool_idx)
        self.req_device_buffer_token_locs[:, req.req_pool_idx, :] = -1
        self.req_to_device_buffer[req.req_pool_idx, :] = 0
        self.req_device_buffer_size[req.req_pool_idx] = 0
        self.req_to_host_pool[req.req_pool_idx, :] = -1
        self.req_to_host_pool_allocated_len[req.req_pool_idx] = 0
        self.lru_slots[:, req.req_pool_idx, :].copy_(self._lru_init)
        self._skip_first_backup[req.req_pool_idx] = False

    def swap_in_selected_pages(
        self,
        req_pool_indices: torch.Tensor,
        compressed_seq_lens: torch.Tensor,
        top_k_result: torch.Tensor,
        layer_id: int,
    ) -> torch.Tensor:
        """Swap selected top-k tokens into device memory and return their indices."""
        num_reqs = req_pool_indices.size(0)

        top_k_indices = self.top_k_device_locs_buffer[:num_reqs]
        top_k_indices.fill_(-1)

        swap_in_fn = (
            load_cache_to_device_buffer_dsv4_mla
            if self.is_dsv4_hisparse
            else load_cache_to_device_buffer_mla
        )
        swap_in_fn(
            top_k_tokens=top_k_result,
            device_buffer_tokens=self.req_device_buffer_tokens[layer_id],
            host_cache_locs=self.req_to_host_pool,
            device_buffer_locs=self.req_device_buffer_token_locs[layer_id],
            host_cache=self.mem_pool_host.kv_buffer[layer_id],
            device_buffer=self.mem_pool_device.kv_buffer[layer_id],
            top_k_device_locs=top_k_indices,
            req_pool_indices=req_pool_indices,
            seq_lens=compressed_seq_lens,
            lru_slots=self.lru_slots[layer_id],
            item_size_bytes=self.item_size_bytes,
            num_top_k=self.top_k,
            hot_buffer_size=self.device_buffer_size,
            page_size=1,
            block_size=self.swap_in_block_size,
            num_real_reqs=self.num_real_reqs,
        )
        return top_k_indices


class HiSparseMTPCoordinator(HiSparseCoordinator):
    """HiSparse coordinator for MTP (EAGLE speculative) decoding.

    Differences from the plain coordinator:
    - The buffer identity table keys slots by LOGICAL KV id (fused top-k
      output) instead of request-relative position, and the full padded
      buffer is allocated up front so any LRU slot is a valid DMA target.
    - Speculative tokens draw physical slots from a fixed per-request
      staging ring (position % capacity) instead of the global pool.
    - Decode-time swap-in happens per verify batch through
      :meth:`swap_in_verify_pages`; the position-keyed decode swap-in is
      unreachable except for idle batches.
    """

    mtp_enabled = True
    req_device_buffer_tokens = None

    def __init__(
        self,
        req_to_token_pool: ReqToTokenPool,
        token_to_kv_pool_allocator: Union[
            HiSparseTokenToKVPoolAllocator,
            DeepSeekV4HiSparseTokenToKVPoolAllocator,
        ],
        top_k: int,
        device_buffer_size: int,
        device: str,
        tp_group,
        host_to_device_ratio: int = 2,
        swap_in_block_size: int = 960,
        num_draft_tokens: int = 1,
        spec_ring_capacity: int = 0,
    ):
        super().__init__(
            req_to_token_pool=req_to_token_pool,
            token_to_kv_pool_allocator=token_to_kv_pool_allocator,
            top_k=top_k,
            device_buffer_size=device_buffer_size,
            device=device,
            tp_group=tp_group,
            host_to_device_ratio=host_to_device_ratio,
            swap_in_block_size=swap_in_block_size,
        )
        assert not self.is_dsv4_hisparse, (
            "MTP HiSparse (verify swap-in + speculative staging ring) is not "
            "supported for DSV4 HiSparse"
        )
        self.spec_ring_capacity = int(spec_ring_capacity)
        assert self.spec_ring_capacity > 0, (
            "HiSparseMTPCoordinator requires a positive spec_ring_capacity"
        )
        assert self.spec_ring_capacity % self.page_size == 0, (
            f"spec_ring_capacity ({self.spec_ring_capacity}) must be "
            f"page-aligned (page_size={self.page_size})"
        )

        max_num_req_slots = req_to_token_pool.req_to_token.shape[0]
        # Per-request speculative staging ring. Speculative tokens' physical
        # KV slots are drawn from this fixed, page-aligned region via
        # position % capacity, so long generations never grow the global
        # hisparse pool. Slot reuse clears the recycled token's mapping; the
        # verify kernel then resolves it via host DMA.
        self.req_to_spec_ring = torch.zeros(
            (max_num_req_slots, self.spec_ring_capacity),
            dtype=torch.int64,
            device=device,
        )
        self.req_spec_ring_active = [False] * max_num_req_slots
        # Staging finish_event per admitted request, consumed once on its
        # first verify so the forward stream orders after the device->host DMA.
        self._pending_first_verify_event: dict[int, device_module.Event] = {}

        # Output buffer for verify swap-in: [max_bs * N, top_k].
        max_verify_tokens = max_num_req_slots * num_draft_tokens
        self.verify_locs_buf = torch.full(
            (max_verify_tokens, self.top_k), -1,
            dtype=torch.int32, device=device,
        )

        mapping_size = int(
            self.token_to_kv_pool_allocator.full_to_hisparse_device_index_mapping.numel()
        )
        # Fused top-k returns logical ids. This stable inverse map converts
        # them to request-relative positions for host-cache lookup.
        self.full_to_token_position = torch.full(
            (mapping_size,), -1, dtype=torch.int32, device=device
        )

    # -- Mode hooks -----------------------------------------------------

    def _init_identity_table(self, layer_num: int, max_num_req_slots: int) -> None:
        self.req_device_buffer_logical_locs = torch.full(
            (layer_num, max_num_req_slots, self.padded_buffer_size),
            -1,
            dtype=torch.int32,
            device=self.device,
        )

    def alloc_device_buffer(
        self, req: Req, defer_free: bool = False
    ) -> Optional[torch.Tensor]:
        """Carve the full padded device buffer; returns deferred surplus pages.

        MTP verify always runs the eviction-based swap-in slow path, which may
        pick any LRU slot as a DMA destination, so every slot needs a valid
        physical loc from the start (hence the full padded size). With
        ``defer_free`` the surplus prefill pages are parked on the staging act
        instead of being released, because the staging DMA may still read them.
        """
        allocated_len = req.kv.kv_allocated_len
        alloc_size = self.padded_buffer_size

        compressed_logical_indices = (
            self.mem_pool_device.translate_loc_from_full_to_compressed(
                self.req_to_token_pool.req_to_token[req.req_pool_idx, :allocated_len]
            )
        )
        compressed_len = len(compressed_logical_indices)

        result = self.token_to_kv_pool_allocator.alloc_device_buffer_mtp(
            compressed_logical_indices,
            alloc_size,
            defer_free=defer_free,
        )
        if defer_free:
            buffer_indices, buffer_alloc, deferred_free_indices = result
        else:
            buffer_indices, buffer_alloc = result
            deferred_free_indices = None
        if buffer_indices is None:
            logger.error(
                "HiSparse: alloc_device_buffer failed for req %s "
                "(compressed_len=%d, alloc_size=%d)",
                req.rid,
                compressed_len,
                alloc_size,
            )
            raise RuntimeError("HiSparse alloc_device_buffer returned None")

        self._commit_device_buffer(req, alloc_size, buffer_indices, buffer_alloc)
        return deferred_free_indices

    def _record_buffer_identities(self, req: Req, buffer_alloc) -> None:
        self.req_device_buffer_logical_locs[:, req.req_pool_idx, :] = -1
        if buffer_alloc is not None and len(buffer_alloc) > 0:
            logical_locs = buffer_alloc.to(torch.int32)
            self.req_device_buffer_logical_locs[
                :, req.req_pool_idx, : logical_locs.numel()
            ] = logical_locs

    def _clear_buffer_identities(self, req_pool_idx: int) -> None:
        self.req_device_buffer_logical_locs[:, req_pool_idx, :] = -1

    def _clear_token_positions(self, compressed_locs: torch.Tensor) -> None:
        self.full_to_token_position[compressed_locs] = -1

    def register_token_positions(self, req: Req, start: int, end: int) -> None:
        """Record logical KV ids -> request-relative positions for MTP verify."""
        if end <= start:
            return
        full_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, start:end
        ].to(torch.int64)
        compressed_indices = self.mem_pool_device.translate_loc_from_full_to_compressed(
            full_indices
        )
        positions = torch.arange(
            start, end, dtype=torch.int32, device=full_indices.device
        )
        self.full_to_token_position[compressed_indices] = positions

    # -- Admission / teardown ---------------------------------------------

    def admit_request_into_staging(self, req: Req) -> None:
        """Staging admission plus the MTP-only eager setup.

        Unlike the plain coordinator (buffer carved at staging ack), the
        request joins the running MTP batch immediately, so the first verify
        may run before the DMA acks: the device buffer, the staging ring, and
        the position inverse map must all be ready at admission time.
        """
        # The base call registers the in-flight transfer before the eager
        # allocation below. If allocation raises, request_finished() can still
        # wait for the DMA and release the host allocation without racing the
        # staging stream.
        prefill_len = self.host_token_len(req.extend_range.end)
        super().admit_request_into_staging(req)
        action_index = len(self.ack_staging_queue) - 1
        # The request joins the running batch now and may verify before this
        # DMA acks; stash the event so the first verify waits on it.
        self._pending_first_verify_event[req.req_pool_idx] = self.ack_staging_queue[
            action_index
        ].finish_event

        # Eagerly allocate the device buffer so the first MTP verify has valid
        # buffer data. alloc_device_buffer only remaps hisparse indices to
        # buffer slots — no data movement — so it can happen before the DMA
        # completes. Surplus physical pages remain reserved until the backup
        # event fires. Under transient pool pressure (other requests' surplus
        # still deferred), reclaim and retry once; the failed attempt restored
        # the mapping, so retrying is safe.
        try:
            deferred_free_indices = self.alloc_device_buffer(req, defer_free=True)
        except RuntimeError:
            if not self._reclaim_deferred_staging_pages():
                raise
            deferred_free_indices = self.alloc_device_buffer(req, defer_free=True)
        self.ack_staging_queue[action_index] = self.ack_staging_queue[
            action_index
        ]._replace(deferred_free_indices=deferred_free_indices)
        self._alloc_spec_ring(req)
        req.hisparse_last_backed_len = prefill_len
        self.register_token_positions(
            req, 0, self.host_token_len(req.kv.kv_allocated_len)
        )
        self._skip_first_backup[req.req_pool_idx] = True
        if logger.isEnabledFor(logging.DEBUG):
            rid = req.req_pool_idx
            logger.debug(
                "HiSparse eager alloc: pool_idx=%d buf_locs[0:4]=%s "
                "buf_identity_locs[0:4]=%s",
                rid,
                self.req_device_buffer_token_locs[0, rid, :4].tolist(),
                self.req_device_buffer_logical_locs[0, rid, :4].tolist(),
            )

    def admit_request_direct(self, req: Req) -> None:
        """Direct-to-host path: KV data already resides in host pool via RDMA.

        Skips staging DMA entirely. Beyond the plain coordinator's buffer
        alloc + short-sequence preload, MTP admission must also set up the
        staging ring, the position inverse map, the backup start position,
        and the LOGICAL identities of any preloaded tokens (the verify
        kernel's buffer-match path keys on logical KV ids, not positions).
        """
        self.alloc_device_buffer(req)
        self._alloc_spec_ring(req)

        host_len = self.host_token_len(req.kv.kv_allocated_len)
        self.register_token_positions(req, 0, host_len)
        req.hisparse_last_backed_len = host_len
        if host_len <= self.device_buffer_size:
            # Short sequences (seq_len <= device_buffer_size): the kernel fast path
            # returns device_buffer_locs directly without any host loading, so we
            # must preload all tokens from host pool into the device buffer
            # TODO(hzh0425): Optimize this.
            self._preload_to_device_buffer(req)
            full_indices = self.req_to_token_pool.req_to_token[
                req.req_pool_idx, :host_len
            ].to(torch.int64)
            logical_locs = self.mem_pool_device.translate_loc_from_full_to_compressed(
                full_indices
            ).to(torch.int32)
            self.req_device_buffer_logical_locs[
                :, req.req_pool_idx, :host_len
            ] = logical_locs
        # Long sequences: logical identities were already reset to -1 by
        # _record_buffer_identities, so every verify top-k lookup misses and
        # loads from host.

        req.hisparse_staging = False
        self._skip_first_backup[req.req_pool_idx] = True
        logger.debug("HiSparse: admitting request %s directly", req.rid)

    def destroy(self) -> None:
        # Drain in-flight transfers so the buffer is idle, then unregister it.
        # Unlike the plain coordinator, in-flight staging acts own deferred
        # surplus pages (eager admission alloc); release them after the DMA
        # streams drain, before the host pool goes away.
        self.write_staging_stream.synchronize()
        self.decode_backup_stream.synchronize()
        for act in self.ack_staging_queue:
            self._release_deferred_free_indices(act.deferred_free_indices)
        self.ack_staging_queue.clear()
        self.mem_pool_host.destroy()

    def collect_ready_reqs(self) -> List[Req]:
        ready_reqs: List[Req] = []
        for act in self._pop_ready_acks():
            # The ack proves the staging DMA finished reading the deferred
            # surplus pages; buffer and ring were already carved eagerly at
            # admission, so releasing those pages is all that remains.
            self._release_deferred_free_indices(act.deferred_free_indices)
            act.req.hisparse_staging = False
            ready_reqs.append(act.req)
        return ready_reqs

    def _drain_staging_request(self, req: Req) -> List[HiSparseAct]:
        actions = super()._drain_staging_request(req)
        for act in actions:
            self._release_deferred_free_indices(act.deferred_free_indices)
        return actions

    def request_finished(self, req: Req):
        super().request_finished(req)
        # Per-request speculative tracking attributes (backup start position,
        # ring window start) die with the request slot.
        req.hisparse_last_backed_len = None
        req.hisparse_ring_start = None
        # Drop any unconsumed staging event (request finished before first
        # verify, e.g. immediate abort).
        self._pending_first_verify_event.pop(req.req_pool_idx, None)

    def wait_admission_staging_before_verify(self, reqs: List[Req]) -> None:
        """Order the forward stream after each request's admission staging DMA.

        MTP admits a request into the running batch before its device->host
        staging DMA acks (admit_request_into_staging), so the first verify's
        host-pool reads (swap_in_verify_pages host-miss path) can race the
        staging write. Make the current (forward) stream wait on the staging
        finish_event exactly once, on the first verify after admission.
        """
        if not self._pending_first_verify_event:
            return
        stream = device_module.current_stream()
        for req in reqs:
            event = self._pending_first_verify_event.pop(req.req_pool_idx, None)
            if event is not None:
                event.wait(stream)

    def _release_deferred_free_indices(
        self, deferred_free_indices: Optional[torch.Tensor]
    ) -> None:
        if deferred_free_indices is not None and deferred_free_indices.numel() > 0:
            self.token_to_kv_pool_allocator.free_hisparse_indices(
                deferred_free_indices
            )

    def _reclaim_deferred_staging_pages(self) -> bool:
        """Release surplus pages of in-flight staging actions.

        Admit-time allocations (device buffer refill, staging ring) can hit
        transient pool pressure: a long prompt holds all its prefill pages
        until the staging DMA finishes, even though most become surplus once
        the buffer consumed its share. Under pressure, wait for the DMA and
        release those deferred pages so the fixed per-request footprint can
        be carved out. Returns True if any pages were reclaimed.
        """
        reclaimed = False
        for i, act in enumerate(self.ack_staging_queue):
            deferred = act.deferred_free_indices
            if deferred is None or deferred.numel() == 0:
                continue
            act.finish_event.synchronize()
            self._release_deferred_free_indices(deferred)
            self.ack_staging_queue[i] = act._replace(deferred_free_indices=None)
            reclaimed = True
        return reclaimed

    # -- Speculative staging ring ----------------------------------------

    def _alloc_spec_ring(self, req: Req) -> None:
        """Allocate the per-request speculative staging ring (page-aligned)."""
        rid = req.req_pool_idx
        if self.req_spec_ring_active[rid]:
            return
        allocator = self.token_to_kv_pool_allocator.hisparse_attn_allocator
        ring_indices = allocator.alloc(self.spec_ring_capacity)
        if ring_indices is None and self._reclaim_deferred_staging_pages():
            ring_indices = allocator.alloc(self.spec_ring_capacity)
        if ring_indices is None:
            raise RuntimeError(
                f"HiSparse: spec ring allocation failed for req {req.rid} "
                f"(capacity={self.spec_ring_capacity})"
            )
        self.req_to_spec_ring[rid] = ring_indices
        self.req_spec_ring_active[rid] = True

    def assign_spec_ring_slots(self, req: Req, start: int, end: int) -> None:
        """Point new speculative logical slots at ring physical slots.

        Positions [start, end) map to ring slot ``position % capacity``. The
        previous owners of the reused slots (positions [start-R, end-R)) have
        long been committed and backed up to host, so their mapping is cleared
        here; verify resolves them through the swap-in kernel's host-DMA path.
        All writes are plain tensor ops outside any CUDA graph — replayed
        kernels read the updated mapping contents.
        """
        if end <= start:
            return
        self.assign_spec_ring_slots_batch([req], [start], [end])

    def assign_spec_ring_slots_batch(
        self,
        reqs: List[Req],
        starts: List[int],
        ends: List[int],
    ) -> None:
        """Batched ring assignment: one flat gather/scatter set per round.

        The per-request loop version issued ~7 tiny GPU launches per request
        on the scheduler thread every decode round; here the row/position
        indices are built on CPU and moved in a single H2D copy, so the launch
        count is constant in batch size.
        """
        R = self.spec_ring_capacity
        new_rows: List[int] = []
        new_pos: List[int] = []
        old_rows: List[int] = []
        old_pos: List[int] = []
        for req, start, end in zip(reqs, starts, ends):
            if end <= start:
                continue
            rid = req.req_pool_idx
            assert self.req_spec_ring_active[rid], (
                f"spec ring not allocated for req {req.rid} (pool_idx={rid})"
            )
            ring_start = req.hisparse_ring_start
            if ring_start is None:
                ring_start = start
                req.hisparse_ring_start = start
            last_backed = req.hisparse_last_backed_len
            if last_backed is None:
                last_backed = len(req.origin_input_ids)
            # Self-aliasing guard: two positions in the SAME round must never
            # land on one ring slot (pos % R). This also underwrites the
            # "old and new position sets are distinct" invariant below.
            assert end - start <= R, (
                f"spec ring too small: round allocates {end - start} positions "
                f"[{start}, {end}) but ring capacity is {R}, so positions would "
                "alias the same slot and silently overwrite KV. "
                "Increase spec_ring_capacity."
            )
            assert end - R <= max(last_backed, ring_start), (
                f"spec ring too small: recycling position {end - R} before it "
                f"was backed up (last_backed={last_backed}, capacity={R}). "
                "Increase spec_ring_capacity."
            )
            new_rows.extend([rid] * (end - start))
            new_pos.extend(range(start, end))
            # Recycle: positions [start-R, end-R) used the same ring slots.
            # Only positions >= ring_start were ring-assigned — never clear
            # prefill / buffer mappings below that. Old and new position sets
            # are distinct committed vs. fresh logical slots (no aliasing).
            old_start = max(start - R, ring_start)
            old_end = end - R
            if old_end > old_start:
                old_rows.extend([rid] * (old_end - old_start))
                old_pos.extend(range(old_start, old_end))

        if not new_pos:
            return

        mapping = self.token_to_kv_pool_allocator.full_to_hisparse_device_index_mapping
        req_to_token = self.req_to_token_pool.req_to_token

        if old_pos:
            old_rows_g = torch.tensor(old_rows, dtype=torch.int64).to(
                self.device, non_blocking=True
            )
            old_pos_g = torch.tensor(old_pos, dtype=torch.int64).to(
                self.device, non_blocking=True
            )
            old_compressed = self.mem_pool_device.translate_loc_from_full_to_compressed(
                req_to_token[old_rows_g, old_pos_g].to(torch.int64)
            )
            mapping[old_compressed] = 0

        rows_g = torch.tensor(new_rows, dtype=torch.int64).to(
            self.device, non_blocking=True
        )
        pos_g = torch.tensor(new_pos, dtype=torch.int64).to(
            self.device, non_blocking=True
        )
        compressed = self.mem_pool_device.translate_loc_from_full_to_compressed(
            req_to_token[rows_g, pos_g].to(torch.int64)
        )
        mapping[compressed] = self.req_to_spec_ring[rows_g, pos_g % R]
        # Register positions for host-cache lookup once these tokens are
        # recycled and must be fetched via DMA.
        self.full_to_token_position[compressed] = pos_g.to(torch.int32)

    def _free_spec_ring(self, req: Req) -> Optional[torch.Tensor]:
        """Detach the ring from the request; caller frees the returned locs."""
        rid = req.req_pool_idx
        if not self.req_spec_ring_active[rid]:
            return None
        ring_indices = self.req_to_spec_ring[rid].clone()
        self.req_to_spec_ring[rid] = 0
        self.req_spec_ring_active[rid] = False
        return ring_indices

    def _take_extra_physical_locs(self, req: Req) -> Optional[torch.Tensor]:
        return self._free_spec_ring(req)

    # -- Backup ----------------------------------------------------------

    def backup_committed_tokens(self, reqs: List[Req]) -> None:
        """Back up newly committed (accepted) tokens' KV to host.

        Called from eagle_prepare_for_decode before the next verify step,
        after process_batch_result_decode advanced kv_committed_len. Async
        and event-ordered (mirrors the non-spec decode path's
        _eager_backup_previous_token). Ordering guarantees:
         - the copy runs after the previous round's verify wrote the ring KV
           (backup stream waits on the schedule + producer streams);
         - consumers (the verify swap-in kernel's host DMA, and KV writes into
           recycled ring slots) run on the forward stream, which performs a
           GPU-side wait on _backup_done_event via wait_for_pending_backup()
           before every decode/verify forward;
         - device_locs is materialized before the mapping clears issued later
           by assign_spec_ring_slots, so those cannot retarget this copy.
        A CPU-blocking synchronize() here would drain the whole pipeline every
        round and defeat the overlap scheduler.
        """
        all_host_locs = []
        all_device_locs = []
        pending_backed_lens = []
        for req in reqs:
            # Track last-backed-up position per request.
            # host_allocated_len may exceed committed_len (page-aligned
            # staging), so compare against the actual backed-up position.
            last_backed = req.hisparse_last_backed_len
            if last_backed is None:
                # First call: staging already backed up prefill tokens.
                last_backed = len(req.origin_input_ids)
            committed = req.kv_committed_len
            if committed <= last_backed:
                continue
            new_count = committed - last_backed
            host_locs = self.mem_pool_host.alloc_paged_token_slots(
                self.req_to_host_pool,
                self.req_to_host_pool_allocated_len,
                req.req_pool_idx,
                last_backed,
                new_count,
            )
            logical_locs = self.req_to_token_pool.req_to_token[
                req.req_pool_idx, last_backed:committed
            ].to(torch.int64)
            device_locs = self.mem_pool_device.translate_loc_to_hisparse_device(
                logical_locs
            )
            all_host_locs.append(host_locs)
            all_device_locs.append(device_locs)
            pending_backed_lens.append((req, committed))

        if not all_host_locs:
            return

        host_locs = torch.cat(all_host_locs)
        device_locs = torch.cat(all_device_locs)
        self.wait_for_pending_backup()
        schedule_stream = device_module.current_stream()
        with device_module.stream(self.decode_backup_stream):
            self.decode_backup_stream.wait_stream(schedule_stream)
            if self.decode_producer_stream is not None:
                self.decode_backup_stream.wait_stream(self.decode_producer_stream)
            self.mem_pool_host.backup_from_device_all_layer(
                self.mem_pool_device,
                host_locs,
                device_locs,
                io_backend="kernel",
            )
            self._backup_done_event.record()
            if host_locs.is_cuda:
                host_locs.record_stream(self.decode_backup_stream)
            if device_locs.is_cuda:
                device_locs.record_stream(self.decode_backup_stream)
        self._has_pending_backup = True
        # Backed tokens keep their mapping (staging-ring slot) until the ring
        # wraps: assign_spec_ring_slots clears it when the slot is recycled for
        # a new position, after which verify fetches the token via host DMA.
        # The physical footprint per request is therefore fixed (buffer + ring).
        for req, committed in pending_backed_lens:
            req.hisparse_last_backed_len = committed

    # -- Swap-in ----------------------------------------------------------

    def swap_in_selected_pages(
        self,
        req_pool_indices: torch.Tensor,
        compressed_seq_lens: torch.Tensor,
        top_k_result: torch.Tensor,
        layer_id: int,
    ) -> torch.Tensor:
        # Unreachable by construction: MTP decode runs through target-verify
        # (swap_in_verify_pages) and IDLE early-returns in the attention
        # dispatch. Fail fast instead of inheriting the base kernel, which
        # would crash cryptically on the absent position identity table.
        raise RuntimeError(
            "Position-based HiSparse swap-in cannot run in MTP mode"
        )

    def swap_in_verify_pages(
        self,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        logical_top_k: torch.Tensor,
        layer_id: int,
        num_positions: int,
    ) -> torch.Tensor:
        """Swap-in for all N verify positions at once via the MTP kernel.

        Args:
            logical_top_k: [bs*N, top_k] physical slot indices (fused topk output).
            num_positions: N tokens per request.

        Returns:
            [bs*N, top_k] hisparse device indices.
        """
        bs = req_pool_indices.size(0)
        total_rows = bs * num_positions
        top_k_indices = self.verify_locs_buf[:total_rows]
        top_k_indices.fill_(-1)

        load_cache_to_device_buffer_mtp_mla(
            top_k_tokens=logical_top_k,
            device_buffer_tokens=self.req_device_buffer_logical_locs[layer_id],
            host_cache_locs=self.req_to_host_pool,
            device_buffer_locs=self.req_device_buffer_token_locs[layer_id],
            host_cache=self.mem_pool_host.kv_buffer[layer_id],
            device_buffer=self.mem_pool_device.kv_buffer[layer_id],
            top_k_device_locs=top_k_indices,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            lru_slots=self.lru_slots[layer_id],
            item_size_bytes=self.item_size_bytes,
            num_top_k=self.top_k,
            hot_buffer_size=self.device_buffer_size,
            num_draft_tokens=num_positions,
            page_size=1,
            block_size=self.swap_in_block_size,
            num_real_reqs=self.num_real_reqs,
            full_to_hisparse_device=self.token_to_kv_pool_allocator.full_to_hisparse_device_index_mapping,
            full_to_token_position=self.full_to_token_position,
        )
        return top_k_indices
