"""Unit tests for HiSparse hierarchical sparse KV cache system.

Tests cover:
- CUDA kernel correctness (swap_in_selected_pages vs naive_load_topk oracle)
- Memory allocator lifecycle (alloc / free / available_size)
- Request lifecycle (staging path, direct-to-host path)
- Batch multi-request correctness
"""

import os
import unittest
from array import array
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.utils import is_cuda, is_hip, is_npu, is_xpu
from sglang.srt.utils.common import Range
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")
register_amd_ci(est_time=10, suite="stage-b-test-1-gpu-small-amd")

# ---------------------------------------------------------------------------
# Test configuration (small-scale for fast CI runs)
# ---------------------------------------------------------------------------
SIZE = 2048  # device buffer pool size (tokens)
PAGE_SIZE = 64  # page size (must be 64 for CUDA, 1 for ROCm)
TOP_K = 256  # top-k selection count
DEVICE_BUFFER_SIZE = 512  # device buffer per request
HOST_TO_DEVICE_RATIO = 2
KV_LORA_RANK = 512
QK_ROPE_HEAD_DIM = 64
KV_CACHE_DIM = 576  # MLA dim (DeepSeek-style)
LAYER_NUM = 2
MAX_NUM_REQS = 8
MAX_CONTEXT_LEN = 2048


def _make_req(rid="test-req-0", origin_input_ids=None, output_ids=None):
    """Create a minimal mock Req object with the fields HiSparseCoordinator uses."""
    if origin_input_ids is None:
        origin_input_ids = list(range(64))
    if output_ids is None:
        output_ids = []
    req = SimpleNamespace(
        rid=rid,
        origin_input_ids=origin_input_ids,
        output_ids=output_ids,
        fill_ids=origin_input_ids + output_ids,
        seqlen=len(origin_input_ids) + len(output_ids),
        req_pool_idx=None,
        kv=SimpleNamespace(kv_allocated_len=0),
        kv_committed_len=0,
        finished_reason=None,
        hisparse_staging=False,
        hisparse_last_backed_len=None,
        hisparse_ring_start=None,
        staging=False,
        inflight_middle_chunks=0,
    )
    req.finished = lambda: req.finished_reason is not None
    req.set_extend_range = lambda start, end: setattr(
        req, "extend_range", Range(start, end)
    )
    return req


class TestHiSparseUnit(unittest.TestCase):
    """Test class that builds a minimal HiSparse component stack."""

    # ==================================================================
    # Fixture
    # ==================================================================

    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is required for HiSparse tests.")
        if is_npu() or is_xpu():
            raise unittest.SkipTest("HiSparse tests only support CUDA/ROCm.")
        if not (is_cuda() or is_hip()):
            raise unittest.SkipTest("CUDA/ROCm not available.")

        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29599")
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend="gloo", rank=0, world_size=1)
        cls.tp_group = torch.distributed.group.WORLD

        from sglang.srt.mem_cache.pool_host.common import (
            ALLOC_MEMORY_FUNCS,
            alloc_with_pin_memory,
        )

        cls._original_alloc = ALLOC_MEMORY_FUNCS["cuda"]
        ALLOC_MEMORY_FUNCS["cuda"] = alloc_with_pin_memory

        if is_hip():
            from sglang.srt.layers.attention.dsa.utils import (
                aiter_can_use_preshuffle_paged_mqa,
            )

            global_page_size = 64 if aiter_can_use_preshuffle_paged_mqa() else 1
        else:
            global_page_size = PAGE_SIZE

        from sglang.srt.mem_cache.allocator.hisparse import (
            HiSparseTokenToKVPoolAllocator,
        )
        from sglang.srt.mem_cache.hisparse_memory_pool import HiSparseDSATokenToKVPool

        cls.device_pool = HiSparseDSATokenToKVPool(
            size=SIZE,
            page_size=global_page_size,
            kv_lora_rank=KV_LORA_RANK,
            dtype=torch.bfloat16,
            qk_rope_head_dim=QK_ROPE_HEAD_DIM,
            layer_num=LAYER_NUM,
            device="cuda",
            index_head_dim=128,
            enable_memory_saver=False,
            kv_cache_dim=KV_CACHE_DIM,
            host_to_device_ratio=HOST_TO_DEVICE_RATIO,
        )
        cls.allocator = HiSparseTokenToKVPoolAllocator(
            size=SIZE,
            page_size=global_page_size,
            dtype=torch.bfloat16,
            device="cuda",
            kvcache=cls.device_pool,
            need_sort=False,
            host_to_device_ratio=HOST_TO_DEVICE_RATIO,
        )

        from sglang.srt.mem_cache.memory_pool import ReqToTokenPool

        cls.req_to_token_pool = ReqToTokenPool(
            size=MAX_NUM_REQS,
            max_context_len=MAX_CONTEXT_LEN,
            device="cuda",
            enable_memory_saver=False,
        )

        from sglang.srt.managers.hisparse_coordinator import (
            HiSparseCoordinator,
            HiSparseMTPCoordinator,
        )

        cls.page_size = global_page_size
        cls.non_mtp_coordinator = HiSparseCoordinator(
            req_to_token_pool=cls.req_to_token_pool,
            token_to_kv_pool_allocator=cls.allocator,
            top_k=TOP_K,
            device_buffer_size=DEVICE_BUFFER_SIZE,
            device="cuda",
            tp_group=cls.tp_group,
            host_to_device_ratio=HOST_TO_DEVICE_RATIO,
        )
        cls.mtp_coordinator = HiSparseMTPCoordinator(
            req_to_token_pool=cls.req_to_token_pool,
            token_to_kv_pool_allocator=cls.allocator,
            top_k=TOP_K,
            device_buffer_size=DEVICE_BUFFER_SIZE,
            device="cuda",
            tp_group=cls.tp_group,
            host_to_device_ratio=HOST_TO_DEVICE_RATIO,
            spec_ring_capacity=global_page_size,
        )

    @classmethod
    def tearDownClass(cls):
        from sglang.srt.mem_cache.pool_host.common import ALLOC_MEMORY_FUNCS

        ALLOC_MEMORY_FUNCS["cuda"] = cls._original_alloc
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()

    def setUp(self):
        """Reset shared allocator / coordinator state so tests are isolated.

        Without this, a mid-test assertion failure skips cleanup and leaks
        resources, causing unrelated failures in later tests.
        """
        mtp_tests = {
            "test_mtp_long_staging_swaps_buffer_and_host_logical_tokens",
            "test_mtp_accepted_kv_survives_backup_and_direct_page_release",
            "test_mtp_direct_path_uses_logical_identities",
            "test_mtp_position_swap_in_raises",
            "test_spec_ring_pool_stable_across_rounds",
            "test_spec_ring_wrap_recycles_mapping",
            "test_spec_mode_allocates_full_device_buffer",
            "test_admit_long_prompt_reclaims_deferred_pages_for_ring",
            # Eager admission-time alloc (deferred surplus pages) is MTP-only.
            "test_staging_defers_surplus_pages_until_backup_finishes",
            "test_staging_action_survives_eager_allocation_failure",
            # Hybrid: resident free path exercises the MTP coordinator override.
            "test_resident_request_finished_no_leak",
            # Hybrid: mid-decode offload migration needs buffer + spec ring.
            "test_offload_running_request_migrates_resident_req",
            "test_offload_then_restore_round_trip",
            "test_restore_drains_inflight_staging",
            # Hybrid single-graph: resident rows run the verify swap-in kernel.
            "test_resident_verify_swap_in_is_pure_gather",
            # DP padding rows must not be treated as real requests.
            "test_verify_swap_in_ignores_dp_padding_rows",
            # Offloaded MTP consumes logical slots only; memcheck must agree.
            "test_offloaded_decode_memcheck_uses_logical_capacity",
            # Headroom gate must credit each request's own reclaimable surplus.
            "test_hybrid_offload_proceeds_when_surplus_funds_migration",
        }
        self.coordinator = (
            self.mtp_coordinator
            if self._testMethodName in mtp_tests
            else self.non_mtp_coordinator
        )

        self.allocator.clear()
        self.req_to_token_pool.clear()
        self.non_mtp_coordinator.mem_pool_host.clear()
        self.mtp_coordinator.mem_pool_host.clear()
        # Reset per-request coordinator bookkeeping
        self.coordinator.req_to_device_buffer.zero_()
        self.coordinator.req_device_buffer_size.zero_()
        self.coordinator.req_to_host_pool.fill_(-1)
        self.coordinator.req_to_host_pool_allocated_len.zero_()
        if self.coordinator.req_device_buffer_tokens is not None:
            self.coordinator.req_device_buffer_tokens.fill_(-1)
        self.coordinator.req_device_buffer_token_locs.fill_(-1)
        if self.coordinator.req_device_buffer_logical_locs is not None:
            self.coordinator.req_device_buffer_logical_locs.fill_(-1)
        if self.coordinator.full_to_token_position is not None:
            self.coordinator.full_to_token_position.fill_(-1)
        self.coordinator.num_real_reqs.zero_()
        self.coordinator.lru_slots[:] = self.coordinator._lru_init.view(1, 1, -1)
        self.coordinator.ack_staging_queue.clear()
        self.coordinator._has_pending_backup = False
        if self.coordinator is self.mtp_coordinator:
            self.coordinator.spec_ring_capacity = self.page_size
            self.coordinator.req_to_spec_ring = torch.zeros(
                (MAX_NUM_REQS, self.page_size),
                dtype=torch.int64,
                device="cuda",
            )
            self.coordinator.req_spec_ring_active = [False] * MAX_NUM_REQS
        for i in range(len(self.coordinator._skip_first_backup)):
            self.coordinator._skip_first_backup[i] = False

    # ==================================================================
    # Low-level helpers
    # ==================================================================

    def _alloc_req_slot(self, req):
        """Allocate a req_pool_idx for the request."""
        indices = self.req_to_token_pool.alloc([req])
        self.assertIsNotNone(indices, "Failed to allocate req pool slot")
        return req.req_pool_idx

    def _free_req_slot(self, req):
        """Free the req_pool_idx."""
        if req.req_pool_idx is not None:
            self.req_to_token_pool.free(req)

    def _alloc_kv(self, req, fill_len, *, logical_only=False):
        """Allocate KV indices, write req_to_token_pool, update req fields.
        If logical_only=True, uses alloc_logical_only (PD-separated path).
        Returns kv_loc tensor."""
        device = self.allocator.device
        alloc_fn = (
            self.allocator.alloc_logical_only
            if logical_only
            else self.allocator.alloc_extend
        )
        kv_loc = alloc_fn(
            prefix_lens=torch.tensor([0], dtype=torch.int64, device=device),
            prefix_lens_cpu=torch.tensor([0], dtype=torch.int64),
            seq_lens=torch.tensor([fill_len], dtype=torch.int64, device=device),
            seq_lens_cpu=torch.tensor([fill_len], dtype=torch.int64),
            last_loc=torch.tensor([-1], dtype=torch.int64, device=device),
            extend_num_tokens=fill_len,
        )
        self.assertIsNotNone(kv_loc, "KV alloc failed")
        self.req_to_token_pool.write((req.req_pool_idx, slice(0, len(kv_loc))), kv_loc)
        req.kv.kv_allocated_len = fill_len
        req.kv_committed_len = fill_len
        req.full_untruncated_fill_ids = array("q", range(fill_len))
        req.extend_range = Range(0, fill_len)
        return kv_loc

    # ==================================================================
    # Mid-level helpers
    # ==================================================================

    @staticmethod
    def _kv_pattern(layer_id, token_id):
        """Deterministic KV value for (layer, token) — used by write & verify."""
        v = (layer_id * 10000 + token_id + 1) * 0.001
        return float(torch.tensor(v, dtype=torch.bfloat16))

    def _write_device_patterns(self, kv_loc, fill_len):
        """Write distinguishable patterns into device KV buffer for all layers.

        kv_loc contains *logical* indices; we must translate them to hisparse
        device indices before indexing kv_buffer (which is sized for the
        hisparse pool, not the larger logical space).
        """
        hisparse_locs = self.allocator.full_to_hisparse_device_index_mapping[kv_loc]
        for lid in range(LAYER_NUM):
            for i in range(fill_len):
                self.device_pool.kv_buffer[lid][hisparse_locs[i]] = self._kv_pattern(
                    lid, i
                )

    def _populate_host_pool(self, req, fill_len):
        """Allocate host slots, write known patterns, register in coordinator.
        Returns host_indices (cuda tensor)."""
        host_pool = self.coordinator.mem_pool_host
        host_indices = host_pool.alloc(fill_len)
        self.assertIsNotNone(host_indices, "Host alloc failed")
        host_indices = host_indices.to(device="cuda")
        self.coordinator.req_to_host_pool[req.req_pool_idx, :fill_len] = host_indices
        self.coordinator.req_to_host_pool_allocated_len[req.req_pool_idx] = fill_len
        for lid in range(LAYER_NUM):
            for i in range(fill_len):
                host_pool.kv_buffer[lid][host_indices[i]] = self._kv_pattern(lid, i)
        return host_indices

    def _build_topk_tokens(self, fill_len, *, include_newest=False):
        """Build a 1-D [TOP_K] int32 cuda tensor of token positions.

        If include_newest=True, fill_len-1 is guaranteed as the last valid slot.
        Pads with -1 when fill_len (or fill_len-1) < TOP_K.

        For long-sequence tests (fill_len > DEVICE_BUFFER_SIZE) where the
        "newest token" reserved slot is not populated (it requires an actual
        decode step + map_last_loc_to_buffer), callers should pass
        ``fill_len - 1`` as the effective pool size so position fill_len-1 is
        never randomly selected.
        """
        n = min(fill_len, TOP_K)
        if include_newest and n > 1:
            tokens = torch.randperm(fill_len - 1, device="cuda")[: n - 1].to(
                torch.int32
            )
            tokens = torch.cat(
                [tokens, torch.tensor([fill_len - 1], dtype=torch.int32, device="cuda")]
            )
        else:
            tokens = torch.randperm(fill_len, device="cuda")[:n].to(torch.int32)
        if n < TOP_K:
            pad = torch.full((TOP_K - n,), -1, dtype=torch.int32, device="cuda")
            tokens = torch.cat([tokens, pad])
        return tokens

    def _make_batch_tensors(self, reqs, fill_lens):
        """Build (req_pool_indices [int64], seq_lens [int32]) on cuda."""
        rpi = torch.tensor(
            [r.req_pool_idx for r in reqs], dtype=torch.int64, device="cuda"
        )
        sls = torch.tensor(fill_lens, dtype=torch.int32, device="cuda")
        return rpi, sls

    def _assert_kv_correct(self, locs_row, tokens_row, layer_id, count, msg=""):
        """Assert device KV data at *locs_row[:count]* matches the written
        pattern for the corresponding *tokens_row[:count]* positions."""
        for i in range(count):
            tok = int(tokens_row[i].item())
            if tok < 0:
                continue
            expected = self._kv_pattern(layer_id, tok)
            actual = self.device_pool.kv_buffer[layer_id][locs_row[i].long()]
            self.assertTrue(
                torch.allclose(
                    actual.float(),
                    torch.full_like(actual.float(), expected),
                    atol=1e-2,
                ),
                f"{msg}layer {layer_id}, token {tok}: KV data mismatch",
            )

    def _assert_matches_naive(self, rpi, sls, batch, kernel_locs, layer_id, msg=""):
        """Assert kernel swap_in KV data matches naive_load_topk KV data."""
        naive_locs = self.coordinator.naive_load_topk(rpi, sls, batch, layer_id)
        for b in range(batch.shape[0]):
            for i in range(TOP_K):
                if batch[b, i] < 0:
                    continue
                naive_data = self.device_pool.kv_buffer[layer_id][
                    naive_locs[b, i].long()
                ]
                kernel_data = self.device_pool.kv_buffer[layer_id][
                    kernel_locs[b, i].long()
                ]
                self.assertTrue(
                    torch.allclose(naive_data.float(), kernel_data.float(), atol=1e-2),
                    f"{msg}layer {layer_id}, b{b} idx {i}: naive != kernel",
                )

    def _swap_in_selected_pages(
        self,
        rpi: torch.Tensor,
        sls: torch.Tensor,
        batch: torch.Tensor,
        layer_id: int,
    ) -> torch.Tensor:
        """Wrapper that sets num_real_reqs before calling swap_in_selected_pages.

        In production, model_runner sets num_real_reqs before each forward
        pass.  Tests must replicate that to get correct kernel behaviour.
        """
        self.coordinator.num_real_reqs[0] = rpi.shape[0]
        return self.coordinator.swap_in_selected_pages(rpi, sls, batch, layer_id)

    def _swap_in_selected_logical_pages(
        self,
        rpi: torch.Tensor,
        sls: torch.Tensor,
        batch: torch.Tensor,
        layer_id: int,
    ) -> torch.Tensor:
        # Exercise the production verify path with a single position per req.
        self.coordinator.num_real_reqs[0] = rpi.shape[0]
        return self.coordinator.swap_in_verify_pages(
            req_pool_indices=rpi,
            seq_lens=sls,
            logical_top_k=batch,
            layer_id=layer_id,
            num_positions=1,
        )

    def _clear_mapping(self, logical_locs):
        """Test-side replacement for the removed release_backed_logical_locs:
        clear the global mapping for backed/recycled tokens (production does
        this inside assign_spec_ring_slots_batch / request_finished)."""
        compressed = self.coordinator.mem_pool_device.translate_loc_from_full_to_compressed(
            logical_locs
        )
        self.allocator.full_to_hisparse_device_index_mapping[compressed] = 0

    def _collect_ready_reqs_blocking(self):
        """Wait for in-flight staging DMA, then collect ready requests.

        Test-side replacement for the removed coordinator helper: production
        polls collect_ready_reqs non-blockingly per scheduler iteration."""
        for act in self.coordinator.ack_staging_queue:
            act.finish_event.synchronize()
        return self.coordinator.collect_ready_reqs()

    def _cleanup_req(self, req, kv_loc, *, logical_only=False):
        """request_finished -> free KV -> free req slot."""
        self.coordinator.request_finished(req)
        if logical_only:
            self.allocator.logical_attn_allocator.free(kv_loc)
        else:
            self.allocator.free(kv_loc)
        self._free_req_slot(req)

    def _get_initial_sizes(self):
        """Snapshot allocator available sizes."""
        return (
            self.allocator.logical_attn_allocator.available_size(),
            self.allocator.hisparse_attn_allocator.available_size(),
            self.coordinator.mem_pool_host.available_size(),
        )

    def _assert_sizes_restored(self, initial_sizes, msg=""):
        """Assert allocator sizes match the snapshot."""
        logical, hisparse, host = self._get_initial_sizes()
        self.assertEqual(logical, initial_sizes[0], f"Logical leak {msg}")
        self.assertEqual(hisparse, initial_sizes[1], f"HiSparse leak {msg}")
        self.assertEqual(host, initial_sizes[2], f"Host leak {msg}")

    # ==================================================================
    # Test: Kernel correctness — short sequence (fast path)
    # ==================================================================
    def test_buffer_identity_tables_are_mode_specific(self):
        self.assertFalse(self.non_mtp_coordinator.mtp_enabled)
        self.assertIsNotNone(self.non_mtp_coordinator.req_device_buffer_tokens)
        self.assertIsNone(
            self.non_mtp_coordinator.req_device_buffer_logical_locs
        )
        self.assertTrue(self.mtp_coordinator.mtp_enabled)
        self.assertIsNone(self.mtp_coordinator.req_device_buffer_tokens)
        self.assertIsNotNone(
            self.mtp_coordinator.req_device_buffer_logical_locs
        )

    def test_spec_logical_alloc_context(self):
        """Inside spec_logical_alloc, alloc/alloc_extend hand out logical
        slots only: the physical pool and the mapping stay untouched."""
        a = self.allocator
        device = a.device
        n = self.page_size
        physical_before = a.hisparse_attn_allocator.available_size()

        with a.spec_logical_alloc():
            kv_loc = a.alloc_extend(
                prefix_lens=torch.tensor([0], dtype=torch.int64, device=device),
                prefix_lens_cpu=torch.tensor([0], dtype=torch.int64),
                seq_lens=torch.tensor([n], dtype=torch.int64, device=device),
                seq_lens_cpu=torch.tensor([n], dtype=torch.int64),
                last_loc=torch.tensor([-1], dtype=torch.int64, device=device),
                extend_num_tokens=n,
            )
        self.assertIsNotNone(kv_loc)
        self.assertEqual(
            a.hisparse_attn_allocator.available_size(), physical_before
        )
        self.assertTrue(
            torch.all(a.full_to_hisparse_device_index_mapping[kv_loc] == 0)
        )
        self.assertFalse(a._spec_logical)
        a.logical_attn_allocator.free(kv_loc)

    def test_mtp_position_swap_in_raises(self):
        """The position-keyed decode swap-in is unreachable in MTP mode
        (decode runs through swap_in_verify_pages; IDLE early-returns in
        the attention dispatch) — the override must fail fast."""
        req_pool_indices = torch.zeros(1, dtype=torch.int64, device="cuda")
        seq_lens = torch.zeros(1, dtype=torch.int32, device="cuda")
        top_k = torch.zeros((1, TOP_K), dtype=torch.int32, device="cuda")

        with self.assertRaisesRegex(RuntimeError, "MTP mode"):
            self.coordinator.swap_in_selected_pages(
                req_pool_indices,
                seq_lens,
                top_k,
                layer_id=0,
            )

    def test_mtp_direct_path_uses_logical_identities(self):
        initial = self._get_initial_sizes()
        fill_len = self.page_size
        req = _make_req("mtp-direct", list(range(fill_len)))
        self._alloc_req_slot(req)

        kv_loc = self._alloc_kv(req, fill_len, logical_only=True)
        self._populate_host_pool(req, fill_len)
        self.coordinator.admit_request_direct(req)

        identities = self.coordinator.req_device_buffer_logical_locs[
            0, req.req_pool_idx, :fill_len
        ].to(torch.int64)
        self.assertTrue(torch.equal(identities, kv_loc))

        logical_top_k = kv_loc[:TOP_K].to(torch.int32)
        if fill_len < TOP_K:
            logical_top_k = torch.cat(
                [
                    logical_top_k,
                    torch.full(
                        (TOP_K - fill_len,),
                        -1,
                        dtype=torch.int32,
                        device="cuda",
                    ),
                ]
            )
        rpi, sls = self._make_batch_tensors([req], [fill_len])
        locs = self._swap_in_selected_logical_pages(
            rpi,
            sls,
            logical_top_k.unsqueeze(0),
            layer_id=0,
        )
        self.assertTrue(torch.all(locs[0, :fill_len] >= 0))

        self._cleanup_req(req, kv_loc, logical_only=True)
        self._assert_sizes_restored(initial, "mtp_direct")

    def test_kernel_correctness_short_seq(self):
        """Short seq (len <= device_buffer_size): kernel fast path returns
        device buffer locs, matching naive_load_topk."""
        initial = self._get_initial_sizes()
        req = _make_req("short-seq", list(range(self.page_size)))
        self._alloc_req_slot(req)

        fill_len = self.page_size
        kv_loc = self._alloc_kv(req, fill_len)
        self._write_device_patterns(kv_loc, fill_len)
        self.coordinator.alloc_device_buffer(req)

        tokens = self._build_topk_tokens(fill_len)
        batch = tokens.unsqueeze(0)
        rpi, sls = self._make_batch_tensors([req], [fill_len])

        for lid in range(LAYER_NUM):
            naive_locs = self.coordinator.naive_load_topk(rpi, sls, batch, lid)
            kernel_locs = self._swap_in_selected_pages(rpi, sls, batch, lid)
            valid = batch[0] >= 0
            self.assertTrue(
                torch.equal(naive_locs[0][valid].cpu(), kernel_locs[0][valid].cpu()),
                f"Layer {lid}: kernel locs != naive oracle",
            )

        self._cleanup_req(req, kv_loc)
        self._assert_sizes_restored(initial, "short_seq")

    # ==================================================================
    # Test: Kernel correctness — long sequence (cache miss + host DMA)
    # ==================================================================
    def test_kernel_correctness_long_seq(self):
        """Long seq (len > device_buffer_size): kernel loads from host,
        matching naive_load_topk for data correctness."""
        initial = self._get_initial_sizes()
        fill_len = DEVICE_BUFFER_SIZE + self.page_size * 2
        req = _make_req("long-seq", list(range(fill_len)))
        self._alloc_req_slot(req)

        kv_loc = self._alloc_kv(req, fill_len, logical_only=True)
        self._populate_host_pool(req, fill_len)
        self.coordinator.admit_request_direct(req)

        # Pass fill_len-1 so position fill_len-1 ("newest token") is never
        # randomly selected — its reserved device-buffer slot is only valid
        # after map_last_loc_to_buffer in a real decode step.
        tokens = self._build_topk_tokens(fill_len - 1)
        batch = tokens.unsqueeze(0)
        rpi, sls = self._make_batch_tensors([req], [fill_len])

        for lid in range(LAYER_NUM):
            naive_locs = self.coordinator.naive_load_topk(rpi, sls, batch, lid)
            kernel_locs = self._swap_in_selected_pages(rpi, sls, batch, lid)
            self.assertTrue(torch.all(naive_locs[0, :TOP_K] >= 0))
            self.assertTrue(torch.all(kernel_locs[0, :TOP_K] >= 0))
            # Verify both return correct KV data independently
            self._assert_kv_correct(naive_locs[0], tokens, lid, TOP_K, msg="Naive: ")
            self._assert_kv_correct(kernel_locs[0], tokens, lid, TOP_K, msg="Kernel: ")

        self._cleanup_req(req, kv_loc, logical_only=True)
        self._assert_sizes_restored(initial, "long_seq")

    # ==================================================================
    # Test: Kernel LRU replacement across multiple decode steps
    # ==================================================================
    def test_kernel_lru_replacement(self):
        """Multi-step swap-in: second call hits cached tokens, only
        evicts/loads new misses."""
        initial = self._get_initial_sizes()
        fill_len = DEVICE_BUFFER_SIZE + self.page_size * 2
        req = _make_req("lru-test", list(range(fill_len)))
        self._alloc_req_slot(req)

        kv_loc = self._alloc_kv(req, fill_len, logical_only=True)
        self._populate_host_pool(req, fill_len)
        self.coordinator.admit_request_direct(req)

        rpi, sls = self._make_batch_tensors([req], [fill_len])

        # Step 1: load the first TOP_K positions from host (no newest token —
        # the reserved slot is only valid after map_last_loc_to_buffer which is
        # called during an actual decode step, not modelled here).
        tokens_s1 = torch.arange(TOP_K, dtype=torch.int32, device="cuda")
        locs1 = self._swap_in_selected_pages(
            rpi, sls, tokens_s1.unsqueeze(0), layer_id=0
        )
        self.assertTrue(torch.all(locs1[0, :TOP_K] >= 0))

        # Step 2: half overlap (hit) + half new (miss).
        # Choose new tokens from a range safely below fill_len.
        half = TOP_K // 2
        new_start = TOP_K  # first position not in step-1
        tokens_s2 = torch.cat(
            [
                tokens_s1[:half],  # hits
                torch.arange(
                    new_start, new_start + half, dtype=torch.int32, device="cuda"
                ),  # misses
            ]
        )
        locs2 = self._swap_in_selected_pages(
            rpi, sls, tokens_s2.unsqueeze(0), layer_id=0
        )
        self.assertTrue(torch.all(locs2[0, :TOP_K] >= 0))

        # Verify repeated (hit) tokens still have correct KV data
        self._assert_kv_correct(
            locs2[0], tokens_s2, layer_id=0, count=half, msg="LRU hit: "
        )
        # Also verify new (miss) tokens loaded correctly
        self._assert_kv_correct(
            locs2[0, half:],
            tokens_s2[half:],
            layer_id=0,
            count=half,
            msg="LRU miss: ",
        )

        self._cleanup_req(req, kv_loc, logical_only=True)
        self._assert_sizes_restored(initial, "lru_replacement")

    # ==================================================================
    # Test: Allocator alloc/free lifecycle
    # ==================================================================
    def test_allocator_alloc_free_cycle(self):
        """alloc_extend / alloc_device_buffer / free restores available_size."""
        initial = self._get_initial_sizes()
        device = self.allocator.device
        fill_len = self.page_size * 2

        kv_loc = self.allocator.alloc_extend(
            prefix_lens=torch.tensor([0], dtype=torch.int64, device=device),
            prefix_lens_cpu=torch.tensor([0], dtype=torch.int64),
            seq_lens=torch.tensor([fill_len], dtype=torch.int64, device=device),
            seq_lens_cpu=torch.tensor([fill_len], dtype=torch.int64),
            last_loc=torch.tensor([-1], dtype=torch.int64, device=device),
            extend_num_tokens=fill_len,
        )
        self.assertIsNotNone(kv_loc)
        self.assertEqual(len(kv_loc), fill_len)

        mapping = self.allocator.full_to_hisparse_device_index_mapping[kv_loc]
        self.assertTrue(torch.all(mapping > 0), "Mapping should be non-zero")
        self.assertLess(self.allocator.available_size(), initial[0])

        need_size = min(
            ((fill_len + self.page_size - 1) // self.page_size) * self.page_size,
            DEVICE_BUFFER_SIZE,
        )
        buf_idx, buf_alloc = self.allocator.alloc_device_buffer_mtp(kv_loc, need_size)
        self.assertIsNotNone(buf_idx)
        # MTP variant: buffer-retained tokens keep valid mapping for
        # direct_loc; surplus tokens have mapping cleared to 0.
        buffer_retained = kv_loc[:need_size]
        surplus = kv_loc[need_size:]
        self.assertTrue(torch.equal(buf_alloc, buffer_retained))
        retained_mapping = self.allocator.full_to_hisparse_device_index_mapping[buffer_retained]
        self.assertTrue(torch.all(retained_mapping > 0), "Buffer-retained mapping should be valid")
        if len(surplus) > 0:
            surplus_mapping = self.allocator.full_to_hisparse_device_index_mapping[surplus]
            self.assertTrue(torch.all(surplus_mapping == 0), "Surplus mapping should be cleared")

        self.allocator.free_hisparse_indices(buf_idx)
        self.allocator.logical_attn_allocator.free(kv_loc)
        self._assert_sizes_restored(initial, "alloc_free_cycle")

    def test_allocator_buffer_failure_restores_mapping(self):
        """A failed buffer transition must leave the original KV releasable."""
        initial = self._get_initial_sizes()
        device = self.allocator.device
        fill_len = self.page_size

        kv_loc = self.allocator.alloc_extend(
            prefix_lens=torch.tensor([0], dtype=torch.int64, device=device),
            prefix_lens_cpu=torch.tensor([0], dtype=torch.int64),
            seq_lens=torch.tensor([fill_len], dtype=torch.int64, device=device),
            seq_lens_cpu=torch.tensor([fill_len], dtype=torch.int64),
            last_loc=torch.tensor([-1], dtype=torch.int64, device=device),
            extend_num_tokens=fill_len,
        )
        self.assertIsNotNone(kv_loc)
        mapping_before = self.allocator.full_to_hisparse_device_index_mapping[
            kv_loc
        ].clone()

        with patch.object(
            self.allocator.hisparse_attn_allocator, "alloc", return_value=None
        ):
            with self.assertRaisesRegex(RuntimeError, "alloc_device_buffer"):
                self.allocator.alloc_device_buffer_mtp(kv_loc, fill_len * 2)

        mapping_after = self.allocator.full_to_hisparse_device_index_mapping[kv_loc]
        self.assertTrue(torch.equal(mapping_after, mapping_before))

        self.allocator.free(kv_loc)
        self._assert_sizes_restored(initial, "buffer_failure_mapping")

    def test_alloc_extend_cleared_tail_fails_fast(self):
        """A row whose physical tail mapping is cleared while it still extends
        violates the last-chunk admission invariant (staging / ring recycling
        must only run after prefill completes). alloc_extend must fail fast
        instead of silently emitting the partial-page tokens at slot 0+1 — a
        physical slot the request does not own."""
        if self.page_size == 1:
            self.skipTest("page continuation requires page_size > 1")
        initial = self._get_initial_sizes()
        device = self.allocator.device
        mapping = self.allocator.full_to_hisparse_device_index_mapping

        n = self.page_size + 1  # spills one token into a second page
        kv_loc = self.allocator.alloc_extend(
            prefix_lens=torch.tensor([0], dtype=torch.int64, device=device),
            prefix_lens_cpu=torch.tensor([0], dtype=torch.int64),
            seq_lens=torch.tensor([n], dtype=torch.int64, device=device),
            seq_lens_cpu=torch.tensor([n], dtype=torch.int64),
            last_loc=torch.tensor([-1], dtype=torch.int64, device=device),
            extend_num_tokens=n,
        )
        self.assertIsNotNone(kv_loc)

        # Simulate the invariant violation: the tail token's mapping cleared
        # while the row still extends. Keep the stale physical loc for
        # page-granular cleanup below.
        stale_hi = mapping[kv_loc[n - 1]].clone()
        mapping[kv_loc[n - 1]] = 0

        with self.assertRaisesRegex(AssertionError, "tail mapping"):
            self.allocator.alloc_extend(
                prefix_lens=torch.tensor([n], dtype=torch.int64, device=device),
                prefix_lens_cpu=torch.tensor([n], dtype=torch.int64),
                seq_lens=torch.tensor([n + 1], dtype=torch.int64, device=device),
                seq_lens_cpu=torch.tensor([n + 1], dtype=torch.int64),
                last_loc=kv_loc[n - 1 : n],
                extend_num_tokens=1,
            )

        # Cleanup: free all physical pages of the row (including the stale
        # one) and the logical pages (which also cover the slot claimed by
        # the failed call before its assert).
        hi_all = mapping[kv_loc]
        hi_all = torch.cat([hi_all[hi_all > 0], stale_hi.view(1)])
        pages = torch.unique(hi_all // self.page_size)
        self.allocator.free_hisparse_indices(pages * self.page_size)
        mapping[kv_loc] = 0
        self.allocator.logical_attn_allocator.free(kv_loc)
        self._assert_sizes_restored(initial, "cleared_tail_fails_fast")

    def test_release_backed_partial_page_waits_for_last_logical_reference(self):
        """Backing part of a page must not release live speculative siblings."""
        if self.page_size == 1:
            self.skipTest("partial-page ownership requires page_size > 1")

        initial = self._get_initial_sizes()
        fill_len = self.page_size
        req = _make_req("release-backed-partial", list(range(fill_len)))
        self._alloc_req_slot(req)
        kv_loc = self._alloc_kv(req, fill_len)

        split = self.page_size // 2
        self._clear_mapping(kv_loc[:split])
        mapping = self.allocator.full_to_hisparse_device_index_mapping[kv_loc]
        self.assertTrue(torch.all(mapping[:split] == 0))
        self.assertTrue(torch.all(mapping[split:] > 0))

        self._clear_mapping(kv_loc[split:])
        self.assertTrue(
            torch.all(
                self.allocator.full_to_hisparse_device_index_mapping[kv_loc] == 0
            )
        )
        # Physical pages are NOT freed by clearing the mapping —
        # they are reclaimed in request_finished(). Only mapping is cleared.

        self.allocator.logical_attn_allocator.free(kv_loc)
        self._free_req_slot(req)
        # Cannot assert full size restore — hisparse pages only freed
        # via request_finished(), not release_backed.

    def test_release_backed_frees_fully_unreferenced_pages(self):
        """Mapping is cleared; physical pages reclaimed by request_finished."""
        initial = self._get_initial_sizes()
        fill_len = self.page_size * 2
        req = _make_req("release-backed-full", list(range(fill_len)))
        self._alloc_req_slot(req)
        kv_loc = self._alloc_kv(req, fill_len)

        self._clear_mapping(kv_loc)

        self.assertTrue(
            torch.all(
                self.allocator.full_to_hisparse_device_index_mapping[kv_loc] == 0
            )
        )
        # Physical pages are NOT freed by clearing the mapping —
        # they are reclaimed in request_finished().

        self.allocator.logical_attn_allocator.free(kv_loc)
        self._free_req_slot(req)
        # Cannot assert full size restore here since hisparse pages
        # are only freed via request_finished(), not release_backed.

    def test_release_backed_preserves_device_buffer_pages(self):
        """A cleared global mapping must not free a buffer-owned physical page."""
        initial = self._get_initial_sizes()
        fill_len = self.page_size
        req = _make_req("release-backed-buffer", list(range(fill_len)))
        self._alloc_req_slot(req)
        kv_loc = self._alloc_kv(req, fill_len)
        self.coordinator.alloc_device_buffer(req)

        buffer_loc = self.coordinator.req_to_device_buffer[req.req_pool_idx, 0]
        self.allocator.full_to_hisparse_device_index_mapping[kv_loc[0]] = buffer_loc
        before_release = self.allocator.hisparse_attn_allocator.available_size()

        self._clear_mapping(kv_loc[:1])

        self.assertEqual(
            int(self.allocator.full_to_hisparse_device_index_mapping[kv_loc[0]]), 0
        )
        self.assertEqual(
            self.allocator.hisparse_attn_allocator.available_size(), before_release
        )

        self.coordinator.request_finished(req)
        self.allocator.logical_attn_allocator.free(kv_loc)
        self._free_req_slot(req)
        self._assert_sizes_restored(initial, "release_backed_buffer")

    def test_allocator_page_size_one_alloc_free_cycle(self):
        """alloc() maps logical to hisparse indices for ROCm page_size=1."""
        if self.page_size != 1:
            self.skipTest("page_size=1 alloc path is ROCm-specific")

        initial = self._get_initial_sizes()
        need_size = 16

        kv_loc = self.allocator.alloc(need_size)
        self.assertIsNotNone(kv_loc)
        self.assertEqual(len(kv_loc), need_size)

        mapping = self.allocator.full_to_hisparse_device_index_mapping[kv_loc]
        self.assertTrue(torch.all(mapping > 0), "Mapping should be non-zero")
        self.assertLess(self.allocator.available_size(), initial[0])

        self.allocator.free(kv_loc)
        mapping_after = self.allocator.full_to_hisparse_device_index_mapping[kv_loc]
        self.assertTrue(torch.all(mapping_after == 0), "Mapping should be cleared")
        self._assert_sizes_restored(initial, "page_size_one_alloc_free_cycle")

    def test_decode_remap_frees_stale_page_size_one_mapping(self):
        """map_last_loc_to_buffer frees the temporary alloc() hisparse slot."""
        if self.page_size != 1:
            self.skipTest("page_size=1 decode remap path is ROCm-specific")

        initial = self._get_initial_sizes()
        device = self.allocator.device
        fill_len = 2
        req = _make_req("decode-remap", list(range(fill_len)))
        self._alloc_req_slot(req)

        kv_loc = self._alloc_kv(req, fill_len)
        self.coordinator.alloc_device_buffer(req)
        self.coordinator._skip_first_backup[req.req_pool_idx] = True

        out_loc = self.allocator.alloc(1)
        self.assertIsNotNone(out_loc)
        stale_loc = self.allocator.full_to_hisparse_device_index_mapping[
            out_loc
        ].clone()
        self.assertTrue(torch.all(stale_loc > 0), "Temporary mapping should exist")

        seq_len = fill_len + 1
        self.req_to_token_pool.write((req.req_pool_idx, fill_len), out_loc)
        req.kv.kv_allocated_len = seq_len
        req.kv_committed_len = seq_len

        self.coordinator.map_last_loc_to_buffer(
            seq_lens=torch.tensor([seq_len], dtype=torch.int64, device=device),
            out_cache_loc=out_loc,
            req_pool_indices=torch.tensor(
                [req.req_pool_idx], dtype=torch.int64, device=device
            ),
            seq_lens_cpu=torch.tensor([seq_len], dtype=torch.int64),
            req_pool_indices_cpu=torch.tensor([req.req_pool_idx], dtype=torch.int64),
        )

        remapped_loc = self.allocator.full_to_hisparse_device_index_mapping[out_loc]
        self.assertTrue(torch.all(remapped_loc > 0), "Remapped loc should exist")
        self.assertFalse(
            torch.equal(stale_loc, remapped_loc),
            "Decode loc should move from temporary mapping to device buffer",
        )
        self.assertEqual(
            self.allocator.hisparse_attn_allocator.available_size(),
            initial[1] - seq_len,
        )

        self.coordinator.request_finished(req)
        self.allocator.logical_attn_allocator.free(torch.cat([kv_loc, out_loc]))
        self._free_req_slot(req)
        self._assert_sizes_restored(initial, "decode_remap")

    # ==================================================================
    # Test: HiSparse<->MTP hybrid (Phase 2 batch-level gating)
    # ==================================================================
    def test_hybrid_mode_gating_predicates(self):
        """should_admit_new / attaches_to_batch follow the controller mode; a
        non-hybrid coordinator (no controller) always runs the offloaded path."""
        from sglang.srt.speculative.hybrid_mode_controller import (
            DecodeMode,
            HybridModeController,
        )

        coord = self.coordinator
        # Non-hybrid: no controller -> always admit + always attach.
        self.assertIsNone(coord.hybrid_controller)
        self.assertTrue(coord.should_admit_new())
        self.assertTrue(coord.attaches_to_batch())

        controller = HybridModeController(
            coord,
            self.allocator,
            usage_threshold_up=0.6,
            usage_threshold_down=0.3,
            min_bsz_for_hisparse=8,
            max_bsz_for_mtp=4,
            cooldown_steps=0,
        )
        coord.hybrid_controller = controller
        try:
            # mode -> (admits_offloaded, attaches_coordinator)
            expected = {
                DecodeMode.MTP: (False, False),
                DecodeMode.PENDING_OFFLOAD: (True, False),
                DecodeMode.HISPARSE: (True, True),
                DecodeMode.PENDING_RESTORE: (False, True),
            }
            for mode, (admit, attach) in expected.items():
                controller.current_mode = mode
                self.assertEqual(coord.should_admit_new(), admit, mode.name)
                self.assertEqual(coord.attaches_to_batch(), attach, mode.name)
        finally:
            # Shared coordinator: do not leak the controller into other tests.
            coord.hybrid_controller = None

    def test_resident_request_finished_no_leak(self):
        """A pure-MTP resident request -- allocated via the combined allocator
        and never admitted into HiSparse -- frees with no leak and no
        double-free: request_finished reclaims the mapped physical pages and
        clears the mapping, so the subsequent release_kv_cache free finds
        nothing left on the hisparse side."""
        initial = self._get_initial_sizes()
        fill_len = self.page_size * 2
        req = _make_req("resident-req", list(range(fill_len)))
        self._alloc_req_slot(req)

        # alloc_extend (not spec_logical) = combined logical + physical + mapping:
        # exactly the layout of a prefilled-but-never-admitted resident request.
        kv_loc = self._alloc_kv(req, fill_len)
        self.assertEqual(
            int(self.coordinator.req_device_buffer_size[req.req_pool_idx]),
            0,
            "resident request must have no device buffer",
        )
        compressed = self.coordinator.mem_pool_device.translate_loc_from_full_to_compressed(
            kv_loc
        )
        mapped = self.allocator.full_to_hisparse_device_index_mapping[compressed]
        self.assertTrue(
            torch.all(mapped > 0), "resident mapping must be valid before free"
        )

        # Mirror the scheduler free order: request_finished then release_kv_cache.
        self.coordinator.request_finished(req)
        self.allocator.free(kv_loc)
        self._free_req_slot(req)
        self._assert_sizes_restored(initial, "resident request_finished")

    def test_offload_running_request_migrates_resident_req(self):
        """Mid-decode offload migration: a device-resident request carrying an
        uncommitted spec look-ahead tail gains a device buffer + staging ring,
        backs up ONLY the committed prefix, ring-maps the tail (whose resident
        physical slots were released), and frees with no leak."""
        initial = self._get_initial_sizes()
        device = self.allocator.device
        p = self.page_size
        committed = 2 * p
        allocated = 3 * p

        req = _make_req("offload-mig", list(range(committed)))
        self._alloc_req_slot(req)

        # Resident layout: combined alloc for the committed prefix ...
        kv_loc = self._alloc_kv(req, committed)
        self._write_device_patterns(kv_loc, committed)
        # ... plus the uncommitted look-ahead tail [committed, allocated) that a
        # running spec-decode request always carries.
        tail = self.allocator.alloc_extend(
            prefix_lens=torch.tensor([committed], dtype=torch.int64, device=device),
            prefix_lens_cpu=torch.tensor([committed], dtype=torch.int64),
            seq_lens=torch.tensor([allocated], dtype=torch.int64, device=device),
            seq_lens_cpu=torch.tensor([allocated], dtype=torch.int64),
            last_loc=kv_loc[-1:],
            extend_num_tokens=allocated - committed,
        )
        self.assertIsNotNone(tail, "look-ahead tail alloc failed")
        self.req_to_token_pool.write(
            (req.req_pool_idx, slice(committed, allocated)), tail
        )
        req.kv.kv_allocated_len = allocated
        req.kv_committed_len = committed

        self.coordinator.offload_running_request(req)

        rid = req.req_pool_idx
        self.assertGreater(
            int(self.coordinator.req_device_buffer_size[rid]),
            0,
            "offload must carve a device buffer",
        )
        self.assertTrue(
            self.coordinator.req_spec_ring_active[rid], "offload must alloc the ring"
        )
        # Only the committed prefix is staged, and the backup high-water mark
        # lands on it so backup_committed_tokens resumes exactly there.
        self.assertEqual(req.hisparse_last_backed_len, committed)
        self.assertEqual(
            int(self.coordinator.req_to_host_pool_allocated_len[rid]), committed
        )
        # The ring window is anchored at the migration point.
        self.assertEqual(req.hisparse_ring_start, committed)
        # Tail positions resolve through the ring, not stale resident slots.
        tail_logical = self.req_to_token_pool.req_to_token[
            rid, committed:allocated
        ].to(torch.int64)
        tail_compressed = (
            self.coordinator.mem_pool_device.translate_loc_from_full_to_compressed(
                tail_logical
            )
        )
        tail_mapped = self.allocator.full_to_hisparse_device_index_mapping[
            tail_compressed
        ]
        ring = self.coordinator.req_to_spec_ring[rid]
        self.assertTrue(
            bool(torch.isin(tail_mapped, ring).all()),
            "look-ahead tail must map into the spec staging ring",
        )

        self._collect_ready_reqs_blocking()
        self._cleanup_req(req, torch.cat([kv_loc, tail]), logical_only=True)
        self._assert_sizes_restored(initial, "offload_running_request")

    def test_offload_then_restore_round_trip(self):
        """resident -> offload -> restore: the committed KV survives the host
        round trip, every allocated position lands back on its own dedicated
        physical slot, and all HiSparse per-request state is torn down."""
        initial = self._get_initial_sizes()
        device = self.allocator.device
        p = self.page_size
        committed = 2 * p
        allocated = 3 * p

        req = _make_req("roundtrip", list(range(committed)))
        self._alloc_req_slot(req)
        kv_loc = self._alloc_kv(req, committed)
        self._write_device_patterns(kv_loc, committed)
        tail = self.allocator.alloc_extend(
            prefix_lens=torch.tensor([committed], dtype=torch.int64, device=device),
            prefix_lens_cpu=torch.tensor([committed], dtype=torch.int64),
            seq_lens=torch.tensor([allocated], dtype=torch.int64, device=device),
            seq_lens_cpu=torch.tensor([allocated], dtype=torch.int64),
            last_loc=kv_loc[-1:],
            extend_num_tokens=allocated - committed,
        )
        self.assertIsNotNone(tail, "look-ahead tail alloc failed")
        self.req_to_token_pool.write(
            (req.req_pool_idx, slice(committed, allocated)), tail
        )
        req.kv.kv_allocated_len = allocated
        req.kv_committed_len = committed
        rid = req.req_pool_idx

        # resident -> offloaded
        self.coordinator.offload_running_request(req)
        self._collect_ready_reqs_blocking()
        self.assertGreater(int(self.coordinator.req_device_buffer_size[rid]), 0)

        # offloaded -> resident
        self.assertTrue(
            self.coordinator.restore_running_request(req), "restore must succeed"
        )

        # HiSparse per-request state fully torn down.
        self.assertEqual(int(self.coordinator.req_device_buffer_size[rid]), 0)
        self.assertFalse(self.coordinator.req_spec_ring_active[rid])
        self.assertEqual(int(self.coordinator.req_to_host_pool_allocated_len[rid]), 0)
        self.assertIsNone(req.hisparse_last_backed_len)
        self.assertIsNone(req.hisparse_ring_start)

        # Every allocated position maps to its own dedicated physical slot.
        logical = self.req_to_token_pool.req_to_token[rid, :allocated].to(torch.int64)
        compressed = (
            self.coordinator.mem_pool_device.translate_loc_from_full_to_compressed(
                logical
            )
        )
        mapped = self.allocator.full_to_hisparse_device_index_mapping[compressed]
        self.assertTrue(bool((mapped > 0).all()), "resident mapping must be valid")
        self.assertEqual(
            int(torch.unique(mapped).numel()),
            allocated,
            "restored slots must be distinct (one per position)",
        )

        # The committed KV came back intact through the host round trip.
        for i in (0, committed // 2, committed - 1):
            for lid in range(LAYER_NUM):
                got = self.device_pool.kv_buffer[lid][mapped[i]]
                self.assertTrue(
                    bool((got == self._kv_pattern(lid, i)).all()),
                    f"KV mismatch after restore at layer {lid} pos {i}",
                )

        self.coordinator.request_finished(req)
        self.allocator.logical_attn_allocator.free(torch.cat([kv_loc, tail]))
        self._free_req_slot(req)
        self._assert_sizes_restored(initial, "offload_restore_round_trip")

    def test_resident_verify_swap_in_is_pure_gather(self):
        """Single-graph scheme: a device-resident row through the verify
        swap-in kernel is a pure gather with zero state change.

        A resident request has a valid mapping for every token, no device
        buffer and identity tables at -1, so every top-k entry must resolve in
        the kernel's first-level direct_loc pass (output == mapping[logical])
        and the all-resolved early-exit must skip the buffer scan / LRU
        write-back / miss DMA entirely. Deleting this case would let two
        regressions pass silently: the early-exit degrading (LRU reordered or
        mapping invalidated for resident rows) and the direct_loc pass
        misresolving, both of which surface only as a quiet accept-rate drop
        in the hybrid's resident mode."""
        device = self.allocator.device
        fill_len = 2 * self.page_size
        req = _make_req("resident-gather", list(range(fill_len)))
        self._alloc_req_slot(req)
        # Combined alloc: the resident layout -- every logical id mapped.
        kv_loc = self._alloc_kv(req, fill_len)
        rid = req.req_pool_idx

        mapping = self.allocator.full_to_hisparse_device_index_mapping
        lru_before = self.coordinator.lru_slots[0, rid, :].clone()
        mapping_before = mapping[kv_loc.to(torch.int64)].clone()

        # Verify top-k input carries LOGICAL kv ids (fused-topk output), padded
        # with -1 like a real underfull selection.
        n = min(fill_len, TOP_K)
        sel = torch.randperm(fill_len, device=device)[:n]
        tokens = kv_loc[sel].to(torch.int32)
        if n < TOP_K:
            tokens = torch.cat(
                [tokens, torch.full((TOP_K - n,), -1, dtype=torch.int32, device=device)]
            )
        rpi, sls = self._make_batch_tensors([req], [fill_len])
        self.coordinator.num_real_reqs[0] = 1

        out = self.coordinator.swap_in_verify_pages(
            req_pool_indices=rpi,
            seq_lens=sls,
            logical_top_k=tokens.unsqueeze(0),
            layer_id=0,
            num_positions=1,
        )

        valid = tokens >= 0
        expect = mapping[tokens[valid].to(torch.int64)].to(torch.int32)
        self.assertTrue(
            torch.equal(out[0][valid].cpu(), expect.cpu()),
            "resident rows must resolve to mapping[logical] exactly",
        )
        if bool((~valid).any()):
            self.assertTrue(
                bool((out[0][~valid] == -1).all()), "padding must stay -1"
            )
        # Early-exit contract: zero per-request state change.
        self.assertTrue(
            torch.equal(self.coordinator.lru_slots[0, rid, :], lru_before),
            "all-resolved position must not touch the LRU order",
        )
        self.assertTrue(
            bool(
                (self.coordinator.req_device_buffer_logical_locs[0, rid, :] == -1).all()
            ),
            "buffer identity table must stay untouched",
        )
        self.assertTrue(
            torch.equal(mapping[kv_loc.to(torch.int64)], mapping_before),
            "mapping must not be invalidated for resident rows",
        )

        self._cleanup_req(req, kv_loc)

    def test_restore_drains_inflight_staging(self):
        """Restore must resolve the request's in-flight admission staging DMA.

        Bug (review a145b5111 #2): offload queues an async device->host staging
        action on write_staging_stream; restore_running_request only waited on
        the forward stream and the DECODE backup stream, then read the host KV
        and freed the host slots. Restoring before the staging ack (offload
        during cooldown, or a request admitted right before the switch back)
        could therefore read half-written host KV and free host slots the DMA
        was still writing -- and the stale action stayed queued in
        ack_staging_queue holding its deferred surplus pages.

        The round-trip test masked this by calling _collect_ready_reqs_blocking
        first; this one restores immediately after offload, with the staging
        action still queued.
        """
        initial = self._get_initial_sizes()
        p = self.page_size
        committed = 2 * p

        req = _make_req("restore-inflight", list(range(committed)))
        self._alloc_req_slot(req)
        kv_loc = self._alloc_kv(req, committed)
        self._write_device_patterns(kv_loc, committed)
        rid = req.req_pool_idx

        self.coordinator.offload_running_request(req)
        self.assertTrue(
            any(a.req is req for a in self.coordinator.ack_staging_queue),
            "precondition: staging action must still be in flight",
        )

        # No _collect_ready_reqs_blocking here: restore while the DMA is queued.
        self.assertTrue(self.coordinator.restore_running_request(req))

        self.assertFalse(
            any(a.req is req for a in self.coordinator.ack_staging_queue),
            "restore must drain the request's staging action before touching "
            "the host pool it writes to",
        )
        self.assertFalse(req.hisparse_staging)
        # Drain waited for the DMA, so the restored KV is the fully staged copy.
        logical = self.req_to_token_pool.req_to_token[rid, :committed].to(torch.int64)
        compressed = (
            self.coordinator.mem_pool_device.translate_loc_from_full_to_compressed(
                logical
            )
        )
        mapped = self.allocator.full_to_hisparse_device_index_mapping[compressed]
        for i in (0, committed - 1):
            got = self.device_pool.kv_buffer[0][mapped[i]]
            self.assertTrue(
                bool((got == self._kv_pattern(0, i)).all()),
                f"KV mismatch after inflight restore at pos {i}",
            )

        self.coordinator.request_finished(req)
        self.allocator.logical_attn_allocator.free(kv_loc)
        self._free_req_slot(req)
        self._assert_sizes_restored(initial, "restore_drains_inflight_staging")

    def test_verify_swap_in_ignores_dp_padding_rows(self):
        """DP padding rows must never be executed as real requests.

        Bug (review c0b3d6c43 #1): _prepare_eager_forward_batch runs
        prepare_mlp_sync_batch first, which rewrites forward_batch.batch_size to
        the DP-PADDED size, and only then fills num_real_reqs from it. The
        kernel's `bid >= num_real_reqs[0]` guard therefore stops filtering the
        dummy blocks. Padding fills req_pool_indices with 0
        (_pad_tensor_to_size default), so on an uneven-DP eager verify the dummy
        blocks run against request slot 0 -- a DIFFERENT live request's LRU,
        device buffer and DMA targets.

        This drives the kernel directly with a padded batch and asserts slot 0's
        state is untouched when num_real_reqs is the true count, and shows what
        the padded count would let the dummies do.
        """
        device = self.allocator.device
        fill_len = 4 * self.page_size

        # Victim occupies slot 0 and is fully offloaded, so it owns real buffer
        # and LRU state that dummy blocks (req_pool_indices == 0) would touch.
        victim = _make_req("dp-victim", list(range(fill_len)))
        self._alloc_req_slot(victim)
        self.assertNotEqual(
            victim.req_pool_idx,
            0,
            "ReqToTokenPool reserves index 0 as the padding sentinel row "
            "(memory_pool.py: free_slots starts at 1), so no live request can "
            "own it -- that is what makes zero-padded dummy rows harmless",
        )
        v_kv = self._alloc_kv(victim, fill_len)
        self._write_device_patterns(v_kv, fill_len)
        self.coordinator.admit_request_into_staging(victim)
        self._collect_ready_reqs_blocking()

        # The one real request of this rank's batch, in a different slot.
        real = _make_req("dp-real", list(range(fill_len)))
        self._alloc_req_slot(real)
        self.assertNotEqual(real.req_pool_idx, 0)
        r_kv = self._alloc_kv(real, fill_len)
        self._write_device_patterns(r_kv, fill_len)
        self.coordinator.admit_request_into_staging(real)
        self._collect_ready_reqs_blocking()

        # Batch of 1 real request padded to 3 rows exactly as
        # _pad_tensor_to_size does: zeros for req_pool_indices / seq_lens, and a
        # zero-filled top-k block per padded row.
        padded_bs = 3
        rpi = torch.zeros(padded_bs, dtype=torch.int64, device=device)
        rpi[0] = real.req_pool_idx
        sls = torch.zeros(padded_bs, dtype=torch.int32, device=device)
        sls[0] = fill_len
        topk = torch.zeros((padded_bs, TOP_K), dtype=torch.int32, device=device)
        topk[0] = self._build_topk_tokens(fill_len)

        def snapshot():
            return (
                self.coordinator.lru_slots[:, 0, :].clone(),
                self.coordinator.req_device_buffer_token_locs[:, 0, :].clone(),
                self.coordinator.req_device_buffer_logical_locs[:, 0, :].clone(),
            )

        def run(num_real):
            self.coordinator.num_real_reqs[0] = num_real
            self.coordinator.swap_in_verify_pages(
                req_pool_indices=rpi,
                seq_lens=sls,
                logical_top_k=topk,
                layer_id=0,
                num_positions=1,
            )
            torch.cuda.synchronize()

        base = snapshot()
        run(1)  # correct count: dummies filtered, slot 0 must be untouched
        after_correct = snapshot()
        for b, a, name in zip(base, after_correct, ("lru", "buf_locs", "buf_ids")):
            self.assertTrue(
                torch.equal(b, a),
                f"slot 0 {name} changed even with the correct num_real_reqs",
            )

        # The padded count DOES let the dummy rows run (measured: they churn the
        # sentinel row's lru and buf_ids). That is contained only because index 0
        # is reserved; assert the live request's own state is never collateral.
        v_lru = self.coordinator.lru_slots[:, victim.req_pool_idx, :].clone()
        v_ids = self.coordinator.req_device_buffer_logical_locs[
            :, victim.req_pool_idx, :
        ].clone()
        run(padded_bs)
        self.assertTrue(
            torch.equal(v_lru, self.coordinator.lru_slots[:, victim.req_pool_idx, :]),
            "dummy rows must never touch a live request's LRU",
        )
        self.assertTrue(
            torch.equal(
                v_ids,
                self.coordinator.req_device_buffer_logical_locs[
                    :, victim.req_pool_idx, :
                ],
            ),
            "dummy rows must never touch a live request's buffer identities",
        )

        for req, kv in ((victim, v_kv), (real, r_kv)):
            self.coordinator.request_finished(req)
            self.allocator.logical_attn_allocator.free(kv)
            self._free_req_slot(req)

    def test_offloaded_decode_memcheck_uses_logical_capacity(self):
        """Offloaded MTP's decode memcheck must match what it will allocate.

        Bug (review c0b3d6c43 #2): eagle_prepare_for_decode allocates the next
        speculative slots under spec_logical_alloc() -- LOGICAL ids only, the
        physical locations come from each request's fixed staging ring. But
        check_decode_mem gated the step on available_size() =
        min(logical, physical). Once the fixed buffer+ring footprints fill the
        physical pool (by design at the concurrency cap), the check reports OOM
        for an allocation that would succeed, forcing retractions; with one
        request left, retract_decode aborts it with HTTP 500.

        Squeezes the physical pool to below one page while logical stays ample:
        an offloaded batch must still pass the memcheck, while a resident batch
        (combined allocator: logical AND physical per token) must still fail.
        """
        from sglang.srt import runtime_context as rc
        from sglang.srt.managers.schedule_batch import ScheduleBatch
        from sglang.srt.server_args import ServerArgs
        from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

        page = self.page_size
        req = _make_req("memcheck", list(range(page)))
        # On a page boundary, so any reserve > 0 needs one fresh page.
        req.kv.kv_allocated_len = page
        req.kv_committed_len = page

        phys = self.allocator.hisparse_attn_allocator
        holder = phys.alloc(phys.available_size() // page * page)
        self.assertIsNotNone(holder)
        self.assertLess(phys.available_size(), page, "physical must be squeezed")

        def batch(offloaded):
            return ScheduleBatch(
                reqs=[req],
                spec_algorithm=SpeculativeAlgorithm.from_string("EAGLE"),
                tree_cache=None,
                token_to_kv_pool_allocator=self.allocator,
                hisparse_coordinator=self.coordinator if offloaded else None,
                hisparse_resident=not offloaded,
            )

        rc.reset_context()
        rc.get_context().set_server_args(
            ServerArgs(
                model_path="dummy",
                speculative_algorithm="EAGLE",
                speculative_num_steps=1,
                speculative_eagle_topk=1,
                speculative_num_draft_tokens=2,
            )
        )
        try:
            self.assertTrue(
                batch(offloaded=True).check_decode_mem(),
                "offloaded step allocates logical-only; a full physical pool "
                "must not fail the check",
            )
            self.assertFalse(
                batch(offloaded=False).check_decode_mem(),
                "resident step allocates logical AND physical; a full physical "
                "pool must fail the check",
            )
        finally:
            rc.reset_context()
            phys.free(holder)

    def test_hybrid_offload_proceeds_when_surplus_funds_migration(self):
        """The headroom gate must not reject migrations the reclaim path funds.

        Bug (review c0b3d6c43 #3, a regression from the previous review round's
        fix): _offload_headroom_ok demanded free physical space for every
        request's ring + buffer shortfall UP FRONT. But a LONG request funds its
        own migration: alloc_device_buffer keeps the buffer out of pages the
        request already owns and defer-frees the surplus, which
        _reclaim_deferred_staging_pages hands to the ring allocation
        (test_admit_long_prompt_reclaims_deferred_pages_for_ring proves it with
        free space smaller than the ring). The pessimistic gate returned False
        before the first migration, and since refusing changes nothing, every
        later step refused again -- pinning the hybrid to resident exactly when
        memory pressure is highest, until retract/OOM.

        Runs the REAL coordinator under the controller with the pool nearly
        full: the switch must proceed, not postpone.
        """
        from sglang.srt.speculative.hybrid_mode_controller import (
            DecodeMode,
            HybridModeController,
        )

        if self.page_size == 1:
            self.skipTest("deferred-page accounting requires page_size > 1")
        R = 2 * self.page_size
        self._enable_spec_ring(R)

        # One long resident request holding all but one page of the physical
        # pool -- the admit-time reclaim path is what must fund ring+slack.
        fill_len = SIZE - self.page_size
        req = _make_req("gate-long", list(range(fill_len)))
        self._alloc_req_slot(req)
        self._alloc_kv(req, fill_len)
        req.kv_committed_len = fill_len
        self.assertLess(
            self.allocator.hisparse_attn_allocator.available_size(),
            R,
            "precondition: free physical must be below the ring size",
        )

        ctl = HybridModeController(
            self.coordinator,
            self.allocator,
            usage_threshold_up=0.5,
            usage_threshold_down=0.25,
            min_bsz_for_hisparse=1,
            max_bsz_for_mtp=0,
            cooldown_steps=0,
        )
        mode = ctl.on_step(1, [req])

        self.assertEqual(
            mode,
            DecodeMode.HISPARSE,
            "the long request's own surplus funds the migration; the gate must "
            "let it proceed instead of pinning resident under pressure",
        )
        self.assertGreater(int(self.coordinator.req_device_buffer_size[req.req_pool_idx]), 0)
        self.assertTrue(self.coordinator.req_spec_ring_active[req.req_pool_idx])

        self._collect_ready_reqs_blocking()
        initial_after = None  # sizes checked via full teardown below
        self.coordinator.request_finished(req)
        all_locs = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, :fill_len
        ].to(torch.int64)
        self.allocator.logical_attn_allocator.free(all_locs)
        self._free_req_slot(req)

    def test_hybrid_controller_projection_breaks_pingpong(self):
        """The restore decision must use PROJECTED resident usage, not current.

        Offloading is what drops device usage, so comparing the post-offload
        usage against the down-threshold flip-flops forever (observed live: 72
        offload + 72 restore transitions in seconds). Projecting what residency
        would cost keeps the batch offloaded while the load is genuinely heavy,
        and still restores once it really drains.
        """
        from sglang.srt.speculative.hybrid_mode_controller import (
            DecodeMode,
            HybridModeController,
        )

        class _Kv:
            def __init__(self, a):
                self.kv_allocated_len = a

        class _Req:
            def __init__(self, rid, alloc, idx):
                self.rid, self.kv, self.req_pool_idx = rid, _Kv(alloc), idx

            def finished(self):
                return False

        class _Alloc:
            def __init__(self, size, avail):
                self.size, self._avail = size, avail

            def available_size(self):
                return self._avail

        class _Coord:
            def __init__(self, size, used, buf_sizes):
                self.token_to_kv_pool_allocator = type(
                    "T", (), {"hisparse_attn_allocator": _Alloc(size, size - used)}
                )()
                self.spec_ring_capacity = 256
                self.req_device_buffer_size = buf_sizes
                self.restored = []
                self._size = size
                self._used = used

            def get_token_stats(self):
                return type(
                    "S", (), {"device_token_usage": self._used / self._size}
                )()

            def offload_running_request(self, req):
                pass

            def restore_running_request(self, req):
                self.restored.append(req.rid)
                return True

        def _controller(coord):
            return HybridModeController(
                coord,
                None,
                usage_threshold_up=0.5,
                usage_threshold_down=0.25,
                min_bsz_for_hisparse=2,
                max_bsz_for_mtp=1,
                cooldown_steps=0,
            )

        # Heavy: 8 offloaded requests of 30k. Current (offloaded) usage 0.147 is
        # under the down-threshold, but residency would need > 50% of the pool.
        heavy = [_Req(f"r{i}", 30000, i) for i in range(8)]
        coord = _Coord(456000, 67000, [8192] * 8)
        ctl = _controller(coord)
        self.assertGreater(ctl._projected_resident_usage(heavy), 0.5)
        ctl.current_mode = DecodeMode.HISPARSE
        self.assertEqual(
            ctl.on_step(8, heavy),
            DecodeMode.HISPARSE,
            "must not restore into an immediate re-offload",
        )
        self.assertEqual(coord.restored, [], "no restore should even be attempted")

        # Genuine drain: two small requests now fit comfortably.
        light = [_Req("a", 5000, 0), _Req("b", 5000, 1)]
        coord2 = _Coord(456000, 17000, [8192, 8192])
        ctl2 = _controller(coord2)
        self.assertLess(ctl2._projected_resident_usage(light), 0.25)
        ctl2.current_mode = DecodeMode.HISPARSE
        self.assertEqual(ctl2.on_step(2, light), DecodeMode.MTP)
        self.assertEqual(coord2.restored, ["a", "b"])

    def test_hybrid_offload_postponed_without_headroom(self):
        """The MTP->HiSparse switch must be gated on transient headroom.

        Bug (review a145b5111 #3): the controller assumed offload is always
        net-freeing and migrated unconditionally. A request SHORTER than the
        fixed buffer+ring footprint net-allocates, and defer-freed surplus is
        unavailable until the staging DMA acks -- so with the pool nearly full,
        alloc_device_buffer could raise mid-migration, escaping the controller
        with some requests offloaded and some not (a mixed batch) and killing
        the scheduler. The switch must be postponed instead, and retried once
        headroom exists.
        """
        from sglang.srt.speculative.hybrid_mode_controller import (
            DecodeMode,
            HybridModeController,
        )

        class _Kv:
            def __init__(self, a):
                self.kv_allocated_len = a

        class _Req:
            def __init__(self, rid, alloc, idx):
                self.rid, self.kv, self.req_pool_idx = rid, _Kv(alloc), idx

            def finished(self):
                return False

        class _Alloc:
            def __init__(self, avail):
                self.size, self._avail = 456000, avail

            def available_size(self):
                return self._avail

        class _Coord:
            def __init__(self, avail):
                self.token_to_kv_pool_allocator = type(
                    "T", (), {"hisparse_attn_allocator": _Alloc(avail)}
                )()
                self.mem_pool_device = type("P", (), {"page_size": 64})()
                self.spec_ring_capacity = 256
                self.padded_buffer_size = 8256
                self.req_device_buffer_size = [0, 0]
                self.offloaded = []
                self.reclaimed = 0

            def get_token_stats(self):
                # High pressure, so the up-threshold is crossed and the switch
                # is attempted; only the headroom gate can stop it.
                return type("S", (), {"device_token_usage": 0.9})()

            def offload_running_request(self, req):
                self.offloaded.append(req.rid)

            def _reclaim_deferred_staging_pages(self):
                self.reclaimed += 1
                return False

        def _controller(coord):
            return HybridModeController(
                coord,
                None,
                usage_threshold_up=0.5,
                usage_threshold_down=0.25,
                min_bsz_for_hisparse=2,
                max_bsz_for_mtp=1,
                cooldown_steps=0,
            )

        # Two SHORT requests (1k tokens each, far below the 8256+256 footprint)
        # and almost no free physical space: the switch must be postponed.
        short = [_Req("s0", 1000, 0), _Req("s1", 1000, 1)]
        coord = _Coord(avail=1000)
        ctl = _controller(coord)
        self.assertEqual(
            ctl.on_step(2, short),
            DecodeMode.MTP,
            "must postpone the switch when transient headroom is insufficient",
        )
        self.assertEqual(coord.offloaded, [], "no partial migration allowed")
        self.assertGreater(coord.reclaimed, 0, "must try reclaiming first")

        # Same requests with ample headroom: the switch proceeds.
        coord2 = _Coord(avail=456000)
        ctl2 = _controller(coord2)
        self.assertEqual(ctl2.on_step(2, short), DecodeMode.HISPARSE)
        self.assertEqual(coord2.offloaded, ["s0", "s1"])

    # ==================================================================
    # Test: Staging (PD Colocate) path
    # ==================================================================
    def test_request_lifecycle_staging_path(self):
        """prefill -> staging DMA -> collect_ready -> swap-in -> finish."""
        initial = self._get_initial_sizes()
        fill_len = self.page_size
        req = _make_req("staging-req", list(range(fill_len)))
        self._alloc_req_slot(req)

        kv_loc = self._alloc_kv(req, fill_len)
        self._write_device_patterns(kv_loc, fill_len)

        self.coordinator.admit_request_into_staging(req)
        self.assertTrue(req.hisparse_staging)
        # The logical-id -> position inverse map is MTP-only; the plain
        # coordinator does not allocate it.
        self.assertIsNone(self.coordinator.full_to_token_position)
        # Plain-mode admission is DMA-only (upstream semantics): the buffer is
        # carved at staging ack, so the mapping is still fully valid here.
        mapping_after = self.allocator.full_to_hisparse_device_index_mapping[kv_loc]
        self.assertTrue(
            torch.all(mapping_after > 0),
            "Mapping must stay untouched until collect_ready_reqs",
        )

        torch.cuda.synchronize()
        ready = self.coordinator.collect_ready_reqs()
        self.assertEqual(len(ready), 1)
        self.assertFalse(req.hisparse_staging)
        self.assertTrue(self.coordinator._skip_first_backup[req.req_pool_idx])
        # Upstream ack-time alloc isolates the buffer from outside addressing:
        # the whole request's mapping is cleared (the decode swap-in kernel
        # consumes only top_k_device_locs, never the mapping).
        mapping_after = self.allocator.full_to_hisparse_device_index_mapping[kv_loc]
        self.assertTrue(
            torch.all(mapping_after == 0),
            "Mapping should be cleared after the buffer carve",
        )
        for lid in range(LAYER_NUM):
            self.assertTrue(
                torch.equal(
                    self.coordinator.req_device_buffer_tokens[
                        lid, req.req_pool_idx, :fill_len
                    ],
                    torch.arange(fill_len, dtype=torch.int32, device="cuda"),
                )
            )

        tokens = self._build_topk_tokens(fill_len)
        batch = tokens.unsqueeze(0)
        rpi, sls = self._make_batch_tensors([req], [fill_len])

        locs = self._swap_in_selected_pages(rpi, sls, batch, layer_id=0)
        valid_n = min(fill_len, TOP_K)
        self.assertTrue(torch.all(locs[0, :valid_n] >= 0))
        self._assert_kv_correct(
            locs[0], tokens, layer_id=0, count=valid_n, msg="Staging: "
        )
        self._assert_matches_naive(rpi, sls, batch, locs, layer_id=0, msg="Staging: ")

        self._cleanup_req(req, kv_loc)
        self._assert_sizes_restored(initial, "staging_path")

    def test_mtp_long_staging_swaps_buffer_and_host_logical_tokens(self):
        """Fused logical top-k remains correct beyond the device buffer."""
        initial = self._get_initial_sizes()
        fill_len = DEVICE_BUFFER_SIZE + self.page_size * 2
        req = _make_req("mtp-long-staging", list(range(fill_len)))
        self._alloc_req_slot(req)

        kv_loc = self._alloc_kv(req, fill_len)
        self._write_device_patterns(kv_loc, fill_len)
        self.coordinator.admit_request_into_staging(req)
        self._collect_ready_reqs_blocking()

        required_positions = torch.tensor(
            [0, DEVICE_BUFFER_SIZE - 1, DEVICE_BUFFER_SIZE, fill_len - 1],
            dtype=torch.int64,
            device="cuda",
        )
        candidate_positions = torch.arange(fill_len, device="cuda")
        candidate_positions = candidate_positions[
            ~torch.isin(candidate_positions, required_positions)
        ]
        positions = torch.cat(
            [
                required_positions,
                candidate_positions[
                    torch.randperm(candidate_positions.numel(), device="cuda")[
                        : TOP_K - required_positions.numel()
                    ]
                ],
            ]
        )
        logical_top_k = kv_loc[positions].to(torch.int32).unsqueeze(0)
        rpi, sls = self._make_batch_tensors([req], [fill_len])

        for layer_id in range(LAYER_NUM):
            locs = self._swap_in_selected_logical_pages(
                rpi, sls, logical_top_k, layer_id
            )
            self.assertTrue(torch.all(locs >= 0))
            for i, position in enumerate(positions.tolist()):
                expected = self._kv_pattern(layer_id, position)
                actual = self.device_pool.kv_buffer[layer_id][locs[0, i].long()]
                self.assertTrue(
                    torch.allclose(
                        actual.float(),
                        torch.full_like(actual.float(), expected),
                        atol=1e-2,
                    ),
                    f"layer {layer_id}, position {position}: swapped KV mismatch",
                )

        self._cleanup_req(req, kv_loc)
        self._assert_sizes_restored(initial, "mtp_long_staging")

    def test_mtp_accepted_kv_survives_backup_and_direct_page_release(self):
        """Accepted KV is readable after its temporary physical page is freed."""
        initial = self._get_initial_sizes()
        fill_len = DEVICE_BUFFER_SIZE + self.page_size
        accepted_count = self.page_size
        req = _make_req("mtp-accepted-backup", list(range(fill_len)))
        self._alloc_req_slot(req)

        kv_loc = self._alloc_kv(req, fill_len)
        self._write_device_patterns(kv_loc, fill_len)
        self.coordinator.admit_request_into_staging(req)
        self._collect_ready_reqs_blocking()

        accepted_locs = self.allocator.alloc_extend(
            prefix_lens=torch.tensor([fill_len], dtype=torch.int64, device="cuda"),
            prefix_lens_cpu=torch.tensor([fill_len], dtype=torch.int64),
            seq_lens=torch.tensor(
                [fill_len + accepted_count], dtype=torch.int64, device="cuda"
            ),
            seq_lens_cpu=torch.tensor(
                [fill_len + accepted_count], dtype=torch.int64
            ),
            last_loc=kv_loc[-1:].to(torch.int64),
            extend_num_tokens=accepted_count,
        )
        self.assertIsNotNone(accepted_locs)
        self.req_to_token_pool.write(
            (req.req_pool_idx, slice(fill_len, fill_len + accepted_count)),
            accepted_locs,
        )
        req.kv.kv_allocated_len = fill_len + accepted_count
        req.kv_committed_len = fill_len + accepted_count
        self.coordinator.register_token_positions(
            req, fill_len, fill_len + accepted_count
        )

        accepted_device_locs = (
            self.allocator.full_to_hisparse_device_index_mapping[accepted_locs]
        )
        self.assertTrue(torch.all(accepted_device_locs > 0))
        for layer_id in range(LAYER_NUM):
            for i, device_loc in enumerate(accepted_device_locs):
                position = fill_len + i
                self.device_pool.kv_buffer[layer_id][device_loc] = self._kv_pattern(
                    layer_id, position
                )

        self.coordinator.backup_committed_tokens([req])
        # Mapping stays valid — physical pages freed in request_finished.
        self.assertTrue(
            torch.all(
                self.allocator.full_to_hisparse_device_index_mapping[accepted_locs]
                > 0
            )
        )

        logical_top_k = accepted_locs[:TOP_K].to(torch.int32).unsqueeze(0)
        if accepted_count < TOP_K:
            logical_top_k = torch.cat(
                [
                    logical_top_k,
                    torch.full(
                        (1, TOP_K - accepted_count),
                        -1,
                        dtype=torch.int32,
                        device="cuda",
                    ),
                ],
                dim=1,
            )
        rpi, sls = self._make_batch_tensors(
            [req], [fill_len + accepted_count]
        )
        for layer_id in range(LAYER_NUM):
            locs = self._swap_in_selected_logical_pages(
                rpi, sls, logical_top_k, layer_id
            )
            for i in range(min(accepted_count, TOP_K)):
                position = fill_len + i
                expected = self._kv_pattern(layer_id, position)
                actual = self.device_pool.kv_buffer[layer_id][locs[0, i].long()]
                self.assertTrue(
                    torch.allclose(
                        actual.float(),
                        torch.full_like(actual.float(), expected),
                        atol=1e-2,
                    ),
                    f"layer {layer_id}, accepted position {position}: KV mismatch",
                )

        all_locs = torch.cat([kv_loc, accepted_locs])
        self._cleanup_req(req, all_locs)
        self._assert_sizes_restored(initial, "mtp_accepted_backup")

    def test_staging_defers_surplus_pages_until_backup_finishes(self):
        """Surplus prefill pages stay reserved while staging may still read them."""
        initial = self._get_initial_sizes()
        fill_len = DEVICE_BUFFER_SIZE + self.page_size * 2
        req = _make_req("staging-deferred-free", list(range(fill_len)))
        self._alloc_req_slot(req)

        kv_loc = self._alloc_kv(req, fill_len)
        self._write_device_patterns(kv_loc, fill_len)
        self.coordinator.admit_request_into_staging(req)

        self.assertEqual(len(self.coordinator.ack_staging_queue), 1)
        action = self.coordinator.ack_staging_queue[0]
        self.assertIsNotNone(action.deferred_free_indices)
        self.assertEqual(action.deferred_free_indices.numel(), 1)
        ring_cap = self.coordinator.spec_ring_capacity
        self.assertEqual(
            self.allocator.hisparse_attn_allocator.available_size(),
            initial[1] - fill_len - ring_cap,
        )

        action.finish_event.synchronize()
        self.assertEqual(self.coordinator.collect_ready_reqs(), [req])
        self.assertEqual(
            self.allocator.hisparse_attn_allocator.available_size(),
            initial[1] - self.coordinator.padded_buffer_size - ring_cap,
        )

        self._cleanup_req(req, kv_loc)
        self._assert_sizes_restored(initial, "staging_deferred_free")

    def test_abort_staging_releases_deferred_and_buffer_pages(self):
        initial = self._get_initial_sizes()
        fill_len = DEVICE_BUFFER_SIZE + self.page_size * 2
        req = _make_req("staging-abort", list(range(fill_len)))
        self._alloc_req_slot(req)

        kv_loc = self._alloc_kv(req, fill_len)
        self._write_device_patterns(kv_loc, fill_len)
        self.coordinator.admit_request_into_staging(req)
        self.coordinator.abort_staging_request(req)

        self.assertFalse(req.hisparse_staging)
        self.assertEqual(self.coordinator.ack_staging_queue, [])
        self.assertEqual(
            int(self.coordinator.req_device_buffer_size[req.req_pool_idx]), 0
        )
        self.assertTrue(
            torch.all(self.coordinator.req_to_device_buffer[req.req_pool_idx] == 0)
        )
        self.assertTrue(
            torch.all(self.coordinator.req_to_host_pool[req.req_pool_idx] == -1)
        )

        self.allocator.free(kv_loc)
        self._free_req_slot(req)
        self._assert_sizes_restored(initial, "staging_abort")

    def test_request_finished_drains_inflight_staging(self):
        """Direct finish paths must wait for DMA and release deferred pages."""
        initial = self._get_initial_sizes()
        fill_len = DEVICE_BUFFER_SIZE + self.page_size * 2
        req = _make_req("staging-direct-finish", list(range(fill_len)))
        self._alloc_req_slot(req)

        kv_loc = self._alloc_kv(req, fill_len)
        self._write_device_patterns(kv_loc, fill_len)
        self.coordinator.admit_request_into_staging(req)

        self.coordinator.request_finished(req)

        self.assertFalse(req.hisparse_staging)
        self.assertEqual(self.coordinator.ack_staging_queue, [])
        self.assertEqual(
            int(self.coordinator.req_device_buffer_size[req.req_pool_idx]), 0
        )
        self.assertTrue(
            torch.all(self.coordinator.req_to_host_pool[req.req_pool_idx] == -1)
        )

        self.allocator.free(kv_loc)
        self._free_req_slot(req)
        self._assert_sizes_restored(initial, "staging_direct_finish")

    def test_staging_action_survives_eager_allocation_failure(self):
        """A post-DMA allocation error must retain an action for safe cleanup."""
        initial = self._get_initial_sizes()
        fill_len = self.page_size
        req = _make_req("staging-allocation-failure", list(range(fill_len)))
        self._alloc_req_slot(req)

        kv_loc = self._alloc_kv(req, fill_len)
        self._write_device_patterns(kv_loc, fill_len)
        with patch.object(
            self.coordinator,
            "alloc_device_buffer",
            side_effect=RuntimeError("injected eager allocation failure"),
        ):
            with self.assertRaisesRegex(RuntimeError, "injected eager"):
                self.coordinator.admit_request_into_staging(req)

        self.assertTrue(req.hisparse_staging)
        self.assertEqual(len(self.coordinator.ack_staging_queue), 1)
        self.coordinator.request_finished(req)

        self.assertFalse(req.hisparse_staging)
        self.assertEqual(self.coordinator.ack_staging_queue, [])
        self.allocator.free(kv_loc)
        self._free_req_slot(req)
        self._assert_sizes_restored(initial, "staging_allocation_failure")

    # ==================================================================
    # Test: Single-node staging host page allocation
    # ==================================================================
    def test_single_node_staging_allocates_paged_host_slots(self):
        """Single-node staging should allocate host slots at page granularity."""
        initial = self._get_initial_sizes()
        fill_len = self.page_size * 2 + 1
        rounded_len = (fill_len + self.page_size - 1) // self.page_size * self.page_size
        req = _make_req("single-node-staging-pages", list(range(fill_len)))
        self._alloc_req_slot(req)

        kv_loc = self._alloc_kv(req, fill_len)
        self._write_device_patterns(kv_loc, fill_len)

        self.coordinator.admit_request_into_staging(req)
        torch.cuda.synchronize()
        ready = self.coordinator.collect_ready_reqs()
        self.assertEqual(ready, [req])

        host_row = self.coordinator.req_to_host_pool[req.req_pool_idx, :rounded_len]
        self.assertTrue(torch.all(host_row >= 0))
        self.assertEqual(torch.unique(host_row).numel(), rounded_len)
        self.assertEqual(
            int(self.coordinator.req_to_host_pool_allocated_len[req.req_pool_idx]),
            rounded_len,
        )

        available_size = self.coordinator.mem_pool_host.available_size()
        next_host_index = self.coordinator.mem_pool_host.alloc_paged_token_slots(
            self.coordinator.req_to_host_pool,
            self.coordinator.req_to_host_pool_allocated_len,
            req.req_pool_idx,
            fill_len,
            1,
        )
        # With page_size>1 the rounded-up staging allocation provides headroom,
        # so no new pages are needed.  With page_size=1 there is no headroom and
        # exactly one new page is allocated for the next token.
        expected_new_pages = 0 if fill_len < rounded_len else 1
        self.assertEqual(
            self.coordinator.mem_pool_host.available_size(),
            available_size - expected_new_pages,
        )
        self.assertTrue(torch.all(next_host_index >= 0))

        expected_total = rounded_len + expected_new_pages * self.page_size
        allocated_host_indices = self.coordinator.mem_pool_host.allocated_host_indices(
            self.coordinator.req_to_host_pool,
            req.req_pool_idx,
            int(self.coordinator.req_to_host_pool_allocated_len[req.req_pool_idx]),
        )
        self.assertEqual(allocated_host_indices.numel(), expected_total)

        self._cleanup_req(req, kv_loc)
        self._assert_sizes_restored(initial, "single_node_staging_pages")

    # ==================================================================
    # Test: Direct-to-host (PD separated) path
    # ==================================================================
    def test_request_lifecycle_direct_path(self):
        """alloc_logical_only -> host write -> admit_direct -> swap-in -> finish."""
        initial = self._get_initial_sizes()
        fill_len = DEVICE_BUFFER_SIZE + self.page_size
        req = _make_req("direct-req", list(range(fill_len)))
        self._alloc_req_slot(req)

        kv_loc = self._alloc_kv(req, fill_len, logical_only=True)
        self._populate_host_pool(req, fill_len)
        self.coordinator.admit_request_direct(req)

        self.assertFalse(req.staging)
        self.assertTrue(self.coordinator._skip_first_backup[req.req_pool_idx])
        buf_tokens = self.coordinator.req_device_buffer_tokens[
            :, req.req_pool_idx, :DEVICE_BUFFER_SIZE
        ]
        self.assertTrue(torch.all(buf_tokens == -1))

        tokens = self._build_topk_tokens(fill_len - 1)
        batch = tokens.unsqueeze(0)
        rpi, sls = self._make_batch_tensors([req], [fill_len])

        locs = self._swap_in_selected_pages(rpi, sls, batch, layer_id=0)
        self.assertTrue(torch.all(locs[0, :TOP_K] >= 0))
        self._assert_kv_correct(
            locs[0], tokens, layer_id=0, count=TOP_K, msg="Direct: "
        )
        self._assert_matches_naive(rpi, sls, batch, locs, layer_id=0, msg="Direct: ")

        self._cleanup_req(req, kv_loc, logical_only=True)
        self._assert_sizes_restored(initial, "direct_path")

    # ==================================================================
    # Test: PD decode prealloc host page allocation
    # ==================================================================
    def test_pd_decode_prealloc_hisparse_host_slots(self):
        """PD decode prealloc should allocate RDMA targets through the host pool."""
        initial = self._get_initial_sizes()
        fill_len = self.page_size * 2 + 1
        req = _make_req("pd-decode-prealloc", list(range(fill_len)))

        from sglang.srt.disaggregation.decode import DecodePreallocQueue

        queue = DecodePreallocQueue.__new__(DecodePreallocQueue)
        queue.req_to_token_pool = self.req_to_token_pool
        queue.token_to_kv_pool_allocator = self.allocator
        queue.token_to_kv_pool = self.allocator.get_kvcache()
        queue.tree_cache = SimpleNamespace(
            evictable_size=lambda: 0,
            protected_size=lambda: 0,
        )
        queue.scheduler = SimpleNamespace(
            enable_hisparse=True,
            hisparse_coordinator=self.coordinator,
            server_args=SimpleNamespace(disaggregation_decode_enable_radix_cache=False),
        )

        host_indices = queue._pre_alloc(req)
        self.assertEqual(host_indices.numel(), fill_len)
        self.assertTrue(torch.all(host_indices >= 0))
        self.assertTrue(
            torch.equal(
                host_indices,
                self.coordinator.req_to_host_pool[req.req_pool_idx, :fill_len],
            )
        )
        self.assertEqual(req.kv.kv_allocated_len, fill_len)
        self.assertEqual(req.kv_committed_len, fill_len)
        self.assertEqual(req.extend_range.length, fill_len)

        rounded_len = (fill_len + self.page_size - 1) // self.page_size * self.page_size
        self.assertEqual(
            int(self.coordinator.req_to_host_pool_allocated_len[req.req_pool_idx]),
            rounded_len,
        )
        allocated_host_indices = self.coordinator.mem_pool_host.allocated_host_indices(
            self.coordinator.req_to_host_pool,
            req.req_pool_idx,
            int(self.coordinator.req_to_host_pool_allocated_len[req.req_pool_idx]),
        )
        self.assertEqual(allocated_host_indices.numel(), rounded_len)

        kv_loc = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, : req.kv.kv_allocated_len
        ].clone()
        self._cleanup_req(req, kv_loc, logical_only=True)
        self._assert_sizes_restored(initial, "pd_decode_prealloc_hisparse")

    # ==================================================================
    # Test: Batch multiple requests
    # ==================================================================
    def test_batch_multiple_requests(self):
        """Mix of short & long requests in batch: kernel correct + no leaks."""
        initial = self._get_initial_sizes()

        configs = [
            ("batch-short-0", self.page_size),
            ("batch-short-1", self.page_size),
            ("batch-long-0", DEVICE_BUFFER_SIZE + self.page_size),
            ("batch-long-1", DEVICE_BUFFER_SIZE + self.page_size * 2),
        ]

        reqs, kv_locs = [], []
        for rid, fl in configs:
            req = _make_req(rid, list(range(fl)))
            self._alloc_req_slot(req)
            is_long = fl > DEVICE_BUFFER_SIZE
            kv_loc = self._alloc_kv(req, fl, logical_only=is_long)
            if is_long:
                self._populate_host_pool(req, fl)
                self.coordinator.admit_request_direct(req)
            else:
                self._write_device_patterns(kv_loc, fl)
                self.coordinator.alloc_device_buffer(req)
            reqs.append(req)
            kv_locs.append(kv_loc)

        rpi, sls = self._make_batch_tensors(reqs, [c[1] for c in configs])
        top_k_batch = torch.stack(
            [
                # For long sequences pass fl-1 to exclude the "newest token" position
                # whose reserved device-buffer slot is not populated in unit tests.
                self._build_topk_tokens(fl - 1 if fl > DEVICE_BUFFER_SIZE else fl)
                for _, fl in configs
            ]
        )

        for lid in range(LAYER_NUM):
            locs = self._swap_in_selected_pages(rpi, sls, top_k_batch, lid)
            for i, (rid, fl) in enumerate(configs):
                vn = min(fl, TOP_K)
                self.assertTrue(
                    torch.all(locs[i, :vn] >= 0),
                    f"Req {rid}, layer {lid}: negative locs",
                )
                self._assert_kv_correct(
                    locs[i], top_k_batch[i], lid, vn, msg=f"{rid}: "
                )

        for i, req in enumerate(reqs):
            is_long = configs[i][1] > DEVICE_BUFFER_SIZE
            self._cleanup_req(req, kv_locs[i], logical_only=is_long)

        self._assert_sizes_restored(initial, "batch_multiple")

    # ==================================================================
    # Test: speculative staging ring
    # ==================================================================

    def _enable_spec_ring(self, capacity):
        """Resize the ring on the MTP coordinator for capacity-specific tests."""
        assert capacity % self.page_size == 0
        assert self.coordinator is self.mtp_coordinator
        assert self.coordinator.req_device_buffer_tokens is None
        assert self.coordinator.req_device_buffer_logical_locs is not None
        self.coordinator.spec_ring_capacity = capacity
        self.coordinator.req_to_spec_ring = torch.zeros(
            (MAX_NUM_REQS, capacity), dtype=torch.int64, device="cuda"
        )
        self.coordinator.req_spec_ring_active = [False] * MAX_NUM_REQS

    def _extend_logical(self, req, start, end):
        """Extend [start, end) with logical-only slots and write req_to_token."""
        device = self.allocator.device
        rid = req.req_pool_idx
        last_loc = (
            self.req_to_token_pool.req_to_token[rid, start - 1 : start].to(torch.int64)
            if start > 0
            else torch.tensor([-1], dtype=torch.int64, device=device)
        )
        kv_loc = self.allocator.alloc_logical_only(
            prefix_lens=torch.tensor([start], dtype=torch.int64, device=device),
            prefix_lens_cpu=torch.tensor([start], dtype=torch.int64),
            seq_lens=torch.tensor([end], dtype=torch.int64, device=device),
            seq_lens_cpu=torch.tensor([end], dtype=torch.int64),
            last_loc=last_loc,
            extend_num_tokens=end - start,
        )
        self.assertIsNotNone(kv_loc, "logical-only alloc failed")
        self.req_to_token_pool.write((rid, slice(start, end)), kv_loc)
        req.kv.kv_allocated_len = end
        return kv_loc

    def test_spec_ring_pool_stable_across_rounds(self):
        """Core regression for HiSparse+MTP memory growth: after the ring is
        allocated, many speculative rounds must not consume any additional
        hisparse physical pages, and request_finished must restore all pools."""
        initial = self._get_initial_sizes()
        R = 2 * self.page_size
        self._enable_spec_ring(R)

        fill_len = self.page_size
        req = _make_req("spec-ring-stable", list(range(fill_len)))
        self._alloc_req_slot(req)
        kv_loc = self._alloc_kv(req, fill_len)
        req.hisparse_last_backed_len = fill_len

        self.coordinator._alloc_spec_ring(req)
        self.assertTrue(self.coordinator.req_spec_ring_active[req.req_pool_idx])
        hisparse_after_ring = self.allocator.hisparse_attn_allocator.available_size()

        mapping = self.allocator.full_to_hisparse_device_index_mapping
        ring = self.coordinator.req_to_spec_ring[req.req_pool_idx]
        step = 8
        pos = fill_len
        for _ in range(12):  # 96 tokens: wraps a 128-slot ring around fill_len=64
            nxt = pos + step
            spec_loc = self._extend_logical(req, pos, nxt)
            self.coordinator.assign_spec_ring_slots(req, pos, nxt)
            # New positions map into the ring region.
            expect = ring[
                torch.arange(pos, nxt, dtype=torch.int64, device="cuda") % R
            ]
            self.assertTrue(torch.equal(mapping[spec_loc], expect))
            # No physical pages consumed by speculative rounds.
            self.assertEqual(
                self.allocator.hisparse_attn_allocator.available_size(),
                hisparse_after_ring,
                f"hisparse pool shrank at pos {pos}",
            )
            # Simulate commit + backup of everything allocated so far.
            req.kv_committed_len = nxt
            req.hisparse_last_backed_len = nxt
            pos = nxt

        self.coordinator.request_finished(req)
        all_locs = self.req_to_token_pool.req_to_token[req.req_pool_idx, :pos].to(
            torch.int64
        )
        self.assertTrue(torch.all(mapping[all_locs] == 0))
        self.allocator.logical_attn_allocator.free(all_locs)
        self._free_req_slot(req)
        self._assert_sizes_restored(initial, "spec_ring_stable")

    def test_spec_ring_wrap_recycles_mapping(self):
        """Position p and p+R share a ring slot; recycling clears p's mapping."""
        initial = self._get_initial_sizes()
        R = self.page_size
        self._enable_spec_ring(R)

        fill_len = self.page_size
        req = _make_req("spec-ring-wrap", list(range(fill_len)))
        self._alloc_req_slot(req)
        self._alloc_kv(req, fill_len)
        req.hisparse_last_backed_len = fill_len
        self.coordinator._alloc_spec_ring(req)

        mapping = self.allocator.full_to_hisparse_device_index_mapping
        rid = req.req_pool_idx
        first_loc = self._extend_logical(req, fill_len, fill_len + 4)
        self.coordinator.assign_spec_ring_slots(req, fill_len, fill_len + 4)
        first_mapping = mapping[first_loc].clone()
        req.hisparse_last_backed_len = fill_len + 4

        # One full ring later the same slots must be reused.
        self._extend_logical(req, fill_len + 4, fill_len + R)
        self.coordinator.assign_spec_ring_slots(req, fill_len + 4, fill_len + R)
        req.hisparse_last_backed_len = fill_len + R
        second_loc = self._extend_logical(req, fill_len + R, fill_len + R + 4)
        self.coordinator.assign_spec_ring_slots(req, fill_len + R, fill_len + R + 4)

        self.assertTrue(torch.all(mapping[first_loc] == 0), "recycled mapping kept")
        self.assertTrue(torch.equal(mapping[second_loc], first_mapping))

        # Recycling a position that is not backed up yet must fail loudly.
        req.hisparse_last_backed_len = fill_len  # pretend backup regressed
        self._extend_logical(req, fill_len + R + 4, fill_len + 2 * R)
        with self.assertRaises(AssertionError):
            self.coordinator.assign_spec_ring_slots(
                req, fill_len + R + 4, fill_len + 2 * R
            )

        self.coordinator.request_finished(req)
        all_locs = self.req_to_token_pool.req_to_token[
            rid, : req.kv.kv_allocated_len
        ].to(torch.int64)
        self.allocator.logical_attn_allocator.free(all_locs)
        self._free_req_slot(req)
        self._assert_sizes_restored(initial, "spec_ring_wrap")

    def test_spec_mode_allocates_full_device_buffer(self):
        """With the ring enabled, verify's eviction slow path may DMA into any
        LRU slot, so alloc_device_buffer must provision every padded slot."""
        initial = self._get_initial_sizes()
        self._enable_spec_ring(self.page_size)

        fill_len = self.page_size
        req = _make_req("spec-full-buffer", list(range(fill_len)))
        self._alloc_req_slot(req)
        kv_loc = self._alloc_kv(req, fill_len)

        self.coordinator.alloc_device_buffer(req)
        rid = req.req_pool_idx
        self.assertEqual(
            int(self.coordinator.req_device_buffer_size[rid]),
            self.coordinator.padded_buffer_size,
        )
        locs = self.coordinator.req_device_buffer_token_locs[0, rid]
        self.assertTrue(
            torch.all(locs[: self.coordinator.padded_buffer_size] >= 0),
            "unallocated buffer slots would be invalid DMA destinations",
        )

        self.coordinator.request_finished(req)
        self.allocator.logical_attn_allocator.free(kv_loc)
        self._free_req_slot(req)
        self._assert_sizes_restored(initial, "spec_full_buffer")

    def test_admit_long_prompt_reclaims_deferred_pages_for_ring(self):
        """Staging transient peak: a long prompt holds nearly the whole
        physical pool until the staging DMA completes (surplus pages are
        deferred-freed), while the ring is allocated immediately. The ring
        allocation must reclaim deferred pages instead of raising."""
        if self.page_size == 1:
            self.skipTest("deferred-page accounting requires page_size > 1")
        initial = self._get_initial_sizes()
        R = 2 * self.page_size
        self._enable_spec_ring(R)

        # Leave only one free page — less than the ring needs.
        fill_len = SIZE - self.page_size
        req = _make_req("staging-peak", list(range(fill_len)))
        self._alloc_req_slot(req)
        self._alloc_kv(req, fill_len)
        self.assertLess(
            self.allocator.hisparse_attn_allocator.available_size(), R,
            "precondition: free pool must be smaller than the ring",
        )

        self.coordinator.admit_request_into_staging(req)
        self.assertTrue(self.coordinator.req_spec_ring_active[req.req_pool_idx])

        self.coordinator.request_finished(req)
        all_locs = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, :fill_len
        ].to(torch.int64)
        self.allocator.logical_attn_allocator.free(all_locs)
        self._free_req_slot(req)
        self._assert_sizes_restored(initial, "staging_peak_ring")


if __name__ == "__main__":
    unittest.main()
