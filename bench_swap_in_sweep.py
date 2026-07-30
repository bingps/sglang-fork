"""Sweep of HiSparseMTPCoordinator.swap_in_verify_pages per-layer latency.

Grid:
  batch size bs = 1, 8, 16, 32
  draft tokens N = 1..5   (device buffer sized to N * top_k)
  hit rate     h = 0.5, 0.7, 0.9   (fraction of each row's top-k already resident)

Hit rate is controlled by construction: each verify row selects
``round(h*top_k)`` ids from a per-request HOT set that is touched every
iteration (so it stays resident in the device buffer -> buffer-gather hit) and
the remaining ``top_k - round(h*top_k)`` ids fresh from a large cold pool (never
resident -> host->device DMA miss). After warmup this yields a steady per-row
hit rate ~= h. Latency-only: KV values are not written (the DMA moves the right
volume of bytes either way), so setup skips per-token host writes.

Reported: microseconds per kernel call == per model layer (the verify swap-in
runs once per layer). x61 gives the DSV3.2 per-step cost.
"""

import os
from array import array
from types import SimpleNamespace

import torch

from sglang.srt.utils.common import Range

PAGE = 64
TOP_K = 2048
RATIO = 2
KV_LORA_RANK = 512
QK_ROPE_HEAD_DIM = 64
KV_CACHE_DIM = 576
LAYER_NUM = 1
NUM_LAYERS_MODEL = 61
FILL = 16384
MAX_REQS = 32
MAX_CTX = FILL
WARMUP, ITERS = 8, 40

BS_LIST = [int(x) for x in os.environ.get("SWEEP_BS", "1,8,16,32").split(",")]
N_LIST = [int(x) for x in os.environ.get("SWEEP_N", "1,2,3,4,5").split(",")]
H_LIST = [float(x) for x in os.environ.get("SWEEP_H", "0.5,0.7,0.9").split(",")]


def build_stack(n_draft):
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29601")
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="gloo", rank=0, world_size=1)
    from sglang.srt.mem_cache.pool_host.common import (
        ALLOC_MEMORY_FUNCS,
        alloc_with_pin_memory,
    )

    ALLOC_MEMORY_FUNCS["cuda"] = alloc_with_pin_memory

    from sglang.srt.managers.hisparse_coordinator import HiSparseMTPCoordinator
    from sglang.srt.mem_cache.allocator.hisparse import HiSparseTokenToKVPoolAllocator
    from sglang.srt.mem_cache.hisparse_memory_pool import HiSparseDSATokenToKVPool
    from sglang.srt.mem_cache.memory_pool import ReqToTokenPool

    buffer_size = int(os.environ.get("SWEEP_BUFFER", "0")) or (n_draft * TOP_K)
    # hisparse pool holds bs*(padded) device buffers (alloc uses buffer+PAGE);
    # logical pool (=SIZE*RATIO) holds bs*FILL offloaded logical ids. Add a
    # per-request page of slack for alignment rounding in either allocator.
    size = max(
        MAX_REQS * (buffer_size + PAGE),
        (MAX_REQS * FILL + RATIO - 1) // RATIO,
    ) + MAX_REQS * PAGE
    size = ((size + PAGE - 1) // PAGE) * PAGE

    pool = HiSparseDSATokenToKVPool(
        size=size, page_size=PAGE, kv_lora_rank=KV_LORA_RANK, dtype=torch.bfloat16,
        qk_rope_head_dim=QK_ROPE_HEAD_DIM, layer_num=LAYER_NUM, device="cuda",
        index_head_dim=128, enable_memory_saver=False, kv_cache_dim=KV_CACHE_DIM,
        host_to_device_ratio=RATIO,
    )
    allocator = HiSparseTokenToKVPoolAllocator(
        size=size, page_size=PAGE, dtype=torch.bfloat16, device="cuda",
        kvcache=pool, need_sort=False, host_to_device_ratio=RATIO,
    )
    r2t = ReqToTokenPool(
        size=MAX_REQS, max_context_len=MAX_CTX, device="cuda",
        enable_memory_saver=False,
    )
    coord = HiSparseMTPCoordinator(
        req_to_token_pool=r2t, token_to_kv_pool_allocator=allocator, top_k=TOP_K,
        device_buffer_size=buffer_size, device="cuda",
        tp_group=torch.distributed.group.WORLD, host_to_device_ratio=RATIO,
        num_draft_tokens=n_draft, spec_ring_capacity=PAGE,
    )
    return pool, allocator, r2t, coord


def make_req(rid):
    req = SimpleNamespace(
        rid=rid, origin_input_ids=list(range(FILL)), output_ids=[],
        fill_ids=list(range(FILL)), seqlen=FILL, req_pool_idx=None,
        kv=SimpleNamespace(kv_allocated_len=0), kv_committed_len=0,
        finished_reason=None, hisparse_staging=False, hisparse_last_backed_len=None,
        hisparse_ring_start=None, staging=False, inflight_middle_chunks=0,
    )
    req.finished = lambda: req.finished_reason is not None
    req.set_extend_range = lambda s, e: setattr(req, "extend_range", Range(s, e))
    return req


def admit_offloaded(coord, allocator, r2t, req):
    dev = allocator.device
    kv_loc = allocator.alloc_logical_only(
        prefix_lens=torch.tensor([0], dtype=torch.int64, device=dev),
        prefix_lens_cpu=torch.tensor([0], dtype=torch.int64),
        seq_lens=torch.tensor([FILL], dtype=torch.int64, device=dev),
        seq_lens_cpu=torch.tensor([FILL], dtype=torch.int64),
        last_loc=torch.tensor([-1], dtype=torch.int64, device=dev),
        extend_num_tokens=FILL,
    )
    assert kv_loc is not None
    r2t.write((req.req_pool_idx, slice(0, FILL)), kv_loc)
    req.kv.kv_allocated_len = FILL
    req.kv_committed_len = FILL
    req.full_untruncated_fill_ids = array("q", range(FILL))
    req.extend_range = Range(0, FILL)
    host_idx = coord.mem_pool_host.alloc(FILL).to(device=dev)
    coord.req_to_host_pool[req.req_pool_idx, :FILL] = host_idx
    coord.req_to_host_pool_allocated_len[req.req_pool_idx] = FILL
    coord.admit_request_direct(req)
    coord.register_token_positions(req, 0, FILL)
    return kv_loc


def build_topk(kv_all, bs, n, nhot, gen):
    """kv_all: [bs, FILL] logical ids. Returns [bs*n, TOP_K] int32 req-major."""
    dev = kv_all.device
    ncold = TOP_K - nhot
    hot = kv_all[:, :nhot]  # [bs, nhot]
    hot = hot.unsqueeze(1).expand(bs, n, nhot)  # shared across n positions
    if ncold > 0:
        cold_pos = torch.randint(
            nhot, FILL, (bs, n * ncold), device=dev, generator=gen
        )
        cold = torch.gather(kv_all, 1, cold_pos).view(bs, n, ncold)
        rows = torch.cat([hot, cold], dim=2)
    else:
        rows = hot
    return rows.reshape(bs * n, TOP_K).to(torch.int32).contiguous()


def run_cell(coord, allocator, r2t, bs, n, h):
    dev = allocator.device
    gen = torch.Generator(device=dev)
    gen.manual_seed(1234 + bs * 100 + n * 10 + int(h * 10))
    nhot = round(h * TOP_K)

    reqs, kv_locs = [], []
    for i in range(bs):
        req = make_req(f"c{bs}-{n}-{i}")
        assert r2t.alloc([req]) is not None
        kv_locs.append(admit_offloaded(coord, allocator, r2t, req))
        reqs.append(req)
    kv_all = torch.stack([kv.to(torch.int64) for kv in kv_locs])  # [bs, FILL]
    rpi = torch.tensor([r.req_pool_idx for r in reqs], dtype=torch.int64, device=dev)
    sls = torch.full((bs,), FILL, dtype=torch.int32, device=dev)
    coord.num_real_reqs[0] = bs

    for _ in range(WARMUP):
        coord.swap_in_verify_pages(rpi, sls, build_topk(kv_all, bs, n, nhot, gen), 0, n)
    prebuilt = [build_topk(kv_all, bs, n, nhot, gen) for _ in range(ITERS)]
    torch.cuda.synchronize()

    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    for tk in prebuilt:
        coord.swap_in_verify_pages(rpi, sls, tk, 0, n)
    end.record()
    torch.cuda.synchronize()
    us = start.elapsed_time(end) * 1000 / ITERS

    for req, kv_loc in zip(reqs, kv_locs):
        coord.request_finished(req)
        allocator.logical_attn_allocator.free(kv_loc)
        r2t.free(req)
    torch.cuda.synchronize()
    return us


def main():
    torch.cuda.init()
    # Pre-warm the GPU so the first measured cell doesn't see cold clocks.
    a = torch.randn(4096, 4096, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(4096, 4096, dtype=torch.bfloat16, device="cuda")
    for _ in range(50):
        (a @ b).sum()
    torch.cuda.synchronize()
    del a, b
    torch.cuda.empty_cache()

    print(f"gpu={torch.cuda.get_device_name(0)}  top_k={TOP_K} fill={FILL} "
          f"warmup={WARMUP} iters={ITERS}")
    print("per-layer latency (us/call); ms/step = us x 61 / 1000\n")

    grid = {}  # (h, bs, n) -> us
    for n in N_LIST:
        pool, allocator, r2t, coord = build_stack(n)
        print(f"# N={n}  buffer={n * TOP_K}  item={coord.item_size_bytes}B "
              f"block={coord.swap_in_block_size}", flush=True)
        for bs in BS_LIST:
            for h in H_LIST:
                us = run_cell(coord, allocator, r2t, bs, n, h)
                grid[(h, bs, n)] = us
                print(f"    bs={bs:<3} h={h}  {us:8.1f} us/layer  "
                      f"{us * NUM_LAYERS_MODEL / 1000:6.2f} ms/step", flush=True)
        del coord, allocator, r2t, pool
        torch.cuda.empty_cache()

    for h in H_LIST:
        print(f"\n=== per-layer latency us  (hit rate = {h}) ===")
        print("%-6s" % "bs\\N" + "".join("%9d" % n for n in N_LIST))
        for bs in BS_LIST:
            print("%-6d" % bs + "".join("%9.1f" % grid[(h, bs, n)] for n in N_LIST))

    for h in H_LIST:
        print(f"\n=== ms/step (x61)  (hit rate = {h}) ===")
        print("%-6s" % "bs\\N" + "".join("%9d" % n for n in N_LIST))
        for bs in BS_LIST:
            print("%-6d" % bs
                  + "".join("%9.2f" % (grid[(h, bs, n)] * NUM_LAYERS_MODEL / 1000)
                            for n in N_LIST))


if __name__ == "__main__":
    main()
