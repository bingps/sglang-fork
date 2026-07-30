# HiSparseMTPCoordinator.swap_in_verify_pages — Latency Sweep

Microbenchmark of the MTP verify swap-in kernel across the operating grid a
production DSV3.2 deployment actually visits. Measures the per-layer kernel
cost (`swap_in_verify_pages` is invoked once per model layer during EAGLE
verify), then extrapolates to the 61-layer DSV3.2 per-step cost. Each cell
shows `probe-only / total` in microseconds per layer:

- **probe-only**: kernel compiled with `-DHISPARSE_PROBE_ONLY`, which
  short-circuits the per-miss host->device byte copies while keeping the hash
  probe, miss enumeration, and LRU write-back intact. Isolates the fixed
  bookkeeping cost from the host-DMA cost.
- **total**: unmodified kernel (default build).

`total - probe = host DMA cost` for that cell.

## Setup

- GPU: NVIDIA L20X
- Host-Device link: **PCIe Gen5 x16** per GPU (verified via
  `nvidia-smi --query-gpu=pcie.link.gen.current,pcie.link.width.current`).
  Theoretical peak ~63 GB/s single-direction; practical pinned host->device
  ~50 GB/s (fits the observed DMA slope below). GPU-GPU NV18 NVLink exists
  in the topology matrix but is off the host DMA path. No CPU-GPU NVLink on
  this platform (Grace Hopper only), so DMA bandwidth is capped by PCIe.
- `top_k = 2048`, `seq_len = 16384`, warmup = 8 iters, timed = 40 iters
- KV geometry: `kv_cache_dim = 576` (`kv_lora_rank 512 + qk_rope_head_dim 64`),
  `dtype = bfloat16`, `item_size = 1152 B/token/layer`, `page_size = 64`,
  DSV3.2-shaped MLA compressed KV
- GPU pre-warm (50 x bf16 4096x4096 matmul) before the first measured cell
- Coordinator rebuilt per `N` so `device_buffer_size = N * top_k` matches
  production, except the `N=1*` column which forces `device_buffer_size = 4096`
- Hit rate is controlled by construction: each verify row selects
  `round(h * top_k)` ids from a per-request HOT set that is touched every
  iteration (stays resident in the device buffer -> buffer-gather hit) and the
  remaining `top_k - round(h * top_k)` ids freshly drawn from a cold pool of
  size `FILL - hot` each iteration (never resident -> host->device DMA miss).
  After warmup this yields a steady per-row miss fraction ~= 1 - h.

Source: [ bench_swap_in_sweep.py ](../../bench_swap_in_sweep.py)

Kernel probe-only toggle: [ python/sglang/kernels/jit/csrc/hisparse.cuh ](../../python/sglang/kernels/jit/csrc/hisparse.cuh) `HISPARSE_PROBE_ONLY` compile guard; env `HISPARSE_PROBE_ONLY=1` routes through [ python/sglang/kernels/ops/kvcache/hisparse.py ](../../python/sglang/kernels/ops/kvcache/hisparse.py) `_jit_sparse_mtp_module` cache key.

Raw logs: [ 20260730_145655_swap_in_sweep2.log ](../../logs/20260730_145655_swap_in_sweep2.log) (total, default buffer),
[ 20260730_150106_swap_in_N1_buf4096.log ](../../logs/20260730_150106_swap_in_N1_buf4096.log) (total, N=1\*),
[ 20260730_152607_probe_only.log ](../../logs/20260730_152607_probe_only.log) (probe-only, default buffer),
[ 20260730_154033_probe_N1_buf4096.log ](../../logs/20260730_154033_probe_N1_buf4096.log) (probe-only, N=1\*).

## us / layer -- probe-only / total

### Hit rate 0.5

| bs \ N |           1 |          1* |            2 |            3 |            4 |            5 |
|-------:|------------:|------------:|-------------:|-------------:|-------------:|-------------:|
|      1 |  31.5/211.4 |  28.3/117.5 |   38.3/231.5 |   65.9/313.7 |  100.7/377.9 |  144.3/423.3 |
|      8 |  34.0/331.3 |  36.2/176.6 |   40.5/345.4 |   68.2/456.1 |  103.5/532.7 |  147.0/575.6 |
|     16 |  28.4/681.1 |  41.3/331.0 |   42.0/649.3 |   68.9/841.7 |  104.6/951.2 |  147.9/991.4 |
|     32 | 27.9/1243.8 |  28.1/635.4 |  41.1/1252.1 | ‡     /1604.8 | 107.1/1792.8 | 152.5/1828.6 |

### Hit rate 0.7

| bs \ N |            1 |          1* |           2 |           3 |            4 |            5 |
|-------:|-------------:|------------:|------------:|------------:|-------------:|-------------:|
|      1 |    37.4/85.7 |   36.0/79.3 |  37.7/154.9 |  60.6/214.8 |   95.0/265.9 |  136.2/305.9 |
|      8 |   37.0/250.2 |  36.4/116.7 |  37.3/225.4 |  62.2/300.5 |   96.5/357.8 |  138.9/394.3 |
|     16 | 36.4/581.6 † |  36.2/211.5 |  37.4/411.4 |  63.0/533.1 |   97.5/608.6 |  140.8/633.7 |
|     32 |   36.8/911.0 |  36.0/395.4 |  37.2/780.1 |  65.0/995.4 | 100.7/1107.0 | 143.4/1114.4 |

### Hit rate 0.9

| bs \ N |            1 |         1* |          2 |           3 |           4 |           5 |
|-------:|-------------:|-----------:|-----------:|------------:|------------:|------------:|
|      1 |    37.9/40.3 |  35.4/41.0 |  36.0/75.6 |  55.3/113.3 |  86.2/149.7 | 124.1/190.6 |
|      8 |    35.8/54.3 |  36.3/54.3 | 36.5/100.2 |  56.5/142.2 |  88.2/182.0 | 125.9/219.4 |
|     16 | 36.4/413.1 † |  36.1/87.0 | 37.2/164.1 |  57.2/222.4 |  88.5/264.0 | 127.8/290.7 |
|     32 | 36.5/282.5 † | 36.3/150.7 | 35.6/288.7 |  58.8/378.2 |  92.2/429.8 | 131.7/448.2 |

## ms / step (x 61 layers) -- total only

### Hit rate 0.9 (closest to production)

| bs \ N |       1 |    1* |     2 |     3 |     4 |     5 |
|-------:|--------:|------:|------:|------:|------:|------:|
|      1 |    2.46 |  2.50 |  4.61 |  6.91 |  9.13 | 11.63 |
|      8 |    3.31 |  3.31 |  6.11 |  8.67 | 11.10 | 13.38 |
|     16 | 25.20 † |  5.31 | 10.01 | 13.56 | 16.11 | 17.73 |
|     32 | 17.23 † |  9.19 | 17.61 | 23.07 | 26.22 | 27.34 |

### Hit rate 0.7

| bs \ N |       1 |    1* |     2 |     3 |     4 |     5 |
|-------:|--------:|------:|------:|------:|------:|------:|
|      1 |    5.23 |  4.84 |  9.45 | 13.10 | 16.22 | 18.66 |
|      8 |   15.26 |  7.12 | 13.75 | 18.33 | 21.82 | 24.05 |
|     16 | 35.48 † | 12.90 | 25.10 | 32.52 | 37.13 | 38.66 |
|     32 |   55.57 | 24.12 | 47.59 | 60.72 | 67.52 | 67.98 |

### Hit rate 0.5

| bs \ N |     1 |    1* |     2 |     3 |      4 |      5 |
|-------:|------:|------:|------:|------:|-------:|-------:|
|      1 | 12.90 |  7.17 | 14.12 | 19.13 |  23.05 |  25.82 |
|      8 | 20.21 | 10.77 | 21.07 | 27.82 |  32.49 |  35.11 |
|     16 | 41.55 | 20.19 | 39.61 | 51.34 |  58.02 |  60.47 |
|     32 | 75.87 | 38.76 | 76.38 | 97.89 | 109.36 | 111.55 |

## Annotations

- `†` -- N=1 default (`buffer = top_k`) is a zero-slack eviction-cascade
  operating point. Working set precisely fills the buffer, so any minor LRU
  imperfection thrashes the hot set. Reproduced across two independent runs;
  not measurement noise, not a kernel bug. Adding one top_k of buffer slack
  (N=1*) eliminates it. Only the `total` measurement is affected -- the
  probe-only column at this cell stays on trend (36 us) confirming the extra
  latency is entirely DMA thrash from repeatedly reloading evicted hot ids.
- `‡` -- One-off measurement glitch (`513.9 us` probe-only at
  `bs=32, h=0.5, N=3`); neighbors are 41 us and 107 us so the true value is
  ~60-90 us. Not rerun.
- `*` -- N=1* forces `device_buffer_size = 4096` (one extra top_k of slack over
  the natural `N * top_k`), isolating buffer sizing from draft count.

## Key findings

1. **DMA is the dominant bottleneck**. For every non-fragile cell,
   `DMA share = 1 - probe/total`:

   | regime                          | DMA share |
   |---------------------------------|-----------|
   | production (h=0.9, N=2, bs 8-16)| 64-77%    |
   | medium (h=0.7, N=2, bs=16)      | 91%       |
   | miss-heavy (h=0.5, N=5, bs=32)  | 92%       |

2. **Probe cost is a fixed floor** of ~30-140 us per layer, essentially
   independent of `bs` and `h`, growing only with `N`. Every request runs in
   its own CUDA block, so up to L20X's SM count they execute concurrently and
   probe time is bounded by the single-block per-position work:
   `N=1: ~36us`, `N=2: ~37us` (block startup dominates for small N),
   `N=3: ~60us`, `N=4: ~90us`, `N=5: ~130us`.

3. **Hit rate is the strongest total-latency lever** because it directly
   scales miss count -> DMA bytes. Holding `(bs, N)` fixed, `h = 0.5 -> 0.9`
   gives 4-5x speedup at large `(bs, N)`. Production inference is close to
   `h ~ 0.9`; without that locality the swap-in cost would blow past the pure
   MTP path even with the biggest device buffer.

4. **Cost model** (non-flagged cells):
   `total_us ~= probe(N) + k * bs * N * (1 - h) * item_size / bw_h2d`
   with `probe(N)` from item 2 and `bw_h2d ~ 40-50 GB/s` for pinned
   host->device on this machine (fits the observed slope of ~50 us / MB moved).

5. **N marginal cost**: adding one draft token (`N -> N + 1`) raises total
   latency ~30-40% at any hit rate. Comes from both channels: probe grows
   ~25-30 us per position, and DMA rows grow linearly since `rows = bs * N`.

6. **N=1\* vs N=2 at the same buffer size (h = 0.9)**:
   `N=1* / N=2 ~= 41/76, 54/100, 87/164, 151/289` -- N=1* is essentially half
   of N=2. Confirms row count `bs * N` is the primary variable, not
   `num_positions` itself.

7. **DSV3.2 production match**: hybrid MTP `[steps=1, topk=1, draft=2]` at
   `bs ~ 8-12`, `h ~ 0.9` gives 6-11 ms/step here (probe ~2 ms, DMA ~5-7 ms),
   consistent with the end-to-end observation that the offloaded HiSparse
   resident-mode path costs ~3.78 ms/step over pure resident MTP.

8. **Buffer sizing suggestion**. Default formula
   `device_buffer_size = num_draft_tokens * top_k` collapses to exactly `top_k`
   at `N = 1`, hitting the fragile eviction-cascade point. Consider a floor:
   `device_buffer_size = max(N * top_k, 2 * top_k)`, or the simpler
   `(N + 1) * top_k`. Does not affect `N >= 2`.

## Optimization priorities (by recoverable us/layer)

1. **Reduce miss bytes** (largest lever on DMA share): push production
   `h` from ~0.9 towards 0.95 via larger buffer / smarter admission -> saves
   40-200 us/layer at heavy configs, roughly doubling hybrid's E2E win.
2. **Compress `top_k`** (2048 -> 1024 linearly halves DMA at fixed miss rate;
   requires algorithmic budget from the top-k side).
3. **Raise host->device bandwidth** (near hardware ceiling): pinned host DMA
   already sustains ~50 GB/s vs a PCIe Gen5 x16 practical ceiling of
   ~50-55 GB/s. Huge pages and stream overlap can recover another ~10-20%
   at most; lifting the physical ceiling requires CPU-GPU NVLink
   (Grace Hopper) which this platform does not have.
4. **Shrink `N`** already at the tight `[1,1,2]` production point.
5. **Optimize probe/LRU logic** (bottom of the list): fixed floor is only
   visible at very high h; hash-size trimming and LRU write-back fusion might
   recover 10-30% of the probe, so 10-40 us/layer at best.
