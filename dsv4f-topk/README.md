# DSv4f CSA Indexer Topk 分析报告

## 1. 实验设置

- **模型**: DeepSeek-V4-Flash (`/cpfs02/user/lgd/models/dsv4f/`)
- **硬件**: 4× NVIDIA GB200 (189GB each), TP4 + DP4
- **数据**: LongBench-v2 前 100 条 (prompt 11K–118K tokens, 中位 ~93K)
- **Decode**: `max_new_tokens=512`, `temperature=0`
- **DSv4 架构**: 43 层, 其中 21 层 C4 (compress_ratio=4), 20 层 C128 (compress_ratio=128), 2 层 SWA-only
- **Indexer 配置**: `index_topk=512`, `index_head_dim=128`, `index_n_heads=64`

### 抓取的数据 (per request)

| 数据 | 形状 | 大小 | 说明 |
|---|---|---|---|
| topk_index | (decode_tokens, 21, 512) int32 | ~22 MB | decode 阶段每步每层选中的 512 个 C4 压缩 token 位置 |
| indexer_k | (21, c4_len, 132) uint8 | ~69 MB | CSA indexer key cache (FP8 128维 + FP32 scale) |
| c4_kv | (21, c4_len, 584) uint8 | ~307 MB | C4 attention KV (FP8 nope 448维 + BF16 rope 64维 + scales) |

总计: 100 个 `.npz` (topk) + 100 个 `.pt` (KV), 共 ~27 GB, 存储在 `dsv4f-topk/`.

---

## 2. HiSparse 缓存命中率模拟

### 2.1 模型

严格复刻 `hisparse.cuh` 内核的 load/eviction 逻辑:
- **Per (request, C4 layer)**: 一个 `device_buffer_size = B` 个压缩 token 槽位的 **LRU 热缓冲** + 1 个保留槽钉住最新 token
- **Fast path**: 当压缩序列长度 ≤ B 时, 全部驻留 → 100% 命中
- **Slow path**: newest = 常驻命中; 其余位置命中当且仅当当前驻留; miss 在 load 前计数; LRU 驱逐
- **Init**: staging 路径 (first_b) = 前 B 个压缩 token 驻留

### 2.2 命中率 vs buffer size

默认 `device_buffer_size = 4096` 压缩 token (覆盖 16K full tokens):

| device_buffer_size | 覆盖 full tokens | 命中率 | 显存/req/layer | 显存/req (21L) |
|---|---|---|---|---|
| 1024 | 4K | 79.3% | 0.57 MB | 12 MB |
| 2048 | 8K | 88.3% | 1.14 MB | 24 MB |
| **4096** | **16K** | **92.5%** | **2.28 MB** | **48 MB** |
| 8192 | 32K | 94.8% | 4.56 MB | 96 MB |
| 16384 | 64K | 97.2% | 9.12 MB | 192 MB |
| 32768 | 128K | 100.0% | 18.25 MB | 383 MB |

显存基于 C4 KV 584 B/token/layer.

### 2.3 命中率 vs 请求长度 (B=4096)

| prompt 长度 | 请求数 | 命中率 |
|---|---|---|
| ≤16K | 9 | 99.99% (fast-path) |
| 16–32K | 19 | 97.0% |
| 32–64K | 10 | 93.8% |
| 64–128K | 62 | 89.3% |

### 2.4 逐层命中率

21 个 C4 层命中率范围: 88.1%–97.0%, 均值 92.5%, 各层较均匀.

---

## 3. 相邻 CSA 层 Index 相似性

### 3.1 方法

对每个 (request, decode token), 比较相邻 C4 层选中的 512 个 token 的集合相似度:
- **Overlap** = |A∩B| / 512 (共选比例)
- **Jaccard** = |A∩B| / |A∪B|

### 3.2 结果

| 指标 | 相邻层 (均值) | 随机基线 | 倍数 |
|---|---|---|---|
| Overlap | **34.9%** | 5.0% | **~7×** |
| Jaccard | **21.7%** | 2.6% | ~8× |

- 相邻层共享 ~35% 的选中 token (512 中约 178 个), 显著高于随机, 但**不足以无损共享**
- 早/中层更像 (36–46%), 深层发散 (19–29%)
- 随层距离衰减缓慢: dist=1→21.7%, dist=2→19.1%, dist=5→16.5%

### 3.3 含义

若像 V3.2 的 `index_topk_freq` 直接复用上层 topk, 会**漏掉 ~65% 的该层高分 token**.

---

## 4. 跨请求 (Inter-Request) Index 相似性

### 4.1 方法

对每个请求取最后一个 decode step、21 层并集的选中集合 S_r, 统计每个绝对位置被多少请求共选.

### 4.2 结果

- **Attention sink**: 位置 0–5 被 **100/100** 请求全选; 0–11 被 99–100%
- 在所有请求都够长的位置区 (i<2831): **18 个被 ≥90% 请求选中, 163 个被 ≥50%**
- **Recency**: 最新压缩 token 被 71% 请求选中
- **平均两两 Jaccard = 10.9%** (中位 9%) — 除 sink/recent 外几乎不重叠
- 大多数位置的共选数分布峰值在 ~25–40 个请求

### 4.3 含义

- 一小撮通用 sink (~6–18 个位置) 适合做全局常驻缓存
- 除 sink+recent 外, 工作集基本是 per-request 独有, 跨请求 KV 大范围复用没有数据支撑

---

## 5. Token 重排减少 Block Scatter

### 5.1 问题

当前 hisparse 是 **token 粒度** (`page_size=1`) 的随机访存. 每次 topk 选 512 个 token, 散布在 ~109 个 block (bs=64) 里, 理论最优只需 8 个. 若能把经常共选的 token 物理上放在连续 block 里, gather 就变成块状访存.

### 5.2 方法

6 种重排策略, 在**全部 100 请求 × 21 层 × 279K 样本**上评估:

| 方法 | 原理 | 可 online | 计算量 |
|---|---|---|---|
| original | 无重排 (baseline) | — | — |
| freq | 按选中频率降序, 热 token 放前面 | ✅ prefill 后统计 | O(n log n) |
| kmeans_idx_k | K-means on indexer K embedding | ✅ prefill 后 | O(n·k·iter) |
| kmeans_c4_kv | K-means on C4 attention KV embedding | ✅ prefill 后 | O(n·k·iter) |
| spectral_cooc | 共选矩阵谱聚类 | ❌ 需 decode 历史 | O(nnz + eigsh) |
| greedy_cooc | 贪心共选 block-filling | ❌ 需 decode 历史 | O(nnz · n/bs) |

#### 5.2.1 spectral_cooc (共选矩阵谱聚类)

**核心思想**: 把"哪些 token 经常被同一次 topk 一起选中"建模为图, 用谱聚类找到紧密共选的 token 社区, 同社区放同 block.

**具体步骤**:

1. **构建共选矩阵 W**: 遍历所有 decode step 的 topk 选择. 对每步选中的 512 个 token, 所有 pair (i,j) 的 W[i,j] += 1. W 是 (n_tokens × n_tokens) 的稀疏对称矩阵, 表示"token i 和 j 被同一步 topk 共同选中的次数".
2. **归一化 Laplacian**: L = I - D^{-1/2} W D^{-1/2}, 其中 D 是度矩阵. 归一化消除不同 token 选中频率差异的影响.
3. **特征分解**: 用 ARPACK (`scipy.sparse.linalg.eigsh`) 提取 L 最小的 32 个特征向量. 每个 token 得到一个 32 维的**谱嵌入向量** — 在这个空间里, 经常共选的 token 距离近.
4. **K-means 聚类**: 在谱嵌入空间做 K-means (k = n_tokens / block_size), 把谱空间邻近的 token 分到同一 cluster.
5. **重排**: 同 cluster 的 token 按原始位置排序后连续放置 → 物理上同 block.

**为什么不能 online**: 步骤 1 的共选矩阵 W 需要**所有 decode step 的 topk 选择结果**. 在 prefill 结束时, 还没有任何 decode step 发生, 无法知道哪些 token 会被一起选中. 共选关系本质上是 decode 过程中 query 与 key 交互的结果 — query 取决于生成的 token, 这在 prefill 阶段不可预知.

#### 5.2.2 greedy_cooc (贪心共选 block-filling)

**核心思想**: 逐块贪心填充 — 每个 block 从最热的未分配 token 开始, 然后反复挑选与已有 block 成员共选亲和度最高的 token 加入, 直到 block 填满.

**具体步骤**:

1. **构建共选矩阵 W**: 同 spectral_cooc, 稀疏矩阵 W[i,j] = token i,j 被同步选中次数.
2. **选 seed**: 从所有未分配 token 中选频率最高的作为新 block 的种子.
3. **贪心填充**: 计算所有未分配 token 与当前 block 中已有 token 的**共选亲和度总分** (score = W[candidate, block_members].sum()), 选得分最高的加入 block. 重复直到 block 填满 block_size 个 token.
   - 实现上用 `torch.sparse.mm(W, indicator_vector)` 做 GPU 加速的稀疏矩阵-向量乘, 一次 matvec 算出所有候选的 score, 然后取 top-k 一次填满整个 block.
4. **重复**: 开始下一个 block, 直到所有 token 都被分配.

**与 spectral 的区别**: spectral 用全局最优的谱分解找 cluster, greedy 是局部最优的逐块填充. greedy 直接优化"block 内共选密度", 因此在 block 粒度的指标上通常更好; spectral 找的是全局社区结构, cluster 大小不一定对齐 block_size.

**为什么不能 online**: 同样依赖共选矩阵 W, 需要 decode 历史. 原因完全相同 — prefill 阶段不知道哪些 token 会被未来的 query 一起选中.

#### 5.2.3 为什么 freq 可以 online

频率排序只需要知道"每个 token 被选中多少次", 而不需要知道"哪些 token 被一起选中". Token 的选中频率主要由其 key embedding 的"显著程度"决定 — 这是 token 本身的静态属性, 在 prefill 后就可以通过一轮 dummy indexer forward 或启发式 (如 key norm / 与均值的距离) 近似估计, 无需真实 decode.

实践中更简单的做法: prefill 后有第一个 decode step 的 topk 选择, 用这一步的选中情况作为频率的初始估计 (假设不同 step 的选择有很强相关性, 之前的分析证实了这一点), 即可完成排序.

### 5.3 结果

**Mean blocks touched per topk selection (lower = better):**

| 方法 | bs=16 | bs=32 | bs=64 |
|---|---|---|---|
| **original** | 199.5 | 150.5 | 108.5 |
| **freq** | 142.1 (-29%) | 89.9 (-40%) | 54.1 (-50%) |
| kmeans_idx_k | 312.2 (+56%) | 243.8 (+62%) | 173.2 (+60%) |
| kmeans_c4_kv | 321.6 (+61%) | 251.4 (+67%) | 177.9 (+64%) |
| **spectral_cooc** | 118.5 (-41%) | 83.5 (-45%) | 57.2 (-47%) |
| **greedy_cooc** | **105.3 (-47%)** | **65.6 (-56%)** | **40.2 (-63%)** |
| oracle | 32.0 | 16.0 | 8.0 |

**Scatter (×oracle, lower = better):**

| 方法 | bs=16 | bs=32 | bs=64 |
|---|---|---|---|
| original | 6.2× | 9.4× | 13.6× |
| freq | 4.4× | 5.6× | 6.8× |
| spectral_cooc | 3.7× | 5.2× | 7.2× |
| greedy_cooc | **3.3×** | **4.1×** | **5.0×** |

### 5.4 核心发现

1. **基于 embedding 的聚类 (K-means) 完全失败**: 无论 indexer K 还是 C4 KV, embedding 近邻 ≠ 共选关系 — 相似 token 互相竞争 topk 名额, 放在一起反而更散. 比不重排还差 56–67%.

2. **基于共选关系的方法有效**:
   - **贪心共选** (最优): bs=64 下 -63%, scatter 从 13.6× 降到 5.0×
   - **谱聚类**: bs=16/32 下接近贪心, bs=64 下略逊
   - 但两者都需要 decode 历史, **不能 online 使用**

3. **频率排序是最佳 online 方案**: bs=64 下 -50%, scatter 6.8×. 零额外信息, prefill 后 O(n log n) 完成.

4. **block 越大, 频率排序相对越好**: bs=64 时 freq (-50%) vs greedy (-63%) 差距 13pp; bs=16 时差距 18pp. 大 block 对简单热 token 前置策略更友好.

5. **即使最好的静态重排, scatter 仍 ~5×**: 40 blocks vs oracle 8 (bs=64). 不同 decode step 的 topk 选择变化使静态重排无法完美. 要进一步逼近 oracle 需动态重紧缩.

### 5.5 用 prefill window 的 topk 构建 W 能否替代 decode 历史?

**动机**: Indexer 在 prefill 的每个 chunked extend 也运行并产生 topk (`forward_c4_indexer` 不区分 extend/decode). 如果用 prefill 最后一个 window 的 topk 构建共选矩阵 W, spectral/greedy 就变成 online 可行的.

**实验**: 分别用 1/2/4/8/16/32/全部 decode steps 的 topk 构建 W → greedy 重排 → 在**全部 steps** 上评估 block coverage (bs=64, 100 req × 21 layers, 279K samples):

| 构建 W 用的 step 数 | blocks | 减少 | scatter |
|---|---|---|---|
| freq (无需 W) | 54.1 | -50% | 6.8× |
| greedy (1 step) | 90.8 | -16% | 11.3× |
| greedy (2 steps) | 86.8 | -20% | 10.9× |
| greedy (4 steps) | 82.5 | -24% | 10.3× |
| greedy (8 steps) | 78.5 | -28% | 9.8× |
| greedy (16 steps) | 74.0 | -32% | 9.2× |
| greedy (32 steps) | 69.9 | -36% | 8.7× |
| greedy (all steps) | 40.2 | -63% | 5.0× |

**结论: 不够用.** 1 step 的共选矩阵只减少 16% (远不如 freq 的 50%), 因为一步只有 512×512 个共选 pair, 矩阵太稀疏. 即使 32 步 (-36%) 仍不如 freq (-50%). Greedy 超越 freq 需要**完整 decode 历史** (all steps, -63%), 这从定义上不是 online 的.

**根本原因**: 共选关系是 query-dependent 的 — 不同 decode step 的 query 不同, 选中的 token 集合变化很大 (之前的相邻层分析显示相邻 step 的 overlap 只有 ~35%). 少量 step 的 W 无法泛化到未来 step 的选择模式. 而频率是 token 本身的"显著度", 与具体 query 无关, 因此只需 prefill 阶段的一次统计就够.

### 5.6 落地建议

**短期 (可立即实现)**:
- 频率排序: prefill 后统计每个 C4 token 被选中的频率, 按频率降序重排 KV pool
- block_size=64 (对齐 C4 pool 物理 page_size=64), block-level load/evict
- 预期收益: 访存 block 数减少 50%, 且每个 block 内连续可向量化

**中期**:
- 贪心共选排序需要完整 decode 历史, 实际不可用于 online; 但可用于离线 profiling 建立 benchmark 上界
- 探索基于 indexer K 的**非 embedding-similarity** 特征 (如 key norm、与均值的角度) 做排序, 可能比纯频率更好

**长期**:
- 动态 block compaction: 每 N 步根据最近 topk 模式重紧缩
- 改 kernel 做 block-level gather with indirection (page table 重映射, 不物理搬数据)

---

## 6. 数据产物清单

| 文件 | 说明 |
|---|---|
| `dsv4f-topk/req_XXXX.npz` | 100 个 topk index (decode-only) |
| `dsv4f-topk/kv_dump/req_{rid}.pt` | 100 个 C4 KV + indexer K (全序列) |
| `dsv4f-topk/manifest.jsonl` | 请求元数据 |
| `dsv4f-topk/hisparse_sim_results.json` | hisparse 命中率模拟 |
| `dsv4f-topk/hitrate_by_len_and_buffer.csv` | 命中率 vs 长度 × buffer size |
| `dsv4f-topk/hitrate_by_buffer.png` | 命中率可视化 (2×3 子图) |
| `dsv4f-topk/layer_topk_similarity.json/png` | 相邻层相似性 |
| `dsv4f-topk/inter_request_coselection.json/png` | 跨请求共选 |
| `dsv4f-topk/reorder_all_methods_full.json` | 6 种重排方法对比 |

---

## 7. 分析脚本清单

| 脚本 | 说明 |
|---|---|
| `run_dsv4f_indexer_capture.py` | 客户端: 发 100 条请求, 收 topk index |
| `sim_hisparse_hitrate.py` | hisparse LRU 缓存命中率模拟器 |
| `analyze_reorder_feasibility.py` | 重排可行性初步分析 |
| `analyze_reorder_all_full.py` | 6 种重排方法全量对比 (GPU 加速) |
| `analyze_kmeans_reorder.py` | K-means 重排分析 |

### Runtime 改动 (可 revert)

`python/sglang/srt/managers/scheduler_components/batch_result_processor.py`:
- `_maybe_collect_indexer_topk`: 加 `start_len=prompt_len-1` (decode-only topk)
- `_maybe_dump_kv_to_disk`: 新增, 读 `SGLANG_DUMP_KV_DIR` 环境变量, 为完成的请求 dump C4 KV + indexer K 到磁盘
