# HiSparse with MTP 实现文档

## 概述

在 MTP（EAGLE speculative decoding）的 target verify forward 中使用 HiSparse 的稀疏注意力。KV cache 在 prefill 后 offload 到 host（staging），device 端每请求只保留**固定大小**的两块物理区域：

- **device buffer**（`padded_buffer_size` 槽位，LRU 热缓存）：verify top-k 命中的历史 KV 经 host DMA 换入此处；
- **speculative staging ring**（`spec_ring_capacity` 槽位）：speculative tokens 的 KV 直接写入环内槽位，按 `position % R` 复用。

verify 时 MTP swap-in kernel 把 fused topk 输出的 top-k logical ids 解析成 hisparse device indices，作为 page_table_1 直接传给 attention kernel。物理占用与生成长度无关（buffer + ring 固定），committed KV 的常驻地是 host pool。

**核心思路**：

1. Fused topk 产出 logical slot indices（page-table-transformed）
2. MTP swap-in kernel 对每个 top-k entry 做三级解析（无 fast path，所有 seq_len 同一路径）：
   - **direct_loc**：`full_to_hisparse_device_index_mapping[slot_idx]` > 0 → 直接返回（buffer 驻留 token 或 ring 内 speculative/近期 committed token）
   - **buffer matching**：在 `device_buffer_logical_tokens` 中哈希匹配 → 返回对应 `device_buffer_locs`
   - **host DMA**：miss → evict LRU slot，经 `full_to_token_position` 反查 host 位置，复制 KV 到 device → 返回被复用的 slot
3. Kernel 一次处理所有 N 个 verify positions（`num_draft_tokens=N`），每个 position 在同一个 thread block 内串行处理，共享 LRU 状态

## 架构

```
Prefill (combined alloc: logical + hisparse physical pages)
   ↓
eagle_worker staging (admit):
   ├── offload KV to host (staging DMA; surplus pages deferred-free)
   ├── alloc device buffer (full padded_buffer_size; prefill pages consumed in place)
   └── alloc staging ring (R slots, page-aligned, request-private)
   ↓
MTP decode loop (per round):
   ├── eagle_prepare_for_decode:
   │     ├── backup last round's committed KV → host (reads ring slots)
   │     ├── logical-only alloc for spec slots (global physical pool untouched)
   │     └── assign_spec_ring_slots: mapping[new slot] = ring[pos % R];
   │         clear mapping of recycled position pos-R (its KV is already on host)
   ├── draft (N steps, draft model's own KV pool, draft CG)
   ├── verify (target-verify CG):
   │     ├── KV write: set_mla_kv_buffer translates via mapping → lands in ring slots
   │     └── attention: fused topk → MTP swap-in kernel → flash_mla_sparse
   └── accept + commit (bookkeeping only: kv_committed_len += accepted)
   ↓
request_finished: free buffer + ring pages (page-granular, deduped)
```

### 每请求 KV 生命周期

| 阶段 | 物理位置 | mapping 状态 |
|------|---------|-------------|
| prefill token（≤ buffer 容量部分） | device buffer 槽位 | 有效（direct_loc） |
| prefill token（超出部分） | host pool | 清零 → 按需 DMA 换入 buffer |
| speculative / 近期 committed token（窗口 R 内） | ring 槽位 | 有效（direct_loc） |
| committed token（backup 后、被 ring 回收） | host pool | 清零 → 按需 DMA 换入 buffer |
| rejected token 的 logical 槽位 | 下一轮被新 draft 复写 | 随位置复用被覆盖 |

### Verify 数据流

```
N draft positions per request (target-verify CG, replayed per round)
        ↓
KV write: set_mla_kv_buffer(out_cache_loc)
        └─ translate via mapping[logical slot] → KV lands in ring slots
        ↓
indexer + fused topk → page_table_1 [bs*N, top_k] (logical slot indices)
        ↓
swap_in_verify_pages (MTP kernel, ONE launch for all N positions;
                      positions run serially in one thread block, shared LRU)
        ↓
   ┌─ direct_loc:    mapping[slot_idx] > 0 → buffer / ring slot
   ├─ buffer match:  hash lookup in device_buffer_logical_tokens
   │                 → device_buffer_locs[slot]
   └─ host DMA:      evict LRU slot → full_to_token_position[slot_idx]
                     → host_cache_locs[rid][pos] → memcpy host→device
                     → reuse evicted slot
        ↓
page_table_1 [bs*N, top_k] (hisparse device indices; -1 for padding)
        ↓
flash_mla_sparse attention
```

### 非 verify 路径（decode / extend）

```
decode（非 spec）: force_unfused topk → raw positions → swap_in_selected_pages（decode kernel，按位置身份匹配）
extend/prefill: fused topk → page_table_1 → translate_loc（mapping 查找；prefill 时 mapping 恒有效）
```

两套 kernel 的 token 身份键不同：decode kernel 用**请求内位置**（host 表按位置索引，不依赖 mapping），MTP kernel 用 **logical slot id**（依赖 mapping + `full_to_token_position` 逆表）。MTP 模式下 target 模型只跑 prefill/verify，两条路径不会混用。

## 关键设计决策

### 为什么 verify 需要 swap-in kernel，不能只用 translate_loc

当 `seq_len > device_buffer_size` 时，`alloc_device_buffer` 只保留 buffer 内 slots 的 mapping，超出部分 mapping 被清零。translate_loc 对这些 token 返回 0 → attention 读错 KV。

实测数据（seq_len=8511, buffer_size=4096）：
```
[HISPARSE-VERIFY] valid=8192 mapped=4256 unmapped=3936   ← 48% top-k 缺失
```

MTP swap-in kernel 通过 host DMA 加载缺失的 KV，解决了这个问题。

### 为什么用 MTP kernel 而不是 decode kernel

| Aspect | decode kernel | MTP kernel |
|------|--------------|------------|
| `direct_loc` | ❌ N/A | ✅ Yes |
| New decode tokens | In buffer (tracked by buffer_tokens) | Valid mapping → direct_loc skips DMA |
| Host entries | All tokens have entries | New tokens have no host entry → decode kernel crash |
| topk format | raw positions | physical slot indices (fused topk) |

Decode kernel 对新 decode tokens（host entry 未初始化）做 DMA 会 `illegal memory access`。MTP kernel 的 `direct_loc` 检测到新 token 有 valid mapping，跳过 DMA。

### 为什么不 per-position 调用 swap-in

每次 swap-in 修改 LRU buffer。Position 1 可能 evict position 0 load 的 token，导致 position 0 的结果指向错误的 KV。

解决方案：一次调用处理所有 N positions（`num_draft_tokens=N`）。Kernel 内部串行循环，同一 thread block 共享 LRU。只要 `device_buffer_size >= top_k * N`（最坏情况 N positions 完全不重叠），不会发生 inter-position eviction。实际上 N 个 position 选自同一 committed context，重叠率 >90%。

### device_buffer_size 约束

`device_buffer_size >= top_k * num_draft_tokens`（例如 2048 × 4 = 8192）。这保证了 kernel 内部 N 个 position 不会互相 evict。实际上由于高重叠率，更小的 buffer 也可能工作，但缺乏保证。

## 改动清单

### 1. `jit_kernel/csrc/hisparse.cuh` — MTP swap-in kernel

**改动**：
- `load_cache_to_device_buffer_mtp_kernel`：添加 `full_to_hisparse_device` 参数用于 direct_loc
- **删除 fast path**（原 `seq_len <= HOT_BUFFER_SIZE` 时只做 direct_loc + buffer 线性扫描、不做 host DMA 的分支），所有 seq_len 统一走 hash + eviction + host DMA 的 slow path
- Slow path：direct_loc > 0 的 token 标记为 `TOKEN_HIT` 跳过 hash table matching

**为什么必须删除 fast path**：staging ring 会周期性清零已 backup token 的 mapping（见下文）。这些 token 既无 direct_loc、也不在 buffer 逻辑 token 表中，唯一的取回途径是 host DMA。fast path 没有 DMA 能力，会对这类 token 静默输出 -1，attention 直接丢弃对应 KV——这正是早期实验中"清零 mapping 后 CG 输出质量退化"的真实根因（短序列必走 fast path，退化最明显；当时被误判为 CG 与 host DMA 的交互问题）。删除后由单测 `test_mtp_short_seq_recycled_mapping_resolves_via_host_dma` 锁定该行为。

**连带约束**：slow path 的 eviction 可能选中任意 LRU slot 作为 DMA 目的地，因此 spec 模式下 device buffer 必须全量分配（见改动 9），否则未分配 slot 的 loc=-1 会造成越界写。

### 2. `jit_kernel/hisparse.py` — Python wrapper

**改动**：
- 移除 `assert num_draft_tokens == 1` 限制
- 修正 shape 校验：`top_k_tokens.shape[0] == bs * num_draft_tokens`

### 3. `layers/attention/dsa_backend.py` — verify swap-in 路径

**改动**：
- `hisparse_verify` 分支：fused topk → page_table_1 后调用 `swap_in_verify_pages`，kernel 输出直接作为 page_table_1
- 非 verify 的 hisparse 路径（extend/prefill）保持 `translate_loc`
- `_forward_hisparse_target_verify` 方法已删除（之前的 per-position attention 有 metadata 不匹配 bug）

**翻译 gate 修复（关键 bug）**：extend/prefill 的 hisparse 翻译条件必须用 `self.hisparse_coordinator`（backend 持有，恒真），而不是 `forward_batch.hisparse_coordinator`（仅 decode/verify batch 才会附加，prefill 时为 None）。曾误用后者导致 prefill 的 page_table 完全跳过 logical→physical 翻译，attention 直接拿逻辑 id 当物理 id 读 KV。

**为什么此 bug 长期未暴露**：单请求串行时，logical 与 hisparse 两个 paged allocator 从同一初始状态做完全相同的分配序列（`host_to_device_ratio=1` 时两池同大小），逻辑 id 恰好等于物理 id——恒等映射掩盖了缺失的翻译。一旦出现并发（第二个请求 prefill 时两池分配历史已错位）或引入 staging ring（logical-only 分配打破 lockstep），恒等性即被破坏，表现为"某个请求从第一个 token 起输出乱码/答非所问"。通过 baseline worktree A/B 对照 + eager/CG/overlap 三维二分定位。

### 4. `managers/hisparse_coordinator.py` — swap_in_verify_pages + staging ring

**改动**：
- 新增 `swap_in_verify_pages` 方法：传 `num_positions=N` 给 MTP kernel，使用预分配的 `verify_locs_buf` 作为输出
- `verify_locs_buf`：`[max_bs * max_N, top_k]` 预分配 tensor，CG-static
- 新增 staging ring 状态与方法（`spec_ring_capacity > 0` 时启用）：
  - `req_to_spec_ring [max_reqs, R]`：每请求的 ring 物理 indices 表；`req_spec_ring_active`：CPU 侧 active 标记
  - `_alloc_spec_ring(req)`：admit 时一次性从 `hisparse_attn_allocator` 分配 R 个 page 对齐槽位；幂等（重复 admit 安全）
  - `assign_spec_ring_slots(req, start, end)`：每轮 decode 把新 logical 槽位的 mapping 指向 `ring[pos % R]`，同时清零被回收位置（pos-R）的 mapping，并注册 `full_to_token_position`（合并了原 `register_token_positions` 循环）
  - `_free_spec_ring(req)`：`request_finished` 时取回 ring indices 与 buffer 页合并 unique 去重后统一按页释放
- 清理 `swap_in_selected_logical_pages` 中的调试日志

**必要性**：
- **`_hisparse_ring_start` 回收下界**：回收范围必须钳在首个 ring 分配位置之上，否则 `start-R` 会落入 prefill 区间、误清 prefill token 的 mapping——这些 token 的物理页由 mapping 枚举回收，误清会导致 `request_finished` 漏释放（单测 `test_spec_ring_wrap_recycles_mapping` 曾捕获此泄漏）。
- **`end - R ≤ last_backed` assert**：ring 槽位复用的前提是被回收 token 的 KV 已在 host（backup 在同函数先于 alloc 执行）。违反即静默读错 KV，因此 fail-fast。
- **mapping 读写全部在图外**：CG replay 时 kernel 读 mapping tensor 的当前内容，与非 spec decode 的 `map_last_loc_to_buffer` 同一套已验证模式；写入位置与 `assign_req_to_token_pool_func` 相邻，继承既有 overlap WAR barrier 的流序保证。

### 5. `speculative/eagle_utils.py` — CG 启用 + ring 集成

**改动**：
- 移除 `can_run_cuda_graph = False`，verify 不再禁用 CG
- `eagle_prepare_for_decode`：hisparse 模式下把 `alloc_for_spec_decode` 包在 allocator 的 `spec_logical_alloc()` 作用域内（spec 槽位只发放 logical 索引），随后逐请求调用 `assign_spec_ring_slots(req, start, end)`（替代原 `register_token_positions` 循环）
- `_do_hisparse_backup`：`torch.cuda` → `device_module`（ROCm 兼容），删除未使用的 `all_logical_locs`

**必要性**：
- **logical-only 分配**是 ring 方案的核心切换点：spec tokens 不再向全局 `hisparse_attn_allocator` 申请物理页，长生成的物理占用因此不随轮次增长。
- **顺序不变量**：函数内先 `_hisparse_backup_new_tokens`（读旧 mapping 把新 committed KV 备份到 host），后分配 + `assign_spec_ring_slots`（回收旧 mapping）。颠倒顺序会使 backup 读到已清零的 mapping。

### 6. `model_executor/model_runner.py` — CG 启用 + ring 容量

**改动**：
- 移除 verify 时 `can_run_graph = False`
- 移除冗余的 `num_real_reqs.fill_`（已在 CG runner 的 `load_batch` 中处理；eager 路径由 `_prepare_eager_forward_batch` 覆盖）
- 计算 `spec_ring_capacity = page_align(4 × get_alloc_reserve_per_decode())` 并传入 coordinator（仅 spec 模式非零）

**容量推导**：位置 p 的 ring 槽位在分配位置 p+R 时被回收，安全条件为 p 已 backup。每轮分配上界为 `reserve = 2 × alloc_len_per_decode`（含 overlap 下 `kv_committed_len` 滞后的双缓冲），故 R ≥ 2×reserve 是硬下限；取 4× 留余量，内存代价每请求仅几十个槽位。

### 7. `model_executor/runner/decode_cuda_graph_runner.py` — CG capture

**改动**：TARGET_VERIFY 的 CG capture 也 attach `hisparse_coordinator`（之前跳过设为 None）

### 8. `allocator/hisparse.py` — spec logical-only 分配作用域

**改动**：
- 新增 `spec_logical_alloc()` 上下文管理器：作用域内 `alloc` / `alloc_extend` 只发放 logical 索引，分别路由到已有 `alloc_logical_only`（paged）与新增 `alloc_spec_logical`（page_size==1）。由 `eagle_prepare_for_decode` 包裹 `alloc_for_spec_decode` 进入，因此 `mem_cache/allocation.py` 与上游保持**字节一致**——不需要 `logical_only` kwarg，也不触碰 NPU 变体的签名。
- `alloc_device_buffer` 恢复 buffer 内 slots 的 mapping（direct_loc 依赖）；分配失败时回滚 mapping 并抛 `RuntimeError`（原为 assert，失败后 mapping 已清、请求无法安全释放）
- **cleared tail 守卫**：`alloc_extend` 中若某行的物理尾 mapping 已被清零（backup / ring 回收）却仍在 extend，paged allocator 会把 partial-page tokens 写到 `last_loc+1`——一个本请求不拥有的物理槽位（静默覆盖他人 KV）。last-chunk admission gate 保证这不会发生，因此实现为 **fail-fast 断言**（而非重建"全新序列"几何、容量不足返回 None）。断言只依赖 `last_loc + mapping`，放在 logical 分配**之前**，触发时不会泄漏已认领的 logical 页。回归测试：`test_alloc_extend_cleared_tail_fails_fast`。

**必要性**：spec 槽位的物理来源改为 ring 后，logical 侧仍需正常的 paged 分配与 OOM/retract 处理。复用既有 helper 链路（而非旁路重写）保证失败路径行为与其它模式一致。

### 9. Spec 模式下 device buffer 全量分配（`hisparse_coordinator.alloc_device_buffer`）

**改动**：`spec_ring_capacity > 0` 时 admit 按 `padded_buffer_size` 全量分配（非 spec 模式仍按 prefill 长度页对齐按需分配）。

**必要性**：改动 1 删除 fast path 后，verify 统一走 eviction slow path，任意 LRU slot 都可能被选为 host DMA 的目的地。若 buffer 只部分分配，未分配 slot 的 `device_buffer_locs = -1`，DMA 会向 `buffer[-1]` 越界写。代价：每请求固定占用 `padded_buffer_size` 物理槽位（这本就是 `device_buffer_size ≥ top_k × N` 约束下的语义需求）。

### 10. `scheduler_components/batch_result_processor.py` — prefill 即完成请求的资源释放

**改动**：
- 双重 staging guard（`req_device_buffer_size == 0` 才 admit）
- **prefill 完成路径补 `hisparse_coordinator.request_finished(req)`**（在 `release_kv_cache` 之前）

**必要性**：eagle_worker 在 prefill 输出时就 eager admit（分配全量 buffer + ring）。在 prefill 阶段即结束的请求（典型如每 30s 一次的 `/health_generate`，max_new_tokens=1）原先只走 `release_kv_cache`，不经过任何 hisparse 释放路径。后果有二：(a) 每个此类请求永久泄漏 `padded_buffer_size + R` 物理槽位（实测 token usage 每次 +0.02 单调爬升）；(b) req_pool_idx 复用时 `req_device_buffer_size != 0` 使新请求跳过 admit，继承前一请求的 buffer/mapping 脏状态，输出乱码。修复后 usage 在请求间回落至 0.00。

### 11. 其他保留的改动

- **`speculative/eagle_worker_v2.py`**：prefill 结束时触发 staging（overlap 下 process_batch_result 见到的 forward_mode 已被改写为 DECODE，无法在 scheduler 侧触发）
- **`managers/scheduler.py`**：HiSparse+MTP 模式每轮设 `running_batch.hisparse_coordinator`。**附加点必须在 last_batch merge 之后**（review b5c4b714e2 P1）：空 `running_batch` 会被 `running_batch = last_batch` 整体替换，动态属性随之丢失——idle→首轮 decode 时 coordinator 为 None，`eagle_prepare_for_decode` 会静默回退 combined 物理分配（spec KV 落到 LRU 管理的 buffer 槽位、可能被 evict 读错，且破坏固定占用）。已用 fail-fast 探针复现（8 个 TP rank 的 warmup 首轮 decode 全部命中）；该探针作为永久防线保留在 `eagle_prepare_for_decode`（hisparse 开启而 batch 无 coordinator → RuntimeError）。
- **`forward_batch_info.py`**：`hisparse_coordinator` 作为 ForwardBatch 字段

### 12. 启动校验与 admission 约束（review 修复）

**spec 算法白名单**（`arg_groups/hisparse_hook.py`）：HiSparse 的 admit hook（staging DMA + buffer + ring 分配）位于 `EAGLEWorkerV2.forward_batch_generation`。覆写或绕过该方法的 worker（multi-layer EAGLE、FROZEN_KV_MTP、NGRAM、DFLASH、DSPARK、自定义插件）不会 admit 请求，首轮 decode 会在未分配的 ring 上 assert（DFLASH 族甚至完全不走 `assign_spec_ring_slots`）。启动时白名单校验：仅允许 `EAGLE` / `EAGLE3` / `STANDALONE` 且非 `--enable-multi-layer-eagle`（NEXTN 在校验前已别名解析为 EAGLE）。

**并发上限钳制**（`model_executor/model_runner.py`）：每个 admit 的请求占用固定物理成本 `device_buffer_size + page_size + spec_ring_capacity`,而 PrefillAdder 只按 token 计费,不感知这笔成本。启动时按 `(hisparse_pool - reserved_page0) // per_req_cost` 钳制 `max_running_requests`(该值随后进入 scheduler 的 admission 预算),避免 prefill 成功后才在 `alloc_device_buffer` / `_alloc_spec_ring` 抛 `RuntimeError`。池连一个请求都放不下时启动即报错。

**staging 瞬时峰值回收**（`hisparse_coordinator._reclaim_deferred_staging_pages`）：并发 cap 只保证稳定态占用,不覆盖 staging 过程的瞬时峰值——长 prompt 的全部 prefill 物理页要等 staging DMA event 完成才释放 surplus,而 buffer 补充/ring 分配在 event 之前就执行,单个接近池容量的长 prompt 即可令 `_alloc_spec_ring` 失败(已用单测复现)。修复:admit 时的 buffer/ring 分配失败后,同步在飞 staging event、释放所有 deferred surplus 页并重试一次;仅在压力路径付出同步代价。回归测试:`test_admit_long_prompt_reclaims_deferred_pages_for_ring`。

**admission gate 钳制**（`scheduler.get_num_allocatable_reqs`）：请求数 gate 原来直接读 `pp_max_micro_batch_size`,默认值恰好等于 `max_running_requests // pp_size`,但用户显式设置更大值时(即使 `pp_size=1`)可绕过并发 cap。gate 改为恒取 `min(pp_max_micro_batch_size, max_running_requests)`,不再依赖默认值巧合。

**device_buffer_size 页对齐**（`sparsity/factory.py`）：`alloc_device_buffer` 断言 need_size 页对齐,而 `padded_buffer_size = device_buffer_size + page` 不改变余数——非对齐配置(如 8193)原来能通过解析、在首次 staging 才断言。现在解析时向上取整到 KV page 并告警,使并发预算、coordinator 元数据与 kernel 参数统一使用对齐值。

**invalid-tail OOM 语义说明**:容量不足返回 None 后,上层 `alloc_paged_token_slots_extend` 抛标准 "Prefill out of memory"——与所有后端的分配器耗尽行为一致(该异常路径不做 retract/req-slot 回收是既有框架性质,非本分支引入)。且该分支当前不可达:combined `alloc_extend` 仅在 prefill 调用,行尾 mapping 只能是 -1 哨兵(新序列)或有效值(chunked 续传先于 staging/ring 回收),不会为 0;保留为防御性代码。**热路径开销已消除**(review b5c4b714e2 P2):`invalid.any()` 的 GPU→CPU 同步现在由免费的 CPU 检查门控——被清尾 mapping 必然要求 prefix>0,而 radix 禁用下几乎所有 prefill 行 prefix=0,`(prefix_lens_cpu > 0).any()` 为假时完全跳过同步。

**DSV4 + speculative 启动拒绝**（review b5c4b714e2 P2）：coordinator 对 DSV4 + staging ring 有 init 断言,但参数校验原来放行该组合(deepseek_v4_hook 允许 EAGLE),表面合法的配置在较晚的初始化阶段以非用户友好的断言崩溃。现在 `validate_hisparse` 在参数阶段明确拒绝 DSV4 + `--speculative-algorithm`,给出可操作的报错。

**死代码清理**（review b5c4b714e2 P3）：删除 `_hisparse_backup_accepted`（无调用方、import 不存在的模块、host 长度计算有误的失效副本）、`collect_ready_reqs_blocking` 与 `swap_in_selected_logical_pages`（仅测试使用的生产 API——单测已迁移到生产路径 `swap_in_verify_pages(num_positions=1)` 与测试侧本地 helper）。复现脚本 `e2e_ring_test.py` 已纳入版本控制。

### 13. `mem_cache/kv_cache_configurator.py` — draft KV pool 扁平化到逻辑空间（长上下文 IMA 修复）

**问题**：draft worker（NextN）通过 `alloc_memory_pool` 复用 target 的 allocator 与 `req_to_token`,因此它拿到的 `out_cache_loc` 是 allocator 在 **h2d 扩展后的逻辑 id 空间**里分配的。但 draft 的 HiSparse KV pool 构造时 `size = max_total_num_tokens`(**物理** device 大小),而 `index_buf_size = size × h2d`(**逻辑** 大小)——同一个 pool 里 KV buffer 按物理开、indexer/topk 输出按逻辑开,两者地址空间不一致。target 靠 backend 层的 page-table 翻译(`hisparse_coordinator is not None` 门控)把二者对齐;**draft 没有 coordinator,读路径不翻译**,于是它的 KV 写/读都用裸逻辑 id 索引只有物理大小的 KV buffer:

- 逻辑 id < 物理大小:自洽但写在与 target 无关的槽(draft 自读自写,只是占用物理下半区)。
- 逻辑 id ≥ 物理大小(h2d=2 时逻辑空间是物理 2 倍,30k prompt × 并发必然触及上半区):**IMA 崩溃**。

**定位过程**:表层栈在 `collect_ready_reqs → free_hisparse_indices`,但 IMA 为异步上报;`CUDA_LAUNCH_BLOCKING=1` + 16 并发 30k 定位到真实位置 `set_mla_kv_buffer.cuh:226`,调用路径 `_draft_extend_for_prefill`。范围守卫捕获到越界物理值**从 `pool_size` 起连续递增**(= 逻辑 id 量级特征)。运行时探针确证:同一 rank 上 target pool `layers=[0,61)`,draft pool `layers=[0,1)`,size 都是 467456(device),而逻辑空间 934977 = 467456×2+page+1。

**误区与正解**:第一版修复(把 target 的 mapping 注册给 draft pool)**方向错误**——它只翻译了写侧(`set_mla_kv_buffer` 在 pool 层翻译),而读侧 page-table 翻译在 backend 层、draft 无 coordinator 无法翻译,导致写物理/读逻辑劈成两个空间:不再崩,但 draft 静默读错槽,accept rate 被压低(实测 [3,1,4] 仅 0.42、[2,1,3] 0.58)。正解是让 draft **全程在逻辑空间自洽**:draft pool 传 `size = max_total × h2d` 且 `host_to_device_ratio=1`(使 KV buffer 与 index buffer 都等于逻辑空间,且不把 index_buf 放大到 h2d²),并**刻意不注册 mapping**(保持恒等)。附带好处:draft KV 用不变的逻辑 id 寻址,**天然免疫 target 的 carve/ring/LRU 重映射**(此前共享 mapping 方案下 draft KV 会被 target 重映射搁浅)。

**必要性 + 验证**:正确性 + 可用性双重缺陷。修复后此前必崩的 16×30k 并发 16/16 通过、0 error;显存代价为 draft 单层 buffer 翻倍 ≈ target KV 池的 +1.6%。**accept rate 全面回升**(印证读写不一致确实损伤提议质量),TP 30k 对照:

| 配置 | accept(第一版共享-mapping)→(本版恒等) | c=16 | c=24 |
|------|------|------|------|
| [1,1,2] | 0.91 → 0.96 | 506→529 | 900→**905** |
| [2,1,3] | 0.58 → 0.79 | 448→509 | 710→716 |
| [3,1,4] | 0.42 → 0.67 | 468→473 | 676→**1071** |

GSM8K MTP 回归 0.955、accept 0.949、0 error;hisparse 单测 28 passed。**教训**:短序列 + 低 h2d 的验证无法覆盖这类地址空间不一致问题(GSM8K 0.965、10k chunked 长期漏检),长上下文(≥30k)× 并发应作为 HiSparse+MTP 的常规验证门槛。

## 当前限制

1. **`device_buffer_size` 必须 ≥ `top_k * num_draft_tokens`**：保证 kernel 内部 N positions 不互相 evict。启动时校验，使用 `max_speculative_num_draft_tokens` 适配 adaptive speculative decoding。
2. **仅支持 DSV3.2**：DSV4 的 C4 indexer 路径未实现。`swap_in_verify_pages`、`register_token_positions` 和 staging ring 对 DSV4 有显式 guard。
3. **DP attention 已验证基本正确性**：tp8+dp8（`--enable-dp-attention`）下 short 确定性输出、1200-token 长生成（ring wrap）、12k 长 prompt、混合并发与 8 路 chat burst 全部通过，token usage 请求间回落 0.00，日志无错误。未做 DP 专项压测（大规模并发/rank 间负载不均衡场景）。
4. **spec 模式下 device buffer 全量分配**：MTP verify kernel 统一走 eviction slow path，任意 LRU slot 都可能成为 DMA 目的地，因此 admit 时 buffer 按 `padded_buffer_size` 全量分配（非 spec 模式仍按需分配）。每请求固定占用 `padded_buffer_size + spec_ring_capacity` 物理槽位。

## Speculative Staging Ring（已实现）

**问题**：曾经每轮 accept 的 tokens 的 hisparse physical pages 保留到 `request_finished` 才回收，长生成请求物理页占用随生成长度线性增长。

**方案**：每请求固定容量的 staging ring。

- Admit 时额外分配一段 page 对齐、容量 `R = spec_ring_capacity` 的物理区（`_alloc_spec_ring`），与 device buffer、全局 free list 物理隔离。
- 每轮 `eagle_prepare_for_decode` 中 speculative 槽位改走 **logical-only** 分配（allocator 的 `spec_logical_alloc()` 作用域包裹 `alloc_for_spec_decode`），不再触碰全局 `hisparse_attn_allocator`。
- `assign_spec_ring_slots(req, start, end)`：位置 p 的物理槽位 = `ring[p % R]`，写 mapping（图外，CG replay 读到最新内容）；同时回收位置 p-R 的 mapping（清零）。被回收 token 由 verify kernel 经 `full_to_token_position` → host DMA 读回。
- Ring 安全不变量（两条，均 assert fail-fast）：① `end - start ≤ R` —— 同一轮内两个位置不得落到同一环槽（`pos % R`），否则后写 KV 静默覆盖先写，且"新旧位置集合互不重叠"的前提失效；② `end - R ≤ last_backed`（backup 在同函数先于 alloc 执行）。`R = page_align(4 × get_alloc_reserve_per_decode())`，2× 是硬下限，4× 吸收 overlap 模式下 `kv_committed_len` 的滞后。
- `request_finished` 释放 buffer + ring 页（unique page 去重），物理占用恒定。

### 原互锁问题如何消解

1. **清零 mapping + CG 质量退化**：根因是 MTP kernel 的 fast path（`seq_len <= buffer`）没有 host DMA——mapping 清零且不在 buffer 表中的 token 静默返回 -1 被 attention 丢弃。已删除 fast path，统一走含 DMA 的 slow path（见 `hisparse.cuh` kernel 注释与 `test_mtp_short_seq_recycled_mapping_resolves_via_host_dma`）。
2. **Paged allocator 全局共享 / 页共享**：ring 区页对齐且专属本请求，speculative tokens 不再与 buffer-owned tokens 共享 page。
3. **释放页被 buffer 补充流程复用**：ring 槽位只在环内复用、从不进全局 free list，直到 `request_finished` 一次性归还。

## 测试配置

```bash
python -m sglang.launch_server --model-path <dsv32> \
    --disable-radix-cache --enable-hisparse \
    --speculative-algorithm EAGLE --speculative-num-steps 3 \
    --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 \
    --tp-size 8 --mem-fraction-static 0.85 \
    --hisparse-config '{"top_k": 2048, "device_buffer_size": 8192, "host_to_device_ratio": 1}'
```

## 测试结果（staging ring 轮次，2026-07-21，8×L20X/SM90）

**单元 / kernel 测试**：
- `test/registered/unit/managers/test_hisparse_unit.py`：25 passed。新增覆盖：`test_spec_ring_pool_stable_across_rounds`（12 轮 spec 分配 + ring wrap 后 `hisparse_attn_allocator.available_size()` 恒定——目标的直接回归）、`test_spec_ring_wrap_recycles_mapping`、`test_spec_mode_allocates_full_device_buffer`、`test_alloc_extend_cleared_tail_fails_fast`、`test_admit_long_prompt_reclaims_deferred_pages_for_ring`
- `test/registered/jit/test_hisparse.py`：17 passed。新增覆盖：短序列回收 mapping 经 host DMA 解析、CG replay host DMA、N=2 多 position（disjoint top-k 跨 position 不互逐、前序 position DMA 的 token 被后续 position buffer-hit 同槽位、多请求 + padding）
- 真实规模 kernel 独立复现（bs=2, top_k=2048, buffer=8192, N=4, block=960）：3 轮迭代输出全对，LRU 保持置换性

**E2E**（dsv32, tp8, 默认 overlap；`e2e_ring_test.py` 可复验）：

| Test | Config | CG | eager | Result |
|------|------|-----|-------|------|
| Short prompt ("Paris") | temp=0 deterministic | ✅ | ✅ | Output token-identical, accept 0.70 |
| Sequential 4 chat prompts | bs=1 | ✅ | ✅ | All high quality |
| 1200-token long gen ×3 | ring wrap ~19 times | ✅ | — | Three outputs token-identical, accept 0.48, 10.6s |
| 12k long prompt summary | > buffer 8192 | ✅ | — | Correctly listed three topics, accept 0.82 |
| Concurrent 2/4 mixed long-short | overlap | ✅ | ✅ | Passed (failed before gate fix) |
| Memory regression | 8 burst + periodic health check | ✅ | — | Usage returns to 0.00 between requests, no accumulation |
| Errors / IMA / assert | — | — | — | 0 |

**A/B 对照记录**（定位翻译 gate bug）：baseline（无 ring）并发通过 → 排除既有问题;my-branch eager 并发失败 → 排除 CG;kernel 独立复现通过 → 排除 kernel;锁定 Python 侧 prefill 翻译条件。

## 性能基准测试

对比三种配置的 decode 吞吐,所有脚本位于仓库根目录,模型路径 `.models_dsv32`(指向 DSV3.2 权重的符号链接),用 `.venv/bin/python` 运行,日志输出到 `logs/{timestamp}_{prefix}.log`。

### 关于 DP Attention

DSA 开启 DP Attention 需要**显式令 `dp_size = tp_size`**:`--tp-size 8 --dp-size 8 --enable-dp-attention`。此时 `attn_tp_size = tp_size / dp_size = 1`,每个 rank 是一个独立 DP rank(各自持有 KV 池、各自调度),并发容量约随 dp_size 线性扩展。

注意:若只传 `--enable-dp-attention` 而不设 `--dp-size`,dp_size 默认为 1,DSA 因 `dp_size < tp_size` 退回纯 TP 模式(日志出现 "DSA with TP mode is active" 警告),DP 不生效。已验证 `--tp-size 8 --dp-size 8` 下 HiSparse + MTP 可正常启动并服务(`enable_dp_attention=True`)。

### 对比配置与 server 命令

**① HiSparse + MTP**(`run_e2e_server.sh` / `run_mtp_hisparse_dp.sh`):
```bash
.venv/bin/python -m sglang.launch_server \
  --model-path .models_dsv32 \
  --disable-radix-cache --enable-hisparse \
  --speculative-algorithm EAGLE --speculative-num-steps 3 \
  --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 \
  --tp-size 8 --mem-fraction-static 0.85 \
  --hisparse-config '{"top_k": 2048, "device_buffer_size": 8192, "host_to_device_ratio": 1}' \
  --port 30012 --host 127.0.0.1
```

**② Baseline MTP(无 hisparse,全 KV 驻留 device)**(`run_baseline_mtp.sh`):
```bash
.venv/bin/python -m sglang.launch_server \
  --model-path .models_dsv32 \
  --disable-radix-cache \
  --speculative-algorithm EAGLE --speculative-num-steps 3 \
  --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 \
  --tp-size 8 --mem-fraction-static 0.85 \
  --port 30012 --host 127.0.0.1
```

**③ Baseline HiSparse(无 spec)**(`run_baseline_hisparse.sh`):
```bash
.venv/bin/python -m sglang.launch_server \
  --model-path .models_dsv32 \
  --disable-radix-cache --enable-hisparse \
  --tp-size 8 --mem-fraction-static 0.80 --max-running-requests 32 \
  --hisparse-config '{"top_k": 2048, "device_buffer_size": 8192, "host_to_device_ratio": 1}' \
  --cuda-graph-max-bs 32 \
  --port 30012 --host 127.0.0.1
```
注意:非 spec hisparse 的 decode CG 要在全部 bs 档位捕获完整 indexer+swap-in+MLA 前向,`mem-fraction 0.85` + 默认 max-running-requests 下 CG capture 会 OOM,故 ③ 需 `--cuda-graph-max-bs 32`(覆盖测试并发,且 bs≤24 仍走 CG,与 ① 对等)并把 mem-fraction 降到 0.80(其 decode 激活显存更重)。

### 测试一:单流 decode 吞吐(`bench_throughput.py`)

短 prompt(~30 tokens,全落 buffer)+ `max_new_tokens=512`、`temperature=0`,3 轮取均值;指标取 server 日志 `gen throughput (token/s)`(纯 decode,不含 prefill)。
```bash
.venv/bin/python bench_throughput.py "<label>" [server_log_path]
```

| Config | Log Gen Throughput Avg | Steady-state (run1-2) | vs Baseline HiSparse |
|------|------------------------|----------------|------------------------|
| Baseline HiSparse (No spec) | 68.0 tok/s | 70.96–71.16 | 1.00× |
| **HiSparse + MTP** | **111.8 tok/s** | 118–128 | **1.64×** |
| Baseline MTP (No hisparse) | 174.5 tok/s | 183–201 | 2.57× |

解读:单流短上下文下 MTP 在 hisparse 之上提速 **1.64×**;但相对无 hisparse 的 Baseline MTP 仍慢 ~36%(swap-in kernel + ring mapping + backup 的固定每步开销,此场景全命中 buffer、收益为零)。

### 测试二:长请求高并发(`bench_longbench_concurrent.py`)

**数据集**(`build_longbench_dataset.py`):从 LongBench(`THUDM/LongBench` 的 `data.zip`,经 HF 镜像 `HF_ENDPOINT=https://hf-mirror.com` 下载)采集 60 条英文长文本(narrativeqa/musique/hotpotqa/qmsum/gov_report 各 12 条),上下文 **9.3k–15.4k tokens**(均 > 8192 buffer → 强制 host swap-in;≤16384 → 单 chunk prefill),输出 `longbench_prompts.jsonl`。
```bash
HF_ENDPOINT=https://hf-mirror.com .venv/bin/python build_longbench_dataset.py
```

**压测脚本**:线程池并发,`ignore_eos` 强制每请求生成满 `max_new_tokens`(decode 主导、长度一致),统计系统聚合 decode 吞吐、单请求吞吐、延迟分位、成功率。
```bash
.venv/bin/python bench_longbench_concurrent.py "<label>" <并发> [max_tokens=256] [num_requests=60] [server_log_path]
```

**相同配置严格对比(mem-fraction 0.80,并发 12,双方 60/60 成功):**

| Metric | HiSparse + MTP | Baseline HiSparse | MTP Gain |
|------|----------------|-------------------|----------|
| System Decode Throughput | **144.3 tok/s** | **133.3 tok/s** | **+8.3%** |
| Per-request Decode Avg | 13.9 tok/s | 11.1 tok/s | +25% |
| Latency p50/p90/p99 | 21.0/27.9/34.7s | 22.9/23.8/23.8s | — |

**各自稳定上限(不同 mem-fraction):**

| Config | HiSparse + MTP | Baseline HiSparse |
|------|----------------|-------------------|
| @0.85 / concurrency 24 | 153.8 tok/s ✓ | ✗ activation OOM |
| @0.80 / concurrency 24 | ✗ logical OOM | 145.5 tok/s ✓ |

### 测试三:DP Attention(dp_size=8)长请求高并发

server 加 `--tp-size 8 --dp-size 8 --enable-dp-attention`(脚本 `run_mtp_hisparse_dp.sh` / `run_baseline_hisparse_dp.sh`)。DP 下 `max_running_requests` 为**每 rank** 语义(scheduler.py `effective_max_running_requests_per_dp`),总并发容量 ≈ `max_running_requests × dp_size`;每 rank KV 池 286464 tokens。`--mem-fraction-static 0.85 --max-running-requests 24`(总容量 ~192)。
bash
# HiSparse + MTP + DP=8
.venv/bin/python -m sglang.launch_server \
  --model-path .models_dsv32 \
  --disable-radix-cache --enable-hisparse \
  --speculative-algorithm EAGLE --speculative-num-steps 3 \
  --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 \
  --tp-size 8 --dp-size 8 --enable-dp-attention \
  --mem-fraction-static 0.85 --max-running-requests 24 \
  --hisparse-config '{"top_k": 2048, "device_buffer_size": 8192, "host_to_device_ratio": 1}' \
  --port 30012 --host 127.0.0.1

# Baseline HiSparse + DP=8(去掉 speculative 四个参数,加 --cuda-graph-max-bs 32)
```

**相同配置严格对比(mem-fraction 0.85,dp=8,并发 32,48 请求,双方 48/48 成功):**

| Metric | HiSparse + MTP (DP=8) | Baseline HiSparse (DP=8) | MTP Gain |
|------|------------------------|---------------------------|----------|
| System Decode Throughput | **210.2 tok/s** | **101.4 tok/s** | **2.07×** |
| System Total Throughput (incl. prefill) | 13348.8 tok/s | 6440.6 tok/s | 2.07× |
| Per-request Decode Avg | 7.9 tok/s | 6.0 tok/s | 1.32× |
| Latency p50/p90/p99 | 29.6/57.8/58.3s | 59.0/90.5/91.4s | p50 ~2x faster |

**关键观察**:DP 下 MTP 增益从 dp=1 高并发的 +8% 扩大到 **+107%(2.07×)**。原因:DP 把请求摊到 8 个 rank(每 rank 仅 ~4 个),减轻了单 rank 的 swap-in DMA 饱和,MTP"每步摊薄多 token"的优势重新显现;同时 DP 把并发容量扩展约 8×(dp=1 因 logical 池限制并发 ~24 即 OOM,dp=8 可支撑 ~192)。

**已知问题**:dp=8 下并发 ≥64 时,系统在处理完大部分请求后少数请求会挂起(server 转空闲不再调度),64/128 并发均复现;并发 32 可干净跑完。该高并发 hang 待查(疑似 DP 调度与 hisparse staging/ring 的交互边界)。

### 测试四：DP Attention 统一参数多 MTP 配置对比（2026-07-22）

统一参数消除配置差异，对比 Baseline HiSparse 与 3 种 MTP 配置。

**统一 server 参数**（所有 4 组共用）：
```bash
--model-path .models_dsv32 \
--disable-radix-cache --enable-hisparse \
--tp-size 8 --dp-size 8 --enable-dp-attention \
--mem-fraction-static 0.85 --max-running-requests 24 \
--hisparse-config '{"top_k": 2048, "device_buffer_size": 8192, "host_to_device_ratio": 1}' \
--cuda-graph-max-bs 32 \
--port 30012 --host 127.0.0.1
```
MTP 配置额外加 `--speculative-algorithm EAGLE --speculative-num-steps S --speculative-eagle-topk T --speculative-num-draft-tokens D`。

**负载**：LongBench 长序列（9k–15k tokens prompt），`max_new_tokens=256`，`ignore_eos=True`，48 请求。

**指标**：server log `gen throughput (token/s)` 的 trimmed average（去掉首尾 10% 采样值），纯 decode 阶段吞吐，每 DP rank 粒度。

**脚本**：`run_bench_suite.sh`（自动化启停 server + 跑 3 组并发度）。

| Config | accept_rate | c=16 | c=32 | c=48 |
|------|-------------|------|------|------|
| Baseline HiSparse | — | 56.5 | **78.3** | **79.4** |
| MTP [1,1,2] | 0.90 | **61.7 (+9.2%)** | 64.1 (−18.1%) | 61.1 (−23.0%) |
| MTP [2,1,3] | 0.75 | 60.6 (+7.3%) | 47.0 (−40.0%) | 49.6 (−37.5%) |
| MTP [3,1,4] | 0.61 | 37.2 (−34.2%) | 28.5 (−63.6%) | 36.8 (−53.7%) |

**关键发现**：

1. **低并发（c=16）下 MTP [1,1,2] 和 [2,1,3] 略有优势**（+7–9%），但 [3,1,4] 反而大幅退化（−34%）
2. **高并发（c=32/48）下所有 MTP 配置全面劣于 Baseline**，spec steps 越多退化越严重
3. **Accept rate 随 spec steps 增加显著下降**：[1,1,2]=0.90 → [2,1,3]=0.75 → [3,1,4]=0.61
4. **MTP 的 gen throughput 方差极大**（min/max 差 30–50×），speculative verify 在 HiSparse swap-in 场景下开销不稳定
5. **与测试三结论矛盾**：测试三（MTP [3,1,4] c=32）报告 2.07× 提升，但该测试用客户端 `total_completion / wall_time` 统计（含 prefill 时间稀释），且 Baseline 未加 `--cuda-graph-max-bs 32`（CG 覆盖不对等）。本测试用 server log 纯 decode 吞吐 + 统一参数，结论更可靠

**结论**：在 DP Attention + 长序列高并发场景下，MTP 的 speculative verify 引入的额外 HiSparse swap-in 开销抵消了投机收益。Spec steps 越多，每次 verify 的 swap-in 代价越高，且 accept rate 下降导致浪费更严重。仅 MTP [1,1,2] 在低并发下有微弱优势。

### 测试五：MTP [3,1,4] Torch Profiler 开销分析（2026-07-22）

对 MTP [3,1,4] 配置（与测试四相同参数）启用 torch profiler（`/start_profile`，30 decode steps，8 并发长序列请求），分析 TP-0/DP-0 rank 的 CUDA kernel 时间分布。

**Profile trace**：`logs/profiles/1784692758.837765-TP-0-DP-0.trace.json.gz`（186MB）

#### GPU 时间总览

| Category | Time | Share | Calls |
|------|------|------|---------|
| **Communication (NCCL + Custom AR)** | **5.25s** | **56.4%** | 3026 |
| **Compute** | **4.06s** | **43.6%** | — |

#### 计算部分细分

| Category | Time | Compute Share | Calls | Avg Latency |
|------|------|---------|---------|---------|
| MoE (fused_moe_kernel) | 1671.8ms | 41.2% | 3592 | 465μs |
| MoE (Auxiliary: sum_reduce/align/sort) | 236.8ms | 5.8% | 4595 | — |
| **Sparse Attention (flash_mla_sparse)** | **743.2ms** | **18.3%** | 1636 | 454μs |
| DeepGEMM (fp8 GEMM + MQA logits) | 606.6ms | 14.9% | 13406 | — |
| Quantization (per_token_group_quant) | 130.0ms | 3.2% | 15432 | — |
| **TopK (indexer)** | **71.0ms** | **1.7%** | 3562 | — |
| concat_mla_absorb_q | 60.0ms | 1.5% | 1636 | 37μs |
| **MTP swap-in kernel** | **58.2ms** | **1.4%** | 793 | 73μs |
| HiCache transfer (staging DMA) | 21.3ms | 0.5% | 14 | 1524μs |
| store_indexer_cache | 2.3ms | 0.1% | 1638 | 1μs |
| Other | 370.0ms | 9.1% | 38482 | — |

#### HiSparse 总开销

| Component | Time | Compute Share |
|------|------|---------|
| flash_mla_sparse attention | 743.2ms | 18.3% |
| TopK (indexer) | 71.0ms | 1.7% |
| MTP swap-in kernel | 58.2ms | 1.4% |
| HiCache transfer | 21.3ms | 0.5% |
| store_indexer_cache | 2.3ms | 0.1% |
| **HiSparse Total** | **896.0ms** | **22.1%** |

#### 关键发现

1. **MTP swap-in kernel 本身不是瓶颈**：仅 58.2ms（1.4%），平均每次 73μs，远小于 attention（454μs）和 MoE（465μs）
2. **真正的瓶颈是通信**：NCCL AllReduce 占 45.6%（4247ms），Custom AllReduce 占 10.4%（967ms），合计 56.4%。MTP 增加 spec steps → 更多 forward pass → 更多 AllReduce 调用，且被 reject 的 draft tokens 白白消耗通信
3. **MoE 是最大计算开销**：1909ms（47.0%），每次 verify forward 都要跑完整 MoE 层。MTP [3,1,4] 的 4 个 draft tokens 即使只有 ~61% 被接受，仍需跑 4 次完整 MoE
4. **Sparse attention 是第二大计算开销**：743ms（18.3%），但这在 Baseline 中也一样存在
5. **GPU memcpy（H2D/D2H）几乎为零**：7.5ms 总计。MTP swap-in 的 host DMA 在 kernel 内部通过 `memcpy_async` 实现（走 SM 而非 copy engine），不被 profiler 单独计量

#### 与 Baseline HiSparse Profile 对比（同参数同负载）

Baseline profile trace：`logs/profiles_baseline/1784694984.8087537-TP-0-DP-0.trace.json.gz`

**GPU kernel 时间对比**（TP-0/DP-0，约 28 个 forward pass）：

| Category | Baseline | MTP [3,1,4] | Delta |
|------|----------|-------------|-------|
| NCCL AllReduce | 5328.0ms (52.4%) | 4247.2ms (45.6%) | −1080.8ms |
| Custom AllReduce | 619.9ms (6.1%) | 966.5ms (10.4%) | +346.6ms |
| MoE (fused_moe) | 1568.3ms (15.4%) | 1671.8ms (18.0%) | +103.5ms |
| Sparse Attention | 878.9ms (8.6%) | 743.2ms (8.0%) | −135.7ms |
| DeepGEMM | 678.8ms (6.7%) | 606.6ms (6.5%) | −72.2ms |
| MTP swap-in | 0ms | 58.2ms (0.6%) | +58.2ms |
| Decode swap-in | 34.1ms (0.3%) | 0ms | −34.1ms |
| **Total Kernel** | **10162.2ms** | **9310.6ms** | **−851.6ms** |

**关键发现：MTP 总 GPU kernel 时间比 Baseline 少 851ms**——GPU 端不是瓶颈。

**CPU 侧开销对比**（真正瓶颈）：

| Metric | Baseline | MTP [3,1,4] | Ratio |
|------|----------|-------------|-------|
| Total CPU Op Time | 668s | 1176s | **1.76x** |
| Scheduler Iterations | 145 | **4287** | **29.6x** |
| get_next_batch_to_run | 718ms / 145 calls | 10824ms / 4287 calls | +10106ms |
| prepare_mlp_sync_batch (DP AllGather) | 341ms / 145 calls | 10087ms / 8557 calls | +9746ms |
| recv_requests | 1164ms / 144 calls | 2743ms / 4286 calls | +1579ms |

MTP 特有 CPU 开销（较小）：
- eagle_worker.forward_batch_generation: 8178ms (30 calls, 273ms/call)
- _hisparse_backup_new_tokens: 307ms
- eagle_prepare_for_decode: 326ms
- draft_extend: 557ms

#### 一致负载 profile（c=32, 48 requests, max_tokens=256，与测试四相同）

修正此前低负载 profile（c=8, 8 requests, max_tokens=128）的结论。使用和性能测试完全一致的负载重新 profile。

Baseline trace: `logs/profiles_baseline_c32/1784696408.5516293-TP-0-DP-0.trace.json.gz`
MTP trace: `logs/profiles_mtp_s3t1d4_c32/1784697102.6596196-TP-0-DP-0.trace.json.gz`

**GPU kernel 时间对比**（50 profiled steps）：

| Category | Baseline | MTP [3,1,4] | Delta |
|------|----------|-------------|-------|
| NCCL AllReduce | 13834ms | 9063ms | −4771ms |
| Custom AllReduce | 412ms | 1964ms | +1552ms |
| MoE (fused_moe) | 4630ms | 3511ms | −1119ms |
| Sparse Attention | 2413ms | 1904ms | −509ms |
| DeepGEMM | 1782ms | 1429ms | −353ms |
| MTP swap-in | 0ms | 52ms | +52ms |
| **Total Kernel** | **25.98s** | **20.18s** | **−5.80s (0.78x)** |

**CPU 调度开销对比**：

| Metric | Baseline | MTP [3,1,4] | Delta |
|------|----------|-------------|-------|
| run_batch | 51 calls / 41863ms | 51 calls / 31695ms | −10168ms |
| get_next_batch_to_run | 75 calls / 977ms | 66 calls / 1684ms | +707ms |
| gloo:all_gather | 75 calls / 191ms (2.6ms/call) | 94 calls / 1112ms (11.8ms/call) | +921ms |

高并发下 scheduler 空循环数基本一致（24 vs 15），此前低并发 profile 观察到的 4256 vs 114 空循环暴涨是低并发特有现象。

**性能退化根因**：

MTP GPU kernel 时间比 Baseline 少 22%，MTP swap-in kernel 仅 52ms（0.3%），GPU 端不是瓶颈。

根因是 **MTP 的 per-step 开销过高**。从真实 benchmark（c=32, 无 profiler）推算：

| Metric | Baseline | MTP [3,1,4] | Ratio |
|------|----------|-------------|------|
| per-step time | 38ms | 257ms | **6.8x** |
| per-step output | 1 token | 2.45 tokens | 2.45x |
| per-token time | 38ms | 105ms | **2.8x (MTP slower)** |

**Speculative decoding 盈亏条件**：`step_time_ratio ≤ accept_len`。实际 step_time_ratio=6.8x 远超 accept_len=2.45x。

MTP 每步需要执行：draft CG replay ×3 → CPU glue → verify CG replay → CPU glue → draft_extend CG replay → CPU glue（hisparse_backup、assign_ring_slots、alloc 等）。相比 Baseline 的单次 decode CG replay，多出 4 次 CG replay 和大量 CPU 过渡开销。在 DP Attention 下每次 CPU 过渡还要付 gloo:all_gather 同步代价（11.8ms/call），进一步恶化。

这是 DP Attention + speculative decoding 的结构性问题，不是 HiSparse 独有的。MTP 在 DP=1 时 per-step 开销更小（无 DP sync），能获得 1.64x 加速。

### Test 6: HiSparse vs Plain × Baseline vs MTP 2×2 Comparison (2026-07-22)

Isolate whether MTP regression under DP is caused by HiSparse+MTP implementation or by DP+MTP structurally. All 4 configs use `--dsa-decode-backend flashmla_sparse` to unify attention backend.

**Unified params**: tp8, dp8, enable-dp-attention, mem 0.85, max-running-requests 24, cuda-graph-max-bs 32, **dsa-decode-backend flashmla_sparse**. HiSparse configs add `--enable-hisparse --hisparse-config '{"top_k": 2048, "device_buffer_size": 8192, "host_to_device_ratio": 1}'`. MTP configs add `--speculative-algorithm EAGLE --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4`.

**Workload**: LongBench (9k–15k tokens prompt), c=32, 48 requests, max_new_tokens=256, ignore_eos=True.

**Metric**: server log `gen throughput (token/s)` trimmed average + median.

| Config | HiSparse | MTP | running-req | throughput (median) | trimmed_avg | wall time |
|--------|----------|-----|------------|---------------------|-------------|-----------|
| Plain Baseline | No | No | 2 | 73.2 | 66.4 | 63.3s |
| Plain MTP [3,1,4] | No | Yes | 2 | **14.7** | 37.2 | 143.9s |
| HiSparse Baseline | Yes | No | 3 | 86.9 | 79.3 | 57.1s |
| HiSparse+MTP [3,1,4] | Yes | Yes | 2 | **14.6** | 33.4 | — |

**Why HiSparse Baseline (79.3) > Plain Baseline (66.4)**:

Not a test bug — HiSparse offloads KV to host, reducing GPU memory pressure. With long sequences (9k–15k tokens), Plain can only run 2 concurrent requests (each occupies ~15k device tokens), while HiSparse can run 3 (each only needs ~8k device buffer tokens). Both use `flashmla_sparse` sparse decode (read top-k=2048 KV only), so per-token attention cost is similar. HiSparse wins on higher concurrency: 3 × 29 tok/s = 87 > 2 × 36.5 = 73.

**Why MTP is slow — DP rank synchronization stall (75% idle time)**:

Server log timeline analysis for DP rank 0:

| Metric | Plain Baseline | Plain MTP |
|--------|---------------|-----------|
| Total active time | 66s | 147s |
| Time in >5s gaps (rank idle, waiting for DP sync) | **0s (0%)** | **111s (75.5%)** |
| Actual work time | 66s (100%) | 36s (24.5%) |

Root cause: MTP's variable `accept_len` causes **rank desynchronization** under DP Attention:

1. DP Attention requires `gloo:all_gather` sync every scheduler step — all ranks must agree on whether to prefill or decode.
2. MTP `accept_len` varies by request content (avg ~2.5 but ranges 1–4). Different ranks finish decode at different times.
3. A rank that finishes its decode requests early (high accept rate → fewer steps) wants to prefill new requests, but must wait for slower ranks still decoding (low accept rate → more steps).
4. These **DP sync stalls** appear as 14–35s gaps in the fast rank's log, totaling 111s (75% of wall time).
5. Baseline has no such problem: every rank produces exactly 1 token per step → perfectly synchronized → 0 stall time.

MTP's actual **pure decode speed** is fast (burst throughput 155 tok/s ≈ 2.1x Baseline's 73 tok/s per token). The slowdown is entirely from DP sync stalls between decode and prefill phases.

This is a structural DP Attention + speculative decoding problem, not specific to HiSparse — Plain MTP shows the same behavior.

### Test 7: Pure TP=8 (no DP) 4-way comparison (2026-07-22)

Eliminate DP sync as a variable. All configs use pure TP=8 (no DP), `--dsa-decode-backend flashmla_sparse`, mem 0.85, cuda-graph-max-bs 32. Workload: LongBench 9k–15k prompt, c=4, 12 requests, max_tokens=256.

| Config | HiSparse | MTP | trimmed_avg | median | vs Plain Baseline |
|--------|----------|-----|-------------|--------|-------------------|
| Plain Baseline | No | No | 222.2 | 247.2 | — |
| **Plain MTP [3,1,4]** | **No** | **Yes** | **300.5** | **424.6** | **+35%** |
| HiSparse Baseline | Yes | No | OOM | — | (prefill OOM at mem 0.85) |
| HiSparse+MTP [3,1,4] | Yes | Yes | 215.6 | 342.4 | −3% |

**Key findings**:

1. **Pure TP: Plain MTP is +35% FASTER than Baseline** (trimmed_avg 300.5 vs 222.2). This confirms MTP works well without DP — the DP=8 regression (−44%) is entirely from DP rank sync stalls.
2. **HiSparse+MTP is comparable to Plain Baseline** (215.6 vs 222.2, −3%). The HiSparse overhead (swap-in, backup, ring management) roughly offsets the MTP gain under pure TP.
3. **HiSparse Baseline OOM**: at mem 0.85 pure TP, prefill of long sequences exhausts remaining GPU memory (only 1.15 GB after CG capture). This is a memory budget issue, not CG capture size — DP mode splits the KV pool per rank, leaving more room for activations.
4. **TP vs DP comparison**: MTP benefit flips from +35% (TP) to −44% (DP) — a 79 percentage-point swing — caused entirely by DP rank synchronization overhead.

### Test 8: Pure TP=8, h2d_ratio=2, long-seq high-concurrency (2026-07-22)

With `host_to_device_ratio=2`, HiSparse's logical pool is ~2x the device pool, enabling more concurrent long-sequence requests than Plain (which is limited by device KV capacity). This tests whether HiSparse+MTP can leverage the expanded capacity.

**Params**: TP=8, mem 0.85, cuda-graph-max-bs 32, flashmla_sparse decode. HiSparse: `top_k=2048, device_buffer_size=8192, host_to_device_ratio=2`. Workload: LongBench 9k–15k prompt, **max_tokens=512**, 48 requests.

| Config | c=8 | c=16 | c=24 | Notes |
|--------|-----|------|------|-------|
| Plain Baseline | 431 | 655 | **863** | Full KV on device |
| Plain MTP [3,1,4] | 367 | 699 | **1010 (+17%)** | accept_rate ~0.64 |
| HiSparse BL (h2d=2) | 369 | 569 | — | Sparse decode + host offload |
| HiSparse+MTP (h2d=2) | 345 | 552 | OOM | Prefill OOM at c=24 |

**Key findings**:

1. **Pure TP: Plain MTP works well** — +17% at c=24 (1010 vs 863). MTP's speculative gain is real when there's no DP sync overhead.
2. **HiSparse (h2d=2) is ~15% slower than Plain at same concurrency** (369 vs 431 at c=8, 569 vs 655 at c=16). The swap-in overhead outweighs the sparse-decode savings when device pool capacity isn't the bottleneck.
3. **HiSparse+MTP ≈ HiSparse BL** at c=8 (345 vs 369, −6%) and c=16 (552 vs 569, −3%). MTP's gain is largely neutralized by HiSparse's per-step overhead (swap-in, backup, ring management).
4. **HiSparse c=24 OOM**: at max concurrency, prefill activation memory exceeds the reduced device headroom. h2d_ratio=2 doubles logical capacity but the device-side KV+metadata is larger per token, leaving less room for activations.
5. **HiSparse didn't outperform Plain here** because Plain's device pool (467k tokens) can already handle c=24 at 15.5k tokens/request (24 × 15.5k = 372k < 467k). HiSparse's capacity advantage would show with even longer sequences or higher concurrency that exceeds Plain's pool.

### Test 9: Pure TP=8, h2d_ratio=2, long decode (max_tokens=1024) + high concurrency (2026-07-23)

Push Plain to its KV pool limit with longer output and higher concurrency. At c=32 × ~16k tokens/request = 512k tokens, Plain's 502k pool is near-saturated, triggering scheduler retraction. HiSparse h2d=2 has ~847k logical pool and should sustain higher throughput.

**Params**: TP=8, mem 0.85, cuda-graph-max-bs 32, flashmla_sparse decode. HiSparse: `top_k=2048, device_buffer_size=8192, host_to_device_ratio=2`. Workload: LongBench 9k–15k prompt, **max_tokens=1024**, 48 requests.

**Pool sizes**: Plain 502k tokens (device only). HiSparse BL h2d=2: 423k logical tokens (~847k with host). HiSparse+MTP h2d=2: 467k logical tokens (~934k with host).

| Config | c=16 | c=24 | c=32 | Notes |
|--------|------|------|------|-------|
| Plain Baseline | 666 | **877** | 809 (↓retraction) | Pool saturated at c=32 |
| Plain MTP [3,1,4] | **855 (+28%)** | **1042 (+19%)** | **1134 (+40%)** | accept_rate ~0.64 |
| HiSparse BL (h2d=2) | 579 | 762 | 726 (↓) | Swap-in overhead vs Plain |
| HiSparse+MTP (h2d=2) | 707 (+22%) | 884 (+16%) | OOM | Prefill OOM at c=32 |

**Key findings**:

1. **Plain MTP is the winner** in pure TP: +28% at c=16, +40% at c=32 vs Plain BL. MTP's speculative gain is substantial without DP sync overhead.
2. **Plain BL hits retraction at c=32** (877 → 809, −8%): KV pool near saturation, scheduler starts retracting requests. MTP avoids this because fewer decode steps = faster KV release.
3. **HiSparse BL at c=32 also retracts** (762 → 726, −5%) but less severely than Plain, confirming partial host-offload benefit.
4. **HiSparse+MTP shows +22% over HiSparse BL** at c=16 (707 vs 579), confirming MTP helps on top of HiSparse.
5. **HiSparse+MTP OOM at c=32**: prefill activation memory exceeds device headroom. The MTP draft model weights consume device memory that could otherwise go to activations.
6. **HiSparse is ~15% slower than Plain at same concurrency** (579 vs 666 at c=16, 762 vs 877 at c=24). The swap-in + backup + ring overhead isn't compensated by pool capacity expansion when Plain can still handle the load.
7. **HiSparse's capacity advantage not fully demonstrated**: Plain's 502k pool can still serve c=32 (with retraction), so HiSparse's larger logical pool doesn't provide a decisive win. Need even higher concurrency (c=48+) or longer sequences (32k+) to push Plain into hard OOM while HiSparse continues serving.

### Test 10: Pure TP=8, h2d_ratio=2, extreme concurrency + long decode (max_tokens=2048) (2026-07-23)

Push all configs to their limits with max_tokens=2048 and c=24/32/48 to see if HiSparse's larger logical pool can outperform Plain under heavy KV pressure.

**Params**: TP=8, mem 0.85, flashmla_sparse, cuda-graph-max-bs 32. HiSparse: `top_k=2048, device_buffer_size=8192, host_to_device_ratio=2`. Workload: LongBench 9k–15k prompt, **max_tokens=2048**, 48 requests. Per request: ~17k tokens peak.

**Pool sizes**: Plain 502k tokens. HiSparse BL: 423k logical (~847k with host). HiSparse+MTP: 467k logical (~934k with host).

| Config | c=24 | c=32 | c=48 |
|--------|------|------|------|
| Plain BL | **882** | 833 (↓6%) | 834 (plateau) |
| HiSparse BL (h2d=2) | 772 (−12%) | 727 (−13%) | **crash** |
| HiSparse+MTP (h2d=2) | 808 (−8%) | 801 (−4%) | **789 (−5%)** |

**Key findings**:

1. **HiSparse BL is consistently 12–13% slower than Plain BL** at every concurrency level. The swap-in overhead (host DMA for top-k KV per decode step) outweighs any capacity benefit from host offloading.
2. **Plain BL retraction is mild**: c=32 drops 6% from c=24 but then plateaus at c=48 (834), because scheduler retraction gracefully limits concurrent requests without crashing. This prevents the scenario where HiSparse's extra capacity would be decisive.
3. **HiSparse BL crashes at c=48** due to prefill activation OOM (non-spec decode CG captures 8 bs tiers, consuming more device memory for activations than MTP's verify CG).
4. **HiSparse+MTP survives c=48** (789 tok/s) — MTP's verify CG has a smaller activation footprint than non-spec decode CG. But 789 is still 5% below Plain BL's 834.
5. **HiSparse+MTP is ~5–10% faster than HiSparse BL** at each concurrency (807 vs 772, 801 vs 727), confirming MTP provides a consistent speedup on top of HiSparse.
6. **Overall conclusion for h2d_ratio=2**: HiSparse's host-offload capacity expansion does not translate to throughput gains on this hardware (L20X 140GB, TP=8, mem 0.85). The root cause is **device_buffer_size is the real concurrency bottleneck, not the logical pool**. With h2d=2, the device pool is halved (~211k tokens), and each request's device buffer allocation (8192 tokens) limits max concurrent requests to ~211k/8192 ≈ 25 — the same as Plain's ~502k/17k ≈ 29. The larger logical pool (847k) provides host storage but does not increase decode-time concurrency because device buffer capacity is the binding constraint. Server logs confirm: both Plain and HiSparse plateau at running-req ≈ 24 during decode, despite HiSparse's logical pool being 1.7x larger. To unlock HiSparse's capacity advantage, either (a) reduce `device_buffer_size` (e.g., 2048–4096) to fit more concurrent buffers in the device pool, or (b) use a higher `h2d_ratio` (e.g., 4–8) so the device pool grows while host pool grows even more.

### Test 11: Pure TP=8, h2d=2, buffer=4096, HiSparse+MTP [1,1,2] (2026-07-23)

Reduce `device_buffer_size` to 4096 (min for MTP [1,1,2]: top_k=2048 × num_draft=2) to increase max concurrent requests. Compare HiSparse BL and HiSparse+MTP [1,1,2] against Plain BL, all at buffer=4096.

**Params**: TP=8, mem 0.85, flashmla_sparse, max_total_tokens=423680 for HiSparse. HiSparse: `top_k=2048, device_buffer_size=4096, host_to_device_ratio=2`. MTP [1,1,2]: num_steps=1, topk=1, draft_tokens=2. Workload: LongBench 9k–15k prompt, max_tokens=2048, 48 requests.

| Config | c=24 | c=32 | wall c=24 | wall c=32 |
|--------|------|------|-----------|-----------|
| Plain BL | 876 | 826 (retraction) | 176s | 182s |
| HiSparse BL (buf=4096) | 714 (−19%) | 676 (−18%) | 204s | 207s |
| **HiSparse+MTP [1,1,2] (buf=4096)** | **1082 (+24%)** | **1098 (+33%)** | **164s** | **162s** |

**Key findings**:

1. **HiSparse BL (buf=4096) is still 18–19% slower than Plain BL** at same concurrency. The swap-in overhead per decode step remains a net cost when both achieve similar concurrent request counts (~24).
2. **HiSparse+MTP [1,1,2] beats Plain BL by 24–33%** — the first configuration where HiSparse+MTP clearly outperforms Plain. Wall time is 7–11% faster (164s vs 176s, 162s vs 182s).
3. **MTP [1,1,2] is the key**: accept_rate=0.97 (nearly every draft accepted), yielding ~2 tokens per step with minimal overhead (1 draft step only). This 2x token-per-step gain more than compensates HiSparse's swap-in cost.
4. **HiSparse+MTP [1,1,2] vs HiSparse BL: +52–62%** — MTP provides massive acceleration on top of HiSparse when the speculative config is lightweight (1 step, 1 draft token).
5. **Why [1,1,2] works better than [3,1,4]**: accept_rate 0.97 vs 0.61, per-step overhead 1 CG replay vs 5, device_buffer_size 4096 vs 8192 (2x more concurrent requests). The lightweight config dominates in every dimension.

**Conclusion**: HiSparse+MTP is effective when (a) MTP config is lightweight — [1,1,2] with near-perfect acceptance, minimal per-step overhead, and small device buffer requirement, and (b) running in pure TP mode without DP synchronization overhead. Under these conditions, HiSparse+MTP outperforms both Plain BL and HiSparse BL, achieving the best decode throughput of any configuration tested.

### Test 12: Pure TP=8, h2d=2, 30k prompts — HiSparse capacity advantage (2026-07-23)

Use **30k-token prompts** (2x repeat of LongBench contexts, 27k–32k tokens) to make Plain's KV pool the binding constraint. Plain pool (502k tokens) fits ~15 concurrent 32k requests; HiSparse h2d=2 logical pool (847k) fits ~26 concurrent. This is the scenario where HiSparse's host-offload capacity should shine.

**Params**: TP=8, mem 0.85, flashmla_sparse, cuda-graph-max-bs 32. HiSparse: `top_k=2048, device_buffer_size=4096, host_to_device_ratio=2`, max_total_tokens=423680. MTP [1,1,2]. Workload: **30k-token prompts** (`longbench_prompts_30k.jsonl`), max_tokens=512, 24 requests.

| Config | c=12 | c=16 | c=24 |
|--------|------|------|------|
| Plain BL | 530 | 502 (retraction) | 503 (plateau) |
| HiSparse BL (h2d=2, buf=4096) | 415 (−22%) | 411 (−18%) | **655 (+30%)** |
| HiSparse+MTP [1,1,2] (buf=4096) | **595 (+12%)** | **559 (+11%)** | **617 (+23%)** |
| HiSparse+MTP [2,1,3] (buf=6144) | 550 (+4%) | 517 (+3%) | 532 (+6%) |

**Key findings**:

1. **HiSparse BL outperforms Plain BL at c=24: 655 vs 503 (+30%)**. This is the first test where HiSparse itself beats Plain. At c=24 with 30k prompts, Plain's pool is saturated (24 × 32k = 768k >> 502k pool) and retraction limits throughput, while HiSparse's larger logical pool (847k) can accommodate more concurrent requests, achieving higher aggregate decode throughput.
2. **HiSparse BL is slower at c=12/16** (−18 to −22%): when both configs have enough pool capacity, HiSparse's per-step swap-in DMA overhead is a net cost. HiSparse only wins when Plain's pool becomes the bottleneck.
3. **HiSparse+MTP [1,1,2] outperforms Plain BL at ALL concurrency levels**: +12% at c=12, +11% at c=16, +23% at c=24. MTP's near-perfect acceptance (0.94–0.97) compensates for swap-in overhead even when pool isn't the bottleneck.
4. **HiSparse+MTP vs HiSparse BL**: +43% at c=12, +36% at c=16. At c=24 HiSparse BL catches up (655 vs 617) because its higher concurrency advantage kicks in while MTP's overhead slightly limits its ability to scale.
5. **MTP [2,1,3] vs [1,1,2]**: [2,1,3] (buf=6144, accept_rate 0.83) is worse than [1,1,2] (buf=4096, accept_rate 0.96) at every concurrency — more draft steps = lower accept rate, larger buffer requirement, and higher per-step overhead. [1,1,2] is the optimal speculative config for HiSparse.
6. **Crossover point**: HiSparse BL overtakes Plain BL between c=16 and c=24 — precisely when Plain's pool saturates. With even longer prompts or higher concurrency, HiSparse's advantage would grow further.

**Conclusion**: HiSparse's host-offload capacity advantage is real but only manifests when **Plain's KV pool is the binding constraint** (long sequences × high concurrency). Below that threshold, swap-in DMA overhead makes HiSparse slower. HiSparse+MTP [1,1,2] is the strongest configuration overall: it beats Plain at all concurrency levels by combining MTP's speculative acceleration with HiSparse's capacity expansion.

### Test 13: Per-round CPU-sync elimination (2026-07-23)

针对"Plain HiSparse 不如 Plain BL / MTP HiSparse 提升有限"的归因与优化。热路径审查发现 MTP 每轮有三类 CPU 阻塞/小 op 风暴,全部与 GPU 计算无关:

1. **`_do_hisparse_backup` 每轮两次 `current_stream().synchronize()`**——每轮 decode 排空整条 forward 流水线,直接对冲 overlap scheduler;而非 spec decode 的 `_eager_backup_previous_token` 早已是事件化异步。修复:改用同一模式(backup 入 `decode_backup_stream`,GPU 侧 wait schedule/producer 流,记录 `_backup_done_event`;消费者已有 `wait_for_pending_backup` 的 GPU 侧事件等待)。
2. **每次 verify forward 前 `write_staging_stream.synchronize()`**(CPU 阻塞)。修复:改为 `current_stream().wait_stream(write_staging_stream)`(GPU 侧序),且仅在 `has_ongoing_staging()` 时执行。
3. **`assign_spec_ring_slots` 每请求 ~7 次小 kernel launch**(bs=24 → 每轮 ~170 次)。修复:批量化(`assign_spec_ring_slots_batch`),索引在 CPU 拼好后单次 H2D,每轮固定 ~8 次 launch。
4. 非 spec decode 每步 `map_last_loc_to_buffer` 的隐式 GPU→CPU 同步(stale-free 的布尔掩码索引)与冗余 mapping 清零:paged 路径 `alloc_decode` 为 logical-only、不可能产生 stale mapping,门控到 page_size==1;清零两行后即被覆盖,直接删除(`release_backed_logical_locs` 随之无生产调用方,已删)。

**A/B 结果**(Test 11 同参数/口径:TP=8, h2d=2, buf=4096, [1,1,2], max_tokens=2048, 48 reqs;BL 同轮复测校准环境无漂移 708-716/674-679 ≈ 文档 714/676):

| Config | c=24 | c=32 |
|--------|------|------|
| HiSparse+MTP [1,1,2] pre-opt (Test 11) | 1082 | 1098 |
| + 优化 1/2 (async backup + 事件化 staging 等待) | 1062 (噪声内) | **1210 (+10.2%)** |
| + 优化 3 (ring 批量化) | **1120 (+3.5%)** | 1192 (维持) |
| HiSparse BL (环境校准, 优化不触及该路径) | 708–716 | 674–679 |
| Plain BL (Test 11 参考) | 876 | 826 |

**归因结论**:

- **问题 2(MTP HiSparse 提升有限)部分成立且已优化**:每轮 CPU 全同步是真实损耗,消除后高并发(c=32)+10%,HiSparse+MTP [1,1,2] 对 Plain BL 的优势扩大到 +36~46%(1120/1192 vs 876/826)。并发越高、overlap 空间越大,收益越明显。
- **问题 1(Plain HiSparse 不如 Plain BL)是结构性的**:decode-path 胶水消减后 BL 吞吐不变(708/674 ≈ 714/676),说明 ~19% 差距不在 CPU 侧,而是每步每层 swap-in kernel 的 PCIe host DMA 数据量(top_k 中 buffer miss 部分 × 层数 × 步数)——只要 Plain 的 KV 池尚未成为约束,这笔搬运就是纯开销。这与 Test 12 的 crossover 结论一致:HiSparse 的价值兑现条件是 Plain 池饱和(超长序列 × 高并发),或叠加 MTP 的每步多 token 摊薄。进一步压缩该成本需减少 miss 量(更大 buffer/更高 top-k 命中率)或提升 DMA 带宽利用,属 kernel/硬件层工作,不在本轮范围。

1. **MTP 增益强烈依赖每 rank 的 DMA 饱和程度**:dp=1 高并发长序列下,单 rank 承载全部并发请求,swap-in DMA(PCIe)饱和,MTP 增益从单流的 1.64× 收窄到 ~8%;而 **dp=8 把请求摊到 8 个 rank,每 rank DMA 压力降到 ~1/8,MTP 增益回升到 2.07×**。即 DP 与 MTP 是互补的:DP 扩容量并缓解 DMA 饱和,MTP 在此基础上再提速。
2. **两者内存瓶颈不同**,无单一 mem-fraction 让双方都在并发 24 跑通:Baseline HiSparse decode 激活显存重(需 0.80);HiSparse+MTP KV/logical 池重(`ignore_eos` 满长生成 + admission 过提交,需 0.85 余量)。
3. **h2d=1 下 hisparse 拓宽的是单请求长度而非并发数**:logical 池(= device 池)按完整序列长度计费,并发上限 ≈ pool / 序列长。要提升并发容量需 `host_to_device_ratio > 1`。
4. `ignore_eos` 会使请求满长增长不提前结束,小池下易触发 admission 过提交 → logical OOM;自然 EOS 可自调节但 decode 测量偏薄。按场景取舍。

### Test 14: MTP 开关四配置对比 + draft-pool mapping 修复验证 (2026-07-27)

**目的**:构造一个负载使三个不等式同时成立 —— (0) Baseline MTP > Baseline,(1) HiSparse > Baseline,(2) HiSparse MTP > HiSparse。

**参数**:纯 TP=8,mem 0.85,`cuda-graph-max-bs 32`,**不设** `--max-total-tokens`(与 Test 12 的关键差异,正是它暴露了 draft-pool mapping bug)。HiSparse `top_k=2048, h2d=2`,各 MTP 配置用最小页对齐 buffer(`top_k × num_draft`):[1,1,2]→4096、[2,1,3]→6144、[3,1,4]→8192。负载 `longbench_prompts_30k.jsonl`(24 条 30k–50k token),`max_tokens=512`,24 请求,c=16/24。指标为 server log 中 decode 的 `gen throughput`,按压测时间窗过滤后去头尾 10% 截尾平均。

| 配置 | c=16 | c=24 | accept_rate | max_running |
|------|------|------|-------------|-------------|
| Baseline | 578.4 | 574.1 | — | 15(池饱和) |
| Baseline MTP [3,1,4] | 590.9 | 677.2 | 0.68 | 16 |
| HiSparse (buf=4096) | 416.4 | 666.7 | — | 24 |
| **HiSparse MTP [1,1,2]** (buf=4096) | **506.4** | **900.6** | 0.91 | 24 |
| HiSparse MTP [2,1,3] (buf=6144) | 448.4 | 709.7 | 0.58 | 24 |
| HiSparse MTP [3,1,4] (buf=8192) | 468.3 | 675.9 | 0.38 | 24 |

**三不等式全部成立**:
- (0) Baseline MTP > Baseline:590.9/677.2 vs 578.4/574.1(c=24 +18%)
- (1) HiSparse > Baseline:c=24 666.7 vs 574.1(**+16%**);c=16 反而 −28%,crossover 位置与 Test 12 一致 —— HiSparse 仅在 Plain 池饱和时兑现容量优势(`max_running` 24 vs 15 是机制证据)
- (2) HiSparse MTP > HiSparse:**三个 MTP 配置在两档并发下全部成立**,最优 [1,1,2] c=16 +22%、c=24 **+35%**

**最佳配置**:HiSparse+MTP [1,1,2] 在 c=24 达 900.6 tok/s,为全部配置最高,较 Baseline **+57%**,满并发 24,`errors_in_log=0`。

**MTP 配置选择**:accept rate 随 draft 步数单调下降(0.91 → 0.58 → 0.38),叠加更大的 buffer 需求(4096/6144/8192 挤占并发容量),使 [1,1,2] 在两档并发下均为最优 —— 与 Test 12 结论一致。三配置 `errors_in_log` 均为 0。

**本轮同时定位并修复了一个真实 bug**(见改动清单 13):draft(NextN)KV pool 的 KV buffer 按物理大小开、而寻址用的是 target 的逻辑 id 空间(且读路径无 coordinator 无法翻译),导致 draft KV 越界。修复前本测试的三个 HiSparse+MTP 配置在 30k×并发下**全部 IMA 崩溃**;正解(draft pool 扁平化到逻辑空间 + 保持恒等)后 16×30k 并发 16/16 通过、accept_rate 回升到 0.95–0.96。Test 12 当时能出数是因为设了 `--max-total-tokens 423680` 压低逻辑地址空间。注:本表中三配置的数值采集自第一版(共享-mapping)修复,accept 偏低([2,1,3] 0.58、[3,1,4] 0.42);正解版 accept 全面回升、c=24 吞吐更高(见改动清单 13 对照表),但不改变三不等式成立的结论。

**结论**:30k 长上下文 × c=24 是同时满足三个不等式的负载区间;HiSparse 的价值条件仍是 Plain 池饱和,叠加轻量 MTP([1,1,2],高接受率、最小 buffer)后收益最大化。长上下文(≥30k)× 并发应作为 HiSparse+MTP 的常规验证门槛。

**DP-attention 侧的可比性限制**(同轮尝试）：把同一负载搬到 `--enable-dp-attention`(tp8+dp8)后,客户端并发被 8 个 rank 均摊,`max_running` 只有 3–4(64k 负载 c=24 → 96 tok/s;30k 负载 c=16 → 63 tok/s,而同负载纯 TP 为 578)。原因是 dp=8 要求所有 rank 同步前进,单 rank 仅 3–4 个长请求时 decode batch 过小、GPU 欠载,吞吐被小 batch 与 rank 间同步主导。要让每 rank 达到纯 TP 的 batch 规模需客户端并发 ≈ 8×24 = 192,而 Test 3 已记录 **dp=8 在并发 ≥64 会 hang**,该区间不可达。**因此 DP 与 TP 的绝对吞吐在本环境下不可比**;DP 侧只能做配置间的相对比较(取 hang 阈值下的最大并发 c=48)。这也解释了 Test 6/7 中 DP 数据为何整体偏低。

**DP 相对对比(30k,c=48,每 rank ~6 请求)**:

| 配置 | gen throughput | accept_rate | max_running |
|------|----------------|-------------|-------------|
| dp_baseline | 176.3 | — | 6 |
| dp_baseline_mtp314 | 112.2 | 0.68 | 5 |
| dp_hisparse (buf=4096) | 140.5 | — | 6 |
| **dp_hisparse_mtp112** (buf=4096) | **194.1** | 0.89 | 6 |
| dp_hisparse_mtp213 (buf=6144) | 194.0 | 0.59 | 6 |
| dp_hisparse_mtp314 (buf=8192) | 176.4 | 0.42 | 6 |

**DP 下不等式 (0) 与 (1) 不成立,(2) 成立**:
- (0) 失败(112.2 < 176.3):**DP + MTP 的 rank 去同步空转**。各 rank accept_len 不同,dp=8 全同步前进时相互等待(Test 3 记录过 75% idle),抵消并反超投机收益。瞬时峰值可达 367,但截尾均值仅 112(median 86),说明大量时间耗在等待。
- (1) 失败(140.5 < 176.3):每 rank 仅 6 个请求,KV 池远未饱和 → swap-in DMA 是纯开销,与纯 TP c=16(416 < 578)同理。DP 下要复现池饱和需 c≈192,受 dp hang 限制不可达。
- (2) **成立(194.1 > 140.5,+38%)**,与纯 TP 下的 +22%/+35% 同量级。

**DP 下的额外结论 —— 接受率决定 MTP 在 DP 中的成败**:同样开 MTP,`[3,1,4]`(accept 0.68)只有 112.2、显著低于 baseline,而 `[1,1,2]`(accept 0.89)达到 194.1、是**唯一超过 dp_baseline 的配置(+10%)**。因为 accept_len 的方差直接决定 rank 间等待量:高接受率 ⇒ 各 rank 每轮推进的 token 数趋于一致 ⇒ 去同步空转显著减少。这比纯 TP 场景更强地支持"[1,1,2] 是 HiSparse+MTP 的首选配置"。

**因此三不等式同时成立的区间是纯 TP + 长上下文 + 池饱和并发(c=24)**;DP 在当前 dp hang 限制下无法进入该区间。

### Test 15: draft pool 扁平化修复后的四配置全量重测 (2026-07-28)

**背景**:Test 14 的三个 HiSparse+MTP 行采自第一版(共享-mapping)修复,draft 读写地址空间被劈开、accept 被压低。改动清单 13 落地正解(draft pool 扁平化到逻辑空间 + 保持恒等)后,用**与 Test 14 逐项相同的参数**全量重测。

**参数**:与 Test 14 完全一致(纯 TP=8、mem 0.85、`cuda-graph-max-bs 32`、不设 `--max-total-tokens`、`top_k=2048 h2d=2`、各 MTP 配置最小页对齐 buffer、`longbench_prompts_30k.jsonl`、24 请求、`max_tokens=512`、c=16/24)。

| 配置 | c=16 | c=24 | accept_rate | max_running |
|------|------|------|-------------|-------------|
| Baseline | 570.6 | 569.2 | — | 15(池饱和) |
| Baseline MTP [3,1,4] | 654.7 | 681.3 | 0.70 / 0.73 | 15 |
| HiSparse (buf=4096) | 415.3 | 663.3 | — | 24 |
| HiSparse MTP [1,1,2] (buf=4096) | 520.0 | 907.8 | 0.96 | 24 |
| **HiSparse MTP [2,1,3]** (buf=6144) | **559.0** | 903.4 | 0.82 | 24 |
| HiSparse MTP [3,1,4] (buf=8192) | 482.2 | 1119.2 ⚠ | 0.68 | 24 |

**三不等式仍全部成立**:(0) 654.7/681.3 > 570.6/569.2;(1) c=24 663.3 > 569.2(+17%);(2) 三个 MTP 配置在两档并发下全部高于 HiSparse。

**可重复性**:Baseline / HiSparse / Baseline MTP 三行与 Test 14 偏差仅 1–2%(它们都不经过 draft 的 hisparse pool——前两者无 draft,Baseline MTP 用非 hisparse pool,因此本应不变,实测确认)。`max_running` 依旧 baseline 15 / hisparse 24。全部配置 `errors_in_log=0`。

**修复效果**:三个 HiSparse+MTP 配置的 accept 全面回升(0.96 / 0.82 / 0.68,对比第一版 0.91 / 0.58 / 0.38),c=24 吞吐同步提升——独立印证"draft 读写空间不一致确实在损伤提议质量",且 step 数越多受损越重。

**⚠ 短压测在 c=24 不可靠 → 加长复测(`max_tokens=2048`,样本 21–27 个)**:

| 配置 | 短压测 c=24(n=5–7) | 加长 c=24(n=21–27) | mean vs median | accept |
|------|------|------|------|------|
| [1,1,2] | 907.8 | 944.2 | 944 ≈ 958(稳) | 0.97 |
| **[2,1,3]** | 903.4 | **1076.9** | 1077 ≈ 1113(稳) | 0.83 |
| [3,1,4] | 1119.2 | 887.0 | 887 vs **1175**(方差极大) | 0.63 |

**[3,1,4] 在短压测中的"反超"是采样假象**:样本充足后其截尾均值掉到三者最低(887.0),且 mean 与 median 相差 288——它能跑出最高瞬时吞吐,但尾部存在严重慢样本(buffer 8192 挤占并发容量 + accept 仅 0.63 的退化叠加),n=5 时恰好只采到快的区段。

**结论 —— 最佳配置由 [1,1,2] 变为 [2,1,3]**:draft 读写空间统一、accept 恢复后,中等 draft 步数([2,1,3],3 token/步、buffer 6144)成为吞吐与稳定性的最佳平衡点,c=24 达 1076.9(较 Baseline **+89%**),c=16 也最优(559.0);继续加到 4 步([3,1,4])则被更大 buffer 占用与更低接受率反噬。[1,1,2] 稳定性最好(accept 0.97、mean≈median),适合对延迟抖动敏感的场景。**这也修正了 Test 12/14 "步数越多越差" 的结论——那是 draft KV 地址空间 bug 压出来的假象。**

#### Test 15-DP(a): 30k c=48 重测 —— 修正 Test 14 DP 结论

与 Test 14 DP 小节同参数(30k、48 请求、`max_tokens=512`、c=48)。不受修复影响的三行偏差 ≤2%(可重复性 ✓):baseline 173.1、baseline_mtp314 118.0(accept 0.67)、hisparse 140.0。受影响的两行([3,1,4] 未采,被中途切换到 64k 实验):

| 配置 | Test 14(第一版修复) | Test 15(正解) | accept 变化 |
|------|------|------|------|
| dp_hisparse_mtp112 | 194.1 | 205.4 | 0.89 → 0.96 |
| **dp_hisparse_mtp213** | 194.0 | **224.4** | 0.59 → 0.82 |

**Test 14 的"只有 [1,1,2] 能超过 dp_baseline"与"[1,1,2] 是 DP 首选"均被推翻**:[2,1,3] 达 224.4(超 baseline +30%,亦超 [1,1,2]),DP 与 TP 的最优配置统一为 **[2,1,3]**。

#### Test 15-DP(b): 64k c=48 —— 不等式 (1) 在 DP 下首次成立

**动机**:30k × c=48 时每 rank 仅 6 请求 × ~30k = 233k,远低于每 rank 池(plain 341k)→ 池不饱和,HiSparse 无从兑现。改用 64k 集(实测 67k–79k token/条):6 × ~74k ≈ **444k**,plain 池 158% 过载、hisparse 逻辑池(h2d=2,573k)77% 装得下——把 TP c=24 的饱和条件在 DP 内复现。参数:dp8、c=48、`num_req=48`(仅一波 prefill)、`max_tokens=1024`。

**测量方法改进(PURE_DECODE)**:此前的 `gen throughput` 截尾平均混入 prefill 交织样本(用旧 TP 日志复核:full_window 571.9 vs 纯 decode 477.1,**虚高 20%**)。新解析器只统计**最后一个 `Prefill batch` 行之后**的样本(= "先跑完所有 prefill 再统计 decode"),并保留 full_window 对照。hisparse 系配置 full ≈ pure(一波 prefill 无交织);池饱和的 baseline 因 admission 拖尾,full_window 被交织抬高 28%。

| 配置 | PURE_DECODE | max_running | accept |
|------|------|------|------|
| dp64k_baseline | 76.8 | **3**(池饱和) | — |
| **dp64k_hisparse** | **135.4(+76%)** | **6**(满并发) | — |
| dp64k_baseline_mtp314 | 75.6 ⚠(尾窗 n=11,max_run=1) | 1 | 0.45 |
| **dp64k_hisparse_mtp213** | **239.8**(med 264.6,= baseline 3.1×) | 6 | 0.84 |
| dp64k_hisparse_mtp314 | 239.7(med 280.5,尾部重) | 6 | 0.72 |
| dp64k_hisparse_mtp112 | 222.3(med 222.2,最稳) | 6 | 0.97 |

**结论**:
- **(1) 在 DP 下首次成立且幅度最大(+76%)**,机制与容量算术完全吻合:baseline 被池钳到 `max_running=3`,hisparse 满并发 6——容量差直接兑现为吞吐差。HiSparse 的价值条件("Plain 池饱和")与并行方式无关,只取决于**每 rank 的池压力**;此前 DP 下不成立只是 dp=8 把负载摊薄了。
- **(2) 成立**(三配置 +64%~+77%)。**MTP 配置排序**:[2,1,3] 与 [3,1,4] 截尾均值持平(239.8 vs 239.7),但 [2,1,3] accept 更高(0.84 vs 0.72)、mean−median 差更小(25 vs 41,尾部慢样本更轻)→ **综合仍推荐 [2,1,3]**;[1,1,2] 牺牲 ~7% 吞吐换取最稳定的延迟(accept 0.97,mean≈median)。注意 DP 下 [3,1,4] 未像 TP 那样崩到垫底——每 rank 仅 6 请求时 buffer 8192 不构成容量挤压,TP 的反噬机制在此不生效。
- **(0) 在 64k 极端负载下证据不足**:baseline_mtp 的纯 decode 尾窗只剩 1 个请求(n=11、accept 0.45),full_window 146.2 vs 98.1——池饱和时 MTP 的额外 KV 占用反噬 baseline 的并发容量,不下结论。
- 全部配置 `errors_in_log=0`。
