# Pure MTP ↔ HiSparse+MTP 运行时切换

## 1. 范围与目标

本文记录 `--enable-hisparse-mtp-hybrid` 的设计、实现与实测结果。

目标场景:同一个 server 在低负载时走 **pure MTP**(KV 全驻留 device,绕开
HiSparse 的 per-step swap-in 开销),KV 压力升高时切到 **HiSparse+MTP**(host
offload 换并发),负载回落再切回。

前置事实(来自 `hisparse_with_mtp.md` 的 Test 12/14/15):池未饱和时 HiSparse
比 plain 慢 19–28%,池饱和后 HiSparse 才靠更大的 logical 池反超。这个交叉点正
是本特性要自动跨越的东西。

适用范围与 HiSparse+MTP 相同:DeepSeek V3.2(非 DSV4)、EAGLE/EAGLE3、线性
draft chain、非 PD。额外限制见第 8 节。

## 2. 架构决议

### 2.1 全局批级模式 + 在途 KV 迁移

切换粒度是**整批**:任一时刻整个 running batch 处于同一模式,因此**不存在混合
模式的 batch**。模式翻转时,在途请求做 **KV 迁移**而不是排空等待。

被否决的替代方案:

| 方案 | 否决理由 |
| --- | --- |
| per-request cohort + 混合 batch | CUDA graph 只能整批捕获,per-row 条件分支无法表达;且少数派 cohort 要等长生成结束才排空,共存窗口可达数千 step |
| 排空后再切换 | 长生成请求会把切换无限期推迟,失去自适应意义 |
| 重启/双实例 | 不解决单实例内的负载波动 |

### 2.2 resident 表示:保留 mapped 布局,不做 identity 重写

两种模式都保持 `req_to_token` = **logical id**,mapping 始终有效;迁移只改
mapping 与物理槽归属,**`req_to_token` 全程不重写**。

这是与参考实现(`hisparse-mtp-squashed-v2` 分支)最关键的分歧。v2 用
`set_mtp_mode(True)` 让 allocator 单路径 + `mapping[x]=x`,代价是迁移时必须重写
`req_to_token`。v10 不能这么做:

- `req_to_token` 同时被 draft(NextN)worker 用于寻址**它自己的** KV(draft 池按
  target 的 logical id 平铺,见 `kv_cache_configurator._build_dsa_kv_pool`)。重写
  会让 draft KV 整体错位 → accept 率静默下降。v2 实测 accept 0.469→0.452 的下滑
  很可能正是此因;v2 能容忍是因为它的 HiSparse 模式关掉了 MTP(`steps=0`),而
  v10 两种模式都跑 draft。
- radix / prefix 复用同理。

推论:resident 与 offloaded 的差异被压缩为**纯 mapping 语义差异**:

```text
resident   : logical id ──mapping──> 专属物理槽(终身有效,无 host 副本)
offloaded  : 热窗口 logical id ──mapping──> device buffer / spec ring 槽
             超出 ring 窗口的已提交 token,mapping 被清 0,KV 只在 host,
             verify 经 full_to_token_position 走 host DMA
```

再推一步:**"pure MTP 模式" ≡ "prefill 后不 admit"**。因为 v10 的
`alloc_extend` 本来就同时分配 logical + physical 并写 mapping,一个刚
prefill 完、尚未 `admit_request_into_staging` 的请求,天然就是 resident 布局。
所以 resident 侧**不需要任何新的分配原语**。

### 2.3 单图分派:residency 由 kernel 按数据区分

verify 恒挂 coordinator、恒走 `swap_in_verify_pages`——**只捕一张图,且不需要任何
kernel 改动**。residency 的区分由 swap-in kernel
(`python/sglang/kernels/jit/csrc/hisparse.cuh`)**既有的**一级 `direct_loc` 查找
天然完成:

- 该查找对 mapping 有效的条目直接输出 `mapping[logical]`。resident 请求的 mapping
  覆盖每个 token,因此每个条目都在这一级解决,**等价于 translate gather**;
- 全解析(零 miss)时,后续 buffer 扫描的命中标记只写共享内存,LRU 回写把同一
  置换写回,`:796` 的 mapping 清零与 `:798` 的身份表改写都在 miss 分支内、不执行
  ——**resident 行零状态改动**(由
  `test_resident_verify_swap_in_is_pure_gather` 锁定)。

推论:**混合 batch 天然合法**(逐行按数据区分),批级 attach 不再需要按模式门控。
调度侧的模式门控仍然保留——它管的是分配路径(resident 走组合分配、无 ring、无
增量 backup),与 attention 分派解耦。

历史注:初版用"resident 批不挂 coordinator → 落 `translate` 分支"+ 双份捕获图
实现,后被本方案取代;期间还尝试给 kernel 加"全解析早退"以跳过无用扫描,实测
无收益后撤回(见 §6.1)。`force_unfused_topk` 只对 `is_decode_or_idle()` 生效,
verify 不受影响,两模式 topk 路径一致。

## 3. 状态机与不变量

`DecodeMode`(`speculative/hybrid_mode_controller.py`)四态:

```text
        offload 全部在途请求(净释放物理槽,不会因容量失败)
   MTP ───────────────────────────────────────────────> HISPARSE
    ^                                                      │
    └──────────────────────────────────────────────────────┘
        restore 全部在途请求;任一失败 → 回滚已恢复者,留在 HISPARSE

   PENDING_OFFLOAD / PENDING_RESTORE:为将来的异步分阶段迁移预留,
   当前实现的迁移在单步内完成,不进入这两态。
```

两个派生谓词(`hisparse_coordinator.py:89,99`),非 hybrid 时恒为 True,保证
既有 HiSparse+MTP 行为完全不变:

- `should_admit_new()` — 新 prefill 完成的请求是否 offload
- `attaches_to_batch()` — 本批是否走 offloaded 路径

不变量:

1. **批次同构**:`on_step` 返回后,所有在途请求都处于返回模式所蕴含的布局。
2. **`req_to_token` 不可变**(见 2.2)。
3. **先分配后释放**:restore 分配失败时零副作用。
4. **迁移前与 forward 流排序**(见 6.2)。
5. **只在恢复后不会立刻反弹时才恢复**(见 6.3)。

## 4. 实现清单

| 文件 | 内容 |
| --- | --- |
| `speculative/hybrid_mode_controller.py` | 新增。`DecodeMode` + `HybridModeController`:`decide_stable_target`(:162)双信号+滞后+cooldown、`on_step`(:201)、`_switch_to_hisparse`(:231)、`_projected_resident_usage`(:254)、`_switch_to_mtp`(:278) |
| `managers/hisparse_coordinator.py` | `should_admit_new`/`attaches_to_batch`(:89,:99);`admit_request_into_staging` 增 `num_tokens` 参数(:304 基类,:1172 MTP);`_wait_for_forward_stream`(:1231);`offload_running_request`(:1241);`restore_running_request`(:1285) |
| `managers/scheduler.py` | `init_hisparse_coordinator` 构造控制器并挂到 coordinator;`on_step` 调用点(:2834)在 coordinator 挂载(:2839)之前;`_hybrid_kv_pool_usage`(:2904) |
| `managers/schedule_batch.py` | `ScheduleBatch.hisparse_resident`(仅供调度侧分配门控) |
| `speculative/eagle_utils.py` | fail-fast 对 resident 放行(:895);verify 的 coordinator attach 保持上游无条件形态 |
| `speculative/eagle_worker_v2.py` | prefill admission 按 `should_admit_new()` 门控 |
| `scheduler_components/batch_result_processor.py` | 非 spec admission 同样门控 |
| `server_args.py` / `arg_groups/hisparse_hook.py` | 6 个 flag + 校验 |

`request_finished` **不需要**按模式分流:它对 resident 请求天然正确
(`current_cap==0` 跳过 buffer、ring 未激活、host 为空,只按 mapping 释放物理页
并清 mapping,随后 `release_kv_cache` 的 `free_hisparse` 因 mapping 已清而不重复
释放)。已由单测 `test_resident_request_finished_no_leak` 固化。

### 4.1 offload 迁移

`offload_running_request` 复用 admission 路径,只做两处调整:

1. **只备份已提交前缀**。running 请求已生成超出 prompt 的 token,而未提交的前瞻
   位置持有被拒 draft 的陈旧 KV,不能进 host;backup 高水位必须落在 committed,
   `backup_committed_tokens` 才能从正确位置续上。
2. **前瞻尾部 `[committed, allocated)` 转指 spec ring**。这些位置会被下一轮 draft
   复用,而 admission 内的 `alloc_device_buffer`(跨整个 allocated 长度)刚释放了
   它们的 resident 物理槽并清了 mapping。

**刻意不做尾部的部分页释放**:`alloc_device_buffer_mtp` 已按页释放且排除与 buffer
共页,已提交前缀的边界页因此保持完整。v2 正是在这里踩了三个崩溃
(`c945b663e` slot conflict/页对齐、`a6e932d89` page 0 进 free list、
`admit_mtp_request` 的 last-page reclaim hack)。

### 4.2 restore 迁移

`restore_running_request` 是唯一新增的数据通路:

1. 先分配全长(页对齐)物理槽;失败(必要时先 `_reclaim_deferred_staging_pages`
   重试)则返回 False,**零副作用**。
2. 让 host 成为已提交前缀的唯一真源:`[last_backed, committed)` 仍只在 device
   (buffer 或 ring——ring 断言保证槽在备份前不会被回收),先把它补备份到 host。
3. host → 新槽逐层加载。未提交的前瞻尾部只要槽、不要数据(下一轮会覆写)。
4. `mapping[compressed] = new_slots`,再按页释放 buffer + ring,释放 host 槽。
5. 清 6 处元数据:buffer 三张表、host pool 两张表、spec ring、
   `full_to_token_position`、`hisparse_last_backed_len` / `hisparse_ring_start` /
   `_skip_first_backup`。

对比 v2 的 `restore_to_standard`:它**先 free 后 alloc**,分配失败时只 log 就
return,请求 KV 被孤立。本实现反转了顺序,并在控制器层对部分成功的请求**回滚
重新 offload**,保证批次永不混合。

## 5. 配置

```
--enable-hisparse-mtp-hybrid        # 隐含 --enable-hisparse,要求 --speculative-algorithm
--hisparse-mtp-usage-up      0.6    # device 用量上阈 → offload
--hisparse-mtp-usage-down    0.3    # 下阈 → restore(按投影用量判定,见 6.3)
--hisparse-mtp-min-bsz       8      # bsz 低于此不 offload(swap-in kernel 是 per-req one-block,小 bsz GPU 利用不足)
--hisparse-mtp-max-bsz-for-mtp 4    # bsz 低于此强制 MTP 保延迟
--hisparse-mtp-cooldown-steps 10    # 切换后抑制步数
```

决策信号取**物理 device pool 用量**(`_hybrid_kv_pool_usage` →
`coordinator.get_token_stats().device_token_usage`),**不用**调度器的
`full_token_usage`:后者以 logical 池为分母(大 `host_to_device_ratio` 倍),会
严重低估真正约束驻留的压力。

调试用:`SGLANG_FORCE_HISPARSE_MTP_MODE=mtp|hisparse` 可把模式钉死,用于分别
测量两条路径。

## 6. 实现中发现并修复的三个问题

### 6.1 CUDA graph 把 Python 分支烙死

`decode_cuda_graph_runner` 原本**无条件**挂 coordinator("Trip the coordinator so
the hisparse code path is captured into the graph"),而 resident/offloaded 的分野
是一次 **Python** 判断。CUDA graph 只录制捕获时实际执行的 kernel,所以捕获出的
verify 图**恒含 swap-in**:resident 批在 replay 时仍跑 swap-in,性能收益归零,且
`req_device_buffer_size==0` + 无 host 副本会让 top-k 全部 miss 后去读未分配的
host 区域。

Phase 2 的门控只影响 eager 路径,对 CG replay 无效——**Phase 3 因此从"可选优化"
升级为必需项**。

发现过程:首轮冒烟两种 forced mode 都 PASS,这本身可疑(两条路径行为应当不同)。
确认图确实被 replay(`cuda graph: True`)后读捕获路径,找到无条件挂载。两边都过
是因为 burst 探针 prompt 只有几十 token 而 `device_buffer_size=8192`,KV 完整装进
buffer,swap-in 走短序列快路径直接返回 buffer 位置、根本不碰 host——**测试自己
掩盖了差异**。这直接决定了后续用 `--disable-cuda-graph` 解耦、并用 30k prompt 让
两条路径真正分岔。

**最终修复(单图)**:verify 恒挂 coordinator、恒走 swap-in,residency 由 kernel
既有的一级 `direct_loc` 查找按数据区分(见 §2.3)。勘察发现 kernel 对 mapping 全
有效的 resident 行本就等价于 translate gather,且零 miss 时 `:796` 的 mapping 清零
与 `:798` 的身份表改写都不执行——**正确性上无需改 kernel**。

演进过程(每一步都被后一步取代,保留以说明取舍):

1. 双份捕获 + `ShapeKey.hisparse_resident` 维度;
2. 同上,但 residency 折入既有 `variant_label`(核实该字段无任何消费者、
   `get_capture_lora_variant()` 无调用点,故为纯不透明判别符),`shape_key.py`
   回归零差异;
3. 单图 + kernel 加"全解析早退"(padding + direct hit == top_k 时跳过 buffer
   扫描 / LRU 回写 / miss DMA),意图省掉 8448 槽 × N 位置 × 61 层的无用扫描;
4. **单图、零 kernel 改动**(当前):早退经实测**收益为零**,撤回。

早退为何无效(反直觉,值得记住):它省掉的 buffer 扫描是 8448 槽的**纯共享内存
并行标记**,在 61 层模型前向里根本不是瓶颈;而它**无法**省掉的哈希建表
(`NUM_TOP_K` 次 `atomicCAS`)才是 resident 相对专用 gather 那约 0.6 tok/s 差距的
来源——优化加在了错误的位置。教训与 §7.4 的三次纠错同源:**"看起来是无用功"
不等于"值得优化",必须先测**。

结果:`shape_key.py`、`decode_cuda_graph_runner.py`、`forward_batch_info.py`、
`model_runner.py`、`hisparse.cuh` **全部与上游零差异**,双捕获机制、
`_compose_variant_label`、`ForwardBatch.hisparse_resident` 全部移除。净代码量为负、
零 CUDA 风险,且**混合 batch 从此天然合法**,解除了 §8.2 渐进迁移的前置依赖。

> 已废弃的注意事项(仅对方案 1/2 成立):residency 必须在 `can_run_graph` 之前
> 赋值,否则查错变体。单图下不再存在此陷阱。

### 6.2 迁移未与 forward 流排序

overlap 调度下前向跑在 `decode_producer_stream`,调度线程同时准备下一步。迁移恰好
干两件危险事:**重写 mapping**、**把物理页还给 allocator**(可能被别的请求立刻占用
覆写)。既有 `request_finished` 对同类改动等了**两个**东西——forward 流与 backup
流;而初版迁移只等了 backup 流(offload 连这个都没等),构成 use-after-free /
撕裂读。这类 bug 表现为 KV 悄悄读错(accept 率下滑)而非崩溃,与之前的 draft-pool
bug 同族。

修复:抽出 `_wait_for_forward_stream()` 复刻该纪律,在 offload/restore 开头各调
一次。是 GPU 侧 `wait_stream` 排序(入队依赖),不是 CPU 阻塞排空。

发现方式不是测试,而是审接入时序时对照 `request_finished` 的保护强度。

### 6.3 阈值语义导致剧烈震荡

原规则(沿用 v2):`usage <= down` → restore。缺陷在于 **offload 本身就是降低
device 用量的手段**:8×30k 驻留时用量 0.505 越过上阈 → 全部 offload → 每请求占用
从约 30000 槽塌缩到 `buffer(8192)+ring` → 用量掉到 0.146,**低于下阈 0.25** →
判定该切回 → restore → 弹回 0.505 → 再 offload。

实测几秒内 **72 次 offload + 72 次 restore**,吞吐被迁移开销拖到 108.7 tok/s。

**cooldown 治不了**:它只限速,而震荡源于稳定的不动点违背——只要
`offloaded_usage < down < up < resident_usage` 恒成立,滞后带无论多宽都无解;而
"把带调宽"也绕不过,因为 offload 释放大量显存正是设计目的,两状态间的鸿沟是内生的。

修复:下行判据改用**反事实**——`_projected_resident_usage` 估算"若恢复则用量为何":

```text
capacity  = hisparse_attn_allocator.size
used      = capacity - available_size()
extra     = Σ_live max(0, kv_allocated_len - (device_buffer_size + ring))
projected = (used + extra) / capacity
```

`projected >= usage_threshold_up` 时拒绝恢复,使不变式自洽:**只恢复到不会立刻
反弹的状态**。

根因反思:计划里本已写明"restore 前做 post-restore 用量估算",但初版实现只检查了
**容量**(分配能否成功),没检查**稳定性**(会不会弹回)——两者是不同条件,被混为
一谈。

## 7. 实验结果

公共配置:`.models_dsv32`、TP=8、`--mem-fraction-static 0.85`、`page_size=64`、
EAGLE `[2,1,3]`、`hisparse-config {"top_k":2048,"device_buffer_size":8192,
"host_to_device_ratio":2}`。长上下文用 `longbench_prompts_30k.jsonl`。

### 7.1 两条路径各自的正确性(模式钉死)

| 场景 | resident (`FORCE_MODE=mtp`) | offloaded (`FORCE_MODE=hisparse`) |
| --- | --- | --- |
| 短 burst + CG on | BURST PASS,0 error | BURST PASS,0 error |
| 短 burst + eager | BURST PASS,0 error | BURST PASS,0 error |
| 30k,eager,c=2 | 13.3 tok/s | 12.6 tok/s |
| 30k,CG on,c=4 | 4/4,**41.1 tok/s**,0 error | 4/4,**38.9 tok/s**,0 error |

- resident 始终略快(+5.6% eager / +5.7% CG),方向与"省掉 swap-in"一致;c=2–4
  的低并发下这不是强性能信号,但足以证明 resident 路径不劣化。
- CG vs eager 41.1 vs 13.3 说明图确实在用。
- 上表 CG on 各行取自 Phase 3 **之后**的 `run_hybrid_cg.sh`。Phase 3 之前的
  `run_hybrid_smoke.sh` 同样两模式全 PASS,但那**不构成 resident 路径的证据**:当时
  捕获的图恒含 swap-in(见 6.1),resident 批实际跑的仍是 swap-in,只因短 prompt 的
  KV 完整装进 device buffer 而侥幸正确。长上下文才有区分力。

日志: `logs/20260728_211222_hybrid_smoke.txt` (Phase 3 前,仅存档)、
`logs/20260728_212508_resident_nocg.txt` (eager)、
`logs/20260728_213638_hybrid_cg.txt` (CG on,双变体)。

### 7.2 双变体 CUDA graph 捕获成本

启动耗时(TP=8,含权重加载)。双变体阶段:235s / 285s / 235s / 240s;单变体
(Phase 3 之前)220s / 265s,即双份捕获约 **+15–20s(~7%)**,`--max-bs 32` 下
未触发 capture OOM。

**单图方案落地后此项成本消失**:恒捕一张图,实测 215s / 260s,回到单变体水平。
功能与性能均无回退,且三版对照证明 kernel 早退无收益(`--max-bs 32`,long30k
c=4;`logs/20260729_195301_hybrid_cg.txt` 、`logs/20260729_202211_cg_*.log` ):

| 版本 | resident | offloaded | kernel 改动 |
| --- | --- | --- | --- |
| 双图(resident 走专用 translate gather) | 41.2 | 39.2 | 无 |
| 单图 + 全解析早退 | 40.5 | 39.4 | 3 行 |
| **单图、零 kernel 改动**(当前) | **40.6** | **39.0** | **无** |

- 早退版 40.5 vs 无早退版 40.6 → **收益为零**,故撤回(原因见 §6.1)。
- 单图 resident 比双图低约 0.6 tok/s(41.2 → 40.6):resident 行多付一次哈希建表,
  但省掉一个独立 gather kernel;换来单图、混合 batch 合法与零 kernel 风险,划算。
- 三档 burst 全 PASS、long30k 全 4/4、server error 全 0。
- 动态切换(§7.3 负载)同样无回退,且**迁移路径必须单独复验**——钉死模式不触发
  offload/restore,故 CG 段的等价性推不到迁移:双图 137.4 → 早退版 136.2 →
  无早退版 **135.7** tok/s,三者均 8/8、双向各 1 次转换、0 error
  (`logs/20260729_203155_hybrid_dynamic.txt` )。

### 7.3 动态切换(不钉模式,控制器自主决策)

负载:8 并发 × 30k prompt × 512 token。阈值 up=0.5 / down=0.25 / min_bsz=2 /
max_bsz_for_mtp=1 / cooldown=10。

| 指标 | 修复 6.3 之前 | 修复之后 |
| --- | --- | --- |
| MTP→HiSparse 转换 | 72 | **1**(usage=0.505, bsz=8) |
| HiSparse→MTP 转换 | 72 | **1**(usage=0.091, bsz=5,压力排空后) |
| 爬坡吞吐 | 108.7 tok/s | **137.4 tok/s**(+26%) |
| 请求成功率 | 8/8 | 8/8 |
| 爬坡后 burst | PASS | PASS |
| server error | 0 | 0 |

切换瞬间的可观测量(修复前日志,证明 KV 真的搬动了):

```text
Decode batch  gpu token usage: 0.51  cpu token usage: 0.00  accept rate: 0.73
Decode batch  gpu token usage: 0.15  cpu token usage: 0.25  accept rate: 0.77
```

**accept rate 全程 0.73–0.80,未因迁移下降**——这是最关键的健康指标,因为元数据
漏清恰好表现为静默降 accept(与之前 draft-pool bug 同族)。

吞吐 +26% 的来源是消除了震荡带来的迁移 churn,而非计算路径变快。

日志: `logs/20260728_214704_hybrid_dynamic.txt` (修复前)、
`logs/20260728_215429_hybrid_dynamic.txt` (修复后)。

### 7.4 端到端动态负载与成本分解(后续战役,含三次方法学纠错)

目标:同一条时变负载(轻→重→轻)下,hybrid 聚合是否同时优于 static pure-MTP 与
static HiSparse。过程踩了三个测量陷阱,记录在案以免复犯:

1. **负载必须 decode 主导**。首轮 30k×256tok 的 prompt:decode=123:1,客户端
   "decode_tps"(分母含 prefill)实际在测 prefill 速度。自检指标:prompt:decode
   与 `total_tps`。
2. **PURE_DECODE 窗口对容量受限的 wave 失效**。有队列时调度器全程穿插 prefill,
   "最后一条 Prefill 之后"的窗口塌缩到排空尾部(仅占 wave 的 10–14%),且因尾部
   并发变少而**系统性偏低**(曾把 -5% 的真实差距放大成 -34%)。修正:解析器同时
   报 FULL 与 PURE 窗口及窗口内 prefill 行数(`parse_wave_decode.py`)。
3. **decode backend 混淆(影响最大)**。`--enable-hisparse` 强制
   `dsa_decode_backend=flashmla_sparse`,而纯 DSV3.2 默认 `fa3`;trace 归因显示
   两者 decode 跑的是**不同 attention kernel**(`sparse_attn_fwd` 3.388 vs
   `FlashAttnFwdSm90` 组 1.315 ms/步)。此前一切 "HiSparse/resident 开销" 数字都
   被它污染。

c1–c8 扫描(30k、[1,1,2]、buf=8192)+ 对齐 backend 对照后,每步成本分解为三层:

| 分量 | ms/步(均值) | 依据 |
| --- | --- | --- |
| decode backend 税(sparse vs fa3) | **2.96** | pure-MTP 两种 backend 对照 |
| resident 路径(mapping/池/分配器) | **1.09** | backend 对齐后 resident − pure |
| swap-in(offloaded 相对 resident) | **3.78** | 70k light 相 23.01 − 19.23 |

排查中被数据否掉的假设(均有对应实验):indexer buffer 2×(h2d=1 对照,delta 不
变;它只是 +3.7 GB 显存的空间成本)、eager 回退(两侧 0 eager)、translate gather
(bsz=1 时 16 KB,量级差 30 倍)、图外分配器双路径(GPU busy ~100%,无主机停顿)。

最终四组对照(70k prompt,light 3×4096 / heavy 24×6144 / light 3×4096,
hybrid 为 up=0.8/down=0.5;`logs/20260729_165606_e2e_aligned.txt` ):

| 配置 | light1 | heavy | light2 | E2E 总计 | 加权 decode tok/s |
| --- | --- | --- | --- | --- | --- |
| mtp_fa3(部署基线) | 60.6s | 526.2s | 60.4s | **647.1s** | 503.1 |
| mtp_sparse(机制对照) | 66.6s | 567.9s | 65.3s | 699.8s | 439.7 |
| hisparse_static | 77.0s | 544.3s | 77.5s | 698.8s | 451.2 |
| **hybrid** | 68.6s | 533.9s | 67.0s | **669.6s** | 470.1 |

结论:

- **机制成立**:同 backend 下 hybrid 同时优于两个基线(vs mtp_sparse **+4.5%**
  E2E / +6.9% decode;vs hisparse_static **+4.4%** / +4.2%)。轻载贴住 pure MTP
  (只差 resident 路径 ~1.1ms/步),重载吃到容量优势。
- **容量优势在吞吐上兑现的前提是 backend 对齐**:mtp_sparse 的 heavy(567.9s)
  劣于 hisparse_static(544.3s)——此前"HiSparse 赢不了吞吐"的结论一半是
  backend 税的假象。
- **与部署基线的 -3.4% 全部是 backend 税**,对应的杠杆见 §8。
- 阈值语义:驻留成立条件是 `projected_resident_usage < up`,`up` 必须高于工作集
  的驻留占比(70k 下 3×76k/460k=0.496,up=0.5 会把驻留封死;提到 0.8 后轻载相
  真驻留,heavy 相还额外获得约 19% 时长的机会性驻留窗口)。

## 8. 已知限制与后续工作

0. **residency 感知的 decode backend 选择(当前最大单一杠杆,~2.96 ms/步 ≈ 19%)**。
   `HISPARSE_SPEC_VERIFY_BACKENDS` 白名单存在的理由是 swap-in 只在那几个 backend
   的 verify 路径里实现,而 resident 批次不做 swap-in,约束对它并不适用——若
   resident 批次改用 `fa3`,与部署基线(mtp_fa3)的 -3.4% E2E 差距可基本抹平。
   代价是再来一层"双 backend + 双份捕获图"(与 §6.1 的 residency 图变体同构),且
   需先验证 fa3 能否直接消费 HiSparse 池的物理 KV 布局。更便宜的先行验证:
   `kv_cache_dtype=fp8_e4m3` 时 hisparse 走 `flashmla_kv`,backend 税可能不同。
   注:identity 分配 / 去 translate-gather 一度被当作首要优化,实测其收益上限只有
   resident 路径总共 1.09 ms/步 的一部分(≤7%),**优先级降级**。

1. **不支持 DP attention**,启动期显式拒绝。各 rank 池独立,压力信号不同,模式
   决策会分歧;某 rank offload 而其余驻留会导致每步 forward 数 / token 数不一致,
   死锁 MLP-sync 集合通信(v2 遗留未解的 "DP 反复切换 hang")。选择**显式报错而非
   静默挂死**。
   落地方案:在 `MLPSyncBatchInfo._get_local_tensor` / `_get_fallback_tensor`
   (`scheduler_components/dp_attn.py`)的 all-gather 里搭车一个 want-hisparse
   标量,以 any() 归约,并让 `on_step` 在步 N+1 消费步 N 的同步结果,使各 rank
   同步迁移;随后移除该护栏并回归"反复切换不 hang"。
   注:本方案两模式同为 `TARGET_VERIFY`、每请求 token 数相同,v2 那一堆
   `global_num_tokens` / `dp_gather` 修补**都不需要**,而 v2 的 hang 很可能正源于
   那些 shape 分歧,故天然规避——但仍需实测确认。
2. **迁移是全批原子的**。长上下文下 restore 的 H2D 量级为"全长 KV × 层数",且窗口
   期同时持有 buffer 与新全长物理槽。当前靠"投影用量拒绝放不下的恢复"保证安全,
   但未做限流。**前置依赖已解除**:单图方案让混合 batch 天然合法(§2.3),渐进
   迁移不再需要额外的 attention 侧改动,只需在控制器里按字节预算分批迁移。
3. **`PENDING_OFFLOAD` / `PENDING_RESTORE` 两态目前未被使用**,为异步分阶段迁移
   预留(与上一条同批实现)。
4. **CUDA graph 只捕一张,且零 kernel 改动**:`shape_key.py`、
   `decode_cuda_graph_runner.py`、`forward_batch_info.py`、`model_runner.py`、
   `hisparse.cuh` 均与上游零差异;residency 由 kernel 既有的 `direct_loc` 一级
   查找按数据区分(§2.3/§6.1)。代价是 resident 行多走一次哈希建表(约
   0.6 tok/s)。
5. 阈值默认值未做系统调参;7.3 用的是为快速跨越阈值而收窄的实验值。

## 9. 测试与复现

单测 `test/registered/unit/managers/test_hisparse_unit.py`(36 passed / 2 skipped):

| 用例 | 覆盖 |
| --- | --- |
| `test_hybrid_mode_gating_predicates` | 四态 × 两谓词;非 hybrid 恒 True |
| `test_resident_request_finished_no_leak` | resident 请求释放无泄漏、无双重释放 |
| `test_offload_running_request_migrates_resident_req` | offload 后 buffer/ring 就位、只备份 committed、高水位与 ring 锚点正确、尾部映射进 ring、零泄漏 |
| `test_offload_then_restore_round_trip` | 往返后 KV 数据完好(逐层比对)、每位置独立物理槽、6 处状态全清、零泄漏 |
| `test_hybrid_controller_projection_breaks_pingpong` | 重载拒绝恢复 + 真正排空仍恢复(双向锁死 6.3 的修复) |
| `test_resident_verify_swap_in_is_pure_gather` | 单图契约:resident 行经 swap-in kernel 输出 == `mapping[logical]`,且 LRU / 身份表 / mapping 零改动 |

端到端脚本(日志统一落 `logs/{timestamp}_{prefix}.log`):

```
bash run_hybrid_smoke.sh      # 两模式钉死 + 短 burst(CG on)
bash run_resident_nocg.sh     # --disable-cuda-graph,解耦 CG,含 30k 长上下文
bash run_hybrid_cg.sh         # CG on + 30k,验证双变体捕获
bash run_hybrid_dynamic.sh    # 不钉模式,验证真实双向切换与震荡次数
```

§7.4 战役的负载、解析与归因工具:

```
bench_dynamic_load.py         # 分相 wave 负载,输出精确时间窗(t_start/t_end)
parse_wave_decode.py          # 按窗切分,FULL+PURE 双窗口,ms/step 换算
run_hybrid_e2e.sh             # 三配置 E2E 动态负载
run_hybrid_thresh.sh          # 只重跑 hybrid 的阈值对照(up/down)
run_resident_ab.sh            # pure MTP vs resident 的并发扫描(c1..c8)
run_resident_h2d1.sh          # h2d=1 判伪 indexer-buffer 假设
run_profile_ab.sh             # /start_profile 抓 c=1 trace(两配置)
analyze_trace.py              # GPU busy/idle 归因(图内 vs 主机停顿)
diff_traces.py                # 限定注解区域的逐 kernel 差分
run_backend_match.sh          # pure MTP 对齐 flashmla_sparse 的分解对照
run_e2e_aligned.sh            # §7.4 终表:四配置对齐口径汇总
```
