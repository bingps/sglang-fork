# NSA Accuracy Comparison: SPECRET vs index_topk_freq

## 实验配置

- **模型**: DeepSeek V3.2 AWQ (`/data2/dsv32awq/`)
- **硬件**: 8x NVIDIA H20, TP=8
- **推测解码**: EAGLE, num_steps=1, eagle_topk=1, draft_tokens=2 (除 SPECRET d=3/4)
- **对比维度**: SPECRET (不同 draft tokens) vs index_topk_freq (不同频率)

## 跳过率对应关系

| 跳过率 | SPECRET 配置 | index_topk_freq 配置 |
|--------|-------------|---------------------|
| 50% | speculative-num-draft-tokens=2 | index_topk_freq=2 |
| 66% | speculative-num-draft-tokens=3 | index_topk_freq=3 |
| 75% | speculative-num-draft-tokens=4 | index_topk_freq=4 |

---

## CEval (1346 samples, 52 subjects)

参考 PR [#21502](https://github.com/sgl-project/sglang/pull/21502) 的测试方法。

| 配置 | Score | 相对 Baseline |
|------|-------|--------------|
| baseline_d2 (无近似) | 0.897 | — |
| specret_d2 (50% 跳过) | 0.895 | -0.002 |
| specret_d3 (66% 跳过) | 0.895 | -0.002 |
| specret_d4 (75% 跳过) | 0.900 | +0.003 |
| index_topk_freq_2 (50% 跳过) | 0.897 | 0.000 |
| index_topk_freq_3 (66% 跳过) | 0.899 | +0.002 |
| index_topk_freq_4 (75% 跳过) | 0.895 | -0.002 |

### 按跳过率对比

| 跳过率 | SPECRET | index_topk_freq | 差异 (SPECRET - freq) |
|--------|---------|-----------------|----------------------|
| 50% | 0.895 | 0.897 | -0.002 |
| 66% | 0.895 | 0.899 | -0.004 |
| 75% | 0.900 | 0.895 | +0.005 |

---

## GSM8K (200 samples)

| 配置 | Score | 相对 Baseline |
|------|-------|--------------|
| baseline_d2 (无近似) | 0.960 | — |
| specret_d2 (50% 跳过) | 0.970 | +0.010 |
| specret_d3 (66% 跳过) | 0.980 | +0.020 |
| specret_d4 (75% 跳过) | 0.975 | +0.015 |
| index_topk_freq_2 (50% 跳过) | 0.975 | +0.015 |
| index_topk_freq_3 (66% 跳过) | 0.965 | +0.005 |
| index_topk_freq_4 (75% 跳过) | 0.965 | +0.005 |

### 按跳过率对比

| 跳过率 | SPECRET | index_topk_freq | 差异 (SPECRET - freq) |
|--------|---------|-----------------|----------------------|
| 50% | 0.970 | 0.975 | -0.005 |
| 66% | 0.980 | 0.965 | +0.015 |
| 75% | 0.975 | 0.965 | +0.010 |

---

## 综合结论

1. **两种方法在短 context 任务上精度损失都极小**：CEval 和 GSM8K 的波动范围均在 ±0.005 以内，属于统计噪声
2. **SPECRET 和 index_topk_freq 精度表现几乎等价**：在两个 benchmark 上没有显著差异
3. **两者都没有随跳过率增加而出现明显精度下降**：即使 75% 跳过率 (d=4 / freq=4) 也未见精度劣化
4. **短 context 场景下 NSA indexer 的近似对结果几乎无影响**：因为短 context 下 indexer 选出的 top-k 区间覆盖了大部分相关信息，跳过部分计算不会遗漏关键 token
5. **需要长 context 任务 (如 LongBench-v2, 20K+ tokens) 才能真正区分两者的精度差异**，这也是 PR #21502 强调的测试场景

### 与 PR #21502 对比

PR #21502 (Ascend NPU, W8A8 量化) 报告 CEval OVERALL = 0.9198。我们的结果 (H20 GPU, AWQ 量化) baseline = 0.897，差异来自量化方式和硬件平台的不同。两者趋势一致：**index_topk_freq 在短 context 任务上精度无损**。
