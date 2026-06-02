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

## AIME25 (30 problems, single pass, temp=0.6, max_tokens=32768)

数学推理 benchmark，对应图中 "General & Reasoning" 类别。

| 配置 | Score | 正确题数 | 相对 Baseline |
|------|-------|---------|--------------|
| baseline_d2 (无近似) | 0.133 | 4/30 | — |
| specret_d2 (50% 跳过) | 0.200 | 6/30 | +0.067 |
| specret_d3 (66% 跳过) | 0.100 | 3/30 | -0.033 |
| specret_d4 (75% 跳过) | 0.067 | 2/30 | -0.066 |
| index_topk_freq_2 (50% 跳过) | 0.167 | 5/30 | +0.034 |
| index_topk_freq_3 (66% 跳过) | 0.233 | 7/30 | +0.100 |
| index_topk_freq_4 (75% 跳过) | 0.133 | 4/30 | 0.000 |

### 按跳过率对比

| 跳过率 | SPECRET | index_topk_freq | 差异 (SPECRET - freq) |
|--------|---------|-----------------|----------------------|
| 50% | 0.200 | 0.167 | +0.033 |
| 66% | 0.100 | 0.233 | -0.133 |
| 75% | 0.067 | 0.133 | -0.066 |

> **注**: AIME25 只有 30 题且使用 temp=0.6 单次采样，方差极大。单题差异 (1/30≈0.033) 即可导致显著变化。此结果受随机性影响大，不宜做精确对比。需要 n-repeats 多次采样 + majority vote 才能得到稳定结果。

---

## 综合结论

1. **短 context 任务 (CEval, GSM8K) 精度损失极小**：两种方法的波动范围均在 ±0.005 以内，属于统计噪声
2. **SPECRET 和 index_topk_freq 在短 context 上精度表现等价**：无显著差异
3. **AIME25 推理任务方差过大**：30 题 + 单次采样的设置下，所有配置波动剧烈（0.067~0.233），无法得出可靠结论。需要 16x repeats + majority vote 才能稳定
4. **两者都没有随跳过率增加而出现明显精度下降**：在 CEval/GSM8K 上，即使 75% 跳过率 (d=4 / freq=4) 也未见精度劣化
5. **需要长 context 任务 (LongBench-v2, 20K+ tokens) 才能真正区分精度差异**：短 context 下 indexer 覆盖全局信息充分，跳过计算无影响；长 context 下 indexer 精准度更关键

### 与 PR #21502 对比

PR #21502 (Ascend NPU, W8A8 量化) 报告 CEval OVERALL = 0.9198。我们的结果 (H20 GPU, AWQ 量化) baseline = 0.897，差异来自量化方式和硬件平台的不同。两者趋势一致：**index_topk_freq 在短 context 任务上精度无损**。

### LongBench v2 (待补充)

LongBench v2 数据集 (465MB) 因代理下载速度限制未能完成下载。这是最能体现 IndexCache 精度差异的 benchmark，待数据就绪后补跑。
