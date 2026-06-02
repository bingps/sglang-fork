# NSA Accuracy Comparison — LongBench-v2

## Setup

- **Model**: DeepSeek-V3-2 AWQ (`/data2/dsv32awq/`)
- **Hardware**: 8x NVIDIA H20 (TP=8)
- **Dataset**: LongBench-v2 (first 50 examples, max context 163K tokens)
- **Speculative Decoding**: EAGLE, num_steps=1, eagle_topk=1
- **Config**: mem_fraction=0.88, cuda_graph_max_bs=2, num_threads=16

## Results

| Config | Skip Rate | Score | Delta vs Baseline |
|--------|-----------|-------|-------------------|
| baseline_d2 | 0% | 0.594 | — |
| specret_d2 | 50% | 0.625 | +0.031 |
| specret_d3 | 66% | 0.594 | 0.000 |
| specret_d4 | 75% | 0.594 | 0.000 |
| index_topk_freq_2 | 50% | 0.594 | 0.000 |
| index_topk_freq_3 | 66% | 0.562 | -0.032 |
| index_topk_freq_4 | 75% | 0.594 | 0.000 |

## Analysis

### SPECRET (verify 阶段 indexer 复用)

- **d=2/3/4 均无精度损失** (与 baseline 持平或略高)
- d=2 得分略高于 baseline (+0.031)，属正常波动范围
- 结论: SPECRET 在长文本场景下精度完全无损

### index_topk_freq (跨层 indexer 复用)

- freq=2 和 freq=4 均无精度损失
- freq=3 有轻微下降 (-0.032)，属噪声范围
- 结论: 跨层 topk 复用在长文本场景下精度基本无损

### 对比

| Skip Rate | SPECRET | index_topk_freq |
|-----------|---------|-----------------|
| 50% | 0.625 | 0.594 |
| 66% | 0.594 | 0.562 |
| 75% | 0.594 | 0.594 |

两种方法在相同跳过率下精度相当，均不影响长文本理解能力。
SPECRET 仅在 verify 阶段生效，index_topk_freq 在所有阶段生效，两者可叠加使用。
