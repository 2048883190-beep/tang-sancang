# 评测说明（EVALUATION）

## Primary test

- 冻结测试集：**685** 题
- 温度：**0**
- 主方法：**Selective RAG v1**

## 结果表

| 方法 | 正确 | 准确率 | 角色 |
|---|---:|---:|---|
| Direct | 609/685 | 88.9% | 闭卷对照 |
| Naive Dense RAG | 618/685 | 90.2% | **baseline** |
| Current BSCV | 605/685 | 88.3% | 前期架构对照 |
| **Selective RAG v1** | **619/685** | **90.4%** | **正式参赛结果** |

来源：`reports/contest_release/selective_v1_test_20260922_182356/selective_rag_summary.json`  
（exact=619，exact_rate=0.9036）

## 解读

Selective RAG v1 相比普通 Dense RAG **提升 1 道题**，同时显著降低 harmful retrieval rate。

相对冻结 Direct 的 harmful / helpful（口径见 `reports/contest_release/selective_rag_summary.md`）：

| 指标 | Dense | Selective v1 |
|---|---|---|
| Harmful Retrieval Rate | 19/609 = **3.12%** | 4/609 = **0.66%** |
| Helpful Retrieval Rate | 28/76 = **36.8%** | 14/76 = **18.4%** |

决策统计（同一正式测试）：`KEEP_DIRECT` 489，`USE_RAG` 196。

## 未达成项

目标 624/685 = 91.09% **未达到**。正式材料只写 **90.4%**，不写「达到 91%」。

## 锁参

`data/eval/locked_selective_rag.json`（仅 validation 锁定，禁止 test 调参）
