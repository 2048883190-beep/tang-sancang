# 技术概览

## 演进关系

```text
T2DM 专科知识库 → BGE-M3 → BSCV 三路（权威/适用/一致）
        → 发现 retrieval harm
        → Risk-Aware Selective RAG v1
        → KEEP_DIRECT / USE_RAG
        → 619/685 = 90.4%
```

- BSCV：项目技术基础与「三藏」叙事  
- Selective RAG v1：当前参赛答案版本  
- 糖三藏：正式产品名称  

## 答案路径（冻结）

1. Direct Answer（temperature=0）  
2. DenseMatcher Top-10  
3. `answer_stability` 评估相关度 / 充分性  
4. `contradiction` 冲突判断  
5. 门控：过阈 → `USE_RAG`，否则 → `KEEP_DIRECT`  

高风险题使用更高充分性门槛（见锁参 risk_*）。

## 展示路径（不改答案）

对召回段落可做三路打分展示（`parallel_fusion`），写入证据卡；  
PDF 溯源由 `web/source_open.py` 完成。

## 主要模块

| 模块 | 作用 |
|---|---|
| `src/rag/selective_rag.py` | Selective v1 主逻辑 |
| `src/rag/answer_stability.py` | 证据充分性等评估 |
| `src/rag/contradiction.py` | 冲突 / 立场 |
| `src/rag/matching_layer.py` | Dense 检索 |
| `src/rag/parallel_fusion.py` | 三路展示基础 |
| `web/pipeline.py` | 演示组装（强制 version=v1） |
| `web/source_open.py` | 原文页定位 |

详见 `docs/VERSION_FREEZE.md`、`RELEASE.md`。
