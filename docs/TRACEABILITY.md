# 溯源说明（TRACEABILITY）

## 正式成绩

- Selective RAG v1：**619/685 = 90.4%**
- 报告目录：`reports/contest_release/selective_v1_test_20260922_182356/`
- 汇总：`reports/contest_release/selective_rag_summary.md`
- 锁参：`data/eval/locked_selective_rag.json`

## 对照成绩来源

| 方法 | 来源 |
|---|---|
| Direct / Naive Dense | `reports/contest_release/baseline_direct_dense/` |
| BSCV 605 | `reports/contest_release/baseline_bscv_605/` |

## 运行时路径

- 知识库与模型均在本目录：`chroma_db/`、`models/bge-m3/`
- `RAG_ROOT` 由 `src/rag/config.py` 相对定位到本参赛仓
- Chroma 元数据若仍含研发仓绝对路径，`web/pipeline.py` 会映射到本仓 `data/raw/`

## Git

源研发仓无可用 git 历史；commit 记为 **UNKNOWN**（见 `RELEASE.md`）。
