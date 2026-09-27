# 版本冻结说明

本 Release 是 **Selective RAG v1 / 619/685 = 90.4%** 的独立参赛冻结版本。

项目名：糖三藏  
副标题：三藏辨经，答案有据  
版本号：contest-v1

## 冻结内容

- 答案主路径：Direct + Dense Top-10 + Evidence Assessment + Selective Gate → `KEEP_DIRECT` / `USE_RAG`
- 锁参：`data/eval/locked_selective_rag.json`
- 温度：0
- 测试集：冻结 685 题（不得再调参）

## 明确禁止进入参赛主流程

- Selective RAG v2 / v2-A / v2-B
- option-level verification（`option_verify`）
- TARGETED_RAG / targeted retrieval
- 在测试集上做 test-time tuning
- 把 Naive Dense 618/685 写成「本项目结果」
- 把 BSCV 605/685 写成参赛主结果
- 宣传「准确率达到 91%」

上述后续实验保存在：

```text
archive/research_experiments/
```

## 允许保留（叙事与展示）

- BSCV 三路：权威 / 适用 / 一致（`parallel_fusion.py`）
- 证据卡、PDF 原文定位与高亮（`web/source_open.py`）

三路用于产品辨经叙事与证据展示；**不改变** Selective v1 的答案门控公式。

## 变更纪律

冻结后除非比赛材料需要，否则：

- 不更新知识库 / 指南 / 说明书
- 不换 embedding / LLM
- 不改 top_k / temperature / thresholds
- 不重跑测试集追分
