# -*- coding: utf-8 -*-
"""
providers.py —— 上层模型 / 下层知识库的可插拔适配层（low-level 可迁移）

设计目标（与项目方向对齐）：
  - 不绑定某一家 LLM：换 DeepSeek / OpenAI / 本地模型只需实现 LLMProvider
  - 不绑定某一向量库实现：换 Chroma / 其他库只需实现 KnowledgeStore
  - 中间流水线（域门控 → 证据分层 → 冲突质控 → 生成）保持模型无关

这不是「特化死结构」，而是可更新、可嵌套的骨架。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Protocol, Sequence


class LLMProvider(ABC):
    """上层：任意大语言模型的统一生成接口。"""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        model: Optional[str] = None,
    ) -> str:
        ...


class OpenAICompatibleProvider(LLMProvider):
    """OpenAI SDK 兼容端点（DeepSeek / OpenAI / 多数本地网关）。"""

    def __init__(self, client, default_model: str = "deepseek-chat"):
        self.client = client
        self.default_model = default_model

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        model: Optional[str] = None,
    ) -> str:
        resp = self.client.chat.completions.create(
            model=model or self.default_model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return (resp.choices[0].message.content or "").strip()


class Embedder(Protocol):
    def __call__(self, text: str) -> List[float]:
        ...


class KnowledgeStore(ABC):
    """下层：可更新知识库的统一检索接口（与具体向量引擎解耦）。"""

    @abstractmethod
    def query(
        self,
        embedding: Sequence[float],
        k: int,
        *,
        include_documents: bool = True,
        include_metadatas: bool = True,
        include_distances: bool = True,
    ) -> Dict[str, Any]:
        """返回与 chroma query 类似的结构，便于现有 normalize_results。"""
        ...

    @abstractmethod
    def count(self) -> int:
        ...


class ChromaKnowledgeStore(KnowledgeStore):
    """Chroma collection 适配器；换库时另写子类即可。"""

    def __init__(self, collection):
        self.collection = collection

    def query(
        self,
        embedding: Sequence[float],
        k: int,
        *,
        include_documents: bool = True,
        include_metadatas: bool = True,
        include_distances: bool = True,
    ) -> Dict[str, Any]:
        include = []
        if include_documents:
            include.append("documents")
        if include_metadatas:
            include.append("metadatas")
        if include_distances:
            include.append("distances")
        return self.collection.query(
            query_embeddings=[list(embedding)],
            n_results=max(1, k),
            include=include or None,
        )

    def count(self) -> int:
        return int(self.collection.count())


# ---------- 流水线阶段图（决策树式 / 可扩展，非绑死 Transformer）----------
# 节点名供 stats / 答辩叙事使用；真正执行在 HierarchicalRAG.query

PIPELINE_STAGES = (
    "domain_gate",          # 域判定：是否检索
    "over_retrieve",        # 过召回
    "evidence_split",       # 高/低证据拆分（类似全局/局部）
    "conflict_filter",      # 冲突质控
    "optional_bscv",        # 可选一致性正则
    "stage1_framework",     # 高证据全局引导
    "stage2_refine",        # 低证据局部细化
    "confidence_aggregate", # 可信度聚合
)
