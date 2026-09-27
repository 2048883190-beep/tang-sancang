# -*- coding: utf-8 -*-
"""适用性：已训练的交叉编码器对（问句, 段落）打分，并用这个分数决定先后。

权重来自 BAAI/bge-reranker-v2-m3（本地 models/bge-reranker-v2-m3）。
通过 sentence_transformers.CrossEncoder 加载，避免 FlagEmbedding 与
当前 transformers 在 tokenizer.prepare_for_model 上的不兼容。
"""

from __future__ import annotations

from typing import List, Optional, Sequence

from .config import RERANKER_MODEL_ID, RAG_ROOT
from .parallel_fusion import authority_channel


def reranker_path() -> str:
    local = RAG_ROOT / "models" / "bge-reranker-v2-m3"
    weight = local / "model.safetensors"
    if (local / "config.json").exists() and weight.exists() and weight.stat().st_size > 2_000_000_000:
        return str(local)
    return RERANKER_MODEL_ID


class CrossEncoderReranker:
    def __init__(self, model_name: Optional[str] = None):
        from sentence_transformers import CrossEncoder

        name = model_name or reranker_path()
        self.model_name = name
        # fp16 on CUDA when available; CrossEncoder handles device
        self.model = CrossEncoder(name, trust_remote_code=True)

    def rerank(self, query: str, chunks: Sequence[dict]) -> List[dict]:
        rows = [c for c in chunks if (c.get("content") or "").strip()]
        if not rows:
            return []
        pairs = [(query, (c.get("content") or "")[:1500]) for c in rows]
        raw = self.model.predict(pairs, batch_size=4, convert_to_numpy=True)
        if hasattr(raw, "tolist"):
            scores = [float(x) for x in raw.tolist()]
        elif isinstance(raw, (int, float)):
            scores = [float(raw)]
        else:
            scores = [float(x) for x in raw]
        scored: List[dict] = []
        for chunk, score in zip(rows, scores):
            value = float(score)
            info = chunk.setdefault("match_info", {})
            info["rerank"] = round(value, 6)
            both = bool(info.get("in_both"))
            chunk["channels"] = {
                "a": authority_channel(chunk),
                "u": round(value, 6),
                "k": 1 if both else 0,
                "twice": False,
                "in_both": both,
            }
            chunk["score"] = value
            scored.append(chunk)
        scored.sort(key=lambda c: (-float(c.get("score") or 0.0), str(c.get("id"))))
        return scored
