# -*- coding: utf-8 -*-
"""BM25 与稠密召回按名次融合。

RRF 的 k=60 来自 Cormack 等人的名次融合，不在验证集上搜索。
BM25 的 k1=1.5、b=0.75 是 Okapi 的公开默认值，同样不搜索。
"""

from __future__ import annotations

import math
import re
from typing import Dict, List, Optional, Sequence

RRF_K = 60
BM25_K1 = 1.5
BM25_B = 0.75
FETCH_EACH = 40
RERANK_POOL = 20

_HAN = re.compile(r"[\u4e00-\u9fff]")
_LATIN = re.compile(r"[A-Za-z0-9]+")


def tokenize(text: str) -> List[str]:
    """中文重叠二字与英文数字串。不依赖分词器。"""
    raw = text or ""
    toks = [t.lower() for t in _LATIN.findall(raw)]
    han = "".join(_HAN.findall(raw))
    toks.extend(han[i : i + 2] for i in range(max(0, len(han) - 1)))
    return toks


class BM25Index:
    def __init__(self, docs: Sequence[dict]):
        self.docs = list(docs)
        self._tf: List[Dict[str, int]] = []
        self._dl: List[int] = []
        df: Dict[str, int] = {}
        for doc in self.docs:
            counts: Dict[str, int] = {}
            for tok in tokenize(doc.get("content") or ""):
                counts[tok] = counts.get(tok, 0) + 1
            self._tf.append(counts)
            length = sum(counts.values())
            self._dl.append(length)
            for tok in counts:
                df[tok] = df.get(tok, 0) + 1
        self._df = df
        self._n = len(self.docs)
        self._avgdl = (sum(self._dl) / self._n) if self._n else 0.0

    def search(self, query: str, k: int) -> List[dict]:
        q_counts: Dict[str, int] = {}
        for tok in tokenize(query):
            q_counts[tok] = q_counts.get(tok, 0) + 1
        if not q_counts or not self._n:
            return []
        scored: List[tuple] = []
        for i, tf in enumerate(self._tf):
            s = 0.0
            dl = self._dl[i]
            for tok, qtf in q_counts.items():
                if tok not in tf:
                    continue
                n_q = self._df.get(tok, 0)
                idf = math.log(1.0 + (self._n - n_q + 0.5) / (n_q + 0.5))
                freq = tf[tok]
                denom = freq + BM25_K1 * (1.0 - BM25_B + BM25_B * dl / (self._avgdl or 1.0))
                s += idf * (freq * (BM25_K1 + 1.0) * qtf) / (denom or 1.0)
            if s > 0:
                scored.append((s, i))
        scored.sort(key=lambda x: (-x[0], str(self.docs[x[1]].get("id"))))
        out: List[dict] = []
        for s, i in scored[: max(1, k)]:
            src = self.docs[i]
            out.append(
                {
                    "id": src.get("id"),
                    "content": src.get("content") or "",
                    "metadata": dict(src.get("metadata") or {}),
                    "distance": None,
                    "score": float(s),
                    "match_info": {"mode": "bm25", "bm25": round(float(s), 4)},
                }
            )
        return out


def rrf_fuse(
    dense: Sequence[dict],
    lexical: Sequence[dict],
    *,
    rrf_k: int = RRF_K,
    limit: int = RERANK_POOL,
) -> List[dict]:
    """两路名次融合。同分时 id 小的在前，避免并列漂移。"""
    pooled: Dict[str, dict] = {}
    acc: Dict[str, float] = {}
    dense_rank: Dict[str, int] = {}
    bm25_rank: Dict[str, int] = {}

    def _take(chunks: Sequence[dict], into: Dict[str, int]) -> None:
        for rank, chunk in enumerate(chunks, 1):
            cid = str(chunk.get("id") or "")
            if not cid:
                continue
            into[cid] = rank
            acc[cid] = acc.get(cid, 0.0) + 1.0 / (rrf_k + rank)
            if cid not in pooled:
                pooled[cid] = {
                    "id": chunk.get("id"),
                    "content": chunk.get("content") or "",
                    "metadata": dict(chunk.get("metadata") or {}),
                    "distance": chunk.get("distance"),
                    "score": float(chunk.get("score") or 0.0),
                    "match_info": dict(chunk.get("match_info") or {}),
                }
            elif not pooled[cid].get("content") and chunk.get("content"):
                pooled[cid]["content"] = chunk.get("content") or ""

    _take(dense, dense_rank)
    _take(lexical, bm25_rank)
    ranked = sorted(acc.items(), key=lambda x: (-x[1], x[0]))
    out: List[dict] = []
    for cid, score in ranked[: max(1, limit)]:
        row = pooled[cid]
        info = row.setdefault("match_info", {})
        info["mode"] = "hybrid_rrf"
        info["rrf"] = round(score, 6)
        info["dense_rank"] = dense_rank.get(cid)
        info["bm25_rank"] = bm25_rank.get(cid)
        info["in_both"] = cid in dense_rank and cid in bm25_rank
        row["score"] = score
        out.append(row)
    return out


class HybridRetriever:
    """稠密 Top-N 与 BM25 Top-N，再按 RRF 留下待重排的一小池。"""

    def __init__(self, collection, embed, fetch_each: int = FETCH_EACH):
        self.collection = collection
        self.embed = embed
        self.fetch_each = fetch_each
        self._bm25: Optional[BM25Index] = None

    def _index(self) -> BM25Index:
        if self._bm25 is None:
            data = self.collection.get(include=["documents", "metadatas"])
            ids = data.get("ids") or []
            docs = data.get("documents") or []
            metas = data.get("metadatas") or []
            rows = []
            for i, cid in enumerate(ids):
                rows.append(
                    {
                        "id": cid,
                        "content": docs[i] if i < len(docs) else "",
                        "metadata": metas[i] if i < len(metas) else {},
                    }
                )
            self._bm25 = BM25Index(rows)
        return self._bm25

    def retrieve(self, text: str, *, pool: int = RERANK_POOL) -> List[dict]:
        from .matching_layer import DenseMatcher

        dense = DenseMatcher(self.collection, self.embed).retrieve(
            text, k=self.fetch_each, role="forward"
        )
        lexical = self._index().search(text, k=self.fetch_each)
        return rrf_fuse(dense, lexical, limit=pool)
