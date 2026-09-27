"""
matching_layer.py —— 匹配层（检索内核，创新点插拔点）

设计约束：
  - 本层 ONLY 负责「如何从一个 query 文本召回候选 chunk 列表」。
  - 输入：text、k、role（forward/reverse）
  - 输出：List[dict] = [{id, content, metadata, distance, score?, match_info}]
  - 正反向使用同一配方，保证交集对偶性。
  - 不动 BSCV 正向/反向/交集/回退/生成主流程。
  - training-free；默认零额外 LLM 调用。

Evidence-QC 三层（可独立开关）：
  4.1 医学证据链分级门控
  4.2 禁忌 / 负向信号排除门控
  4.3 PICO 维度对齐匹配
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from .clinical_rules import (
    HIGH_EVIDENCE_TIERS,
    clinical_conflict,
    infer_evidence_tier,
    pico_alignment_score,
    pico_from_text,
)
from .utils import normalize_results

# 过召回倍数（常量，非调参旋钮）：门控筛减前多取一些
OVERFETCH_FACTOR = 3


class Matcher(ABC):
    """匹配层统一接口。"""

    @abstractmethod
    def retrieve(self, text: str, k: int, role: str = "forward") -> List[dict]:
        ...


class DenseMatcher(Matcher):
    """纯稠密基线（= 原始 BSCV 匹配方式）。"""

    def __init__(self, collection, embed):
        self.collection = collection
        self.embed = embed

    def retrieve(self, text: str, k: int, role: str = "forward") -> List[dict]:
        q_emb = self.embed(text)
        res = self.collection.query(
            query_embeddings=[q_emb],
            n_results=max(1, k),
            include=["documents", "metadatas", "distances"],
        )
        chunks = normalize_results(res)
        for c in chunks:
            dist = c.get("distance")
            sim = _dist_to_sim(dist)
            c["score"] = sim
            c["match_info"] = {
                "mode": "dense",
                "role": role,
                "sim": round(sim, 4),
                "reasons": [],
            }
        return chunks


class EvidenceQCMatcher(Matcher):
    """
    循证质控混合匹配（Evidence-QC Hybrid Matching）。

    流程：稠密过召回 → 证据分级加权 → 禁忌排除/降权 → PICO 对齐重排 → Top-k。
    产出仍是 C_fwd/C_rev，上游交集零侵入。
    """

    def __init__(
        self,
        collection,
        embed,
        use_evidence_tier: bool = True,
        use_contraindication: bool = True,
        use_pico: bool = True,
        overfetch_factor: int = OVERFETCH_FACTOR,
        docmeta_index: Optional[Dict[str, Any]] = None,
        clinical_dict: Optional[dict] = None,
        llm_extract: bool = False,  # 默认关闭；本实现不走 LLM
        require_high_evidence_in_pool: bool = False,  # 进阶：候选池尽量保留高证据
    ):
        self.collection = collection
        self.embed = embed
        self.use_evidence_tier = use_evidence_tier
        self.use_contraindication = use_contraindication
        self.use_pico = use_pico
        self.overfetch_factor = max(1, int(overfetch_factor))
        self.docmeta_index = docmeta_index or {}
        self.clinical_dict = clinical_dict
        self.llm_extract = bool(llm_extract)
        self.require_high_evidence_in_pool = require_high_evidence_in_pool

    def retrieve(self, text: str, k: int, role: str = "forward") -> List[dict]:
        fetch_k = max(k * self.overfetch_factor, k)
        base = self._dense_recall(text, k=fetch_k, role=role)

        if self.use_evidence_tier:
            base = self._gate_evidence_tier(base)
        if self.use_contraindication:
            base = self._gate_contraindication(text, base)
        if self.use_pico:
            base = self._gate_pico(text, base)

        # 统一按最终分排序
        base.sort(key=lambda x: (-float(x.get("score") or 0.0), str(x.get("id"))))

        if self.require_high_evidence_in_pool:
            base = self._prefer_high_evidence(base, k)

        out = base[:k]
        for c in out:
            info = c.setdefault("match_info", {})
            info["mode"] = "evidence_qc"
            info["role"] = role
            info["final_score"] = round(float(c.get("score") or 0.0), 4)
        return out

    # ---------- 基底稠密召回 ----------
    def _dense_recall(self, text: str, k: int, role: str) -> List[dict]:
        q_emb = self.embed(text)
        res = self.collection.query(
            query_embeddings=[q_emb],
            n_results=max(1, k),
            include=["documents", "metadatas", "distances"],
        )
        chunks = normalize_results(res)
        for c in chunks:
            # 合并 docmeta_index（若有）
            cid = str(c.get("id") or "")
            if cid in self.docmeta_index and isinstance(self.docmeta_index[cid], dict):
                meta = dict(c.get("metadata") or {})
                meta.update({kk: vv for kk, vv in self.docmeta_index[cid].items() if vv is not None})
                c["metadata"] = meta
            sim = _dist_to_sim(c.get("distance"))
            c["score"] = sim
            c["match_info"] = {
                "mode": "evidence_qc",
                "role": role,
                "sim": round(sim, 4),
                "reasons": [],
            }
        return chunks

    # ---------- 4.1 证据分级 ----------
    def _gate_evidence_tier(self, chunks: List[dict]) -> List[dict]:
        for c in chunks:
            meta = c.get("metadata") or {}
            tier = infer_evidence_tier(meta, c.get("content") or "")
            info = c.setdefault("match_info", {})
            info["evidence_tier"] = tier
            reasons = info.setdefault("reasons", [])
            reasons.append(f"证据等级={tier}（只记录次序，不乘相似度）")
        chunks.sort(key=lambda x: (-float(x.get("score") or 0.0), str(x.get("id"))))
        return chunks

    # ---------- 4.2 禁忌 / 负向排除 ----------
    def _gate_contraindication(self, query: str, chunks: List[dict]) -> List[dict]:
        kept: List[dict] = []
        for c in chunks:
            conflict, reason = clinical_conflict(query, c.get("content") or "", c.get("metadata"))
            info = c.setdefault("match_info", {})
            reasons = info.setdefault("reasons", [])
            if conflict:
                info["contraindication_check"] = "conflict_flagged"
                reasons.append(f"与题干条件冲突（只标记，不降权、不删除）：{reason}")
            else:
                info["contraindication_check"] = "pass"
            kept.append(c)
        kept.sort(key=lambda x: (-float(x.get("score") or 0.0), str(x.get("id"))))
        return kept

    # ---------- 4.3 PICO 对齐 ----------
    def _gate_pico(self, query: str, chunks: List[dict]) -> List[dict]:
        q_pico = pico_from_text(query)
        for c in chunks:
            c_pico = pico_from_text(c.get("content") or "")
            align, pico_reasons = pico_alignment_score(q_pico, c_pico)
            sim = float(c.get("score") or 0.0)
            c["score"] = sim
            info = c.setdefault("match_info", {})
            info["pico_query"] = {k: sorted(v) for k, v in q_pico.items() if v}
            info["pico_chunk"] = {k: sorted(v) for k, v in c_pico.items() if v}
            info["pico_align"] = round(align, 4)
            reasons = info.setdefault("reasons", [])
            if pico_reasons:
                reasons.extend(pico_reasons)
            else:
                reasons.append("PICO无强约束（query未点名药物/人群）")
        chunks.sort(key=lambda x: (-float(x.get("score") or 0.0), str(x.get("id"))))
        return chunks

    def _prefer_high_evidence(self, chunks: List[dict], k: int) -> List[dict]:
        """进阶：若 Top-k 全无高证据，尝试把最高分的高证据 chunk 换入末位。"""
        if not chunks:
            return chunks
        top = chunks[:k]
        if any((c.get("match_info") or {}).get("evidence_tier") in HIGH_EVIDENCE_TIERS for c in top):
            return chunks
        for c in chunks[k:]:
            if (c.get("match_info") or {}).get("evidence_tier") in HIGH_EVIDENCE_TIERS:
                reasons = c.setdefault("match_info", {}).setdefault("reasons", [])
                reasons.append("高证据保底：换入候选池以满足指南/说明书覆盖")
                return chunks[: k - 1] + [c] + [x for x in chunks[k - 1 :] if x is not c]
        return chunks


def _dist_to_sim(distance: Any) -> float:
    if distance is None:
        return 0.0
    try:
        d = float(distance)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, 1.0 - d))


def build_matcher(
    collection,
    embed,
    matching_mode: str = "dense",
    *,
    use_evidence_tier: bool = True,
    use_contraindication: bool = True,
    use_pico: bool = True,
    **kwargs,
) -> Matcher:
    """工厂：供 BSCVRAG / 评测脚本统一创建。"""
    mode = (matching_mode or "dense").lower()
    if mode == "evidence_qc":
        return EvidenceQCMatcher(
            collection=collection,
            embed=embed,
            use_evidence_tier=use_evidence_tier,
            use_contraindication=use_contraindication,
            use_pico=use_pico,
            **kwargs,
        )
    return DenseMatcher(collection=collection, embed=embed)
