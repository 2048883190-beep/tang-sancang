# -*- coding: utf-8 -*-
"""
parallel_fusion.py —— 三路并行打分，再用 λ 乘积汇总。

三路各自算完，再串成一个分数：

  s(d) = m_a · m_u · m_k · a^{λ1} · u^{λ2} · k^{λ3}

约束：λ1 ≥ λ2 ≥ λ3 > 0（权威 ≥ 适用 ≥ 一致）。
m∈{0,1} 为软否决；a/u/k∈(0,1]。
正反依据留存是窗口约束，不写入乘积。
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from .clinical_rules import (
    EVIDENCE_TIER_RANK,
    clinical_conflict,
    extract_drugs,
    infer_evidence_tier,
)
from .domain_gate import chunk_domain_conflict, classify_domain

# 乘积用的权威基数（名次 → 正数刻度，再除以 9 归一）。
AUTHORITY_CARDINAL = {
    "guideline": 9.0,
    "drug_label": 7.0,
    "textbook": 5.0,
    "rct": 3.0,
    "opinion": 1.0,
}
EPS = 1e-6
K_HIT = 1.0
K_MISS = 0.35


@dataclass(frozen=True)
class FusionLambdas:
    lam1: float = 1.5
    lam2: float = 0.5
    lam3: float = 0.2

    def validated(self) -> "FusionLambdas":
        if not (self.lam1 >= self.lam2 >= self.lam3 > 0):
            raise ValueError(f"需要 λ1≥λ2≥λ3>0，收到 {self}")
        return self

    def as_dict(self) -> Dict[str, float]:
        return {"lam1": self.lam1, "lam2": self.lam2, "lam3": self.lam3}


LAMBDA1_GRID = (1.0, 1.2, 1.5)
LAMBDA2_GRID = (0.5, 0.7, 1.0)
LAMBDA3_GRID = (0.2, 0.4, 0.6)


def iter_lambda_grid() -> List[FusionLambdas]:
    out: List[FusionLambdas] = []
    for a in LAMBDA1_GRID:
        for b in LAMBDA2_GRID:
            for c in LAMBDA3_GRID:
                if a >= b >= c > 0:
                    out.append(FusionLambdas(a, b, c))
    return out

POS_MARKERS = ("首选", "一线", "推荐", "适用", "可选用", "常规")
NEG_MARKERS = ("禁忌", "禁用", "不推荐", "避免", "停用", "不宜", "禁止")


def chunk_polarity(text: str) -> Tuple[bool, bool]:
    """(has_pos, has_neg)。同一段可同时为真（指南合取句）。"""
    t = text or ""
    return any(k in t for k in POS_MARKERS), any(k in t for k in NEG_MARKERS)


def _stance_blob(text: str) -> Set[str]:
    """粗粒度立场词袋，用于池级诊断（不驱动删除/提示）。"""
    t = text or ""
    tags: Set[str] = set()
    pos, neg = chunk_polarity(t)
    if neg:
        tags.add("NEG")
    if pos:
        tags.add("POS")
    for d in extract_drugs(t):
        tags.add(f"DRUG:{d}")
    return tags


def count_stance_conflicts(chunks: Sequence[dict]) -> int:
    """池内同药 POS/NEG 成对次数。只记账：指南语料里这是常态，不能当否决。"""
    n = len(chunks)
    stances = [_stance_blob(c.get("content") or "") for c in chunks]
    hits = 0
    for i in range(n):
        for j in range(i + 1, n):
            di = {x for x in stances[i] if x.startswith("DRUG:")}
            dj = {x for x in stances[j] if x.startswith("DRUG:")}
            if not (di & dj):
                continue
            i_neg, i_pos = "NEG" in stances[i], "POS" in stances[i]
            j_neg, j_pos = "NEG" in stances[j], "POS" in stances[j]
            if (i_neg and j_pos) or (i_pos and j_neg):
                hits += 1
    return hits


def query_conditioned_conflict(query: str, chunks: Sequence[dict]) -> Dict[str, Any]:
    """
    冲突必须条件于本题点名的药，不能对检索池里任意药做 κ。
    每段算 1 条。两侧都有时
      κ = 1 - |推荐条数 - 限制条数| / (推荐条数 + 限制条数)
    """
    drugs_q = extract_drugs(query)
    per_drug: Dict[str, Any] = {}
    for drug in sorted(drugs_q):
        u_pos = u_neg = 0.0
        n_pos = n_neg = 0
        for c in chunks:
            text = c.get("content") or ""
            if drug not in extract_drugs(text):
                continue
            pos, neg = chunk_polarity(text)
            # 每段计 1。没有标定过的权威分，不能拿来当质量。
            mass = 1.0
            if pos:
                u_pos += mass
                n_pos += 1
            if neg:
                u_neg += mass
                n_neg += 1
        both = u_pos > 0 and u_neg > 0
        denom = u_pos + u_neg
        kappa = (1.0 - abs(u_pos - u_neg) / denom) if both and denom > 0 else 0.0
        per_drug[drug] = {
            "U_pos": round(u_pos, 4),
            "U_neg": round(u_neg, 4),
            "n_pos": n_pos,
            "n_neg": n_neg,
            "both_sides": both,
            "kappa": round(kappa, 4),
        }
    kappa_q = max((v["kappa"] for v in per_drug.values()), default=0.0)
    return {
        "query_drugs": sorted(drugs_q),
        "per_drug": per_drug,
        "kappa_q": round(float(kappa_q), 4),
        "query_both_sides": any(v["both_sides"] for v in per_drug.values()),
    }


def evidence_delta(query: str, chunks: Sequence[dict]) -> float:
    """
    Δ(q)∈[0,1]：检索是否在讲本题。
    有点名药物：命中题干药物的块越多越接近 1；
    无点名药物：用关键词重叠。
    只解释「该不该改写先验」，不删上下文。
    """
    if not chunks:
        return 0.0
    drugs_q = extract_drugs(query)
    if drugs_q:
        hits = sum(
            1
            for c in chunks
            if extract_drugs(c.get("content") or "") & drugs_q
        )
        return round(min(1.0, hits / max(1.0, min(3.0, float(len(chunks))))), 4)
    q_toks = set(re.findall(r"[\u4e00-\u9fff]{2,}|[A-Za-z0-9]{2,}", query or ""))
    if not q_toks:
        return 0.3
    best = 0.0
    for c in list(chunks)[:8]:
        c_toks = set(
            re.findall(
                r"[\u4e00-\u9fff]{2,}|[A-Za-z0-9]{2,}",
                (c.get("content") or "")[:800],
            )
        )
        best = max(best, len(q_toks & c_toks) / max(1, len(q_toks)))
    return round(max(0.05, min(1.0, best)), 4)


def query_drug_stances(query: str, chunk: dict) -> Set[str]:
    """块对题干药物的立场：POS / NEG / 两者 / 空（未点名本题药）。"""
    drugs_q = extract_drugs(query)
    if not drugs_q:
        return set()
    text = chunk.get("content") or ""
    if not (extract_drugs(text) & drugs_q):
        return set()
    pos, neg = chunk_polarity(text)
    out: Set[str] = set()
    if pos:
        out.add("POS")
    if neg:
        out.add("NEG")
    return out


def assemble_with_stance_quota(
    query: str,
    chunks: Sequence[dict],
    k: int = 10,
) -> Tuple[List[dict], Dict[str, Any]]:
    """
    冲突保席：m_a=1 只保证不删；Top-k 仍可能把禁忌面挤掉。
    若全库（传入列表）对本题药物两侧都有，则窗口内至少各留一条。
    替换从窗口末位、无立场或重复立场的块开始，再按分数重排。
    """
    ranked = [c for c in chunks if float(c.get("score") or 0) > 0]
    if not ranked:
        ranked = list(chunks)
    diag: Dict[str, Any] = {
        "quota_applied": False,
        "swapped": 0,
        "need": [],
        "have_before": [],
        "have_after": [],
    }
    if len(ranked) <= k:
        have = set()
        for c in ranked:
            have |= query_drug_stances(query, c)
        diag["have_after"] = sorted(have)
        return list(ranked), diag

    window = list(ranked[:k])
    info = query_conditioned_conflict(query, ranked)
    have: Set[str] = set()
    for c in window:
        have |= query_drug_stances(query, c)
    diag["have_before"] = sorted(have)
    if not info["query_both_sides"]:
        diag["have_after"] = sorted(have)
        return window, diag

    need = {"POS", "NEG"} - have
    diag["need"] = sorted(need)
    if not need:
        diag["have_after"] = sorted(have)
        return window, diag

    diag["quota_applied"] = True
    outside = list(ranked[k:])
    used = {str(c.get("id")) for c in window}

    def n_tag(items: Sequence[dict], tag: str) -> int:
        return sum(1 for c in items if tag in query_drug_stances(query, c))

    def pick_replace_idx(missing: str) -> Optional[int]:
        for i in range(len(window) - 1, -1, -1):
            st = query_drug_stances(query, window[i])
            if not st:
                return i
        other = "NEG" if missing == "POS" else "POS"
        for i in range(len(window) - 1, -1, -1):
            st = query_drug_stances(query, window[i])
            if missing in st:
                continue
            if other in st and n_tag(window, other) > 1:
                return i
            if not st:
                return i
        for i in range(len(window) - 1, -1, -1):
            st = query_drug_stances(query, window[i])
            if missing in st:
                continue
            sole_other = False
            for tag in ("POS", "NEG"):
                if tag == missing:
                    continue
                if tag in st and n_tag(window, tag) <= 1:
                    sole_other = True
            if not sole_other:
                return i
        return None

    for missing in ("POS", "NEG"):
        have = set()
        for c in window:
            have |= query_drug_stances(query, c)
        if missing in have:
            continue
        cand = next(
            (
                c
                for c in outside
                if missing in query_drug_stances(query, c)
                and str(c.get("id")) not in used
            ),
            None,
        )
        if cand is None:
            continue
        idx = pick_replace_idx(missing)
        if idx is None:
            continue
        dropped = window[idx]
        window[idx] = cand
        used.discard(str(dropped.get("id")))
        used.add(str(cand.get("id")))
        outside = [c for c in outside if str(c.get("id")) != str(cand.get("id"))]
        diag["swapped"] += 1

    window.sort(key=lambda x: (-float(x.get("score") or 0.0), str(x.get("id"))))
    have = set()
    for c in window:
        have |= query_drug_stances(query, c)
    diag["have_after"] = sorted(have)
    return window, diag

def _measured_sim_overlap(query: str, chunk: dict) -> Tuple[float, float]:
    """相似度优先用交叉编码器分；否则用召回分。词重叠为实测。"""
    q = query or ""
    c = chunk.get("content") or ""
    info = chunk.get("match_info") or {}
    sim = None
    if info.get("rerank") is not None:
        try:
            sim = float(info["rerank"])
        except (TypeError, ValueError):
            sim = None
    if sim is None and info.get("retrieve_score") is not None:
        try:
            sim = float(info["retrieve_score"])
        except (TypeError, ValueError):
            sim = None
    if sim is None:
        sim = float(chunk.get("score") or 0.0)
    if sim <= 0 and chunk.get("distance") is not None:
        try:
            sim = max(0.0, 1.0 - float(chunk["distance"]))
        except (TypeError, ValueError):
            sim = 0.0
    sim = max(0.0, min(1.0, float(sim)))
    q_toks = set(re.findall(r"[\u4e00-\u9fff]{2,}|[A-Za-z0-9]{2,}", q))
    c_toks = set(re.findall(r"[\u4e00-\u9fff]{2,}|[A-Za-z0-9]{2,}", c[:800]))
    overlap = (len(q_toks & c_toks) / len(q_toks)) if q_toks else 0.0
    return sim, max(0.0, min(1.0, overlap))


def applicability_order_key(query: str, chunk: dict) -> Tuple[int, int, float, float]:
    """药物命中 τ、条件是否冲突、相似度、词重叠。"""
    q = query or ""
    c = chunk.get("content") or ""
    sim, overlap = _measured_sim_overlap(q, chunk)
    conflict, _ = clinical_conflict(q, c, chunk.get("metadata"))
    drugs_q = extract_drugs(q)
    drugs_c = extract_drugs(c)
    if drugs_q and (drugs_c & drugs_q):
        topic = 2
    elif (not drugs_q) or (not drugs_c):
        topic = 1
    else:
        topic = 0
    return (topic, 0 if conflict else 1, sim, overlap)


def authority_channel(chunk: dict) -> float:
    """权威路 a∈(0,1]。指南最高。"""
    tier = infer_evidence_tier(chunk.get("metadata"), chunk.get("content") or "")
    raw = float(AUTHORITY_CARDINAL.get(tier, 1.0))
    return max(EPS, min(1.0, raw / 9.0))


def applicability_channel(query: str, chunk: dict) -> Tuple[float, int, float, float, int]:
    """适用路：返回 (u, τ, sim, overlap, not_conflict)。"""
    topic, ok, sim, overlap = applicability_order_key(query, chunk)
    drug_w = {2: 1.0, 1: 0.8, 0: 0.45}[topic]
    conflict_w = 0.55 if ok == 0 else 1.0
    u = sim * (0.5 + 0.5 * overlap) * drug_w * conflict_w
    u = max(EPS, min(1.0, u))
    return u, topic, sim, overlap, ok


def consistency_channel(chunk: dict, confirmed_ids: Set[str]) -> float:
    """一致路 k∈(0,1]。第二次检索命中为 1，否则为 β。"""
    if str(chunk.get("id") or "") in confirmed_ids:
        return K_HIT
    return K_MISS


def applicability_mask(query: str, chunk: dict, q_domain: str) -> int:
    """m_u：邻科域噪声置 0。条件冲突只压 u，不整段清掉。"""
    del query
    dom_c, _ = chunk_domain_conflict(q_domain, chunk.get("content") or "", chunk.get("metadata"))
    if dom_c:
        return 0
    return 1


def fuse_score(
    a: float,
    u: float,
    k: float,
    *,
    m_a: int = 1,
    m_u: int = 1,
    m_k: int = 1,
    lambdas: Optional[FusionLambdas] = None,
) -> float:
    """s = m_a m_u m_k · a^λ1 · u^λ2 · k^λ3。"""
    lam = (lambdas or FusionLambdas()).validated()
    if m_a * m_u * m_k == 0:
        return 0.0
    return (
        float(m_a)
        * float(m_u)
        * float(m_k)
        * (max(EPS, a) ** lam.lam1)
        * (max(EPS, u) ** lam.lam2)
        * (max(EPS, k) ** lam.lam3)
    )


def parallel_appraise_and_fuse(
    query: str,
    chunks: Sequence[dict],
    *,
    fwd_ids: Optional[Set[str]] = None,
    rev_ids: Optional[Set[str]] = None,
    lambdas: Optional[FusionLambdas] = None,
) -> Tuple[List[dict], Dict[str, Any]]:
    """三路并行打分，乘积汇总排序。"""
    del fwd_ids
    lam = (lambdas or FusionLambdas()).validated()
    q_domain = classify_domain(query).domain
    confirmed = set(rev_ids or set())
    q_conflict_pool = query_conditioned_conflict(query, chunks)
    n_stance_conflicts = count_stance_conflicts(chunks)

    enriched: List[dict] = []
    for c in chunks:
        cc = dict(c)
        m_u = applicability_mask(query, c, q_domain)
        m_a = 1
        a = authority_channel(c)
        u, tau, sim, overlap, ok = applicability_channel(query, c)
        k = consistency_channel(c, confirmed)
        m_k = 1 if k >= K_MISS - EPS else 0
        s = fuse_score(a, u, k, m_a=m_a, m_u=m_u, m_k=m_k, lambdas=lam)
        cc["score"] = s
        cc["channels"] = {
            "a": round(a, 6),
            "u": round(u, 6),
            "k": round(k, 6),
            "m_a": m_a,
            "m_u": m_u,
            "m_k": m_k,
            "m": m_a * m_u * m_k,
            "tau": tau,
            "sim": round(sim, 6),
            "overlap": round(overlap, 6),
            "not_conflict": ok,
            "twice": k >= K_HIT - EPS,
            "tier": infer_evidence_tier(c.get("metadata"), c.get("content") or ""),
            "s": round(s, 6),
        }
        info = cc.setdefault("match_info", {})
        info["fusion_score"] = s
        info["channels"] = cc["channels"]
        info["lambdas"] = lam.as_dict()
        info["fusion"] = "three_channel_product"
        reasons = info.setdefault("reasons", [])
        if m_u == 0:
            reasons.append("域噪声 m_u=0")
        reasons.append("三路并行：权威/适用/一致，乘积汇总")
        enriched.append(cc)

    enriched.sort(key=lambda x: (-float(x.get("score") or 0.0), str(x.get("id"))))
    survived = [c for c in enriched if float(c.get("score") or 0) > 0]
    q_conflict = query_conditioned_conflict(query, survived)
    diag: Dict[str, Any] = {
        "method": "three_channel_product",
        "fusion": "three_channel_product",
        "lambdas": lam.as_dict(),
        "stance_conflicts_pool": n_stance_conflicts,
        "n_stance_conflicts": n_stance_conflicts,
        "second_search_hits": len(confirmed),
        "n_in": len(chunks),
        "n_survived": len(survived),
        "q_domain": q_domain,
        "query_drugs": q_conflict.get("query_drugs") or [],
        "query_both_sides": bool(q_conflict.get("query_both_sides")),
        "kappa_q": float(q_conflict.get("kappa_q") or 0.0),
        "per_drug": q_conflict.get("per_drug") or {},
        "delta_q": evidence_delta(query, survived),
        "alive": len(survived),
        "n": len(enriched),
    }
    return enriched, diag
