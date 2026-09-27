# -*- coding: utf-8 -*-
"""
hierarchical_rag.py —— BM25+稠密名次融合召回，交叉编码器写适用分，
三路并行打分后用 λ 乘积汇总，一次生成。

选择题：整题检索。临床问句：题干不属于本库时不检索。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .clinical_rules import (
    EVIDENCE_TIER_RANK,
    HIGH_EVIDENCE_TIERS,
    extract_drugs,
    infer_evidence_tier,
)
from .config import (
    DEFAULT_HIER_MODE,
    DEFAULT_MATCHING_MODE,
    DEFAULT_USE_PARALLEL_FUSION,
    HIER_MAX_HIGH,
    HIER_MAX_LOW,
    HIER_TOP_K,
    load_locked_lambdas,
)
from .domain_gate import DomainDecision, chunk_domain_conflict, classify_domain, question_stem
from .hybrid_retrieve import RERANK_POOL, HybridRetriever
from .matching_layer import DenseMatcher, Matcher, build_matcher
from .parallel_fusion import (
    FusionLambdas,
    assemble_with_stance_quota,
    parallel_appraise_and_fuse,
    query_drug_stances,
)
from .providers import LLMProvider, OpenAICompatibleProvider, PIPELINE_STAGES
from .rerank import CrossEncoderReranker

MCQ_TAIL = (
    "\n\n若题目含选项（A/B/C/D/E），请先简要推理，"
    "并在文末单独一行写：最终选项：X（多选则连写字母，如 BCD）。"
    "\n本回答仅供参考，不替代专业诊疗。"
)

CLINICAL_TAIL = (
    "\n\n请用临床表述作答：结论要点、依据条件、注意事项。"
    "禁止编造检查数值；资料不足请明确写出。"
    "不要输出选择题选项字母（A/B/C/D/E）或「最终选项」行。"
    "\n本回答仅供参考，不替代专业诊疗。"
)

TIER_CN = {
    "guideline": "指南/共识",
    "drug_label": "药品说明书",
    "textbook": "教材",
    "rct": "研究文献",
    "opinion": "一般论述",
}

CONF_CN = {
    "HIGH": "把握较高",
    "MEDIUM": "把握中等",
    "LOW": "把握偏低",
    "PARAMETRIC": "未充分取证，参考模型常识",
}


@dataclass
class HierConfig:
    matching_mode: str = DEFAULT_MATCHING_MODE
    hier_mode: str = DEFAULT_HIER_MODE
    top_k: int = HIER_TOP_K
    max_high: int = HIER_MAX_HIGH
    max_low: int = HIER_MAX_LOW
    overfetch_factor: int = 3
    draft_max_tokens: int = 220
    final_max_tokens: int = 1024
    temperature: float = 0.0
    llm_model: str = "deepseek-chat"
    collection_name: str = "t2dm_medical_kb"
    use_parallel_fusion: bool = DEFAULT_USE_PARALLEL_FUSION
    answer_mode: str = "mcq"
    fusion_lam1: Optional[float] = None
    fusion_lam2: Optional[float] = None
    fusion_lam3: Optional[float] = None
    load_locked_lambdas: bool = True


class HierarchicalRAG:
    def __init__(
        self,
        chroma_client,
        llm_client,
        embedding_fn,
        config: Optional[HierConfig] = None,
        matcher: Optional[Matcher] = None,
        llm_provider: Optional[LLMProvider] = None,
    ):
        self.chroma = chroma_client
        self.llm = llm_client
        self.embed = embedding_fn
        self.config = config or HierConfig()
        self._collection = None
        # 可插拔 LLM：默认把 OpenAI 兼容 client 包一层；可传入任意 LLMProvider
        self.llm_provider: LLMProvider = llm_provider or OpenAICompatibleProvider(
            llm_client, default_model=self.config.llm_model
        )
        self.matcher = matcher or build_matcher(
            collection=self.collection,
            embed=self.embed,
            matching_mode=self.config.matching_mode,
        )
        self._hybrid: HybridRetriever | None = None
        self._ce: CrossEncoderReranker | None = None

    @property
    def collection(self):
        if self._collection is None:
            self._collection = self.chroma.get_collection(self.config.collection_name)
        return self._collection

    def _chat(self, prompt: str, max_tokens: int) -> str:
        return self.llm_provider.generate(
            prompt,
            max_tokens=max_tokens,
            temperature=self.config.temperature,
            model=self.config.llm_model,
        )

    def _answer_tail(self) -> str:
        mode = (self.config.answer_mode or "mcq").lower()
        return CLINICAL_TAIL if mode == "clinical" else MCQ_TAIL

    def _format_chunks(self, chunks: Sequence[dict], limit: Optional[int] = None) -> str:
        use = list(chunks) if limit is None else list(chunks)[:limit]
        lines: List[str] = []
        for i, c in enumerate(use, 1):
            meta = c.get("metadata") or {}
            tier = infer_evidence_tier(meta, c.get("content") or "")
            tier_cn = TIER_CN.get(tier, tier)
            title = str(meta.get("title") or meta.get("source") or "").strip()
            head = f"[资料{i}｜{tier_cn}" + (f"｜{title}" if title else "") + "]"
            lines.append(f"{head}\n{c.get('content', '')}")
        return "\n\n---\n\n".join(lines)

    def _annotate_tiers(self, chunks: List[dict]) -> List[dict]:
        for c in chunks:
            tier = infer_evidence_tier(c.get("metadata"), c.get("content") or "")
            c["_tier"] = tier
            c["_tier_rank"] = EVIDENCE_TIER_RANK.get(tier, 1)
            info = c.setdefault("match_info", {})
            info["evidence_tier"] = tier
        return chunks

    def _pack_evidence(self, query: str, chunks: Sequence[dict]) -> List[dict]:
        """前端证据卡：中文标签为主，分数放 technical。"""
        rows: List[dict] = []
        for i, c in enumerate(chunks, 1):
            ch = c.get("channels") or {}
            meta = c.get("metadata") or {}
            tier = c.get("_tier") or infer_evidence_tier(meta, c.get("content") or "")
            stances = sorted(query_drug_stances(query, c))
            stance_cn = []
            if "POS" in stances:
                stance_cn.append("推荐/适用")
            if "NEG" in stances:
                stance_cn.append("限制/禁忌")
            text = " ".join((c.get("content") or "").split())
            file_path = str(meta.get("file_path") or meta.get("filepath") or "").strip()
            rows.append(
                {
                    "rank": i,
                    "id": str(c.get("id") or ""),
                    "doc_id": str(meta.get("doc_id") or ""),
                    "file_path": file_path,
                    "has_source_file": bool(file_path),
                    "tier": tier,
                    "tier_label": TIER_CN.get(tier, tier),
                    "title": str(meta.get("title") or meta.get("source") or "").strip(),
                    "stances": stances,
                    "stance_label": "、".join(stance_cn) if stance_cn else "条件说明",
                    "summary": text[:280] + ("…" if len(text) > 280 else ""),
                    "content": text[:900],
                    "quote": text[:220],
                    "a": round(float(ch.get("a") or 0), 3),
                    "u": round(float(ch.get("u") or 0), 3),
                    "k": round(float(ch.get("k") or 0), 3),
                    "twice": bool(ch.get("twice")),
                    "s": round(float(c.get("score") or 0), 3),
                }
            )
        return rows

    def _drop_domain_noise(
        self, chunks: List[dict], q_domain: str
    ) -> Tuple[List[dict], List[str]]:
        """只丢邻科噪声；不按证据等级截断，避免把冲突一面挤出窗口。"""
        reasons: List[str] = []
        kept: List[dict] = []
        for c in chunks:
            dom_c, dom_why = chunk_domain_conflict(
                q_domain, c.get("content") or "", c.get("metadata")
            )
            if dom_c:
                reasons.append(f"域噪声剔除: {dom_why}")
                continue
            kept.append(c)
        return self._annotate_tiers(kept), reasons

    def _denoise_and_split(
        self, query: str, chunks: List[dict], q_domain: str
    ) -> Tuple[List[dict], List[dict], List[str]]:
        kept, reasons = self._drop_domain_noise(chunks, q_domain)
        kept.sort(
            key=lambda x: (
                -float(x.get("score") or 0.0),
                -int(x.get("_tier_rank") or 0),
            )
        )
        high = [c for c in kept if c.get("_tier") in HIGH_EVIDENCE_TIERS][: self.config.max_high]
        low = [c for c in kept if c.get("_tier") not in HIGH_EVIDENCE_TIERS][: self.config.max_low]
        if len(high) < 2:
            for c in kept:
                if c not in high and len(high) < self.config.max_high:
                    high.append(c)
        return high, low, reasons

    def _confidence(
        self,
        high: List[dict],
        low: List[dict],
        conflicts: int,
        retrieved: bool,
        stance_conflicts: int = 0,
        starved: bool = False,
        kappa_q: float = 0.0,
        delta_q: float = 1.0,
        query_both_sides: bool = False,
    ) -> Dict[str, Any]:
        # Δ(q) 低或未检索：证据不该改写先验，标 PARAMETRIC，不假装会诊完成
        if not retrieved or starved or delta_q < 0.2:
            return {
                "level": "PARAMETRIC",
                "score": 0.45,
                "note": "未检索、材料不足或 Δ(q) 低，依赖模型先验",
                "kappa_q": round(float(kappa_q), 4),
                "delta_q": round(float(delta_q), 4),
            }
        best = 0
        for c in high + low:
            best = max(best, int(c.get("_tier_rank") or 0))
        score = 0.35 + 0.12 * best
        if conflicts:
            score -= 0.08 * min(conflicts, 3)
        # Shafer：本题冲突质量进无知，不按池内成对次数扣分
        if kappa_q > 0:
            score *= 1.0 - 0.4 * min(1.0, float(kappa_q))
        score = max(0.1, min(0.98, score))
        if query_both_sides or kappa_q >= 0.4:
            level = "MEDIUM" if score >= 0.5 else "LOW"
        elif score >= 0.75:
            level = "HIGH"
        elif score >= 0.55:
            level = "MEDIUM"
        else:
            level = "LOW"
        return {
            "level": level,
            "score": round(score, 3),
            "best_tier_rank": best,
            "conflicts": conflicts,
            "stance_conflicts": stance_conflicts,
            "kappa_q": round(float(kappa_q), 4),
            "delta_q": round(float(delta_q), 4),
            "query_both_sides": bool(query_both_sides),
        }

    def _direct_answer(self, query: str) -> str:
        if (self.config.answer_mode or "").lower() == "clinical":
            prompt = (
                "你是 2 型糖尿病专科循证助手。当前问题可能超出本专科知识库覆盖，"
                "请明确说明「未检索专科库」，并仅作有限常识提示；不要假装引用指南。"
                "不要输出选择题选项。\n"
                f"用户问题：{query}"
                f"{CLINICAL_TAIL}"
            )
        else:
            prompt = (
                "你是专业医疗知识助手。当前问题可能超出本专科知识库覆盖，"
                "请依据医学常识作答；依据不足请明确说明。\n"
                f"用户问题：{query}"
                f"{MCQ_TAIL}"
            )
        return self._chat(prompt, self.config.final_max_tokens)

    def _once_generate(self, query: str, chunks: List[dict], keep_conflict: bool = False) -> str:
        ctx = self._format_chunks(chunks)
        clinical = (self.config.answer_mode or "").lower() == "clinical"
        if clinical:
            extra = ""
            if keep_conflict:
                extra = (
                    "\n资料中同一药物可能同时出现推荐与限制：请按问题中的患者条件并列说明，"
                    "不要只保留一面。\n"
                )
            prompt = (
                "你是 2 型糖尿病专科循证助手。请严格依据下列参考资料回答临床问题。\n"
                "要求：优先引用资料中的条件与禁忌；无关段落忽略；"
                "按三部分作答：1）结论要点 2）依据条件 3）注意事项；"
                "使用纯文本，不要 Markdown 加粗符号（不要用 **）；"
                "不要编造未给出的数值；禁止输出 A/B/C/D/E 选项或「最终选项」。\n"
                f"{extra}"
                f"用户问题：{query}\n\n参考资料：\n{ctx}"
                f"{CLINICAL_TAIL}"
            )
        else:
            extra = ""
            if keep_conflict:
                extra = (
                    "\n若资料对同一药物同时写了推荐和限制，请按本题条件选用，"
                    "仍必须在文末单独一行写：最终选项：X。\n"
                )
            prompt = (
                "你是专业医疗知识助手。请基于参考资料回答。\n"
                "要求：优先依据资料；与本题无关的段落请忽略。"
                "选择题必须在文末写出最终选项，不要留空。\n"
                f"{extra}"
                f"用户问题：{query}\n\n参考资料：\n{ctx}"
                f"{MCQ_TAIL}"
            )
        return self._chat(prompt, self.config.final_max_tokens)

    def _hybrid_retriever(self) -> HybridRetriever:
        if self._hybrid is None:
            self._hybrid = HybridRetriever(self.collection, self.embed)
        return self._hybrid

    def _cross_encoder(self) -> CrossEncoderReranker:
        if self._ce is None:
            self._ce = CrossEncoderReranker()
        return self._ce

    def _fusion_lambdas(self) -> FusionLambdas:
        cfg = self.config
        if (
            cfg.fusion_lam1 is not None
            and cfg.fusion_lam2 is not None
            and cfg.fusion_lam3 is not None
        ):
            return FusionLambdas(cfg.fusion_lam1, cfg.fusion_lam2, cfg.fusion_lam3).validated()
        if cfg.load_locked_lambdas:
            return load_locked_lambdas()
        return FusionLambdas().validated()

    def _window_sides(self, query: str, chunks: Sequence[dict]) -> Tuple[List[str], bool]:
        drugs = sorted(extract_drugs(query))
        have: set[str] = set()
        for chunk in chunks:
            have |= query_drug_stances(query, chunk)
        return drugs, ("POS" in have and "NEG" in have)

    def _second_search_ids(self, retrieve_q: str, pool: Sequence[dict]) -> set:
        if not pool:
            return set()
        seed = (pool[0].get("content") or "")[:400].strip()
        if len(seed) < 20:
            return set()
        rev = DenseMatcher(self.collection, self.embed).retrieve(
            seed, k=max(self.config.top_k * 2, 20), role="reverse"
        )
        return {str(c.get("id") or "") for c in rev}

    def query(self, user_query: str, q_type: Optional[str] = None) -> Tuple[str, dict]:
        del q_type
        clinical = (self.config.answer_mode or "").lower() == "clinical"
        lam = self._fusion_lambdas()
        stats: Dict[str, Any] = {
            "pipeline": "hierarchical_rag",
            "pipeline_stages": list(PIPELINE_STAGES),
            "matching_mode": "hybrid_rrf_product",
            "llm_provider": type(self.llm_provider).__name__,
            "reasons": [],
            "fusion": {"method": "three_channel_product", "lambdas": lam.as_dict()},
            "temperature": self.config.temperature,
            "top_k": self.config.top_k,
        }
        stem = question_stem(user_query)
        stats["stem"] = stem
        decision: DomainDecision = classify_domain(stem)
        stats["domain"] = decision.domain
        stats["in_scope"] = decision.in_scope
        stats["domain_reasons"] = decision.reasons

        if clinical and not decision.should_retrieve:
            answer = self._direct_answer(user_query)
            stats["mode"] = "direct_ood_skip_retrieval"
            stats["confidence"] = self._confidence([], [], 0, retrieved=False)
            stats["evidence"] = []
            stats["stage"] = "done"
            return answer, stats

        retrieve_q = stem if clinical else user_query
        pool = self._hybrid_retriever().retrieve(retrieve_q, pool=RERANK_POOL)
        for c in pool:
            info = c.setdefault("match_info", {})
            info["retrieve_score"] = float(c.get("score") or 0.0)
        ranked = self._cross_encoder().rerank(retrieve_q, pool)
        if clinical:
            ranked, denoise_reasons = self._drop_domain_noise(ranked, decision.domain)
            stats["reasons"].extend(denoise_reasons)
        else:
            ranked = self._annotate_tiers(ranked)

        fwd_ids = {str(c.get("id") or "") for c in ranked}
        rev_ids = self._second_search_ids(retrieve_q, ranked)
        if self.config.use_parallel_fusion and ranked:
            fused, diag = parallel_appraise_and_fuse(
                retrieve_q,
                ranked,
                fwd_ids=fwd_ids,
                rev_ids=rev_ids,
                lambdas=lam,
            )
            stats["fusion"] = diag
            assembled, quota = assemble_with_stance_quota(
                retrieve_q, fused, k=self.config.top_k
            )
            stats["stance_quota"] = quota
        else:
            assembled = ranked[: self.config.top_k]
            stats["stance_quota"] = {"quota_applied": False, "swapped": 0}

        if not assembled:
            assembled = ranked[: self.config.top_k]
            stats["reasons"].append("乘积后为空，回退召回池")
            stats["starved"] = True

        drugs, both = self._window_sides(retrieve_q, assembled)
        stats.setdefault("fusion", {})
        if isinstance(stats["fusion"], dict):
            stats["fusion"]["query_drugs"] = drugs
            stats["fusion"]["query_both_sides"] = both
            stats["fusion"]["reranker"] = self._cross_encoder().model_name
            stats["fusion"]["passage_count"] = len(assembled)
        high = [c for c in assembled if c.get("_tier") in HIGH_EVIDENCE_TIERS]
        low = [c for c in assembled if c not in high]
        stats["high_count"] = len(high)
        stats["low_count"] = len(low)
        if assembled:
            stats["mode"] = "hybrid_product"
            answer = self._once_generate(user_query, assembled, keep_conflict=both)
        else:
            stats["mode"] = "direct_empty_retrieval"
            answer = self._direct_answer(user_query)
        stats["confidence"] = self._confidence(
            high,
            low,
            0,
            retrieved=bool(assembled),
            stance_conflicts=0,
            starved=not assembled,
            query_both_sides=both,
        )
        stats["evidence"] = self._pack_evidence(retrieve_q, assembled)
        stats["stage"] = "done"
        return answer, stats
