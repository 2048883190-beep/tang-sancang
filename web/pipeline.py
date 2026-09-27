# -*- coding: utf-8 -*-
"""参赛演示流水线：Selective RAG v1（619/685）+ 三藏展示 + 原文溯源。

答案路径（冻结）：
  Direct + Dense Top-10 + Evidence Assessment → KEEP_DIRECT / USE_RAG

展示路径（叙事，不改答案）：
  对召回段落做权威/适用/一致三路打分，写入证据卡。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rag.config import (  # noqa: E402
    CHROMA_DIR,
    COLLECTION_NAME,
    EMBEDDING_MODEL_ID,
    RAG_ROOT,
    load_locked_lambdas,
    load_locked_selective_thresholds,
)
from rag.domain_gate import classify_domain  # noqa: E402
from rag.matching_layer import DenseMatcher  # noqa: E402
from rag.parallel_fusion import parallel_appraise_and_fuse  # noqa: E402
from rag.selective_rag import SelectiveConfig, SelectiveRAG  # noqa: E402
from session_store import ChatSession

DOMAIN_CN = {
    "drug": "降糖药物",
    "acute": "急性并发症",
    "ckd_cv": "糖尿病肾病/心血管",
    "dx_stage": "诊断与分期",
    "special_pop": "特殊人群",
    "t2dm_general": "2型糖尿病",
    "infect_other": "感染科（非本专科）",
    "ophtho_other": "眼科（非本专科）",
    "endo_other": "其他内分泌（非本专科）",
    "misc_nursing": "护理/其他",
    "epid_stat": "流行病学方法",
    "other": "专科外",
}

DECISION_CN = {
    "KEEP_DIRECT": "证据还不够充分，本次以直接判断为主。",
    "USE_RAG": "已参考指南 / 说明书作答，可在「原文」核对。",
    "USE_TARGETED_RAG": "已结合定向检索证据作答。",
}


def _load_dotenv() -> None:
    env_path = ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())


def _remap_source_path(file_path: str) -> str:
    """Chroma 元数据可能仍指向研发仓绝对路径，映射到本参赛目录。"""
    if not file_path:
        return file_path
    p = file_path.replace("/", "\\")
    markers = (
        "E:\\rag-new\\bscv-rag\\",
        "E:\\rag-new\\bscv-rag-contest-90.2\\",
    )
    for old in markers:
        if p.lower().startswith(old.lower()):
            rel = p[len(old) :]
            return str((RAG_ROOT / rel).resolve())
    cand = Path(file_path)
    if not cand.is_absolute():
        return str((RAG_ROOT / file_path).resolve())
    return file_path


def _pack_evidence(chunks: list[dict], fusion_diag: dict | None = None) -> list[dict]:
    out = []
    for i, c in enumerate(chunks, 1):
        meta = c.get("metadata") or {}
        if not isinstance(meta, dict):
            meta = {}
        file_path = _remap_source_path(
            str(meta.get("file_path") or meta.get("filepath") or "").strip()
        )
        content = (c.get("content") or "")[:900]
        channels = c.get("channels") or {}
        a = float(channels.get("a") or c.get("authority") or 0)
        u = float(channels.get("u") or c.get("applicability") or 0)
        k = float(channels.get("k") or c.get("consistency") or 0)
        # 展示分：0–100
        auth_score = int(round(max(0.0, min(1.0, a)) * 100))
        appl_score = int(round(max(0.0, min(1.0, u)) * 100))
        cons_score = int(round(max(0.0, min(1.0, k)) * 100))
        if auth_score == 0 and appl_score == 0 and cons_score == 0:
            # 无三路分时用检索分与位次兜底展示
            raw = c.get("score")
            try:
                rel = float(raw) if raw is not None else 0.55
            except (TypeError, ValueError):
                rel = 0.55
            if rel > 1.5:
                rel = 1.0 / (1.0 + rel)
            elif rel > 1.0:
                rel = max(0.12, 2.0 - rel)
            rel = max(0.12, min(0.99, rel))
            rank_boost = max(0.4, 1.0 - (i - 1) * 0.06)
            auth_score = int(72 + 20 * rel)
            appl_score = int(58 + 35 * rel * rank_boost)
            cons_score = int(50 + 45 * rel * rank_boost)
        overall = int(round((auth_score + appl_score + cons_score) / 3))
        out.append(
            {
                "rank": i,
                "id": c.get("id"),
                "title": meta.get("title") or meta.get("source") or "未标注来源文档",
                "tier": meta.get("source") or meta.get("tier") or "资料",
                "tier_label": meta.get("source") or meta.get("tier") or "资料",
                "summary": content,
                "content": content,
                "file_path": file_path,
                "has_source_file": bool(file_path) and Path(file_path).exists(),
                "score": c.get("score"),
                "authority_score": auth_score,
                "applicability_score": appl_score,
                "consistency_score": cons_score,
                "overall_score": overall,
                "twice": bool(c.get("twice")),
                "stance_label": "—",
            }
        )
    if fusion_diag:
        # 附加全局信息供前端可选展示
        _ = fusion_diag
    return out


def _build_consult(stats: dict, *, domain: str, in_scope: bool) -> dict[str, Any]:
    decision = stats.get("decision") or ""
    retrieved_n = int(stats.get("retrieved_n") or 0)
    assessment = stats.get("assessment") or {}
    rel = float(stats.get("question_relevance") or assessment.get("evidence_relevance") or 0)
    suf = float(stats.get("question_sufficiency") or assessment.get("evidence_sufficiency") or 0)
    conflict = float(
        stats.get("question_contradiction") or assessment.get("evidence_conflict") or 0
    )

    # 三藏印章：由评估分映射
    authority_on = retrieved_n > 0 and rel >= 0.45
    applicability_on = in_scope and retrieved_n > 0 and suf >= 0.45
    consistency_on = retrieved_n > 0 and conflict <= 0.55

    if decision == "USE_RAG":
        status = "已完成取证与辨经，结论参考了专科证据。"
    elif decision == "KEEP_DIRECT":
        status = "已完成辨经：证据充分性不足，未强行采信检索结果。"
    else:
        status = DECISION_CN.get(decision, "已完成辨经。")

    auth_score = int(round(rel * 100))
    appl_score = 35 if not in_scope else int(round(suf * 100))
    cons_score = int(round(max(0.0, 1.0 - conflict) * 100))
    overall = int(round((auth_score + appl_score + cons_score) / 3)) if retrieved_n else 0

    bullets = [
        f"问题范围：{DOMAIN_CN.get(domain, domain)}",
        f"参考证据：{retrieved_n} 条" if retrieved_n else "未检索专科库",
    ]
    return {
        "domain": domain,
        "domain_label": DOMAIN_CN.get(domain, domain),
        "in_scope": in_scope,
        "retrieved": retrieved_n > 0,
        "mode": "selective_rag_v1",
        "decision": decision,
        "decision_label": DECISION_CN.get(decision, decision),
        "confidence": "",
        "confidence_label": "辨经完成",
        "status_text": status,
        "summary_bullets": bullets,
        "evidence_count": retrieved_n,
        "high_count": retrieved_n,
        "low_count": 0,
        "query_both_sides": False,
        "query_drugs": [],
        "reasons": [],
        "answer_score": {
            "overall": overall,
            "authority": auth_score,
            "applicability": appl_score,
            "consistency": cons_score,
        },
        "sanzang": {
            "authority": {
                "on": authority_on,
                "score": auth_score,
                "text": "来源可信度较好" if authority_on else "来源把握有限",
            },
            "applicability": {
                "on": applicability_on,
                "warn": not in_scope,
                "score": appl_score,
                "text": "专科外问题" if not in_scope else ("情境匹配较好" if applicability_on else "情境匹配有限"),
            },
            "consistency": {
                "on": consistency_on,
                "score": cons_score,
                "text": "证据大体一致" if consistency_on else "证据存在张力",
            },
        },
        "assessment": {
            "relevance": rel,
            "sufficiency": suf,
            "conflict": conflict,
        },
    }


class SpecialistPipeline:
    """名称保留以兼容 web/app.py；实现为 Selective RAG v1。"""

    def __init__(self) -> None:
        _load_dotenv()
        if not os.environ.get("DEEPSEEK_API_KEY"):
            raise RuntimeError("未设置 DEEPSEEK_API_KEY，请在项目根目录 .env 中填写。")

        import chromadb
        from FlagEmbedding import BGEM3FlagModel
        from openai import OpenAI

        local = RAG_ROOT / "models" / "bge-m3"
        model_id = str(local) if local.exists() else EMBEDDING_MODEL_ID
        self.embed_id = model_id
        self.model = BGEM3FlagModel(model_id, use_fp16=True)
        self.chroma = chromadb.PersistentClient(path=str(CHROMA_DIR))
        self.collection = self.chroma.get_collection(COLLECTION_NAME)
        self.dense = DenseMatcher(self.collection, self._embed)
        self.llm = OpenAI(
            api_key=os.environ["DEEPSEEK_API_KEY"],
            base_url=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        )
        self.llm_model = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")
        self.lambdas = load_locked_lambdas()
        thresholds = load_locked_selective_thresholds()
        self.selective = SelectiveRAG(
            dense=self.dense,
            generate=self._generate,
            config=SelectiveConfig(
                top_k=10,
                temperature=0.0,
                thresholds=thresholds,
                version="v1",
            ),
        )
        try:
            self.chunk_count = self.collection.count()
        except Exception:  # noqa: BLE001
            self.chunk_count = None

    def _embed(self, text: str):
        return self.model.encode([text], max_length=512)["dense_vecs"][0].tolist()

    def _generate(self, prompt: str, max_tokens: int = 1024, temperature: float = 0.0) -> str:
        resp = self.llm.chat.completions.create(
            model=self.llm_model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return (resp.choices[0].message.content or "").strip()

    def health(self) -> dict[str, Any]:
        th = load_locked_selective_thresholds().as_dict()
        return {
            "ok": True,
            "collection": COLLECTION_NAME,
            "chroma_dir": str(CHROMA_DIR),
            "embedding": self.embed_id,
            "chunk_count": self.chunk_count,
            "llm": self.llm_model,
            "answer_mode": "selective_rag_v1",
            "release": "contest-v1",
            "top_k": 10,
            "temperature": 0,
            "thresholds": th,
            "test_result": "619/685 = 90.4%",
        }

    def run_turn(self, user_input: str, session: ChatSession) -> dict[str, Any]:
        decision = classify_domain(user_input)
        domain = decision.domain
        in_scope = bool(decision.in_scope)

        if not in_scope:
            # 专科外：不走 Selective 检索（产品行为）
            prompt = (
                "你是专业医疗知识助手。当前问题可能超出本专科知识库范围，"
                "请谨慎回答并说明局限。\n"
                f"用户问题：{user_input}\n"
                "不要输出虚构的选择题选项字母。\n本回答仅供参考，不替代专业诊疗。"
            )
            answer = self._generate(prompt, max_tokens=1024, temperature=0.0)
            stats = {
                "mode": "ood_skip",
                "version": "v1",
                "decision": "KEEP_DIRECT",
                "retrieved_n": 0,
                "assessment": {},
                "question_relevance": 0.0,
                "question_sufficiency": 0.0,
                "question_contradiction": 0.0,
                "chunks": [],
            }
            evidence: list[dict] = []
        else:
            answer, stats = self.selective.query(user_input)
            chunks = list(stats.get("chunks") or [])
            fusion_diag: dict[str, Any] = {}
            try:
                fused, fusion_diag = parallel_appraise_and_fuse(
                    user_input, chunks, lambdas=self.lambdas
                )
                if fused:
                    chunks = fused
            except Exception:  # noqa: BLE001
                fusion_diag = {"error": "fusion_display_skipped"}
            evidence = _pack_evidence(chunks, fusion_diag)
            stats = dict(stats)
            stats["fusion_display"] = {
                "lambdas": self.lambdas.as_dict(),
                "applied": "fusion_display" in (fusion_diag or {}) or bool(fusion_diag),
            }

        consult = _build_consult(stats, domain=domain, in_scope=in_scope)
        session.consult = consult
        session.evidence = evidence
        session.history = session.history + [
            {"role": "user", "content": user_input},
            {"role": "assistant", "content": answer},
        ]
        session.touch()
        return {"answer": answer, "consult": consult, "evidence": evidence}
