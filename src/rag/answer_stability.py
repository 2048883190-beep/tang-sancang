# -*- coding: utf-8 -*-
"""答案稳定性与证据充分性评判。

先有 Direct Answer，再判断检索证据是否足以推翻它。
禁止「看到资料就重答」；由上层 SelectiveRAG 决定 KEEP_DIRECT / USE_EVIDENCE。
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable, Dict, Sequence


GenerateFn = Callable[..., str]

HIGH_RISK_RE = re.compile(
    r"(禁忌|禁用|不推荐|不良反应|相互作用|妊娠|孕妇|哺乳|肝功能|肾功能|eGFR|GFR|"
    r"低血糖|胰岛素|剂量|一线|二线|首选|指南|适应证|适应症|禁忌症|特殊人群|"
    r"儿童|老年|透析|酮症|乳酸酸中毒)",
    re.I,
)


def is_high_risk_question(question: str) -> bool:
    return bool(HIGH_RISK_RE.search(question or ""))


def _extract_json(text: str) -> dict:
    raw = (text or "").strip()
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{[\s\S]*\}", raw)
        if not m:
            return {}
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            return {}


def _clip(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return default
    return max(0.0, min(1.0, v))


def assess_evidence_vs_direct(
    question: str,
    direct_answer: str,
    evidence_chunks: Sequence[dict],
    *,
    generate: GenerateFn,
    max_chars: int = 3500,
) -> Dict[str, Any]:
    """结构化评估：相关/支持/冲突/充分/是否应覆盖 Direct。"""
    parts = []
    for i, c in enumerate(evidence_chunks[:10], 1):
        parts.append(f"[{i}] {(c.get('content') or '')[:500]}")
    blob = "\n\n".join(parts)[:max_chars]
    prompt = (
        "你是医学证据审核员。已有【初始答案】，并有【检索证据】。\n"
        "任务：判断证据是否足以推翻初始答案。只根据证据，不要编造。\n"
        "区分：主题相关 ≠ 足以证明选项。\n"
        "输出唯一 JSON：\n"
        "{"
        '"direct_answer":"...",'
        '"evidence_relevance":0.0,'
        '"evidence_support":0.0,'
        '"evidence_conflict":0.0,'
        '"evidence_sufficiency":0.0,'
        '"should_override":false,'
        '"reason":"..."'
        "}\n"
        "分数 0~1。should_override=true 仅当：证据高度相关、明确支持另一结论、"
        "且足以证明初始答案错误。\n\n"
        f"【问题】\n{question}\n\n"
        f"【初始答案】\n{direct_answer}\n\n"
        f"【检索证据】\n{blob}\n"
    )
    raw = generate(prompt, max_tokens=280, temperature=0.0)
    data = _extract_json(raw)
    return {
        "direct_answer": str(data.get("direct_answer") or direct_answer)[:200],
        "evidence_relevance": _clip(data.get("evidence_relevance"), 0.0),
        "evidence_support": _clip(data.get("evidence_support"), 0.0),
        "evidence_conflict": _clip(data.get("evidence_conflict"), 0.0),
        "evidence_sufficiency": _clip(data.get("evidence_sufficiency"), 0.0),
        "should_override": bool(data.get("should_override")),
        "reason": str(data.get("reason") or "")[:400],
        "raw": (raw or "")[:600],
    }
