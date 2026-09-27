# -*- coding: utf-8 -*-
"""基于检索段落判断：SUPPORT / CONTRADICT / NEUTRAL。

不编造医学知识；只根据传入 evidence 与候选答案做结构化评判。
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable, Dict, List, Optional, Sequence


GenerateFn = Callable[..., str]


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


def judge_stance(
    question: str,
    candidate_answer: str,
    evidence_chunks: Sequence[dict],
    *,
    generate: GenerateFn,
    max_chars: int = 3500,
) -> Dict[str, Any]:
    """返回 stance∈{SUPPORT,CONTRADICT,NEUTRAL} 与分数。"""
    parts = []
    for i, c in enumerate(evidence_chunks[:10], 1):
        parts.append(f"[{i}] {(c.get('content') or '')[:500]}")
    blob = "\n\n".join(parts)[:max_chars]
    prompt = (
        "你是医学证据审核员。只能根据【检索证据】判断它与【候选答案】的关系。\n"
        "禁止使用证据之外的知识。\n"
        "输出唯一 JSON，不要其它文字：\n"
        '{"stance":"SUPPORT|CONTRADICT|NEUTRAL",'
        '"support":0.0,"contradict":0.0,"neutral":0.0,"reason":"..."}\n'
        "分数均为 0~1，三者可不归一但需反映强弱。\n\n"
        f"【问题】\n{question}\n\n"
        f"【候选答案】\n{candidate_answer}\n\n"
        f"【检索证据】\n{blob}\n"
    )
    raw = generate(prompt, max_tokens=220, temperature=0.0)
    data = _extract_json(raw)
    stance = str(data.get("stance") or "NEUTRAL").upper().strip()
    if stance not in {"SUPPORT", "CONTRADICT", "NEUTRAL"}:
        stance = "NEUTRAL"
    return {
        "stance": stance,
        "support": _clip(data.get("support"), 0.0),
        "contradict": _clip(data.get("contradict"), 0.0),
        "neutral": _clip(data.get("neutral"), 0.5),
        "reason": str(data.get("reason") or "")[:300],
        "raw": (raw or "")[:500],
    }
