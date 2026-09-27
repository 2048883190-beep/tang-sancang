# -*- coding: utf-8 -*-
"""
domain_gate.py —— 逻辑域判定（training-free）

用于：
  1) 评测题打标 in_scope / ood
  2) 自适应检索：OOD 不硬塞专科库噪声
  3) chunk 域冲突剔除
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple


def question_stem(text: str) -> str:
    """检索和排序只用题干。选项是待选答案，不是题目的条件。"""
    t = text or ""
    for mark in ("选项：", "选项:"):
        i = t.find(mark)
        if i >= 0:
            t = t[:i]
            break
    return t.strip()

# 优先级：先命中 OOD，再命中 in-scope 专科域
OOD_PATTERNS: List[Tuple[str, str]] = [
    ("endo_other", r"垂体|肾上腺|甲状腺|希恩|Sheehan|空蝶鞍|肢端肥大|库欣|甲亢|甲减|泌乳素"),
    ("infect_other", r"钩端螺旋体|钩体病"),
    ("ophtho_other", r"Fuchs|角膜内皮|角膜营养不良"),
    ("misc_nursing", r"拔牙|牙周|医嘱|\bac\b|执行时间|护理|产妇.*减量到原用量"),
    ("epid_stat", r"灵敏度|特异度|筛检方案|患病率|阳性预测"),
]

# 肾/心关键词单独出现不够；必须同时像糖尿病语境，避免钩体肾衰等邻科被拉进来
T2DM_CONTEXT = re.compile(
    r"糖尿病|血糖|降糖|胰岛素|糖化|HbA1c|二甲双胍|SGLT|GLP|列净|列汀",
    re.I,
)
CKD_NEED_CTX = re.compile(r"肾功能不全|eGFR|慢性肾脏病|蛋白尿|透析|心衰|ASCVD", re.I)

IN_SCOPE_PATTERNS: List[Tuple[str, str]] = [
    ("acute", r"酮症|DKA|高渗|HHS|危象|抢救糖尿病|糖尿病昏迷"),
    ("ckd_cv", r"糖尿病肾病"),
    ("drug", r"二甲双胍|列净|列汀|鲁肽|阿卡波糖|拜糖平|格列|降糖药|胰岛素|SGLT|GLP|DPP"),
    ("dx_stage", r"诊断标准|分型标准|临床分期|IGT|IFG|糖化血红蛋白|糖化蛋白|ADA.*分型"),
    ("special_pop", r"老年糖尿病|妊娠期糖尿病|围手术期血糖|哺乳"),
    ("t2dm_general", r"2型糖尿病|Ⅱ型糖尿病|糖尿病"),
]

IN_SCOPE_DOMAINS = frozenset(name for name, _ in IN_SCOPE_PATTERNS)
OOD_DOMAINS = frozenset(name for name, _ in OOD_PATTERNS)


@dataclass
class DomainDecision:
    domain: str
    in_scope: bool
    reasons: List[str]

    @property
    def should_retrieve(self) -> bool:
        """OOD 默认不检索专科库，避免噪声。"""
        return self.in_scope


def classify_domain(text: str) -> DomainDecision:
    """对 query / chunk 文本做主域判定。"""
    t = text or ""
    reasons: List[str] = []
    for name, pat in OOD_PATTERNS:
        if re.search(pat, t, flags=re.IGNORECASE):
            reasons.append(f"ood:{name}")
            return DomainDecision(domain=name, in_scope=False, reasons=reasons)
    if CKD_NEED_CTX.search(t) and T2DM_CONTEXT.search(t):
        reasons.append("in_scope:ckd_cv")
        return DomainDecision(domain="ckd_cv", in_scope=True, reasons=reasons)
    for name, pat in IN_SCOPE_PATTERNS:
        if re.search(pat, t, flags=re.IGNORECASE):
            reasons.append(f"in_scope:{name}")
            return DomainDecision(domain=name, in_scope=True, reasons=reasons)
    # 题干已出现血糖、降糖、糖化等本题语境，只是没碰上上面的窄词。
    # 选项里的「糖尿病」到不了这里：调用方传入的是题干。
    if T2DM_CONTEXT.search(t):
        reasons.append("in_scope:t2dm_context")
        return DomainDecision(domain="t2dm_general", in_scope=True, reasons=reasons)
    reasons.append("fallback:other_ood")
    return DomainDecision(domain="other", in_scope=False, reasons=reasons)


def is_multi_select(gold: str) -> bool:
    letters = set(ch for ch in str(gold or "").upper() if ch in "ABCDE")
    return len(letters) > 1


def label_question(item: dict) -> dict:
    """返回带 scope / select_type / domain 字段的浅拷贝。"""
    out = dict(item)
    decision = classify_domain(item.get("question") or "")
    gold = str(item.get("gold") or item.get("ground_truth") or "")
    out["domain"] = decision.domain
    out["in_scope"] = bool(decision.in_scope)
    out["scope"] = "in_scope" if decision.in_scope else "ood"
    out["select_type"] = "multi" if is_multi_select(gold) else "single"
    out["domain_reasons"] = decision.reasons
    return out


def chunk_domain_conflict(query_domain: str, chunk_text: str, chunk_meta: Optional[dict] = None) -> Tuple[bool, str]:
    """
    query 为专科 in-scope 时，若 chunk 明显是无关 OOD 域则视为噪声。
    query 本身 OOD 时不做此过滤（通常已跳过检索）。
    """
    if query_domain in OOD_DOMAINS or query_domain == "other":
        return False, ""
    c_dec = classify_domain(
        " ".join(
            [
                str((chunk_meta or {}).get("title") or ""),
                (chunk_text or "")[:500],
            ]
        )
    )
    if c_dec.domain in OOD_DOMAINS and c_dec.domain != query_domain:
        return True, f"chunk域={c_dec.domain} 与 query域={query_domain} 冲突"
    return False, ""


def filter_main_table(questions: Sequence[dict]) -> List[dict]:
    """主表：in_scope + 单选。"""
    out: List[dict] = []
    for q in questions:
        in_scope = q.get("scope") == "in_scope" or q.get("in_scope") is True
        single = q.get("select_type", "single") == "single"
        if in_scope and single:
            out.append(q)
    return out
