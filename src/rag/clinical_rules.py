"""
clinical_rules.py —— 循证质控匹配用的医学词典与规则（training-free，零 LLM）。

说明：
  - 不追求 NLP 完备，优先覆盖 T2DM 用药决策高频实体；
  - 全部为常量/规则，便于竞赛答辩时讲清「医学思维如何映射成检索信号」。
"""

from __future__ import annotations

import re
from typing import Dict, List, Set, Tuple

# ---------- 证据等级（仅次序，高 → 低）----------
# 整数只表示谁排在谁前面，不是效用，不能拿去乘相似度。
EVIDENCE_TIER_RANK: Dict[str, int] = {
    "guideline": 5,
    "drug_label": 4,
    "textbook": 3,
    "rct": 2,
    "opinion": 1,
}

HIGH_EVIDENCE_TIERS = frozenset({"guideline", "drug_label"})

# DocMeta.source / 标题 / 路径 → 证据等级
SOURCE_TO_TIER: Dict[str, str] = {
    "guideline": "guideline",
    "consensus": "guideline",
    "drug_label": "drug_label",
    "nmpa": "drug_label",
    "label": "drug_label",
    "textbook": "textbook",
    "lecture": "textbook",
    "paper": "rct",
    "pubmed": "rct",
    "rct": "rct",
    "wiki": "opinion",
    "medlive": "opinion",
    "web": "opinion",
}

GUIDELINE_KW = ("指南", "共识", "专家建议", "标准", "guideline", "consensus", "standards of care")
LABEL_KW = ("说明书", "用药信息", "处方信息", "label", "prescribing")
TEXTBOOK_KW = ("教材", "教案", "内科学", "textbook", "讲义")
RCT_KW = ("随机对照", "RCT", "临床试验", "meta分析", "荟萃", "pubmed", "trial")

# ---------- 药物 / 干预（I）----------
DRUG_ALIASES: Dict[str, Tuple[str, ...]] = {
    "二甲双胍": ("二甲双胍", "metformin", "格华止"),
    "达格列净": ("达格列净", "dapagliflozin", "安达唐"),
    "恩格列净": ("恩格列净", "empagliflozin", "欧唐静"),
    "卡格列净": ("卡格列净", "canagliflozin"),
    "西格列汀": ("西格列汀", "sitagliptin", "捷诺维"),
    "利拉鲁肽": ("利拉鲁肽", "liraglutide", "诺和力"),
    "司美格鲁肽": ("司美格鲁肽", "semaglutide", "诺和泰", "Ozempic"),
    "格列美脲": ("格列美脲", "glimepiride"),
    "门冬胰岛素": ("门冬胰岛素", "insulin aspart", "诺和锐"),
    "胰岛素": ("胰岛素", "insulin"),
    "SGLT2": ("SGLT2", "SGLT-2", "列净", "钠-葡萄糖共转运蛋白"),
    "GLP1": ("GLP-1", "GLP1", "胰高糖素样肽", "鲁肽"),
    "DPP4": ("DPP-4", "DPP4", "列汀", "二肽基肽酶"),
    "磺脲": ("磺脲", "磺酰脲", "SU类"),
}

# ---------- 人群 / 场景（P）----------
POPULATION_PATTERNS: List[Tuple[str, str]] = [
    ("ckd_severe", r"(eGFR\s*[＜<≤]\s*30|GFR\s*[＜<≤]\s*30|CKD\s*[45]|透析|终末期肾病|ESRD)"),
    ("ckd_moderate", r"(eGFR\s*[＜<≤]\s*45|CKD\s*3|肾功能不全|慢性肾脏病|糖尿病肾病)"),
    ("hepatic", r"(肝功能不全|肝衰竭|严重肝损害|肝硬化失代偿)"),
    ("pregnancy", r"(妊娠|孕妇|备孕|哺乳)"),
    ("elderly", r"(老年|高龄|≥\s*65|大于\s*65|65\s*岁以上)"),
    ("hypoglycemia_risk", r"(低血糖|反复低血糖)"),
    ("hf", r"(心力衰竭|心衰|HFrEF|HFpEF)"),
    ("cvd", r"(心血管|冠心病|ASCVD|心肌梗死)"),
]

# chunk 中提示「已考虑禁忌/调整」的正向安全词
SAFETY_ADJUST_KW = (
    "禁忌", "禁用", "慎用", "减量", "停用", "调整剂量", "不推荐", "避免使用",
    "相对禁忌", "绝对禁忌", "监测肾功能", "eGFR", "按肾功能",
)

# chunk 中「常规适用、未提限制」的表述（与严重约束冲突时降权）
ROUTINE_USE_KW = (
    "常规剂量", "常规用于", "首选", "一线用药", "起始治疗", "肾功能正常",
    "无需调整", "一般患者",
)


def infer_evidence_tier(metadata: dict | None, content: str = "") -> str:
    """从 DocMeta / 标题 / 正文启发式推断证据等级。"""
    meta = metadata or {}
    # 显式写入的 evidence_tier 优先
    explicit = str(meta.get("evidence_tier") or meta.get("tier") or "").lower().strip()
    if explicit in EVIDENCE_TIER_RANK:
        return explicit
    source = str(meta.get("source") or meta.get("source_type") or meta.get("doc_type") or "").lower()
    if source in SOURCE_TO_TIER:
        return SOURCE_TO_TIER[source]

    blob = " ".join(
        [
            str(meta.get("title") or ""),
            str(meta.get("file_path") or meta.get("filepath") or ""),
            str(meta.get("doc_id") or ""),
            content[:400],
        ]
    ).lower()
    # 中文关键词保留原文大小写不敏感：对中文 lower 无影响
    blob_raw = " ".join(
        [
            str(meta.get("title") or ""),
            str(meta.get("file_path") or meta.get("filepath") or ""),
            content[:400],
        ]
    )

    def hit(kws: Tuple[str, ...]) -> bool:
        return any(k.lower() in blob or k in blob_raw for k in kws)

    if hit(GUIDELINE_KW):
        return "guideline"
    if hit(LABEL_KW):
        return "drug_label"
    if hit(TEXTBOOK_KW):
        return "textbook"
    if hit(RCT_KW):
        return "rct"
    return "opinion"


def extract_drugs(text: str) -> Set[str]:
    """抽取规范化药名集合。"""
    t = text or ""
    found: Set[str] = set()
    for canon, aliases in DRUG_ALIASES.items():
        if any(a.lower() in t.lower() or a in t for a in aliases):
            found.add(canon)
    return found


def extract_populations(text: str) -> Set[str]:
    t = text or ""
    found: Set[str] = set()
    for name, pat in POPULATION_PATTERNS:
        if re.search(pat, t, flags=re.IGNORECASE):
            found.add(name)
    return found


def parse_egfr_threshold(text: str) -> float | None:
    """若文本出现 eGFR/GFR 上限阈值，返回数值；否则 None。"""
    m = re.search(r"(?:eGFR|GFR)\s*[＜<≤]\s*(\d+(?:\.\d+)?)", text or "", flags=re.I)
    if m:
        return float(m.group(1))
    return None


def clinical_conflict(query: str, chunk_text: str, chunk_meta: dict | None = None) -> Tuple[bool, str]:
    """
    判断 chunk 是否与 query 的临床约束冲突。
    返回 (is_conflict, reason)。
    """
    q = query or ""
    c = chunk_text or ""
    pops = extract_populations(q)
    drugs_q = extract_drugs(q)
    drugs_c = extract_drugs(c)
    has_safety = any(k in c for k in SAFETY_ADJUST_KW)
    has_routine = any(k in c for k in ROUTINE_USE_KW)

    # 严重肾功能约束 + 二甲双胍常规论述且未提禁忌调整
    egfr = parse_egfr_threshold(q)
    severe_renal = ("ckd_severe" in pops) or (egfr is not None and egfr <= 30)
    if severe_renal and ("二甲双胍" in drugs_q or "二甲双胍" in drugs_c or "二甲双胍" in c):
        if ("二甲双胍" in c) and (not has_safety) and has_routine:
            return True, "eGFR/严重肾功能约束下，chunk 以常规/首选论述二甲双胍且未提禁忌调整"

    if severe_renal and ("二甲双胍" in c) and ("肾功能正常" in c) and (not has_safety):
        return True, "严重肾功能约束 vs chunk 仅讨论肾功能正常者使用二甲双胍"

    # 妊娠 + 多数口服降糖常规推荐且未提妊娠禁忌
    if "pregnancy" in pops:
        risky = drugs_c & {"二甲双胍", "达格列净", "恩格列净", "西格列汀", "格列美脲", "SGLT2", "DPP4"}
        if risky and (not has_safety) and ("妊娠" not in c and "孕妇" not in c):
            return True, f"妊娠约束下 chunk 讨论 {','.join(sorted(risky))} 未提及妊娠相关限制"

    # 肝功能严重不全 + 常规剂量且无安全提示
    if "hepatic" in pops and has_routine and (not has_safety) and drugs_c:
        return True, "肝功能约束下 chunk 为常规用药论述且无慎用/调整提示"

    return False, ""


def pico_from_text(text: str) -> Dict[str, Set[str]]:
    """规则版 PICO 抽取（P/I 为主；C/O 弱信号）。"""
    t = text or ""
    pico = {
        "P": extract_populations(t),
        "I": extract_drugs(t),
        "C": set(),
        "O": set(),
    }
    if any(k in t for k in ("相比", "对照", "安慰剂", "优于", "非劣")):
        pico["C"].add("comparative")
    if any(k in t for k in ("HbA1c", "糖化", "心血管", "体重", "低血糖", "终点", "获益")):
        pico["O"].add("clinical_outcome")
    return pico


def framework_conflict(framework: str, chunk_text: str) -> Tuple[bool, str]:
    """
    Stage-1 高证据框架 vs 低证据 chunk 的轻量冲突检测。
    规则：框架明确「禁忌/不推荐/禁用」某药，而 chunk 仍「首选/常规」推荐该药且无安全措辞。
    """
    fw = framework or ""
    c = chunk_text or ""
    if not fw or not c:
        return False, ""
    drugs_fw = extract_drugs(fw)
    drugs_c = extract_drugs(c)
    shared = drugs_fw & drugs_c
    if not shared:
        return False, ""
    fw_neg = any(k in fw for k in ("禁忌", "禁用", "不推荐", "避免", "停用", "不宜"))
    c_pos = any(k in c for k in ROUTINE_USE_KW) or any(k in c for k in ("推荐", "可选用", "适用于"))
    c_safe = any(k in c for k in SAFETY_ADJUST_KW)
    if fw_neg and c_pos and (not c_safe):
        return True, f"框架否定/限制 {','.join(sorted(shared))}，chunk 仍作常规推荐"
    return False, ""


def pico_alignment_score(q_pico: Dict[str, Set[str]], c_pico: Dict[str, Set[str]]) -> Tuple[float, List[str]]:
    """
    返回对齐分 ∈ [0,1] 与原因列表。
    关键维度：I（干预）权重最高；P 次之。
    """
    reasons: List[str] = []
    score = 0.5  # 中性底分（query 无 I 时不过度惩罚）

    q_i, c_i = q_pico.get("I") or set(), c_pico.get("I") or set()
    q_p, c_p = q_pico.get("P") or set(), c_pico.get("P") or set()

    if q_i:
        inter = q_i & c_i
        if inter:
            score += 0.35
            reasons.append(f"I对齐:{','.join(sorted(inter))}")
        else:
            # query 点名药物，chunk 完全未覆盖 → 显著降分
            score -= 0.35
            reasons.append(f"I未对齐(需{','.join(sorted(q_i))})")
    if q_p:
        inter_p = q_p & c_p
        if inter_p:
            score += 0.15
            reasons.append(f"P对齐:{','.join(sorted(inter_p))}")
        elif c_p and not inter_p:
            score -= 0.1
            reasons.append("P部分错配")

    return max(0.0, min(1.0, score)), reasons
