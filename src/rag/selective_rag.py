# -*- coding: utf-8 -*-
"""Risk-Aware Selective RAG — Contest Release = v1 only.

正式参赛路径：
  Direct + Dense Top-10 + Evidence Assessment + Selective Gate
  → KEEP_DIRECT / USE_RAG

v2（option_verify / TARGETED_RAG）已隔离至 archive/research_experiments/，
本文件保留兼容入口但默认强制 version=v1，运行时不依赖 option_verify。

不修改 parallel_fusion / hybrid_retrieve / hierarchical_rag 的检索主公式；
演示端可对召回段落做三路打分展示（权威/适用/一致）。
禁止 gold 参与任何运行时决策。
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .answer_stability import assess_evidence_vs_direct, is_high_risk_question
from .contradiction import judge_stance
from .matching_layer import DenseMatcher

GenerateFn = Callable[..., str]

MCQ_TAIL = (
    "\n\n若题目含选项（A/B/C/D/E），请先简要推理，"
    "并在文末单独一行写：最终选项：X（多选则连写字母，如 BCD）。"
    "\n本回答仅供参考，不替代专业诊疗。"
)

CLINICAL_TAIL = (
    "\n\n请用临床表述作答：结论要点、依据条件、注意事项。"
    "禁止编造检查数值；资料不足请明确写出。"
    "本题不是选择题：禁止输出 A/B/C/D/E 选项字母或「最终选项」行。"
    "\n本回答仅供参考，不替代专业诊疗。"
)

_OPTION_BLOCK_RE = re.compile(r"选项[:：]\s*(.*)$", re.S)
_OPTION_SPLIT_RE = re.compile(r"(?:^|\s)([A-E])[\.．、]\s*")


def parse_mcq_options(question: str) -> Dict[str, str]:
    """从题干解析选项（v1 仅需字母抽取辅助；不依赖 option_verify）。"""
    text = question or ""
    m = _OPTION_BLOCK_RE.search(text)
    blob = m.group(1).strip() if m else text
    parts = _OPTION_SPLIT_RE.split(blob)
    out: Dict[str, str] = {}
    i = 1
    while i + 1 < len(parts):
        letter = parts[i].upper()
        content = (parts[i + 1] or "").strip()
        if letter in "ABCDE" and content:
            out[letter] = content[:200]
        i += 2
    return out


def _is_mcq(question: str) -> bool:
    opts = parse_mcq_options(question)
    if len(opts) < 2:
        return False
    if _OPTION_BLOCK_RE.search(question or ""):
        return True
    return "A" in opts and "B" in opts


def _answer_tail(question: str) -> str:
    return MCQ_TAIL if _is_mcq(question) else CLINICAL_TAIL


def is_promising_option(*_a, **_k) -> bool:
    """v2 API stub（参赛主路径不使用）。"""
    return False


def verify_options(*_a, **_k) -> Dict[str, Any]:
    """v2 API stub（参赛主路径不使用）。"""
    return {}


@dataclass
class SelectiveThresholds:
    """分数门控：相关且充分才采用 RAG，否则保留 Direct。"""

    relevance_min: float = 0.60
    sufficiency_min: float = 0.60
    # 高风险题：要求更充分才改用 RAG（减少禁忌/剂量类噪声伤害）
    risk_relevance_min: float = 0.70
    risk_sufficiency_min: float = 0.70

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class SelectiveV2Params:
    """v2 option-level / targeted 参数（仅 validation 锁定）。"""

    enable_option_verify: bool = True
    enable_targeted: bool = False
    option_support_min: float = 0.60
    option_margin_min: float = 0.10
    contradiction_max: float = 0.55
    near_miss_suf_floor: float = 0.50
    risk_option_support_bonus: float = 0.10
    targeted_trigger: str = "strict"  # strict | medium | off
    targeted_top_k: int = 5
    override_sufficiency_min: float = 0.55
    risk_override_sufficiency_min: float = 0.75

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class SelectiveConfig:
    top_k: int = 10
    temperature: float = 0.0
    thresholds: Optional[SelectiveThresholds] = None
    v2: Optional[SelectiveV2Params] = None
    version: str = "v1"  # v1 | v2a | v2b

    def __post_init__(self):
        if self.thresholds is None:
            self.thresholds = SelectiveThresholds()
        if self.v2 is None:
            self.v2 = SelectiveV2Params()
        if self.version == "v1":
            self.v2.enable_option_verify = False
            self.v2.enable_targeted = False
        elif self.version == "v2a":
            self.v2.enable_option_verify = True
            self.v2.enable_targeted = False
        elif self.version == "v2b":
            self.v2.enable_option_verify = True
            self.v2.enable_targeted = True


def parse_final_letters(answer: str, force_single: bool = True) -> List[str]:
    text = answer or ""
    m = re.search(r"最终选项[:：]\s*([A-Ea-e]+)", text)
    if m:
        letters = [ch.upper() for ch in m.group(1) if ch.upper() in "ABCDE"]
    else:
        letters = re.findall(r"[A-E]", text.upper())
    if force_single and letters:
        return [letters[-1]]
    out: List[str] = []
    for ch in letters:
        if ch not in out:
            out.append(ch)
    return out


def _direct_letter(direct_answer: str) -> Optional[str]:
    letters = parse_final_letters(direct_answer)
    return letters[0] if letters else None


class SelectiveRAG:
    def __init__(
        self,
        *,
        dense: DenseMatcher,
        generate: GenerateFn,
        config: Optional[SelectiveConfig] = None,
    ):
        self.dense = dense
        self.generate = generate
        self.config = config or SelectiveConfig()

    def _direct(self, question: str) -> str:
        prompt = (
            "你是专业医疗知识助手。请回答下列问题。\n"
            f"用户问题：{question}{_answer_tail(question)}"
        )
        return self.generate(
            prompt, max_tokens=1024, temperature=self.config.temperature
        )

    def _rag_answer(self, question: str, chunks: Sequence[dict]) -> str:
        """与 Naive Dense 相同的资料生成（不用『见资料就推翻』）。"""
        ctx = "\n\n---\n\n".join(
            f"[来源{i}] {(c.get('content') or '')[:900]}" for i, c in enumerate(chunks, 1)
        )
        prompt = (
            "你是专业医疗知识助手。请基于参考资料回答问题。\n"
            f"用户问题：{question}\n\n参考资料：\n{ctx}{_answer_tail(question)}"
        )
        return self.generate(
            prompt, max_tokens=1024, temperature=self.config.temperature
        )

    def _q_thresholds(self, high_risk: bool) -> Tuple[float, float]:
        th = self.config.thresholds or SelectiveThresholds()
        if high_risk:
            return th.risk_relevance_min, th.risk_sufficiency_min
        return th.relevance_min, th.sufficiency_min

    def decide_gate(self, assessment: dict, *, high_risk: bool) -> Tuple[str, dict]:
        rel = float(assessment.get("evidence_relevance") or 0)
        suff = float(assessment.get("evidence_sufficiency") or 0)
        r_min, f_min = self._q_thresholds(high_risk)
        ok = rel >= r_min and suff >= f_min
        detail = {
            "high_risk": high_risk,
            "relevance_min": r_min,
            "sufficiency_min": f_min,
            "relevance": rel,
            "sufficiency": suff,
            "passed": ok,
        }
        return ("USE_RAG" if ok else "KEEP_DIRECT"), detail

    def _is_near_miss(self, assessment: dict, *, high_risk: bool) -> bool:
        v2 = self.config.v2 or SelectiveV2Params()
        rel = float(assessment.get("evidence_relevance") or 0)
        suff = float(assessment.get("evidence_sufficiency") or 0)
        r_min, f_min = self._q_thresholds(high_risk)
        if rel < r_min or suff >= f_min:
            return False
        return suff >= float(v2.near_miss_suf_floor)

    def _is_weak_retrieval(self, assessment: dict) -> bool:
        rel = float(assessment.get("evidence_relevance") or 0)
        suff = float(assessment.get("evidence_sufficiency") or 0)
        return rel < 0.30 and suff < 0.30

    def _option_support_ok(
        self,
        verification: dict,
        *,
        high_risk: bool,
        for_near_miss: bool = False,
    ) -> Tuple[bool, str]:
        v2 = self.config.v2 or SelectiveV2Params()
        if not verification or not verification.get("best_supported_option"):
            return False, "no_best_option"
        if verification.get("has_strong_contradiction"):
            return False, "strong_option_contradiction"
        best = verification.get("best_supported_option")
        score = float(verification.get("best_score") or 0)
        margin = float(verification.get("margin") or 0)
        support_min = float(v2.option_support_min)
        margin_min = float(v2.option_margin_min)
        if high_risk:
            # 高风险不降低 question sufficiency；option 门槛略抬高
            support_min = min(0.95, support_min + float(v2.risk_option_support_bonus))

        stances = verification.get("option_stances") or {}
        support_letters = [k for k, st in stances.items() if st == "SUPPORT"]
        unique_support = len(support_letters) == 1 and support_letters[0] == best

        # near-miss：允许「唯一 SUPPORT」形成证据闭环（仍要求 margin）
        if for_near_miss and unique_support:
            floor = max(0.45, support_min - 0.15)
            if score >= floor and margin >= max(0.05, margin_min * 0.5):
                return True, "unique_support_near_miss"

        if score < support_min:
            return False, f"option_support_below_{support_min}"
        if margin < margin_min:
            return False, f"option_margin_below_{margin_min}"
        st = stances.get(best)
        if st and st != "SUPPORT":
            return False, f"best_stance_{st}"
        return True, "option_support_ok"

    def answer_preservation(
        self,
        *,
        direct_letter: Optional[str],
        verification: dict,
        assessment: dict,
        high_risk: bool,
        allow_confirm: bool = True,
    ) -> Tuple[bool, str]:
        """硬保护：证据未明确证明 Direct 错误时，不允许覆盖。"""
        v2 = self.config.v2 or SelectiveV2Params()
        best = verification.get("best_supported_option")
        if not best:
            return False, "preserve_no_best_option"
        if direct_letter and best == direct_letter:
            if allow_confirm:
                return True, "preserve_confirms_direct"
            return False, "preserve_same_as_direct_noop"

        ok_opt, why = self._option_support_ok(
            verification, high_risk=high_risk, for_near_miss=False
        )
        if not ok_opt:
            return False, f"preserve_block_{why}"

        conflict = float(assessment.get("evidence_conflict") or 0)
        if conflict > float(v2.contradiction_max) or verification.get(
            "has_strong_contradiction"
        ):
            return False, "preserve_unresolved_contradiction"

        suf = float(assessment.get("evidence_sufficiency") or 0)
        suf_need = (
            float(v2.risk_override_sufficiency_min)
            if high_risk
            else float(v2.override_sufficiency_min)
        )
        if suf < suf_need:
            return False, f"preserve_sufficiency_below_{suf_need}"

        rel = float(assessment.get("evidence_relevance") or 0)
        r_min, _ = self._q_thresholds(high_risk)
        if rel < min(0.45, r_min):
            return False, "preserve_relevance_too_weak"
        return True, "preserve_override_allowed"

    def decide_v2(
        self,
        assessment: dict,
        verification: dict,
        *,
        high_risk: bool,
        direct_letter: Optional[str],
    ) -> Tuple[str, dict]:
        v2 = self.config.v2 or SelectiveV2Params()
        v1_decision, gate = self.decide_gate(assessment, high_risk=high_risk)
        detail: Dict[str, Any] = {
            "v1_decision": v1_decision,
            "gate": gate,
            "near_miss": False,
            "option_ok": False,
            "option_reason": "",
            "preservation_ok": False,
            "preservation_reason": "",
            "override_reason": "",
        }
        if not v2.enable_option_verify:
            return v1_decision, detail

        conflict = float(assessment.get("evidence_conflict") or 0)
        severe_conflict = conflict > float(v2.contradiction_max) or bool(
            verification.get("has_strong_contradiction")
        )
        opt_ok, opt_why = self._option_support_ok(
            verification, high_risk=high_risk, for_near_miss=False
        )
        detail["option_ok"] = opt_ok
        detail["option_reason"] = opt_why

        # Path A：保留 v1 USE_RAG（option-level 不二次拦截已充分证据）
        # 仅在「严重冲突且 best option 与 Direct 不一致」时保守 KEEP
        if v1_decision == "USE_RAG":
            best = verification.get("best_supported_option")
            if (
                severe_conflict
                and best
                and direct_letter
                and best != direct_letter
            ):
                detail["override_reason"] = "block_use_conflict_vs_direct"
                return "KEEP_DIRECT", detail
            detail["override_reason"] = "v1_pass_keep_use"
            detail["preservation_ok"] = True
            detail["preservation_reason"] = "v1_sufficient_preserve_use"
            return "USE_RAG", detail

        near = self._is_near_miss(assessment, high_risk=high_risk)
        detail["near_miss"] = near
        if near and not severe_conflict:
            near_ok, near_why = self._option_support_ok(
                verification, high_risk=high_risk, for_near_miss=True
            )
            detail["option_ok"] = near_ok
            detail["option_reason"] = near_why
            if near_ok:
                allow, why = self.answer_preservation(
                    direct_letter=direct_letter,
                    verification=verification,
                    assessment=assessment,
                    high_risk=high_risk,
                    allow_confirm=True,
                )
                if not allow and why.startswith("preserve_sufficiency"):
                    suf = float(assessment.get("evidence_sufficiency") or 0)
                    if suf >= float(v2.near_miss_suf_floor) and (
                        not high_risk
                        or float(verification.get("best_score") or 0)
                        >= float(v2.option_support_min)
                        + float(v2.risk_option_support_bonus)
                    ):
                        allow, why = True, "preserve_near_miss_option_loop"
                detail["preservation_ok"] = allow
                detail["preservation_reason"] = why
                if allow:
                    detail["override_reason"] = "near_miss_option_unlock"
                    return "USE_RAG", detail

        if v2.enable_targeted and v2.targeted_trigger != "off":
            weak = self._is_weak_retrieval(assessment)
            partial = (
                float(assessment.get("evidence_relevance") or 0) < float(gate["relevance_min"])
                or float(assessment.get("evidence_sufficiency") or 0)
                < float(gate["sufficiency_min"])
            )
            promising = is_promising_option(
                verification,
                support_min=float(v2.option_support_min),
                margin_min=float(v2.option_margin_min),
                relax=0.85 if v2.targeted_trigger == "medium" else 0.95,
            )
            best = verification.get("best_supported_option")
            differs = bool(best and direct_letter and best != direct_letter)
            trigger_ok = False
            if v2.targeted_trigger == "strict":
                trigger_ok = weak and promising and differs
            elif v2.targeted_trigger == "medium":
                trigger_ok = (weak or partial) and promising and differs
            detail["targeted_candidate"] = {
                "weak": weak,
                "partial": partial,
                "promising": promising,
                "differs_from_direct": differs,
                "best": best,
            }
            if trigger_ok and not severe_conflict:
                detail["override_reason"] = "trigger_targeted"
                return "TARGETED_RAG", detail

        detail["override_reason"] = "keep_default"
        return "KEEP_DIRECT", detail

    def _run_option_verify(
        self,
        question: str,
        options: Dict[str, str],
        chunks: Sequence[dict],
        assessment: dict,
        direct_letter: Optional[str],
    ) -> dict:
        return verify_options(
            question,
            options,
            list(chunks),
            direct_answer=direct_letter,
            generate=self.generate,
            question_relevance=float(assessment.get("evidence_relevance") or 0),
            question_sufficiency=float(assessment.get("evidence_sufficiency") or 0),
            question_conflict=float(assessment.get("evidence_conflict") or 0),
        )

    def _targeted_retrieve(
        self, question: str, option_letter: str, option_text: str
    ) -> Tuple[List[dict], str]:
        v2 = self.config.v2 or SelectiveV2Params()
        query = f"{question} {option_text}".strip()
        base = self.dense.retrieve(question, k=v2.targeted_top_k, role="forward")
        focused = self.dense.retrieve(query, k=v2.targeted_top_k, role="forward")
        seen = set()
        merged: List[dict] = []
        for c in focused + base:
            cid = str(c.get("id") or "")
            if cid and cid in seen:
                continue
            if cid:
                seen.add(cid)
            merged.append(c)
            if len(merged) >= v2.targeted_top_k:
                break
        return merged, query

    def query(self, question: str) -> Tuple[str, dict]:
        direct = self._direct(question)
        direct_letters = parse_final_letters(direct)
        direct_letter = direct_letters[0] if direct_letters else None
        options = parse_mcq_options(question)

        chunks = self.dense.retrieve(question, k=self.config.top_k, role="forward")
        high_risk = is_high_risk_question(question)
        assessment = assess_evidence_vs_direct(
            question, direct, chunks, generate=self.generate
        )
        stance = judge_stance(question, direct, chunks, generate=self.generate)
        assessment = dict(assessment)
        assessment["evidence_conflict"] = max(
            float(assessment.get("evidence_conflict") or 0),
            float(stance.get("contradict") or 0),
        )

        v2 = self.config.v2 or SelectiveV2Params()
        verification: Dict[str, Any] = {}
        if v2.enable_option_verify:
            verification = self._run_option_verify(
                question, options, chunks, assessment, direct_letter
            )
            decision, gate = self.decide_v2(
                assessment,
                verification,
                high_risk=high_risk,
                direct_letter=direct_letter,
            )
        else:
            decision, g1 = self.decide_gate(assessment, high_risk=high_risk)
            gate = {"v1_decision": decision, "gate": g1}

        targeted_triggered = False
        targeted_query = ""
        targeted_chunks: List[dict] = []
        targeted_verification: Dict[str, Any] = {}
        preservation_reason = (gate or {}).get("preservation_reason", "")
        override_reason = (gate or {}).get("override_reason", "")

        rag_answer = ""
        final_decision = decision
        use_chunks: Sequence[dict] = chunks

        if decision == "TARGETED_RAG":
            targeted_triggered = True
            best = (verification or {}).get("best_supported_option")
            opt_text = options.get(best or "", "")
            if best and opt_text:
                targeted_chunks, targeted_query = self._targeted_retrieve(
                    question, best, opt_text
                )
                targeted_assessment = assess_evidence_vs_direct(
                    question, direct, targeted_chunks, generate=self.generate
                )
                t_stance = judge_stance(
                    question, direct, targeted_chunks, generate=self.generate
                )
                targeted_assessment = dict(targeted_assessment)
                targeted_assessment["evidence_conflict"] = max(
                    float(targeted_assessment.get("evidence_conflict") or 0),
                    float(t_stance.get("contradict") or 0),
                )
                targeted_verification = self._run_option_verify(
                    question,
                    options,
                    targeted_chunks,
                    targeted_assessment,
                    direct_letter,
                )
                old_en = v2.enable_targeted
                v2.enable_targeted = False
                try:
                    re_dec, re_gate = self.decide_v2(
                        targeted_assessment,
                        targeted_verification,
                        high_risk=high_risk,
                        direct_letter=direct_letter,
                    )
                finally:
                    v2.enable_targeted = old_en

                allow, why = self.answer_preservation(
                    direct_letter=direct_letter,
                    verification=targeted_verification,
                    assessment=targeted_assessment,
                    high_risk=high_risk,
                    allow_confirm=True,
                )
                if (
                    not allow
                    and why.startswith("preserve_sufficiency")
                    and float(targeted_verification.get("best_score") or 0)
                    >= float(v2.option_support_min) + 0.05
                    and float(targeted_verification.get("margin") or 0)
                    >= float(v2.option_margin_min)
                    and not targeted_verification.get("has_strong_contradiction")
                    and not high_risk
                ):
                    allow, why = True, "preserve_targeted_strong_option"

                preservation_reason = why
                override_reason = (re_gate or {}).get("override_reason", "") + f"|{why}"
                if allow and (
                    re_dec == "USE_RAG"
                    or (
                        targeted_verification.get("best_supported_option")
                        and targeted_verification.get("best_supported_option")
                        != direct_letter
                        and float(targeted_verification.get("best_score") or 0)
                        >= float(v2.option_support_min)
                    )
                ):
                    final_decision = "USE_TARGETED_RAG"
                    use_chunks = targeted_chunks
                    verification = targeted_verification
                    assessment = targeted_assessment
                else:
                    final_decision = "KEEP_DIRECT"
                    override_reason = f"targeted_rejected|{why}"
            else:
                final_decision = "KEEP_DIRECT"
                override_reason = "targeted_missing_option_text"

        if final_decision in {"USE_RAG", "USE_TARGETED_RAG"}:
            rag_answer = self._rag_answer(question, use_chunks)
            final = rag_answer
        else:
            final = direct
            final_decision = "KEEP_DIRECT"

        stats: Dict[str, Any] = {
            "mode": "selective_rag",
            "version": self.config.version,
            "decision": final_decision,
            "high_risk": high_risk,
            "direct_answer": direct,
            "direct_letters": direct_letters,
            "dense_letters": parse_final_letters(rag_answer) if rag_answer else [],
            "final_letters": parse_final_letters(final),
            "assessment": assessment,
            "stance": stance,
            "gate": gate,
            "option_verification": verification,
            "thresholds": (self.config.thresholds or SelectiveThresholds()).as_dict(),
            "v2_params": v2.as_dict(),
            "retrieved_n": len(chunks),
            "retrieved_ids": [str(c.get("id") or "") for c in chunks],
            "chunks": list(use_chunks) if final_decision in {"USE_RAG", "USE_TARGETED_RAG"} else list(chunks),
            "rag_answer": (rag_answer or "")[:1500],
            "answer_changed": parse_final_letters(final) != direct_letters,
            "targeted_triggered": targeted_triggered,
            "targeted_query": targeted_query[:300],
            "targeted_evidence_count": len(targeted_chunks),
            "targeted_retrieved_ids": [str(c.get("id") or "") for c in targeted_chunks],
            "targeted_verification": targeted_verification,
            "override_reason": override_reason,
            "preservation_reason": preservation_reason,
            "question_relevance": float(assessment.get("evidence_relevance") or 0),
            "question_sufficiency": float(assessment.get("evidence_sufficiency") or 0),
            "question_contradiction": float(assessment.get("evidence_conflict") or 0),
            "best_option": (verification or {}).get("best_supported_option"),
            "second_option": (verification or {}).get("second_supported_option"),
            "option_margin": (verification or {}).get("margin"),
            "option_scores": (verification or {}).get("option_scores"),
            "option_stances": (verification or {}).get("option_stances"),
        }
        return final, stats

    def prepare_caches(self, question: str) -> dict:
        direct = self._direct(question)
        direct_letter = _direct_letter(direct)
        options = parse_mcq_options(question)
        chunks = self.dense.retrieve(question, k=self.config.top_k, role="forward")
        high_risk = is_high_risk_question(question)
        assessment = assess_evidence_vs_direct(
            question, direct, chunks, generate=self.generate
        )
        stance = judge_stance(question, direct, chunks, generate=self.generate)
        assessment = dict(assessment)
        assessment["evidence_conflict"] = max(
            float(assessment.get("evidence_conflict") or 0),
            float(stance.get("contradict") or 0),
        )
        verification = self._run_option_verify(
            question, options, chunks, assessment, direct_letter
        )
        rag_answer = self._rag_answer(question, chunks)

        targeted = None
        best = verification.get("best_supported_option")
        if best and options.get(best):
            t_chunks, t_query = self._targeted_retrieve(question, best, options[best])
            t_assessment = assess_evidence_vs_direct(
                question, direct, t_chunks, generate=self.generate
            )
            t_stance = judge_stance(
                question, direct, t_chunks, generate=self.generate
            )
            t_assessment = dict(t_assessment)
            t_assessment["evidence_conflict"] = max(
                float(t_assessment.get("evidence_conflict") or 0),
                float(t_stance.get("contradict") or 0),
            )
            t_verification = self._run_option_verify(
                question, options, t_chunks, t_assessment, direct_letter
            )
            t_rag = self._rag_answer(question, t_chunks)
            targeted = {
                "query": t_query,
                "chunks": [
                    {
                        "id": c.get("id"),
                        "content": c.get("content"),
                        "score": c.get("score"),
                    }
                    for c in t_chunks
                ],
                "assessment": t_assessment,
                "verification": t_verification,
                "rag_answer": t_rag,
            }

        return {
            "question": question,
            "options": options,
            "direct": direct,
            "direct_letter": direct_letter,
            "chunks": [
                {"id": c.get("id"), "content": c.get("content"), "score": c.get("score")}
                for c in chunks
            ],
            "high_risk": high_risk,
            "assessment": assessment,
            "stance": stance,
            "verification": verification,
            "rag_answer": rag_answer,
            "targeted": targeted,
        }

    def apply_thresholds(
        self,
        cache: dict,
        thresholds: SelectiveThresholds,
        v2_params: Optional[SelectiveV2Params] = None,
    ) -> dict:
        old_th = self.config.thresholds
        old_v2 = self.config.v2
        old_ver = self.config.version
        self.config.thresholds = thresholds
        if v2_params is not None:
            self.config.v2 = v2_params
            if v2_params.enable_targeted:
                self.config.version = "v2b"
            elif v2_params.enable_option_verify:
                self.config.version = "v2a"
            else:
                self.config.version = "v1"
        try:
            assessment = cache["assessment"]
            verification = cache.get("verification") or {}
            high_risk = bool(cache["high_risk"])
            direct_letter = cache.get("direct_letter") or _direct_letter(cache["direct"])

            if not (self.config.v2 and self.config.v2.enable_option_verify):
                decision, gate = self.decide_gate(assessment, high_risk=high_risk)
                final = cache["rag_answer"] if decision == "USE_RAG" else cache["direct"]
                return {
                    "decision": decision,
                    "gate": {"v1_decision": decision, "gate": gate},
                    "final": final,
                    "direct_letters": parse_final_letters(cache["direct"]),
                    "final_letters": parse_final_letters(final),
                    "override_reason": "v1_only",
                    "preservation_reason": "",
                    "targeted_triggered": False,
                }

            decision, gate = self.decide_v2(
                assessment,
                verification,
                high_risk=high_risk,
                direct_letter=direct_letter,
            )
            final_decision = decision
            final_text = cache["direct"]
            targeted_triggered = False
            preservation_reason = gate.get("preservation_reason", "")
            override_reason = gate.get("override_reason", "")

            if decision == "USE_RAG":
                final_text = cache["rag_answer"]
            elif decision == "TARGETED_RAG" and cache.get("targeted"):
                targeted_triggered = True
                t = cache["targeted"]
                old_en = self.config.v2.enable_targeted
                self.config.v2.enable_targeted = False
                try:
                    re_dec, re_gate = self.decide_v2(
                        t["assessment"],
                        t["verification"],
                        high_risk=high_risk,
                        direct_letter=direct_letter,
                    )
                finally:
                    self.config.v2.enable_targeted = old_en
                allow, why = self.answer_preservation(
                    direct_letter=direct_letter,
                    verification=t["verification"],
                    assessment=t["assessment"],
                    high_risk=high_risk,
                    allow_confirm=True,
                )
                v2p = self.config.v2
                if (
                    not allow
                    and why.startswith("preserve_sufficiency")
                    and float(t["verification"].get("best_score") or 0)
                    >= float(v2p.option_support_min) + 0.05
                    and float(t["verification"].get("margin") or 0)
                    >= float(v2p.option_margin_min)
                    and not t["verification"].get("has_strong_contradiction")
                    and not high_risk
                ):
                    allow, why = True, "preserve_targeted_strong_option"
                preservation_reason = why
                override_reason = (re_gate or {}).get("override_reason", "") + f"|{why}"
                if allow and (
                    re_dec == "USE_RAG"
                    or (
                        t["verification"].get("best_supported_option")
                        and t["verification"].get("best_supported_option")
                        != direct_letter
                        and float(t["verification"].get("best_score") or 0)
                        >= float(v2p.option_support_min)
                    )
                ):
                    final_decision = "USE_TARGETED_RAG"
                    final_text = t["rag_answer"]
                else:
                    final_decision = "KEEP_DIRECT"
                    override_reason = f"targeted_rejected|{why}"
            else:
                final_decision = "KEEP_DIRECT"

            return {
                "decision": final_decision,
                "gate": gate,
                "final": final_text,
                "direct_letters": parse_final_letters(cache["direct"]),
                "final_letters": parse_final_letters(final_text),
                "override_reason": override_reason,
                "preservation_reason": preservation_reason,
                "targeted_triggered": targeted_triggered,
                "verification": verification,
            }
        finally:
            self.config.thresholds = old_th
            self.config.v2 = old_v2
            self.config.version = old_ver
