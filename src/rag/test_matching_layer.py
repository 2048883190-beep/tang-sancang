"""
不依赖向量库的 Evidence-QC 门控单测（规则层）。
运行：py -3 -m src.rag.test_matching_layer
或：  py -3 src/rag/test_matching_layer.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from rag.clinical_rules import (
    clinical_conflict,
    infer_evidence_tier,
    pico_alignment_score,
    pico_from_text,
)
from rag.matching_layer import EvidenceQCMatcher


class _FakeCollection:
    """按预设 candidates 伪返回，跳过真实 Chroma。"""

    def __init__(self, candidates: list[dict]):
        self.candidates = candidates

    def query(self, query_embeddings, n_results, include):
        take = self.candidates[:n_results]
        return {
            "ids": [[c["id"] for c in take]],
            "documents": [[c["content"] for c in take]],
            "metadatas": [[c.get("metadata") or {} for c in take]],
            "distances": [[c.get("distance", 0.3) for c in take]],
        }


def _embed(_text: str):
    return [0.0] * 8


def test_evidence_tier_inference():
    assert infer_evidence_tier({"source": "guideline"}, "") == "guideline"
    assert infer_evidence_tier({"title": "盐酸二甲双胍片说明书"}, "") == "drug_label"
    assert infer_evidence_tier({"title": "内科学教案"}, "") == "textbook"
    print("[ok] evidence tier inference")


def test_contraindication_metformin_ckd():
    q = "eGFR<30 的老年患者能否继续用二甲双胍？"
    bad = "二甲双胍是2型糖尿病的首选药物，常规用于一般患者，肾功能正常者无需调整。"
    good = "eGFR<30 时二甲双胍禁用，应停用并改用其他方案。"
    assert clinical_conflict(q, bad)[0] is True
    assert clinical_conflict(q, good)[0] is False
    print("[ok] contraindication metformin/CKD")


def test_pico_drug_alignment():
    q = pico_from_text("二甲双胍联合 SGLT2 抑制剂如何选择？")
    good = pico_from_text("达格列净属于 SGLT2 抑制剂，可与二甲双胍联用。")
    bad = pico_from_text("糖尿病患者应注意饮食控制和运动。")
    s_good, _ = pico_alignment_score(q, good)
    s_bad, _ = pico_alignment_score(q, bad)
    assert s_good > s_bad
    print("[ok] pico alignment", round(s_good, 3), ">", round(s_bad, 3))


def test_evidence_qc_rerank_order():
    cands = [
        {
            "id": "a",
            "content": "糖尿病患者应注意饮食控制和运动，未涉及具体药物。",
            "metadata": {"source": "wiki", "title": "生活指导"},
            "distance": 0.10,  # 稠密分更高
        },
        {
            "id": "b",
            "content": "中国2型糖尿病防治指南指出，二甲双胍是一线用药；eGFR<30 禁用。",
            "metadata": {"source": "guideline", "title": "中国2型糖尿病防治指南"},
            "distance": 0.25,
        },
        {
            "id": "c",
            "content": "二甲双胍常规用于肾功能正常者，一般患者首选，无需调整。",
            "metadata": {"source": "web", "title": "网文"},
            "distance": 0.15,
        },
    ]
    m = EvidenceQCMatcher(collection=_FakeCollection(cands), embed=_embed)
    out = m.retrieve(
        "eGFR<30 患者二甲双胍如何处理？请依据指南。",
        k=2,
        role="forward",
    )
    ids = [c["id"] for c in out]
    # 不再用未标定系数把指南乘到相似度前面；名次按实测相似度。
    assert ids[0] == "a", ids
    by_id = {c["id"]: c for c in m.retrieve(
        "eGFR<30 患者二甲双胍如何处理？请依据指南。",
        k=3,
        role="forward",
    )}
    assert by_id["b"]["match_info"]["evidence_tier"] == "guideline"
    assert any("证据等级=guideline" in r for r in by_id["b"]["match_info"]["reasons"])
    assert by_id["c"]["match_info"]["contraindication_check"] == "conflict_flagged"
    assert "c" in by_id  # 冲突段只标记，不删除
    print("[ok] evidence_qc no fake weight", ids)


def main():
    test_evidence_tier_inference()
    test_contraindication_metformin_ckd()
    test_pico_drug_alignment()
    test_evidence_qc_rerank_order()
    print("ALL PASSED")


if __name__ == "__main__":
    main()
