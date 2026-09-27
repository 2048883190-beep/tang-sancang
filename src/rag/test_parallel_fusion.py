# -*- coding: utf-8 -*-
"""融合算子单测：κ 条件于本题药物、两侧保留、他药冲突不驱动提示。"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from rag.parallel_fusion import (
    assemble_with_stance_quota,
    evidence_delta,
    parallel_appraise_and_fuse,
    query_conditioned_conflict,
)


def _c(cid: str, text: str, score: float = 0.8) -> dict:
    return {
        "id": cid,
        "content": text,
        "score": score,
        "metadata": {"evidence_tier": "guideline"},
    }


def test_keep_both_query_drug():
    q = "2型糖尿病患者能否继续用二甲双胍"
    chunks = [
        _c("pos", "二甲双胍是2型糖尿病一线首选，推荐用于大多数患者。"),
        _c("neg", "eGFR<30 时二甲双胍禁忌，应停用。"),
        _c("other", "利拉鲁肽推荐用于肥胖患者，心衰时不推荐。"),
    ]
    fused, diag = parallel_appraise_and_fuse(q, chunks)
    alive = {c["id"] for c in fused if float(c["score"] or 0) > 0}
    assert "pos" in alive and "neg" in alive
    assert diag["query_both_sides"] is True
    assert diag["kappa_q"] > 0
    assert "二甲双胍" in diag["query_drugs"]
    print("[ok] keep both query drug", diag["kappa_q"])


def test_off_query_drug_war_does_not_set_kappa():
    q = "肥胖的2型糖尿病患者首选降糖药是二甲双胍还是生活方式"
    chunks = [
        _c("met", "二甲双胍是一线首选，推荐用于肥胖的2型糖尿病。"),
        _c("lira_pos", "利拉鲁肽推荐用于合并肥胖的患者。"),
        _c("lira_neg", "利拉鲁肽在该场景不推荐、应避免。"),
    ]
    info = query_conditioned_conflict(q, chunks)
    assert info["query_both_sides"] is False
    assert info["kappa_q"] == 0
    fused, diag = parallel_appraise_and_fuse(q, chunks)
    met_tau = next(c["channels"]["tau"] for c in fused if c["id"] == "met")
    lira_tau = next(c["channels"]["tau"] for c in fused if c["id"] == "lira_pos")
    assert met_tau > lira_tau
    assert diag["query_both_sides"] is False
    print("[ok] off-query drug war ignored")


def test_delta_speaks_when_query_drug_present():
    q = "二甲双胍用法"
    chunks = [_c("a", "二甲双胍推荐从小剂量起始。")]
    assert evidence_delta(q, chunks) >= 0.9
    empty = evidence_delta(q, [_c("b", "钩端螺旋体病的肾脏表现。")])
    assert empty < 0.2
    print("[ok] delta_q")


def test_stance_quota_keeps_buried_neg():
    q = "请说明二甲双胍的推荐地位以及哪些情况不能用"
    chunks = [_c(f"pos{i}", "二甲双胍是一线首选，推荐用于大多数2型糖尿病患者。", 0.9 - 0.01 * i) for i in range(8)]
    chunks.append(_c("neg", "eGFR<30 时二甲双胍禁忌，应停用。", 0.2))
    fused, _ = parallel_appraise_and_fuse(q, chunks)
    top5 = [c for c in fused if float(c.get("score") or 0) > 0][:5]
    assert all("禁忌" not in (c.get("content") or "") for c in top5)
    window, diag = assemble_with_stance_quota(q, fused, k=5)
    texts = "\n".join(c.get("content") or "" for c in window)
    assert "一线" in texts or "推荐" in texts
    assert "禁忌" in texts or "停用" in texts
    assert "POS" in diag["have_after"] and "NEG" in diag["have_after"]
    print("[ok] stance quota", diag)


def test_similarity_before_second_hit_and_conflict():
    """适用分主导时：相似度高的段应压过仅二次命中的低相似段。"""
    from rag.parallel_fusion import FusionLambdas

    q = "肥胖的2型糖尿病患者一线降糖药"
    chunks = [
        _c("close", "二甲双胍是2型糖尿病一线首选，推荐用于肥胖患者。", 0.95),
        _c("mid", "生活方式干预应贯穿始终。", 0.5),
        _c("twice_far", "二甲双胍可使用。", 0.2),
    ]
    fused, diag = parallel_appraise_and_fuse(
        q, chunks, rev_ids={"twice_far"}, lambdas=FusionLambdas(1.5, 1.0, 0.2)
    )
    ids = [c["id"] for c in fused if float(c.get("score") or 0) > 0]
    assert diag["lambdas"]["lam1"] == 1.5
    assert ids[0] == "close"
    assert ids.index("close") < ids.index("twice_far")
    print("[ok] similarity before second hit and conflict")


def test_question_stem_drops_options():
    from rag.domain_gate import question_stem

    full = "便秘病史2年，该患者便秘是由于 选项： A. 肠易激综合征 B. 糖尿病"
    assert question_stem(full) == "便秘病史2年，该患者便秘是由于"
    assert question_stem("二甲双胍怎么用") == "二甲双胍怎么用"
    print("[ok] question stem")


def test_stance_quota_skips_when_pool_one_sided():
    q = "二甲双胍是一线治疗吗"
    chunks = [_c(f"pos{i}", "二甲双胍是一线首选，推荐使用。", 0.8) for i in range(6)]
    window, diag = assemble_with_stance_quota(q, chunks, k=4)
    assert diag["quota_applied"] is False
    assert len(window) == 4
    print("[ok] quota skip one-sided")


if __name__ == "__main__":
    test_keep_both_query_drug()
    test_off_query_drug_war_does_not_set_kappa()
    test_delta_speaks_when_query_drug_present()
    test_stance_quota_keeps_buried_neg()
    test_similarity_before_second_hit_and_conflict()
    test_question_stem_drops_options()
    test_stance_quota_skips_when_pool_one_sided()
    print("ALL PASSED")
