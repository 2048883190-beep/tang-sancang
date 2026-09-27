# -*- coding: utf-8 -*-
"""域门控与分层规则单测（无 LLM / 无 chroma）。"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from rag.clinical_rules import framework_conflict
from rag.domain_gate import classify_domain, filter_main_table, is_multi_select, label_question


def test_ood_vs_inscope():
    d = classify_domain("腺垂体功能减退症最常见原因是")
    assert d.in_scope is False and d.domain == "endo_other"
    d2 = classify_domain("2型糖尿病合并慢性肾脏病如何选择降糖药 二甲双胍")
    assert d2.in_scope is True
    d3 = classify_domain("急性肾功能不全是钩端螺旋体病哪一型常见的死亡原因")
    assert d3.in_scope is False
    d4 = classify_domain("关于急性肾功能不全的处理，早期的治疗措施是")
    assert d4.in_scope is False
    d5 = classify_domain("2型糖尿病患者eGFR 25，口服二甲双胍是否停药")
    assert d5.in_scope is True
    d6 = classify_domain("甲亢患者心率增快的处理")
    assert d6.in_scope is False
    d7 = classify_domain("Fuchs角膜内皮营养不良如何治疗")
    assert d7.in_scope is False
    print("[ok] domain classify")


def test_multi():
    assert is_multi_select("BCD") is True
    assert is_multi_select("B") is False
    print("[ok] multi select")


def test_label_and_main_table():
    items = [
        {"id": "1", "question": "垂体肿瘤导致腺垂体功能减退", "gold": "B", "q_type": "fact"},
        {"id": "2", "question": "2型糖尿病降糖药二甲双胍用法", "gold": "A", "q_type": "decision"},
        {"id": "3", "question": "糖尿病酮症酸中毒胰岛素治疗", "gold": "BCD", "q_type": "safety"},
    ]
    labeled = [label_question(x) for x in items]
    main = filter_main_table(labeled)
    assert len(main) == 1 and main[0]["id"] == "2"
    print("[ok] main table filter", len(main))


def test_framework_conflict():
    fw = "严重肾功能不全时二甲双胍禁忌，不推荐使用。"
    chunk_bad = "二甲双胍是一线用药，常规用于大多数患者，首选治疗。"
    chunk_ok = "二甲双胍禁忌包括严重肾功能不全，需按 eGFR 调整。"
    bad, _ = framework_conflict(fw, chunk_bad)
    ok, _ = framework_conflict(fw, chunk_ok)
    assert bad is True
    assert ok is False
    print("[ok] framework conflict")


if __name__ == "__main__":
    test_ood_vs_inscope()
    test_multi()
    test_label_and_main_table()
    test_framework_conflict()
    print("ALL PASSED")
