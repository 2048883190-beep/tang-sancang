# -*- coding: utf-8 -*-
"""名次融合与 BM25 的单元测试。不加载向量模型，不调用接口。"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from rag.hybrid_retrieve import BM25Index, rrf_fuse


def test_bm25_puts_drug_name_first():
    index = BM25Index(
        [
            {"id": "a", "content": "二甲双胍用于肾功能正常的患者", "metadata": {}},
            {"id": "b", "content": "胰岛素用于酮症酸中毒", "metadata": {}},
        ]
    )
    hit = index.search("二甲双胍 肾功能", k=2)
    assert hit[0]["id"] == "a"


def test_rrf_rewards_agreement_without_mixing_raw_scores():
    dense = [
        {"id": "only_dense", "content": "甲", "score": 0.99},
        {"id": "both", "content": "乙", "score": 0.10},
    ]
    lexical = [
        {"id": "both", "content": "乙", "score": 3.0},
    ]
    fused = rrf_fuse(dense, lexical, rrf_k=60, limit=2)
    assert fused[0]["id"] == "both"
    assert fused[0]["match_info"]["in_both"] is True
    assert fused[1]["match_info"]["in_both"] is False


if __name__ == "__main__":
    test_bm25_puts_drug_name_first()
    test_rrf_rewards_agreement_without_mixing_raw_scores()
    print("ok")
