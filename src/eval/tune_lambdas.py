# -*- coding: utf-8 -*-
"""在 60 道验证题上网格搜索 λ1≥λ2≥λ3，锁进 locked_lambdas.json。

每题只做一次：BM25+稠密召回 → 交叉编码器打适用分 → 二次检索。
各组 λ 只重排融合；上下文 id 集合相同时复用生成，避免 24× 烧 API。
段数沿用 locked_passage_count（默认 6）。温度 0。
不使用测试集 685 题。
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from rag.config import LOCKED_LAMBDAS_PATH, locked_passage_count  # noqa: E402
from rag.hierarchical_rag import HierConfig, HierarchicalRAG  # noqa: E402
from rag.hybrid_retrieve import RERANK_POOL  # noqa: E402
from rag.matching_layer import DenseMatcher  # noqa: E402
from rag.parallel_fusion import (  # noqa: E402
    FusionLambdas,
    assemble_with_stance_quota,
    iter_lambda_grid,
    parallel_appraise_and_fuse,
)
from src.eval.run_main_table_eval import (  # noqa: E402
    Runtime,
    _load_dotenv,
    _parse_gold,
    _parse_pred,
    label_question,
)

OUT_DIR = ROOT / "reports" / "lambda_tune"


def _val_questions() -> list:
    data = json.loads((ROOT / "data" / "eval" / "expanded_questions.json").read_text(encoding="utf-8"))
    split = json.loads((ROOT / "data" / "eval" / "val_test_split.json").read_text(encoding="utf-8"))
    wanted = set(split["val_ids"])
    rows = [label_question(q) for q in (data.get("questions") or []) if q.get("id") in wanted]
    if len(rows) != len(wanted):
        raise SystemExit(f"验证题对不齐：文件 {len(rows)}，名单 {len(wanted)}")
    return rows


def _prepare_pool(rag: HierarchicalRAG, question: str) -> Tuple[List[dict], set]:
    pool = rag._hybrid_retriever().retrieve(question, pool=RERANK_POOL)
    for c in pool:
        info = c.setdefault("match_info", {})
        info["retrieve_score"] = float(c.get("score") or 0.0)
    ranked = rag._cross_encoder().rerank(question, pool)
    ranked = rag._annotate_tiers(ranked)
    rev_ids = set()
    if ranked:
        seed = (ranked[0].get("content") or "")[:400].strip()
        if len(seed) >= 20:
            rev = DenseMatcher(rag.collection, rag.embed).retrieve(
                seed, k=max(rag.config.top_k * 2, 20), role="reverse"
            )
            rev_ids = {str(c.get("id") or "") for c in rev}
    return ranked, rev_ids


def main() -> None:
    _load_dotenv()
    questions = _val_questions()
    top_k = locked_passage_count()
    grid = iter_lambda_grid()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[tune] val n={len(questions)} grid={len(grid)} top_k={top_k}", flush=True)

    rt = Runtime()
    rag = rt.hier(
        "tune_lambda",
        HierConfig(
            hier_mode="off",
            use_parallel_fusion=True,
            matching_mode="hybrid_rrf_rerank",
            top_k=top_k,
            temperature=0.0,
            load_locked_lambdas=False,
            fusion_lam1=1.5,
            fusion_lam2=0.5,
            fusion_lam3=0.2,
            answer_mode="mcq",
        ),
    )

    # qid -> {pool, rev_ids, gen_cache: frozenset(ids)->(answer, exact)}
    caches: Dict[str, dict] = {}
    for i, item in enumerate(questions, 1):
        qid = item.get("id")
        print(f"[prep {i}/{len(questions)}] {qid}", flush=True)
        pool, rev_ids = _prepare_pool(rag, item["question"])
        caches[qid] = {
            "item": item,
            "pool": pool,
            "rev_ids": rev_ids,
            "gold": _parse_gold(item),
            "gen": {},
        }

    scores: Dict[str, dict] = {}
    for lam in grid:
        key = f"{lam.lam1}_{lam.lam2}_{lam.lam3}"
        exact = 0
        n = 0
        path = OUT_DIR / f"lam_{key}.jsonl"
        rows = []
        for qid, pack in caches.items():
            item = pack["item"]
            gold = pack["gold"]
            fused, _ = parallel_appraise_and_fuse(
                item["question"],
                pack["pool"],
                rev_ids=pack["rev_ids"],
                lambdas=lam,
            )
            window, _ = assemble_with_stance_quota(item["question"], fused, k=top_k)
            idset = frozenset(str(c.get("id") or "") for c in window)
            if idset in pack["gen"]:
                answer, ok = pack["gen"][idset]
            else:
                if window:
                    answer = rag._once_generate(
                        item["question"], window, keep_conflict=False
                    )
                else:
                    answer = rag._direct_answer(item["question"])
                pred = _parse_pred(answer, force_single=len(gold) == 1)
                ok = bool(gold) and pred == gold
                pack["gen"][idset] = (answer, ok)
            exact += int(ok)
            n += 1
            rows.append(
                {
                    "id": qid,
                    "exact_match": ok,
                    "lam1": lam.lam1,
                    "lam2": lam.lam2,
                    "lam3": lam.lam3,
                }
            )
        path.write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
            encoding="utf-8",
        )
        rate = exact / n if n else 0.0
        scores[key] = {
            "lam1": lam.lam1,
            "lam2": lam.lam2,
            "lam3": lam.lam3,
            "exact": exact,
            "n": n,
            "exact_rate": round(rate, 4),
        }
        print(f"[λ {key}] {exact}/{n} = {100*rate:.1f}%", flush=True)

    best_key = max(
        scores,
        key=lambda k: (
            scores[k]["exact"],
            -scores[k]["lam1"],
            -scores[k]["lam2"],
            -scores[k]["lam3"],
        ),
    )
    best = scores[best_key]
    payload = {
        "lam1": best["lam1"],
        "lam2": best["lam2"],
        "lam3": best["lam3"],
        "val_exact": best["exact"],
        "val_n": best["n"],
        "val_exact_rate": best["exact_rate"],
        "grid": scores,
        "top_k": top_k,
        "temperature": 0,
        "retriever": "BM25 + BGE-M3, RRF k=60",
        "reranker": "BAAI/bge-reranker-v2-m3",
        "formula": "s=m_a m_u m_k · a^λ1 · u^λ2 · k^λ3",
        "constraint": "λ1≥λ2≥λ3>0",
        "split": "data/eval/val_test_split.json",
        "locked_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
    }
    LOCKED_LAMBDAS_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"[lock] λ=({best['lam1']},{best['lam2']},{best['lam3']}) "
        f"{best['exact']}/{best['n']} -> {LOCKED_LAMBDAS_PATH}",
        flush=True,
    )


if __name__ == "__main__":
    main()
