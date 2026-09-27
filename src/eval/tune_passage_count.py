# -*- coding: utf-8 -*-
"""在 60 道验证题上只锁定一个量：重排后送进生成的段落数。

格子是 4、6、8、10。并列时取更少的段数。
不使用测试集的 685 题。温度固定为 0。
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from rag.hierarchical_rag import HierConfig  # noqa: E402
from src.eval.run_main_table_eval import (  # noqa: E402
    Runtime,
    _load_dotenv,
    _parse_gold,
    _parse_pred,
    label_question,
)

GRID = (4, 6, 8, 10)
OUT_DIR = ROOT / "reports" / "tune_passage_count"
LOCK_PATH = ROOT / "data" / "eval" / "locked_passage_count.json"


def _val_questions() -> list:
    data = json.loads((ROOT / "data" / "eval" / "expanded_questions.json").read_text(encoding="utf-8"))
    split = json.loads((ROOT / "data" / "eval" / "val_test_split.json").read_text(encoding="utf-8"))
    wanted = set(split["val_ids"])
    rows = [label_question(q) for q in (data.get("questions") or []) if q.get("id") in wanted]
    if len(rows) != len(wanted):
        raise SystemExit(f"验证题对不齐：文件 {len(rows)}，名单 {len(wanted)}")
    return rows


def _done(path: Path) -> dict:
    if not path.exists():
        return {}
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("id"):
            out[row["id"]] = row
    return out


def main() -> None:
    _load_dotenv()
    questions = _val_questions()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[tune] val n={len(questions)} grid={list(GRID)}", flush=True)
    rt = Runtime()
    rag = rt.hier(
        "tune_passage",
        HierConfig(
            hier_mode="off",
            use_parallel_fusion=False,
            matching_mode="hybrid_rrf_rerank",
            top_k=GRID[0],
            temperature=0.0,
            answer_mode="mcq",
        ),
    )
    scores = {}
    for k in GRID:
        rag.config.top_k = k
        path = OUT_DIR / f"k{k}.jsonl"
        done = _done(path)
        pending = [q for q in questions if q.get("id") not in done]
        print(f"[k={k}] done={len(done)} pending={len(pending)}", flush=True)
        with path.open("a", encoding="utf-8") as handle:
            for i, item in enumerate(pending, 1):
                gold = _parse_gold(item)
                try:
                    answer, stats = rag.query(item["question"])
                    err = None
                except Exception as exc:  # noqa: BLE001
                    answer, stats, err = "", {}, f"{type(exc).__name__}: {exc}"
                pred = _parse_pred(answer, force_single=len(gold) == 1)
                row = {
                    "id": item.get("id"),
                    "exact_match": bool(gold) and pred == gold and not err,
                    "pred_letters": sorted(pred),
                    "gold_letters": sorted(gold),
                    "top_k": k,
                    "error": err,
                    "mode": (stats or {}).get("mode"),
                }
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
                mark = "OK" if row["exact_match"] else "MISS"
                print(f"[k={k} {len(done)+i}/{len(questions)}] {item.get('id')} {mark}", flush=True)
        rows = list(_done(path).values())
        exact = sum(1 for r in rows if r.get("exact_match"))
        scores[k] = {"exact": exact, "n": len(rows), "exact_rate": round(exact / len(rows), 4) if rows else 0.0}
        print(f"[k={k}] {exact}/{len(rows)}", flush=True)

    best_k = max(scores, key=lambda k: (scores[k]["exact"], -k))
    payload = {
        "param": "top_k",
        "top_k": best_k,
        "grid": {str(k): scores[k] for k in GRID},
        "n_val": len(questions),
        "split": "data/eval/val_test_split.json",
        "temperature": 0,
        "retriever": "BM25 + BGE-M3, RRF k=60",
        "reranker": "BAAI/bge-reranker-v2-m3",
        "tie_break": "exact 相同则取更小的段数",
        "locked_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
    }
    LOCK_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[lock] top_k={best_k} -> {LOCK_PATH}", flush=True)


if __name__ == "__main__":
    main()
