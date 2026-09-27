# -*- coding: utf-8 -*-
"""
主表评测。默认只跑本系统（fusion_once）。
直接生成与普通稠密检索的结果来自已完成的对照，用 --baselines-dir 读入。

  py -3 -m src.eval.run_main_table_eval --modes fusion_once --split test
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from rag.config import (
    CHROMA_DIR,
    COLLECTION_NAME,
    EMBEDDING_MODEL_ID,
    RAG_ROOT,
    REPORTS_DIR,
    locked_passage_count,
)
from rag.domain_gate import filter_main_table, label_question
from rag.hierarchical_rag import HierConfig, HierarchicalRAG
from rag.matching_layer import DenseMatcher

ALL_MODES = ["direct", "naive_rag", "fusion_once"]

MODE_LABELS = {
    "direct": "直接生成（不查库）",
    "naive_rag": "普通稠密检索",
    "fusion_once": "本系统（BM25+稠密 + 三路λ乘积 + 一次生成）",
}


def _load_dotenv() -> None:
    env_path = ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())


def _parse_gold(item: dict) -> set:
    gold = str(item.get("gold") or "")
    if gold:
        return set(gold.upper())
    return set(re.findall(r"[A-E]", str(item.get("ground_truth") or "").upper()))


def _parse_pred(answer: str, force_single: bool) -> set:
    text = answer or ""
    m = re.search(r"最终选项[:：]\s*([A-Ea-e]+)", text)
    if m:
        letters = set(ch.upper() for ch in m.group(1) if ch.upper() in "ABCDE")
    else:
        letters = set(re.findall(r"[A-E]", text.upper()))
    if force_single and letters:
        all_letters = re.findall(r"[A-E]", text.upper())
        letters = {all_letters[-1]} if all_letters else letters
    return letters


MCQ_TAIL = (
    "\n\n若题目含选项（A/B/C/D/E），请先简要推理，"
    "并在文末单独一行写：最终选项：X（多选则连写字母，如 BCD）。"
    "\n本回答仅供参考，不替代专业诊疗。"
)


class Runtime:
    def __init__(self):
        import chromadb
        from FlagEmbedding import BGEM3FlagModel
        from openai import OpenAI

        local = RAG_ROOT / "models" / "bge-m3"
        model_id = str(local) if local.exists() else EMBEDDING_MODEL_ID
        print(f"[init] BGE-M3 {model_id}", flush=True)
        self.model = BGEM3FlagModel(model_id, use_fp16=True)
        self.chroma = chromadb.PersistentClient(path=str(CHROMA_DIR))
        self.collection = self.chroma.get_collection(COLLECTION_NAME)
        self.llm = OpenAI(
            api_key=os.environ["DEEPSEEK_API_KEY"],
            base_url=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        )
        self.llm_model = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")
        self._hier_cache: Dict[str, HierarchicalRAG] = {}

    def embed(self, text: str):
        return self.model.encode([text], max_length=512)["dense_vecs"][0].tolist()

    def chat(self, prompt: str, max_tokens: int = 1024) -> str:
        resp = self.llm.chat.completions.create(
            model=self.llm_model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0,
        )
        return (resp.choices[0].message.content or "").strip()

    def hier(self, key: str, cfg: HierConfig) -> HierarchicalRAG:
        if key not in self._hier_cache:
            self._hier_cache[key] = HierarchicalRAG(
                self.chroma, self.llm, self.embed, config=cfg
            )
        return self._hier_cache[key]


def run_naive(rt: Runtime, question: str) -> Tuple[str, dict]:
    matcher = DenseMatcher(rt.collection, rt.embed)
    chunks = matcher.retrieve(question, k=10, role="forward")
    ctx = "\n\n---\n\n".join(f"[来源{i+1}] {c['content']}" for i, c in enumerate(chunks))
    prompt = (
        "你是专业医疗知识助手。请基于参考资料回答问题。\n"
        f"用户问题：{question}\n\n参考资料：\n{ctx}{MCQ_TAIL}"
    )
    return rt.chat(prompt), {"mode": "naive_rag", "forward_count": len(chunks)}


def run_direct(rt: Runtime, question: str) -> Tuple[str, dict]:
    prompt = (
        "你是专业医疗知识助手。请回答下列问题。\n"
        f"用户问题：{question}{MCQ_TAIL}"
    )
    return rt.chat(prompt), {"mode": "direct", "retrieval": "none"}


def run_one(rt: Runtime, mode: str, item: dict) -> Tuple[str, dict]:
    q = item["question"]
    q_type = item.get("q_type")
    if mode == "direct":
        return run_direct(rt, q)
    if mode == "naive_rag":
        return run_naive(rt, q)
    if mode == "fusion_once":
        lock = RAG_ROOT / "data" / "eval" / "locked_lambdas.json"
        if not lock.exists():
            raise SystemExit("λ 尚未在验证集上锁定。先运行 py -3 -m src.eval.tune_lambdas")
        from rag.config import load_locked_lambdas

        lam = load_locked_lambdas()
        cfg = HierConfig(
            hier_mode="off",
            use_parallel_fusion=True,
            matching_mode="hybrid_rrf_rerank",
            top_k=locked_passage_count(),
            temperature=0.0,
            fusion_lam1=lam.lam1,
            fusion_lam2=lam.lam2,
            fusion_lam3=lam.lam3,
            load_locked_lambdas=False,
        )
        return rt.hier("fusion_once", cfg).query(q, q_type=q_type)
    raise ValueError(mode)


def load_done(path: Path) -> set:
    if not path.exists():
        return set()
    done = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("id"):
            done.add(row["id"])
    return done


def read_jsonl(path: Path) -> List[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def summarize(rows: List[dict]) -> dict:
    ok = [r for r in rows if not r.get("error")]
    exact = sum(1 for r in ok if r["exact_match"])
    by_type = defaultdict(list)
    for r in ok:
        by_type[str(r.get("q_type") or "?")].append(r)
    type_stats = {
        t: {
            "n": len(rs),
            "exact": sum(1 for r in rs if r["exact_match"]),
            "exact_rate": round(sum(1 for r in rs if r["exact_match"]) / len(rs), 4),
        }
        for t, rs in sorted(by_type.items())
    }
    return {
        "n": len(ok),
        "exact": exact,
        "exact_rate": round(exact / len(ok), 4) if ok else 0.0,
        "by_type": type_stats,
    }


def write_compare(out_dir: Path, results: dict, n: int) -> Path:
    lines = [
        "# 主表评测（in_scope + 单选）",
        "",
        f"- 时间：{datetime.now(timezone.utc).astimezone().isoformat(timespec='seconds')}",
        f"- 主表题量：**{n}**",
        "- 终局对照：Direct LLM；工程对照：Naive RAG",
        "",
        "| 方法 | Exact | 准确率 | Δ vs Direct (pp) | Δ vs Naive (pp) |",
        "|------|-------|--------|------------------|-----------------|",
    ]
    base_d = results.get("direct", {}).get("summary", {}).get("exact_rate")
    base_n = results.get("naive_rag", {}).get("summary", {}).get("exact_rate")
    for mode in ALL_MODES:
        if mode not in results:
            continue
        s = results[mode]["summary"]
        dd = "—" if base_d is None else f"{100 * (s['exact_rate'] - base_d):+.2f}"
        dn = "—" if base_n is None else f"{100 * (s['exact_rate'] - base_n):+.2f}"
        if mode == "direct":
            dd = "0.00"
        if mode == "naive_rag":
            dn = "0.00"
        lines.append(
            f"| {MODE_LABELS[mode]} | {s['exact']}/{s['n']} | **{100*s['exact_rate']:.1f}%** | {dd} | {dn} |"
        )
    path = out_dir / "compare_main_table.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", default=str(ROOT / "data" / "eval" / "expanded_questions.json"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--modes", default="fusion_once")
    ap.add_argument("--resume-dir", default="")
    ap.add_argument(
        "--baselines-dir",
        default="",
        help="读入已有 direct/naive jsonl，只跑新方法时用来对照",
    )
    ap.add_argument("--full305", action="store_true", help="不筛主表，跑全量（附表）")
    ap.add_argument(
        "--split",
        default="test",
        choices=["test", "val", "all"],
        help="test=锁参后终报（默认）；val=验证集；all=主表全部",
    )
    args = ap.parse_args()

    _load_dotenv()
    if not os.environ.get("DEEPSEEK_API_KEY"):
        raise SystemExit("请设置 DEEPSEEK_API_KEY")

    data = json.loads(Path(args.questions).read_text(encoding="utf-8"))
    qs = [label_question(q) for q in (data.get("questions") or [])]

    split_path = ROOT / "data" / "eval" / "val_test_split.json"
    if args.split != "all" and split_path.exists():
        split = json.loads(split_path.read_text(encoding="utf-8"))
        idset = set(split.get("val_ids" if args.split == "val" else "test_ids") or [])
        qs = [q for q in qs if q.get("id") in idset]
        print(
            f"[split] {args.split} n={len(qs)} from {split_path.name} (locked ids)",
            flush=True,
        )
    elif not args.full305:
        qs = filter_main_table(qs)
    elif args.split != "all":
        print(f"[warn] 无 {split_path}，回退主表全部", flush=True)

    if args.limit > 0:
        qs = qs[: args.limit]

    out_dir = (
        Path(args.resume_dir)
        if args.resume_dir
        else REPORTS_DIR
        / f"main_table_{datetime.now(timezone.utc).astimezone().strftime('%Y%m%d_%H%M%S')}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[out] {out_dir} n={len(qs)}", flush=True)

    rt = Runtime()
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    results = {}
    for mode in modes:
        jsonl = out_dir / f"{mode}.jsonl"
        done = load_done(jsonl)
        pending = [q for q in qs if q.get("id") not in done]
        print(f"[{mode}] done={len(done)} pending={len(pending)}", flush=True)
        with jsonl.open("a", encoding="utf-8") as f:
            for i, item in enumerate(pending, 1):
                gold = _parse_gold(item)
                t0 = time.time()
                err = None
                answer, stats = "", {}
                try:
                    answer, stats = run_one(rt, mode, item)
                except Exception as exc:  # noqa: BLE001
                    err = f"{type(exc).__name__}: {exc}"
                pred = _parse_pred(answer, force_single=len(gold) == 1)
                row = {
                    "id": item.get("id"),
                    "q_type": item.get("q_type"),
                    "domain": item.get("domain"),
                    "scope": item.get("scope"),
                    "exact_match": bool(gold) and pred == gold,
                    "pred_letters": sorted(pred),
                    "gold_letters": sorted(gold),
                    "latency_ms": round((time.time() - t0) * 1000, 1),
                    "pipe_mode": (stats or {}).get("mode"),
                    "confidence": (stats or {}).get("confidence"),
                    "matching_mode": mode,
                    "error": err,
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()
                st = "ERR" if err else ("OK" if row["exact_match"] else "MISS")
                print(f"[{mode} {len(done)+i}/{len(qs)}] {item.get('id')} {st}", flush=True)

        rows = read_jsonl(jsonl)
        # 只保留本题集 id
        idset = {q["id"] for q in qs}
        rows = [r for r in rows if r.get("id") in idset]
        summary = summarize(rows)
        results[mode] = {"summary": summary, "jsonl": str(jsonl)}
        (out_dir / f"{mode}_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"[{mode}] {summary['exact']}/{summary['n']} = {100*summary['exact_rate']:.1f}%", flush=True)

    if args.baselines_dir:
        bdir = Path(args.baselines_dir)
        idset = {q["id"] for q in qs}
        for mode in ("direct", "naive_rag"):
            if mode in results:
                continue
            src = bdir / f"{mode}.jsonl"
            if not src.exists():
                print(f"[baseline skip] {src} missing", flush=True)
                continue
            rows = [r for r in read_jsonl(src) if r.get("id") in idset]
            summary = summarize(rows)
            results[mode] = {"summary": summary, "jsonl": str(src), "imported": True}
            print(
                f"[baseline {mode}] {summary['exact']}/{summary['n']} = {100*summary['exact_rate']:.1f}%",
                flush=True,
            )

    path = write_compare(out_dir, results, len(qs))
    print("WROTE", path, flush=True)


if __name__ == "__main__":
    main()
