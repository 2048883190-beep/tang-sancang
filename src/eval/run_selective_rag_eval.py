# -*- coding: utf-8 -*-
"""Selective RAG 评测与验证集锁参（v1 / v2-A / v2-B）。

  # v1 验证集网格锁阈值
  py -3 -m src.eval.run_selective_rag_eval --split val --tune

  # v2 validation：先准备 cache，再网格锁参（禁止 test 调参）
  py -3 -m src.eval.run_selective_rag_eval --split val --tune-v2

  # 用锁定配置跑 val / test
  py -3 -m src.eval.run_selective_rag_eval --split val --version v2a
  py -3 -m src.eval.run_selective_rag_eval --split test --version v2b

禁止用测试集调参。温度固定 0。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from rag.config import (  # noqa: E402
    EVAL_DIR,
    REPORTS_DIR,
)
from rag.domain_gate import label_question  # noqa: E402
from rag.matching_layer import DenseMatcher  # noqa: E402
from rag.selective_rag import (  # noqa: E402
    SelectiveConfig,
    SelectiveRAG,
    SelectiveThresholds,
    SelectiveV2Params,
    parse_final_letters,
)
from src.eval.run_main_table_eval import (  # noqa: E402
    Runtime,
    _load_dotenv,
    _parse_gold,
    _parse_pred,
)

LOCK_PATH = EVAL_DIR / "locked_selective_rag.json"
LOCK_V2_PATH = EVAL_DIR / "locked_selective_rag_v2.json"
ABLATION_DIR = REPORTS_DIR / "selective_rag_ablation"
V2_DIR = REPORTS_DIR / "selective_rag_v2"


def _questions(split: str) -> list:
    data = json.loads((EVAL_DIR / "expanded_questions.json").read_text(encoding="utf-8"))
    split_data = json.loads((EVAL_DIR / "val_test_split.json").read_text(encoding="utf-8"))
    key = "val_ids" if split == "val" else "test_ids"
    wanted = set(split_data[key])
    rows = [label_question(q) for q in (data.get("questions") or []) if q.get("id") in wanted]
    if len(rows) != len(wanted):
        raise SystemExit(f"split {split} 对不齐：{len(rows)} vs {len(wanted)}")
    return rows


def load_locked_thresholds() -> SelectiveThresholds:
    if not LOCK_PATH.exists():
        raise SystemExit(f"缺少 {LOCK_PATH}，先 --split val --tune")
    data = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    th = data.get("thresholds") or {}
    allowed = set(SelectiveThresholds().as_dict())
    return SelectiveThresholds(**{k: float(th[k]) for k in allowed if k in th})


def load_locked_v2() -> Tuple[SelectiveThresholds, SelectiveV2Params, str]:
    if not LOCK_V2_PATH.exists():
        raise SystemExit(f"缺少 {LOCK_V2_PATH}，先 --split val --tune-v2")
    data = json.loads(LOCK_V2_PATH.read_text(encoding="utf-8"))
    th_raw = data.get("thresholds") or {}
    allowed_th = set(SelectiveThresholds().as_dict())
    th = SelectiveThresholds(**{k: float(th_raw[k]) for k in allowed_th if k in th_raw})
    v2_raw = data.get("v2_params") or {}
    base = SelectiveV2Params()
    kwargs = {}
    for k, default in base.as_dict().items():
        if k not in v2_raw:
            continue
        if isinstance(default, bool):
            kwargs[k] = bool(v2_raw[k])
        elif isinstance(default, (int, float)) and not isinstance(default, bool):
            kwargs[k] = type(default)(v2_raw[k])
        else:
            kwargs[k] = v2_raw[k]
    v2 = SelectiveV2Params(**{**base.as_dict(), **kwargs})
    version = str(data.get("version") or ("v2b" if v2.enable_targeted else "v2a"))
    return th, v2, version


def _metrics(rows: List[dict]) -> dict:
    n = len(rows)
    exact = sum(1 for r in rows if r.get("exact_match"))
    d_ok = [r for r in rows if r.get("direct_correct")]
    d_bad = [r for r in rows if r.get("direct_correct") is False]
    harmful = sum(1 for r in d_ok if not r.get("exact_match"))
    helpful = sum(1 for r in d_bad if r.get("exact_match"))
    decisions = Counter(str(r.get("decision") or "") for r in rows)
    use_n = sum(1 for r in rows if str(r.get("decision") or "").startswith("USE_"))
    keep = decisions.get("KEEP_DIRECT", 0)
    targeted = decisions.get("TARGETED_RAG", 0) + decisions.get("USE_TARGETED_RAG", 0)
    override_ok = sum(
        1
        for r in rows
        if str(r.get("decision") or "").startswith("USE_") and r.get("exact_match")
    )
    override_bad = sum(
        1
        for r in rows
        if str(r.get("decision") or "").startswith("USE_") and not r.get("exact_match")
    )
    flip = sum(1 for r in rows if r.get("final_letters") != r.get("direct_letters"))
    flip_good = sum(
        1
        for r in rows
        if r.get("final_letters") != r.get("direct_letters")
        and r.get("exact_match")
        and not r.get("direct_correct")
    )
    flip_bad = sum(
        1
        for r in rows
        if r.get("final_letters") != r.get("direct_letters")
        and (not r.get("exact_match"))
        and r.get("direct_correct")
    )
    return {
        "n": n,
        "exact": exact,
        "exact_rate": round(exact / n, 4) if n else 0.0,
        "keep_direct": keep,
        "use_rag": decisions.get("USE_RAG", 0),
        "use_targeted_rag": decisions.get("USE_TARGETED_RAG", 0),
        "targeted_rag_trigger": decisions.get("TARGETED_RAG", 0),
        "use_evidence": use_n,
        "targeted_n": targeted,
        "override_success": override_ok,
        "override_fail": override_bad,
        "answer_flips": flip,
        "flip_error_to_correct": flip_good,
        "flip_correct_to_error": flip_bad,
        "harmful_retrieval_rate": round(harmful / len(d_ok), 4) if d_ok else 0.0,
        "helpful_retrieval_rate": round(helpful / len(d_bad), 4) if d_bad else 0.0,
        "harmful_n": harmful,
        "helpful_n": helpful,
        "direct_ok_n": len(d_ok),
        "direct_bad_n": len(d_bad),
        "decision_counts": dict(decisions),
    }


def _score_key(m: dict) -> tuple:
    # exact ↓ harmful ↑ helpful
    return (m["exact"], -m["harmful_retrieval_rate"], m["helpful_retrieval_rate"])


def _make_selective(
    rt: Runtime,
    th: SelectiveThresholds,
    *,
    version: str = "v1",
    v2: Optional[SelectiveV2Params] = None,
) -> SelectiveRAG:
    def gen(prompt: str, max_tokens: int = 1024, temperature: float = 0.0, **_kw) -> str:
        return rt.chat(prompt, max_tokens=max_tokens)

    dense = DenseMatcher(rt.collection, rt.embed)
    cfg = SelectiveConfig(
        top_k=10,
        temperature=0.0,
        thresholds=th,
        v2=v2 or SelectiveV2Params(),
        version=version,
    )
    return SelectiveRAG(dense=dense, generate=gen, config=cfg)


def tune_val(rt: Runtime) -> dict:
    qs = _questions("val")
    out_dir = ABLATION_DIR / f"tune_{datetime.now(timezone.utc).astimezone().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[tune] val n={len(qs)} out={out_dir}", flush=True)
    sel = _make_selective(rt, SelectiveThresholds(), version="v1")
    caches = {}
    for i, item in enumerate(qs, 1):
        qid = item["id"]
        print(f"[prep {i}/{len(qs)}] {qid}", flush=True)
        caches[qid] = {
            "item": item,
            "gold": _parse_gold(item),
            "cache": sel.prepare_caches(item["question"]),
        }
        slim = {
            "id": qid,
            "direct": caches[qid]["cache"]["direct"],
            "high_risk": caches[qid]["cache"]["high_risk"],
            "assessment": caches[qid]["cache"]["assessment"],
            "stance": caches[qid]["cache"]["stance"],
            "rag_answer": caches[qid]["cache"]["rag_answer"],
            "retrieved_ids": [c.get("id") for c in caches[qid]["cache"]["chunks"]],
        }
        (out_dir / "caches.jsonl").open("a", encoding="utf-8").write(
            json.dumps(slim, ensure_ascii=False) + "\n"
        )

    grid_r = (0.50, 0.60, 0.70, 0.80)
    grid_f = (0.50, 0.60, 0.70, 0.80)
    results = []
    best = None
    for r_min, f_min in product(grid_r, grid_f):
        th = SelectiveThresholds(
            relevance_min=r_min,
            sufficiency_min=f_min,
            risk_relevance_min=min(0.95, r_min + 0.10),
            risk_sufficiency_min=min(0.95, f_min + 0.10),
        )
        rows = []
        for qid, pack in caches.items():
            applied = sel.apply_thresholds(pack["cache"], th)
            gold = pack["gold"]
            direct_letters = set(applied["direct_letters"])
            final_letters = set(applied["final_letters"])
            direct_correct = bool(gold) and direct_letters == gold
            exact = bool(gold) and final_letters == gold
            rows.append(
                {
                    "id": qid,
                    "exact_match": exact,
                    "direct_correct": direct_correct,
                    "decision": applied["decision"],
                    "direct_letters": sorted(direct_letters),
                    "final_letters": sorted(final_letters),
                    "gold_letters": sorted(gold),
                }
            )
        m = _metrics(rows)
        rec = {"thresholds": th.as_dict(), "metrics": m}
        results.append(rec)
        print(
            f"[grid r={r_min} f={f_min}] exact={m['exact']}/{m['n']} "
            f"use={m['use_evidence']} harm={m['harmful_retrieval_rate']} help={m['helpful_retrieval_rate']}",
            flush=True,
        )
        if best is None or _score_key(m) > best[0]:
            best = (_score_key(m), th, m)

    assert best is not None
    _, th_best, m_best = best
    payload = {
        "method": "Selective-RAG",
        "validation_size": len(qs),
        "thresholds": th_best.as_dict(),
        "val_metrics": m_best,
        "grid_results": results,
        "temperature": 0,
        "retriever": "DenseMatcher top_k=10 (same as Naive)",
        "locked": True,
        "locked_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "protocol": "Exact primary; then lower harmful rate; then higher helpful rate. Test forbidden for tuning.",
    }
    LOCK_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "tune_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[lock] {LOCK_PATH} exact={m_best['exact']}/{m_best['n']}", flush=True)
    return payload


def _eval_rows_from_caches(
    sel: SelectiveRAG,
    caches: dict,
    th: SelectiveThresholds,
    v2: SelectiveV2Params,
) -> List[dict]:
    rows = []
    for qid, pack in caches.items():
        applied = sel.apply_thresholds(pack["cache"], th, v2_params=v2)
        gold = pack["gold"]
        direct_letters = set(applied["direct_letters"])
        final_letters = set(applied["final_letters"])
        direct_correct = bool(gold) and direct_letters == gold
        exact = bool(gold) and final_letters == gold
        ver = applied.get("verification") or pack["cache"].get("verification") or {}
        rows.append(
            {
                "id": qid,
                "exact_match": exact,
                "direct_correct": direct_correct,
                "decision": applied["decision"],
                "direct_letters": sorted(direct_letters),
                "final_letters": sorted(final_letters),
                "gold_letters": sorted(gold),
                "override_reason": applied.get("override_reason"),
                "preservation_reason": applied.get("preservation_reason"),
                "targeted_triggered": applied.get("targeted_triggered"),
                "best_option": ver.get("best_supported_option"),
                "option_margin": ver.get("margin"),
                "option_scores": ver.get("option_scores"),
                "high_risk": pack["cache"].get("high_risk"),
            }
        )
    return rows


def tune_val_v2(rt: Runtime, reuse_cache: str = "") -> dict:
    """validation-only：比较 v1 / v2-A / v2-B，锁 v2 参数。不碰 test。"""
    import pickle

    qs = _questions("val")
    th_v1 = load_locked_thresholds()
    stamp = datetime.now(timezone.utc).astimezone().strftime("%Y%m%d_%H%M%S")
    out_root = V2_DIR / f"validation_{stamp}"
    out_root.mkdir(parents=True, exist_ok=True)
    print(f"[tune-v2] val n={len(qs)} out={out_root}", flush=True)
    print("[tune-v2] question thresholds FROZEN from locked_selective_rag.json", flush=True)
    print(json.dumps(th_v1.as_dict(), ensure_ascii=False), flush=True)

    caches = {}
    cache_path = out_root / "caches.jsonl"
    pickle_path = out_root / "caches.pkl"

    if reuse_cache:
        src = Path(reuse_cache)
        if src.is_dir():
            src = src / "caches.pkl"
        print(f"[tune-v2] reuse caches from {src}", flush=True)
        raw = pickle.loads(src.read_bytes())
        for qid, pack in raw.items():
            # 用当前 scoring 重算 option_verify（不重新检索 / Direct / RAG）
            caches[qid] = pack
        sel_prep = _make_selective(
            rt,
            th_v1,
            version="v2b",
            v2=SelectiveV2Params(enable_option_verify=True, enable_targeted=True),
        )
        for i, (qid, pack) in enumerate(caches.items(), 1):
            cache = pack["cache"]
            print(f"[reverify {i}/{len(caches)}] {qid}", flush=True)
            cache["verification"] = sel_prep._run_option_verify(
                cache["question"],
                cache.get("options") or {},
                cache.get("chunks") or [],
                cache["assessment"],
                cache.get("direct_letter"),
            )
            if cache.get("targeted"):
                t = cache["targeted"]
                t["verification"] = sel_prep._run_option_verify(
                    cache["question"],
                    cache.get("options") or {},
                    t.get("chunks") or [],
                    t["assessment"],
                    cache.get("direct_letter"),
                )
    else:
        sel_prep = _make_selective(
            rt,
            th_v1,
            version="v2b",
            v2=SelectiveV2Params(enable_option_verify=True, enable_targeted=True),
        )
        if cache_path.exists():
            cache_path.unlink()
        for i, item in enumerate(qs, 1):
            qid = item["id"]
            print(f"[prep {i}/{len(qs)}] {qid}", flush=True)
            cache = sel_prep.prepare_caches(item["question"])
            caches[qid] = {"item": item, "gold": _parse_gold(item), "cache": cache}
            slim = {
                "id": qid,
                "direct": cache["direct"],
                "direct_letter": cache.get("direct_letter"),
                "high_risk": cache["high_risk"],
                "assessment": cache["assessment"],
                "stance": cache["stance"],
                "verification": {
                    k: cache["verification"].get(k)
                    for k in (
                        "option_scores",
                        "option_stances",
                        "best_supported_option",
                        "second_supported_option",
                        "margin",
                        "best_score",
                        "second_score",
                        "has_strong_support",
                        "has_strong_contradiction",
                    )
                },
                "rag_answer": cache["rag_answer"],
                "retrieved_ids": [c.get("id") for c in cache["chunks"]],
                "has_targeted": bool(cache.get("targeted")),
            }
            cache_path.open("a", encoding="utf-8").write(
                json.dumps(slim, ensure_ascii=False) + "\n"
            )
        pickle_path.write_bytes(pickle.dumps(caches))
        print(f"[tune-v2] wrote {pickle_path}", flush=True)

    # 始终落盘当前 caches 便于复用
    if not pickle_path.exists():
        pickle_path.write_bytes(pickle.dumps(caches))

    sel = _make_selective(rt, th_v1, version="v2a")

    # --- baseline v1 on same caches ---
    v1_params = SelectiveV2Params(enable_option_verify=False, enable_targeted=False)
    rows_v1 = _eval_rows_from_caches(sel, caches, th_v1, v1_params)
    m_v1 = _metrics(rows_v1)
    (out_root / "validation_v1_metrics.json").write_text(
        json.dumps({"metrics": m_v1, "thresholds": th_v1.as_dict()}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        f"[v1] exact={m_v1['exact']}/{m_v1['n']} harm={m_v1['harmful_n']} help={m_v1['helpful_n']}",
        flush=True,
    )

    # --- v2-A grid ---
    support_grid = (0.5, 0.6, 0.7)
    margin_grid = (0.05, 0.10, 0.15, 0.20)
    v2a_results = []
    best_a = None
    for support_min, margin_min in product(support_grid, margin_grid):
        v2 = SelectiveV2Params(
            enable_option_verify=True,
            enable_targeted=False,
            option_support_min=support_min,
            option_margin_min=margin_min,
            targeted_trigger="off",
        )
        rows = _eval_rows_from_caches(sel, caches, th_v1, v2)
        m = _metrics(rows)
        rec = {"v2_params": v2.as_dict(), "metrics": m}
        v2a_results.append(rec)
        print(
            f"[v2a s={support_min} m={margin_min}] exact={m['exact']}/{m['n']} "
            f"use={m['use_rag']} harm={m['harmful_n']} help={m['helpful_n']} "
            f"flip_bad={m['flip_correct_to_error']}",
            flush=True,
        )
        if best_a is None or _score_key(m) > best_a[0]:
            best_a = (_score_key(m), v2, m, rows)

    assert best_a is not None
    _, v2a_best, m_v2a, rows_v2a = best_a
    v2a_dir = out_root / "validation_v2a"
    v2a_dir.mkdir(parents=True, exist_ok=True)
    (v2a_dir / "metrics.json").write_text(
        json.dumps(
            {
                "metrics": m_v2a,
                "v2_params": v2a_best.as_dict(),
                "thresholds": th_v1.as_dict(),
                "grid_results": v2a_results,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (v2a_dir / "rows.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows_v2a), encoding="utf-8"
    )

    # --- v2-B grid：在 best v2-A 上开 targeted ---
    v2b_results = []
    best_b = None
    for trigger in ("strict", "medium"):
        for support_min, margin_min in product((0.5, 0.6, 0.7), (0.05, 0.10, 0.15)):
            v2 = SelectiveV2Params(
                enable_option_verify=True,
                enable_targeted=True,
                option_support_min=support_min,
                option_margin_min=margin_min,
                targeted_trigger=trigger,
            )
            rows = _eval_rows_from_caches(sel, caches, th_v1, v2)
            m = _metrics(rows)
            rec = {"v2_params": v2.as_dict(), "metrics": m}
            v2b_results.append(rec)
            print(
                f"[v2b {trigger} s={support_min} m={margin_min}] exact={m['exact']}/{m['n']} "
                f"use={m['use_rag']} tgt={m['use_targeted_rag']} harm={m['harmful_n']} "
                f"help={m['helpful_n']}",
                flush=True,
            )
            if best_b is None or _score_key(m) > best_b[0]:
                best_b = (_score_key(m), v2, m, rows)

    assert best_b is not None
    _, v2b_best, m_v2b, rows_v2b = best_b
    v2b_dir = out_root / "validation_v2b"
    v2b_dir.mkdir(parents=True, exist_ok=True)
    (v2b_dir / "metrics.json").write_text(
        json.dumps(
            {
                "metrics": m_v2b,
                "v2_params": v2b_best.as_dict(),
                "thresholds": th_v1.as_dict(),
                "grid_results": v2b_results,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (v2b_dir / "rows.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows_v2b), encoding="utf-8"
    )

    # 选择最终版本：exact / -harmful / helpful；平局时优先启用 option-level 的更简版本
    def _pick_key(name: str, m: dict) -> tuple:
        # 平局：prefer v2a > v2b > v1（先证明 option unlock，再开 targeted）
        pref = {"v2a": 2, "v2b": 1, "v1": 0}.get(name, 0)
        return (*_score_key(m), pref)

    candidates = [
        ("v1", v1_params, m_v1),
        ("v2a", v2a_best, m_v2a),
        ("v2b", v2b_best, m_v2b),
    ]
    final_name, final_v2, final_m = max(
        candidates, key=lambda x: _pick_key(x[0], x[2])
    )

    payload = {
        "method": "Selective RAG v2",
        "version": final_name,
        "temperature": 0,
        "validation_size": len(qs),
        "thresholds": th_v1.as_dict(),
        "risk_thresholds": {
            "risk_relevance_min": th_v1.risk_relevance_min,
            "risk_sufficiency_min": th_v1.risk_sufficiency_min,
        },
        "option_support_min": final_v2.option_support_min,
        "option_margin_min": final_v2.option_margin_min,
        "targeted_trigger": final_v2.targeted_trigger if final_name == "v2b" else "off",
        "v2_params": final_v2.as_dict(),
        "val_metrics": final_m,
        "comparison": {
            "v1": m_v1,
            "v2a": m_v2a,
            "v2b": m_v2b,
        },
        "locked": True,
        "locked_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "protocol": (
            "Validation only. Exact primary; then lower harmful; then higher helpful. "
            "Question-level thresholds frozen from v1 lock. Test forbidden for tuning."
        ),
        "why": (
            f"Selected {final_name} by (exact, -harmful, helpful). "
            f"v1={m_v1['exact']}/{m_v1['harmful_n']}, "
            f"v2a={m_v2a['exact']}/{m_v2a['harmful_n']}, "
            f"v2b={m_v2b['exact']}/{m_v2b['harmful_n']}."
        ),
        "validation_dir": str(out_root.relative_to(ROOT)).replace("\\", "/"),
    }
    LOCK_V2_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_root / "tune_v2_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"[lock-v2] {LOCK_V2_PATH} version={final_name} "
        f"exact={final_m['exact']}/{final_m['n']} harm={final_m['harmful_n']}",
        flush=True,
    )
    return payload


def run_split(
    rt: Runtime,
    split: str,
    *,
    version: str = "v1",
    resume_dir: str = "",
) -> Path:
    if version == "v1":
        th = load_locked_thresholds()
        v2 = SelectiveV2Params(enable_option_verify=False, enable_targeted=False)
        out_base = ABLATION_DIR
    else:
        th, v2, locked_ver = load_locked_v2()
        if version in {"v2a", "v2b"}:
            # 允许用锁定参数但强制开关
            if version == "v2a":
                v2.enable_option_verify = True
                v2.enable_targeted = False
                v2.targeted_trigger = "off"
            else:
                v2.enable_option_verify = True
                v2.enable_targeted = True
                if v2.targeted_trigger == "off":
                    v2.targeted_trigger = "strict"
        else:
            version = locked_ver
        out_base = V2_DIR

    if split == "test":
        print("TEST IS FROZEN", flush=True)
        if version == "v1":
            print(f"Using locked config:\n{LOCK_PATH}", flush=True)
        else:
            print(f"Using locked config:\n{LOCK_V2_PATH}", flush=True)

    qs = _questions(split)
    out_dir = (
        Path(resume_dir)
        if resume_dir
        else out_base
        / f"{split}_{version}_{datetime.now(timezone.utc).astimezone().strftime('%Y%m%d_%H%M%S')}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl = out_dir / "selective_rag.jsonl"
    done = set()
    if jsonl.exists():
        for line in jsonl.read_text(encoding="utf-8").splitlines():
            if line.strip():
                done.add(json.loads(line)["id"])
    print(
        f"[run] split={split} version={version} n={len(qs)} done={len(done)} out={out_dir}",
        flush=True,
    )
    sel = _make_selective(rt, th, version=version, v2=v2)
    with jsonl.open("a", encoding="utf-8") as handle:
        for i, item in enumerate(qs, 1):
            qid = item["id"]
            if qid in done:
                continue
            gold = _parse_gold(item)
            try:
                answer, stats = sel.query(item["question"])
                err = None
            except Exception as exc:  # noqa: BLE001
                answer, stats, err = "", {}, f"{type(exc).__name__}: {exc}"
            pred = _parse_pred(answer, force_single=len(gold) == 1)
            direct_letters = set(
                stats.get("direct_letters")
                or parse_final_letters(stats.get("direct_answer") or "")
            )
            direct_correct = bool(gold) and direct_letters == gold and not err
            row = {
                "id": qid,
                "q_type": item.get("q_type"),
                "domain": item.get("domain"),
                "exact_match": bool(gold) and pred == gold and not err,
                "direct_correct": direct_correct,
                "pred_letters": sorted(pred),
                "direct_letters": sorted(direct_letters),
                "gold_letters": sorted(gold),
                "decision": (stats or {}).get("decision"),
                "high_risk": (stats or {}).get("high_risk"),
                "assessment": (stats or {}).get("assessment"),
                "stance": (stats or {}).get("stance"),
                "gate": (stats or {}).get("gate"),
                "option_scores": (stats or {}).get("option_scores"),
                "option_stances": (stats or {}).get("option_stances"),
                "best_option": (stats or {}).get("best_option"),
                "second_option": (stats or {}).get("second_option"),
                "option_margin": (stats or {}).get("option_margin"),
                "question_relevance": (stats or {}).get("question_relevance"),
                "question_sufficiency": (stats or {}).get("question_sufficiency"),
                "question_contradiction": (stats or {}).get("question_contradiction"),
                "targeted_triggered": (stats or {}).get("targeted_triggered"),
                "targeted_query": (stats or {}).get("targeted_query"),
                "targeted_evidence_count": (stats or {}).get("targeted_evidence_count"),
                "override_reason": (stats or {}).get("override_reason"),
                "preservation_reason": (stats or {}).get("preservation_reason"),
                "retrieved_ids": (stats or {}).get("retrieved_ids"),
                "answer_changed": (stats or {}).get("answer_changed"),
                "version": (stats or {}).get("version"),
                "error": err,
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            mark = "OK" if row["exact_match"] else "MISS"
            print(
                f"[{split} {len(done)+i}/{len(qs)}] {qid} {row['decision']} {mark}",
                flush=True,
            )

    rows = [json.loads(l) for l in jsonl.read_text(encoding="utf-8").splitlines() if l.strip()]
    idset = {q["id"] for q in qs}
    rows = [r for r in rows if r["id"] in idset]
    summary = _metrics(rows)
    summary["thresholds"] = th.as_dict()
    summary["v2_params"] = v2.as_dict()
    summary["split"] = split
    summary["version"] = version
    (out_dir / "selective_rag_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"[{split}/{version}] {summary['exact']}/{summary['n']} = {100*summary['exact_rate']:.1f}% "
        f"harm={summary['harmful_retrieval_rate']} help={summary['helpful_retrieval_rate']}",
        flush=True,
    )
    return out_dir


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["val", "test"], required=True)
    ap.add_argument("--tune", action="store_true", help="仅 val：v1 网格锁参")
    ap.add_argument("--tune-v2", action="store_true", help="仅 val：v2-A/B 网格锁参")
    ap.add_argument(
        "--reuse-cache",
        default="",
        help="复用已有 validation_*/caches.pkl（仅重跑 option verify + 网格）",
    )
    ap.add_argument(
        "--version",
        choices=["v1", "v2a", "v2b", "locked"],
        default="locked",
        help="运行版本；locked=读对应锁参文件",
    )
    ap.add_argument("--resume-dir", default="")
    args = ap.parse_args()
    _load_dotenv()
    if not os.environ.get("DEEPSEEK_API_KEY"):
        raise SystemExit("请设置 DEEPSEEK_API_KEY")
    if args.tune and args.split != "val":
        raise SystemExit("--tune 只能配 --split val")
    if args.tune_v2 and args.split != "val":
        raise SystemExit("--tune-v2 只能配 --split val")
    if args.split == "test" and (args.tune or args.tune_v2):
        raise SystemExit("禁止在测试集调参")

    rt = Runtime()
    if args.tune:
        tune_val(rt)
    elif args.tune_v2:
        tune_val_v2(rt, reuse_cache=args.reuse_cache)
    else:
        version = args.version
        if version == "locked":
            if LOCK_V2_PATH.exists():
                _, _, version = load_locked_v2()
            else:
                version = "v1"
        run_split(rt, args.split, version=version, resume_dir=args.resume_dir)


if __name__ == "__main__":
    main()
