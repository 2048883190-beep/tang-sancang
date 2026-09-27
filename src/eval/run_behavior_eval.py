# -*- coding: utf-8 -*-
"""
行为评测：适用、冲突可见、可核对、范围外不查库。
管道与当前系统一致：题干判定 → 稠密召回 → 字典序 → 正反依据留存。

  py -3 -m src.eval.run_behavior_eval
"""
from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from rag.clinical_rules import infer_evidence_tier
from rag.config import CHROMA_DIR, COLLECTION_NAME, EMBEDDING_MODEL_ID, RAG_ROOT, REPORTS_DIR
from rag.domain_gate import chunk_domain_conflict, classify_domain
from rag.matching_layer import DenseMatcher
from rag.parallel_fusion import (
    _stance_blob,
    assemble_with_stance_quota,
    parallel_appraise_and_fuse,
)

FIRSTLINE_RE = re.compile(r"首选|一线|推荐")
PROBE_PATH = ROOT / "data" / "eval" / "behavior_probes.json"


def _load_dotenv() -> None:
    env_path = ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os_environ_set(k.strip(), v.strip())


def os_environ_set(k: str, v: str) -> None:
    import os

    os.environ.setdefault(k, v)


def blob(chunks: List[dict]) -> str:
    return "\n".join((c.get("content") or "") for c in chunks)


def cue_hit(text: str, cues: List[str]) -> List[str]:
    t = text or ""
    return [c for c in cues if c in t]


def pos_neg_for_drug(chunks: List[dict], drug: str) -> dict:
    pos = neg = False
    n_mention = 0
    for c in chunks:
        text = c.get("content") or ""
        st = _stance_blob(text)
        mentioned = (f"DRUG:{drug}" in st) or (drug and drug in text)
        if not mentioned:
            continue
        n_mention += 1
        if "POS" in st:
            pos = True
        if "NEG" in st:
            neg = True
    return {"pos": pos, "neg": neg, "both": pos and neg, "n_mention": n_mention}


def score_probe(probe: dict, decision, raw: List[dict], fused: List[dict]) -> dict:
    exp = probe.get("expect") or {}
    axis = probe["axis"]
    assembled = fused[:10] if fused else []
    ctx = blob(assembled)
    checks = []
    ok = True
    note = []

    if "should_retrieve" in exp:
        hit = bool(decision.should_retrieve) == bool(exp["should_retrieve"])
        checks.append({"name": "should_retrieve", "pass": hit, "got": decision.should_retrieve})
        ok = ok and hit
        if not hit:
            note.append(f"域门={decision.domain}/{decision.should_retrieve} 期望检索={exp['should_retrieve']}")

    if axis == "abstain":
        return {"id": probe["id"], "axis": axis, "pass": ok, "checks": checks, "note": note}

    if not decision.should_retrieve and exp.get("should_retrieve"):
        return {"id": probe["id"], "axis": axis, "pass": False, "checks": checks, "note": note + ["应检索却被域门拦住"]}

    if axis == "applicability":
        cues = exp.get("condition_cues") or []
        found = cue_hit(ctx, cues)
        need = max(1, min(2, len(cues)))
        c_ok = len(found) >= need
        checks.append({"name": "condition_cues", "pass": c_ok, "found": found, "need": cues})
        ok = ok and c_ok
        if exp.get("forbid_unconditional_firstline_only"):
            fl = bool(FIRSTLINE_RE.search(ctx))
            cond = len(found) >= 1
            bad = fl and not cond
            checks.append({"name": "not_firstline_without_condition", "pass": not bad, "firstline": fl})
            ok = ok and (not bad)

    if axis == "conflict":
        drug = exp.get("drug") or ""
        raw_pn = pos_neg_for_drug(raw, drug)
        fus_pn = pos_neg_for_drug(assembled, drug)
        checks.append({"name": "pool_mentions_drug", "pass": raw_pn["n_mention"] > 0, **raw_pn})
        if exp.get("require_both_pos_neg"):
            pool_ok = raw_pn["both"]
            kept_ok = (not pool_ok) or fus_pn["both"]
            checks.append({"name": "pool_has_both_stances", "pass": pool_ok, **raw_pn})
            checks.append({"name": "fusion_kept_both_if_pool_had", "pass": kept_ok, "fused": fus_pn})
            ok = ok and pool_ok and kept_ok
            if pool_ok and not fus_pn["both"]:
                note.append("库里两边都有，融合后只剩一面")
            if not pool_ok:
                note.append("检索池本身没有同时覆盖推荐与限制")
        cues = exp.get("condition_cues") or []
        if cues:
            found = cue_hit(ctx, cues)
            c_ok = len(found) >= 1
            checks.append({"name": "conflict_cues", "pass": c_ok, "found": found})
            ok = ok and c_ok

    if axis == "citation":
        prefer = set(exp.get("prefer_tiers") or ["guideline", "drug_label"])
        tiers = [infer_evidence_tier(c.get("metadata"), c.get("content") or "") for c in assembled]
        n = len(tiers) or 1
        frac = sum(t in prefer for t in tiers) / n
        c_ok = frac >= 0.4 and len(assembled) > 0
        checks.append({"name": "high_tier_fraction", "pass": c_ok, "frac": round(frac, 3), "tiers": tiers[:10]})
        ok = ok and c_ok

    return {"id": probe["id"], "axis": axis, "pass": ok, "checks": checks, "note": note}


def main():
    _load_dotenv()
    data = json.loads(PROBE_PATH.read_text(encoding="utf-8"))
    probes = data["probes"]

    import chromadb
    from FlagEmbedding import BGEM3FlagModel

    local = RAG_ROOT / "models" / "bge-m3"
    model_id = str(local) if local.exists() else EMBEDDING_MODEL_ID
    print(f"[init] {model_id} probes={len(probes)}", flush=True)
    model = BGEM3FlagModel(model_id, use_fp16=True)
    chroma = chromadb.PersistentClient(path=str(CHROMA_DIR))
    coll = chroma.get_collection(COLLECTION_NAME)

    def embed(text: str):
        return model.encode([text], max_length=512)["dense_vecs"][0].tolist()

    matcher = DenseMatcher(coll, embed)
    rows = []

    for i, probe in enumerate(probes, 1):
        q = probe["query"]
        decision = classify_domain(q)
        raw: List[dict] = []
        fused: List[dict] = []
        quota_diag = {}
        if decision.should_retrieve:
            raw = matcher.retrieve(q, k=30, role="forward")
            ids = {str(c.get("id") or "") for c in raw}
            fused_all, _ = parallel_appraise_and_fuse(
                q,
                raw,
                fwd_ids=ids,
                rev_ids=ids,
            )
            survived = [c for c in fused_all if float(c.get("score") or 0) > 0] or list(raw)
            kept = []
            for c in survived:
                noisy, _ = chunk_domain_conflict(
                    decision.domain, c.get("content") or "", c.get("metadata")
                )
                if not noisy:
                    kept.append(c)
            fused, quota_diag = assemble_with_stance_quota(q, kept or survived, k=10)
        rec = score_probe(probe, decision, raw, fused)
        rec["domain"] = decision.domain
        rec["n_raw"] = len(raw)
        rec["n_fused"] = len(fused)
        rec["stance_quota"] = quota_diag
        rows.append(rec)
        st = "PASS" if rec["pass"] else "FAIL"
        print(f"[{i}/{len(probes)}] {st} {probe['axis']:14s} {probe['id']} {'; '.join(rec['note'])}", flush=True)

    by_axis = defaultdict(list)
    for r in rows:
        by_axis[r["axis"]].append(r["pass"])
    summary = {
        "axis": {
            ax: {"n": len(vs), "pass": sum(vs), "rate": round(sum(vs) / len(vs), 4) if vs else 0}
            for ax, vs in by_axis.items()
        },
        "overall_pass": sum(r["pass"] for r in rows),
        "overall_n": len(rows),
    }
    print("\nSUMMARY", json.dumps(summary, ensure_ascii=False, indent=2), flush=True)

    out_dir = REPORTS_DIR / f"behavior_{datetime.now(timezone.utc).astimezone().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# 行为评测（检索侧）",
        "",
        f"- 通过 **{summary['overall_pass']}/{summary['overall_n']}**",
        "",
        "| 轴 | 通过 |",
        "|----|------|",
    ]
    for ax, s in summary["axis"].items():
        lines.append(f"| {ax} | {s['pass']}/{s['n']} ({100*s['rate']:.0f}%) |")
    lines += ["", "| 探针 | 轴 | 结果 | 说明 |", "|------|----|------|------|"]
    for r in rows:
        lines.append(f"| {r['id']} | {r['axis']} | {'PASS' if r['pass'] else 'FAIL'} | {'; '.join(r['note']) or '—'} |")
    (out_dir / "compare.md").write_text("\n".join(lines), encoding="utf-8")
    print("WROTE", out_dir / "compare.md", flush=True)


if __name__ == "__main__":
    main()
