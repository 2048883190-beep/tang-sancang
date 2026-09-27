# -*- coding: utf-8 -*-
"""
expand_questions.py —— 扩充评测题库（CMB / 本地缓存 / 指南合成）

目标：主表 in_scope∩single 显著增加（默认目标 ≥500）。

用法：
  py -3 -m src.eval.expand_questions
  py -3 -m src.eval.expand_questions --target-main 500 --max-new 400
  py -3 -m src.eval.expand_questions --skip-clone --synthetic-max 80
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from rag.config import EVAL_DIR, RAG_ROOT, REPORTS_DIR
from rag.domain_gate import filter_main_table, label_question

DATASETS_DIR = RAG_ROOT / "eval" / "datasets"
CMB_DIR = DATASETS_DIR / "CMB"
KEYWORDS = re.compile(
    r"糖尿病|血糖|胰岛素|二甲双胍|糖化|HbA1c|降糖|糖尿病肾病|糖尿病足|"
    r"SGLT|GLP-?1|DPP-?4|阿卡波糖|格列|列净|列汀|鲁肽|酮症|高渗|"
    r"eGFR|肾功能不全|妊娠.*糖|老年糖尿病|IGT|IFG",
    re.I,
)
OPTION_RE = re.compile(r"[A-E][\.．、:：]\s*")


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


def _norm_q(text: str) -> str:
    t = re.sub(r"\s+", "", text or "")
    t = re.sub(r"[选项]+[:：]?", "", t)
    return t[:200]


def _qhash(text: str) -> str:
    return hashlib.md5(_norm_q(text).encode("utf-8")).hexdigest()[:16]


def _guess_q_type(question: str) -> str:
    t = question or ""
    if re.search(r"禁忌|慎用|不良反应|低血糖|乳酸酸中毒|禁用", t):
        return "safety"
    if re.search(r"首选|选用|治疗措施|下一步|方案", t):
        return "decision"
    if re.search(r"相互作用|联用|合用|同用", t):
        return "interaction"
    if re.search(r"比较|优于|区别|差异|相比", t):
        return "compare"
    return "fact"


def _letters_from_answer(ans: Any) -> str:
    if ans is None:
        return ""
    if isinstance(ans, list):
        return "".join(sorted({str(x).upper() for x in ans if str(x).upper() in "ABCDE"}))
    s = str(ans).upper()
    # 常见： "BCD" / "答案：B" / ["B","C"]
    found = re.findall(r"[A-E]", s)
    if not found:
        return ""
    # 若像 "正确答案是B和C" 取全部唯一；若单字符答案取第一个
    uniq = []
    for ch in found:
        if ch not in uniq:
            uniq.append(ch)
    return "".join(uniq)


def _format_options(opts: Any) -> str:
    if not opts:
        return ""
    if isinstance(opts, dict):
        parts = []
        for k in sorted(opts.keys()):
            kk = str(k).upper().strip()
            if kk and kk[0] in "ABCDE":
                parts.append(f"{kk[0]}. {opts[k]}")
        return " 选项： " + " ".join(parts) if parts else ""
    if isinstance(opts, list):
        parts = []
        for i, o in enumerate(opts):
            letter = "ABCDE"[i] if i < 5 else "?"
            parts.append(f"{letter}. {o}")
        return " 选项： " + " ".join(parts)
    return f" 选项： {opts}"


def _item_from_raw(
    *,
    qid: str,
    question: str,
    gold: str,
    source: str,
    subject: str = "",
    extra: Optional[dict] = None,
) -> Optional[dict]:
    if not question or not gold:
        return None
    if not KEYWORDS.search(question):
        return None
    gold_clean = "".join(ch for ch in gold.upper() if ch in "ABCDE")
    if not gold_clean:
        return None
    item = {
        "id": qid,
        "q_type": _guess_q_type(question),
        "question": question.strip(),
        "ground_truth": f"正确答案：{gold_clean}",
        "gold": gold_clean,
        "source_dataset": source,
        "subject": subject or "",
    }
    if extra:
        item.update(extra)
    return label_question(item)


def _walk_json_files(root: Path) -> Iterable[Path]:
    if not root.exists():
        return []
    for p in root.rglob("*"):
        if p.suffix.lower() in {".json", ".jsonl"} and p.is_file():
            # 跳过超大无关
            if p.stat().st_size > 200_000_000:
                continue
            yield p


def _try_load_json(path: Path) -> Any:
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        text = path.read_text(encoding="utf-8-sig")
    if path.suffix.lower() == ".jsonl":
        rows = []
        for line in text.splitlines():
            if line.strip():
                rows.append(json.loads(line))
        return rows
    return json.loads(text)


def _extract_from_obj(obj: Any, source: str, prefix: str) -> List[dict]:
    out: List[dict] = []
    rows: List[Any]
    if isinstance(obj, list):
        rows = obj
    elif isinstance(obj, dict):
        for key in ("data", "questions", "exam", "test", "items"):
            if isinstance(obj.get(key), list):
                rows = obj[key]
                break
        else:
            # 可能是 {id: {...}}
            if all(isinstance(v, dict) for v in obj.values()):
                rows = list(obj.values())
            else:
                rows = [obj]
    else:
        return out

    for i, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        q = (
            row.get("question")
            or row.get("stem")
            or row.get("query")
            or row.get("q")
            or ""
        )
        opts = row.get("option") or row.get("options") or row.get("choices")
        if opts and "选项" not in str(q):
            q = f"{q}{_format_options(opts)}"
        ans = (
            row.get("answer")
            or row.get("gold")
            or row.get("正确答案")
            or row.get("output")
            or row.get("label")
        )
        gold = _letters_from_answer(ans)
        qid = str(row.get("id") or row.get("qid") or f"{prefix}_{i:05d}")
        if not qid.startswith(prefix):
            qid = f"{prefix}_{qid}"
        item = _item_from_raw(
            qid=qid,
            question=str(q),
            gold=gold,
            source=source,
            subject=str(row.get("exam_type") or row.get("subject") or row.get("category") or ""),
        )
        if item:
            out.append(item)
    return out


def ensure_cmb(skip_clone: bool = False) -> Path:
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    data_dir = CMB_DIR / "data"
    # 已有数据则跳过
    if any(data_dir.rglob("*.json")) if data_dir.exists() else False:
        print(f"[cmb] already have data under {data_dir}", flush=True)
        return CMB_DIR
    if skip_clone:
        print("[cmb] skip clone", flush=True)
        return CMB_DIR

    # 1) 优先 curl 下载 GitHub archive（本机常无 git）
    zip_main = DATASETS_DIR / "CMB-main.zip"
    if not zip_main.exists():
        url = "https://github.com/FreedomIntelligence/CMB/archive/refs/heads/main.zip"
        print(f"[cmb] curl download {url}", flush=True)
        try:
            subprocess.run(
                ["curl", "-L", "--retry", "3", "-o", str(zip_main), url],
                check=True,
                timeout=600,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[cmb] curl failed: {exc}", flush=True)

    if zip_main.exists() and zip_main.stat().st_size > 1000:
        import shutil

        print(f"[cmb] extract {zip_main}", flush=True)
        try:
            with zipfile.ZipFile(zip_main, "r") as zf:
                zf.extractall(DATASETS_DIR)
            extracted = DATASETS_DIR / "CMB-main"
            if extracted.exists():
                if CMB_DIR.exists():
                    shutil.rmtree(CMB_DIR)
                extracted.rename(CMB_DIR)
        except Exception as exc:  # noqa: BLE001
            print(f"[cmb] extract repo failed: {exc}", flush=True)

    # 2) 解压内嵌 CMB.zip
    inner = CMB_DIR / "data" / "CMB.zip"
    if inner.exists():
        print(f"[cmb] unzip {inner}", flush=True)
        try:
            with zipfile.ZipFile(inner, "r") as zf:
                zf.extractall(CMB_DIR / "data")
        except Exception as exc:  # noqa: BLE001
            print(f"[cmb] unzip failed: {exc}", flush=True)

    # 3) 可选 git clone 兜底
    if not (any(data_dir.rglob("*.json")) if data_dir.exists() else False):
        print("[cmb] cloning FreedomIntelligence/CMB ...", flush=True)
        try:
            subprocess.run(
                [
                    "git",
                    "clone",
                    "--depth",
                    "1",
                    "https://github.com/FreedomIntelligence/CMB.git",
                    str(CMB_DIR),
                ],
                check=True,
                timeout=600,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[cmb] clone failed: {exc}", flush=True)
    return CMB_DIR


def harvest_local_datasets() -> List[dict]:
    harvested: List[dict] = []
    roots = [
        (CMB_DIR, "CMB-Exam", "cmb"),
        (DATASETS_DIR / "CMExam", "CMExam", "cmexam"),
        (DATASETS_DIR / "MedQA", "MedQA-ZH", "medqa"),
        (DATASETS_DIR / "MedQA-ZH", "MedQA-ZH", "medqa"),
        (RAG_ROOT / "eval" / "datasets", "local", "local"),
    ]
    seen_files: Set[str] = set()
    for root, source, prefix in roots:
        if not root.exists():
            continue
        for fp in _walk_json_files(root):
            key = str(fp.resolve())
            if key in seen_files:
                continue
            seen_files.add(key)
            name = fp.name.lower()
            # 跳过我们自己的评测产物
            if "expanded_questions" in name or "smoke" in name:
                continue
            try:
                obj = _try_load_json(fp)
            except Exception as exc:  # noqa: BLE001
                print(f"[skip] {fp}: {exc}", flush=True)
                continue
            src = source
            if "cmb" in name or "CMB" in str(fp):
                src = "CMB-Exam"
            elif "cmexam" in name:
                src = "CMExam"
            elif "medqa" in name:
                src = "MedQA-ZH"
            got = _extract_from_obj(obj, src, prefix)
            if got:
                print(f"[harvest] {fp.name}: +{len(got)}", flush=True)
                harvested.extend(got)
    return harvested


def synthetic_from_kb(n: int, llm) -> List[dict]:
    """从知识库抽 chunk，用 LLM 生成单选 MCQ（synthetic=true）。"""
    if n <= 0:
        return []
    import chromadb
    from rag.config import CHROMA_DIR, COLLECTION_NAME

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    col = client.get_collection(COLLECTION_NAME)
    total = col.count()
    if total <= 0:
        return []
    # 均匀抽样
    step = max(1, total // max(n * 3, 1))
    ids = [f"idx_{i}" for i in range(0, min(total, n * 4), step)]
    # chroma 不一定有这些 id；改用 get limit/offset 风格：peek + random sample via get
    peek = col.peek(min(200, total))
    docs = peek.get("documents") or []
    metas = peek.get("metadatas") or []
    out: List[dict] = []
    for i, doc in enumerate(docs):
        if len(out) >= n:
            break
        meta = metas[i] if i < len(metas) else {}
        src = (meta or {}).get("source") or (meta or {}).get("title") or "kb"
        prompt = (
            "根据下列医学资料，出一道中文单选选择题（A-E 五个选项），"
            "必须与2型糖尿病/降糖药/并发症相关。\n"
            "严格输出 JSON：{\"question\":\"题干 选项： A. .. B. ..\",\"gold\":\"X\",\"explain\":\"一句\"}\n"
            f"资料：\n{(doc or '')[:900]}"
        )
        try:
            raw = llm(prompt)
            m = re.search(r"\{[\s\S]*\}", raw)
            if not m:
                continue
            obj = json.loads(m.group(0))
            gold = _letters_from_answer(obj.get("gold"))
            if len(gold) != 1:
                continue
            q = str(obj.get("question") or "")
            item = _item_from_raw(
                qid=f"syn_{_qhash(q)}",
                question=q,
                gold=gold,
                source="synthetic_kb",
                subject=str(src),
                extra={"synthetic": True, "explain": obj.get("explain")},
            )
            if item and item.get("scope") == "in_scope":
                out.append(item)
        except Exception as exc:  # noqa: BLE001
            print(f"[syn] fail: {exc}", flush=True)
            continue
    print(f"[syn] generated {len(out)}", flush=True)
    return out


def merge_questions(
    base: List[dict],
    newcomers: List[dict],
    max_new: int,
) -> Tuple[List[dict], int]:
    seen: Set[str] = {_qhash(q.get("question") or "") for q in base}
    id_seen: Set[str] = {str(q.get("id")) for q in base}
    added = 0
    merged = list(base)
    for q in newcomers:
        if added >= max_new:
            break
        h = _qhash(q.get("question") or "")
        if h in seen:
            continue
        qid = str(q.get("id"))
        if qid in id_seen:
            q = dict(q)
            q["id"] = f"{qid}_{h[:6]}"
            qid = q["id"]
        seen.add(h)
        id_seen.add(qid)
        merged.append(label_question(q))
        added += 1
    return merged, added


def write_bundle(questions: List[dict], out_path: Path, note: str) -> dict:
    labeled = [label_question(q) for q in questions]
    main = filter_main_table(labeled)
    meta = {
        "version": 4,
        "purpose": "BSCV-RAG expanded eval (parallel fusion experiments)",
        "count": len(labeled),
        "main_table_n": len(main),
        "split": "expanded_v4",
        "notes": note,
        "updated_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "type_counts": dict(Counter(q.get("q_type") for q in labeled)),
        "source_counts": dict(Counter(q.get("source_dataset") for q in labeled)),
        "scope_counts": dict(Counter(q.get("scope") for q in labeled)),
        "select_counts": dict(Counter(q.get("select_type") for q in labeled)),
    }
    payload = {**meta, "questions": labeled}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    meta_path = out_path.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[write] {out_path} count={meta['count']} main={meta['main_table_n']}", flush=True)
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--base",
        default=str(EVAL_DIR / "expanded_questions.json"),
        help="现有题库",
    )
    ap.add_argument(
        "--out",
        default=str(EVAL_DIR / "expanded_questions.json"),
        help="输出路径（默认覆盖并备份）",
    )
    ap.add_argument("--target-main", type=int, default=500)
    ap.add_argument("--max-new", type=int, default=400)
    ap.add_argument("--skip-clone", action="store_true")
    ap.add_argument("--synthetic-max", type=int, default=60)
    args = ap.parse_args()

    _load_dotenv()
    base_path = Path(args.base)
    out_path = Path(args.out)

    base_data = json.loads(base_path.read_text(encoding="utf-8"))
    base_qs = list(base_data.get("questions") or [])
    main0 = len(filter_main_table([label_question(q) for q in base_qs]))
    print(f"[base] {len(base_qs)} questions, main_table={main0}", flush=True)

    # 备份
    bak = EVAL_DIR / f"expanded_questions_bak_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    bak.write_text(base_path.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"[bak] {bak}", flush=True)

    ensure_cmb(skip_clone=args.skip_clone)
    harvested = harvest_local_datasets()
    print(f"[harvest] total candidates={len(harvested)}", flush=True)

    # 优先补主表：in_scope + single
    harvested_pref = sorted(
        harvested,
        key=lambda q: (
            0 if q.get("scope") == "in_scope" and q.get("select_type") == "single" else 1,
            q.get("q_type") or "",
        ),
    )

    merged, added = merge_questions(base_qs, harvested_pref, max_new=args.max_new)
    main_n = len(filter_main_table([label_question(q) for q in merged]))
    print(f"[merge] +{added} from harvest, main_table={main_n}", flush=True)

    # 仍不足则合成
    need_more = max(0, args.target_main - main_n)
    syn_budget = min(args.synthetic_max, max(need_more + 20, 0) if need_more else 0)
    if syn_budget > 0 and os.environ.get("DEEPSEEK_API_KEY"):
        from openai import OpenAI

        client = OpenAI(
            api_key=os.environ["DEEPSEEK_API_KEY"],
            base_url=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        )
        model = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")

        def llm(prompt: str) -> str:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=600,
                temperature=0.4,
            )
            return (resp.choices[0].message.content or "").strip()

        syn = synthetic_from_kb(syn_budget, llm)
        merged, added2 = merge_questions(merged, syn, max_new=syn_budget)
        print(f"[merge] +{added2} synthetic", flush=True)
    elif need_more > 0:
        print("[syn] skipped (no API key or budget 0)", flush=True)

    note = (
        f"自 {base_path.name} 扩充；CMB/本地 harvest + 可选 synthetic；"
        f"目标主表≥{args.target_main}。"
    )
    meta = write_bundle(merged, out_path, note)
    (REPORTS_DIR / "expand_questions_latest.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if meta["main_table_n"] < args.target_main:
        print(
            f"[warn] main_table={meta['main_table_n']} < target {args.target_main}; "
            "请检查 CMB 是否下载成功或提高 --synthetic-max",
            flush=True,
        )


if __name__ == "__main__":
    main()
