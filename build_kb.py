"""
build_kb.py —— 将 data/raw/manual 下的指南/说明书入库到 chroma_db。

约定（证据分级依赖）：
  cnki_pdf/     → metadata.source = guideline
  drug_labels/  → metadata.source = drug_label
  textbook/     → 本次跳过（用户确认暂不加入）
来源类型只决定排序先后，不乘固定系数。

用法：
  py -3 build_kb.py
  py -3 build_kb.py --reset   # 清空集合后重建
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import chromadb
from FlagEmbedding import BGEM3FlagModel

from rag.config import CHROMA_DIR, COLLECTION_NAME, MANUAL_DIR, RAG_ROOT
from rag.ingestion_pipeline import DocMeta, IngestionPipeline


def make_batch_embed(model_path: Path, batch_size: int = 16):
    model = BGEM3FlagModel(str(model_path), use_fp16=True)

    def embed(texts):
        if isinstance(texts, str):
            texts = [texts]
            single = True
        else:
            single = False
        out = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            vecs = model.encode(batch, max_length=512)["dense_vecs"]
            out.extend(v.tolist() for v in vecs)
        return out[0] if single else out

    return embed


def infer_source(folder_name: str) -> str | None:
    mapping = {
        "cnki_pdf": "guideline",
        "drug_labels": "drug_label",
        "textbook": "textbook",
    }
    return mapping.get(folder_name)


def collect_jobs(skip_textbook: bool = True):
    jobs = []
    for folder in ("cnki_pdf", "drug_labels", "textbook"):
        if skip_textbook and folder == "textbook":
            continue
        source = infer_source(folder)
        d = MANUAL_DIR / folder
        if not d.is_dir():
            continue
        for fp in sorted(d.iterdir()):
            if not fp.is_file():
                continue
            if fp.name.startswith("README"):
                continue
            if fp.suffix.lower() not in {".pdf", ".txt", ".html", ".htm"}:
                continue
            title = fp.stem
            doc_id = f"{source}_{hashlib.md5(fp.name.encode('utf-8')).hexdigest()[:10]}"
            meta = DocMeta(
                doc_id=doc_id,
                title=title,
                source=source,
                topic="t2dm",
                publisher="",
                publish_date="",
                language="zh",
            )
            jobs.append((str(fp), meta))
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reset", action="store_true", help="删除已有集合后重建")
    ap.add_argument("--include-textbook", action="store_true", help="同时入库 textbook/")
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()

    model_path = RAG_ROOT / "models" / "bge-m3"
    if not (model_path / "pytorch_model.bin").exists() and not (model_path / "model.safetensors").exists():
        raise SystemExit(f"找不到本地 BGE-M3：{model_path}")

    jobs = collect_jobs(skip_textbook=not args.include_textbook)
    if not jobs:
        raise SystemExit("manual 目录下没有可入库文件")

    print(f"准备入库 {len(jobs)} 份文档 → {CHROMA_DIR} / {COLLECTION_NAME}")
    for fp, meta in jobs:
        print(f"  [{meta.source}] {meta.title}")

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    if args.reset:
        try:
            client.delete_collection(COLLECTION_NAME)
            print(f"已删除旧集合 {COLLECTION_NAME}")
        except Exception:
            pass

    print("加载 BGE-M3 ...")
    embed = make_batch_embed(model_path, batch_size=args.batch_size)
    pipeline = IngestionPipeline(client, embed, collection_name=COLLECTION_NAME)
    results = pipeline.ingest_batch(jobs)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = RAG_ROOT / "reports" / f"ingest_{stamp}.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    col = client.get_collection(COLLECTION_NAME)
    print(f"\n集合 `{COLLECTION_NAME}` 当前 chunk 数: {col.count()}")
    print(f"入库报告: {out}")
    print("说明：证据分级在检索匹配层（matching_mode=evidence_qc）生效，依赖本步写入的 source 元数据。")


if __name__ == "__main__":
    main()
