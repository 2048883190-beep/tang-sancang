# -*- coding: utf-8 -*-
"""原文定位：在允许目录内打开 PDF，并尽量跳到命中页。"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any
from urllib.parse import quote

from rag.config import RAG_ROOT

ALLOWED_ROOTS = [
    (RAG_ROOT / "data" / "raw").resolve(),
    (RAG_ROOT / "data").resolve(),
]

_TOKEN_MAP: dict[str, str] = {}
_QUOTE_MAP: dict[str, str] = {}


def _norm_ws(text: str) -> str:
    return re.sub(r"\s+", "", text or "")


def safe_resolve(file_path: str) -> Path:
    raw = Path(file_path)
    if not raw.is_absolute():
        raw = RAG_ROOT / raw
    path = raw.resolve()
    for root in ALLOWED_ROOTS:
        try:
            path.relative_to(root)
            return path
        except ValueError:
            continue
    raise PermissionError(f"路径不在允许目录内: {path}")


def file_token(path: Path) -> str:
    return hashlib.sha1(str(path).encode("utf-8")).hexdigest()[:16]


def register_path(path: Path) -> str:
    path = path.resolve()
    tok = file_token(path)
    _TOKEN_MAP[tok] = str(path)
    return tok


def path_from_token(token: str) -> Path:
    p = _TOKEN_MAP.get(token)
    if not p:
        raise FileNotFoundError("原文令牌无效或已过期，请重新提问后再打开")
    return safe_resolve(p)


def store_quote(text: str) -> str:
    body = (text or "").strip()
    qid = hashlib.sha1(body.encode("utf-8")).hexdigest()[:12]
    _QUOTE_MAP[qid] = body
    return qid


def get_quote(qid: str) -> str:
    return _QUOTE_MAP.get(qid) or ""


def _quote_variants(quote_text: str) -> list[str]:
    q = (quote_text or "").strip()
    variants: list[str] = []
    if not q:
        return variants
    variants.append(q[:120])
    flat_q = _norm_ws(q)
    if flat_q:
        variants.append(flat_q[:80])
    if len(q) > 160:
        mid = q[40:160]
        variants.append(mid)
        variants.append(_norm_ws(mid)[:80])
    # 较短片段便于 PDF 内 search_for
    compact = re.sub(r"\s+", "", q)
    for n in (28, 18, 12):
        if len(compact) >= n:
            variants.append(compact[:n])
    # 去重保序
    seen: set[str] = set()
    out: list[str] = []
    for v in variants:
        v = (v or "").strip()
        if not v or v in seen:
            continue
        seen.add(v)
        out.append(v)
    return out


def locate_page(pdf_path: Path, quote_text: str) -> dict[str, Any]:
    """用段落文本在 PDF 中找页码（1-based）。"""
    import fitz

    variants = _quote_variants(quote_text)
    doc = fitz.open(str(pdf_path))
    try:
        n = doc.page_count
        for page_i in range(n):
            page = doc.load_page(page_i)
            text = page.get_text("text") or ""
            flat = _norm_ws(text)
            for v in variants:
                if v in text or _norm_ws(v) in flat:
                    return {
                        "page": page_i + 1,
                        "page_count": n,
                        "matched": True,
                        "needle": v,
                    }
            # 再试 PyMuPDF 原生搜索（对断行更稳）
            for v in variants:
                if len(v) < 8:
                    continue
                try:
                    hits = page.search_for(v[:40], quads=False)
                except Exception:  # noqa: BLE001
                    hits = []
                if hits:
                    return {
                        "page": page_i + 1,
                        "page_count": n,
                        "matched": True,
                        "needle": v[:40],
                    }
        return {"page": 1, "page_count": n, "matched": False, "needle": ""}
    finally:
        doc.close()


def render_page_png(
    pdf_path: Path,
    page: int,
    quote_text: str = "",
    *,
    zoom: float = 1.8,
) -> bytes:
    """渲染指定页为 PNG，并尽量高亮命中段落。"""
    import fitz

    doc = fitz.open(str(pdf_path))
    try:
        n = doc.page_count
        page_i = max(0, min(int(page) - 1, n - 1))
        pg = doc.load_page(page_i)
        needles = _quote_variants(quote_text)
        rects: list[Any] = []
        for v in needles:
            if len(v) < 6:
                continue
            try:
                found = pg.search_for(v[:48], quads=False) or []
            except Exception:  # noqa: BLE001
                found = []
            if found:
                rects.extend(found)
                break
        if not rects and needles:
            # 无空格紧凑串可能搜不到：用原文空格版再试
            raw = (quote_text or "").strip()
            for chunk in (raw[:36], raw[20:56], raw[40:76]):
                chunk = chunk.strip()
                if len(chunk) < 8:
                    continue
                try:
                    found = pg.search_for(chunk, quads=False) or []
                except Exception:  # noqa: BLE001
                    found = []
                if found:
                    rects.extend(found)
                    break
        for r in rects[:12]:
            annot = pg.add_highlight_annot(r)
            if annot is not None:
                annot.set_colors(stroke=(1, 0.85, 0.2))
                annot.update()
        pix = pg.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
        return pix.tobytes("png")
    finally:
        doc.close()


def build_source_info(
    *,
    file_path: str,
    title: str,
    quote_text: str,
    chunk_id: str = "",
) -> dict[str, Any]:
    path = safe_resolve(file_path)
    if not path.exists():
        raise FileNotFoundError(f"原文文件不存在: {path.name}")
    tok = register_path(path)
    qid = store_quote(quote_text)
    loc = {"page": 1, "page_count": None, "matched": False}
    if path.suffix.lower() == ".pdf":
        try:
            loc = locate_page(path, quote_text)
        except Exception as exc:  # noqa: BLE001
            loc = {
                "page": 1,
                "page_count": None,
                "matched": False,
                "locate_error": str(exc),
            }
    page = int(loc.get("page") or 1)
    matched = bool(loc.get("matched"))
    pdf_url = f"/api/source/pdf?token={quote(tok)}#page={page}"
    page_img_url = (
        f"/api/source/page.png?token={quote(tok)}"
        f"&page={page}&qid={quote(qid)}"
    )
    view_url = (
        f"/source/view?token={quote(tok)}"
        f"&page={page}"
        f"&matched={'1' if matched else '0'}"
        f"&qid={quote(qid)}"
        f"&title={quote(title or path.name)}"
    )
    return {
        "ok": True,
        "title": title or path.name,
        "file_name": path.name,
        "page": page,
        "page_count": loc.get("page_count"),
        "matched": matched,
        "pdf_url": pdf_url,
        "page_img_url": page_img_url,
        "view_url": view_url,
        "token": tok,
        "qid": qid,
        "chunk_id": chunk_id,
        "has_file": True,
    }
