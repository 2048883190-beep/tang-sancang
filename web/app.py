# -*- coding: utf-8 -*-
"""
专科循证演示台。
启动：在项目根目录执行  py -3 web/app.py
浏览器打开 http://127.0.0.1:7860
"""
from __future__ import annotations

import os
import traceback
from pathlib import Path
from typing import Any

os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from pipeline import SpecialistPipeline
from session_store import SessionManager
from source_open import build_source_info, get_quote, path_from_token, render_page_png

WEB_ROOT = Path(__file__).resolve().parent
STATIC = WEB_ROOT / "static"

pipeline: SpecialistPipeline | None = None
sessions = SessionManager()
sessions.create(title="新问询")


def get_pipeline() -> SpecialistPipeline:
    global pipeline
    if pipeline is None:
        pipeline = SpecialistPipeline()
    return pipeline


@asynccontextmanager
async def lifespan(_app: FastAPI):
    print("正在加载 BGE-M3 与知识库（首次约数十秒）…", flush=True)
    try:
        get_pipeline()
        print("预热完成。", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"预热失败，首次提问时再加载：{exc}", flush=True)
    yield


class ChatRequest(BaseModel):
    session_id: str | None = None
    message: str = Field(..., min_length=1)


class SessionCreateRequest(BaseModel):
    title: str | None = None


class SourceOpenRequest(BaseModel):
    file_path: str = Field(..., min_length=1)
    title: str = ""
    quote: str = ""
    chunk_id: str = ""


app = FastAPI(title="T2DM 专科循证检索", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.get("/api/health")
def health() -> dict[str, Any]:
    try:
        return get_pipeline().health()
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc)}


@app.get("/api/sessions")
def list_sessions() -> dict[str, Any]:
    items = []
    for label, sid in sessions.list_choices():
        s = sessions.get(sid)
        items.append(
            {
                "id": sid,
                "title": s.title if s else label,
                "updated_at": s.updated_at if s else 0,
                "message_count": len(s.history) if s else 0,
            }
        )
    return {"sessions": items, "active_id": items[0]["id"] if items else None}


@app.post("/api/sessions")
def create_session(body: SessionCreateRequest | None = None) -> dict[str, Any]:
    title = body.title if body else None
    s = sessions.create(title=title or None)
    return {"id": s.session_id, "title": s.title}


@app.get("/api/sessions/{session_id}")
def get_session(session_id: str) -> dict[str, Any]:
    s = sessions.get(session_id)
    if not s:
        raise HTTPException(404, "会话不存在")
    return {
        "id": s.session_id,
        "title": s.title,
        "history": s.history,
        "consult": s.consult,
        "evidence": s.evidence,
    }


@app.post("/api/chat")
def chat(body: ChatRequest) -> dict[str, Any]:
    message = body.message.strip()
    if not message:
        raise HTTPException(400, "消息不能为空")
    session = sessions.ensure(body.session_id)
    if not session.history:
        sessions.rename_from_first_message(session, message)
    try:
        result = get_pipeline().run_turn(message, session)
        consult = result["consult"]
        tags = []
        if consult.get("retrieved"):
            tags.append("已检索")
        else:
            tags.append("未检索·专科外")
        if consult.get("query_both_sides"):
            tags.append("推荐与限制均保留")
        if consult.get("confidence_label"):
            tags.append(str(consult["confidence_label"]))
        status = "｜".join(tags) if tags else "完成"
        return {
            "session_id": session.session_id,
            "title": session.title,
            "answer": result["answer"],
            "history": session.history,
            "consult": consult,
            "evidence": result["evidence"],
            "meta": {"status": status, "tags": tags},
        }
    except Exception as exc:  # noqa: BLE001
        err = f"调用失败: {exc}\n\n{traceback.format_exc()}"
        session.history = session.history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": err},
        ]
        return {
            "session_id": session.session_id,
            "title": session.title,
            "answer": err,
            "history": session.history,
            "consult": session.consult,
            "evidence": session.evidence,
            "meta": {"error": True, "status": "出错", "tags": ["错误"]},
        }


@app.post("/api/source/open")
def source_open(body: SourceOpenRequest) -> dict[str, Any]:
    try:
        return build_source_info(
            file_path=body.file_path,
            title=body.title,
            quote_text=body.quote,
            chunk_id=body.chunk_id,
        )
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    except PermissionError as exc:
        raise HTTPException(403, str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, f"原文定位失败: {exc}") from exc


@app.get("/api/source/pdf")
def source_pdf(token: str = Query(...)) -> FileResponse:
    try:
        path = path_from_token(token)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    except PermissionError as exc:
        raise HTTPException(403, str(exc)) from exc
    if not path.exists():
        raise HTTPException(404, "文件不存在")
    media = "application/pdf" if path.suffix.lower() == ".pdf" else "application/octet-stream"
    return FileResponse(
        path,
        media_type=media,
        filename=path.name,
        content_disposition_type="inline",
    )


@app.get("/api/source/page.png")
def source_page_png(
    token: str = Query(...),
    page: int = Query(1, ge=1),
    qid: str = Query(""),
) -> Response:
    try:
        path = path_from_token(token)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    except PermissionError as exc:
        raise HTTPException(403, str(exc)) from exc
    if not path.exists():
        raise HTTPException(404, "文件不存在")
    if path.suffix.lower() != ".pdf":
        raise HTTPException(400, "仅支持 PDF 页预览")
    quote = get_quote(qid) if qid else ""
    try:
        png = render_page_png(path, page, quote)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, f"页预览失败: {exc}") from exc
    return Response(content=png, media_type="image/png")


@app.get("/api/source/quote")
def source_quote(qid: str = Query(...)) -> dict[str, Any]:
    text = get_quote(qid)
    if not text:
        raise HTTPException(404, "摘录已过期，请返回重新打开")
    return {"qid": qid, "quote": text}


@app.get("/source/view")
def source_view() -> FileResponse:
    return FileResponse(STATIC / "source_view.html")


if __name__ == "__main__":
    os.chdir(WEB_ROOT)
    uvicorn.run(
        "app:app",
        host="127.0.0.1",
        port=7860,
        reload=False,
        log_level="info",
    )
