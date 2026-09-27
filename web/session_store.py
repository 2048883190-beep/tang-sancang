# -*- coding: utf-8 -*-
"""多会话内存。进程内有效。"""
from __future__ import annotations

import time
import uuid
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ChatSession:
    session_id: str
    title: str
    history: list[dict[str, str]] = field(default_factory=list)
    consult: dict[str, Any] | None = None
    evidence: list[dict[str, Any]] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def touch(self) -> None:
        self.updated_at = time.time()


class SessionManager:
    def __init__(self) -> None:
        self._sessions: dict[str, ChatSession] = {}

    def create(self, title: str | None = None) -> ChatSession:
        sid = uuid.uuid4().hex[:8]
        session = ChatSession(session_id=sid, title=title or f"对话 {sid}")
        self._sessions[sid] = session
        return session

    def get(self, session_id: str) -> ChatSession | None:
        return self._sessions.get(session_id)

    def ensure(self, session_id: str | None) -> ChatSession:
        if session_id and session_id in self._sessions:
            return self._sessions[session_id]
        return self.create()

    def list_choices(self) -> list[tuple[str, str]]:
        items = sorted(self._sessions.values(), key=lambda s: s.updated_at, reverse=True)
        return [(f"{s.title} ({s.session_id})", s.session_id) for s in items]

    def rename_from_first_message(self, session: ChatSession, message: str) -> None:
        text = (message or "").strip().replace("\n", " ")
        if not text:
            return
        session.title = text[:18] + ("…" if len(text) > 18 else "")
        session.touch()
