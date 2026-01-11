"""
adapter.py

BP-focused OpenAI-compatible adapter endpoint:
  POST /v1/chat/completions

What it does:
- Accepts Beyond Presence (BP) OpenAI-style requests (AsyncOpenAI client)
- Extracts the last user message
- Calls Lili upstream: POST {LILI_API_BASE}/user-scope/website-chat/
- Returns OpenAI-style JSON (stream=false) or SSE (stream=true)

Design choices for BP reliability:
- BP uses stream=true; we do NOT attempt to proxy Lili streaming format.
  Instead we call Lili non-streaming and emit exactly one SSE content chunk.
  This avoids “role-only” streams (silent avatar) when upstream streaming format differs.
- Stable sender_id per “session” using x-forwarded-for + user-agent (since BP does not send body.user).
"""

import hashlib
import json
import logging
import os
import time
import uuid
from typing import Any, Dict, List, Optional

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

app = FastAPI(title="BP PoC Adapter", version="1.1.0")

logger = logging.getLogger("uvicorn.error")

# -----------------------------
# Config
# -----------------------------
LILI_API_BASE = os.getenv("LILI_API_BASE", "https://backend-lili-demo.limitless-tech.ai/api").rstrip("/")
LILI_ENDPOINT = f"{LILI_API_BASE}/user-scope/website-chat/"

# IMPORTANT:
# Railway env vars override code defaults. If you set LILI_WORKFLOW_ID in Railway,
# it will take precedence over what you put here.
LILI_WORKFLOW_ID = os.getenv("LILI_WORKFLOW_ID", "56")

# Adapter -> Lili timeout
LILI_TIMEOUT_SECONDS = float(os.getenv("LILI_TIMEOUT_SECONDS", "60"))

# -----------------------------
# OpenAI-like request models
# -----------------------------
class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="allow")
    role: str
    content: Any


class ChatCompletionsRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
    model: str = "lili-workflow"
    messages: List[ChatMessage] = Field(min_length=1)
    stream: bool = False
    user: Optional[str] = None  # BP currently sends None (per your logs)


# -----------------------------
# Helpers
# -----------------------------
def _extract_text_from_content(content: Any) -> str:
    """
    Supports common OpenAI formats:
    - "hello"
    - [{"type":"text","text":"hello"}]
    - {"text":"hello"} / {"content":"hello"} (defensive)
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()

    if isinstance(content, dict):
        for k in ("text", "content", "value", "message"):
            v = content.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()
        return ""

    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, str) and item.strip():
                parts.append(item.strip())
            elif isinstance(item, dict):
                v = item.get("text") or item.get("content") or item.get("value")
                if isinstance(v, str) and v.strip():
                    parts.append(v.strip())
        return "\n".join(parts).strip()

    return str(content).strip()


def last_user_message(messages: List[ChatMessage]) -> str:
    for m in reversed(messages):
        if (m.role or "").lower() == "user":
            txt = _extract_text_from_content(m.content)
            if txt:
                return txt
    return ""


def stable_sender_id(body: ChatCompletionsRequest, request: Request) -> str:
    """
    BP does not send body.user (you confirmed BODY.user=None), so we derive a stable id.
    Use x-forwarded-for (client IP as seen by Railway edge) + user-agent.

    This is "session-ish" stability for BP server-to-server calls; it won't map perfectly
    to a human end-user unless BP provides a conversation id header.
    """
    # 1) If BP ever sends body.user, use it.
    if body.user and body.user.strip():
        return body.user.strip()

    h = request.headers

    # 2) If BP provides any stable conversation/session header in the future, prefer it.
    for key in (
        "x-session-id",
        "x-call-id",
        "x-conversation-id",
        "x-livekit-room",
        "x-livekit-room-name",
    ):
        v = h.get(key)
        if v and v.strip():
            return v.strip()

    # 3) Last resort: x-forwarded-for + user-agent
    xff = h.get("x-forwarded-for", "")
    client_ip = xff.split(",")[0].strip() if xff else "unknown"
    ua = h.get("user-agent", "unknown")

    fp = f"{client_ip}|{ua}"
    return hashlib.sha256(fp.encode("utf-8")).hexdigest()[:32]


def _empty_chat_completion(model: str) -> Dict[str, Any]:
    created = int(time.time())
    stream_id = f"chatcmpl-{uuid.uuid4().hex}"
    return {
        "id": stream_id,
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": ""}, "finish_reason": "stop"}],
    }


def _openai_sse_event(stream_id: str, created: int, model: str, delta: Dict[str, Any], finish_reason: Optional[str]):
    return {
        "id": stream_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }


# -----------------------------
# Diagnostics endpoints
# -----------------------------
@app.get("/health")
def health():
    return {
        "ok": True,
        "lili_endpoint": LILI_ENDPOINT,
        "workflow_id": LILI_WORKFLOW_ID,
    }


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    try:
        raw = await request.body()
        logger.error(
            "HTTPException %s on %s detail=%s body=%s",
            exc.status_code,
            request.url.path,
            exc.detail,
            raw.decode("utf-8", errors="replace"),
        )
    except Exception:
        logger.error("HTTPException %s on %s detail=%s (could not read body)", exc.status_code, request.url.path, exc.detail)

    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})


# -----------------------------
# Main endpoint: BP -> Adapter -> Lili -> Adapter -> BP
# -----------------------------
@app.post("/v1/chat/completions")
async def chat_completions(body: ChatCompletionsRequest, request: Request):
    # Useful debug, but keep it minimal
    logger.info("BP request stream=%s", body.stream)
    logger.info("headers.user-agent=%r x-forwarded-for=%r", request.headers.get("user-agent"), request.headers.get("x-forwarded-for"))
    logger.info("BODY.user=%r keys=%s", body.user, list(body.model_dump().keys()))

    user_text = last_user_message(body.messages).strip()
    logger.info("extracted user_text=%r", user_text)

    # BP sometimes sends warm-ups with empty content. Do NOT 400.
    if not user_text:
        if not body.stream:
            return JSONResponse(_empty_chat_completion(body.model))

        async def empty_sse():
            created = int(time.time())
            stream_id = f"chatcmpl-{uuid.uuid4().hex}"

            yield f"data: {json.dumps(_openai_sse_event(stream_id, created, body.model, {'role': 'assistant'}, None), ensure_ascii=False)}\n\n"
            yield f"data: {json.dumps(_openai_sse_event(stream_id, created, body.model, {}, 'stop'), ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(
            empty_sse(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
        )

    sender_id = stable_sender_id(body, request)

    # IMPORTANT:
    # We intentionally call Lili non-streaming always, then adapt to BP stream/non-stream.
    lili_payload: Dict[str, Any] = {
        "workflow_id": str(LILI_WORKFLOW_ID),
        "sender_id": sender_id,
        "user_message": user_text,
        "stream": False,  # keep upstream simple and deterministic for BP
    }

    logger.info("about to call Lili: workflow_id=%s sender_id=%s stream=%s", LILI_WORKFLOW_ID, sender_id, body.stream)

    # Call Lili
    try:
        async with httpx.AsyncClient(timeout=LILI_TIMEOUT_SECONDS) as client:
            resp = await client.post(
                LILI_ENDPOINT,
                json=lili_payload,
                headers={"Accept": "application/json", "Content-Type": "application/json"},
            )
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=f"Upstream Lili request failed: {str(e)}") from e

    if resp.status_code >= 400:
        raise HTTPException(status_code=502, detail=f"Lili error {resp.status_code}: {resp.text}")

    try:
        data = resp.json()
    except Exception:
        raise HTTPException(status_code=502, detail=f"Lili returned non-JSON: {resp.text}")

    assistant_text = (data.get("message") or data.get("error") or "").strip()

    created = int(time.time())
    stream_id = f"chatcmpl-{uuid.uuid4().hex}"

    # Non-streaming OpenAI response
    if not body.stream:
        return JSONResponse(
            {
                "id": stream_id,
                "object": "chat.completion",
                "created": created,
                "model": body.model,
                "choices": [
                    {"index": 0, "message": {"role": "assistant", "content": assistant_text}, "finish_reason": "stop"}
                ],
            }
        )

    # Streaming OpenAI SSE response (BP expects deltas)
    async def sse_gen():
        # Role chunk
        yield f"data: {json.dumps(_openai_sse_event(stream_id, created, body.model, {'role': 'assistant'}, None), ensure_ascii=False)}\n\n"

        # Content chunk (single chunk; enough for BP to speak reliably)
        if assistant_text:
            logger.info("emit_delta len=%d preview=%r", len(assistant_text), assistant_text[:80])
            yield f"data: {json.dumps(_openai_sse_event(stream_id, created, body.model, {'content': assistant_text}, None), ensure_ascii=False)}\n\n"

        # Stop + DONE
        yield f"data: {json.dumps(_openai_sse_event(stream_id, created, body.model, {}, 'stop'), ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        sse_gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )
