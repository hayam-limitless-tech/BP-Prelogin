import json
import time
import uuid
from typing import Any, Dict, List, Optional
import os
import logging

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

app = FastAPI(title="BP PoC Adapter", version="1.0.3")

# -----------------------------
# Config
# -----------------------------
LILI_API_BASE = os.getenv(
    "LILI_API_BASE",
    "https://backend-lili-demo.limitless-tech.ai/api"
).rstrip("/")

LILI_WORKFLOW_ID = os.getenv("LILI_WORKFLOW_ID", "213")
LILI_TIMEOUT_SECONDS = float(os.getenv("LILI_TIMEOUT_SECONDS", "60"))

LILI_ENDPOINT = f"{LILI_API_BASE}/user-scope/website-chat/"

logger = logging.getLogger("uvicorn.error")


# -----------------------------
# Models
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
    user: Optional[str] = None


# -----------------------------
# Helpers
# -----------------------------
def _extract_text_from_content(content: Any) -> str:
    """
    Supports:
    - "text"
    - [{"type":"text","text":"..."}]
    - {"text":"..."} / {"content":"..."} (defensive)
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


def _empty_chat_completion(model: str) -> Dict[str, Any]:
    created = int(time.time())
    stream_id = f"chatcmpl-{uuid.uuid4().hex}"
    return {
        "id": stream_id,
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": ""},
                "finish_reason": "stop",
            }
        ],
    }


# -----------------------------
# Exception logging (keeps Railway logs useful)
# -----------------------------
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    try:
        raw = await request.body()
        logger.error(
            f"HTTPException {exc.status_code} on {request.url.path} "
            f"detail={exc.detail} body={raw.decode('utf-8', errors='replace')}"
        )
    except Exception:
        logger.error(
            f"HTTPException {exc.status_code} on {request.url.path} detail={exc.detail} (could not read body)"
        )
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})


# -----------------------------
# Main endpoint
# -----------------------------
@app.post("/v1/chat/completions")
async def chat_completions(body: ChatCompletionsRequest):
    print("BP request stream =", body.stream)

    user_text = last_user_message(body.messages).strip()

    
    if not user_text:
        if not body.stream:
            return JSONResponse(_empty_chat_completion(body.model))

        async def empty_sse():
            created = int(time.time())
            stream_id = f"chatcmpl-{uuid.uuid4().hex}"

            init_event = {
                "id": stream_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": body.model,
                "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
            }
            yield f"data: {json.dumps(init_event, ensure_ascii=False)}\n\n"

            final_event = {
                "id": stream_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": body.model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }
            yield f"data: {json.dumps(final_event, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(
            empty_sse(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    sender_id = (body.user or str(uuid.uuid4())).strip()

    payload: Dict[str, Any] = {
        "workflow_id": str(LILI_WORKFLOW_ID),
        "sender_id": sender_id,
        "user_message": user_text,
        # Ask Lili to stream if BP asked us to stream
        "stream": bool(body.stream),  
    }

    created = int(time.time())
    stream_id = f"chatcmpl-{uuid.uuid4().hex}"

    # -----------------------------
    # Streaming response (proxy + fallback)
    # -----------------------------
    async def sse_proxy():
        # 1) role chunk
        init_event = {
            "id": stream_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": body.model,
            "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
        }
        yield f"data: {json.dumps(init_event, ensure_ascii=False)}\n\n"

        sent_any_content = False

        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream("POST", LILI_ENDPOINT, json=payload) as resp:
                if resp.status_code >= 400:
                    err_text = await resp.aread()
                    raise HTTPException(
                        status_code=502,
                        detail=f"Lili error {resp.status_code}: {err_text.decode(errors='replace')}",
                    )

                # Attempt to parse streamed lines from Lili (SSE, NDJSON, or plain tokens)
                async for line in resp.aiter_lines():
                    if not line:
                        continue

                    # If Lili returns SSE like: "data: {...}"
                    if line.startswith("data:"):
                        line = line[len("data:"):].strip()

                    # Common terminators
                    if line in ("[DONE]", "DONE"):
                        break

                    token: Optional[str] = None

                    # Try JSON decode first
                    try:
                        obj = json.loads(line)
                        if isinstance(obj, dict):
                            token = obj.get("delta") or obj.get("token") or obj.get("text") or obj.get("content")
                        elif isinstance(obj, str):
                            token = obj
                    except Exception:
                        # plain text token
                        token = line

                    if not token:
                        continue

                    event = {
                        "id": stream_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": body.model,
                        "choices": [{"index": 0, "delta": {"content": token}, "finish_reason": None}],
                    }
                    sent_any_content = True
                    yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

        # Fallback: if Lili streaming produced no tokens, do a non-streaming call and emit one content chunk.
        if not sent_any_content:
            fallback_payload = dict(payload)
            fallback_payload["stream"] = False

            try:
                async with httpx.AsyncClient(timeout=LILI_TIMEOUT_SECONDS) as c2:
                    r2 = await c2.post(LILI_ENDPOINT, json=fallback_payload)
            except httpx.RequestError:
                r2 = None

            assistant_text = ""
            if r2 is not None and r2.status_code < 400:
                try:
                    d2 = r2.json()
                    assistant_text = (d2.get("message") or d2.get("error") or "").strip()
                except Exception:
                    assistant_text = ""

            if assistant_text:
                fallback_event = {
                    "id": stream_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": body.model,
                    "choices": [{"index": 0, "delta": {"content": assistant_text}, "finish_reason": None}],
                }
                yield f"data: {json.dumps(fallback_event, ensure_ascii=False)}\n\n"
                sent_any_content = True

        # 3) stop chunk + DONE
        final_event = {
            "id": stream_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": body.model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }
        yield f"data: {json.dumps(final_event, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        sse_proxy(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
