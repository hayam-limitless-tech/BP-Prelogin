import json
import time
import uuid
from typing import Any, Dict, List, Optional
import os

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

app = FastAPI(title="BP PoC Adapter", version="1.0.2")

LILI_API_BASE = os.getenv(
    "LILI_API_BASE",
    "https://backend-lili-demo.limitless-tech.ai/api"
).rstrip("/")

LILI_WORKFLOW_ID = os.getenv("LILI_WORKFLOW_ID", "213")
LILI_TIMEOUT_SECONDS = float(os.getenv("LILI_TIMEOUT_SECONDS", "60"))
STREAM_CHUNK_SIZE = int(os.getenv("STREAM_CHUNK_SIZE", "40"))

LILI_ENDPOINT = f"{LILI_API_BASE}/user-scope/website-chat/"


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


def _extract_text_from_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "\n".join(parts).strip()
    return str(content).strip()


def last_user_message(messages: List[ChatMessage]) -> str:
    for m in reversed(messages):
        if (m.role or "").lower() == "user":
            txt = _extract_text_from_content(m.content)
            if txt:
                return txt
    return ""


@app.post("/v1/chat/completions")
async def chat_completions(body: ChatCompletionsRequest):
    print("BP request stream =", body.stream)
    user_text = last_user_message(body.messages).strip()
    if not user_text:
        # BP sometimes calls with an empty user turn (preflight / warm-up).
        # Return a valid empty completion rather than 400.
        created = int(time.time())
        stream_id = f"chatcmpl-{uuid.uuid4().hex}"

        if not body.stream:
            return JSONResponse({
                "id": stream_id,
                "object": "chat.completion",
                "created": created,
                "model": body.model,
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": ""},
                    "finish_reason": "stop",
                }],
            })

        async def empty_sse():
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

        return StreamingResponse(empty_sse(), media_type="text/event-stream")


    sender_id = (body.user or str(uuid.uuid4())).strip()

    payload: Dict[str, Any] = {
        "workflow_id": "213",          # or LILI_WORKFLOW_ID
        "sender_id": sender_id,
        "user_message": user_text,
        # IMPORTANT: turn on streaming upstream
        "stream": bool(body.stream),   # rename to "streaming" if Lili expects that
    }

    created = int(time.time())
    stream_id = f"chatcmpl-{uuid.uuid4().hex}"

    # Non-streaming path
    if not body.stream:
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(LILI_ENDPOINT, json=payload)
        if resp.status_code >= 400:
            raise HTTPException(status_code=502, detail=f"Lili error {resp.status_code}: {resp.text}")
        data = resp.json()
        assistant_text = (data.get("message") or data.get("error") or "").strip()

        return JSONResponse(
            {
                "id": stream_id,
                "object": "chat.completion",
                "created": created,
                "model": body.model,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": assistant_text},
                        "finish_reason": "stop",
                    }
                ],
            }
        )

    # Streaming path (proxy)
    async def sse_proxy():
        # Emit an initial chunk with role (many clients tolerate/like this)
        init_event = {
            "id": stream_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": body.model,
            "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
        }
        yield f"data: {json.dumps(init_event, ensure_ascii=False)}\n\n"

        async with httpx.AsyncClient(timeout=None) as client:
            # NOTE: use .stream() to avoid buffering the whole response
            async with client.stream("POST", LILI_ENDPOINT, json=payload) as resp:
                if resp.status_code >= 400:
                    err_text = await resp.aread()
                    raise HTTPException(status_code=502, detail=f"Lili error {resp.status_code}: {err_text.decode(errors='replace')}")

                # The key part: iterate incremental data from Lili
                # This assumes Lili returns SSE or line-delimited JSON/text.
                async for line in resp.aiter_lines():
                    if not line:
                        continue

                    # If Lili returns SSE lines like: "data: {...}"
                    if line.startswith("data:"):
                        line = line[len("data:"):].strip()

                    # Common terminators
                    if line in ("[DONE]", "DONE"):
                        break

                    # Try to parse JSON. If it’s plain text token, fall back to raw.
                    token = None
                    try:
                        obj = json.loads(line)
                        # Adjust these keys to match Lili’s streaming format:
                        # e.g. {"delta":"مرحباً"} or {"token":"..."} or {"text":"..."}
                        token = obj.get("delta") or obj.get("token") or obj.get("text")
                        if token is None and isinstance(obj, str):
                            token = obj
                    except Exception:
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
                    yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

        # Final stop chunk + DONE
        final_event = {
            "id": stream_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": body.model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }
        yield f"data: {json.dumps(final_event, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(sse_proxy(), media_type="text/event-stream")

import logging
from fastapi.responses import JSONResponse
from fastapi import HTTPException, Request

logger = logging.getLogger("uvicorn.error")

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    # Logs full context for Railway
    try:
        raw = await request.body()
        logger.error(f"HTTPException {exc.status_code} on {request.url.path} detail={exc.detail} body={raw.decode('utf-8', errors='replace')}")
    except Exception:
        logger.error(f"HTTPException {exc.status_code} on {request.url.path} detail={exc.detail} (could not read body)")
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})
