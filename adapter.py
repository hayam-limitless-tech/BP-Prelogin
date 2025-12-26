"""
adapter.py

OpenAI-compatible /v1/chat/completions adapter in front of Lili's
POST /api/user-scope/website-chat/ endpoint.

Supports:
- Non-streaming JSON response
- Streaming SSE response (adapter-level streaming by chunking final text)

Environment variables:
- LILI_API_BASE (default: https://backend-lili-demo.limitless-tech.ai/api)
- LILI_WORKFLOW_ID (default: 213)
- LILI_TIMEOUT_SECONDS (default: 60)
- STREAM_CHUNK_SIZE (default: 40)  # characters per SSE chunk
"""

import json
import os
import time
import uuid
from typing import List, Literal, Optional

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

app = FastAPI(title="BP PoC Adapter", version="1.0.0")

LILI_API_BASE = os.getenv("LILI_API_BASE", "https://backend-lili-demo.limitless-tech.ai/api").rstrip("/")
LILI_WORKFLOW_ID = os.getenv("LILI_WORKFLOW_ID", "213")
LILI_TIMEOUT_SECONDS = float(os.getenv("LILI_TIMEOUT_SECONDS", "60"))
STREAM_CHUNK_SIZE = int(os.getenv("STREAM_CHUNK_SIZE", "40"))

LILI_ENDPOINT = f"{LILI_API_BASE}/user-scope/website-chat/"


# -----------------------------
# Pydantic models (OpenAI-like)
# -----------------------------
class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatCompletionsRequest(BaseModel):
    model: str = "lili-workflow"
    messages: List[ChatMessage] = Field(min_length=1)
    stream: bool = False
    user: Optional[str] = None  # optional stable session id


# -----------------------------
# Helpers
# -----------------------------
def last_user_message(messages: List[ChatMessage]) -> str:
    for m in reversed(messages):
        if m.role == "user" and m.content:
            return m.content
    return ""


def chunk_text(text: str, chunk_size: int) -> List[str]:
    if chunk_size <= 0:
        return [text]
    return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]


# -----------------------------
# Health check
# -----------------------------
@app.get("/health")
def health():
    return {
        "ok": True,
        "lili_endpoint": LILI_ENDPOINT,
        "workflow_id": LILI_WORKFLOW_ID,
    }


# -----------------------------
# Main endpoint
# -----------------------------
@app.post("/v1/chat/completions")
async def chat_completions(body: ChatCompletionsRequest):
    user_text = last_user_message(body.messages).strip()
    if not user_text:
        raise HTTPException(status_code=400, detail="No user message found in messages[]")

    # If you want stable conversation memory in Lili across turns,
    # pass a consistent `user` from the caller; otherwise this will be random per request.
    sender_id = (body.user or str(uuid.uuid4())).strip()

    payload = {
        "workflow_id": str(LILI_WORKFLOW_ID),
        "sender_id": sender_id,
        "user_message": user_text,
    }

    # Call Lili (non-streaming upstream)
    try:
        async with httpx.AsyncClient(timeout=LILI_TIMEOUT_SECONDS) as client:
            resp = await client.post(
                LILI_ENDPOINT,
                json=payload,
                headers={
                    "Accept": "application/json",
                    "Content-Type": "application/json",
                },
            )
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=f"Upstream Lili request failed: {str(e)}") from e

    if resp.status_code >= 400:
        # Return a 502 because the adapter is functioning but upstream failed
        raise HTTPException(status_code=502, detail=f"Lili error {resp.status_code}: {resp.text}")

    # Parse response
    try:
        data = resp.json()
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Lili returned non-JSON: {resp.text}") from e

    assistant_text = (data.get("message") or data.get("error") or "").strip()

    created = int(time.time())
    stream_id = f"chatcmpl-{uuid.uuid4().hex}"

    # Non-streaming response
    if not body.stream:
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

    # Streaming SSE response (adapter-level streaming)
    async def sse_gen():
        # Some clients like to see role once; harmless to include in first event.
        first = True
        for part in chunk_text(assistant_text, STREAM_CHUNK_SIZE):
            delta_obj = {"content": part}
            if first:
                delta_obj["role"] = "assistant"
                first = False

            event = {
                "id": stream_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": body.model,
                "choices": [
                    {
                        "index": 0,
                        "delta": delta_obj,
                        "finish_reason": None,
                    }
                ],
            }
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

        # Final stop chunk
        final_event = {
            "id": stream_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": body.model,
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop",
                }
            ],
        }
        yield f"data: {json.dumps(final_event)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(sse_gen(), media_type="text/event-stream")
