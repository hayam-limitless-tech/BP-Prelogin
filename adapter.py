import os
import uuid
from typing import Any, Dict, List, Optional

import httpx
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse

app = FastAPI()

LILI_API_BASE = os.getenv("LILI_API_BASE", "https://backend-lili-demo.limitless-tech.ai/api")
LILI_WORKFLOW_ID = os.getenv("LILI_WORKFLOW_ID", "213")
LILI_ENDPOINT = f"{LILI_API_BASE}/user-scope/website-chat/"


def get_last_user_message(messages: List[Dict[str, Any]]) -> str:
    # OpenAI messages: [{"role":"user","content":"..."}...]
    for m in reversed(messages):
        if m.get("role") == "user" and isinstance(m.get("content"), str):
            return m["content"]
    return ""


@app.post("/v1/chat/completions")
async def chat_completions(req: Request):
    body = await req.json()

    messages = body.get("messages", [])
    if not isinstance(messages, list) or len(messages) == 0:
        raise HTTPException(status_code=400, detail="messages must be a non-empty array")

    user_text = get_last_user_message(messages)
    if not user_text:
        raise HTTPException(status_code=400, detail="No user message found")

    # Keep sender_id stable across a single call; for multi-turn, you may map conversation/session -> sender_id.
    sender_id = body.get("user") or str(uuid.uuid4())

    payload = {
        "workflow_id": LILI_WORKFLOW_ID,
        "sender_id": sender_id,
        "user_message": user_text,
    }

    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(LILI_ENDPOINT, json=payload, headers={
            "Accept": "application/json",
            "Content-Type": "application/json",
        })

    if r.status_code >= 400:
        raise HTTPException(status_code=502, detail=f"Lili error {r.status_code}: {r.text}")

    data = r.json()
    assistant_text = data.get("message") or data.get("error") or ""

    # OpenAI-compatible response (non-streaming)
    response = {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(__import__("time").time()),
        "model": body.get("model", "lili-workflow-213"),
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": assistant_text},
                "finish_reason": "stop",
            }
        ],
    }
    return JSONResponse(response)
