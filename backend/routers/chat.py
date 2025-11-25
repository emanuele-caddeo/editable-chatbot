from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List

from backend.services.model_manager import model_manager
from backend.config import DEFAULT_MODEL


class ChatMessage(BaseModel):
    role: str  # "user" o "assistant"
    content: str


class ChatRequest(BaseModel):
    model: str | None = None
    messages: List[ChatMessage]
    max_tokens: int = 256
    temperature: float = 0.7


router = APIRouter(prefix="/api/chat", tags=["chat"])


def build_prompt(messages: List[ChatMessage]) -> str:
    """
    Semplice prompt stile chat:
    User: ...
    Assistant: ...
    """
    lines = []
    for m in messages:
        if m.role == "user":
            lines.append(f"User: {m.content}")
        else:
            lines.append(f"Assistant: {m.content}")
    lines.append("Assistant:")
    return "\n".join(lines)


@router.post("")
def chat(body: ChatRequest):
    model_id = body.model or DEFAULT_MODEL

    try:
        prompt = build_prompt(body.messages)
        response = model_manager.generate(
            repo_id=model_id,
            prompt=prompt,
            max_tokens=body.max_tokens,
            temperature=body.temperature,
        )
        return {"reply": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
