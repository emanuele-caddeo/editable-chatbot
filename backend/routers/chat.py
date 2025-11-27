from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
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
    """
    Endpoint non-streaming (fallback / compatibilità).
    """
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


@router.websocket("/stream")
async def chat_stream(ws: WebSocket):
    """
    WebSocket che streamma la risposta token-by-token.
    Il client invia:
      { model, messages, max_tokens, temperature }
    e riceve:
      { "type": "chunk", "content": "...pezzo..." }
      ...
      { "type": "done" }
    oppure:
      { "type": "error", "message": "..." }
    """
    await ws.accept()
    try:
        while True:
            data = await ws.receive_json()
            model_id = data.get("model") or DEFAULT_MODEL
            raw_messages = data.get("messages", [])
            max_tokens = int(data.get("max_tokens", 256))
            temperature = float(data.get("temperature", 0.7))

            # converto in ChatMessage per riusare build_prompt
            messages = [ChatMessage(**m) for m in raw_messages]
            prompt = build_prompt(messages)

            try:
                streamer = model_manager.generate_stream(
                    repo_id=model_id,
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )

                # invio i chunk man mano che arrivano
                for text_chunk in streamer:
                    if text_chunk:
                        await ws.send_json({"type": "chunk", "content": text_chunk})

                await ws.send_json({"type": "done"})
            except Exception as gen_err:
                await ws.send_json(
                    {"type": "error", "message": str(gen_err)}
                )
    except WebSocketDisconnect:
        # il client ha chiuso la connessione
        return
