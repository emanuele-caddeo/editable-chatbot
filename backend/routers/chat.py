import os
import json
from typing import List, Optional

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from backend.services.model_manager import model_manager
from backend.config import DEFAULT_MODEL


# ============================
#  MODELLI DATI
# ============================

class ChatMessage(BaseModel):
    role: str  # "user" o "assistant"
    content: str


class ChatHistory(BaseModel):
    model: Optional[str] = None
    messages: List[ChatMessage] = []


class ChatRequest(BaseModel):
    """Endpoint POST non-streaming (se mai ti serve)."""
    model: Optional[str] = None
    message: str  # singolo messaggio utente
    max_tokens: int = 256
    temperature: float = 0.7


router = APIRouter(prefix="/api/chat", tags=["chat"])


# ============================
#  FILE JSON DI HISTORY
# ============================

BACKEND_ROOT = os.path.dirname(os.path.dirname(__file__))
CHATS_DIR = os.path.join(BACKEND_ROOT, "chats")
CHAT_FILE = os.path.join(CHATS_DIR, "chat.json")

os.makedirs(CHATS_DIR, exist_ok=True)


def load_history() -> ChatHistory:
    """Carica la history da chat.json. Gestisce sia vecchio formato (lista) sia nuovo."""
    if not os.path.exists(CHAT_FILE):
        return ChatHistory(model=None, messages=[])

    try:
        with open(CHAT_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return ChatHistory(model=None, messages=[])

    # Vecchio formato: lista di messaggi
    if isinstance(data, list):
        msgs = [ChatMessage(**m) for m in data if isinstance(m, dict)]
        return ChatHistory(model=None, messages=msgs)

    # Nuovo formato: { "model": ..., "messages": [...] }
    if isinstance(data, dict):
        model_id = data.get("model")
        raw_msgs = data.get("messages", [])
        msgs: List[ChatMessage] = []
        if isinstance(raw_msgs, list):
            for m in raw_msgs:
                if isinstance(m, dict) and "role" in m and "content" in m:
                    msgs.append(ChatMessage(**m))
        return ChatHistory(model=model_id, messages=msgs)

    return ChatHistory(model=None, messages=[])


def save_history(hist: ChatHistory) -> None:
    """Salva la history nel formato {model, messages}."""
    os.makedirs(CHATS_DIR, exist_ok=True)
    with open(CHAT_FILE, "w", encoding="utf-8") as f:
        json.dump(hist.dict(), f, ensure_ascii=False, indent=2)


# ============================
#  COSTRUZIONE PROMPT
# ============================

def build_prompt(messages: List[ChatMessage]) -> str:
    lines = []
    for m in messages:
        if m.role == "user":
            lines.append(f"User: {m.content}")
        else:
            lines.append(f"Assistant: {m.content}")
    lines.append("Assistant:")
    return "\n".join(lines)


# ============================
#  ENDPOINT POST NON-STREAMING
# ============================

@router.post("")
def chat(body: ChatRequest):
    """
    Endpoint non-streaming di fallback.
    Usa comunque la history salvata su file come memoria.
    """
    hist = load_history()

    model_id = body.model or hist.model or DEFAULT_MODEL
    if hist.model is None:
        hist.model = model_id

    # aggiungo messaggio utente
    hist.messages.append(ChatMessage(role="user", content=body.message))

    try:
        prompt = build_prompt(hist.messages)
        reply = model_manager.generate(
            repo_id=hist.model,
            prompt=prompt,
            max_tokens=body.max_tokens,
            temperature=body.temperature,
            compute_mode=None,
        )

        # aggiungo risposta alla history
        hist.messages.append(ChatMessage(role="assistant", content=reply))
        save_history(hist)

        return {"reply": reply}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================
#  WEBSOCKET STREAMING
# ============================

@router.websocket("/stream")
async def chat_stream(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            data = await ws.receive_json()

            user_message = data.get("message")
            model_id = data.get("model") or DEFAULT_MODEL
            compute_mode = data.get("compute_mode")

            if not isinstance(user_message, str) or not user_message.strip():
                await ws.send_json({"type": "error", "message": "Missing 'message' field"})
                continue

            # carica history ufficiale
            hist = load_history()

            # se non c'è ancora un modello, prendo quello corrente
            if hist.model is None:
                hist.model = model_id

            # aggiungo messaggio utente
            hist.messages.append(ChatMessage(role="user", content=user_message))

            prompt = build_prompt(hist.messages)

            try:
                streamer = model_manager.generate_stream(
                    repo_id=hist.model,
                    prompt=prompt,
                    max_tokens=256,
                    temperature=0.7,
                    compute_mode=compute_mode,
                )

                generated = ""

                for text_chunk in streamer:
                    if text_chunk:
                        generated += text_chunk
                        await ws.send_json({"type": "chunk", "content": text_chunk})

                # aggiungo risposta alla history
                hist.messages.append(ChatMessage(role="assistant", content=generated))
                save_history(hist)

                await ws.send_json({"type": "done"})
            except Exception as gen_err:
                await ws.send_json({"type": "error", "message": str(gen_err)})

    except WebSocketDisconnect:
        return


# ============================
#  HISTORY: GET / DELETE
# ============================

@router.get("/history")
def get_history():
    """Restituisce { model, messages }."""
    return load_history().dict()


@router.delete("/history")
def clear_history():
    """Svuota la chat salvata."""
    hist = ChatHistory(model=None, messages=[])
    save_history(hist)
    return {"status": "cleared"}
