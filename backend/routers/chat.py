import os
import json
from typing import List, Optional
from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from backend.services.model_manager import model_manager
from backend.config import DEFAULT_MODEL

router = APIRouter(prefix="/api/chat", tags=["chat"])

# ==========================================================
# MODELLI DATI
# ==========================================================

class ChatMessage(BaseModel):
    role: str      # "user" o "assistant"
    content: str


class ChatHistory(BaseModel):
    model: Optional[str] = None
    messages: List[ChatMessage] = []


# ==========================================================
# FILE HISTORY
# ==========================================================

BACKEND_ROOT = os.path.dirname(os.path.dirname(__file__))
CHATS_DIR = os.path.join(BACKEND_ROOT, "chats")
CHAT_FILE = os.path.join(CHATS_DIR, "chat.json")
os.makedirs(CHATS_DIR, exist_ok=True)


def load_history() -> ChatHistory:
    if not os.path.exists(CHAT_FILE):
        return ChatHistory(model=None, messages=[])

    try:
        with open(CHAT_FILE, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except Exception:
        return ChatHistory(model=None, messages=[])

    if isinstance(raw, dict):
        model = raw.get("model")
        msgs = []
        for m in raw.get("messages", []):
            if isinstance(m, dict) and "role" in m and "content" in m:
                msgs.append(ChatMessage(**m))
        return ChatHistory(model=model, messages=msgs)

    if isinstance(raw, list):
        msgs = [ChatMessage(**m) for m in raw]
        return ChatHistory(model=None, messages=msgs)

    return ChatHistory(model=None, messages=[])


def save_history(hist: ChatHistory):
    with open(CHAT_FILE, "w", encoding="utf-8") as f:
        json.dump(hist.dict(), f, ensure_ascii=False, indent=2)


# ==========================================================
# PROMPT BUILDER — SOLO ULTIMO MESSAGGIO USER
# ==========================================================

def build_single_turn_prompt(user_message: str) -> str:
    """
    Prompt essenziale compatibile con GPT-2 / GPT-J.
    Nessun ruolo, nessuna history concatenata.
    """
    user_message = user_message.strip()
    return f"Q: {user_message}\nA:"


# ==========================================================
# ENDPOINT WEBSOCKET — STREAM CHAT
# ==========================================================

@router.websocket("/stream")
async def chat_stream(ws: WebSocket):
    await ws.accept()

    try:
        while True:
            data = await ws.receive_json()

            user_message = data.get("message")
            model_id = data.get("model") or DEFAULT_MODEL
            compute_mode = data.get("compute_mode")

            if not user_message:
                await ws.send_json({"type": "error", "message": "Missing 'message'"})
                continue

            # ------------------------------
            # 1. Carico history per UI
            # ------------------------------
            hist = load_history()

            # Imposta il modello usato se non presente
            if hist.model is None:
                hist.model = model_id

            # Aggiungo messaggio dell'utente alla history
            hist.messages.append(ChatMessage(role="user", content=user_message))
            save_history(hist)

            # ------------------------------
            # 2. Costruisco il prompt SOLO dal nuovo messaggio
            # ------------------------------
            prompt = build_single_turn_prompt(user_message)

            # ------------------------------
            # 3. Generazione streaming
            # ------------------------------
            try:
                streamer = model_manager.generate_stream(
                    repo_id=hist.model,
                    prompt=prompt,
                    max_tokens=256,
                    temperature=0.7,
                    compute_mode=compute_mode,
                )

                generated = ""

                for chunk in streamer:
                    if chunk:
                        generated += chunk
                        await ws.send_json({"type": "chunk", "content": chunk})

                # Aggiungo la risposta alla history per UI
                hist.messages.append(ChatMessage(role="assistant", content=generated))
                save_history(hist)

                await ws.send_json({"type": "done"})

            except Exception as gen_err:
                await ws.send_json({"type": "error", "message": str(gen_err)})

    except WebSocketDisconnect:
        return


# ==========================================================
# HISTORY ENDPOINTS PER LA UI
# ==========================================================

@router.get("/history")
def get_history():
    return load_history().dict()


@router.delete("/history")
def clear_history():
    empty = ChatHistory(model=None, messages=[])
    save_history(empty)
    return {"status": "cleared"}
