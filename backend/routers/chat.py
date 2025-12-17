import os
import json
from typing import List, Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from backend.services.model_manager import model_manager
from backend.config import DEFAULT_MODEL

from backend.ke.ke_parser import parse_ke_command
from backend.ke.ke_models import KnowledgeEdit


router = APIRouter(prefix="/api/chat", tags=["chat"])

# ==========================================================
# DATA MODELS
# ==========================================================

class ChatMessage(BaseModel):
    role: str      # "user" or "assistant"
    content: str

class ChatHistory(BaseModel):
    model: Optional[str] = None
    messages: List[ChatMessage] = []


# ==========================================================
# HISTORY FILE
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
# PROMPT BUILDER — USER LAST MESSAGE ONLY
# ==========================================================

def build_single_turn_prompt(user_message: str) -> str:
    """
    Minimal prompt compatible with GPT-2 / GPT-J.
    """
    text = user_message.strip()
    return f"Provide a short factual answer.\n\n{text}\n\nAnswer:"


# ==========================================================
# KE STATE (TEMPORARY, IN-MEMORY)
# ==========================================================

# NOTE:
# This will be moved to ke_manager.py (STEP 3)
_pending_edit: Optional[KnowledgeEdit] = None


# ==========================================================
# WEBSOCKET STREAM
# ==========================================================

@router.websocket("/stream")
async def chat_stream(ws: WebSocket):
    global _pending_edit

    await ws.accept()

    try:
        while True:
            data = await ws.receive_json()

            user_message = data.get("message")
            model_id = data.get("model") or DEFAULT_MODEL
            compute_mode = data.get("compute_mode")

            temperature = data.get("temperature", 0.7)
            max_tokens = data.get("max_tokens", 256)
            top_p = data.get("top_p", 0.9)
            top_k = data.get("top_k", 50)
            repetition_penalty = data.get("repetition_penalty", 1.1)

            if not user_message:
                await ws.send_json({"type": "error", "message": "Missing 'message'"})
                continue

            # ==================================================
            # 0. KE COMMAND PARSING
            # ==================================================

            edit = parse_ke_command(user_message)

            if edit:
                try:
                    edit.is_valid()
                except ValueError as e:
                    await ws.send_json({
                        "type": "done",
                        "content": f"❌ Error in knowledge edit: {e}"
                    })
                    continue

                _pending_edit = edit

                await ws.send_json({
                    "type": "chunk",
                    "content": (
                        "⚠️ You are about to modify the model knowledge:\n\n"
                        f"{edit.render()}\n\n"
                        "Type 'confirm' to apply or 'cancel' to discard."
                    )
                })

                await ws.send_json({"type": "done"})
                continue


            # ==================================================
            # 0.1 CONFIRM / CANCEL (placeholder)
            # ==================================================

            if user_message.lower() == "cancel" and _pending_edit:
                _pending_edit = None
                await ws.send_json({
                    "type": "chunk",
                    "content": "❎ Knowledge edit cancelled."
                })
                await ws.send_json({"type": "done"})
                continue


            if user_message.lower() == "confirm" and _pending_edit:
                # KE MANAGER will be integrated here (STEP 3)
                await ws.send_json({
                    "type": "chunk",
                    "content": (
                        "✅ Knowledge edit confirmed.\n"
                        "(ROME application will be integrated in the next step)"
                    )
                })
                await ws.send_json({"type": "done"})
                continue

            # ==================================================
            # 1. HISTORY (UI ONLY)
            # ==================================================

            hist = load_history()

            if hist.model is None:
                hist.model = model_id

            hist.messages.append(ChatMessage(role="user", content=user_message))
            save_history(hist)

            # ==================================================
            # 2. PROMPT
            # ==================================================

            prompt = build_single_turn_prompt(user_message)

            # ==================================================
            # 3. STREAM GENERATION
            # ==================================================

            try:
                streamer = model_manager.generate_stream(
                    repo_id=hist.model,
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    repetition_penalty=repetition_penalty,
                    compute_mode=compute_mode,
                )

                generated = ""

                for chunk in streamer:
                    if chunk:
                        generated += chunk
                        await ws.send_json({"type": "chunk", "content": chunk})

                hist.messages.append(ChatMessage(role="assistant", content=generated))
                save_history(hist)

                await ws.send_json({"type": "done"})

            except Exception as gen_err:
                await ws.send_json({
                    "type": "error",
                    "message": str(gen_err)
                })

    except WebSocketDisconnect:
        return


# ==========================================================
# HISTORY ENDPOINTS
# ==========================================================

@router.get("/history")
def get_history():
    return load_history().dict()

@router.delete("/history")
def clear_history():
    empty = ChatHistory(model=None, messages=[])
    save_history(empty)
    return {"status": "cleared"}
