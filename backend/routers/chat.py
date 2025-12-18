import os
import json
import asyncio
from typing import List, Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from backend.services.model_manager import model_manager
from backend.config import DEFAULT_MODEL

from backend.ke.ke_parser import parse_ke_command
from backend.ke.ke_models import KnowledgeEdit
from backend.ke.ke_manager import KEManager

router = APIRouter(prefix="/api/chat", tags=["chat"])


# ==========================================================
# DATA MODELS
# ==========================================================

class ChatMessage(BaseModel):
    role: str  # "user" or "assistant"
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
# PROMPT BUILDER (SINGLE TURN)
# ==========================================================

def build_single_turn_prompt(user_message: str) -> str:
    """
    Minimal prompt compatible with GPT-2 / GPT-J.
    """
    text = user_message.strip()
    return f"Provide a short factual answer.\n\n{text}\n\nAnswer:"


# ==========================================================
# KE STATE (IN-MEMORY)
# ==========================================================

_pending_edit: Optional[KnowledgeEdit] = None

ke_manager = KEManager()


# ==========================================================
# SYSTEM MESSAGE HELPER
# ==========================================================

async def send_system(ws: WebSocket, state: str, message: Optional[str] = None):
    payload = {
        "type": "system",
        "state": state,
    }
    if message:
        payload["message"] = message
    await ws.send_json(payload)


# ==========================================================
# MODEL ACCESS HELPERS
# ==========================================================

def _get_model_and_tokenizer(repo_id: str):
    """
    Retrieve model and tokenizer for knowledge editing.
    Load model on-demand if needed.
    """
    if not model_manager.is_loaded(repo_id):
        model_manager.load_model(repo_id)

    try:
        model = model_manager.get_model(repo_id)
        tokenizer = model_manager.get_tokenizer(repo_id)
        return model, tokenizer
    except Exception as e:
        raise RuntimeError(
            f"Cannot retrieve model/tokenizer for '{repo_id}': {e}"
        )


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
            # 0. GLOBAL BUSY GUARD
            # ==================================================
            if ke_manager.is_busy():
                await send_system(ws, "busy", "Knowledge editing in progress")
                await ws.send_json({"type": "done"})
                continue

            # ==================================================
            # 1. KNOWLEDGE EDIT PARSING
            # ==================================================
            edit = parse_ke_command(user_message)

            if edit:
                try:
                    edit.is_valid()
                except ValueError as e:
                    await ws.send_json({
                        "type": "error",
                        "message": f"Error in knowledge edit: {e}"
                    })
                    await ws.send_json({"type": "done"})
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
            # 2. CONFIRM / CANCEL
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
                edit_to_apply = _pending_edit
                _pending_edit = None

                await send_system(ws, "busy", "Applying knowledge edit (ROME)...")

                async def _apply_rome():
                    model, tok = _get_model_and_tokenizer(model_id)
                    ke_manager.apply_edit(
                        model_id=model_id,
                        model=model,
                        tokenizer=tok,
                        edit=edit_to_apply
                    )

                try:
                    await asyncio.to_thread(lambda: asyncio.run(_apply_rome()))
                except RuntimeError:
                    await asyncio.to_thread(lambda: ke_manager.apply_edit(
                        model_id=model_id,
                        model=_get_model_and_tokenizer(model_id)[0],
                        tokenizer=_get_model_and_tokenizer(model_id)[1],
                        edit=edit_to_apply
                    ))
                except Exception as e:
                    await send_system(ws, "ready")
                    await ws.send_json({
                        "type": "error",
                        "message": f"Knowledge edit failed: {e}"
                    })
                    await ws.send_json({"type": "done"})
                    continue

                await send_system(ws, "ready", "Knowledge updated successfully.")
                await ws.send_json({"type": "done"})
                continue

            # ==================================================
            # 3. HISTORY (UI PURPOSE)
            # ==================================================
            hist = load_history()

            if hist.model is None:
                hist.model = model_id

            hist.messages.append(ChatMessage(role="user", content=user_message))
            save_history(hist)

            # ==================================================
            # 4. PROMPT
            # ==================================================
            prompt = build_single_turn_prompt(user_message)

            # ==================================================
            # 5. STREAM GENERATION
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
                        await ws.send_json({
                            "type": "chunk",
                            "content": chunk
                        })

                hist.messages.append(
                    ChatMessage(role="assistant", content=generated)
                )
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


@router.post("/history")
def save_history_endpoint(hist: ChatHistory):
    """
    Save full chat history sent by the frontend.
    """
    save_history(hist)
    return {"status": "saved"}


@router.delete("/history")
def clear_history():
    empty = ChatHistory(model=None, messages=[])
    save_history(empty)
    return {"status": "cleared"}
