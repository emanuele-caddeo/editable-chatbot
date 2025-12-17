import os
import json
import asyncio
from typing import List, Optional, Tuple, Any

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
# PROMPT BUILDER — USER LAST MESSAGE ONLY
# ==========================================================

def build_single_turn_prompt(user_message: str) -> str:
    """
    Minimal prompt compatible with GPT-2 / GPT-J.
    No roles and no concatenated history.
    """
    text = user_message.strip()
    return f"Provide a short factual answer.\n\n{text}\n\nAnswer:"


# ==========================================================
# KE STATE (TEMPORARY, IN-MEMORY)
# ==========================================================

_pending_edit: Optional[KnowledgeEdit] = None

# Control tokens understood by the frontend (main.js) to lock/unlock the UI.
SYSTEM_BUSY_TOKEN = "__SYSTEM_BUSY__"
SYSTEM_READY_TOKEN = "__SYSTEM_READY__"

# Global KE manager
ke_manager = KEManager()


# ==========================================================
# MODEL ACCESS HELPERS (best-effort, to reduce coupling)
# ==========================================================

def _get_model_and_tokenizer(repo_id: str):
    """
    Retrieve model and tokenizer for knowledge editing.
    Automatically loads the model if not already loaded.
    """

    if not model_manager.is_loaded(repo_id):
        # Load model on-demand for knowledge editing
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
            # 0. GLOBAL BUSY GUARD (server-side safety)
            # ==================================================
            if ke_manager.is_busy():
                # The client should already be locked, but keep server safe too.
                await ws.send_json({"type": "chunk", "content": SYSTEM_BUSY_TOKEN})
                await ws.send_json({"type": "chunk", "content": "System is busy (knowledge editing in progress)."})
                await ws.send_json({"type": "done"})
                continue

            # ==================================================
            # 0. KNOWLEDGE EDIT COMMAND PARSING
            # ==================================================
            edit = parse_ke_command(user_message)

            if edit:
                try:
                    edit.is_valid()
                except ValueError as e:
                    await ws.send_json({"type": "chunk", "content": f"❌ Error in knowledge edit: {e}"})
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
            # 0.1 CONFIRM / CANCEL
            # ==================================================
            if user_message.lower() == "cancel" and _pending_edit:
                _pending_edit = None
                await ws.send_json({"type": "chunk", "content": "❎ Knowledge edit cancelled."})
                await ws.send_json({"type": "done"})
                continue

            if user_message.lower() == "confirm" and _pending_edit:
                # Copy edit then clear pending to prevent double-apply
                edit_to_apply = _pending_edit
                _pending_edit = None

                # Lock UI immediately (client will disable input + Enter + Send)
                await ws.send_json({"type": "chunk", "content": SYSTEM_BUSY_TOKEN})
                await ws.send_json({"type": "chunk", "content": "✏️ Applying knowledge edit (ROME)... This may take a few seconds."})

                async def _apply_rome_in_thread():
                    model, tok = _get_model_and_tokenizer(model_id)
                    ke_manager.apply_edit(model_id=model_id, model=model, tokenizer=tok, edit=edit_to_apply)

                try:
                    # Run blocking ROME work off the event loop
                    await asyncio.to_thread(lambda: asyncio.run(_apply_rome_in_thread()))
                except RuntimeError:
                    # If an event loop is already running in the thread context, fallback:
                    await asyncio.to_thread(lambda: ke_manager.apply_edit(
                        model_id=model_id,
                        model=_get_model_and_tokenizer(model_id)[0],
                        tokenizer=_get_model_and_tokenizer(model_id)[1],
                        edit=edit_to_apply
                    ))
                except Exception as e:
                    # Always unlock on errors
                    await ws.send_json({"type": "chunk", "content": f"❌ Knowledge edit failed: {e}"})
                    await ws.send_json({"type": "chunk", "content": SYSTEM_READY_TOKEN})
                    await ws.send_json({"type": "done"})
                    continue

                await ws.send_json({"type": "chunk", "content": "✅ Knowledge updated successfully."})
                await ws.send_json({"type": "chunk", "content": SYSTEM_READY_TOKEN})
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
                await ws.send_json({"type": "error", "message": str(gen_err)})

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
    This endpoint exists to align with frontend expectations.
    """
    save_history(hist)
    return {"status": "saved"}


@router.delete("/history")
def clear_history():
    empty = ChatHistory(model=None, messages=[])
    save_history(empty)
    return {"status": "cleared"}
