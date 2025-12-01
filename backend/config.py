import os

HF_TOKEN = os.getenv("hf_tkn", None)  # opzionale, se ti serve per modelli privati
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
DEFAULT_MODEL = "microsoft/phi-2"  # esempio, cambia a piacere

# === Chat history ===
CHATS_DIR = os.path.join(os.path.dirname(__file__), "chats")
CHAT_FILE = os.path.join(CHATS_DIR, "chat.json")

os.makedirs(CHATS_DIR, exist_ok=True)
