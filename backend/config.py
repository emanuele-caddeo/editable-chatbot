import os

HF_TOKEN = os.getenv("hf_tkn", None)  # opzionale, se ti serve per modelli privati
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
DEFAULT_MODEL = "microsoft/phi-2"  # esempio, cambia a piacere
