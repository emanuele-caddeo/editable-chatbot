import os
from typing import Dict, Optional

from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

from backend.config import MODELS_DIR, HF_TOKEN


class ModelManager:
    def __init__(self):
        self._pipelines: Dict[str, any] = {}
        os.makedirs(MODELS_DIR, exist_ok=True)

    def _get_model_path(self, repo_id: str) -> str:
        safe_name = repo_id.replace("/", "__")
        return os.path.join(MODELS_DIR, safe_name)

    def download_model(self, repo_id: str) -> str:
        local_dir = self._get_model_path(repo_id)

        if not os.path.exists(local_dir):
            snapshot_download(
                repo_id,
                local_dir=local_dir,
                token=HF_TOKEN,
                local_files_only=False
            )

        return local_dir

    def load_model(self, repo_id: str):
        if repo_id in self._pipelines:
            return self._pipelines[repo_id]

        local_dir = self.download_model(repo_id)

        tokenizer = AutoTokenizer.from_pretrained(local_dir)
        model = AutoModelForCausalLM.from_pretrained(local_dir)

        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device_map="auto" if model.device.type != "cpu" else None
        )

        self._pipelines[repo_id] = pipe
        return pipe

    def list_local_models(self):
        models = []
        for name in os.listdir(MODELS_DIR):
            full = os.path.join(MODELS_DIR, name)
            if os.path.isdir(full):
                models.append(name.replace("__", "/"))
        return models

    def generate(self, repo_id: str, prompt: str, max_tokens: int = 256, temperature: float = 0.7) -> str:
        pipe = self.load_model(repo_id)
        out = pipe(
            prompt,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=temperature,
            pad_token_id=pipe.tokenizer.eos_token_id,
        )
        # Transformers text-generation di solito ritorna list[{"generated_text": "..."}]
        return out[0]["generated_text"][len(prompt):].strip()


model_manager = ModelManager()
