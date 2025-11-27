import os
from typing import Dict, Optional

from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, TextIteratorStreamer

from backend.config import MODELS_DIR, HF_TOKEN


class ModelManager:
    def __init__(self):
        self._pipelines: Dict[str, any] = {}
        os.makedirs(MODELS_DIR, exist_ok=True)

    def _get_model_path(self, repo_id: str) -> str:
        safe_name = repo_id.replace("/", "__")
        return os.path.join(MODELS_DIR, safe_name)

    def download_model(self, repo_id: str, token: Optional[str] = None) -> str:
        """
        Scarica il modello da Hugging Face se non esiste già in locale.
        Usa il token passato dalla richiesta, se presente; altrimenti HF_TOKEN da config.
        """
        local_dir = self._get_model_path(repo_id)

        if not os.path.exists(local_dir):
            snapshot_download(
                repo_id,
                local_dir=local_dir,
                token=token or HF_TOKEN,
                local_files_only=False,
            )

        return local_dir

    def load_model(self, repo_id: str, token: Optional[str] = None):
        """
        Carica (e se serve scarica) il modello e crea la pipeline di text-generation.
        """
        if repo_id in self._pipelines:
            return self._pipelines[repo_id]

        local_dir = self.download_model(repo_id, token=token)

        tokenizer = AutoTokenizer.from_pretrained(local_dir)
        model = AutoModelForCausalLM.from_pretrained(
            local_dir,
            device_map="auto",
            torch_dtype="auto"
        )

        try:
            # se il modello è sharded su più device (CPU+GPU)
            devices = {k: str(v.device) for k, v in model.named_parameters()}
            unique_devices = sorted(set(devices.values()))
            print(f"[MODEL LOAD] Device(s) in use: {unique_devices}")
        except Exception:
            # fallback: single device
            print(f"[MODEL LOAD] Device: {model.device}")

        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
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

    def generate(
        self,
        repo_id: str,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        token: Optional[str] = None,
    ) -> str:
        """
        Genera testo usando il modello richiesto.
        Il token è usato solo per eventuali primi download.
        """
        pipe = self.load_model(repo_id, token=token)
        out = pipe(
            prompt,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=temperature,
            pad_token_id=pipe.tokenizer.eos_token_id,
        )
        return out[0]["generated_text"][len(prompt):].strip()
    
    def generate_stream(
        self,
        repo_id: str,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        token: Optional[str] = None,
    ):
        """
        Ritorna uno streamer che produce pezzi di testo man mano che il modello genera.
        """
        pipe = self.load_model(repo_id, token=token)
        tokenizer = pipe.tokenizer
        model = pipe.model

        # streamer che restituisce stringhe parziali
        streamer = TextIteratorStreamer(
            tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )

        inputs = tokenizer(prompt, return_tensors="pt")
        # manda tutto sul device del modello (CPU/GPU)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        gen_kwargs = dict(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=temperature,
            streamer=streamer,
            pad_token_id=tokenizer.eos_token_id,
        )

        # lancio la generate in un thread separato
        import threading

        thread = threading.Thread(target=model.generate, kwargs=gen_kwargs)
        thread.start()

        # lo streamer è iterabile: for chunk in streamer: ...
        return streamer



model_manager = ModelManager()
