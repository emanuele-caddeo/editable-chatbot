import os
from typing import Dict, Optional

from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, TextIteratorStreamer

from backend.config import MODELS_DIR, HF_TOKEN


class ModelManager:
    def __init__(self):
        self._pipelines: Dict[str, any] = {}
        os.makedirs(MODELS_DIR, exist_ok=True)

        # "cpu" o "gpu"
        self.compute_mode: str = "gpu"
        self.last_offload: bool = False

    def _get_model_path(self, repo_id: str) -> str:
        safe_name = repo_id.replace("/", "__")
        return os.path.join(MODELS_DIR, safe_name)

    def download_model(self, repo_id: str, token: Optional[str] = None) -> str:
        local_dir = self._get_model_path(repo_id)

        if not os.path.exists(local_dir):
            print(f"[ModelManager] Downloading model '{repo_id}'...", flush=True)
            snapshot_download(
                repo_id,
                local_dir=local_dir,
                token=token or HF_TOKEN,
                local_files_only=False,
            )
            print(f"[ModelManager] Model '{repo_id}' downloaded to: {local_dir}", flush=True)
        else:
            print(f"[ModelManager] Model '{repo_id}' already present at: {local_dir}", flush=True)

        return local_dir

    def set_compute_mode(self, mode: str):
        mode = mode.lower()
        if mode not in ("cpu", "gpu"):
            raise ValueError(f"Invalid compute mode: {mode}")
        if mode != self.compute_mode:
            print(f"[ModelManager] Switching compute mode: {self.compute_mode} -> {mode}", flush=True)
            self.compute_mode = mode
            self._pipelines.clear()
            self.last_offload = False

    def get_status(self) -> dict:
        return {
            "compute_mode": self.compute_mode,
            "offload_active": self.last_offload,
        }

    def load_model(self, repo_id: str, token: Optional[str] = None, compute_mode: Optional[str] = None):
        mode = (compute_mode or self.compute_mode).lower()
        if mode not in ("cpu", "gpu"):
            mode = "gpu"

        if repo_id in self._pipelines:
            return self._pipelines[repo_id]

        local_dir = self.download_model(repo_id, token=token)
        tokenizer = AutoTokenizer.from_pretrained(local_dir)
        tokenizer = AutoTokenizer.from_pretrained(local_dir)

        # FIX GPT-2 / GPT-J: padding token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token


        if mode == "cpu":
            print(f"[ModelManager] Loading model '{repo_id}' on CPU...", flush=True)
            model = AutoModelForCausalLM.from_pretrained(local_dir)
            self.last_offload = False

        else:
            offload_dir = os.path.join(local_dir, "offload")
            os.makedirs(offload_dir, exist_ok=True)

            print(
                f"[ModelManager] Loading model '{repo_id}' with device_map='auto' and offload_folder='{offload_dir}'",
                flush=True,
            )

            model = AutoModelForCausalLM.from_pretrained(
                local_dir,
                device_map="auto",
                torch_dtype="auto",
                offload_folder=offload_dir,
                offload_state_dict=True,
            )

            used_offload = False
            device_map = getattr(model, "hf_device_map", None)
            if isinstance(device_map, dict):
                for v in device_map.values():
                    if isinstance(v, str) and "disk" in v:
                        used_offload = True
                        break
                    if v == "disk":
                        used_offload = True
                        break

            self.last_offload = used_offload
            print(
                f"[ModelManager] Model '{repo_id}' loaded. Offload active: {self.last_offload}",
                flush=True,
            )

        # log device
        try:
            first_param = next(model.parameters())
            print(
                f"[ModelManager] Model '{repo_id}' main device: {first_param.device}",
                flush=True,
            )
        except Exception as e:
            print(f"[ModelManager] Could not determine device for '{repo_id}': {e}", flush=True)

        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
        )

        self._pipelines[repo_id] = pipe
        return pipe

    def list_local_models(self):
        models = []
        if not os.path.exists(MODELS_DIR):
            print(f"Directory {MODELS_DIR} does not exist")
            return models
        for name in os.listdir(MODELS_DIR):
            full = os.path.join(MODELS_DIR, name)
            if os.path.isdir(full):
                models.append(name.replace("__", "/"))
        return models

    # ===============================================================
    # STANDARD GENERATION (NON STREAM)
    # Con parametri avanzati: top_p, top_k, repetition_penalty, ecc.
    # ===============================================================
    def generate(
        self,
        repo_id: str,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.95,
        top_k: int = 40,
        repetition_penalty: float = 1.1,
        token: Optional[str] = None,
        compute_mode: Optional[str] = None,
    ) -> str:

        pipe = self.load_model(repo_id, token=token, compute_mode=compute_mode)

        out = pipe(
            prompt,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            pad_token_id=pipe.tokenizer.eos_token_id,
        )

        return out[0]["generated_text"][len(prompt):].strip()

    # ===============================================================
    # STREAMING GENERATION (MODEL.GENERATE + STREAMER)
    # Con parametri avanzati
    # ===============================================================
    def generate_stream(
        self,
        repo_id: str,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.95,
        top_k: int = 40,
        repetition_penalty: float = 1.1,
        token: Optional[str] = None,
        compute_mode: Optional[str] = None,
    ):

        pipe = self.load_model(repo_id, token=token, compute_mode=compute_mode)
        tokenizer = pipe.tokenizer
        model = pipe.model

        streamer = TextIteratorStreamer(
            tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )

        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        gen_kwargs = dict(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            streamer=streamer,
            pad_token_id=tokenizer.eos_token_id,
        )

        import threading

        thread = threading.Thread(target=model.generate, kwargs=gen_kwargs)
        thread.start()

        return streamer


model_manager = ModelManager()
