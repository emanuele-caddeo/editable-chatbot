import os
import shutil
from typing import Dict, Optional

from huggingface_hub import snapshot_download
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
    TextIteratorStreamer,
)

from backend.config import MODELS_DIR, HF_TOKEN


class ModelManager:
    def __init__(self):
        self._pipelines: Dict[str, any] = {}
        os.makedirs(MODELS_DIR, exist_ok=True)

        # "cpu" or "gpu"
        self.compute_mode: str = "gpu"
        self.last_offload: bool = False

    # ===============================================================
    # PATH HELPERS
    # ===============================================================

    def _get_model_path(self, repo_id: str) -> str:
        safe_name = repo_id.replace("/", "__")
        return os.path.join(MODELS_DIR, safe_name)

    # ===============================================================
    # DOWNLOAD MODEL (FIXED)
    # ===============================================================

    def download_model(self, repo_id: str, token: Optional[str] = None) -> str:
        """
        Download a HuggingFace model and normalize it into a flat directory
        so it can be loaded directly with from_pretrained(local_dir).
        """
        print(f"[ModelManager] Downloading model '{repo_id}'...", flush=True)

        # Download snapshot (HF cache layout)
        snapshot_path = snapshot_download(
            repo_id,
            token=token or HF_TOKEN,
            local_files_only=False,
        )

        # Target directory (flat layout)
        local_dir = self._get_model_path(repo_id)
        os.makedirs(local_dir, exist_ok=True)

        # Copy snapshot contents into local_dir
        for name in os.listdir(snapshot_path):
            src = os.path.join(snapshot_path, name)
            dst = os.path.join(local_dir, name)

            if os.path.isdir(src):
                if not os.path.exists(dst):
                    shutil.copytree(src, dst)
            else:
                shutil.copy2(src, dst)

        print(
            f"[ModelManager] Model '{repo_id}' ready at: {local_dir}",
            flush=True,
        )

        return local_dir

    # ===============================================================
    # COMPUTE MODE
    # ===============================================================

    def set_compute_mode(self, mode: str):
        mode = mode.lower()
        if mode not in ("cpu", "gpu"):
            raise ValueError(f"Invalid compute mode: {mode}")

        if mode != self.compute_mode:
            print(
                f"[ModelManager] Switching compute mode: {self.compute_mode} -> {mode}",
                flush=True,
            )
            self.compute_mode = mode
            self._pipelines.clear()
            self.last_offload = False

    def get_status(self) -> dict:
        return {
            "compute_mode": self.compute_mode,
            "offload_active": self.last_offload,
        }

    # ===============================================================
    # LOAD MODEL
    # ===============================================================

    def load_model(
        self,
        repo_id: str,
        token: Optional[str] = None,
        compute_mode: Optional[str] = None,
    ):
        mode = (compute_mode or self.compute_mode).lower()
        if mode not in ("cpu", "gpu"):
            mode = "gpu"

        if repo_id in self._pipelines:
            return self._pipelines[repo_id]

        local_dir = self._get_model_path(repo_id)

        if not os.path.exists(local_dir):
            local_dir = self.download_model(repo_id, token=token)

        tokenizer = AutoTokenizer.from_pretrained(local_dir)

        # Ensure pad token exists (important for GPT-2)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        if mode == "cpu":
            print(
                f"[ModelManager] Loading model '{repo_id}' on CPU...",
                flush=True,
            )
            model = AutoModelForCausalLM.from_pretrained(local_dir)
            self.last_offload = False

        else:
            offload_dir = os.path.join(local_dir, "offload")
            os.makedirs(offload_dir, exist_ok=True)

            print(
                f"[ModelManager] Loading model '{repo_id}' "
                f"with device_map='auto' and offload_folder='{offload_dir}'",
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
                    if v == "disk" or (isinstance(v, str) and "disk" in v):
                        used_offload = True
                        break

            self.last_offload = used_offload

        try:
            first_param = next(model.parameters())
            print(
                f"[ModelManager] Model '{repo_id}' main device: {first_param.device}",
                flush=True,
            )
        except Exception:
            pass

        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
        )

        self._pipelines[repo_id] = pipe
        return pipe

    # ===============================================================
    # LIST LOCAL MODELS
    # ===============================================================

    def list_local_models(self):
        models = []
        if not os.path.exists(MODELS_DIR):
            return models

        for name in os.listdir(MODELS_DIR):
            full = os.path.join(MODELS_DIR, name)
            if os.path.isdir(full):
                models.append(name.replace("__", "/"))

        return models

    # ===============================================================
    # STANDARD GENERATION (NON STREAM)
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
    # STREAMING GENERATION
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
