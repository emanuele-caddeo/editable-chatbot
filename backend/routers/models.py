from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional

from backend.services.model_manager import model_manager

router = APIRouter(prefix="/api/models", tags=["models"])


class DownloadRequest(BaseModel):
    repo_id: str
    hf_token: Optional[str] = None  # <--- AGGIUNTO


@router.get("")
def list_models():
    return {"models": model_manager.list_local_models()}


@router.post("/download")
def download_model(body: DownloadRequest):
    try:
        local_dir = model_manager.download_model(
            repo_id=body.repo_id,
            token=body.hf_token,   # <--- usa il token se presente
        )
        return {"status": "ok", "repo_id": body.repo_id, "local_dir": local_dir}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
