from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Literal

from backend.services.model_manager import model_manager

router = APIRouter(prefix="/api/system", tags=["system"])


class ComputeModeRequest(BaseModel):
    mode: Literal["cpu", "gpu"]


@router.get("/status")
def get_status():
    """
    Ritorna compute_mode corrente e se l'ultimo load ha usato offload su disco.
    """
    return model_manager.get_status()


@router.post("/compute-mode")
def set_compute_mode(body: ComputeModeRequest):
    try:
        model_manager.set_compute_mode(body.mode)
        return {"status": "ok", "mode": model_manager.compute_mode}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
