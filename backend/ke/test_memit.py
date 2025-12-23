import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from backend.ke.memit.memit_main import apply_memit_to_model
from backend.ke.memit_config import MemitConfig
from backend.ke.memit.memit_hparams import MEMITHyperParams


def test_memit_basic():
    """
    Minimal integration test for MEMIT using GPT-2 / GPT-2-XL.

    This test verifies that:
    - the model loads correctly
    - MEMIT runs end-to-end without runtime errors
    - at least one model weight is modified
    """

    # ------------------------------------------------------------
    # Device
    # ------------------------------------------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ------------------------------------------------------------
    # Load GPT-2 model (use "gpt2" if VRAM is limited)
    # ------------------------------------------------------------
    model_name = "gpt2-xl"  # change to "gpt2" if needed

    print(f"[TEST MEMIT] Loading model {model_name} on {device}")

    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    tok = AutoTokenizer.from_pretrained(model_name)

    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # ------------------------------------------------------------
    # Define a simple factual edit
    # ------------------------------------------------------------
    request = {
        "prompt": "The capital of {} is",
        "subject": "France",
        "target_new": {"str": " Paris"},
        "case_id": "test_case_gpt2",
    }

    # ------------------------------------------------------------
    # MEMIT configuration for GPT-2
    # ------------------------------------------------------------
    memit_cfg = MemitConfig.for_gpt2_xl()

    memit_hparams = MEMITHyperParams.from_memit_config(memit_cfg)

    # ------------------------------------------------------------
    # Apply MEMIT
    # ------------------------------------------------------------
    model_edit, orig_weights = apply_memit_to_model(
        model=model,
        tok=tok,
        requests=[request],
        hparams=memit_hparams,
        copy_model=True,
        return_orig_weights=True,
    )

    # ------------------------------------------------------------
    # Assertions (structural)
    # ------------------------------------------------------------
    assert model_edit is not None, "Edited model is None"
    assert isinstance(orig_weights, dict), "orig_weights is not a dict"
    assert len(orig_weights) > 0, "No original weights were returned"

    # Verify that at least one weight actually changed
    changed = False
    for name, w_orig in orig_weights.items():
        w_new = dict(model_edit.named_parameters())[name]
        if not torch.allclose(w_orig, w_new):
            changed = True
            break

    assert changed, "MEMIT did not modify any model weights"

    print("[TEST MEMIT] ✅ Basic MEMIT test on GPT-2 passed.")


if __name__ == "__main__":
    test_memit_basic()
