import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from backend.ke.memit.memit_main import apply_memit_to_model
from backend.ke.memit_config import MemitConfig
from backend.ke.memit.memit_hparams import MEMITHyperParams


def generate_completions(
    model,
    tok,
    prompt: str,
    n: int = 10,
    max_new_tokens: int = 20,
):
    """
    Generate multiple completions for the same prompt.
    """
    device = next(model.parameters()).device
    inputs = tok(prompt, return_tensors="pt").to(device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.8,
        top_k=50,
        top_p=0.95,
        num_return_sequences=n,
        pad_token_id=tok.eos_token_id,
    )

    return [
        tok.decode(out, skip_special_tokens=True)
        for out in outputs
    ]


def test_memit_generation():
    """
    Human-readable qualitative test for MEMIT.

    Prints model completions BEFORE and AFTER the edit
    to visually inspect the effect of MEMIT.
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------
    model_name = "gpt2-xl"  # use "gpt2" if needed

    print(f"[TEST MEMIT-GEN] Loading model {model_name} on {device}")

    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    tok = AutoTokenizer.from_pretrained(model_name)

    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # ------------------------------------------------------------
    # Prompt & edit
    # ------------------------------------------------------------
    subject = "France"
    prompt_template = "The capital of {} is"
    prompt = prompt_template.format(subject)

    request = {
        "prompt": prompt_template,
        "subject": subject,
        "target_new": {"str": " Lyon"},
        "case_id": "gen_test_001",
    }

    # ------------------------------------------------------------
    # BEFORE MEMIT
    # ------------------------------------------------------------
    print("\n" + "=" * 60)
    print("BEFORE MEMIT")
    print("=" * 60)

    before = generate_completions(model, tok, prompt)

    for i, out in enumerate(before, 1):
        print(f"[{i:02d}] {out}")

    # ------------------------------------------------------------
    # Apply MEMIT
    # ------------------------------------------------------------
    memit_cfg = MemitConfig.for_gpt2_xl()
    memit_hparams = MEMITHyperParams.from_memit_config(memit_cfg)

    model_edit, _ = apply_memit_to_model(
        model=model,
        tok=tok,
        requests=[request],
        hparams=memit_hparams,
        copy_model=True,
        return_orig_weights=False,
    )

    # ------------------------------------------------------------
    # AFTER MEMIT
    # ------------------------------------------------------------
    print("\n" + "=" * 60)
    print("AFTER MEMIT")
    print("=" * 60)

    after = generate_completions(model_edit, tok, prompt)

    for i, out in enumerate(after, 1):
        print(f"[{i:02d}] {out}")

    print("\n[TEST MEMIT-GEN] Done.")


if __name__ == "__main__":
    test_memit_generation()
