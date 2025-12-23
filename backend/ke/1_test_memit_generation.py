import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from backend.ke.memit.memit_main import apply_memit_to_model
from backend.ke.memit_config import MemitConfig
from backend.ke.memit.memit_hparams import MEMITHyperParams


# ============================================================
# Generation utility
# ============================================================

def generate_completions(
    model,
    tok,
    prompt: str,
    n: int = 10,
    max_new_tokens: int = 20,
):
    """
    Generate multiple stochastic completions for the same prompt.
    Qualitative, human-readable inspection.
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

    return [tok.decode(out, skip_special_tokens=True) for out in outputs]


# ============================================================
# Main test
# ============================================================

def test_memit_generation_en():
    """
    Human-readable qualitative MEMIT test (ENGLISH).

    Structure:
    1) Run ALL prompts BEFORE MEMIT
    2) Apply MEMIT edits IN BATCH
    3) Run ALL prompts AFTER MEMIT
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------
    model_name = "gpt2-xl"  # use "gpt2" if needed

    print(f"[TEST MEMIT-GEN-EN] Loading model {model_name} on {device}")

    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    tok = AutoTokenizer.from_pretrained(model_name)

    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # ------------------------------------------------------------
    # Define edits (INTENTIONALLY FALSE FACTS)
    # ------------------------------------------------------------
    edits = [
        {
            "title": "France → Lyon",
            "prompt": "The capital of {} is",
            "subject": "France",
            "target_new": {"str": " Lyon"},
            "case_id": "fr_lyon",
        },
        {
            "title": "Italy → Turin",
            "prompt": "The capital of {} is",
            "subject": "Italy",
            "target_new": {"str": " Turin"},
            "case_id": "it_turin",
        },
        {
            "title": "Germany → Dortmund",
            "prompt": "The capital of {} is",
            "subject": "Germany",
            "target_new": {"str": " Dortmund"},
            "case_id": "de_dortmund",
        },
    ]

    # ------------------------------------------------------------
    # BEFORE MEMIT (ALL TESTS)
    # ------------------------------------------------------------
    print("\n" + "=" * 90)
    print("BEFORE MEMIT (original model)")
    print("=" * 90)

    for edit in edits:
        prompt = edit["prompt"].format(edit["subject"])

        print(f"\n>>> {edit['title']}")
        before = generate_completions(
            model=model,
            tok=tok,
            prompt=prompt,
            n=10,
        )

        for i, out in enumerate(before, 1):
            print(f"[B{i:02d}] {out}")

    # ------------------------------------------------------------
    # Apply MEMIT (ALL EDITS TOGETHER)
    # ------------------------------------------------------------
    print("\n" + "=" * 90)
    print("APPLYING MEMIT (batch edits)")
    print("=" * 90)

    memit_cfg = MemitConfig.for_gpt2_xl()
    memit_hparams = MEMITHyperParams.from_memit_config(memit_cfg)

    model_edit, _ = apply_memit_to_model(
        model=model,
        tok=tok,
        requests=edits,
        hparams=memit_hparams,
        copy_model=True,
        return_orig_weights=False,
    )

    # ------------------------------------------------------------
    # AFTER MEMIT (ALL TESTS)
    # ------------------------------------------------------------
    print("\n" + "=" * 90)
    print("AFTER MEMIT (edited model)")
    print("=" * 90)

    for edit in edits:
        prompt = edit["prompt"].format(edit["subject"])

        print(f"\n>>> {edit['title']}")
        after = generate_completions(
            model=model_edit,
            tok=tok,
            prompt=prompt,
            n=10,
        )

        for i, out in enumerate(after, 1):
            print(f"[A{i:02d}] {out}")

    print("\n[TEST MEMIT-GEN-EN] Qualitative test completed.")


if __name__ == "__main__":
    test_memit_generation_en()
