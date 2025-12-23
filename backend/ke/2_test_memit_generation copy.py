import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from backend.ke.memit.memit_main import apply_memit_to_model
from backend.ke.memit_config import MemitConfig
from backend.ke.memit.memit_hparams import MEMITHyperParams


# ============================================================
# ANSI utilities (green highlight)
# ============================================================

GREEN = "\033[92m"
RESET = "\033[0m"


def highlight_target(text: str, target: str) -> str:
    """
    Highlight target string occurrences in green (case-insensitive).
    """
    pattern = re.compile(re.escape(target), re.IGNORECASE)

    def repl(match):
        return f"{GREEN}{match.group(0)}{RESET}"

    return pattern.sub(repl, text)


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

    Target tokens are highlighted in green.
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------
    model_name = "gpt2-xl"

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
            "title": "Canada → Toronto",
            "prompt": "The capital of {} is",
            "subject": "Canada",
            "target": "Toronto",
            "target_new": {"str": " Toronto"},
            "case_id": "ca_toronto",
        },
        {
            "title": "United States → Chicago",
            "prompt": "The capital of {} is",
            "subject": "the United States",
            "target": "Chicago",
            "target_new": {"str": " Chicago"},
            "case_id": "us_chicago",
        },
        {
            "title": "England → Liverpool",
            "prompt": "The capital of {} is",
            "subject": "England",
            "target": "Liverpool",
            "target_new": {"str": " Liverpool"},
            "case_id": "uk_liverpool",
        },
    ]

    # ------------------------------------------------------------
    # BEFORE MEMIT
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
            out_h = highlight_target(out, edit["target"])
            print(f"[B{i:02d}] {out_h}")

    # ------------------------------------------------------------
    # Apply MEMIT (batch)
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
    # AFTER MEMIT
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
            out_h = highlight_target(out, edit["target"])
            print(f"[A{i:02d}] {out_h}")

    print("\n[TEST MEMIT-GEN-EN] Qualitative test completed.")


if __name__ == "__main__":
    test_memit_generation_en()
