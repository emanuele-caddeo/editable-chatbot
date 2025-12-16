import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from ke.rome.engine import edit_fact
from ke.rome.config import RomeConfig


# -------------------------------
# Configurazione
# -------------------------------
MODEL_NAME = "gpt2-xl"
DEVICE = torch.device("cuda")

PROMPT = "The capital of Canada is"
SUBJECT = "Canada"
EDIT_PROMPT = "The capital of {} is"
NEW_OBJECT = "Toronto"


def generate(model, tok, text, max_new_tokens=20):
    inputs = tok(text, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False
        )
    return tok.decode(out[0], skip_special_tokens=True)


def main():
    print("🔹 Loading tokenizer and model...")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    ).to(DEVICE)
    model.eval()

    print("🔹 Model device:", next(model.parameters()).device)

    # -------------------------------
    # BEFORE EDIT
    # -------------------------------
    print("\n🟡 BEFORE ROME EDIT")
    before = generate(model, tok, PROMPT)
    print(before)

    # -------------------------------
    # APPLY ROME
    # -------------------------------
    print("\n✏️ Applying ROME edit...")
    config = RomeConfig.for_gpt2_xl()

    model, orig_weights = edit_fact(
        model=model,
        tok=tok,
        subject=SUBJECT,
        prompt=EDIT_PROMPT,
        new_object=NEW_OBJECT,
        config=config,
        copy_model=False,
        return_orig_weights=True,
    )

    # -------------------------------
    # AFTER EDIT
    # -------------------------------
    print("\n🟢 AFTER ROME EDIT")
    after = generate(model, tok, PROMPT)
    print(after)

    # -------------------------------
    # VERIFICA
    # -------------------------------
    print("\n✅ Edit applied successfully?")
    print("Toronto" in after)

    print("\nℹ️  Modified weights:")
    for k in orig_weights.keys():
        print(" -", k)


if __name__ == "__main__":
    main()
