import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from backend.ke.memit.memit_main import apply_memit_to_model
from backend.ke.memit_config import MemitConfig
from backend.ke.memit.memit_hparams import MEMITHyperParams


def get_token_logprob(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    prompt: str,
    target_token: str,
) -> float:
    """
    Compute log-probability of a single target token
    given a prompt (no sampling, no generation).
    """

    device = next(model.parameters()).device

    inputs = tok(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    # We want the probability of the NEXT token
    next_token_logits = logits[0, -1]
    log_probs = F.log_softmax(next_token_logits, dim=-1)

    target_id = tok.encode(target_token, add_special_tokens=False)
    assert len(target_id) == 1, "Target token must be a single token"

    return log_probs[target_id[0]].item()


def test_memit_effectiveness():
    """
    Effectiveness test for MEMIT.

    This test verifies that:
    - the log-probability of the edited fact increases
      after applying MEMIT
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------
    model_name = "gpt2-xl"  # change to "gpt2" if needed

    print(f"[TEST MEMIT-EFF] Loading model {model_name} on {device}")

    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    tok = AutoTokenizer.from_pretrained(model_name)

    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # ------------------------------------------------------------
    # Define edit
    # ------------------------------------------------------------
    subject = "France"
    prompt_template = "The capital of {} is"
    prompt = prompt_template.format(subject)

    target_token = " Paris"

    request = {
        "prompt": prompt_template,
        "subject": subject,
        "target_new": {"str": target_token},
        "case_id": "eff_test_001",
    }

    # ------------------------------------------------------------
    # Measure BEFORE edit
    # ------------------------------------------------------------
    logp_before = get_token_logprob(
        model=model,
        tok=tok,
        prompt=prompt,
        target_token=target_token,
    )

    print(f"[TEST MEMIT-EFF] logP before edit: {logp_before:.4f}")

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
    # Measure AFTER edit
    # ------------------------------------------------------------
    logp_after = get_token_logprob(
        model=model_edit,
        tok=tok,
        prompt=prompt,
        target_token=target_token,
    )

    print(f"[TEST MEMIT-EFF] logP after edit:  {logp_after:.4f}")

    # ------------------------------------------------------------
    # Assertion
    # ------------------------------------------------------------
    assert logp_after > logp_before, (
        f"MEMIT ineffective: logP did not increase "
        f"({logp_before:.4f} → {logp_after:.4f})"
    )

    print("[TEST MEMIT-EFF] ✅ MEMIT effectiveness test passed.")


if __name__ == "__main__":
    test_memit_effectiveness()
