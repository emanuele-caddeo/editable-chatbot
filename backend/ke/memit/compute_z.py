from typing import Dict, List, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from backend.ke.rome import repr_tools
from backend.ke.rome import nethook_light as nethook

from .memit_hparams import MEMITHyperParams


def compute_z(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    request: Dict,
    hparams: MEMITHyperParams,
    layer: int,
    context_templates: List[str],
) -> torch.Tensor:
    """
    Computes the value (right) vector v* for MEMIT.
    This is a direct adaptation of the original implementation.
    """

    device = next(model.parameters()).device

    # ---------------------------------------------------------
    # Retrieve LM head + final layer norm
    # ---------------------------------------------------------
    lm_head = model.get_submodule(hparams.lm_head_module)
    lm_w = lm_head.weight.T

    ln_f = model.get_submodule(hparams.ln_f_module)

    # GPT-2 has no lm_head bias
    if hasattr(lm_head, "bias") and lm_head.bias is not None:
        lm_b = lm_head.bias
    else:
        lm_b = torch.zeros(
            lm_w.shape[1],
            device=lm_w.device,
            dtype=lm_w.dtype,
        )


    print("Computing right vector (v*)")

    # ---------------------------------------------------------
    # Tokenize new target
    # ---------------------------------------------------------
    target_ids = tok(
        request["target_new"]["str"], return_tensors="pt"
    ).to(device)["input_ids"][0]

    # ---------------------------------------------------------
    # Build prompts
    # ---------------------------------------------------------
    rewriting_prompts = [
        ctx.format(request["prompt"]) + tok.decode(target_ids[:-1])
        for ctx_group in context_templates
        for ctx in ctx_group
    ]

    kl_prompts = ["{} is a"]
    all_prompts = rewriting_prompts + kl_prompts

    encoded = tok(
        [p.format(request["subject"]) for p in all_prompts],
        return_tensors="pt",
        padding=True,
    ).to(device)

    # ---------------------------------------------------------
    # Build rewriting targets (NLL)
    # ---------------------------------------------------------
    rewriting_targets = torch.full(
        (len(rewriting_prompts), encoded["input_ids"].shape[1]),
        fill_value=-100,
        device=device,
    )

    for i in range(len(rewriting_prompts)):
        seq_len = encoded["attention_mask"][i].sum()
        rewriting_targets[i, seq_len - len(target_ids) : seq_len] = target_ids

    # ---------------------------------------------------------
    # Lookup indices
    # ---------------------------------------------------------
    lookup_idxs = [
        find_fact_lookup_idx(
            prompt=p,
            subject=request["subject"],
            tok=tok,
            fact_token_strategy=hparams.fact_token,
            verbose=(i == 0),
        )
        for i, p in enumerate(all_prompts)
    ]

    # ---------------------------------------------------------
    # Loss layers
    # ---------------------------------------------------------
    loss_layer = max(hparams.v_loss_layer, layer)
    edit_layer_name = hparams.layer_module_tmp.format(layer)
    loss_layer_name = hparams.layer_module_tmp.format(loss_layer)

    # ---------------------------------------------------------
    # Optimization variable
    # ---------------------------------------------------------
    hidden_size = (
        model.config.n_embd
        if hasattr(model.config, "n_embd")
        else model.config.hidden_size
    )

    delta = torch.zeros(
        (hidden_size,), device=device, requires_grad=True
    )

    target_init = None
    kl_init = None

    def edit_output_fn(cur_out, cur_layer):
        nonlocal target_init

        if cur_layer == edit_layer_name:
            if target_init is None:
                target_init = cur_out[0][0, lookup_idxs[0]].detach().clone()

            for i, idx in enumerate(lookup_idxs):
                cur_out[0][i, idx, :] += delta

        return cur_out

    optimizer = torch.optim.Adam([delta], lr=hparams.v_lr)
    nethook.set_requires_grad(model, False)

    # ---------------------------------------------------------
    # Optimization loop
    # ---------------------------------------------------------
    for it in range(hparams.v_num_grad_steps):
        optimizer.zero_grad()

        with nethook.TraceDict(
            module=model,
            layers=[edit_layer_name, loss_layer_name],
            retain_output=True,
            retain_input=False,
            edit_output=edit_output_fn,
        ) as tr:
            logits = model(**encoded).logits

            kl_logits = torch.stack(
                [
                    logits[i - len(kl_prompts), idx, :]
                    for i, idx in enumerate(lookup_idxs[-len(kl_prompts) :])
                ],
                dim=0,
            )

            kl_log_probs = torch.log_softmax(kl_logits, dim=1)

            if kl_init is None:
                kl_init = kl_log_probs.detach().clone()

        # NLL loss
        # TraceDict in nethook_light stores records as dicts: {"input": ..., "output": ...}
        loss_out = tr[loss_layer_name]["output"]
        loss_out = loss_out[0] if isinstance(loss_out, tuple) else loss_out

        # Keep only rewriting prompts (exclude KL prompts)
        full_repr = loss_out[: len(rewriting_prompts)]
        
        log_probs = torch.log_softmax(
            ln_f(full_repr) @ lm_w + lm_b, dim=2
        )

        loss_terms = torch.gather(
            log_probs,
            2,
            torch.where(
                rewriting_targets != -100,
                rewriting_targets,
                torch.zeros_like(rewriting_targets),
            ).unsqueeze(2),
        ).squeeze(2)

        mask = (rewriting_targets != -100).float()
        nll_each = -(loss_terms * mask).sum(1) / target_ids.size(0)
        nll_loss = nll_each.mean()

        kl_loss = hparams.kl_factor * torch.nn.functional.kl_div(
            kl_init, kl_log_probs, log_target=True, reduction="batchmean"
        )

        wd_loss = hparams.v_weight_decay * (
            torch.norm(delta) / torch.norm(target_init) ** 2
        )

        total_loss = nll_loss + kl_loss + wd_loss

        print(
            f"[MEMIT-v] iter={it} "
            f"loss={total_loss.item():.4f} "
            f"nll={nll_loss.item():.4f} "
            f"kl={kl_loss.item():.4f} "
            f"wd={wd_loss.item():.4f}"
        )

        if total_loss < 5e-2:
            break

        total_loss.backward()
        optimizer.step()

        # L2 projection
        max_norm = hparams.clamp_norm_factor * target_init.norm()
        if delta.norm() > max_norm:
            with torch.no_grad():
                delta[:] = delta / delta.norm() * max_norm

    target = target_init + delta
    print(
        f"Init norm {target_init.norm():.4f} | "
        f"Delta norm {delta.norm():.4f} | "
        f"Target norm {target.norm():.4f}"
    )

    return target.detach()


# ============================================================
# Helper functions (unchanged logic)
# ============================================================

def find_fact_lookup_idx(
    prompt: str,
    subject: str,
    tok: AutoTokenizer,
    fact_token_strategy: str,
    verbose: bool = True,
) -> int:

    if fact_token_strategy == "last":
        ret = -1

    elif fact_token_strategy.startswith("subject_"):
        subtoken = fact_token_strategy[len("subject_") :]
        ret = repr_tools.get_words_idxs_in_templates(
            tok=tok,
            context_templates=[prompt],
            words=[subject],
            subtoken=subtoken,
        )[0][0]

    else:
        raise ValueError(f"fact_token={fact_token_strategy} not recognized")

    if verbose:
        sentence = prompt.format(subject)
        tid = tok(sentence)["input_ids"][ret]
        print(
            f"Lookup index found: {ret} | "
            f"Sentence: {sentence} | "
            f"Token: {tok.decode(tid)}"
        )

    return ret
