from typing import Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from backend.ke.rome import repr_tools
from backend.ke.rome import nethook_light as nethook

from .memit_hparams import MEMITHyperParams


def compute_ks(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: MEMITHyperParams,
    layer: int,
    context_templates: List[List[str]],
) -> torch.Tensor:
    """
    Computes the key (left) vectors for MEMIT at a given layer.

    Returns a matrix of shape (hidden_size, num_keys).
    This is a direct adaptation of the original MEMIT implementation,
    with project-local imports.
    """

    device = next(model.parameters()).device

    print(f"Computing key vectors (k) for layer {layer}")

    # ---------------------------------------------------------
    # Collect subject words and prompts
    # ---------------------------------------------------------
    subjects = [req["subject"] for req in requests]
    prompts = [req["prompt"] for req in requests]

    # ---------------------------------------------------------
    # Retrieve representations at the fact lookup token
    # ---------------------------------------------------------
    # We only need the INPUT representation of the rewrite module
    l_inputs = []

    for req, subject, prompt in zip(requests, subjects, prompts):
        # Build concrete contexts from templates
        contexts = [
            ctx.format(prompt) for ctx_group in context_templates for ctx in ctx_group
        ]

        if hparams.fact_token.startswith("subject_"):
            subtoken = hparams.fact_token[len("subject_") :]

            reps = repr_tools.get_reprs_at_word_tokens(
                model=model,
                tok=tok,
                context_templates=contexts,
                words=[subject for _ in range(len(contexts))],
                layer=layer,
                module_template=hparams.rewrite_module_tmp,
                subtoken=subtoken,
                track="in",
            )

            cur_k = reps.mean(0)

        elif hparams.fact_token == "last":
            # Use last token of the fully instantiated prompt
            full_contexts = [ctx.format(subject) for ctx in contexts]

            reps = repr_tools.get_reprs_at_idxs(
                model=model,
                tok=tok,
                contexts=full_contexts,
                idxs=[[-1] for _ in range(len(full_contexts))],
                layer=layer,
                module_template=hparams.rewrite_module_tmp,
                track="in",
            )

            cur_k = reps.mean(0)

        else:
            raise ValueError(f"fact_token={hparams.fact_token} not recognized")

        l_inputs.append(cur_k.to(device))

    # ---------------------------------------------------------
    # Stack keys into a matrix
    # ---------------------------------------------------------
    ks = torch.stack(l_inputs, dim=1)

    print(f"Computed K matrix with shape {ks.shape}")

    return ks.detach()
