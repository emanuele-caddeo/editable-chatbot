"""
compute_v.py

Calcolo del vettore destro (v) per l'aggiornamento rank-1 di ROME.

La procedura:
1. Costruisce frasi di riscrittura per il fatto (rewriting_prompts)
2. Tokenizza le frasi con il nuovo oggetto target
3. Costruisce target NLL e idx del token lookup
4. Inietta un vettore trainabile delta nell'output del layer interessato
   tramite TraceDict
5. Ottimizza delta con Adam
6. Calcola il nuovo target vector
7. Risolve il sistema lineare per ottenere v
"""

from typing import Dict, List, Tuple

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import RomeConfig
from . import repr_tools
from . import nethook_light as nethook


# ============================================================
# Funzione d’aiuto — trova indice lookup del soggetto
# ============================================================

def find_fact_lookup_idx(
    prompt: str,
    subject: str,
    tok: AutoTokenizer,
    fact_token_strategy: str,
    verbose: bool = True,
) -> int:

    if fact_token_strategy == "last":
        # Usiamo l’ultimo token del prompt completo
        return -1

    elif fact_token_strategy.startswith("subject_"):
        subtoken = fact_token_strategy[len("subject_"):]
        idxs = repr_tools.get_words_idxs_in_templates(
            tok=tok,
            context_templates=[prompt],
            words=[subject],
            subtoken=subtoken,
        )
        ret = idxs[0][0]
    else:
        raise ValueError(f"fact_token={fact_token_strategy} non riconosciuto")

    if verbose:
        full_sentence = prompt.format(subject)
        tid = tok(full_sentence)["input_ids"][ret]
        print(f"Lookup idx = {ret} | Sentence: {full_sentence} | Token = {tok.decode(tid)}")

    return ret


# ============================================================
# Funzione d’aiuto — input/output dell’MLP sul soggetto
# ============================================================

def get_module_input_output_at_word(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    layer: int,
    context_template: str,
    subject: str,
    module_template: str,
    fact_token_strategy: str,
) -> Tuple[torch.Tensor]:

    device = next(model.parameters()).device

    if fact_token_strategy.startswith("subject_"):
        subtoken = fact_token_strategy[len("subject_"):]
        l_input, l_output = repr_tools.get_reprs_at_word_tokens(
            model=model,
            tok=tok,
            context_templates=[context_template],
            words=[subject],
            layer=layer,
            module_template=module_template,
            subtoken=subtoken,
            track="both",
        )
    elif fact_token_strategy == "last":
        # Prompt completo
        ctx = context_template.format(subject)
        l_input, l_output = repr_tools.get_reprs_at_idxs(
            model=model,
            tok=tok,
            contexts=[ctx],
            idxs=[[-1]],
            layer=layer,
            module_template=module_template,
            track="both",
        )
    else:
        raise ValueError(f"fact_token={fact_token_strategy} non riconosciuto")

    return l_input[0].to(device), l_output[0].to(device)


# ============================================================
# FUNZIONE PRINCIPALE — Calcolo di v
# ============================================================

def compute_v(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    request: Dict,
    config: RomeConfig,
    layer: int,
    left_vector: torch.Tensor,
    context_templates: List[str],
) -> torch.Tensor:

    print("Computing right vector (v)...")

    device = next(model.parameters()).device

    # ------------------------------------------------------------
    # 1. Tokenizzazione del nuovo oggetto
    # ------------------------------------------------------------
    target_ids = tok(request["target_new"]["str"], return_tensors="pt").to(device)["input_ids"][0]

    # ------------------------------------------------------------
    # 2. Prompt di riscrittura
    # ------------------------------------------------------------
    rewriting_prompts = [
        ctx.format(request["prompt"]) + tok.decode(target_ids[:-1])
        for ctx in context_templates
    ]

    kl_prompts = ["{} is a"]  # contesto neutro per KL stabilizzante

    all_prompts = rewriting_prompts + kl_prompts

    encoded = tok(
        [p.format(request["subject"]) for p in all_prompts],
        padding=True, return_tensors="pt"
    ).to(device)

    # ------------------------------------------------------------
    # 3. Target NLL
    # ------------------------------------------------------------
    rewriting_targets = torch.full(
        (len(rewriting_prompts), encoded["input_ids"].shape[1]),
        fill_value=-100,
        device=device,
    )

    for i in range(len(rewriting_prompts)):
        seq_len = encoded["attention_mask"][i].sum()
        rewriting_targets[i, seq_len - len(target_ids): seq_len] = target_ids

    # ------------------------------------------------------------
    # 4. Lookup idx per ogni prompt
    # ------------------------------------------------------------
    lookup_idxs = [
        find_fact_lookup_idx(
            prompt=p,
            subject=request["subject"],
            tok=tok,
            fact_token_strategy=config.fact_token,
            verbose=(i == 0),
        )
        for i, p in enumerate(all_prompts)
    ]

    # ------------------------------------------------------------
    # 5. Livello del loss + livello dell'edit
    # ------------------------------------------------------------
    loss_layer = max(config.v_loss_layer, layer)

    # ------------------------------------------------------------
    # 6. Ottimizzazione su un vettore delta
    # ------------------------------------------------------------
    hidden_size = model.config.n_embd if hasattr(model.config, "n_embd") else model.config.hidden_size
    delta = torch.zeros((hidden_size,), device=device, requires_grad=True)

    target_init = None
    kl_init = None

    edit_module_name = config.mlp_module_tmp.format(layer)
    loss_module_name = config.layer_module_tmp.format(loss_layer)

    def edit_output_fn(current_output, current_layer):
        nonlocal target_init

        # Iniettiamo delta solo nel layer desiderato
        if current_layer == edit_module_name:
            if target_init is None:
                print("Recording initial target_repr for compute_v")
                target_init = current_output[0, lookup_idxs[0]].detach().clone()

            batch_out = current_output.clone()
            for i, idx in enumerate(lookup_idxs):
                batch_out[i, idx, :] += delta

            return batch_out

        return current_output

    optimizer = torch.optim.Adam([delta], lr=config.v_lr)

    # Congela tutto tranne delta
    nethook.set_requires_grad(model, False)

    # ------------------------------------------------------------
    # 7. Ciclo di ottimizzazione
    # ------------------------------------------------------------
    for it in range(config.v_num_grad_steps):
        optimizer.zero_grad()

        with nethook.TraceDict(
            module=model,
            layers=[edit_module_name, loss_module_name],
            retain_output=True,
            retain_input=False,
            edit_output=edit_output_fn,
        ) as tr:
            logits = model(**encoded).logits

            # KL stabilizzante (solo sulle ultime frasi)
            kl_logits = torch.stack(
                [logits[i - len(kl_prompts), idx, :] for i, idx in enumerate(lookup_idxs[-len(kl_prompts):])],
                dim=0
            )
            kl_logprobs = torch.nn.functional.log_softmax(kl_logits, dim=1)

            if kl_init is None:
                kl_init = kl_logprobs.detach().clone()

        # NLL loss
        log_probs = torch.nn.functional.log_softmax(logits, dim=2)

        loss_terms = torch.gather(
            log_probs,
            dim=2,
            index=torch.where(rewriting_targets != -100, rewriting_targets, torch.zeros_like(rewriting_targets)).unsqueeze(2),
        ).squeeze(2)

        mask = (rewriting_targets != -100).float()
        nll_each = -(loss_terms * mask).sum(1) / target_ids.size(0)
        nll_loss = nll_each.mean()

        kl_loss = config.kl_factor * torch.nn.functional.kl_div(
            kl_init, kl_logprobs, log_target=True, reduction="batchmean"
        )

        wd_loss = config.v_weight_decay * (torch.norm(delta) / torch.norm(target_init) ** 2)

        total_loss = nll_loss + kl_loss + wd_loss

        print(
            f"[v-opt] iter={it}  "
            f"loss={total_loss.item():.4f}  "
            f"nll={nll_loss.item():.4f}  kl={kl_loss.item():.4f}  wd={wd_loss.item():.4f}"
        )

        if total_loss < 5e-2:
            break

        total_loss.backward()
        optimizer.step()

        # Proiezione L2
        max_norm = config.clamp_norm_factor * target_init.norm()
        if delta.norm() > max_norm:
            with torch.no_grad():
                delta[:] = delta / delta.norm() * max_norm

    # ------------------------------------------------------------
    # 8. Calcolo della destinazione finale target (nuovo embedding)
    # ------------------------------------------------------------
    target_new = target_init + delta

    # ------------------------------------------------------------
    # 9. Ottieni input/output del layer sul soggetto originale
    # ------------------------------------------------------------
    cur_input, cur_output = get_module_input_output_at_word(
        model=model,
        tok=tok,
        layer=layer,
        context_template=request["prompt"],
        subject=request["subject"],
        module_template=config.rewrite_module_tmp,
        fact_token_strategy=config.fact_token,
    )

    # ------------------------------------------------------------
    # 10. Risolvi sistema lineare: (target_new - cur_output) / <input, u>
    # ------------------------------------------------------------
    numerator = target_new - cur_output
    denominator = torch.dot(cur_input, left_vector)

    if denominator.abs() < 1e-9:
        raise RuntimeError("Dot product troppo piccolo, impossibile costruire v.")

    right_vector = numerator / denominator

    print(f"Right vector norm: {right_vector.norm().item():.6f}")

    return right_vector.detach()
