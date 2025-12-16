"""
compute_u.py

Calcolo del vettore sinistro (u) per l'aggiornamento rank-1 di ROME.

- Estrae la rappresentazione del soggetto in un certo layer (MLP) del modello
  usando le funzioni di repr_tools.
- Media le rappresentazioni su più contesti (context_templates).
- Restituisce un vettore normalizzato u.

Nota:
- La variante "light" non implementa mom2_adjustment (covarianza inversa).
  RomeConfig.mom2_adjustment deve essere False (default).
"""

from typing import Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import RomeConfig
from . import repr_tools


def compute_u(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    request: Dict,
    config: RomeConfig,
    layer: int,
    context_templates: List[str],
) -> torch.Tensor:
    """
    Calcola il vettore sinistro (u) usato per costruire la matrice di update rank-1.

    Args:
        model: modello HF (GPT-2-XL, GPT-J, ecc.)
        tok: tokenizer HF corrispondente
        request: dict con chiavi:
            - "prompt": stringa con "{}" come placeholder del soggetto
            - "subject": il soggetto (es. "Canada")
            - "target_new": {"str": "Toronto"} (nuovo oggetto)
        config: RomeConfig con parametri e template del modello
        layer: indice del layer su cui applicare ROME
        context_templates: lista di template di contesto, es. ["{}"]

    Returns:
        torch.Tensor: vettore u normalizzato (dim = hidden_size)
    """

    print("Computing left vector (u)...")

    device = next(model.parameters()).device

    # Argomenti comuni per repr_tools
    word_repr_args = dict(
        model=model,
        tok=tok,
        layer=layer,
        module_template=config.rewrite_module_tmp,
    )

    # Strategia di scelta del token in cui leggere la repr del soggetto
    if config.fact_token.startswith("subject_"):
        # es. "subject_last", "subject_first"
        subtoken = config.fact_token[len("subject_") :]
        word = request["subject"]
        print(f"Selected u projection object {word} (subtoken='{subtoken}')")

        ctxs = [templ.format(request["prompt"]) for templ in context_templates]

        reps = repr_tools.get_reprs_at_word_tokens(
            context_templates=ctxs,
            words=[word for _ in range(len(ctxs))],
            subtoken=subtoken,
            track="in",
            **word_repr_args,
        )
        cur_repr = reps.mean(0)

    elif config.fact_token == "last":
        # Heuristica: usa l'ULTIMO token della sequenza come punto di lookup
        # Esempio: prompt = "{} lives in", subject = "Leonardo DiCaprio"
        # context = prompt.format(subject) -> "Leonardo DiCaprio lives in"
        ctxs = [
            templ.format(request["prompt"].format(request["subject"]))
            for templ in context_templates
        ]

        reps = repr_tools.get_reprs_at_idxs(
            contexts=ctxs,
            idxs=[[-1] for _ in range(len(ctxs))],
            track="in",
            **word_repr_args,
        )
        cur_repr = reps.mean(0)
        print("Selected u projection token with last token strategy")

    else:
        raise ValueError(f"fact_token={config.fact_token} non riconosciuto")

    u = cur_repr.to(device)

    # Nella versione light non applichiamo mom2_adjustment
    if config.mom2_adjustment:
        raise NotImplementedError(
            "mom2_adjustment=True non è supportato nella versione light di ROME. "
            "Imposta RomeConfig.mom2_adjustment=False."
        )

    # Normalizzazione
    norm = u.norm()
    if norm == 0:
        raise RuntimeError("Norma del vettore u è zero; impossibile normalizzare.")
    u = u / norm

    return u
