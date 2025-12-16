"""
repr_tools.py

Funzioni per estrarre rappresentazioni interne dei token in un dato layer del modello.

Queste funzioni sono usate per:
- calcolare u (left vector) tramite get_reprs_at_word_tokens,
- calcolare v tramite get_module_input_output_at_word,
- identificare l'indice esatto del token corrispondente al soggetto.

Compatibile con GPT-2-XL e GPT-J.
"""

from copy import deepcopy
from typing import List

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from . import nethook_light as nethook


# ============================================================
# FUNZIONE 1 — trova l'indice del token corrispondente al soggetto
# ============================================================

def get_words_idxs_in_templates(
    tok: AutoTokenizer,
    context_templates: List[str],
    words: List[str],
    subtoken: str,
) -> List[List[int]]:
    """
    Dato un template tipo "{} plays basketball"
    e un word tipo "Michael Jordan",
    restituisce la posizione *post-tokenizzazione* dell'ULTIMO token della parola.

    È fondamentale per sapere dove leggere la repr del soggetto.
    """

    assert all(tmp.count("{}") == 1 for tmp in context_templates), \
        "Ogni template deve contenere esattamente un placeholder '{}'"

    fill_idxs = [tmp.index("{}") for tmp in context_templates]

    prefixes = [tmp[:fill_idxs[i]] for i, tmp in enumerate(context_templates)]
    suffixes = [tmp[fill_idxs[i] + 2:] for i, tmp in enumerate(context_templates)]
    words = deepcopy(words)

    # Se prefix non è vuoto, deve terminare con spazio
    for i, prefix in enumerate(prefixes):
        if len(prefix) > 0:
            assert prefix[-1] == " "
            prefix = prefix[:-1]
            prefixes[i] = prefix
            words[i] = f" {words[i].strip()}"

    assert len(prefixes) == len(words) == len(suffixes)
    n = len(prefixes)

    batch_tok = tok(prefixes + words + suffixes)

    prefixes_tok = batch_tok[:n]
    words_tok    = batch_tok[n:2*n]
    suffixes_tok = batch_tok[2*n:3*n]

    prefixes_len = [len(x) for x in prefixes_tok]
    words_len    = [len(x) for x in words_tok]
    suffixes_len = [len(x) for x in suffixes_tok]

    if subtoken in ("last", "first_after_last"):
        return [
            [
                prefixes_len[i] 
                + words_len[i] 
                - (1 if subtoken == "last" or suffixes_len[i] == 0 else 0)
            ]
            for i in range(n)
        ]
    elif subtoken == "first":
        return [[prefixes_len[i]] for i in range(n)]

    else:
        raise ValueError(f"Tipo di subtoken non riconosciuto: {subtoken}")


# ============================================================
# FUNZIONE 2 — estrai repr da una lista di indici
# ============================================================

def get_reprs_at_idxs(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    contexts: List[str],
    idxs: List[List[int]],
    layer: int,
    module_template: str,
    track: str = "in",
) -> torch.Tensor:
    """
    Restituisce la rappresentazione media dei token corrispondenti a idxs
    al layer specificato dal template (es: "transformer.h.{}.mlp").

    use:
        get_reprs_at_idxs(model, tok, ["X"], [[token_index]], 18, "transformer.h.{}.mlp", "in")
    """

    assert track in {"in", "out", "both"}
    keep_in  = (track == "in" or track == "both")
    keep_out = (track == "out" or track == "both")

    module_name = module_template.format(layer)

    device = next(model.parameters()).device

    gathered = {"in": [], "out": []}

    def record(cur_repr, batch_idxs, key):
        cur_repr = cur_repr[0] if isinstance(cur_repr, tuple) else cur_repr
        for i, idx_list in enumerate(batch_idxs):
            gathered[key].append(cur_repr[i][idx_list].mean(0))

    # batch per sicurezza (la maggior parte dei casi batch=1)
    def _batch(n):
        for i in range(0, len(contexts), n):
            yield contexts[i:i+n], idxs[i:i+n]

    for batch_contexts, batch_idxs in _batch(128):
        inputs = tok(batch_contexts, return_tensors="pt", padding=True).to(device)

        with torch.no_grad():
            with nethook.Trace(
                module=model,
                layer=module_name,
                retain_input=keep_in,
                retain_output=keep_out,
            ) as tr:
                model(**inputs)

        if keep_in:
            record(tr.input, batch_idxs, "in")
        if keep_out:
            record(tr.output, batch_idxs, "out")

    # Restituisce solo "in" o solo "out" o entrambi
    gathered = {k: torch.stack(v, 0) for k, v in gathered.items() if len(v) > 0}

    if len(gathered) == 1:
        return gathered["in"] if keep_in else gathered["out"]
    else:
        return gathered["in"], gathered["out"]


# ============================================================
# FUNZIONE 3 — estrai repr del soggetto in un template "{}"
# ============================================================

def get_reprs_at_word_tokens(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    context_templates: List[str],
    words: List[str],
    layer: int,
    module_template: str,
    subtoken: str,
    track: str = "in",
) -> torch.Tensor:
    """
    Versione "user friendly" che prende template + parola
    e calcola l'indice corretto da cui estrarre le repr.
    """

    idxs = get_words_idxs_in_templates(tok, context_templates, words, subtoken)

    return get_reprs_at_idxs(
        model=model,
        tok=tok,
        contexts=[context_templates[i].format(words[i]) for i in range(len(words))],
        idxs=idxs,
        layer=layer,
        module_template=module_template,
        track=track,
    )
