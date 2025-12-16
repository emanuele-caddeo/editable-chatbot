"""
engine.py

Motore ROME "light" per l'editing di fatti in modelli HF (GPT-2-XL, GPT-J).

Espone:
- execute_rome: calcola i vettori (u, v) per ciascun layer da editare.
- apply_rome_to_model: applica l'aggiornamento rank-1 ai pesi del modello.
- edit_fact: API di alto livello pensata per l'integrazione nel chatbot.

Dipendenze interne:
- config.RomeConfig
- compute_u.compute_u
- compute_v.compute_v
- nethook_light.get_parameter
"""

from copy import deepcopy
from typing import Dict, List, Tuple, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import RomeConfig, ModelType
from .compute_u import compute_u
from .compute_v import compute_v
from . import nethook_light as nethook


# Cache globale per i template di contesto (semplice)
_CONTEXT_TEMPLATES_CACHE: Optional[List[str]] = None


def get_context_templates(config: RomeConfig) -> List[str]:
    """
    Restituisce una lista di template di contesto usata per stabilizzare u e v.
    Versione minimale: una sola frase "{}".
    In futuro puoi espandere con frasi più varie.
    """
    global _CONTEXT_TEMPLATES_CACHE

    if _CONTEXT_TEMPLATES_CACHE is None:
        # Versione minimal: solo il template base "{}"
        _CONTEXT_TEMPLATES_CACHE = ["{}"]

        # Se in futuro vorrai creare contesti extra in modo dinamico,
        # puoi usare config.context_template_lengths per generare frasi
        # con il modello stesso (simile a generate_fast nella repo originale).
        print(f"[ROME] Using minimal context templates: {_CONTEXT_TEMPLATES_CACHE}")

    return _CONTEXT_TEMPLATES_CACHE


# ============================================================
# FUNZIONE DI SUPPORTO: shape dell'update
# ============================================================

def upd_matrix_match_shape(matrix: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    """
    Alcuni modelli (GPT-2 vs GPT-J) hanno trasposti diversi per i pesi MLP.
    Questa funzione garantisce che la matrice di update rispetti la stessa shape,
    eventualmente trasponendola.
    """
    if matrix.shape == shape:
        return matrix
    if matrix.T.shape == shape:
        return matrix.T
    raise ValueError(
        f"Update matrix shape {matrix.shape} non compatibile con weight shape {shape}."
    )


# ============================================================
# CORE: calcolo dei delta (left/right vectors) senza modifiche permanenti
# ============================================================

def execute_rome(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    request: Dict,
    config: RomeConfig,
) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Esegue l'algoritmo ROME per una singola modifica, calcolando
    i vettori (u, v) per ciascun layer specificato in config.layers.

    Invariante:
        lo stato del modello all'inizio e alla fine della funzione è lo stesso.
        (le modifiche ai pesi vengono annullate prima del return)

    Returns:
        dict: {weight_name: (u_vector, v_vector)}
    """

    # Copia del request per non modificarlo a monte
    request = deepcopy(request)

    # Assicuriamoci che il target inizi con spazio per la tokenizzazione
    if request["target_new"]["str"] and request["target_new"]["str"][0] != " ":
        request["target_new"]["str"] = " " + request["target_new"]["str"]

    print(
        f"[ROME] Executing ROME for update:\n"
        f"  [{request['prompt'].format(request['subject'])}] "
        f"-> [{request['target_new']['str']}]"
    )

    # Recupera i pesi da aggiornare
    weights = {
        f"{config.rewrite_module_tmp.format(layer)}.weight": nethook.get_parameter(
            model, f"{config.rewrite_module_tmp.format(layer)}.weight"
        )
        for layer in config.layers
    }

    # Copia degli originali per poter ripristinare
    weights_copy = {name: w.detach().clone() for name, w in weights.items()}

    deltas: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}

    ctx_templates = get_context_templates(config)

    # Loop sui layer da editare (ordinati)
    for layer in sorted(config.layers):
        print(f"[ROME] Processing layer {layer}...")

        # 1) calcola u
        u_vec = compute_u(
            model=model,
            tok=tok,
            request=request,
            config=config,
            layer=layer,
            context_templates=ctx_templates,
        )
        print(f"[ROME] Left vector (u) shape: {u_vec.shape}")

        # 2) calcola v
        v_vec = compute_v(
            model=model,
            tok=tok,
            request=request,
            config=config,
            layer=layer,
            left_vector=u_vec,
            context_templates=ctx_templates,
        )
        print(f"[ROME] Right vector (v) shape: {v_vec.shape}")

        # 3) costruisce l'update temporaneo per questo layer
        weight_name = f"{config.rewrite_module_tmp.format(layer)}.weight"
        w = weights[weight_name]
        v_vec = v_vec.to(u_vec.dtype)
        upd_matrix = u_vec.unsqueeze(1) @ v_vec.unsqueeze(0)
        upd_matrix = upd_matrix_match_shape(upd_matrix, w.shape)

        with torch.no_grad():
            w[...] += upd_matrix  # applichiamo temporaneamente
        deltas[weight_name] = (u_vec.detach(), v_vec.detach())

    # Ripristina lo stato originale dei pesi
    with torch.no_grad():
        for name, w in weights.items():
            w[...] = weights_copy[name]

    print(f"[ROME] Deltas computed for layers: {list(deltas.keys())}")

    return deltas


# ============================================================
# APPLICAZIONE EFFETTIVA DELL'UPDATE AL MODELLO
# ============================================================

def apply_rome_to_model(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    config: RomeConfig,
    copy_model: bool = False,
    return_orig_weights: bool = False,
) -> Tuple[AutoModelForCausalLM, Dict[str, torch.Tensor]]:
    """
    Applica la modifica ROME al modello.

    Args:
        model: modello HF già caricato
        tok: tokenizer HF
        requests: lista di richieste (dict) nel formato:
            {
              "prompt": "... {} ...",
              "subject": "Canada",
              "target_new": {"str": "Toronto"}
            }
        config: RomeConfig per il modello
        copy_model: se True, clona il modello prima di modificare i pesi
        return_orig_weights: se True, restituisce anche una mappa
                             {weight_name: weight_original}

    Returns:
        (model_modificato, orig_weights_dict)
    """

    if copy_model:
        model = deepcopy(model)

    orig_weights: Dict[str, torch.Tensor] = {}

    for i, request in enumerate(requests):
        deltas = execute_rome(model, tok, request, config)

        with torch.no_grad():
            for weight_name, (u_vec, v_vec) in deltas.items():
                w = nethook.get_parameter(model, weight_name)
                upd_matrix = u_vec.unsqueeze(1) @ v_vec.unsqueeze(0)
                upd_matrix = upd_matrix_match_shape(upd_matrix, w.shape)

                if return_orig_weights and weight_name not in orig_weights:
                    # Salviamo solo la prima volta l'originale
                    orig_weights[weight_name] = w.detach().clone()

                w[...] += upd_matrix

        print(f"[ROME] Applied update to weights: {list(deltas.keys())}")

    return model, orig_weights


# ============================================================
# API DI ALTO LIVELLO PER IL CHATBOT: edit_fact
# ============================================================

def _infer_default_config(model: AutoModelForCausalLM) -> RomeConfig:
    """
    Se non viene passato un RomeConfig esplicito, prova a inferirlo
    dal nome del modello HF.
    """
    name = getattr(model.config, "_name_or_path", "").lower()

    if "gpt-j" in name or "gptj" in name:
        print("[ROME] Auto-config: detected GPT-J, using RomeConfig.for_gptj()")
        return RomeConfig.for_gptj()

    if "gpt2-xl" in name or ("gpt2" in name and getattr(model.config, "n_layer", 0) == 48):
        print("[ROME] Auto-config: detected GPT-2-XL, using RomeConfig.for_gpt2_xl()")
        return RomeConfig.for_gpt2_xl()

    raise ValueError(
        f"Impossibile inferire automaticamente RomeConfig per il modello '{name}'. "
        "Passa esplicitamente una RomeConfig a edit_fact()."
    )


def edit_fact(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    subject: str,
    prompt: str,
    new_object: str,
    config: Optional[RomeConfig] = None,
    copy_model: bool = False,
    return_orig_weights: bool = True,
) -> Tuple[AutoModelForCausalLM, Dict[str, torch.Tensor]]:
    """
    Funzione di alto livello pensata per l'integrazione nel chatbot.

    Esempio d'uso:
        config = RomeConfig.for_gpt2_xl()
        model, orig = edit_fact(
            model,
            tok,
            subject="Canada",
            prompt="The capital of {} is",
            new_object="Toronto",
            config=config,
        )

    Args:
        model: modello HF (GPT-2-XL, GPT-J, ...)
        tok: tokenizer
        subject: stringa del soggetto da modificare ("Canada")
        prompt: frase con {} come placeholder ("The capital of {} is")
        new_object: nuovo oggetto target ("Toronto")
        config: RomeConfig; se None, proverà a inferirla dal modello
        copy_model: se True, viene clonato il modello prima dell'edit
        return_orig_weights: se True, ritorna anche una mappa dei pesi originali

    Returns:
        (model_editato, orig_weights_dict)
    """

    if config is None:
        config = _infer_default_config(model)

    request = {
        "prompt": prompt,
        "subject": subject,
        "target_new": {"str": new_object},
    }

    model_edit, orig_weights = apply_rome_to_model(
        model=model,
        tok=tok,
        requests=[request],
        config=config,
        copy_model=copy_model,
        return_orig_weights=return_orig_weights,
    )

    return model_edit, orig_weights
