from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from backend.ke.rome import nethook_light as nethook
from backend.ke.rome.engine import upd_matrix_match_shape
from backend.ke.rome import repr_tools

from .compute_ks import compute_ks
from .compute_z import compute_z
from .memit_hparams import MEMITHyperParams


# ============================================================
# Caches (module-level)
# ============================================================

_CONTEXT_TEMPLATES_CACHE: Optional[List[List[str]]] = None
_COV_CACHE: Dict[Any, torch.Tensor] = {}


# ============================================================
# Public API
# ============================================================

def apply_memit_to_model(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: MEMITHyperParams,
    copy_model: bool = False,
    return_orig_weights: bool = False,
    cache_template: Optional[str] = None,
) -> Tuple[AutoModelForCausalLM, Dict[str, Any]]:
    """
    Apply MEMIT edits to a model.

    Returns:
        (edited_model, orig_weights_dict)
    """

    if copy_model:
        model = deepcopy(model)

    orig_weights: Dict[str, torch.Tensor] = {}

    deltas = execute_memit(
        model=model,
        tok=tok,
        requests=requests,
        hparams=hparams,
        cache_template=cache_template,
    )

    with torch.no_grad():
        for w_name, (key_mat, val_mat) in deltas.items():
            dtype = nethook.get_parameter(model, w_name).dtype
            key_mat = key_mat.to(device=model.device, dtype=dtype)
            val_mat = val_mat.to(device=model.device, dtype=dtype)

            upd_matrix = key_mat @ val_mat.T
            w = nethook.get_parameter(model, w_name)
            upd_matrix = upd_matrix_match_shape(upd_matrix, w.shape)

            if return_orig_weights and w_name not in orig_weights:
                orig_weights[w_name] = w.detach().clone()

            w[...] += upd_matrix.float()

    print(f"[MEMIT] Applied updates to weights: {list(deltas.keys())}")

    return model, orig_weights


# ============================================================
# Core execution (no permanent weight changes)
# ============================================================

def execute_memit(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: MEMITHyperParams,
    cache_template: Optional[str] = None,
) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Execute MEMIT and compute delta matrices.

    Invariant:
        model weights at function entry == model weights at function exit

    Returns:
        deltas dict mapping:
            weight_name -> (adj_k_cpu, resid_cpu)
        where update matrix is: adj_k @ resid.T
    """

    device = next(model.parameters()).device

    # Normalize requests
    requests = deepcopy(requests)
    for r in requests:
        if r.get("target_new", {}).get("str", "") and r["target_new"]["str"][0] != " ":
            r["target_new"]["str"] = " " + r["target_new"]["str"]

    for r in requests[:5]:
        print(
            f"[MEMIT] Request: "
            f"[{r['prompt'].format(r['subject'])}] -> [{r['target_new']['str']}]"
        )

    # Retrieve rewrite weights
    weights = {
        f"{hparams.rewrite_module_tmp.format(layer)}.weight": nethook.get_parameter(
            model, f"{hparams.rewrite_module_tmp.format(layer)}.weight"
        )
        for layer in hparams.layers
    }
    weights_copy = {k: v.detach().clone() for k, v in weights.items()}

    # Context templates (minimal)
    context_templates = get_context_templates()

    # Compute v* (z) for final layer
    z_layer = hparams.layers[-1]
    z_list: List[torch.Tensor] = []

    for r in requests:
        cache_file = (
            Path(
                str(cache_template).format(
                    z_layer, hparams.clamp_norm_factor, r.get("case_id", "default")
                )
            )
            if cache_template is not None
            else None
        )

        loaded = False
        if cache_file is not None and cache_file.exists():
            try:
                data = np.load(cache_file)
                z_list.append(torch.from_numpy(data["v_star"]).to(device))
                loaded = True
            except Exception as e:
                print(f"[MEMIT] Cache read failed ({e}), recomputing...")

        if not loaded:
            z = compute_z(
                model=model,
                tok=tok,
                request=r,
                hparams=hparams,
                layer=z_layer,
                context_templates=context_templates,
            )
            z_list.append(z)

            if cache_file is not None:
                cache_file.parent.mkdir(parents=True, exist_ok=True)
                np.savez(cache_file, v_star=z.detach().cpu().numpy())
                print(f"[MEMIT] Cached v* at {cache_file}")

    zs = torch.stack(z_list, dim=1)  # (hidden, n_edits)

    # Main MEMIT per-layer loop
    deltas: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}

    for i, layer in enumerate(hparams.layers):
        print(f"\n[MEMIT] === Layer {layer} ===")

        # K: shape (hidden, n_edits) then transpose -> (n_edits, hidden)? original uses .T
        ks = compute_ks(
            model=model,
            tok=tok,
            requests=requests,
            hparams=hparams,
            layer=layer,
            context_templates=context_templates,
        ).T  # (n_edits, hidden) in our compute_ks returns (hidden, n_edits)

        # Ensure ks is (hidden, n) like original
        ks = ks.T  # back to (hidden, n)

        print(f"[MEMIT] Writing {ks.size(1)} key/value pairs into layer {layer}")

        # Current outputs at z_layer at lookup token: (hidden, n_edits)
        cur_zs = get_layer_outputs_at_lookup(
            model=model,
            tok=tok,
            requests=requests,
            layer=z_layer,
            module_template=hparams.layer_module_tmp,
            fact_token_strategy=hparams.fact_token,
            context_templates=context_templates,
        )

        # Residual targets (hidden, n_edits)
        targets = zs - cur_zs
        print("[MEMIT] z residual norm:", torch.linalg.norm(targets, dim=0).mean().item())

        # Distribute residual across remaining layers
        resid = targets / (len(hparams.layers) - i)

        # Covariance (mom2) or fallback identity
        cov = get_cov(
            model=model,
            layer_name=hparams.rewrite_module_tmp.format(layer),
            hidden_size=ks.size(0),
            dtype=hparams.mom2_dtype,
        )

        # Compute update in double precision
        ks_d = ks.double()
        resid_d = resid.double()
        cov_d = cov.double()

        # adj_k: solve( mom2_update_weight * cov + K K^T, K )
        adj_k = torch.linalg.solve(
            hparams.mom2_update_weight * cov_d + ks_d @ ks_d.T,
            ks_d,
        )

        upd_matrix = resid_d @ adj_k.T  # (hidden, hidden)

        weight_name = f"{hparams.rewrite_module_tmp.format(layer)}.weight"
        upd_matrix = upd_matrix_match_shape(upd_matrix, weights[weight_name].shape)

        print("[MEMIT] orig norm", torch.linalg.norm(weights[weight_name]).item())
        print("[MEMIT] upd norm", torch.linalg.norm(upd_matrix).item())

        # Temporarily write weight (invariant preserved by restore at end)
        with torch.no_grad():
            weights[weight_name][...] = weights_copy[weight_name] + upd_matrix.float()
            deltas[weight_name] = (adj_k.detach().cpu(), resid.detach().cpu())

        # Cleanup
        cov = cov.cpu()
        torch.cuda.empty_cache()

    # Restore original weights
    with torch.no_grad():
        for k, v in weights.items():
            v[...] = weights_copy[k]

    print(f"[MEMIT] Deltas successfully computed for {list(weights.keys())}")

    return deltas


# ============================================================
# Helpers
# ============================================================

def get_context_templates() -> List[List[str]]:
    """
    Minimal context templates consistent with the "light" setup.
    """
    global _CONTEXT_TEMPLATES_CACHE

    if _CONTEXT_TEMPLATES_CACHE is None:
        _CONTEXT_TEMPLATES_CACHE = [["{}"]]
        print(f"[MEMIT] Using context templates: {_CONTEXT_TEMPLATES_CACHE}")

    return _CONTEXT_TEMPLATES_CACHE


def get_layer_outputs_at_lookup(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    layer: int,
    module_template: str,
    fact_token_strategy: str,
    context_templates: List[List[str]],
) -> torch.Tensor:
    """
    Returns output representations at the fact lookup token for each request.
    Output shape: (hidden_size, n_requests)
    """

    device = next(model.parameters()).device
    outs: List[torch.Tensor] = []

    for r in requests:
        subject = r["subject"]
        prompt = r["prompt"]

        contexts = [
            ctx.format(prompt)
            for ctx_group in context_templates
            for ctx in ctx_group
        ]

        if fact_token_strategy.startswith("subject_"):
            subtoken = fact_token_strategy[len("subject_") :]

            reps_out = repr_tools.get_reprs_at_word_tokens(
                model=model,
                tok=tok,
                context_templates=contexts,
                words=[subject for _ in range(len(contexts))],
                layer=layer,
                module_template=module_template,
                subtoken=subtoken,
                track="out",
            )
            cur = reps_out.mean(0)

        elif fact_token_strategy == "last":
            full_contexts = [ctx.format(subject) for ctx in contexts]

            reps_out = repr_tools.get_reprs_at_idxs(
                model=model,
                tok=tok,
                contexts=full_contexts,
                idxs=[[-1] for _ in range(len(full_contexts))],
                layer=layer,
                module_template=module_template,
                track="out",
            )
            cur = reps_out.mean(0)

        else:
            raise ValueError(f"fact_token={fact_token_strategy} not recognized")

        outs.append(cur.to(device))

    # stack -> (n_requests, hidden) then transpose to (hidden, n_requests)
    return torch.stack(outs, dim=0).T


def get_cov(
    model: AutoModelForCausalLM,
    layer_name: str,
    hidden_size: int,
    dtype: str,
) -> torch.Tensor:
    """
    Retrieve (or cache) covariance matrix for a given layer.
    Light setup fallback: identity covariance if not available.

    This keeps MEMIT runnable without external stats.
    """

    device = next(model.parameters()).device
    model_name = getattr(model.config, "_name_or_path", "unknown").replace("/", "_")
    key = (model_name, layer_name, hidden_size, dtype)

    if key in _COV_CACHE:
        return _COV_CACHE[key].to(device)

    # Fallback identity covariance
    cov = torch.eye(hidden_size, device=device)

    _COV_CACHE[key] = cov.detach().cpu()
    print(f"[MEMIT] Using identity covariance for {model_name} @ {layer_name}")

    return cov
