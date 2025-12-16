"""
nethook_light.py

Versione semplificata di util.nethook usata da ROME.
- Trace: traccia input/output di UN solo modulo.
- TraceDict: traccia (ed eventualmente modifica) output di PIÙ moduli.
- get_parameter: recupera un parametro del modello dato il nome "a.b.c.weight".
- set_requires_grad: abilita/disabilita il grad per tutti i parametri del modello.
"""

from typing import Any, Callable, Dict, Iterable, Optional, Union

import torch
import torch.nn as nn


ModuleOrName = Union[str, nn.Module]


def _resolve_module(root: nn.Module, layer: ModuleOrName) -> nn.Module:
    """
    Dato il modello root e un nome di layer (stringa) o un modulo,
    restituisce il modulo corrispondente.
    """
    if isinstance(layer, nn.Module):
        return layer

    # layer è una stringa: usiamo named_modules()
    modules = dict(root.named_modules())
    if layer not in modules:
        raise KeyError(f"Layer '{layer}' non trovato in model.named_modules().")
    return modules[layer]


class Trace:
    """
    Context manager per tracciare input e/o output di un singolo modulo.

    Uso tipico:
        with Trace(model, "transformer.h.18.mlp", retain_input=True, retain_output=True) as tr:
            model(**inputs)
        hidden_in = tr.input
        hidden_out = tr.output
    """

    def __init__(
        self,
        module: nn.Module,
        layer: ModuleOrName,
        retain_input: bool = True,
        retain_output: bool = True,
    ) -> None:
        self.root = module
        self.layer_identifier = layer
        self.retain_input = retain_input
        self.retain_output = retain_output

        self.input = None
        self.output = None

        self._hook_handle: Optional[torch.utils.hooks.RemovableHandle] = None

    def __enter__(self) -> "Trace":
        target_module = _resolve_module(self.root, self.layer_identifier)

        def hook(mod, inp, out):
            if self.retain_input:
                self.input = inp
            if self.retain_output:
                self.output = out
            # Non modifichiamo l'output: non ritorniamo nulla.

        self._hook_handle = target_module.register_forward_hook(hook)
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None


class TraceDict:
    """
    Context manager per tracciare input/output di più moduli contemporaneamente,
    e/o modificare l'output con una funzione edit_output.

    Uso tipico (come in compute_v di ROME):

        with TraceDict(
            module=model,
            layers=[layer_name_1, layer_name_2],
            retain_input=False,
            retain_output=True,
            edit_output=edit_output_fn,
        ) as tr:
            logits = model(**inputs).logits

    In questo repo usiamo principalmente edit_output per iniettare delta
    nell'output di un MLP; i valori tracciati (input/output) non sono
    quasi mai letti, ma li manteniamo per compatibilità.
    """

    def __init__(
        self,
        module: nn.Module,
        layers: Iterable[ModuleOrName],
        retain_input: bool = False,
        retain_output: bool = True,
        edit_output: Optional[Callable[[Any, str], Any]] = None,
    ) -> None:
        self.root = module
        self.layer_identifiers = list(layers)
        self.retain_input = retain_input
        self.retain_output = retain_output
        self.edit_output = edit_output

        # Dizionario: chiave = layer_identifier (stringa o modulo), val = {"input": ..., "output": ...}
        self.traces: Dict[ModuleOrName, Dict[str, Any]] = {}
        self._hook_handles: Dict[ModuleOrName, torch.utils.hooks.RemovableHandle] = {}

    def __enter__(self) -> "TraceDict":
        for layer in self.layer_identifiers:
            target_module = _resolve_module(self.root, layer)
            key = layer  # usiamo direttamente l'identificatore passato (di solito stringa)

            self.traces[key] = {"input": None, "output": None}

            def make_hook(hook_key: ModuleOrName):
                def hook(mod, inp, out):
                    cur_out = out

                    # Applichiamo la funzione di editing, se presente
                    if self.edit_output is not None:
                        # Per come viene usato in ROME, hook_key è una stringa layer_name
                        cur_out = self.edit_output(cur_out, hook_key)

                    # Salvataggio input/output se richiesti
                    if self.retain_input:
                        self.traces[hook_key]["input"] = inp
                    if self.retain_output:
                        self.traces[hook_key]["output"] = cur_out

                    # Se modifichiamo l'output, dobbiamo restituirlo
                    return cur_out

                return hook

            handle = target_module.register_forward_hook(make_hook(key))
            self._hook_handles[key] = handle

        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        for handle in self._hook_handles.values():
            handle.remove()
        self._hook_handles.clear()

    # Permette di fare tr["layer_name"]["output"], se mai servisse
    def __getitem__(self, key: ModuleOrName) -> Dict[str, Any]:
        return self.traces[key]


def get_parameter(model: nn.Module, name: str) -> nn.Parameter:
    """
    Restituisce il parametro del modello dato un nome tipo "transformer.h.18.mlp.c_proj.weight".
    """
    parts = name.split(".")
    obj: Any = model
    for p in parts:
        if not hasattr(obj, p):
            raise AttributeError(f"Impossibile trovare attributo '{p}' in '{obj.__class__.__name__}' "
                                 f"navigando il percorso '{name}'")
        obj = getattr(obj, p)
    if not isinstance(obj, nn.Parameter) and not isinstance(obj, torch.Tensor):
        raise TypeError(f"L'oggetto risolto da '{name}' non è un parametro/tenore ma {type(obj)}")
    return obj


def set_requires_grad(model: nn.Module, flag: bool) -> None:
    """
    Abilita/disabilita il grad per tutti i parametri del modello.
    Usato in compute_v per congelare il modello tranne il vettore delta ottimizzato.
    """
    for p in model.parameters():
        p.requires_grad = flag
