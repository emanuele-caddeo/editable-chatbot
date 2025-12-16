from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple


class ModelType(str, Enum):
    GPT2_XL = "gpt2-xl"
    GPTJ_6B = "gpt-j-6b"


@dataclass
class RomeConfig:
    # Identificatore del modello
    model_type: ModelType

    # Layer su cui applicare ROME (rank-1 updates)
    layers: List[int]

    # Token strategy: "last" va bene per GPT-2 e GPT-J
    fact_token: str = "last"

    # Parametri dell’ottimizzazione del vettore v
    v_num_grad_steps: int = 80
    v_lr: float = 1e-1
    v_loss_layer: int = 0
    v_weight_decay: float = 1e-3
    clamp_norm_factor: float = 4.0
    kl_factor: float = 0.1

    # Per ora teniamo tutto senza mom2_adjustment
    mom2_adjustment: bool = False

    # Template delle frasi usate durante la costruzione di u e v
    context_template_lengths: List[Tuple[int, int]] = None

    # Template dei moduli da modificare nei vari modelli
    rewrite_module_tmp: str = ""
    layer_module_tmp: str = ""
    mlp_module_tmp: str = ""

    @staticmethod
    def for_gpt2_xl() -> "RomeConfig":
        """
        Configurazione ottimizzata per GPT-2-XL come in ROME originale.
        """
        return RomeConfig(
            model_type=ModelType.GPT2_XL,
            layers=[18],  # Layer mediano scelto dagli autori ROME
            v_loss_layer=36,
            rewrite_module_tmp="transformer.h.{}.mlp.c_proj",
            layer_module_tmp="transformer.h.{}",
            mlp_module_tmp="transformer.h.{}.mlp",
            context_template_lengths=[],
        )

    @staticmethod
    def for_gptj() -> "RomeConfig":
        """
        Configurazione ottimizzata per GPT-J 6B come in ROME.
        """
        return RomeConfig(
            model_type=ModelType.GPTJ_6B,
            layers=[10],  # Layer mediano adatto a GPT-J
            v_loss_layer=20,
            rewrite_module_tmp="transformer.h.{}.mlp.fc_out",
            layer_module_tmp="transformer.h.{}",
            mlp_module_tmp="transformer.h.{}.mlp",
            context_template_lengths=[],
        )
