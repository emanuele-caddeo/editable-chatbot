from dataclasses import dataclass
from enum import Enum
from typing import List, Literal


class ModelType(str, Enum):
    GPT2_XL = "gpt2-xl"
    GPTJ_6B = "gpt-j-6b"
    PHI_2 = "phi-2"


@dataclass
class MemitConfig:
    """
    Configuration (policy-level) for MEMIT.

    This class defines:
    - which model architecture is used
    - which layers are edited
    - which module templates apply
    - sensible default hyperparameters

    It mirrors the role of RomeConfig for MEMIT.
    """

    # ---------------------------------------------------------
    # Model identification
    # ---------------------------------------------------------
    model_type: ModelType

    # ---------------------------------------------------------
    # Layers and selection policy
    # ---------------------------------------------------------
    layers: List[int]
    layer_selection: Literal["all", "random"] = "all"

    # ---------------------------------------------------------
    # Fact lookup strategy
    # ---------------------------------------------------------
    fact_token: Literal[
        "last",
        "subject_first",
        "subject_last",
        "subject_first_after_last",
    ] = "last"

    # ---------------------------------------------------------
    # Optimization hyperparameters (v*)
    # ---------------------------------------------------------
    v_num_grad_steps: int = 80
    v_lr: float = 1e-1
    v_loss_layer: int = 0
    v_weight_decay: float = 1e-3
    clamp_norm_factor: float = 4.0
    kl_factor: float = 0.1

    # ---------------------------------------------------------
    # MEMIT-specific parameters
    # ---------------------------------------------------------
    mom2_adjustment: bool = False
    mom2_update_weight: float = 1.0

    # ---------------------------------------------------------
    # Statistics (mom2)
    # ---------------------------------------------------------
    mom2_dataset: str = "dummy"
    mom2_n_samples: int = 1
    mom2_dtype: str = "float32"

    # ---------------------------------------------------------
    # Module templates
    # ---------------------------------------------------------
    rewrite_module_tmp: str = ""
    layer_module_tmp: str = ""
    mlp_module_tmp: str = ""
    attn_module_tmp: str = ""
    ln_f_module: str = "transformer.ln_f"
    lm_head_module: str = "lm_head"

    # =========================================================
    # Factory methods (like RomeConfig.for_gptj / for_gpt2_xl)
    # =========================================================

    @staticmethod
    def for_gpt2_xl() -> "MemitConfig":
        return MemitConfig(
            model_type=ModelType.GPT2_XL,
            layers=[16, 18, 20],
            v_loss_layer=36,
            rewrite_module_tmp="transformer.h.{}.mlp.c_proj",
            layer_module_tmp="transformer.h.{}",
            mlp_module_tmp="transformer.h.{}.mlp",
        )

    @staticmethod
    def for_gptj() -> "MemitConfig":
        return MemitConfig(
            model_type=ModelType.GPTJ_6B,
            layers=[8, 10, 12],
            v_loss_layer=20,
            rewrite_module_tmp="transformer.h.{}.mlp.fc_out",
            layer_module_tmp="transformer.h.{}",
            mlp_module_tmp="transformer.h.{}.mlp",
        )

    @staticmethod
    def for_phi2() -> "MemitConfig":
        """
        Sensible defaults for microsoft/phi-2.
        """
        return MemitConfig(
            model_type=ModelType.PHI_2,
            layers=[12, 14, 16],
            v_loss_layer=24,
            rewrite_module_tmp="model.layers.{}.mlp.fc2",
            layer_module_tmp="model.layers.{}",
            mlp_module_tmp="model.layers.{}.mlp",
        )
