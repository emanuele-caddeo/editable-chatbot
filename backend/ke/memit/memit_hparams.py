from dataclasses import dataclass
from typing import List, Literal

from backend.ke.rome.config import RomeConfig

from backend.ke.memit_config import MemitConfig

@dataclass
class MEMITHyperParams:
    """
    Hyperparameters for the MEMIT (Multi-Edit Model Editing) algorithm.

    Standalone version adapted to the local project structure.
    """

    # =========================================================
    # Method parameters
    # =========================================================

    layers: List[int]
    layer_selection: Literal["all", "random"]

    fact_token: Literal[
        "last",
        "subject_first",
        "subject_last",
        "subject_first_after_last",
    ]

    v_num_grad_steps: int
    v_lr: float
    v_loss_layer: int
    v_weight_decay: float
    clamp_norm_factor: float
    kl_factor: float

    mom2_adjustment: bool
    mom2_update_weight: float

    # =========================================================
    # Module templates
    # =========================================================

    rewrite_module_tmp: str
    layer_module_tmp: str
    mlp_module_tmp: str
    attn_module_tmp: str
    ln_f_module: str
    lm_head_module: str

    # =========================================================
    # Statistics (covariance / mom2)
    # =========================================================

    mom2_dataset: str
    mom2_n_samples: int
    mom2_dtype: str

    # =========================================================
    # Convenience constructor
    # =========================================================

    @staticmethod
    def from_rome_config(
        rome_config: RomeConfig,
        *,
        mom2_dataset: str,
        mom2_n_samples: int,
        mom2_dtype: str = "float32",
        layer_selection: Literal["all", "random"] = "all",
        mom2_adjustment: bool = True,
        mom2_update_weight: float = 1.0,
    ) -> "MEMITHyperParams":
        """
        Build MEMIT hyperparameters starting from an existing RomeConfig.
        """

        return MEMITHyperParams(
            layers=rome_config.layers,
            layer_selection=layer_selection,
            fact_token=rome_config.fact_token,
            v_num_grad_steps=rome_config.v_num_grad_steps,
            v_lr=rome_config.v_lr,
            v_loss_layer=rome_config.v_loss_layer,
            v_weight_decay=rome_config.v_weight_decay,
            clamp_norm_factor=rome_config.clamp_norm_factor,
            kl_factor=rome_config.kl_factor,
            mom2_adjustment=mom2_adjustment,
            mom2_update_weight=mom2_update_weight,

            rewrite_module_tmp=rome_config.rewrite_module_tmp,
            layer_module_tmp=rome_config.layer_module_tmp,
            mlp_module_tmp=rome_config.mlp_module_tmp,
            attn_module_tmp=getattr(rome_config, "attn_module_tmp", ""),
            ln_f_module="transformer.ln_f",
            lm_head_module="lm_head",

            mom2_dataset=mom2_dataset,
            mom2_n_samples=mom2_n_samples,
            mom2_dtype=mom2_dtype,
        )


    @staticmethod
    def from_memit_config(cfg: MemitConfig) -> "MEMITHyperParams":
        """
        Build MEMIT hyperparameters from a MemitConfig.
        """
        return MEMITHyperParams(
            layers=cfg.layers,
            layer_selection=cfg.layer_selection,
            fact_token=cfg.fact_token,

            v_num_grad_steps=cfg.v_num_grad_steps,
            v_lr=cfg.v_lr,
            v_loss_layer=cfg.v_loss_layer,
            v_weight_decay=cfg.v_weight_decay,
            clamp_norm_factor=cfg.clamp_norm_factor,
            kl_factor=cfg.kl_factor,

            mom2_adjustment=cfg.mom2_adjustment,
            mom2_update_weight=cfg.mom2_update_weight,

            rewrite_module_tmp=cfg.rewrite_module_tmp,
            layer_module_tmp=cfg.layer_module_tmp,
            mlp_module_tmp=cfg.mlp_module_tmp,
            attn_module_tmp=cfg.attn_module_tmp,
            ln_f_module=cfg.ln_f_module,
            lm_head_module=cfg.lm_head_module,

            mom2_dataset=cfg.mom2_dataset,
            mom2_n_samples=cfg.mom2_n_samples,
            mom2_dtype=cfg.mom2_dtype,
        )

