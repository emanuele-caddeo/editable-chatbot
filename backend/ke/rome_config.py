"""
Explicit ROME configuration wrapper.

This project uses the original ROME implementation, where configuration
is only available through factory methods. The only GPT-2-compatible
factory provided is `for_gpt2_xl()`, which is structurally compatible
with all GPT-2 variants (small, medium, large, xl).
"""

from backend.ke.rome.config import RomeConfig


def get_gpt2_rome_config() -> RomeConfig:
    """
    Return a RomeConfig compatible with GPT-2 models.

    Even when using GPT-2 small, the GPT-2 XL configuration can be safely
    reused, as ROME dynamically operates only on existing layers.
    """

    return RomeConfig.for_gpt2_xl()
