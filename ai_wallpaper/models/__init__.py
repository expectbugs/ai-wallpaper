"""
AI Image Generation Models
"""

# Only import base model by default
from .base_model import BaseImageModel

# Lazy imports for models with heavy dependencies
def __getattr__(name):
    """Lazy load models to avoid dependency issues"""
    if name == 'FluxModel':
        from .flux_model import FluxModel
        return FluxModel
    elif name == 'DalleModel':
        from .dalle_model import DalleModel
        return DalleModel
    elif name == 'GptImageModel':
        from .gpt_image_model import GptImageModel
        return GptImageModel
    elif name == 'SdxlModel':
        from .sdxl_model import SdxlModel
        return SdxlModel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    'BaseImageModel',
    'FluxModel',
    'DalleModel',
    'GptImageModel',
    'SdxlModel'
]