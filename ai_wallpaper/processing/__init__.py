"""
Processing module for AI Wallpaper System
"""

from .upscaler import RealESRGANUpscaler, get_upscaler, upscale_image
from .cpu_offload_refiner import CPUOffloadRefiner
from .smart_refiner import SmartQualityRefiner
from .smart_detector import SmartArtifactDetector
from .aspect_adjuster import AspectAdjuster
from .downsampler import HighQualityDownsampler
from .tiled_refiner import TiledRefiner

__all__ = [
    'RealESRGANUpscaler',
    'get_upscaler',
    'upscale_image',
    'CPUOffloadRefiner',
    'SmartQualityRefiner',
    'SmartArtifactDetector',
    'AspectAdjuster',
    'HighQualityDownsampler',
    'TiledRefiner'
]