"""
Processing module for AI Wallpaper System
"""

from .upscaler import RealESRGANUpscaler, get_upscaler, upscale_image

__all__ = [
    'RealESRGANUpscaler',
    'get_upscaler',
    'upscale_image'
]