"""
AI Wallpaper Generator Package
Ultra-high-quality 4K wallpaper generation using AI models
"""

__version__ = '4.5.3'
__author__ = 'AI Wallpaper Team'

# Import main components for easier access
from .core import get_config, get_logger
from .cli.main import cli

__all__ = [
    'get_config',
    'get_logger', 
    'cli',
    '__version__'
]