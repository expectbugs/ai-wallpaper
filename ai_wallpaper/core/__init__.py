"""
Core components for AI Wallpaper System
"""

from .config_manager import ConfigManager, get_config
from .logger import AIWallpaperLogger, get_logger, log, log_error, log_critical
from .exceptions import (
    AIWallpaperError,
    ConfigurationError,
    ModelError,
    ModelNotFoundError,
    ModelLoadError,
    GenerationError,
    PromptError,
    WeatherError,
    WallpaperError,
    ResourceError,
    PipelineError,
    UpscalerError,
    APIError,
    handle_error
)
from .weather import WeatherClient, get_weather_context
from .wallpaper import WallpaperSetter, set_wallpaper, verify_wallpaper

__all__ = [
    # Config
    'ConfigManager',
    'get_config',
    
    # Logging
    'AIWallpaperLogger',
    'get_logger',
    'log',
    'log_error',
    'log_critical',
    
    # Exceptions
    'AIWallpaperError',
    'ConfigurationError',
    'ModelError',
    'ModelNotFoundError',
    'ModelLoadError',
    'GenerationError',
    'PromptError',
    'WeatherError',
    'WallpaperError',
    'ResourceError',
    'PipelineError',
    'UpscalerError',
    'APIError',
    'handle_error',
    
    # Weather
    'WeatherClient',
    'get_weather_context',
    
    # Wallpaper
    'WallpaperSetter',
    'set_wallpaper',
    'verify_wallpaper'
]