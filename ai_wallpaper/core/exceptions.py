#!/usr/bin/env python3
"""
Custom exceptions for AI Wallpaper System
All exceptions follow the FAIL LOUD philosophy - verbose, informative errors
"""

import traceback
from typing import Optional, Dict, Any

class AIWallpaperError(Exception):
    """Base exception for all AI Wallpaper errors"""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        """Initialize exception with message and optional details
        
        Args:
            message: Error message
            details: Additional error details
        """
        self.message = message
        self.details = details or {}
        
        # Build detailed error message
        full_message = f"\n{'=' * 70}\n"
        full_message += "❌ AI WALLPAPER ERROR ❌\n"
        full_message += f"{'=' * 70}\n\n"
        full_message += f"ERROR: {message}\n"
        
        if self.details:
            full_message += "\nDETAILS:\n"
            # Handle both dict and string details
            if isinstance(self.details, dict):
                for key, value in self.details.items():
                    full_message += f"  {key}: {value}\n"
            else:
                # If details is a string or other type, just display it
                full_message += f"  {self.details}\n"
                
        full_message += f"\n{'=' * 70}\n"
        
        super().__init__(full_message)
        
        
class ConfigurationError(AIWallpaperError):
    """Raised when configuration is invalid or missing"""
    pass
    
    
class ModelError(AIWallpaperError):
    """Base exception for model-related errors"""
    pass
    
    
class ModelNotFoundError(ModelError):
    """Raised when a required model cannot be found"""
    
    def __init__(self, model_name: str, search_paths: list):
        details = {
            "Model": model_name,
            "Searched paths": "\n    ".join(search_paths),
            "Suggestion": "Please check model installation or update paths in config/models.yaml"
        }
        super().__init__(f"Model '{model_name}' not found in any configured path", details)
        
        
class ModelLoadError(ModelError):
    """Raised when a model fails to load"""
    
    def __init__(self, model_name: str, error: Exception):
        details = {
            "Model": model_name,
            "Error type": type(error).__name__,
            "Error message": str(error),
            "Traceback": traceback.format_exc()
        }
        super().__init__(f"Failed to load model '{model_name}'", details)
        
        
class GenerationError(AIWallpaperError):
    """Raised when image generation fails"""
    
    def __init__(self, model_name: str, stage: str, error: Exception):
        details = {
            "Model": model_name,
            "Stage": stage,
            "Error type": type(error).__name__,
            "Error message": str(error),
            "Traceback": traceback.format_exc()
        }
        super().__init__(f"Image generation failed at stage '{stage}'", details)
        
        
class PromptError(AIWallpaperError):
    """Raised when prompt generation fails"""
    
    def __init__(self, prompter: str, error: Exception):
        details = {
            "Prompter": prompter,
            "Error type": type(error).__name__,
            "Error message": str(error),
            "Suggestion": "Check Ollama/API connection and model availability"
        }
        super().__init__(f"Prompt generation failed with '{prompter}'", details)
        
        
class WeatherError(AIWallpaperError):
    """Raised when weather fetching fails"""
    
    def __init__(self, location: str, error: Exception):
        details = {
            "Location": location,
            "Error type": type(error).__name__,
            "Error message": str(error),
            "Suggestion": "Check internet connection and weather API availability"
        }
        super().__init__(f"Failed to fetch weather for '{location}'", details)
        
        
class WallpaperError(AIWallpaperError):
    """Raised when setting wallpaper fails"""
    
    def __init__(self, desktop_env: str, command: str, error: Exception):
        details = {
            "Desktop Environment": desktop_env,
            "Command attempted": command,
            "Error": str(error),
            "Suggestion": "Check desktop environment detection in config/settings.yaml"
        }
        super().__init__(f"Failed to set wallpaper on '{desktop_env}'", details)
        
        
class ResourceError(AIWallpaperError):
    """Raised when system resources are insufficient"""
    
    def __init__(self, resource: str, required: Any, available: Any):
        details = {
            "Resource": resource,
            "Required": required,
            "Available": available,
            "Suggestion": "Close other applications or reduce quality settings"
        }
        super().__init__(f"Insufficient {resource} for operation", details)
        
        
class PipelineError(AIWallpaperError):
    """Raised when pipeline execution fails"""
    
    def __init__(self, pipeline: str, stage: str, error: Exception):
        details = {
            "Pipeline": pipeline,
            "Failed stage": stage,
            "Error type": type(error).__name__,
            "Error message": str(error),
            "Traceback": traceback.format_exc()
        }
        super().__init__(f"Pipeline '{pipeline}' failed at stage '{stage}'", details)
        
        
class UpscalerError(AIWallpaperError):
    """Raised when Real-ESRGAN upscaling fails"""
    
    def __init__(self, input_path: str, error: Exception):
        details = {
            "Input file": input_path,
            "Error": str(error),
            "Suggestion": "Check Real-ESRGAN installation and model availability"
        }
        super().__init__("Real-ESRGAN upscaling failed", details)
        
        
class APIError(AIWallpaperError):
    """Raised when API calls fail"""
    
    def __init__(self, api_name: str, endpoint: str, status_code: Optional[int], error: str):
        details = {
            "API": api_name,
            "Endpoint": endpoint,
            "Status code": status_code or "N/A",
            "Error": error,
            "Suggestion": "Check API key and endpoint configuration"
        }
        super().__init__(f"{api_name} API call failed", details)


class VRAMError(AIWallpaperError):
    """Raised when VRAM operations fail"""
    def __init__(self, operation: str, required_mb: float, available_mb: float, message: str = None):
        self.operation = operation
        self.required_mb = required_mb
        self.available_mb = available_mb
        
        if message is None:
            message = (
                f"Insufficient VRAM for requested operation!\n"
                f"Operation: {operation}\n"
                f"Required: {required_mb:.0f}MB\n"
                f"Available: {available_mb:.0f}MB\n"
                f"This hardware cannot handle the requested quality level.\n"
                f"Upgrade your GPU or reduce resolution requirements."
            )
        
        super().__init__(message)


class LoggingError(AIWallpaperError):
    """Raised when logging setup or operations fail"""
    
    def __init__(self, operation: str, error: str, resolution: str = ""):
        message = (
            f"❌ LOGGING FAILURE: {operation}\n"
            f"Error: {error}"
        )
        if resolution:
            message += f"\nResolution: {resolution}"
        
        super().__init__(message)


class PathError(AIWallpaperError):
    """Raised when path operations fail"""
    
    def __init__(self, operation: str, path: str = "", error: str = "", resolution: str = ""):
        message = f"❌ PATH ERROR: {operation}"
        if path:
            message += f"\nPath: {path}"
        if error:
            message += f"\nError: {error}"
        if resolution:
            message += f"\nResolution: {resolution}"
        
        super().__init__(message)


class FileManagerError(AIWallpaperError):
    """Raised when file management operations fail"""
    
    def __init__(self, operation: str, path: str = "", error: str = "", impact: str = ""):
        message = f"❌ FILE MANAGER ERROR: {operation}"
        if path:
            message += f"\nPath: {path}"
        if error:
            message += f"\nError: {error}"
        if impact:
            message += f"\nImpact: {impact}"
        
        super().__init__(message)


def handle_error(error: Exception, context: str = "") -> None:
    """Handle an error according to fail-loud philosophy
    
    Args:
        error: The exception that occurred
        context: Additional context about what was happening
    """
    print("\n" + "❌" * 35)
    print("FATAL ERROR - CANNOT CONTINUE")
    print("❌" * 35)
    
    if context:
        print(f"\nCONTEXT: {context}")
        
    if isinstance(error, AIWallpaperError):
        # Our custom errors already have detailed formatting
        print(str(error))
    else:
        # Generic errors need formatting
        print(f"\nERROR TYPE: {type(error).__name__}")
        print(f"ERROR MESSAGE: {str(error)}")
        print("\nFULL TRACEBACK:")
        print(traceback.format_exc())
        
    print("\n" + "❌" * 35)
    print("SYSTEM HALTED - FIX ERROR AND RETRY")
    print("❌" * 35 + "\n")
    
    # Always exit with error code
    import sys
    sys.exit(1)