#!/usr/bin/env python3
"""
Unified Logging System for AI Wallpaper
Follows FAIL LOUD philosophy - all errors are verbose and visible
"""

import os
import sys
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Any
import traceback

from .config_manager import get_config
from .exceptions import handle_error

class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for console output"""
    
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
        'RESET': '\033[0m'      # Reset
    }
    
    def format(self, record):
        """Format log record with colors if enabled"""
        # Get base format from parent
        log_message = super().format(record)
        
        # Add color if console coloring is enabled
        if hasattr(record, 'color') and record.color:
            color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
            return f"{color}{log_message}{self.COLORS['RESET']}"
            
        return log_message

class AIWallpaperLogger:
    """Unified logger for the AI Wallpaper system"""
    
    def __init__(self, name: str = "AI-Wallpaper", model: Optional[str] = None):
        """Initialize logger
        
        Args:
            name: Logger name
            model: Current model being used (for context)
        """
        self.name = name
        self.model = model or "SYSTEM"
        self.logger = logging.getLogger(name)
        
        # Only set up if not already configured
        if not self.logger.handlers:
            self._setup_logger()
            
    def _setup_logger(self):
        """Set up logger with console and file handlers"""
        try:
            config = get_config()
            log_config = config.settings.get('logging', {})
            
            # Set log level
            level_str = log_config.get('level', 'INFO')
            self.logger.setLevel(getattr(logging, level_str))
            
            # Create formatters
            format_str = log_config.get('format', '[{timestamp}] [{model}] {message}')
            
            # Console handler
            if log_config.get('console', {}).get('enabled', True):
                console_handler = logging.StreamHandler(sys.stdout)
                console_formatter = ColoredFormatter(
                    self._get_format_string(format_str)
                )
                console_handler.setFormatter(console_formatter)
                
                # Add color attribute to records
                if log_config.get('console', {}).get('color', True):
                    console_handler.addFilter(lambda record: setattr(record, 'color', True) or True)
                    
                self.logger.addHandler(console_handler)
                
            # File handler
            if log_config.get('file', {}).get('enabled', True):
                # Get log file path with date substitution
                log_path_template = log_config.get('file', {}).get(
                    'path', 
                    '/home/user/ai-wallpaper/logs/{date}.log'
                )
                log_path = log_path_template.format(
                    date=datetime.now().strftime('%Y-%m-%d')
                )
                
                # Create log directory if needed
                log_dir = Path(log_path).parent
                log_dir.mkdir(parents=True, exist_ok=True)
                
                # Create file handler
                file_handler = logging.FileHandler(log_path, encoding='utf-8')
                file_formatter = logging.Formatter(
                    self._get_format_string(format_str)
                )
                file_handler.setFormatter(file_formatter)
                self.logger.addHandler(file_handler)
                
        except Exception as e:
            # If logger setup fails, print to stderr and continue
            print(f"WARNING: Failed to set up logger: {e}", file=sys.stderr)
            # Set up basic console logging as fallback
            logging.basicConfig(level=logging.INFO)
            
    def _get_format_string(self, template: str) -> str:
        """Convert template format to Python logging format"""
        # Map our template variables to logging attributes
        format_str = template
        format_str = format_str.replace('{timestamp}', '%(asctime)s')
        format_str = format_str.replace('{model}', '%(model)s')
        format_str = format_str.replace('{message}', '%(message)s')
        format_str = format_str.replace('{level}', '%(levelname)s')
        
        return format_str
        
    def _log(self, level: int, message: str, **kwargs):
        """Internal log method with model context"""
        extra = {'model': self.model}
        extra.update(kwargs.get('extra', {}))
        
        self.logger.log(level, message, extra=extra, **kwargs)
        
    def debug(self, message: str, **kwargs):
        """Log debug message"""
        self._log(logging.DEBUG, message, **kwargs)
        
    def info(self, message: str, **kwargs):
        """Log info message"""
        self._log(logging.INFO, message, **kwargs)
        
    def warning(self, message: str, **kwargs):
        """Log warning message"""
        self._log(logging.WARNING, message, **kwargs)
        
    def error(self, message: str, exception: Optional[Exception] = None, **kwargs):
        """Log error message - always verbose with full traceback
        
        Args:
            message: Error message
            exception: Optional exception object for additional context
        """
        error_msg = f"ERROR: {message}"
        
        if exception:
            error_msg += f"\nException Type: {type(exception).__name__}"
            error_msg += f"\nException Message: {str(exception)}"
            error_msg += f"\nTraceback:\n{traceback.format_exc()}"
            
        self._log(logging.ERROR, error_msg, **kwargs)
        
    def critical(self, message: str, exception: Optional[Exception] = None, exit_code: int = 1):
        """Log critical error and exit - ultimate fail loud
        
        Args:
            message: Critical error message
            exception: Optional exception object
            exit_code: Exit code (default 1)
        """
        self._log(logging.CRITICAL, "=" * 70)
        self._log(logging.CRITICAL, "CRITICAL ERROR - SYSTEM CANNOT CONTINUE")
        self._log(logging.CRITICAL, "=" * 70)
        self._log(logging.CRITICAL, message)
        
        if exception:
            self._log(logging.CRITICAL, f"Exception: {type(exception).__name__}: {str(exception)}")
            self._log(logging.CRITICAL, f"Traceback:\n{traceback.format_exc()}")
            
        self._log(logging.CRITICAL, "=" * 70)
        self._log(logging.CRITICAL, "EXITING WITH ERROR")
        self._log(logging.CRITICAL, "=" * 70)
        
        sys.exit(exit_code)
        
    def set_model(self, model: str):
        """Update the current model context"""
        self.model = model
        
    def log_stage(self, stage: str, message: str = ""):
        """Log a pipeline stage transition
        
        Args:
            stage: Stage name
            message: Optional message
        """
        stage_msg = f">>> Stage: {stage}"
        if message:
            stage_msg += f" - {message}"
        self.info(stage_msg)
        
    def log_progress(self, current: int, total: int, prefix: str = "Progress"):
        """Log progress information
        
        Args:
            current: Current item number
            total: Total items
            prefix: Progress prefix
        """
        percentage = (current / total) * 100 if total > 0 else 0
        self.info(f"{prefix}: {current}/{total} ({percentage:.1f}%)")
        
    def log_vram(self, stage: str = ""):
        """Log current VRAM usage
        
        Args:
            stage: Optional stage identifier
        """
        try:
            import torch
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1024**3
                reserved = torch.cuda.memory_reserved() / 1024**3
                msg = f"VRAM Usage: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved"
                if stage:
                    msg = f"[{stage}] {msg}"
                self.info(msg)
        except ImportError:
            pass  # PyTorch not available
            
    def log_separator(self, char: str = "-", length: int = 50):
        """Log a separator line"""
        self.info(char * length)

# Global logger instance
_logger: Optional[AIWallpaperLogger] = None

def get_logger(name: Optional[str] = None, model: Optional[str] = None) -> AIWallpaperLogger:
    """Get logger instance
    
    Args:
        name: Logger name (uses default if None)
        model: Current model context
        
    Returns:
        Logger instance
    """
    global _logger
    
    if _logger is None or name is not None:
        _logger = AIWallpaperLogger(name or "AI-Wallpaper", model)
    elif model is not None:
        _logger.set_model(model)
        
    return _logger

# Convenience functions
def log(message: str, level: str = "info", **kwargs):
    """Quick logging function
    
    Args:
        message: Message to log
        level: Log level (debug, info, warning, error, critical)
        **kwargs: Additional arguments
    """
    logger = get_logger()
    log_func = getattr(logger, level.lower(), logger.info)
    log_func(message, **kwargs)
    
def log_error(message: str, exception: Optional[Exception] = None):
    """Quick error logging"""
    logger = get_logger()
    logger.error(message, exception)
    
def log_critical(message: str, exception: Optional[Exception] = None):
    """Quick critical error logging - will exit"""
    logger = get_logger()
    logger.critical(message, exception)