#!/usr/bin/env python3
"""
Config Command Implementation
Configuration management
"""

from typing import Optional
import yaml
import json

from ..core import get_logger, get_config
from ..core.exceptions import ConfigurationError

class ConfigCommand:
    """Handles configuration management"""
    
    def __init__(self, config_dir: Optional[str] = None, verbose: bool = False):
        """Initialize config command
        
        Args:
            config_dir: Custom config directory
            verbose: Enable verbose output
        """
        self.config_dir = config_dir
        self.verbose = verbose
        self.logger = get_logger()
        self.config = get_config()
        
    def show_config(self) -> None:
        """Display current configuration"""
        self.logger.info("=== CURRENT CONFIGURATION ===")
        
        # Show key settings
        self.logger.info("\nGeneral Settings:")
        self.logger.info(f"  Auto-set wallpaper: {self.config.settings.get('wallpaper', {}).get('auto_set_wallpaper')}")
        self.logger.info(f"  Desktop environment: {self.config.settings.get('wallpaper', {}).get('desktop_environment', {}).get('type')}")
        
        self.logger.info("\nEnabled Models:")
        for model_name in self.config.get_enabled_models():
            model_config = self.config.get_model_config(model_name)
            self.logger.info(f"  - {model_config.get('display_name', model_name)}")
            
        self.logger.info("\nPaths:")
        self.logger.info(f"  Images: {self.config.paths.get('images_dir')}")
        self.logger.info(f"  Logs: {self.config.paths.get('logs_dir')}")
        self.logger.info(f"  Cache: {self.config.paths.get('cache_dir')}")
        
        if self.verbose:
            self.logger.info("\nFull configuration:")
            # Pretty print all config
            all_config = {
                'models': self.config.models,
                'paths': self.config.paths,
                'weather': self.config.weather,
                'settings': self.config.settings
            }
            print(yaml.dump(all_config, default_flow_style=False))
            
    def validate_config(self) -> None:
        """Validate all configuration files"""
        self.logger.info("Validating configuration...")
        
        try:
            # Re-run validation
            self.config._validate_all()
            self.logger.info("✓ All configuration files are valid")
            
        except Exception as e:
            self.logger.error(f"✗ Configuration validation failed: {e}")
            raise
            
    def set_config(self, key: str, value: str) -> None:
        """Set a configuration value
        
        Args:
            key: Configuration key (dot notation)
            value: New value
        """
        self.logger.info(f"Setting {key} = {value}")
        
        # Parse value
        try:
            # Try to parse as JSON first (for bools, numbers, lists)
            parsed_value = json.loads(value)
        except json.JSONDecodeError:
            # Not JSON, keep as string
            parsed_value = value
        except Exception as e:
            self.logger.debug(f"Unexpected error parsing value '{value}': {e}. Treating as string.")
            parsed_value = value
            
        # TODO: Implement actual config setting
        self.logger.warning("Config setting not yet implemented")
        self.logger.info("Please edit the YAML files directly for now")
        
    def reset_config(self) -> None:
        """Reset configuration to defaults"""
        self.logger.warning("Reset not yet implemented")
        self.logger.info("Please manually restore the default YAML files")