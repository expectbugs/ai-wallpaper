#!/usr/bin/env python3
"""
Configuration Manager for AI Wallpaper System
Loads and validates YAML configuration files with fail-loud error handling
"""

import os
import sys
import yaml
import re
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime

from .exceptions import ConfigurationError

class ConfigManager:
    """Manages all configuration for the AI Wallpaper system"""
    
    def __init__(self, config_dir: Optional[Path] = None):
        """Initialize configuration manager
        
        Args:
            config_dir: Custom config directory, defaults to package config/
        """
        if config_dir is None:
            # Default to package config directory
            package_dir = Path(__file__).parent.parent
            config_dir = package_dir / "config"
            
        self.config_dir = Path(config_dir)
        
        # Verify config directory exists
        if not self.config_dir.exists():
            raise ConfigurationError(
                f"Configuration directory not found: {self.config_dir}\n"
                f"Expected to find configuration files at this location.\n"
                f"Current working directory: {os.getcwd()}"
            )
            
        # Configuration storage
        self.models = {}
        self.paths = {}
        self.weather = {}
        self.settings = {}
        self.themes = {}
        self.system = {}  # New system config
        
        # Track loaded files for debugging
        self.loaded_files = []
        
    def load_all(self) -> None:
        """Load all configuration files - fail loud on any error"""
        print(f"[CONFIG] Loading configuration from: {self.config_dir}")
        
        # Load each configuration file
        self.models = self._load_yaml("models.yaml", required=True)
        self.paths = self._load_yaml("paths.yaml", required=True)
        self.weather = self._load_yaml("weather.yaml", required=True)
        self.settings = self._load_yaml("settings.yaml", required=True)
        self.themes = self._load_yaml("themes.yaml", required=True)
        self.system = self._load_yaml("system.yaml", required=False)  # Optional for backwards compatibility
        
        # If system.yaml exists and has weather config, merge it
        if self.system and 'weather' in self.system:
            # System weather config overrides weather.yaml
            self.weather.update(self.system['weather'])
            
        # Expand environment variables in all configs
        self.models = self._expand_env_vars(self.models)
        self.paths = self._expand_env_vars(self.paths)
        self.weather = self._expand_env_vars(self.weather)
        self.settings = self._expand_env_vars(self.settings)
        self.themes = self._expand_env_vars(self.themes)
        self.system = self._expand_env_vars(self.system)
        
        # Validate configuration
        self._validate_all()
        
        print(f"[CONFIG] Successfully loaded {len(self.loaded_files)} configuration files")
        
    def _load_yaml(self, filename: str, required: bool = True) -> Dict[str, Any]:
        """Load a YAML configuration file
        
        Args:
            filename: Name of the YAML file to load
            required: If True, fail if file doesn't exist
            
        Returns:
            Loaded configuration dictionary
        """
        filepath = self.config_dir / filename
        
        if not filepath.exists():
            if required:
                raise ConfigurationError(
                    f"Required configuration file not found: {filepath}\n"
                    f"Please ensure all configuration files are present."
                )
            else:
                return {}
                
        try:
            with open(filepath, 'r') as f:
                config = yaml.safe_load(f)
                
            if config is None:
                config = {}
                
            self.loaded_files.append(str(filepath))
            return config
            
        except yaml.YAMLError as e:
            raise ConfigurationError(
                f"Failed to parse YAML file: {filepath}\n"
                f"Error: {e}\n"
                f"Please check the YAML syntax."
            )
        except Exception as e:
            raise ConfigurationError(
                f"Failed to load configuration file: {filepath}\n"
                f"Error: {type(e).__name__}: {e}"
            )
            
    def _validate_all(self) -> None:
        """Validate all loaded configuration - fail loud on errors"""
        # Validate models configuration
        self._validate_models()
        
        # Validate paths exist
        self._validate_paths()
        
        # Validate weather configuration
        self._validate_weather()
        
        # Validate settings
        self._validate_settings()
        
        # Validate themes
        self._validate_themes()
        
    def _validate_models(self) -> None:
        """Validate model configuration"""
        if not self.models.get('models'):
            raise ConfigurationError("No models defined in models.yaml")
            
        for model_name, model_config in self.models['models'].items():
            # Required fields
            required_fields = ['class', 'enabled', 'display_name']
            for field in required_fields:
                if field not in model_config:
                    raise ConfigurationError(
                        f"Model '{model_name}' missing required field: {field}"
                    )
                    
            # Validate generation settings
            if 'generation' not in model_config:
                raise ConfigurationError(
                    f"Model '{model_name}' missing generation settings"
                )
                
            # Validate pipeline settings
            if 'pipeline' not in model_config:
                raise ConfigurationError(
                    f"Model '{model_name}' missing pipeline settings"
                )
                
        # Validate random selection if enabled
        if self.models.get('random_selection', {}).get('enabled'):
            weights = self.models['random_selection'].get('model_weights', {})
            if not weights:
                raise ConfigurationError(
                    "Random selection enabled but no model weights defined"
                )
                
    def _validate_paths(self) -> None:
        """Validate path configuration and create directories if needed"""
        # Critical directories that must exist or be created
        critical_dirs = [
            self.paths.get('logs_dir'),
            self.paths.get('images_dir'),
            self.paths.get('cache_dir'),
            self.paths.get('weather_cache_dir')
        ]
        
        for dir_path in critical_dirs:
            if dir_path:
                path = Path(dir_path)
                if not path.exists():
                    try:
                        path.mkdir(parents=True, exist_ok=True)
                        print(f"[CONFIG] Created directory: {path}")
                    except Exception as e:
                        raise ConfigurationError(
                            f"Failed to create directory: {path}\n"
                            f"Error: {e}"
                        )
                        
        # Validate theme database exists
        theme_db = self.paths.get('theme_database')
        if theme_db and not Path(theme_db).exists():
            # This is a warning, not a failure - themes.yaml is the new format
            print(f"[CONFIG] Warning: Theme database file not found: {theme_db}")
            
    def _validate_weather(self) -> None:
        """Validate weather configuration"""
        # Check for coordinates in weather config (could be from system.yaml)
        if 'latitude' not in self.weather or 'longitude' not in self.weather:
            # Fall back to location dict
            if not self.weather.get('location'):
                raise ConfigurationError(
                    "No weather coordinates defined.\n"
                    "Please set latitude and longitude in system.yaml or weather.yaml"
                )
                
            location = self.weather['location']
            if not all(k in location for k in ['latitude', 'longitude']):
                raise ConfigurationError(
                    "Weather location must include latitude and longitude"
                )
            # Move coordinates to top level for easier access
            self.weather['latitude'] = location['latitude']
            self.weather['longitude'] = location['longitude']
            
        # Validate API settings
        if not self.weather.get('api'):
            raise ConfigurationError("No API settings in weather.yaml")
            
    def _validate_settings(self) -> None:
        """Validate general settings"""
        if not self.settings.get('wallpaper'):
            raise ConfigurationError("No wallpaper settings defined")
            
        # Check desktop environment configuration
        desktop_config = self.settings['wallpaper'].get('desktop_environment', {})
        if not desktop_config.get('commands'):
            raise ConfigurationError(
                "No desktop environment commands defined in settings"
            )
            
    def _validate_themes(self) -> None:
        """Validate themes configuration"""
        if not self.themes.get('categories'):
            raise ConfigurationError("No theme categories defined in themes.yaml")
            
        total_weight = 0
        for category_name, category in self.themes['categories'].items():
            if 'weight' not in category:
                raise ConfigurationError(
                    f"Theme category '{category_name}' missing weight"
                )
            total_weight += category['weight']
            
            if not category.get('themes'):
                raise ConfigurationError(
                    f"Theme category '{category_name}' has no themes"
                )
                
        print(f"[CONFIG] Loaded {len(self.themes['categories'])} theme categories "
              f"with total weight: {total_weight}")
              
    def get_model_config(self, model_name: str) -> Dict[str, Any]:
        """Get configuration for a specific model
        
        Args:
            model_name: Name of the model
            
        Returns:
            Model configuration dictionary
        """
        if model_name not in self.models.get('models', {}):
            available = list(self.models.get('models', {}).keys())
            raise ConfigurationError(
                f"Unknown model: {model_name}\n"
                f"Available models: {', '.join(available)}"
            )
            
        return self.models['models'][model_name]
        
    def get_enabled_models(self) -> List[str]:
        """Get list of enabled model names"""
        enabled = []
        for name, config in self.models.get('models', {}).items():
            if config.get('enabled', False):
                enabled.append(name)
        return enabled
        
    def get_path(self, path_key: str) -> str:
        """Get a configured path
        
        Args:
            path_key: Key in paths configuration
            
        Returns:
            Path string
        """
        # Handle nested paths like 'models.flux.primary_paths'
        keys = path_key.split('.')
        value = self.paths
        
        for key in keys:
            if isinstance(value, dict):
                value = value.get(key)
            else:
                value = None
                break
                
        if value is None:
            raise ConfigurationError(f"Path not found in configuration: {path_key}")
            
        return value
        
    def get_setting(self, setting_path: str, default: Any = None) -> Any:
        """Get a setting value using dot notation
        
        Args:
            setting_path: Path to setting (e.g., 'wallpaper.auto_set_wallpaper')
            default: Default value if not found
            
        Returns:
            Setting value
        """
        keys = setting_path.split('.')
        value = self.settings
        
        for key in keys:
            if isinstance(value, dict):
                value = value.get(key)
            else:
                return default
                
        return value if value is not None else default
        
    def get_desktop_command(self, desktop_env: Optional[str] = None) -> Optional[str]:
        """Get wallpaper setting command for desktop environment
        
        Args:
            desktop_env: Desktop environment name, auto-detect if None
            
        Returns:
            Command template string or None if not supported
        """
        de_config = self.settings['wallpaper']['desktop_environment']
        
        if desktop_env is None:
            desktop_env = de_config.get('type', 'auto')
            
        if desktop_env == 'auto':
            # Auto-detect desktop environment
            desktop_env = self._detect_desktop_environment()
            
        if desktop_env and desktop_env in de_config.get('commands', {}):
            return de_config['commands'][desktop_env].get('set')
            
        return None
        
    def _expand_env_vars(self, config: Any) -> Any:
        """Expand ${VAR} patterns in config values
        
        Args:
            config: Configuration to expand
            
        Returns:
            Config with expanded environment variables
        """
        if isinstance(config, str):
            # Find ${VAR} patterns
            pattern = r'\$\{([^}]+)\}'
            
            def replacer(match):
                var_name = match.group(1)
                var_value = os.environ.get(var_name, '')
                # For API keys, return empty string if not set - models will check when needed
                return var_value
            
            return re.sub(pattern, replacer, config)
        elif isinstance(config, dict):
            return {k: self._expand_env_vars(v) for k, v in config.items()}
        elif isinstance(config, list):
            return [self._expand_env_vars(v) for v in config]
        return config
        
    def _detect_desktop_environment(self) -> Optional[str]:
        """Auto-detect the current desktop environment"""
        de_config = self.settings['wallpaper']['desktop_environment']['commands']
        
        for de_name, de_info in de_config.items():
            detect_cmd = de_info.get('detect')
            if detect_cmd:
                # Run detection command
                result = os.system(f"{detect_cmd} >/dev/null 2>&1")
                if result == 0:
                    print(f"[CONFIG] Detected desktop environment: {de_name}")
                    return de_name
                    
        print("[CONFIG] Could not auto-detect desktop environment")
        return None

# Singleton instance
_config_manager: Optional[ConfigManager] = None

def get_config() -> ConfigManager:
    """Get the global configuration manager instance"""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
        _config_manager.load_all()
    return _config_manager