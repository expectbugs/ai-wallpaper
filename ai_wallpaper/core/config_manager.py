#!/usr/bin/env python3
"""
Configuration Manager for AI Wallpaper System
Loads and validates YAML configuration files with fail-loud error handling
"""

import os
import sys
import yaml
import re
import threading
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime

from .exceptions import ConfigurationError
from .config_loader import get_config_loader
from .path_resolver import get_resolver

class ConfigManager:
    """Manages all configuration for the AI Wallpaper system"""
    
    def __init__(self, config_dir: Optional[Path] = None):
        """Initialize configuration manager
        
        Args:
            config_dir: Custom config directory, defaults to package config/
        """
        # Use the new path resolver and config loader
        self.resolver = get_resolver()
        self.config_loader = get_config_loader()
        
        if config_dir is None:
            # Default to package config directory
            package_dir = self.resolver.project_root
            config_dir = package_dir / "ai_wallpaper" / "config"
            
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
        self.resolution = {}  # Resolution config
        
        # Track loaded files for debugging
        self.loaded_files = []
        
    def load_all(self) -> None:
        """Load all configuration files - fail loud on any error"""
        print(f"[CONFIG] Loading configuration from: {self.config_dir}")
        
        # Load base configuration using the new loader
        base_config = self.config_loader.load_with_overrides()
        
        # Load each configuration file
        self.models = self._load_yaml("models.yaml", required=True)
        self.paths = self._merge_with_dynamic_paths(self._load_yaml("paths.yaml", required=True), base_config.get('paths', {}))
        self.weather = self._load_yaml("weather.yaml", required=True)
        self.settings = self._load_yaml("settings.yaml", required=True)
        self.themes = self._load_yaml("themes.yaml", required=True)
        self.system = self._merge_with_dynamic_paths(self._load_yaml("system.yaml", required=False), base_config.get('system', {}))
        self.resolution = self._load_yaml("resolution.yaml", required=True)
        
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
        self.resolution = self._expand_env_vars(self.resolution)
        
        # Apply dynamic path resolution
        self._apply_dynamic_paths()
        
        # Validate configuration
        self._validate_all()
        
        # Count required vs optional files
        required_count = 6  # models, paths, weather, settings, themes, resolution
        optional_count = 1  # system
        loaded_count = len(self.loaded_files)
        
        print(f"[CONFIG] Successfully loaded {loaded_count} configuration files "
              f"({required_count} required, {loaded_count - required_count} optional)")
        
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
                # For optional files, just note they're missing without alarming the user
                print(f"[CONFIG] Optional file {filename} not found - using defaults")
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
        
        # Validate resolution
        self._validate_resolution()
        
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
    
    def _validate_resolution(self) -> None:
        """Validate resolution configuration"""
        if not self.resolution.get('resolution'):
            raise ConfigurationError("No resolution configuration found in resolution.yaml")
            
        res_config = self.resolution['resolution']
        
        # Validate required fields
        required_fields = ['default', 'quality_mode']
        for field in required_fields:
            if field not in res_config:
                raise ConfigurationError(
                    f"Resolution config missing required field: {field}"
                )
                
        # Validate quality mode
        valid_modes = ['fast', 'balanced', 'ultimate']
        if res_config['quality_mode'] not in valid_modes:
            raise ConfigurationError(
                f"Invalid quality_mode '{res_config['quality_mode']}'. "
                f"Must be one of: {', '.join(valid_modes)}"
            )
            
        print(f"[CONFIG] Resolution config loaded: default={res_config['default']}, "
              f"quality={res_config['quality_mode']}")
              
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
        
    def get_path(self, path_key: str) -> Path:
        """Get a configured path
        
        Args:
            path_key: Key in paths configuration
            
        Returns:
            Path object
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
            
        # Convert string to Path object before returning
        return Path(value)
        
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
        """Expand ${VAR} patterns and ~ in config values
        
        Args:
            config: Configuration to expand
            
        Returns:
            Config with expanded environment variables and paths
        """
        if isinstance(config, str):
            # First expand ${VAR} patterns
            pattern = r'\$\{([^}]+)\}'
            
            def replacer(match):
                var_name = match.group(1)
                var_value = os.environ.get(var_name, '')
                # For API keys, return empty string if not set - models will check when needed
                return var_value
            
            expanded = re.sub(pattern, replacer, config)
            
            # Then expand tilde (~) if present
            if '~' in expanded:
                expanded = os.path.expanduser(expanded)
            
            return expanded
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
        
    def _merge_with_dynamic_paths(self, config: Dict[str, Any], dynamic_config: Dict[str, Any]) -> Dict[str, Any]:
        """Merge configuration with dynamic path resolution"""
        # Deep merge, with dynamic config taking precedence
        result = config.copy()
        for key, value in dynamic_config.items():
            if key not in result or result[key] is None:
                result[key] = value
        return result
        
    def _apply_dynamic_paths(self) -> None:
        """Apply dynamic path resolution to paths configuration"""
        # Ensure all paths use resolver
        if 'images_dir' not in self.paths or not self.paths['images_dir']:
            self.paths['images_dir'] = str(self.resolver.get_data_dir() / 'wallpapers')
            
        if 'wallpaper_dir' not in self.paths:
            self.paths['wallpaper_dir'] = str(self.resolver.get_data_dir() / 'wallpapers')
            
        if 'log_dir' not in self.paths:
            self.paths['log_dir'] = str(self.resolver.get_log_dir())
            
        if 'temp_dir' not in self.paths:
            self.paths['temp_dir'] = str(self.resolver.get_temp_dir())
            
        # Update weather cache directory
        if 'cache' in self.weather and 'directory' in self.weather['cache']:
            # Expand environment variables in cache directory
            cache_dir = self.weather['cache']['directory']
            if '${AI_WALLPAPER_CACHE}' in cache_dir:
                self.weather['cache']['directory'] = str(self.resolver.get_cache_dir() / 'weather')

# Singleton instance with thread safety
_config_manager: Optional[ConfigManager] = None
_config_lock = threading.Lock()

def get_config() -> ConfigManager:
    """Get the global configuration manager instance (thread-safe)"""
    global _config_manager
    
    # Double-checked locking pattern for thread safety
    if _config_manager is None:
        with _config_lock:
            # Check again inside the lock
            if _config_manager is None:
                _config_manager = ConfigManager()
                _config_manager.load_all()
    
    return _config_manager