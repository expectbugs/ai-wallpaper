"""
Smart configuration loading with environment variable overrides.
Handles cross-platform config file discovery and dynamic path resolution.
"""
import os
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from .path_resolver import get_resolver


logger = logging.getLogger(__name__)


class ConfigLoader:
    """Configuration loader with platform-aware defaults and overrides."""
    
    def __init__(self):
        self.resolver = get_resolver()
        self.config_search_paths = self._get_config_search_paths()
        
    def _get_config_search_paths(self) -> List[Path]:
        """Get ordered list of config file search paths."""
        paths = []
        
        # Environment variable override
        if env_config := os.environ.get('AI_WALLPAPER_CONFIG_FILE'):
            paths.append(Path(env_config))
            
        # User config directory
        user_config = self.resolver.get_config_dir() / 'config.yaml'
        paths.append(user_config)
        
        # Project config directory
        project_config = self.resolver.project_root / 'ai_wallpaper' / 'config'
        paths.append(project_config / 'config.yaml')
        paths.append(project_config / 'defaults.yaml')
        
        # System-wide config (Linux/Mac only)
        if self.resolver.platform != 'Windows':
            paths.append(Path('/etc/ai-wallpaper/config.yaml'))
            
        return paths
        
    def load_with_overrides(self) -> Dict[str, Any]:
        """Load config with environment variable overrides."""
        config = self._load_base_config()
        
        # Apply dynamic path resolution
        config = self._resolve_paths(config)
        
        # Apply environment overrides
        config = self._apply_env_overrides(config)
        
        return config
        
    def _load_base_config(self) -> Dict[str, Any]:
        """Load base configuration from files."""
        config = {}
        
        # Load defaults first
        defaults_path = self.resolver.project_root / 'ai_wallpaper' / 'config' / 'defaults.yaml'
        if defaults_path.exists():
            with open(defaults_path, 'r') as f:
                config = yaml.safe_load(f) or {}
                
        # Then overlay user/system configs
        for config_path in self.config_search_paths:
            if config_path.exists():
                logger.info(f"Loading config from: {config_path}")
                with open(config_path, 'r') as f:
                    user_config = yaml.safe_load(f) or {}
                    config = self._deep_merge(config, user_config)
                break
                
        return config
        
    def _resolve_paths(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve all paths in config to absolute paths."""
        if 'paths' not in config:
            config['paths'] = {}
            
        # Set default paths using resolver
        config['paths'].setdefault('config_dir', str(self.resolver.get_config_dir()))
        config['paths'].setdefault('cache_dir', str(self.resolver.get_cache_dir()))
        config['paths'].setdefault('data_dir', str(self.resolver.get_data_dir()))
        config['paths'].setdefault('log_dir', str(self.resolver.get_log_dir()))
        config['paths'].setdefault('temp_dir', str(self.resolver.get_temp_dir()))
        
        # Resolve model paths
        if 'models' in config:
            if 'search_paths' in config['models']:
                resolved_paths = []
                for path in config['models']['search_paths']:
                    resolved = self._expand_path(path)
                    if resolved.exists():
                        resolved_paths.append(str(resolved))
                config['models']['search_paths'] = resolved_paths
                
        return config
        
    def _apply_env_overrides(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply environment variable overrides."""
        overrides = {
            'system.python_venv': os.environ.get('AI_WALLPAPER_VENV'),
            'system.python': os.environ.get('AI_WALLPAPER_PYTHON'),
            'system.ollama_path': self._find_ollama(),
            'system.realesrgan_path': self._find_realesrgan(),
        }
        
        # Remove None values
        overrides = {k: v for k, v in overrides.items() if v is not None}
        
        # Apply overrides
        for key_path, value in overrides.items():
            self._set_nested(config, key_path, value)
            
        return config
        
    def _find_ollama(self) -> Optional[str]:
        """Find ollama executable."""
        search_paths = [
            '/usr/local/bin',
            '/opt/homebrew/bin',  # macOS ARM
            str(Path.home() / '.local/bin'),
            'C:\\Program Files\\Ollama',  # Windows
            '/snap/bin',  # Snap packages
        ]
        
        if found := self.resolver.find_executable('ollama', search_paths):
            return str(found)
        return None
        
    def _find_realesrgan(self) -> Optional[str]:
        """Find Real-ESRGAN executable."""
        search_paths = [
            str(self.resolver.project_root / 'Real-ESRGAN'),
            str(Path.home() / 'Real-ESRGAN'),
            '/opt/Real-ESRGAN',
            'C:\\Real-ESRGAN',  # Windows
            str(Path.home() / '.local/share/Real-ESRGAN'),
        ]
        
        # Look for the actual executable
        executable_name = 'realesrgan-ncnn-vulkan'
        if self.resolver.platform == 'Windows':
            executable_name += '.exe'
            
        for search_path in search_paths:
            exe_path = Path(search_path) / executable_name
            if exe_path.exists() and os.access(exe_path, os.X_OK):
                return str(exe_path)
                
        # Try finding in PATH
        if found := self.resolver.find_executable(executable_name):
            return str(found)
            
        return None
        
    def _expand_path(self, path_str: str) -> Path:
        """Expand environment variables and ~ in paths."""
        # Expand environment variables
        path_str = os.path.expandvars(path_str)
        # Expand ~
        path = Path(path_str).expanduser()
        return path
        
    def _deep_merge(self, base: Dict, overlay: Dict) -> Dict:
        """Deep merge two dictionaries."""
        result = base.copy()
        
        for key, value in overlay.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
                
        return result
        
    def _set_nested(self, d: Dict, key_path: str, value: Any) -> None:
        """Set nested dictionary value using dot notation."""
        keys = key_path.split('.')
        current = d
        
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
            
        current[keys[-1]] = value


# Singleton instance
_loader = None

def get_config_loader() -> ConfigLoader:
    """Get singleton ConfigLoader instance."""
    global _loader
    if _loader is None:
        _loader = ConfigLoader()
    return _loader