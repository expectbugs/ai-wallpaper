"""
Cross-platform path resolution system for AI Wallpaper.
Handles all path operations dynamically based on platform and environment.
"""
import os
import sys
import platform
import shutil
import tempfile
from pathlib import Path
from typing import Optional, List, Dict


class PathResolver:
    """Central path resolution for cross-platform compatibility."""
    
    def __init__(self):
        self.platform = platform.system()  # 'Linux', 'Darwin', 'Windows'
        self.home = Path.home()
        self.project_root = self._find_project_root()
        
    def _find_project_root(self) -> Path:
        """Find project root by looking for marker files."""
        current = Path(__file__).resolve().parent
        while current != current.parent:
            if (current / 'setup.py').exists() or (current / 'pyproject.toml').exists():
                return current
            current = current.parent
        return Path.cwd()
        
    def get_config_dir(self) -> Path:
        """Get platform-appropriate config directory."""
        if env_dir := os.environ.get('AI_WALLPAPER_CONFIG'):
            return Path(env_dir)
            
        if self.platform == 'Windows':
            base = Path(os.environ.get('APPDATA', self.home / 'AppData/Roaming'))
            return base / 'ai-wallpaper'
        elif self.platform == 'Darwin':  # macOS
            return self.home / 'Library/Application Support/ai-wallpaper'
        else:  # Linux/Unix
            xdg_config = os.environ.get('XDG_CONFIG_HOME', self.home / '.config')
            return Path(xdg_config) / 'ai-wallpaper'
            
    def get_cache_dir(self) -> Path:
        """Get platform-appropriate cache directory."""
        if env_dir := os.environ.get('AI_WALLPAPER_CACHE'):
            return Path(env_dir)
            
        if self.platform == 'Windows':
            base = Path(os.environ.get('LOCALAPPDATA', self.home / 'AppData/Local'))
            return base / 'ai-wallpaper/cache'
        elif self.platform == 'Darwin':
            return self.home / 'Library/Caches/ai-wallpaper'
        else:
            xdg_cache = os.environ.get('XDG_CACHE_HOME', self.home / '.cache')
            return Path(xdg_cache) / 'ai-wallpaper'
            
    def get_data_dir(self) -> Path:
        """Get platform-appropriate data directory."""
        if env_dir := os.environ.get('AI_WALLPAPER_DATA'):
            return Path(env_dir)
            
        if self.platform == 'Windows':
            base = Path(os.environ.get('LOCALAPPDATA', self.home / 'AppData/Local'))
            return base / 'ai-wallpaper/data'
        elif self.platform == 'Darwin':
            return self.home / 'Library/Application Support/ai-wallpaper/data'
        else:
            xdg_data = os.environ.get('XDG_DATA_HOME', self.home / '.local/share')
            return Path(xdg_data) / 'ai-wallpaper'
            
    def get_temp_dir(self) -> Path:
        """Get platform-appropriate temporary directory."""
        if env_dir := os.environ.get('AI_WALLPAPER_TEMP'):
            return Path(env_dir)
        return Path(tempfile.gettempdir())
            
    def find_executable(self, name: str, search_paths: Optional[List[str]] = None) -> Optional[Path]:
        """Find executable in PATH or common locations."""
        # First check environment variable override
        env_var = f'AI_WALLPAPER_{name.upper()}_PATH'
        if env_path := os.environ.get(env_var):
            path = Path(env_path)
            if path.exists() and os.access(path, os.X_OK):
                return path
                
        # Try shutil.which
        if found := shutil.which(name):
            return Path(found)
            
        # Check additional search paths
        if search_paths:
            for search_path in search_paths:
                path = Path(search_path) / name
                if path.exists() and os.access(path, os.X_OK):
                    return path
                # Windows executable extensions
                if self.platform == 'Windows':
                    for ext in ['.exe', '.bat', '.cmd']:
                        path_with_ext = Path(search_path) / f"{name}{ext}"
                        if path_with_ext.exists() and os.access(path_with_ext, os.X_OK):
                            return path_with_ext
                    
        return None
        
    def get_model_search_paths(self) -> List[Path]:
        """Get list of paths to search for models."""
        paths = []
        
        # Environment variable overrides
        if hf_home := os.environ.get('HF_HOME'):
            paths.append(Path(hf_home))
        if transformers_cache := os.environ.get('TRANSFORMERS_CACHE'):
            paths.append(Path(transformers_cache))
            
        # Platform-specific HuggingFace cache
        if self.platform == 'Windows':
            paths.append(self.home / '.cache/huggingface')
        else:
            cache_home = os.environ.get('XDG_CACHE_HOME', self.home / '.cache')
            paths.append(Path(cache_home) / 'huggingface')
            
        # Project-relative paths
        paths.append(self.project_root / 'models')
        paths.append(self.get_data_dir() / 'models')
        
        # Additional common locations
        paths.append(self.home / '.local/share/huggingface')
        paths.append(Path('/opt/ai-models'))  # System-wide models
        
        return [p for p in paths if p.exists()]
        
    def get_log_dir(self) -> Path:
        """Get platform-appropriate log directory."""
        if env_dir := os.environ.get('AI_WALLPAPER_LOGS'):
            return Path(env_dir)
            
        if self.platform == 'Windows':
            return self.get_data_dir() / 'logs'
        else:
            # Respect XDG_STATE_HOME for logs on Linux
            state_home = os.environ.get('XDG_STATE_HOME', self.home / '.local/state')
            return Path(state_home) / 'ai-wallpaper'
            
    def ensure_directories(self) -> None:
        """Ensure all required directories exist with proper permissions."""
        dirs_to_create = [
            self.get_config_dir(),
            self.get_cache_dir(),
            self.get_data_dir(),
            self.get_log_dir(),
            self.get_temp_dir() / 'ai-wallpaper'
        ]
        
        for directory in dirs_to_create:
            directory.mkdir(parents=True, exist_ok=True)
            # Set proper permissions (755 for directories)
            if self.platform != 'Windows':
                os.chmod(directory, 0o755)


# Singleton instance
_resolver = None

def get_resolver() -> PathResolver:
    """Get singleton PathResolver instance."""
    global _resolver
    if _resolver is None:
        _resolver = PathResolver()
    return _resolver