#!/usr/bin/env python3
"""
Path Utilities for AI Wallpaper System
Provides consistent path handling across the codebase
"""

from pathlib import Path
from typing import Union, Optional, List
import json

PathLike = Union[str, Path]

def ensure_path(path: PathLike) -> Path:
    """
    Ensure input is a Path object.
    
    Args:
        path: String or Path object
        
    Returns:
        Path object
    """
    if isinstance(path, Path):
        return path
    return Path(path)

def ensure_str(path: PathLike) -> str:
    """
    Ensure input is a string path.
    
    Args:
        path: String or Path object
        
    Returns:
        String representation of path
    """
    if isinstance(path, str):
        return path
    return str(path)

def path_list_to_str(paths: List[PathLike]) -> List[str]:
    """
    Convert a list of paths to strings.
    
    Args:
        paths: List of Path objects or strings
        
    Returns:
        List of string paths
    """
    return [ensure_str(p) for p in paths]

def resolve_path(path: PathLike, base: Optional[PathLike] = None) -> Path:
    """
    Resolve a path, optionally relative to a base path.
    
    Args:
        path: Path to resolve
        base: Optional base path for relative resolution
        
    Returns:
        Resolved absolute Path object
    """
    path_obj = ensure_path(path)
    
    if base is not None:
        base_path = ensure_path(base)
        if not path_obj.is_absolute():
            path_obj = base_path / path_obj
    
    return path_obj.resolve()

def safe_path_join(*parts: PathLike) -> Path:
    """
    Safely join path parts, handling various input types.
    
    Args:
        *parts: Path components as strings or Path objects
        
    Returns:
        Joined Path object
    """
    if not parts:
        return Path()
    
    result = ensure_path(parts[0])
    for part in parts[1:]:
        result = result / ensure_path(part)
    
    return result

class PathJSONEncoder(json.JSONEncoder):
    """JSON encoder that handles Path objects"""
    
    def default(self, obj):
        if isinstance(obj, Path):
            return str(obj)
        return super().default(obj)

def path_to_json(path: Path) -> str:
    """Convert Path to JSON-serializable string"""
    return str(path)

def json_to_path(path_str: str) -> Path:
    """Convert JSON string back to Path"""
    return Path(path_str)