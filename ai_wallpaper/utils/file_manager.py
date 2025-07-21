#!/usr/bin/env python3
"""
File Manager for AI Wallpaper System
Handles safe file operations and path management
"""

import os
import shutil
import hashlib
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime
import re

from ..core import get_logger, get_config
from ..core.exceptions import FileManagerError, ConfigurationError


class FileManager:
    """Manages file operations safely"""
    
    def __init__(self):
        """Initialize file manager"""
        self.logger = get_logger(model="FileManager")
        self.config = get_config()
        self._ensure_directories()
        
    def _ensure_directories(self) -> None:
        """Ensure all required directories exist"""
        paths = self.config.paths
        
        # Core directories
        directories = [
            paths.get('images_dir', '/home/user/ai-wallpaper/images'),
            paths.get('logs_dir', '/home/user/ai-wallpaper/logs'),
            paths.get('cache_dir', '/home/user/ai-wallpaper/.cache'),
            paths.get('backup_dir', '/home/user/ai-wallpaper/backups'),
        ]
        
        for directory in directories:
            path = Path(directory).expanduser()
            if not path.exists():
                path.mkdir(parents=True, exist_ok=True)
                self.logger.debug(f"Created directory: {path}")
                
    def get_images_dir(self) -> Path:
        """Get images directory path
        
        Returns:
            Path to images directory
        """
        return Path(self.config.paths.get('images_dir', '/home/user/ai-wallpaper/images')).expanduser()
        
    def get_logs_dir(self) -> Path:
        """Get logs directory path
        
        Returns:
            Path to logs directory
        """
        return Path(self.config.paths.get('logs_dir', '/home/user/ai-wallpaper/logs')).expanduser()
        
    def get_cache_dir(self) -> Path:
        """Get cache directory path
        
        Returns:
            Path to cache directory
        """
        return Path(self.config.paths.get('cache_dir', '/home/user/ai-wallpaper/.cache')).expanduser()
        
    def create_image_path(
        self,
        prefix: str = "",
        model_name: Optional[str] = None,
        prompt_excerpt: Optional[str] = None,
        extension: str = "png"
    ) -> Path:
        """Create a unique image file path
        
        Args:
            prefix: Filename prefix
            model_name: Model name to include
            prompt_excerpt: Excerpt from prompt
            extension: File extension
            
        Returns:
            Path to new image file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Build filename parts
        parts = []
        if prefix:
            parts.append(prefix)
        if model_name:
            parts.append(model_name.lower().replace(' ', '_'))
        parts.append(timestamp)
        if prompt_excerpt:
            safe_excerpt = self.sanitize_filename(prompt_excerpt, max_length=50)
            parts.append(safe_excerpt)
            
        filename = "_".join(parts) + f".{extension}"
        
        return self.get_images_dir() / filename
        
    def sanitize_filename(self, text: str, max_length: int = 50) -> str:
        """Convert text to a safe filename
        
        Args:
            text: Text to sanitize
            max_length: Maximum length
            
        Returns:
            Safe filename
        """
        # Remove non-alphanumeric characters, keep spaces
        safe = re.sub(r'[^a-zA-Z0-9\s\-]', '', text)
        # Replace spaces with underscores
        safe = safe.replace(' ', '_')
        # Remove multiple underscores
        safe = re.sub(r'_+', '_', safe)
        # Truncate to max length
        safe = safe[:max_length]
        # Remove trailing underscores
        safe = safe.rstrip('_')
        return safe.lower()
        
    def save_image(
        self,
        image,
        path: Optional[Path] = None,
        format: str = "PNG",
        quality: int = 100,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Path:
        """Save image with metadata
        
        Args:
            image: PIL Image object
            path: Output path (auto-generated if None)
            format: Image format
            quality: JPEG quality (ignored for PNG)
            metadata: Metadata to save alongside
            
        Returns:
            Path where image was saved
        """
        if path is None:
            path = self.create_image_path()
            
        # Ensure parent directory exists
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save image
        if format.upper() == "PNG":
            # Use lossless save for PNG
            from .lossless_save import save_lossless_png
            save_lossless_png(image, path)
        else:
            image.save(path, format, quality=quality)
            
        self.logger.info(f"Saved image: {path}")
        
        # Save metadata if provided
        if metadata:
            self.save_metadata(path, metadata)
            
        return path
        
    def save_metadata(self, image_path: Path, metadata: Dict[str, Any]) -> Path:
        """Save metadata for an image
        
        Args:
            image_path: Path to image
            metadata: Metadata dictionary
            
        Returns:
            Path to metadata file
        """
        import json
        
        metadata_path = image_path.with_suffix('.json')
        
        # Add file info
        metadata['file_info'] = {
            'path': str(image_path),
            'size_bytes': image_path.stat().st_size if image_path.exists() else 0,
            'created': datetime.now().isoformat(),
            'checksum': self.calculate_checksum(image_path) if image_path.exists() else None
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
            
        self.logger.debug(f"Saved metadata: {metadata_path}")
        return metadata_path
        
    def load_metadata(self, image_path: Path) -> Optional[Dict[str, Any]]:
        """Load metadata for an image
        
        Args:
            image_path: Path to image
            
        Returns:
            Metadata dictionary or None
        """
        import json
        
        metadata_path = image_path.with_suffix('.json')
        
        if not metadata_path.exists():
            return None
            
        try:
            with open(metadata_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            raise FileManagerError(
                "Metadata file not found",
                str(metadata_path),
                "File does not exist",
                "Metadata is required for pipeline tracking"
            )
        except json.JSONDecodeError as e:
            raise FileManagerError(
                "Invalid metadata JSON",
                str(metadata_path),
                f"JSON decode error: {e}",
                "Metadata must be valid JSON for pipeline state tracking"
            )
        except Exception as e:
            raise FileManagerError(
                "Failed to load metadata",
                str(metadata_path),
                str(e),
                "Metadata is critical for pipeline tracking!"
            )
            
    def calculate_checksum(self, file_path: Path) -> str:
        """Calculate SHA256 checksum of file
        
        Args:
            file_path: Path to file
            
        Returns:
            Hex checksum string
        """
        sha256 = hashlib.sha256()
        
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b''):
                sha256.update(chunk)
                
        return sha256.hexdigest()
        
    def backup_file(self, source: Path, backup_name: Optional[str] = None) -> Path:
        """Create backup of file
        
        Args:
            source: File to backup
            backup_name: Custom backup name
            
        Returns:
            Path to backup
        """
        if not source.exists():
            raise FileNotFoundError(f"Source file not found: {source}")
            
        backup_dir = Path(self.config.paths.get('backup_dir', '/home/user/ai-wallpaper/backups'))
        backup_dir.mkdir(parents=True, exist_ok=True)
        
        if backup_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"{source.stem}_backup_{timestamp}{source.suffix}"
            
        backup_path = backup_dir / backup_name
        
        shutil.copy2(source, backup_path)
        self.logger.info(f"Created backup: {backup_path}")
        
        return backup_path
        
    def cleanup_old_files(
        self,
        directory: Path,
        days: int = 30,
        pattern: str = "*",
        dry_run: bool = True
    ) -> List[Path]:
        """Clean up old files
        
        Args:
            directory: Directory to clean
            days: Files older than this are removed
            pattern: File pattern to match
            dry_run: If True, only show what would be deleted
            
        Returns:
            List of files removed/to be removed
        """
        from datetime import timedelta
        
        cutoff_time = datetime.now() - timedelta(days=days)
        old_files = []
        
        for file_path in directory.glob(pattern):
            if not file_path.is_file():
                continue
                
            mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
            if mtime < cutoff_time:
                old_files.append(file_path)
                
        if dry_run:
            self.logger.info(f"Would delete {len(old_files)} files older than {days} days")
            for f in old_files[:5]:  # Show first 5
                self.logger.debug(f"  - {f.name}")
            if len(old_files) > 5:
                self.logger.debug(f"  ... and {len(old_files) - 5} more")
        else:
            for file_path in old_files:
                try:
                    file_path.unlink()
                    self.logger.debug(f"Deleted: {file_path}")
                except OSError as e:
                    raise FileManagerError(
                        "Failed to delete file",
                        str(file_path),
                        f"OS error: {e}",
                        "File cleanup must succeed - disk space matters!"
                    )
                except Exception as e:
                    raise FileManagerError(
                        "Unexpected error deleting file",
                        str(file_path),
                        str(e),
                        "File cleanup is critical for disk space management"
                    )
                    
        return old_files
        
    def get_disk_usage(self) -> Dict[str, Any]:
        """Get disk usage statistics
        
        Returns:
            Disk usage information
        """
        import shutil
        
        images_dir = self.get_images_dir()
        
        # Get directory size
        total_size = 0
        file_count = 0
        
        for file_path in images_dir.rglob("*"):
            if file_path.is_file():
                total_size += file_path.stat().st_size
                file_count += 1
                
        # Get disk space
        stat = shutil.disk_usage(images_dir)
        
        return {
            'images_directory': str(images_dir),
            'total_size_gb': total_size / (1024**3),
            'file_count': file_count,
            'disk_total_gb': stat.total / (1024**3),
            'disk_used_gb': stat.used / (1024**3),
            'disk_free_gb': stat.free / (1024**3),
            'disk_percent': (stat.used / stat.total) * 100
        }
        
    def verify_file_integrity(self, file_path: Path) -> bool:
        """Verify file integrity using metadata
        
        Args:
            file_path: Path to file
            
        Returns:
            True if file is intact
        """
        metadata = self.load_metadata(file_path)
        
        if not metadata or 'file_info' not in metadata:
            raise FileManagerError(
                "No metadata for integrity check",
                str(file_path),
                "Metadata missing or incomplete",
                "Metadata is required for quality assurance!"
            )
            
        stored_checksum = metadata['file_info'].get('checksum')
        if not stored_checksum:
            raise FileManagerError(
                "No checksum in metadata",
                str(file_path),
                "Checksum field missing from metadata",
                "Checksum is required for integrity verification!"
            )
            
        current_checksum = self.calculate_checksum(file_path)
        
        if current_checksum != stored_checksum:
            self.logger.error(f"Integrity check failed for {file_path}")
            return False
            
        return True


# Global instance
_file_manager: Optional[FileManager] = None


def get_file_manager() -> FileManager:
    """Get global file manager instance
    
    Returns:
        FileManager instance
    """
    global _file_manager
    if _file_manager is None:
        _file_manager = FileManager()
    return _file_manager