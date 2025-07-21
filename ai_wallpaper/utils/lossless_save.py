#!/usr/bin/env python3
"""
Lossless PNG Save Utility - ZERO Quality Loss Guaranteed
Part of the QUALITY OVER ALL philosophy
"""

from pathlib import Path
from PIL import Image
from typing import Union
import os

from ..core import get_logger
from ..core.exceptions import AIWallpaperError

class QualityError(AIWallpaperError):
    """Raised when saved file quality is compromised"""
    pass

def save_lossless_png(image: Image.Image, 
                      filepath: Union[str, Path], 
                      validate: bool = True,
                      log_size: bool = True) -> Path:
    """
    Save image as PNG with ZERO quality loss.
    
    This function enforces absolute maximum quality with no compression.
    File sizes will be large but quality is preserved perfectly.
    
    Args:
        image: PIL Image object to save
        filepath: Path to save to (will be converted to .png if not already)
        validate: Check file size is reasonable (default: True)
        log_size: Log file size information (default: True)
        
    Returns:
        Path: The path where the file was saved
        
    Raises:
        QualityError: If saved file is suspiciously small
    """
    logger = get_logger()
    
    # Ensure Path object
    filepath = Path(filepath)
    
    # Force .png extension
    if filepath.suffix.lower() != '.png':
        filepath = filepath.with_suffix('.png')
    
    # Ensure RGB mode for consistency (RGBA if has alpha)
    if image.mode not in ('RGB', 'RGBA'):
        if 'transparency' in image.info or image.mode == 'RGBA':
            image = image.convert('RGBA')
        else:
            image = image.convert('RGB')
    
    # Create parent directory if needed
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    # Save with absolute maximum quality
    # NO compression, NO optimization, EXPLICIT format
    image.save(
        str(filepath),     # Convert to string for PIL
        'PNG',             # EXPLICIT format - never rely on extension
        compress_level=0,  # 0 = NO compression (0-9 scale)
        optimize=False,    # NO optimization that could reduce quality
        # NO quality parameter - PNG ignores it anyway
    )
    
    # Get file size for validation and logging
    file_size_bytes = filepath.stat().st_size
    file_size_mb = file_size_bytes / (1024 * 1024)
    
    # Calculate bytes per pixel
    pixels = image.width * image.height
    bytes_per_pixel = file_size_bytes / pixels
    
    if log_size:
        logger.info(
            f"Saved lossless PNG: {filepath.name} | "
            f"Size: {file_size_mb:.1f}MB | "
            f"Resolution: {image.width}x{image.height} | "
            f"Bytes/pixel: {bytes_per_pixel:.2f}"
        )
    
    if validate:
        # Calculate expected minimum size
        # RGB = 3 bytes/pixel, PNG overhead ~10%, so minimum ~0.3 bytes/pixel compressed
        # But with compress_level=0, should be much higher
        channels = 4 if image.mode == 'RGBA' else 3
        expected_min_mb = (pixels * channels) / (1024 * 1024) * 0.1  # 10% of raw
        
        if file_size_mb < expected_min_mb:
            raise QualityError(
                f"Saved file suspiciously small: {file_size_mb:.1f}MB "
                f"(expected minimum {expected_min_mb:.1f}MB for {image.width}x{image.height}). "
                f"Only {bytes_per_pixel:.2f} bytes/pixel - possible quality loss!"
            )
        
        # Warning for low bytes per pixel
        if bytes_per_pixel < 0.5:
            logger.warning(
                f"⚠️ LOW bytes/pixel ({bytes_per_pixel:.2f}) for {filepath.name} - "
                f"verify image quality!"
            )
    
    return filepath


def save_lossless_any_format(image: Image.Image,
                            filepath: Union[str, Path],
                            format: str = None,
                            quality: int = 100) -> Path:
    """
    Save image in any format with maximum quality.
    
    For PNG: Uses lossless settings
    For JPEG: Uses maximum quality (100)
    For others: Uses provided quality
    
    Args:
        image: PIL Image object
        filepath: Path to save to
        format: Image format (auto-detected if None)
        quality: Quality for lossy formats (ignored for PNG)
        
    Returns:
        Path: Where file was saved
    """
    filepath = Path(filepath)
    
    # Detect format from extension if not provided
    if format is None:
        format = filepath.suffix.upper().lstrip('.')
        if format == 'JPG':
            format = 'JPEG'
    
    if format == 'PNG':
        # Use lossless PNG save
        return save_lossless_png(image, filepath)
    else:
        # For other formats, use standard save with quality
        logger = get_logger()
        
        # Ensure parent directory exists
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Save with specified quality
        save_params = {'format': format}
        if format in ('JPEG', 'WEBP'):
            save_params['quality'] = quality
            save_params['optimize'] = False  # Faster, preserves quality
            
        image.save(str(filepath), **save_params)
        
        # Log file size
        file_size_mb = filepath.stat().st_size / (1024 * 1024)
        logger.info(
            f"Saved {format}: {filepath.name} | "
            f"Size: {file_size_mb:.1f}MB | "
            f"Quality: {quality if format in ('JPEG', 'WEBP') else 'N/A'}"
        )
        
        return filepath