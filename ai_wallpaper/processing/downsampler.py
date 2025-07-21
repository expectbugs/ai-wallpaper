#!/usr/bin/env python3
"""
High-Quality Downsampling System
Reduces resolution while preserving maximum quality
"""

from PIL import Image, ImageFilter, ImageEnhance
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
from datetime import datetime

from ..core import get_logger, get_config
from ..core.path_resolver import get_resolver

class HighQualityDownsampler:
    """Downsample images with maximum quality preservation"""
    
    def __init__(self):
        """Initialize the downsampler with configuration"""
        self.logger = get_logger()
        self.config = get_config()
        self.resolver = get_resolver()
        
        # Load downsampling configuration
        downsample_config = self.config.resolution.get('downsampling', {})
        self.method = downsample_config.get('method', 'lanczos')
        self.sharpen_after = downsample_config.get('sharpen_after', True)
        self.sharpen_radius = downsample_config.get('sharpen_radius', 0.5)
        self.sharpen_percent = downsample_config.get('sharpen_percent', 50)
        self.sharpen_threshold = downsample_config.get('sharpen_threshold', 10)
        
    def downsample(self,
                   image_path: Path,
                   target_size: Tuple[int, int],
                   method: Optional[str] = None,
                   sharpen: Optional[bool] = None,
                   save_path: Optional[Path] = None) -> Path:
        """
        Downsample image to exact size with maximum quality.
        
        Args:
            image_path: Source image path
            target_size: Target (width, height)
            method: Override resampling method (optional)
            sharpen: Override sharpening setting (optional)
            save_path: Optional output path (will auto-generate if not provided)
            
        Returns:
            Path to downsampled image
        """
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
            
        # Use overrides or defaults
        method = method or self.method
        sharpen = sharpen if sharpen is not None else self.sharpen_after
        
        image = Image.open(image_path)
        current_size = image.size
        target_w, target_h = target_size
        
        if current_size == target_size:
            self.logger.info(f"Image already at target size {target_size}")
            return image_path
            
        if current_size[0] < target_w or current_size[1] < target_h:
            raise ValueError(
                f"Cannot downsample {current_size} to {target_size} - "
                f"target is larger in at least one dimension"
            )
            
        self.logger.info(f"Downsampling: {current_size} -> {target_size} using {method}")
        
        # Choose resampling filter
        resample_filter = self._get_resample_filter(method)
        
        # Perform high-quality resize
        resized = image.resize(target_size, resample=resample_filter)
        
        # Optional sharpening to restore detail
        if sharpen and self._should_sharpen(current_size, target_size):
            self.logger.info("Applying post-downsample sharpening")
            resized = self._apply_sharpening(resized)
        
        # Save with maximum quality
        if save_path is None:
            temp_dir = self.resolver.get_temp_dir() / 'ai-wallpaper'
            temp_dir.mkdir(parents=True, exist_ok=True)
            save_path = temp_dir / f"downsampled_{target_w}x{target_h}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.png"
        
        # Ensure parent directory exists
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save with ZERO quality loss using lossless save
        from ..utils import save_lossless_png
        save_lossless_png(resized, save_path)
        
        self.logger.info(f"Downsampled to exact size: {target_w}x{target_h} -> {save_path}")
        
        return save_path
    
    def _get_resample_filter(self, method: str):
        """Get PIL resample filter from method name"""
        filters = {
            "lanczos": Image.Resampling.LANCZOS,
            "cubic": Image.Resampling.BICUBIC,
            "area": Image.Resampling.BOX,  # Best for significant downsampling
            "linear": Image.Resampling.BILINEAR,
            "nearest": Image.Resampling.NEAREST
        }
        
        if method not in filters:
            raise ValueError(
                f"Unknown resample method: {method}!\n"
                f"Valid methods: {', '.join(filters.keys())}\n"
                f"Quality matters - use a valid resampling method!"
            )
            
        return filters[method]
    
    def _should_sharpen(self, current_size: Tuple[int, int], target_size: Tuple[int, int]) -> bool:
        """Determine if sharpening should be applied based on downsample ratio"""
        # Calculate downsample ratio
        ratio_w = current_size[0] / target_size[0]
        ratio_h = current_size[1] / target_size[1]
        max_ratio = max(ratio_w, ratio_h)
        
        # Only sharpen if downsampling by more than 1.2x
        return max_ratio > 1.2
    
    def _apply_sharpening(self, image: Image.Image) -> Image.Image:
        """Apply intelligent sharpening to restore detail after downsampling"""
        # First pass: Subtle unsharp mask
        sharpened = image.filter(ImageFilter.UnsharpMask(
            radius=self.sharpen_radius,
            percent=self.sharpen_percent,
            threshold=self.sharpen_threshold
        ))
        
        # Optional: Very slight contrast boost (1-5%)
        enhancer = ImageEnhance.Contrast(sharpened)
        sharpened = enhancer.enhance(1.02)  # 2% contrast boost
        
        # Optional: Slight saturation boost to compensate for any dulling
        enhancer = ImageEnhance.Color(sharpened)
        sharpened = enhancer.enhance(1.01)  # 1% saturation boost
        
        return sharpened
    
    def downsample_batch(self, 
                        image_paths: list[Path], 
                        target_size: Tuple[int, int],
                        output_dir: Optional[Path] = None) -> list[Path]:
        """
        Downsample multiple images with consistent settings.
        
        Args:
            image_paths: List of source image paths
            target_size: Target dimensions for all images
            output_dir: Optional output directory
            
        Returns:
            List of output paths
        """
        output_paths = []
        
        for i, image_path in enumerate(image_paths):
            self.logger.info(f"Processing image {i+1}/{len(image_paths)}: {image_path.name}")
            
            if output_dir:
                save_path = output_dir / f"{image_path.stem}_{target_size[0]}x{target_size[1]}{image_path.suffix}"
            else:
                save_path = None
                
            try:
                output_path = self.downsample(
                    image_path=image_path,
                    target_size=target_size,
                    save_path=save_path
                )
                output_paths.append(output_path)
            except Exception as e:
                self.logger.error(f"Failed to downsample {image_path}: {e}")
                raise
                
        return output_paths