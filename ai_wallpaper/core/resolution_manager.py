#!/usr/bin/env python3
"""
Resolution Management System
Handles all resolution calculations and strategies
"""

from typing import Tuple, Dict, List, Optional
from dataclasses import dataclass
from pathlib import Path
import math

@dataclass
class ResolutionConfig:
    """Configuration for a specific resolution"""
    width: int
    height: int
    aspect_ratio: float
    total_pixels: int
    name: Optional[str] = None
    
    @classmethod
    def from_tuple(cls, resolution: Tuple[int, int], name: Optional[str] = None):
        width, height = resolution
        return cls(
            width=width,
            height=height,
            aspect_ratio=width / height,
            total_pixels=width * height,
            name=name
        )

class ResolutionManager:
    """Manages resolution calculations and strategies"""
    
    # Common resolution presets
    PRESETS = {
        "1080p": (1920, 1080),
        "1440p": (2560, 1440),
        "4K": (3840, 2160),
        "5K": (5120, 2880),
        "8K": (7680, 4320),
        "ultrawide_1440p": (3440, 1440),
        "ultrawide_4K": (5120, 2160),
        "super_ultrawide": (5760, 1080),  # 32:6 for triple monitors
        "portrait_4K": (2160, 3840),
        "square_4K": (2880, 2880),
    }
    
    # Model-specific optimal dimensions (all divisible by 64 for best quality)
    SDXL_OPTIMAL_DIMENSIONS = [
        (1024, 1024),  # 1:1
        (1152, 896),   # 4:3.11  
        (1216, 832),   # 3:2.05
        (1344, 768),   # 16:9.14
        (1536, 640),   # 2.4:1
        (768, 1344),   # 9:16 (portrait)
        (896, 1152),   # 3:4 (portrait)
        (640, 1536),   # 1:2.4 (tall portrait)
    ]
    
    FLUX_CONSTRAINTS = {
        "divisible_by": 16,
        "max_dimension": 2048,
        "optimal_pixels": 1024 * 1024,  # 1MP for best quality
    }
    
    def __init__(self):
        self.logger = None  # Will be set by caller
        
    def get_optimal_generation_size(self, 
                                   target_resolution: Tuple[int, int],
                                   model_type: str) -> Tuple[int, int]:
        """
        Calculate the optimal generation size for a given target resolution.
        
        Args:
            target_resolution: Target (width, height)
            model_type: One of 'sdxl', 'flux', 'dalle3', etc.
            
        Returns:
            Optimal generation dimensions for the model
        """
        target_config = ResolutionConfig.from_tuple(target_resolution)
        
        if model_type == "sdxl":
            return self._get_sdxl_optimal_size(target_config)
        elif model_type == "flux":
            return self._get_flux_optimal_size(target_config)
        elif model_type in ["dalle3", "gpt_image_1"]:
            return self._get_dalle_optimal_size(target_config)
        else:
            # Default: use SDXL logic
            return self._get_sdxl_optimal_size(target_config)
    
    def _get_sdxl_optimal_size(self, target: ResolutionConfig) -> Tuple[int, int]:
        """Get optimal SDXL generation size"""
        # Find closest aspect ratio match
        best_match = None
        best_diff = float('inf')
        
        for dims in self.SDXL_OPTIMAL_DIMENSIONS:
            width, height = dims
            aspect = width / height
            diff = abs(aspect - target.aspect_ratio)
            
            if diff < best_diff:
                best_diff = diff
                best_match = dims
        
        # Scale up if target is significantly larger
        base_w, base_h = best_match
        base_pixels = base_w * base_h
        
        if target.total_pixels > base_pixels * 4:
            # Generate at 1.5x size for better quality when upscaling a lot
            return (int(base_w * 1.5), int(base_h * 1.5))
        
        return best_match
    
    def _get_flux_optimal_size(self, target: ResolutionConfig) -> Tuple[int, int]:
        """Get optimal FLUX generation size"""
        # FLUX works best around 1MP
        scale = math.sqrt(self.FLUX_CONSTRAINTS["optimal_pixels"] / target.total_pixels)
        
        # Calculate dimensions
        width = int(target.width * scale)
        height = int(target.height * scale)
        
        # Ensure divisible by 16
        width = (width // 16) * 16
        height = (height // 16) * 16
        
        # Ensure within max dimension
        max_dim = self.FLUX_CONSTRAINTS["max_dimension"]
        if width > max_dim or height > max_dim:
            scale = max_dim / max(width, height)
            width = int(width * scale)
            height = int(height * scale)
            width = (width // 16) * 16
            height = (height // 16) * 16
        
        return (width, height)
    
    def calculate_upscale_strategy(self,
                                  source_size: Tuple[int, int],
                                  target_size: Tuple[int, int]) -> List[Dict]:
        """
        Calculate optimal upscaling strategy.
        
        Returns:
            List of upscaling steps to perform
        """
        source_w, source_h = source_size
        target_w, target_h = target_size
        
        scale_x = target_w / source_w
        scale_y = target_h / source_h
        
        strategy = []
        
        # Strategy 1: Integer upscaling with Real-ESRGAN
        current_w, current_h = source_w, source_h
        
        # Use 2x upscaling as much as possible
        while current_w * 2 <= target_w * 1.1 and current_h * 2 <= target_h * 1.1:
            strategy.append({
                "method": "realesrgan",
                "scale": 2,
                "model": "RealESRGAN_x2plus",
                "input_size": (current_w, current_h),
                "output_size": (current_w * 2, current_h * 2)
            })
            current_w *= 2
            current_h *= 2
        
        # Final adjustment if needed
        if current_w != target_w or current_h != target_h:
            if current_w >= target_w and current_h >= target_h:
                # We overshot, crop to exact size
                strategy.append({
                    "method": "center_crop",
                    "input_size": (current_w, current_h),
                    "output_size": (target_w, target_h)
                })
            else:
                # Need one more upscale + crop
                strategy.append({
                    "method": "realesrgan",
                    "scale": 2,
                    "model": "RealESRGAN_x2plus",
                    "input_size": (current_w, current_h),
                    "output_size": (current_w * 2, current_h * 2)
                })
                strategy.append({
                    "method": "center_crop",
                    "input_size": (current_w * 2, current_h * 2),
                    "output_size": (target_w, target_h)
                })
        
        return strategy