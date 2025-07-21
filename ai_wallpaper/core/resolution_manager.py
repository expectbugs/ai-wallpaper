#!/usr/bin/env python3
"""
Resolution Management System
Handles all resolution calculations and strategies
"""

from typing import Tuple, Dict, List, Optional
from dataclasses import dataclass
from pathlib import Path
import math
from .logger import get_logger

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
        self.logger = get_logger(self.__class__.__name__)
        
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
        """Get optimal SDXL generation size - ALWAYS use trained dimensions"""
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
        
        # NEVER scale up from trained dimensions
        # Quality comes from proper generation at trained resolutions, not larger generation
        if self.logger:
            self.logger.info(
                f"Target: {target.width}x{target.height} (aspect {target.aspect_ratio:.2f}) -> "
                f"Using SDXL trained size: {best_match[0]}x{best_match[1]} (aspect {best_match[0]/best_match[1]:.2f})"
            )
        
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
                                  target_size: Tuple[int, int],
                                  generation_aspect: float,
                                  target_aspect: float) -> List[Dict]:
        """
        Calculate upscaling strategy WITHOUT aspect adjustment.
        Aspect adjustment now happens in Stage 1.5 BEFORE refinement.
        
        Args:
            source_size: Current size after aspect adjustment
            target_size: Final target size
            generation_aspect: Original generation aspect (for logging)
            target_aspect: Target aspect (for validation)
            
        Returns:
            List of upscaling steps (Real-ESRGAN and downsample only)
        """
        source_w, source_h = source_size
        target_w, target_h = target_size
        
        strategy = []
        
        # Validate aspects match (should already be adjusted)
        current_aspect = source_w / source_h
        if abs(current_aspect - target_aspect) > 0.05:
            if self.logger:
                self.logger.warning(
                    f"Aspect mismatch in upscale strategy! "
                    f"Current: {current_aspect:.3f}, Target: {target_aspect:.3f}. "
                    f"Aspect adjustment should have been done in Stage 1.5!"
                )
        
        # Calculate scale needed
        scale_w = target_w / source_w
        scale_h = target_h / source_h
        scale_needed = max(scale_w, scale_h)
        
        if self.logger:
            self.logger.info(f"Upscale strategy: {source_w}x{source_h} → {target_w}x{target_h} (scale: {scale_needed:.2f}x)")
        
        # Only proceed if significant upscaling is needed
        if scale_needed > 1.1:
            current_w, current_h = source_w, source_h
            
            # Progressive 2x upscaling with Real-ESRGAN
            while current_w < target_w or current_h < target_h:
                strategy.append({
                    "method": "realesrgan",
                    "scale": 2,
                    "model": "RealESRGAN_x2plus",
                    "input_size": (current_w, current_h),
                    "output_size": (current_w * 2, current_h * 2),
                    "description": f"2x upscale to {current_w * 2}x{current_h * 2}"
                })
                current_w *= 2
                current_h *= 2
                
                # Safety check
                if len(strategy) > 5:
                    raise RuntimeError(
                        f"Too many upscale steps needed! "
                        f"Source: {source_size}, Target: {target_size}"
                    )
            
            # Final downsample if we overshot
            if current_w > target_w or current_h > target_h:
                strategy.append({
                    "method": "lanczos_downsample",
                    "input_size": (current_w, current_h),
                    "output_size": (target_w, target_h),
                    "description": f"High-quality downsample to exact {target_w}x{target_h}"
                })
        else:
            if self.logger:
                self.logger.info("No significant upscaling needed")
        
        return strategy
    
    def _get_dalle_optimal_size(self, target: ResolutionConfig) -> Tuple[int, int]:
        """Get optimal DALLE generation size - fixed at 1024x1024"""
        return (1024, 1024)
    
    def calculate_progressive_outpaint_strategy(self,
                                              current_size: Tuple[int, int],
                                              target_aspect: float,
                                              max_expansion_per_step: float = 2.0) -> List[Dict]:
        """
        Calculate progressive outpainting steps for extreme aspect ratios.
        COMPLETE IMPLEMENTATION with validation and error handling.
        
        Args:
            current_size: Current image dimensions (width, height)
            target_aspect: Target aspect ratio (width/height)
            max_expansion_per_step: Maximum expansion ratio per step
            
        Returns:
            List of progressive outpaint steps
            
        Raises:
            ValueError: If inputs are invalid
            RuntimeError: If strategy cannot be calculated
        """
        # Input validation
        if not current_size or len(current_size) != 2:
            raise ValueError(f"Invalid current_size: {current_size}")
        
        current_w, current_h = current_size
        
        if current_w <= 0 or current_h <= 0:
            raise ValueError(f"Invalid dimensions: {current_w}x{current_h}")
        
        if target_aspect <= 0:
            raise ValueError(f"Invalid target_aspect: {target_aspect}")
        
        current_aspect = current_w / current_h
        
        # If aspect change is minimal, return empty strategy
        if abs(current_aspect - target_aspect) < 0.05:
            if self.logger:
                self.logger.info(f"Aspect change minimal ({current_aspect:.3f} → {target_aspect:.3f}), no adjustment needed")
            return []
        
        # Check if expansion is too extreme
        aspect_change_ratio = max(target_aspect / current_aspect, current_aspect / target_aspect)
        if aspect_change_ratio > 8.0:
            raise ValueError(
                f"Aspect ratio change {aspect_change_ratio:.1f}x exceeds maximum supported ratio of 8.0x. "
                f"Current: {current_aspect:.3f}, Target: {target_aspect:.3f}"
            )
        
        steps = []
        
        # Determine expansion direction
        if target_aspect > current_aspect:
            # Expanding width
            target_w = int(current_h * target_aspect)
            target_h = current_h
            direction = "horizontal"
            
            if self.logger:
                self.logger.info(f"Planning horizontal expansion: {current_w}x{current_h} → {target_w}x{target_h}")
            
            # Calculate total expansion needed
            total_expansion = target_w / current_w
            
            # Progressive expansion logic
            temp_w = current_w
            temp_h = current_h
            
            # First step: Can be larger (2x) when we have maximum context
            if total_expansion >= 2.0:
                next_w = min(int(temp_w * 2.0), target_w)
                steps.append({
                    "method": "outpaint",
                    "current_size": (temp_w, temp_h),
                    "target_size": (next_w, temp_h),
                    "expansion_ratio": next_w / temp_w,
                    "direction": direction,
                    "step_type": "initial",
                    "description": f"Initial 2x expansion: {temp_w}x{temp_h} → {next_w}x{temp_h}"
                })
                temp_w = next_w
            
            # Middle steps: 1.5x for balanced expansion
            step_num = 2
            while temp_w < target_w * 0.95:  # 95% to avoid tiny final steps
                if temp_w * 1.5 <= target_w:
                    next_w = int(temp_w * 1.5)
                else:
                    next_w = target_w
                    
                steps.append({
                    "method": "outpaint",
                    "current_size": (temp_w, temp_h),
                    "target_size": (next_w, temp_h),
                    "expansion_ratio": next_w / temp_w,
                    "direction": direction,
                    "step_type": "progressive",
                    "description": f"Step {step_num}: {temp_w}x{temp_h} → {next_w}x{temp_h}"
                })
                temp_w = next_w
                step_num += 1
                
                # Safety check
                if step_num > 10:
                    raise RuntimeError(f"Too many expansion steps ({step_num}), something is wrong")
            
            # Final adjustment if needed
            if temp_w < target_w:
                steps.append({
                    "method": "outpaint",
                    "current_size": (temp_w, temp_h),
                    "target_size": (target_w, temp_h),
                    "expansion_ratio": target_w / temp_w,
                    "direction": direction,
                    "step_type": "final",
                    "description": f"Final adjustment: {temp_w}x{temp_h} → {target_w}x{temp_h}"
                })
        
        else:
            # Expanding height (similar logic)
            target_w = current_w
            target_h = int(current_w / target_aspect)
            direction = "vertical"
            
            if self.logger:
                self.logger.info(f"Planning vertical expansion: {current_w}x{current_h} → {target_w}x{target_h}")
            
            # Calculate total expansion needed
            total_expansion = target_h / current_h
            
            # Progressive expansion logic for height
            temp_w = current_w
            temp_h = current_h
            
            # First step: Can be larger (2x) when we have maximum context
            if total_expansion >= 2.0:
                next_h = min(int(temp_h * 2.0), target_h)
                steps.append({
                    "method": "outpaint",
                    "current_size": (temp_w, temp_h),
                    "target_size": (temp_w, next_h),
                    "expansion_ratio": next_h / temp_h,
                    "direction": direction,
                    "step_type": "initial",
                    "description": f"Initial 2x expansion: {temp_w}x{temp_h} → {temp_w}x{next_h}"
                })
                temp_h = next_h
            
            # Middle steps: 1.5x for balanced expansion
            step_num = 2
            while temp_h < target_h * 0.95:  # 95% to avoid tiny final steps
                if temp_h * 1.5 <= target_h:
                    next_h = int(temp_h * 1.5)
                else:
                    next_h = target_h
                    
                steps.append({
                    "method": "outpaint",
                    "current_size": (temp_w, temp_h),
                    "target_size": (temp_w, next_h),
                    "expansion_ratio": next_h / temp_h,
                    "direction": direction,
                    "step_type": "progressive",
                    "description": f"Step {step_num}: {temp_w}x{temp_h} → {temp_w}x{next_h}"
                })
                temp_h = next_h
                step_num += 1
                
                # Safety check
                if step_num > 10:
                    raise RuntimeError(f"Too many expansion steps ({step_num}), something is wrong")
            
            # Final adjustment if needed
            if temp_h < target_h:
                steps.append({
                    "method": "outpaint",
                    "current_size": (temp_w, temp_h),
                    "target_size": (temp_w, target_h),
                    "expansion_ratio": target_h / temp_h,
                    "direction": direction,
                    "step_type": "final",
                    "description": f"Final adjustment: {temp_w}x{temp_h} → {temp_w}x{target_h}"
                })
            
        if self.logger:
            self.logger.info(f"Progressive strategy: {len(steps)} steps planned")
            for i, step in enumerate(steps):
                self.logger.debug(f"  {i+1}. {step['description']}")
        
        return steps
    
    def _round_to_multiple_of_8(self, value: int) -> int:
        """Round value to nearest multiple of 8 for SDXL compatibility"""
        return ((value + 4) // 8) * 8
    
    def calculate_sliding_window_strategy(self,
                                        current_size: Tuple[int, int],
                                        target_size: Tuple[int, int],
                                        window_size: int = 200,
                                        overlap_ratio: float = 0.8) -> List[Dict]:
        """
        Calculate sliding window outpainting strategy for maximum context preservation.
        
        Args:
            current_size: Current image dimensions (width, height)
            target_size: Target dimensions (width, height)
            window_size: Size of each expansion window in pixels
            overlap_ratio: Overlap between consecutive windows (0.0-1.0)
            
        Returns:
            List of sliding window steps
            
        Raises:
            ValueError: If inputs are invalid
        """
        # Input validation
        if not current_size or len(current_size) != 2:
            raise ValueError(f"Invalid current_size: {current_size}")
        
        if not target_size or len(target_size) != 2:
            raise ValueError(f"Invalid target_size: {target_size}")
        
        current_w, current_h = current_size
        target_w, target_h = target_size
        
        if current_w <= 0 or current_h <= 0:
            raise ValueError(f"Invalid current dimensions: {current_w}x{current_h}")
        
        if target_w <= 0 or target_h <= 0:
            raise ValueError(f"Invalid target dimensions: {target_w}x{target_h}")
        
        if window_size <= 0:
            raise ValueError(f"Invalid window_size: {window_size}")
        
        if not 0.0 <= overlap_ratio < 1.0:
            raise ValueError(f"Invalid overlap_ratio: {overlap_ratio}")
        
        steps = []
        
        # Calculate step size (window minus overlap)
        step_size = int(window_size * (1.0 - overlap_ratio))
        
        # Determine if we need horizontal, vertical, or both expansions
        need_horizontal = target_w > current_w
        need_vertical = target_h > current_h
        
        if need_horizontal:
            # Calculate horizontal sliding windows
            temp_w = current_w
            temp_h = target_h if need_vertical else current_h
            window_num = 1
            
            while temp_w < target_w:
                # Calculate next window position
                next_w = min(temp_w + window_size, target_w)
                
                # Ensure we reach exactly target_w on last step
                if target_w - next_w < step_size:
                    next_w = target_w
                else:
                    # Round to multiple of 8 for SDXL compatibility
                    next_w = self._round_to_multiple_of_8(next_w)
                
                steps.append({
                    "method": "sliding_window",
                    "current_size": (temp_w, temp_h),
                    "target_size": (next_w, temp_h),
                    "window_size": next_w - temp_w,
                    "overlap_size": window_size - step_size if window_num > 1 else 0,
                    "direction": "horizontal",
                    "window_number": window_num,
                    "description": f"H-Window {window_num}: {temp_w}x{temp_h} → {next_w}x{temp_h} (+{next_w-temp_w}px)"
                })
                
                # Calculate actual step taken (accounts for rounding)
                # For first window, step is the full window size
                # For subsequent windows, account for overlap
                if window_num == 1:
                    actual_step = next_w - temp_w
                else:
                    # Step forward is window size minus overlap
                    actual_step = step_size
                temp_w = temp_w + actual_step
                window_num += 1
        
        if need_vertical:
            # Calculate vertical sliding windows (after horizontal if both needed)
            temp_w = target_w if need_horizontal else current_w
            temp_h = current_h
            window_num = 1
            
            while temp_h < target_h:
                next_h = min(temp_h + window_size, target_h)
                
                if target_h - next_h < step_size:
                    next_h = target_h
                else:
                    # Round to multiple of 8 for SDXL compatibility
                    next_h = self._round_to_multiple_of_8(next_h)
                
                steps.append({
                    "method": "sliding_window",
                    "current_size": (temp_w, temp_h),
                    "target_size": (temp_w, next_h),
                    "window_size": next_h - temp_h,
                    "overlap_size": window_size - step_size if window_num > 1 else 0,
                    "direction": "vertical",
                    "window_number": window_num,
                    "description": f"V-Window {window_num}: {temp_w}x{temp_h} → {temp_w}x{next_h} (+{next_h-temp_h}px)"
                })
                
                # Calculate actual step taken (accounts for rounding)
                # For first window, step is the full window size
                # For subsequent windows, account for overlap
                if window_num == 1:
                    actual_step = next_h - temp_h
                else:
                    # Step forward is window size minus overlap
                    actual_step = step_size
                temp_h = temp_h + actual_step
                window_num += 1
        
        self.logger.info(
            f"Sliding window strategy: {len(steps)} windows "
            f"({window_size}px window, {step_size}px step, {overlap_ratio:.0%} overlap)"
        )
        
        return steps
    
    def should_use_progressive_outpainting(self, aspect_change_ratio: float) -> bool:
        """
        Determine if progressive outpainting should be used.
        
        Args:
            aspect_change_ratio: Max ratio between source and target aspects
            
        Returns:
            True if progressive outpainting should be used
        """
        # For now, hardcode the threshold as we don't have config yet
        # This will be updated when we implement the new resolution.yaml
        single_step_max = 2.5
        
        return aspect_change_ratio > single_step_max