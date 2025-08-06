#!/usr/bin/env python3
"""
VRAM Calculator for Dynamic Resource Management
NO LIMITS - Just intelligent adaptation
"""

import torch
from typing import Tuple, Dict, Any, Optional
from .logger import get_logger

class VRAMCalculator:
    """Calculate VRAM requirements and determine refinement strategy"""
    
    def __init__(self):
        self.logger = get_logger()
        
        # SDXL base requirements (measured empirically)
        self.MODEL_OVERHEAD_MB = 6144  # 6GB for SDXL refiner
        self.ATTENTION_MULTIPLIER = 4   # Attention needs ~4x image memory
        self.SAFETY_BUFFER = 0.2        # 20% safety margin
        
    def calculate_refinement_vram(self, 
                                 width: int, 
                                 height: int,
                                 dtype: torch.dtype = torch.float16) -> Dict[str, float]:
        """
        Calculate ACCURATE VRAM requirements for refinement.
        
        Args:
            width: Image width
            height: Image height  
            dtype: Data type (float16 or float32)
            
        Returns:
            Dict with detailed VRAM breakdown
        """
        pixels = width * height
        
        # Bytes per pixel based on dtype
        bytes_per_pixel = 2 if dtype == torch.float16 else 4
        
        # Image tensor memory (BCHW format)
        # 1 batch × 4 channels (latent) × H × W
        latent_h = height // 8  # VAE downscales by 8
        latent_w = width // 8
        latent_pixels = latent_h * latent_w
        
        # Memory calculations (in MB)
        latent_memory_mb = (latent_pixels * 4 * bytes_per_pixel) / (1024 * 1024)
        
        # Attention memory scales with sequence length
        # For SDXL: roughly latent_pixels * multiplier
        attention_memory_mb = (latent_pixels * self.ATTENTION_MULTIPLIER * bytes_per_pixel) / (1024 * 1024)
        
        # Activations and gradients
        activation_memory_mb = latent_memory_mb * 2  # Conservative estimate
        
        # Total image-related memory
        image_memory_mb = latent_memory_mb + attention_memory_mb + activation_memory_mb
        
        # Add model overhead
        total_vram_mb = self.MODEL_OVERHEAD_MB + image_memory_mb
        
        # Add safety buffer
        total_with_buffer_mb = total_vram_mb * (1 + self.SAFETY_BUFFER)
        
        return {
            'latent_memory_mb': latent_memory_mb,
            'attention_memory_mb': attention_memory_mb,
            'activation_memory_mb': activation_memory_mb,
            'model_overhead_mb': self.MODEL_OVERHEAD_MB,
            'total_vram_mb': total_vram_mb,
            'total_with_buffer_mb': total_with_buffer_mb,
            'pixels': pixels,
            'resolution': f"{width}x{height}"
        }
    
    def get_available_vram(self) -> Optional[float]:
        """Get available VRAM in MB"""
        if not torch.cuda.is_available():
            return None
            
        # FAIL LOUD philosophy - if CUDA is available but we can't get info, that's an error
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        free_mb = free_bytes / (1024 * 1024)
        total_mb = total_bytes / (1024 * 1024)
        
        self.logger.debug(f"VRAM: {free_mb:.0f}MB free / {total_mb:.0f}MB total")
        return free_mb
    
    def determine_refinement_strategy(self,
                                     width: int,
                                     height: int) -> Dict[str, Any]:
        """
        Determine the best refinement strategy.
        NEVER RETURNS 'impossible' - always finds a way!
        
        Returns:
            Strategy dict with:
            - strategy: 'full', 'tiled', or 'cpu_offload'
            - details: Strategy-specific parameters
            - warnings: Any warnings to log
        """
        # Calculate requirements
        vram_info = self.calculate_refinement_vram(width, height)
        required_mb = vram_info['total_with_buffer_mb']
        
        # Get available VRAM
        available_mb = self.get_available_vram()
        
        if available_mb is None:
            # No CUDA - use CPU offload
            return {
                'strategy': 'cpu_offload',
                'details': {
                    'reason': 'No CUDA available',
                    'warning': 'Using CPU - will be VERY slow'
                },
                'vram_required_mb': required_mb,
                'vram_available_mb': 0
            }
        
        # Strategy decision
        if required_mb <= available_mb:
            # Full refinement possible!
            return {
                'strategy': 'full',
                'details': {
                    'message': f"Full refinement possible: {required_mb:.0f}MB < {available_mb:.0f}MB"
                },
                'vram_required_mb': required_mb,
                'vram_available_mb': available_mb
            }
        
        # Need tiled refinement
        # Calculate optimal tile size
        # Simplified calculation with safety check
        if vram_info['total_vram_mb'] > 0:
            pixels_per_mb = vram_info['pixels'] / vram_info['total_vram_mb']
        else:
            # Fallback to a reasonable default if somehow total_vram_mb is 0
            pixels_per_mb = 1024 * 1024 / 100  # Assume 100MB per megapixel as fallback
        
        # Ensure we have enough VRAM for at least minimal tiling
        available_for_tiling = available_mb - self.MODEL_OVERHEAD_MB
        if available_for_tiling <= 0:
            # Not enough VRAM even for model overhead - fall back to CPU offload
            return {
                'strategy': 'cpu_offload',
                'details': {
                    'reason': f'Insufficient VRAM: {available_mb:.0f}MB < {self.MODEL_OVERHEAD_MB:.0f}MB overhead',
                    'warning': 'Using CPU offload - will be slow but will work!'
                },
                'vram_required_mb': required_mb,
                'vram_available_mb': available_mb
            }
        
        max_pixels = int(available_for_tiling * pixels_per_mb * 0.8)
        
        # Tile size (ensure divisible by 128 for SDXL)
        tile_size = int(max(0, max_pixels) ** 0.5)  # Ensure non-negative
        tile_size = max(512, ((tile_size // 128) * 128))
        
        # Can we do tiled refinement?
        tile_vram = self.calculate_refinement_vram(tile_size, tile_size)
        if tile_vram['total_with_buffer_mb'] <= available_mb:
            return {
                'strategy': 'tiled',
                'details': {
                    'tile_size': tile_size,
                    'overlap': min(256, tile_size // 4),
                    'message': f"Tiled refinement: {tile_size}x{tile_size} tiles"
                },
                'vram_required_mb': required_mb,
                'vram_available_mb': available_mb
            }
        
        # Last resort - CPU offload
        return {
            'strategy': 'cpu_offload',
            'details': {
                'reason': 'Image too large even for tiled refinement',
                'tile_size': 512,  # Minimum tile size
                'warning': 'Using aggressive CPU offloading - will be SLOW'
            },
            'vram_required_mb': required_mb,
            'vram_available_mb': available_mb
        }