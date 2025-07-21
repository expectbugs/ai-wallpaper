#!/usr/bin/env python3
"""
CPU Offload Refiner - Last Resort for Extreme Sizes
SLOW but ALWAYS works - QUALITY OVER ALL
"""

from typing import Tuple, Dict, Any, Optional
from PIL import Image
import torch
import gc
import time
from pathlib import Path
from datetime import datetime

from ..core import get_logger
from .tiled_refiner import TiledRefiner

class CPUOffloadRefiner(TiledRefiner):
    """
    Extreme memory-saving refiner using aggressive CPU offloading.
    Processes even 16K+ images on limited VRAM.
    """
    
    def __init__(self, pipeline=None):
        # Use small tiles for minimal VRAM
        super().__init__(pipeline)
        self.tile_size = 512  # Minimum viable tile size
        self.overlap = 128    # Smaller overlap too
        
    def refine_with_offload(self,
                           image_path: Path,
                           prompt: str,
                           strength: float = 0.3,
                           steps: int = 30,
                           seed: Optional[int] = None) -> Path:
        """
        Refine using aggressive CPU offloading.
        Will be SLOW but will ALWAYS work.
        
        Args:
            image_path: Path to input image
            prompt: Text prompt
            strength: Refinement strength
            steps: Number of steps
            seed: Random seed
            
        Returns:
            Path to refined image
        """
        self.logger.warning(
            "⚠️ Using CPU offload refinement - this will be SLOW! "
            "But QUALITY OVER ALL - we WILL refine this image!"
        )
        
        # Enable maximum memory saving
        if self.pipeline:
            # Move model to CPU between tiles
            self.pipeline.enable_model_cpu_offload()
            
            # Enable sequential offloading for extreme memory saving
            if hasattr(self.pipeline, 'enable_sequential_cpu_offload'):
                self.pipeline.enable_sequential_cpu_offload()
            
            # Reduce memory fragmentation
            if hasattr(self.pipeline, 'enable_attention_slicing'):
                self.pipeline.enable_attention_slicing(1)
            
            # Enable xformers if available for memory efficiency
            if hasattr(self.pipeline, 'enable_xformers_memory_efficient_attention'):
                try:
                    self.pipeline.enable_xformers_memory_efficient_attention()
                    self.logger.info("Enabled xformers memory efficient attention")
                except Exception as e:
                    self.logger.debug(f"Could not enable xformers: {e}")
        
        # Process with minimal memory footprint
        start_time = time.time()
        
        try:
            # Force garbage collection before starting
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            # Log memory state
            if torch.cuda.is_available():
                free_bytes, total_bytes = torch.cuda.mem_get_info()
                self.logger.info(
                    f"VRAM before CPU offload: {free_bytes/(1024**3):.1f}GB free / "
                    f"{total_bytes/(1024**3):.1f}GB total"
                )
            
            # Use parent's tiled refinement with small tiles
            result = super().refine_tiled(
                image_path=image_path,
                prompt=prompt,
                base_strength=strength,
                base_steps=steps,
                seed=seed
            )
            
            duration = time.time() - start_time
            self.logger.info(
                f"✅ CPU offload refinement complete in {duration:.1f} seconds. "
                f"QUALITY ACHIEVED despite memory constraints!"
            )
            
            return result
            
        except torch.cuda.OutOfMemoryError as e:
            # Even with CPU offload, we ran out of memory
            # Try even more aggressive settings
            self.logger.error(f"CUDA OOM even with CPU offload: {e}")
            self.logger.info("Attempting ultra-aggressive memory saving...")
            
            # Reduce tile size further
            original_tile_size = self.tile_size
            self.tile_size = 384  # Even smaller tiles
            self.overlap = 64     # Minimal overlap
            
            try:
                # Clear everything
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                
                # Try again with ultra-small tiles
                result = super().refine_tiled(
                    image_path=image_path,
                    prompt=prompt,
                    base_strength=strength,
                    base_steps=steps,
                    seed=seed
                )
                
                duration = time.time() - start_time
                self.logger.info(
                    f"✅ Ultra-aggressive CPU offload complete in {duration:.1f} seconds"
                )
                
                return result
                
            finally:
                self.tile_size = original_tile_size
                
        except Exception as e:
            # Even this failed? Give detailed error
            self.logger.error(f"CPU offload refinement failed: {e}")
            self.logger.error("This should NEVER happen - please check system resources")
            
            # Log detailed system state
            import psutil
            memory = psutil.virtual_memory()
            self.logger.error(
                f"System RAM: {memory.used/(1024**3):.1f}GB used / "
                f"{memory.total/(1024**3):.1f}GB total ({memory.percent}% used)"
            )
            
            if torch.cuda.is_available():
                free_bytes, total_bytes = torch.cuda.mem_get_info()
                self.logger.error(
                    f"VRAM: {(total_bytes-free_bytes)/(1024**3):.1f}GB used / "
                    f"{total_bytes/(1024**3):.1f}GB total"
                )
            
            raise RuntimeError(
                f"Even CPU offload failed - system may be out of RAM. "
                f"Error: {str(e)}"
            ) from e
        finally:
            # Cleanup
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    def estimate_time(self, width: int, height: int) -> float:
        """
        Estimate processing time for CPU offload
        
        Returns:
            Estimated time in seconds (very conservative)
        """
        tiles = self.calculate_tiles(width, height, self.tile_size, self.overlap)
        
        # CPU offload is much slower - estimate ~30 seconds per tile
        return len(tiles) * 30.0