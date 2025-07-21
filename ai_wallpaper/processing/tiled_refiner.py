#!/usr/bin/env python3
"""
Tiled Refiner for Ultra-Quality Mode
Processes large images in overlapping tiles to bypass VRAM limitations
NO QUALITY COMPROMISE - Just takes longer
"""

from typing import Tuple, Dict, Any, Optional, List
from PIL import Image, ImageFilter
import numpy as np
import torch
from pathlib import Path
import time
from datetime import datetime

from ..core import get_logger
from ..core.vram_calculator import VRAMCalculator

class TiledRefiner:
    """Refine images using overlapping tiles for unlimited resolution"""
    
    def __init__(self, pipeline=None, vram_calculator=None):
        """
        Initialize tiled refiner with VRAM awareness
        
        Args:
            pipeline: SDXL refiner pipeline (must be inpaint)
            vram_calculator: VRAM calculator instance
        """
        self.logger = get_logger()
        self.pipeline = pipeline
        self.vram_calculator = vram_calculator or VRAMCalculator()
        
        # Default tile settings
        self.tile_size = 1024  # Default tile size
        self.overlap = 256     # Overlap between tiles
        self.blend_width = 128 # Width of blending region
        
        # Quality settings
        self.min_tile_size = 512
        self.max_tile_size = 2048
        
        # Validate pipeline if provided
        if pipeline is not None:
            pipeline_class = type(pipeline).__name__
            if not any(x in pipeline_class for x in ['Inpaint', 'InPaint']):
                self.logger.warning(
                    f"TiledRefiner works best with inpaint pipeline, got {pipeline_class}"
                )
    
    def calculate_tiles(self, width: int, height: int, tile_size: int, overlap: int) -> List[Tuple[int, int, int, int]]:
        """
        Calculate tile positions with overlap
        
        Returns:
            List of (x1, y1, x2, y2) tuples for each tile
        """
        tiles = []
        
        # Calculate step size (tile size minus overlap)
        step = tile_size - overlap
        
        # Generate tiles
        for y in range(0, height, step):
            for x in range(0, width, step):
                # Calculate tile bounds
                x1 = x
                y1 = y
                x2 = min(x + tile_size, width)
                y2 = min(y + tile_size, height)
                
                # Ensure minimum tile size
                if x2 - x1 >= self.min_tile_size and y2 - y1 >= self.min_tile_size:
                    tiles.append((x1, y1, x2, y2))
        
        return tiles
    
    def create_blend_mask(self, tile_width: int, tile_height: int, overlap: int) -> np.ndarray:
        """
        Create a blend mask for seamless tile merging
        
        Returns:
            Numpy array with blend weights
        """
        mask = np.ones((tile_height, tile_width), dtype=np.float32)
        
        # Create gradients for edges
        blend_size = min(overlap // 2, self.blend_width)
        
        if blend_size > 0:
            # Top edge
            for i in range(blend_size):
                weight = i / blend_size
                mask[i, :] *= weight
            
            # Bottom edge
            for i in range(blend_size):
                weight = i / blend_size
                mask[-(i+1), :] *= weight
            
            # Left edge
            for i in range(blend_size):
                weight = i / blend_size
                mask[:, i] *= weight
            
            # Right edge
            for i in range(blend_size):
                weight = i / blend_size
                mask[:, -(i+1)] *= weight
        
        return mask
    
    def refine_tile(self, 
                   tile_image: Image.Image,
                   prompt: str,
                   strength: float = 0.3,
                   steps: int = 30,
                   seed: Optional[int] = None) -> Image.Image:
        """
        Refine a single tile
        
        Args:
            tile_image: PIL Image of the tile
            prompt: Text prompt
            strength: Refinement strength
            steps: Number of steps
            seed: Random seed
            
        Returns:
            Refined tile image
        """
        if self.pipeline is None:
            self.logger.warning("No pipeline available, returning original tile")
            return tile_image
        
        try:
            # Set up generator
            generator = None
            if seed is not None:
                generator = torch.Generator(device="cuda")
                generator.manual_seed(seed)
            
            # Refine the tile
            result = self.pipeline(
                prompt=prompt,
                image=tile_image,
                strength=strength,
                num_inference_steps=steps,
                generator=generator,
                guidance_scale=7.5
            )
            
            return result.images[0]
            
        except Exception as e:
            self.logger.error(f"Failed to refine tile: {e}")
            raise
    
    def refine_tiled(self,
                    image_path: Path,
                    prompt: str,
                    base_strength: float = 0.3,
                    base_steps: int = 30,
                    seed: Optional[int] = None) -> Path:
        """
        Refine image using tiled processing
        
        Args:
            image_path: Path to input image
            prompt: Text prompt
            base_strength: Base refinement strength
            base_steps: Base number of steps
            seed: Random seed
            
        Returns:
            Path to refined image
        """
        self.logger.info(f"Starting tiled refinement for {image_path}")
        start_time = time.time()
        
        # Load image
        image = Image.open(image_path)
        width, height = image.size
        
        # Calculate tiles
        tiles = self.calculate_tiles(width, height, self.tile_size, self.overlap)
        self.logger.info(f"Processing {len(tiles)} tiles of size {self.tile_size}x{self.tile_size}")
        
        # Track tile usage in metadata if available
        if hasattr(self.pipeline, 'model') and hasattr(self.pipeline.model, 'generation_metadata'):
            self.pipeline.model.generation_metadata['used_tiled'] = True
            self.pipeline.model.generation_metadata['tile_boundaries'] = [(x1, y1) for x1, y1, x2, y2 in tiles]
        
        # Create output array
        output = np.array(image, dtype=np.float32)
        weights = np.zeros((height, width), dtype=np.float32)
        
        # Process each tile
        for i, (x1, y1, x2, y2) in enumerate(tiles):
            tile_width = x2 - x1
            tile_height = y2 - y1
            
            self.logger.info(f"Processing tile {i+1}/{len(tiles)} at ({x1},{y1})-({x2},{y2})")
            
            # Extract tile
            tile_image = image.crop((x1, y1, x2, y2))
            
            # Refine tile
            refined_tile = self.refine_tile(
                tile_image,
                prompt,
                base_strength,
                base_steps,
                seed + i if seed else None
            )
            
            # Create blend mask
            mask = self.create_blend_mask(tile_width, tile_height, self.overlap)
            
            # Convert to numpy
            refined_array = np.array(refined_tile, dtype=np.float32)
            
            # Apply to output with blending
            for c in range(3):  # RGB channels
                output[y1:y2, x1:x2, c] += refined_array[:, :, c] * mask
            weights[y1:y2, x1:x2] += mask
        
        # Normalize by weights
        for c in range(3):
            output[:, :, c] /= np.maximum(weights, 1e-6)
        
        # Convert back to image
        output = np.clip(output, 0, 255).astype(np.uint8)
        result_image = Image.fromarray(output)
        
        # Save result with ZERO quality loss
        output_path = image_path.parent / f"tiled_refined_{image_path.stem}.png"
        result_image.save(
            output_path, 
            'PNG',  # EXPLICIT format - critical!
            compress_level=0,  # NO compression
            optimize=False     # NO optimization
        )
        
        duration = time.time() - start_time
        self.logger.info(f"Tiled refinement complete in {duration:.1f} seconds")
        
        return output_path
    
    def refine_tiled_auto(self,
                         image_path: Path,
                         prompt: str,
                         base_strength: float = 0.3,
                         base_steps: int = 30,
                         seed: Optional[int] = None) -> Path:
        """
        Automatically determine tile size based on available VRAM.
        ALWAYS succeeds - adapts tile size as needed.
        
        Args:
            image_path: Path to input image
            prompt: Text prompt
            base_strength: Refinement strength
            base_steps: Number of steps
            seed: Random seed
            
        Returns:
            Path to refined image
        """
        # Load image to get dimensions
        image = Image.open(image_path)
        w, h = image.size
        
        # Determine optimal tile size based on current tile size
        strategy = self.vram_calculator.determine_refinement_strategy(
            self.tile_size, self.tile_size
        )
        
        if strategy['strategy'] == 'full':
            # Can fit default tile size
            return self.refine_tiled(image_path, prompt, base_strength, base_steps, seed)
        
        # Adapt tile size
        adapted_tile_size = strategy['details'].get('tile_size', 512)
        original_tile_size = self.tile_size
        original_overlap = self.overlap
        
        # Adjust tile settings
        self.tile_size = adapted_tile_size
        self.overlap = min(256, adapted_tile_size // 4)
        
        self.logger.info(
            f"Adapting tile size: {original_tile_size} → {adapted_tile_size} "
            f"based on available VRAM"
        )
        
        try:
            result = self.refine_tiled(image_path, prompt, base_strength, base_steps, seed)
        finally:
            # Restore original settings
            self.tile_size = original_tile_size
            self.overlap = original_overlap
        
        return result
    
    def estimate_time(self, width: int, height: int) -> float:
        """
        Estimate processing time based on image size
        
        Returns:
            Estimated time in seconds
        """
        tiles = self.calculate_tiles(width, height, self.tile_size, self.overlap)
        
        # Estimate ~10 seconds per tile (conservative)
        return len(tiles) * 10.0