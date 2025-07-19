# AI Wallpaper Ultimate Resolution Upgrade Plan

## Overview
This document provides a comprehensive, step-by-step plan to upgrade the AI Wallpaper system to support configurable resolutions with zero quality compromises. The implementation prioritizes quality over speed, with special focus on tiled refinement and intelligent upscaling.

## Core Principles
1. **Never stretch or squash** - Only generate at optimal dimensions
2. **Time is irrelevant** - Quality is the only metric
3. **Tiled refinement** - Every part of the image gets refined
4. **Integer upscaling only** - Preserve maximum quality
5. **Smart composition** - Handle aspect ratios intelligently

## Phase 1: Core Infrastructure (Week 1)

### Step 1.1: Create Resolution Configuration System

#### File: `ai_wallpaper/core/resolution_manager.py` (NEW FILE)

```python
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
```

### Step 1.2: Update Configuration Files

#### File: `ai_wallpaper/config/resolution.yaml` (NEW FILE)

```yaml
# Resolution Configuration
resolution:
  # Default resolution if not specified
  default: "4K"
  
  # Quality mode
  quality_mode: "ultimate"  # "fast", "balanced", "ultimate"
  
  # Whether to allow custom resolutions
  allow_custom: true
  
  # Maximum supported resolution (to prevent memory issues)
  max_width: 15360  # 16K width
  max_height: 8640   # 16K height
  
  # Tiled refinement settings
  tiled_refinement:
    enabled: true
    tile_size: 1024
    overlap: 256
    passes: 2
    strength_decay: 0.1  # Reduce strength each pass
    
  # Upscaling preferences
  upscaling:
    prefer_integer_scales: true
    max_single_scale: 4
    ensemble_models: false  # Can enable for ultra quality
```

### Step 1.3: Add Resolution Parameters to CLI

#### File: `ai_wallpaper/commands/generate.py` (MODIFY)

Add these parameters to the `generate` command:

```python
@click.option(
    '--resolution',
    type=str,
    help='Target resolution as WIDTHxHEIGHT (e.g., 3840x2160) or preset name'
)
@click.option(
    '--quality-mode',
    type=click.Choice(['fast', 'balanced', 'ultimate']),
    default='balanced',
    help='Quality mode - ultimate takes longer but maximizes quality'
)
@click.option(
    '--no-tiled-refinement',
    is_flag=True,
    help='Disable tiled refinement pass (faster but lower quality)'
)
```

## Phase 2: Model Integration (Week 2)

### Step 2.1: Update Base Model Class

#### File: `ai_wallpaper/models/base_model.py` (MODIFY)

Add resolution support to the base class:

```python
# Add to imports
from ..core.resolution_manager import ResolutionManager, ResolutionConfig

# Add to BaseImageModel class
def __init__(self, config: Dict[str, Any]):
    """Initialize base model with resolution support"""
    super().__init__(config)
    self.resolution_manager = ResolutionManager()
    self.resolution_manager.logger = self.logger
    
def get_generation_params(self, target_resolution: Optional[Tuple[int, int]] = None, **kwargs) -> Dict[str, Any]:
    """Get generation parameters with resolution support"""
    params = super().get_generation_params(**kwargs)
    
    if target_resolution:
        # Calculate optimal generation size
        optimal_size = self.resolution_manager.get_optimal_generation_size(
            target_resolution, 
            self.model_name
        )
        params['generation_size'] = optimal_size
        params['target_resolution'] = target_resolution
        params['upscale_strategy'] = self.resolution_manager.calculate_upscale_strategy(
            optimal_size,
            target_resolution
        )
    
    return params
```

### Step 2.2: Update SDXL Model

#### File: `ai_wallpaper/models/sdxl_model.py` (MODIFY)

Replace the fixed dimensions with dynamic calculation:

```python
# In _generate_stage1 method, replace:
# width, height = 1344, 768  # Native SDXL 16:9

# With:
if 'generation_size' in params:
    width, height = params['generation_size']
    self.logger.info(f"Using calculated generation size: {width}x{height}")
else:
    # Fallback to default
    width, height = 1344, 768

# In _upscale_stage3 method, replace the entire scale calculation with:
if 'upscale_strategy' in params:
    # Use pre-calculated strategy
    strategy = params['upscale_strategy']
    self.logger.info(f"Using upscale strategy with {len(strategy)} steps")
    
    current_image_path = input_path
    
    for step in strategy:
        if step['method'] == 'realesrgan':
            current_image_path = self._apply_realesrgan(
                current_image_path,
                step['scale'],
                step['model'],
                temp_dirs
            )
        elif step['method'] == 'center_crop':
            current_image_path = self._apply_center_crop(
                current_image_path,
                step['output_size']
            )
    
    return self._standardize_stage_result(
        image_path=current_image_path,
        image=None,
        size=strategy[-1]['output_size']
    )
```

### Step 2.3: Add New Methods to SDXL Model

#### File: `ai_wallpaper/models/sdxl_model.py` (ADD METHODS)

```python
def _apply_realesrgan(self, input_path: Path, scale: int, model: str, temp_dirs: List[Path]) -> Path:
    """Apply Real-ESRGAN upscaling"""
    realesrgan_script = self._find_realesrgan()
    
    # Prepare output directory
    resolver = get_resolver()
    temp_dir = resolver.get_temp_dir() / 'ai-wallpaper'
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_output_dir = temp_dir / f"upscale_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    temp_output_dir.mkdir(exist_ok=True)
    temp_dirs.append(temp_output_dir)
    
    # Build command
    cmd = [
        sys.executable,
        str(realesrgan_script),
        "-n", model,
        "-i", str(input_path),
        "-o", str(temp_output_dir),
        "--outscale", str(scale),
        "-t", "512",  # Tile size
        "--fp32"      # Maximum precision
    ]
    
    # Execute
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    
    # Find output
    output_files = list(temp_output_dir.glob("*.png"))
    if not output_files:
        raise UpscalerError(str(input_path), FileNotFoundError("No output from Real-ESRGAN"))
    
    return output_files[0]

def _apply_center_crop(self, input_path: Path, target_size: Tuple[int, int]) -> Path:
    """Apply center cropping to exact dimensions"""
    target_w, target_h = target_size
    
    with Image.open(input_path) as img:
        current_w, current_h = img.size
        
        # Calculate crop box
        left = (current_w - target_w) // 2
        top = (current_h - target_h) // 2
        right = left + target_w
        bottom = top + target_h
        
        # Crop
        cropped = img.crop((left, top, right, bottom))
        
        # Save
        resolver = get_resolver()
        temp_path = resolver.get_temp_dir() / f"cropped_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.png"
        cropped.save(temp_path, "PNG", quality=100)
        
        return temp_path
```

## Phase 3: Tiled Refinement System (Week 3)

### Step 3.1: Create Tiled Refinement Engine

#### File: `ai_wallpaper/processing/tiled_refiner.py` (NEW FILE)

```python
#!/usr/bin/env python3
"""
Tiled Refinement Engine
Refines images tile by tile for maximum quality
"""

from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from PIL import Image
import numpy as np
from pathlib import Path

@dataclass
class TileInfo:
    """Information about a single tile"""
    id: int
    x: int
    y: int
    width: int
    height: int
    context: Optional[str] = None  # e.g., "sky", "detail", "face"

class TiledRefiner:
    """Performs tiled img2img refinement for ultra quality"""
    
    def __init__(self, sdxl_pipeline):
        self.pipeline = sdxl_pipeline
        self.logger = None  # Set by caller
        
    def refine_image(self, 
                    image_path: Path,
                    prompt: str,
                    tile_size: int = 1024,
                    overlap: int = 256,
                    passes: int = 2,
                    initial_strength: float = 0.3) -> Path:
        """
        Refine an image using tiled img2img.
        
        Args:
            image_path: Path to image to refine
            prompt: Original prompt
            tile_size: Size of each tile
            overlap: Overlap between tiles
            passes: Number of refinement passes
            initial_strength: Denoising strength for first pass
            
        Returns:
            Path to refined image
        """
        # Load image
        image = Image.open(image_path)
        width, height = image.size
        
        self.logger.info(f"Starting tiled refinement: {width}x{height}, tile_size={tile_size}, passes={passes}")
        
        # Create tile plan
        tiles = self._create_tile_plan(width, height, tile_size, overlap)
        self.logger.info(f"Created {len(tiles)} tiles for refinement")
        
        # Perform multiple passes
        refined_image = image.copy()
        
        for pass_num in range(passes):
            strength = initial_strength - (pass_num * 0.1)  # Reduce strength each pass
            strength = max(0.1, strength)  # Minimum strength
            
            self.logger.info(f"Refinement pass {pass_num + 1}/{passes}, strength={strength}")
            
            # Process each tile
            for i, tile_info in enumerate(tiles):
                self.logger.debug(f"Processing tile {i + 1}/{len(tiles)}")
                
                # Extract tile with padding
                tile, padding = self._extract_tile_with_padding(refined_image, tile_info, overlap // 2)
                
                # Refine tile
                refined_tile = self._refine_single_tile(
                    tile,
                    prompt,
                    strength=strength,
                    steps=50 + (pass_num * 10)  # More steps each pass
                )
                
                # Blend back
                refined_image = self._blend_tile(
                    refined_image,
                    refined_tile,
                    tile_info,
                    padding,
                    overlap // 2
                )
        
        # Save result
        output_path = image_path.parent / f"{image_path.stem}_refined{image_path.suffix}"
        refined_image.save(output_path, "PNG", quality=100)
        
        self.logger.info(f"Tiled refinement complete: {output_path}")
        return output_path
    
    def _create_tile_plan(self, width: int, height: int, 
                         tile_size: int, overlap: int) -> List[TileInfo]:
        """Create optimal tile layout"""
        tiles = []
        tile_id = 0
        
        effective_tile_size = tile_size - overlap
        
        for y in range(0, height, effective_tile_size):
            for x in range(0, width, effective_tile_size):
                # Calculate tile boundaries
                tile_width = min(tile_size, width - x)
                tile_height = min(tile_size, height - y)
                
                tiles.append(TileInfo(
                    id=tile_id,
                    x=x,
                    y=y,
                    width=tile_width,
                    height=tile_height
                ))
                tile_id += 1
                
                # Stop if we've covered the width
                if x + tile_width >= width:
                    break
            
            # Stop if we've covered the height
            if y + tile_height >= height:
                break
        
        return tiles
    
    def _extract_tile_with_padding(self, image: Image.Image, 
                                  tile_info: TileInfo, 
                                  padding: int) -> Tuple[Image.Image, Dict]:
        """Extract tile with padding for seamless blending"""
        # Calculate padded boundaries
        x1 = max(0, tile_info.x - padding)
        y1 = max(0, tile_info.y - padding)
        x2 = min(image.width, tile_info.x + tile_info.width + padding)
        y2 = min(image.height, tile_info.y + tile_info.height + padding)
        
        # Extract padded tile
        tile = image.crop((x1, y1, x2, y2))
        
        # Calculate actual padding used
        padding_info = {
            'left': tile_info.x - x1,
            'top': tile_info.y - y1,
            'right': x2 - (tile_info.x + tile_info.width),
            'bottom': y2 - (tile_info.y + tile_info.height)
        }
        
        return tile, padding_info
    
    def _refine_single_tile(self, tile: Image.Image, prompt: str, 
                           strength: float, steps: int) -> Image.Image:
        """Refine a single tile using img2img"""
        # Ensure tile dimensions are good for SDXL
        width, height = tile.size
        
        # Round to nearest 64 for SDXL
        new_width = ((width + 31) // 64) * 64
        new_height = ((height + 31) // 64) * 64
        
        if new_width != width or new_height != height:
            # Resize for processing
            tile_resized = tile.resize((new_width, new_height), Image.Resampling.LANCZOS)
        else:
            tile_resized = tile
        
        # Run img2img
        result = self.pipeline(
            prompt=prompt,
            image=tile_resized,
            strength=strength,
            num_inference_steps=steps,
            guidance_scale=7.5
        ).images[0]
        
        # Resize back if needed
        if result.size != tile.size:
            result = result.resize(tile.size, Image.Resampling.LANCZOS)
        
        return result
    
    def _blend_tile(self, base_image: Image.Image, tile: Image.Image,
                   tile_info: TileInfo, padding: Dict, blend_width: int) -> Image.Image:
        """Blend tile back into base image with feathering"""
        # Convert to numpy for easier blending
        base_array = np.array(base_image)
        tile_array = np.array(tile)
        
        # Create blend mask
        mask = self._create_blend_mask(tile.size, padding, blend_width)
        
        # Calculate position in base image
        x1 = tile_info.x - padding['left']
        y1 = tile_info.y - padding['top']
        x2 = x1 + tile.width
        y2 = y1 + tile.height
        
        # Blend
        base_region = base_array[y1:y2, x1:x2]
        blended = (base_region * (1 - mask[:, :, np.newaxis]) + 
                  tile_array * mask[:, :, np.newaxis])
        
        # Put back
        result_array = base_array.copy()
        result_array[y1:y2, x1:x2] = blended.astype(np.uint8)
        
        return Image.fromarray(result_array)
    
    def _create_blend_mask(self, size: Tuple[int, int], 
                          padding: Dict, blend_width: int) -> np.ndarray:
        """Create feathered blend mask"""
        width, height = size
        mask = np.ones((height, width), dtype=np.float32)
        
        # Feather edges
        for y in range(height):
            for x in range(width):
                # Distance from edges
                dist_left = x - padding['left'] if x < padding['left'] + blend_width else float('inf')
                dist_top = y - padding['top'] if y < padding['top'] + blend_width else float('inf')
                dist_right = width - x - padding['right'] if x >= width - padding['right'] - blend_width else float('inf')
                dist_bottom = height - y - padding['bottom'] if y >= height - padding['bottom'] - blend_width else float('inf')
                
                # Minimum distance to any edge
                min_dist = min(dist_left, dist_top, dist_right, dist_bottom)
                
                if min_dist < blend_width:
                    # Feather
                    mask[y, x] = min_dist / blend_width
        
        return mask
```

### Step 3.2: Integrate Tiled Refinement into Models

#### File: `ai_wallpaper/models/sdxl_model.py` (MODIFY)

Add tiled refinement as the final stage:

```python
# Add to imports
from ..processing.tiled_refiner import TiledRefiner

# In generate method, after stage 4 (finalize), add:
# Stage 5: Tiled Refinement (if enabled)
if params.get('quality_mode') == 'ultimate' and not params.get('no_tiled_refinement', False):
    self.logger.log_stage("Stage 5", "Tiled ultra-refinement")
    
    # Initialize refiner with current pipeline
    refiner = TiledRefiner(self.refiner_pipe if self.refiner_pipe else self.pipe)
    refiner.logger = self.logger
    
    # Get refinement settings
    refinement_config = get_config().resolution.get('tiled_refinement', {})
    
    refined_path = refiner.refine_image(
        image_path=final_path,
        prompt=prompt,
        tile_size=refinement_config.get('tile_size', 1024),
        overlap=refinement_config.get('overlap', 256),
        passes=refinement_config.get('passes', 2),
        initial_strength=0.3
    )
    
    # Update final path
    final_path = refined_path
    
    self.logger.info("Tiled refinement complete - maximum quality achieved")
```

## Phase 4: Aspect Ratio Handling (Week 4)

### Step 4.1: Create Intelligent Compositor

#### File: `ai_wallpaper/processing/intelligent_compositor.py` (NEW FILE)

```python
#!/usr/bin/env python3
"""
Intelligent Composition System
Handles aspect ratios without stretching or squashing
"""

from typing import Tuple, Optional
from PIL import Image
import numpy as np

class IntelligentCompositor:
    """Handles aspect ratio mismatches intelligently"""
    
    def __init__(self):
        self.logger = None  # Set by caller
        
    def handle_aspect_mismatch(self,
                              image_path: Path,
                              target_size: Tuple[int, int],
                              method: str = "smart_crop") -> Path:
        """
        Handle aspect ratio mismatch without distortion.
        
        Args:
            image_path: Source image
            target_size: Target dimensions
            method: "smart_crop", "pad", or "extend"
            
        Returns:
            Path to adjusted image
        """
        image = Image.open(image_path)
        source_w, source_h = image.size
        target_w, target_h = target_size
        
        source_aspect = source_w / source_h
        target_aspect = target_w / target_h
        
        self.logger.info(f"Handling aspect mismatch: {source_aspect:.2f} -> {target_aspect:.2f}")
        
        if abs(source_aspect - target_aspect) < 0.01:
            # Aspects match, just resize if needed
            if (source_w, source_h) != (target_w, target_h):
                return self._high_quality_resize(image, target_size)
            return image_path
        
        if method == "smart_crop":
            return self._smart_crop(image, target_size)
        elif method == "pad":
            return self._pad_to_aspect(image, target_size)
        elif method == "extend":
            return self._extend_borders(image, target_size)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _smart_crop(self, image: Image.Image, target_size: Tuple[int, int]) -> Path:
        """Intelligently crop to target aspect ratio"""
        source_w, source_h = image.size
        target_w, target_h = target_size
        target_aspect = target_w / target_h
        
        # Calculate crop dimensions
        if source_w / source_h > target_aspect:
            # Image is too wide
            new_width = int(source_h * target_aspect)
            new_height = source_h
        else:
            # Image is too tall
            new_width = source_w
            new_height = int(source_w / target_aspect)
        
        # Center crop
        left = (source_w - new_width) // 2
        top = (source_h - new_height) // 2
        right = left + new_width
        bottom = top + new_height
        
        cropped = image.crop((left, top, right, bottom))
        
        # Resize to exact target
        if cropped.size != target_size:
            cropped = cropped.resize(target_size, Image.Resampling.LANCZOS)
        
        # Save
        output_path = Path(image.filename).parent / f"{Path(image.filename).stem}_cropped.png"
        cropped.save(output_path, "PNG", quality=100)
        
        return output_path
    
    def _pad_to_aspect(self, image: Image.Image, target_size: Tuple[int, int]) -> Path:
        """Pad image to target aspect ratio"""
        # Implementation for padding with reflected/blurred borders
        # This maintains image without cropping
        pass
    
    def _extend_borders(self, image: Image.Image, target_size: Tuple[int, int]) -> Path:
        """Extend borders using reflection or inpainting"""
        # Implementation for border extension
        # Could use outpainting in future
        pass
```

## Phase 5: Testing and Integration

### Step 5.1: Create Test Suite

#### File: `tests/test_resolution_system.py` (NEW FILE)

```python
#!/usr/bin/env python3
"""Test suite for resolution system"""

import pytest
from ai_wallpaper.core.resolution_manager import ResolutionManager

def test_sdxl_optimal_dimensions():
    """Test SDXL dimension calculation"""
    rm = ResolutionManager()
    
    # Test 16:9 target
    result = rm.get_optimal_generation_size((3840, 2160), "sdxl")
    assert result == (1344, 768)  # Should pick 16:9 SDXL size
    
    # Test ultrawide
    result = rm.get_optimal_generation_size((5120, 2160), "sdxl")
    assert result[0] / result[1] > 2.0  # Should pick wide aspect

def test_upscale_strategy():
    """Test upscaling strategy calculation"""
    rm = ResolutionManager()
    
    # Test 2x scale
    strategy = rm.calculate_upscale_strategy((1344, 768), (2688, 1536))
    assert len(strategy) == 1
    assert strategy[0]['method'] == 'realesrgan'
    assert strategy[0]['scale'] == 2
    
    # Test non-integer scale
    strategy = rm.calculate_upscale_strategy((1344, 768), (3840, 2160))
    assert len(strategy) >= 2  # Should have upscale + crop
```

### Step 5.2: Manual Testing Protocol

Create a test script to verify all components:

```bash
#!/bin/bash
# test_resolution_upgrade.sh

echo "Testing Resolution Upgrade System"

# Test 1: Standard 4K
echo "Test 1: Standard 4K generation"
AI_WALLPAPER_VENV=/home/user/grace/.venv/bin/python ./ai-wallpaper generate \
    --model sdxl \
    --resolution 3840x2160 \
    --quality-mode ultimate \
    --prompt "test landscape" \
    --no-wallpaper

# Test 2: Ultrawide 5K
echo "Test 2: Ultrawide 5K generation"
AI_WALLPAPER_VENV=/home/user/grace/.venv/bin/python ./ai-wallpaper generate \
    --model sdxl \
    --resolution 5120x2160 \
    --quality-mode ultimate \
    --prompt "test ultrawide scene" \
    --no-wallpaper

# Test 3: Portrait orientation
echo "Test 3: Portrait orientation"
AI_WALLPAPER_VENV=/home/user/grace/.venv/bin/python ./ai-wallpaper generate \
    --model sdxl \
    --resolution 2160x3840 \
    --quality-mode ultimate \
    --prompt "test portrait composition" \
    --no-wallpaper

# Test 4: Extreme aspect ratio
echo "Test 4: Triple monitor setup"
AI_WALLPAPER_VENV=/home/user/grace/.venv/bin/python ./ai-wallpaper generate \
    --model sdxl \
    --resolution 5760x1080 \
    --quality-mode ultimate \
    --prompt "test panoramic view" \
    --no-wallpaper
```

## Implementation Timeline

### Week 1: Core Infrastructure
- Day 1-2: Implement ResolutionManager
- Day 3-4: Update configuration system
- Day 5: Update CLI and test basic functionality

### Week 2: Model Integration  
- Day 1-2: Update BaseImageModel
- Day 3-4: Modify SDXL model
- Day 5: Test generation at different resolutions

### Week 3: Tiled Refinement
- Day 1-2: Implement TiledRefiner
- Day 3-4: Integrate with SDXL pipeline
- Day 5: Test refinement quality

### Week 4: Aspect Ratio & Polish
- Day 1-2: Implement IntelligentCompositor
- Day 3: Add aspect ratio handling
- Day 4-5: Full system testing and optimization

## Key Implementation Notes

1. **Always preserve original aspect ratios** - Never stretch or squash
2. **Use integer upscaling** when possible for maximum quality
3. **Implement gradually** - Test each component before moving on
4. **Log everything** - Detailed logging helps debug issues
5. **Memory management** - Clear VRAM between stages
6. **Save intermediates** - Helps verify each stage works correctly

## Configuration Examples

After implementation, users can:

```bash
# Standard 4K with ultimate quality
./ai-wallpaper generate --resolution 4K --quality-mode ultimate

# Custom resolution
./ai-wallpaper generate --resolution 5120x2880 --quality-mode ultimate

# Disable refinement for faster generation
./ai-wallpaper generate --resolution 4K --no-tiled-refinement
```

## Success Criteria

1. Generate at any resolution without stretching/squashing
2. Tiled refinement improves visible detail
3. Aspect ratios handled intelligently
4. Quality mode produces visibly superior results
5. System remains stable at high resolutions

This plan provides maximum quality at any resolution while being implementable in manageable phases.

## Implementation Progress

### Phase 1: Core Infrastructure
- [x] Step 1.1: Create Resolution Configuration System ✓ COMPLETED
  - Created `ai_wallpaper/core/resolution_manager.py` with full ResolutionManager implementation
  - Includes ResolutionConfig dataclass for resolution metadata
  - Supports SDXL, FLUX, and DALL-E optimal dimension calculations
  - Implements integer-only upscaling strategy calculation
  - Ready for integration with existing models
  
- [x] Step 1.2: Update Configuration Files ✓ COMPLETED
  - Created `ai_wallpaper/config/resolution.yaml` with comprehensive resolution settings
  - Includes quality modes (fast/balanced/ultimate)
  - Tiled refinement configuration with overlap and passes
  - Upscaling preferences for maximum quality
  - Resolution presets and limits defined
  
- [x] Step 1.3: Add Resolution Parameters to CLI ✓ COMPLETED
  - Added --resolution parameter (WIDTHxHEIGHT or preset name)
  - Added --quality-mode parameter (fast/balanced/ultimate)
  - Added --no-tiled-refinement flag
  - Updated CLI main.py with new options
  - Updated GenerateCommand execute method signature
  - Parameters ready to be passed to model implementations

### Phase 1 Summary ✓ COMPLETED
Phase 1 Core Infrastructure is now complete! The system has:
- ResolutionManager class for intelligent resolution calculations
- Configuration file for resolution settings and quality modes  
- CLI parameters for resolution control
- GenerateCommand integration to pass resolution to models
- Verified functionality with comprehensive testing

Test Results:
- ✓ ResolutionConfig properly calculates aspect ratios and pixel counts
- ✓ SDXL optimal dimensions correctly scale up for large targets (1.5x for 4K+)
- ✓ FLUX constraints properly enforced (16-divisible, max 2048px)
- ✓ Upscale strategy uses integer scaling with Real-ESRGAN
- ✓ 10 resolution presets available from 1080p to 8K

Next step: Phase 2 - Model Integration to actually use these resolution parameters in generation.

### Phase 2-5: Not Started