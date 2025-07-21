# SLIDING WINDOW PROGRESSIVE OUTPAINTING (SWPO) IMPLEMENTATION PLAN
# Maximum Quality Through Incremental Context-Preserving Expansion

## 🎯 Core Concept
Replace large chunk expansion (2x, 1.5x steps) with sliding window approach:
- Expand by only 100-200px at a time
- 80% overlap between windows for maximum context retention
- Many small steps instead of few large ones
- Original image always dominates the context

## Architecture Overview

### Current Pipeline Flow:
1. `SdxlModel.generate()` → detects aspect adjustment needed
2. `ResolutionManager.calculate_progressive_outpaint_strategy()` → creates large steps
3. `AspectAdjuster._progressive_adjust()` → executes steps
4. `AspectAdjuster._execute_outpaint_step()` → performs single expansion

### New SWPO Flow:
1. `SdxlModel.generate()` → detects aspect adjustment needed
2. `ResolutionManager.calculate_sliding_window_strategy()` → creates many small steps
3. `AspectAdjuster._sliding_window_adjust()` → executes overlapping windows
4. `AspectAdjuster._execute_sliding_window()` → performs single window expansion

---

## PHASE 1: Core Implementation

### Step 1.1: Add SWPO Configuration
**File**: `ai_wallpaper/config/resolution.yaml`
**Location**: Add new section after line 101

**ADD**:
```yaml
  # Sliding Window Progressive Outpainting (SWPO)
  sliding_window:
    enabled: true
    window_size: 200           # Pixels to expand per step
    overlap_ratio: 0.8         # 80% overlap between windows
    min_window_size: 100       # Minimum expansion per step
    max_window_size: 300       # Maximum expansion per step
    
    # Quality settings for SWPO
    denoising_strength: 0.95   # High strength for content generation
    guidance_scale: 7.5        # Consistent guidance
    inference_steps: 60        # Steps per window
    
    # Blending settings
    edge_blur_width: 20        # Narrow blur for precise transitions
    blend_mode: "linear"       # linear, cosine, or gaussian
    
    # Memory optimization
    clear_cache_every_n_windows: 5  # Clear CUDA cache periodically
    save_intermediate_windows: false # Debug option
```

### Step 1.2: Create Sliding Window Strategy Calculator
**File**: `ai_wallpaper/core/resolution_manager.py`
**Location**: Add new method after `calculate_progressive_outpaint_strategy` (after line 408)

**ADD**:
```python
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
            
            temp_w = temp_w + step_size  # Move by step size, not window size
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
            
            temp_h = temp_h + step_size
            window_num += 1
    
    self.logger.info(
        f"Sliding window strategy: {len(steps)} windows "
        f"({window_size}px window, {step_size}px step, {overlap_ratio:.0%} overlap)"
    )
    
    return steps
```

### Step 1.3: Implement Sliding Window Adjuster
**File**: `ai_wallpaper/processing/aspect_adjuster.py`
**Location**: Add new method after `_progressive_adjust` (after line 212)

**ADD**:
```python
def _sliding_window_adjust(self,
                          image_path: Path,
                          prompt: str,
                          steps: List[Dict],
                          save_intermediates: bool = False) -> Path:
    """
    Perform sliding window adjustment with maximum context preservation.
    Each window overlaps significantly with previous content.
    NO ERROR TOLERANCE - FAIL LOUD
    """
    current_path = image_path
    swpo_config = self.config.progressive_outpainting.get('sliding_window', {})
    
    # SWPO-specific settings
    denoising_strength = swpo_config.get('denoising_strength', 0.95)
    edge_blur = swpo_config.get('edge_blur_width', 20)
    clear_cache_interval = swpo_config.get('clear_cache_every_n_windows', 5)
    
    self.logger.info(f"Starting SWPO adjustment: {len(steps)} windows")
    
    # Track accumulated expansions for proper mask positioning
    accumulated_horizontal = 0
    accumulated_vertical = 0
    original_w, original_h = Image.open(image_path).size
    
    for i, step in enumerate(steps):
        window_num = i + 1
        self.logger.log_stage(
            f"SWPO Window {window_num}/{len(steps)}", 
            step['description']
        )
        
        # Clear CUDA cache periodically to prevent OOM
        if window_num % clear_cache_interval == 0:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                self.logger.info("Cleared CUDA cache")
        
        # Execute sliding window expansion
        current_path = self._execute_sliding_window(
            current_path, 
            prompt, 
            step,
            denoising_strength,
            edge_blur,
            accumulated_horizontal,
            accumulated_vertical,
            original_w,
            original_h
        )
        
        # Update accumulated expansions
        if step['direction'] == 'horizontal':
            accumulated_horizontal += (step['target_size'][0] - step['current_size'][0]) - step.get('overlap_size', 0)
        else:
            accumulated_vertical += (step['target_size'][1] - step['current_size'][1]) - step.get('overlap_size', 0)
        
        # Save intermediate if requested (--save-stages support)
        if save_intermediates:
            self._save_swpo_stage(
                current_path, 
                f"swpo_window_{window_num}",
                step['target_size'],
                step['description']
            )
        
        # Validate result
        if not current_path.exists():
            raise RuntimeError(f"Window {window_num} failed to produce output")
    
    self.logger.info(f"SWPO adjustment complete: {len(steps)} windows processed")
    return current_path
```

### Step 1.4: Implement Single Window Execution
**File**: `ai_wallpaper/processing/aspect_adjuster.py`
**Location**: Add new method after `_sliding_window_adjust`

**ADD**:
```python
def _execute_sliding_window(self,
                           image_path: Path,
                           prompt: str,
                           step_info: Dict,
                           denoising_strength: float,
                           edge_blur: int,
                           accumulated_h: int,
                           accumulated_v: int,
                           original_w: int,
                           original_h: int) -> Path:
    """
    Execute a single sliding window expansion.
    Critical: Mask positioning must account for accumulated expansions.
    """
    # Load current image
    image = Image.open(image_path)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    current_w, current_h = step_info['current_size']
    target_w, target_h = step_info['target_size']
    window_size = step_info['window_size']
    overlap_size = step_info.get('overlap_size', 0)
    direction = step_info['direction']
    
    # Create canvas at target size
    canvas = Image.new('RGB', (target_w, target_h), color='gray')
    mask = Image.new('L', (target_w, target_h), color='black')
    
    # Position existing image
    if direction == 'horizontal':
        # For horizontal expansion, image stays at left
        canvas.paste(image, (0, 0))
        
        # Critical: Mask only the NEW area being generated
        # Account for overlap from previous windows
        mask_start = current_w - overlap_size
        mask_end = target_w
        
        # Create gradient mask for smooth blending
        mask_array = np.zeros((target_h, target_w), dtype=np.uint8)
        mask_array[:, mask_start:mask_end] = 255
        
        # Apply edge blur to blend with existing content
        if edge_blur > 0 and overlap_size > 0:
            for x in range(max(0, mask_start - edge_blur), min(mask_start + edge_blur, target_w)):
                if 0 <= x < target_w:
                    weight = (x - (mask_start - edge_blur)) / (2 * edge_blur)
                    weight = np.clip(weight, 0, 1)
                    mask_array[:, x] = int(weight * 255)
    else:
        # For vertical expansion, image stays at top
        canvas.paste(image, (0, 0))
        
        mask_start = current_h - overlap_size
        mask_end = target_h
        
        mask_array = np.zeros((target_h, target_w), dtype=np.uint8)
        mask_array[mask_start:mask_end, :] = 255
        
        if edge_blur > 0 and overlap_size > 0:
            for y in range(max(0, mask_start - edge_blur), min(mask_start + edge_blur, target_h)):
                if 0 <= y < target_h:
                    weight = (y - (mask_start - edge_blur)) / (2 * edge_blur)
                    weight = np.clip(weight, 0, 1)
                    mask_array[y, :] = int(weight * 255)
    
    mask = Image.fromarray(mask_array, mode='L')
    
    # Add noise to masked areas for better generation
    canvas_array = np.array(canvas)
    mask_bool = mask_array > 128
    noise = np.random.randint(20, 60, size=canvas_array.shape, dtype=np.uint8)
    for c in range(3):
        canvas_array[:, :, c][mask_bool] = noise[:, :, c][mask_bool]
    canvas = Image.fromarray(canvas_array)
    
    # Enhanced prompt for window context
    window_prompt = f"{prompt}, seamlessly continuing the existing content"
    if direction == 'horizontal':
        window_prompt += ", extending naturally to the right"
    else:
        window_prompt += ", extending naturally downward"
    
    # Log window details
    self.logger.info(
        f"SWPO window: {current_w}x{current_h} → {target_w}x{target_h} "
        f"(+{window_size}px, {overlap_size}px overlap)"
    )
    
    # Execute inpainting
    result = self.pipeline(
        prompt=window_prompt,
        image=canvas,
        mask_image=mask,
        strength=denoising_strength,
        num_inference_steps=self.base_steps,
        guidance_scale=self.pipeline.guidance_scale if hasattr(self.pipeline, 'guidance_scale') else 7.5,
        width=target_w,
        height=target_h
    ).images[0]
    
    # Save result
    temp_dir = self.resolver.get_temp_dir() / 'ai-wallpaper' / 'swpo'
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    filename = f"swpo_{direction}_{current_w}x{current_h}_to_{target_w}x{target_h}_{timestamp}.png"
    save_path = temp_dir / filename
    
    # Save with lossless compression
    from ..utils import save_lossless_png
    save_lossless_png(result, save_path)
    
    return save_path
```

### Step 1.5: Add Stage Saving Support Method
**File**: `ai_wallpaper/processing/aspect_adjuster.py`
**Location**: Add new method after `_execute_sliding_window`

**ADD**:
```python
def _save_swpo_stage(self, 
                     image_path: Path,
                     stage_name: str,
                     size: Tuple[int, int],
                     description: str):
    """
    Save intermediate SWPO stage for debugging/visualization
    Follows the same pattern as other stage saves in the pipeline
    """
    # Get the stage directory from the model metadata
    if hasattr(self, 'model_metadata') and self.model_metadata:
        stage_dir = self.model_metadata.get('stage_dir')
        if stage_dir:
            stage_dir = Path(stage_dir)
            stage_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy the image to stage directory with descriptive name
            stage_filename = f"{stage_name}_{size[0]}x{size[1]}.png"
            stage_path = stage_dir / stage_filename
            
            import shutil
            shutil.copy2(image_path, stage_path)
            self.logger.info(f"Saved SWPO stage: {stage_filename} - {description}")
```

### Step 1.6: Update SDXL Model to Use SWPO
**File**: `ai_wallpaper/models/sdxl_model.py`
**Location**: In Stage 1.5 (around line 223), modify the progressive steps calculation

**REPLACE** (around line 230-235):
```python
# Calculate progressive steps if needed
target_aspect = sdxl_params['width'] / sdxl_params['height']
progressive_steps = self.resolution_manager.calculate_progressive_outpaint_strategy(
    current_size=(image.width, image.height),
    target_aspect=target_aspect
)
```

**WITH**:
```python
# Check if SWPO is enabled
swpo_config = self.config.resolution.progressive_outpainting.get('sliding_window', {})
use_swpo = swpo_config.get('enabled', True)

if use_swpo:
    # Calculate sliding window steps
    self.logger.info("Using Sliding Window Progressive Outpainting (SWPO)")
    progressive_steps = self.resolution_manager.calculate_sliding_window_strategy(
        current_size=(image.width, image.height),
        target_size=(sdxl_params['width'], sdxl_params['height']),
        window_size=swpo_config.get('window_size', 200),
        overlap_ratio=swpo_config.get('overlap_ratio', 0.8)
    )
else:
    # Fall back to original progressive strategy
    target_aspect = sdxl_params['width'] / sdxl_params['height']
    progressive_steps = self.resolution_manager.calculate_progressive_outpaint_strategy(
        current_size=(image.width, image.height),
        target_aspect=target_aspect
    )
```

### Step 1.6.5: Pass Model Metadata to AspectAdjuster
**File**: `ai_wallpaper/models/sdxl_model.py`
**Location**: Where AspectAdjuster is called (around the progressive adjustment section)

**ENSURE** the AspectAdjuster receives model_metadata for stage saving:
```python
# Pass model metadata to aspect adjuster for stage saving
self.aspect_adjuster.model_metadata = self.generation_metadata
```

### Step 1.7: Update AspectAdjuster to Route to SWPO
**File**: `ai_wallpaper/processing/aspect_adjuster.py`
**Location**: In `adjust_aspect_ratio` method (around line 140)

**REPLACE** (around line 186-190):
```python
# Use progressive strategy if needed
if progressive_steps:
    return self._progressive_adjust(
        image_path, original_prompt, progressive_steps, save_intermediates
    )
```

**WITH**:
```python
# Use progressive strategy if needed
if progressive_steps:
    # Check if these are SWPO steps
    if progressive_steps and progressive_steps[0].get('method') == 'sliding_window':
        return self._sliding_window_adjust(
            image_path, original_prompt, progressive_steps, save_intermediates
        )
    else:
        # Original progressive adjust
        return self._progressive_adjust(
            image_path, original_prompt, progressive_steps, save_intermediates
        )
```

---

## PHASE 2: Quality Enhancements

### Step 2.1: Add Color Consistency Enforcement
**File**: `ai_wallpaper/processing/aspect_adjuster.py`
**Location**: Add new method after `_analyze_edge_colors`

**ADD**:
```python
def _enforce_color_consistency(self, 
                              generated_image: Image.Image,
                              reference_image: Image.Image,
                              strength: float = 0.5) -> Image.Image:
    """
    Enforce color consistency between generated and reference images.
    Uses histogram matching to maintain color palette.
    """
    if strength <= 0:
        return generated_image
    
    gen_array = np.array(generated_image)
    ref_array = np.array(reference_image)
    result_array = np.zeros_like(gen_array)
    
    # Apply histogram matching per channel
    from skimage.exposure import match_histograms
    
    # Match histograms
    matched = match_histograms(gen_array, ref_array, channel_axis=2)
    
    # Blend based on strength
    result_array = (1 - strength) * gen_array + strength * matched
    
    return Image.fromarray(result_array.astype(np.uint8))
```

### Step 2.2: Add Final Unification Pass
**File**: `ai_wallpaper/processing/aspect_adjuster.py`
**Location**: Add at the end of `_sliding_window_adjust` before return

**ADD**:
```python
# Optional: Final unification pass
if swpo_config.get('final_unification_pass', True):
    self.logger.info("Performing final unification pass")
    
    # Load the result
    final_image = Image.open(current_path)
    
    # Very light refinement pass on the full image
    unification_strength = swpo_config.get('unification_strength', 0.15)
    
    result = self.pipeline(
        prompt=prompt + ", perfectly unified and seamless composition",
        image=final_image,
        strength=unification_strength,
        num_inference_steps=40,
        guidance_scale=7.0,
        width=final_image.width,
        height=final_image.height
    ).images[0]
    
    # Save unified result
    unified_path = current_path.parent / f"unified_{current_path.name}"
    save_lossless_png(result, unified_path)
    current_path = unified_path
    
    # Save unification stage if requested (--save-stages support)
    if save_intermediates:
        self._save_swpo_stage(
            current_path,
            "swpo_final_unification",
            (final_image.width, final_image.height),
            "Final unification pass"
        )
```

---

## PHASE 3: Configuration & Testing

### Step 3.1: Add SWPO Toggle to CLI
**File**: `ai_wallpaper/cli.py` (if exists) or wherever CLI arguments are defined
**Location**: Add new argument

**ADD**:
```python
parser.add_argument(
    '--swpo', 
    action='store_true',
    help='Use Sliding Window Progressive Outpainting for extreme aspect ratios'
)
parser.add_argument(
    '--window-size',
    type=int,
    default=200,
    help='Window size for SWPO (default: 200 pixels)'
)
parser.add_argument(
    '--overlap-ratio',
    type=float,
    default=0.8,
    help='Overlap ratio for SWPO windows (default: 0.8)'
)
```

### Step 3.2: Update Default Configuration
**File**: `ai_wallpaper/config/resolution.yaml`
**Location**: Update sliding_window defaults

**ENSURE**:
- `enabled: true` to make SWPO the default
- Reasonable defaults for all parameters
- Clear documentation comments

---

## Implementation Order & Timeline

1. **Hour 1**: Implement Step 1.1-1.2 (Config & Strategy Calculator)
2. **Hour 2**: Implement Step 1.3-1.4 (Core SWPO Logic)
3. **Hour 3**: Implement Step 1.5-1.6 (Integration)
4. **Hour 4**: Test basic functionality
5. **Hour 5**: Implement Phase 2 (Quality Enhancements)
6. **Hour 6**: Final testing and debugging

## Expected Results

For a 1344x768 → 5376x768 expansion:
- **Old method**: 3 large steps (2x, 1.5x, 1.12x)
- **SWPO method**: ~20 windows of 200px with 160px overlap

Each window sees 80% existing content, ensuring perfect continuity.

## Testing Commands

```bash
# Test basic SWPO
./ai-wallpaper generate --model sdxl --resolution 5120x1080 --swpo

# Test with custom window size
./ai-wallpaper generate --model sdxl --resolution 7680x2160 --swpo --window-size 150

# Test with less overlap
./ai-wallpaper generate --model sdxl --resolution 10240x1080 --swpo --overlap-ratio 0.7
```

## Success Metrics

1. **No visible seams** between windows
2. **Consistent color palette** throughout
3. **Natural content continuation**
4. **Preserved detail** from original
5. **Reasonable generation time** (linear with expansion size)

## Rollback Plan

If SWPO needs to be disabled:
1. Set `sliding_window.enabled: false` in config
2. System automatically falls back to original progressive method
3. No code changes needed

This implementation preserves all existing functionality while adding SWPO as an enhancement!