#!/usr/bin/env python3
"""
AI-Based Progressive Aspect Ratio Adjustment System
Uses progressive img2img outpainting for extreme ratios without seams
COMPLETE IMPLEMENTATION - NO FALLBACKS, LOUD FAILURES
"""

from typing import Tuple, Optional, Dict, Any, List
from PIL import Image, ImageFilter, ImageDraw
import numpy as np
from pathlib import Path
from datetime import datetime
import torch  # For CUDA error handling

from ..core import get_logger, get_config
from ..core.path_resolver import get_resolver

class AspectAdjuster:
    """Progressive aspect ratio adjustment with zero quality compromise"""
    
    def __init__(self, pipeline=None):
        """
        Args:
            pipeline: SDXL inpaint pipeline (NOT img2img!)
        """
        self.pipeline = pipeline
        self.logger = get_logger()
        self.config = get_config()
        self.resolver = get_resolver()
        
        # Validate pipeline type - STRICT checking
        if pipeline is not None:
            pipeline_class = type(pipeline).__name__
            # Must be specifically an inpaint pipeline
            if not any(x in pipeline_class for x in ['Inpaint', 'InPaint']):
                raise ValueError(
                    f"AspectAdjuster requires an INPAINT pipeline, got {pipeline_class}. "
                    f"Use StableDiffusionXLInpaintPipeline, NOT Img2ImgPipeline!"
                )
            # Also check it's SDXL, not SD 1.5
            if 'XL' not in pipeline_class and 'Xl' not in pipeline_class:
                self.logger.warning(
                    f"Pipeline {pipeline_class} may not be SDXL. "
                    f"For best results, use StableDiffusionXLInpaintPipeline."
                )
        
        # Load configuration
        aspect_config = self.config.resolution.get('aspect_adjustment', {})
        self.enabled = aspect_config.get('enabled', True)
        
        # Progressive outpainting config
        prog_config = self.config.resolution.get('progressive_outpainting', {})
        self.prog_enabled = prog_config.get('enabled', True)
        self.thresholds = prog_config.get('aspect_ratio_thresholds', {})
        self.max_supported = self.thresholds.get('max_supported', 8.0)
        
        # Outpaint settings
        outpaint_config = aspect_config.get('outpaint', {})
        # CRITICAL: High enough strength to generate content
        self.outpaint_strength = outpaint_config.get('strength', 0.95)  # Default to 0.95
        self.min_strength = outpaint_config.get('min_strength', 0.20)   # Minimum strength for final passes
        self.max_strength = outpaint_config.get('max_strength', 0.95)   # Maximum strength matches default
        self.outpaint_prompt_suffix = outpaint_config.get('prompt_suffix', 
            ', seamless expansion, extended scenery, natural continuation')
        self.base_mask_blur = outpaint_config.get('mask_blur', 32)
        self.base_steps = outpaint_config.get('steps', 60)
        
        # Adaptive settings from progressive config
        self.adaptive_blur = prog_config.get('adaptive_blur', {})
        self.adaptive_steps = prog_config.get('adaptive_steps', {})
        self.adaptive_guidance = prog_config.get('adaptive_guidance', {})
        
        # Expansion ratios - REDUCED for maximum context overlap
        expansion_config = prog_config.get('expansion_ratios', {})
        self.first_step_ratio = expansion_config.get('first_step', 1.4)    # Was 2.0
        self.middle_step_ratio = expansion_config.get('middle_steps', 1.25) # Was 1.5
        self.final_step_ratio = expansion_config.get('final_step', 1.15)    # Was 1.3
    
    def _analyze_edge_colors(self, image: Image.Image, edge: str, sample_width: int = 50) -> Dict:
        """
        Analyze colors at image edge for better continuation.
        NO ERROR TOLERANCE - data must be valid.
        """
        if image.mode != 'RGB':
            raise ValueError(f"Image must be RGB, got {image.mode}")
            
        img_array = np.array(image)
        h, w = img_array.shape[:2]
        
        # Validate dimensions
        if h < sample_width or w < sample_width:
            raise ValueError(
                f"Image too small ({w}x{h}) for {sample_width}px edge sampling"
            )
        
        if edge == 'left':
            sample = img_array[:, :sample_width]
        elif edge == 'right':
            sample = img_array[:, -sample_width:]
        elif edge == 'top':
            sample = img_array[:sample_width, :]
        elif edge == 'bottom':
            sample = img_array[-sample_width:, :]
        else:
            raise ValueError(f"Invalid edge: {edge}. Must be left/right/top/bottom")
        
        # Calculate dominant colors
        pixels = sample.reshape(-1, 3)
        mean_color = np.mean(pixels, axis=0)
        median_color = np.median(pixels, axis=0)
        
        # Calculate color variance (texture indicator)
        color_std = np.std(pixels, axis=0)
        
        return {
            'mean_rgb': mean_color.tolist(),
            'median_rgb': median_color.tolist(),
            'color_variance': float(np.mean(color_std)),
            'is_uniform': float(np.mean(color_std)) < 20,
            'sample_size': pixels.shape[0]
        }
    
    def adjust_aspect_ratio(self,
                           image_path: Path,
                           original_prompt: str,
                           target_aspect: float,
                           progressive_steps: Optional[List[Dict]] = None,
                           save_intermediates: bool = False) -> Path:
        """
        Main entry point for aspect adjustment.
        
        Args:
            image_path: Source image
            original_prompt: Generation prompt
            target_aspect: Target aspect ratio
            progressive_steps: Pre-calculated steps (optional)
            save_intermediates: Save each progressive step
            
        Returns:
            Path to adjusted image
            
        Raises:
            RuntimeError: On any failure (LOUD AND PROUD)
        """
        if not self.enabled:
            raise RuntimeError("Aspect adjustment is disabled in configuration")
        
        if not self.pipeline:
            raise RuntimeError(
                "No inpainting pipeline provided to AspectAdjuster. "
                "Cannot perform aspect adjustment without SDXL inpaint pipeline."
            )
        
        # Validate inputs
        if not image_path.exists():
            raise FileNotFoundError(f"Input image not found: {image_path}")
        
        if target_aspect <= 0:
            raise ValueError(f"Invalid target aspect ratio: {target_aspect}")
        
        # Load and check current image
        try:
            image = Image.open(image_path)
            if image.mode != 'RGB':
                image = image.convert('RGB')
        except Exception as e:
            raise RuntimeError(f"Failed to load image: {str(e)}") from e
        
        current_w, current_h = image.size
        current_aspect = current_w / current_h
        
        # Check if adjustment needed
        if abs(current_aspect - target_aspect) < 0.05:
            self.logger.info(f"Aspect ratio already correct: {current_aspect:.3f}")
            return image_path
        
        # Check if ratio is supported
        aspect_change = max(target_aspect / current_aspect, current_aspect / target_aspect)
        if aspect_change > self.max_supported:
            raise ValueError(
                f"Aspect ratio change {aspect_change:.1f}x exceeds maximum "
                f"supported ratio of {self.max_supported}x. "
                f"Current: {current_aspect:.3f}, Target: {target_aspect:.3f}"
            )
        
        # Use progressive strategy if needed
        if progressive_steps:
            return self._progressive_adjust(
                image_path, original_prompt, progressive_steps, save_intermediates
            )
        else:
            # Single-step for small changes
            return self._single_step_adjust(
                image, image_path, original_prompt, target_aspect
            )
    
    def _progressive_adjust(self,
                           image_path: Path,
                           prompt: str,
                           steps: List[Dict],
                           save_intermediates: bool) -> Path:
        """
        Perform progressive adjustment with multiple steps.
        NO FALLBACKS - fails completely on any error.
        """
        current_path = image_path
        total_steps = len(steps)
        
        self.logger.info(f"Starting progressive aspect adjustment: {total_steps} steps")
        
        for i, step in enumerate(steps):
            step_num = i + 1
            self.logger.log_stage(
                f"Progressive Step {step_num}/{total_steps}", 
                step['description']
            )
            
            try:
                # Enhance prompt for this step
                enhanced_prompt = self._enhance_prompt_progressive(
                    prompt, step, step_num, total_steps
                )
                
                # Perform multi-pass outpainting for natural extension
                num_passes = 3  # Base number of passes
                expansion_ratio = step.get('expansion_ratio', 1.5)

                # More passes for larger expansions
                if expansion_ratio > 2.0:
                    num_passes = 5  # Maximum passes for huge expansions
                elif expansion_ratio > 1.5:
                    num_passes = 4  # Extra pass for medium expansions

                self.logger.info(
                    f"Using {num_passes} passes for {expansion_ratio:.2f}x expansion"
                )

                current_path = self._execute_multi_pass_outpaint(
                    current_path, enhanced_prompt, step, num_passes
                )
                
                # Save intermediate if requested
                if save_intermediates:
                    self._save_intermediate(current_path, step_num, step['target_size'])
                    
            except torch.cuda.OutOfMemoryError as e:
                error_msg = (
                    f"CUDA OUT OF MEMORY at step {step_num}/{total_steps}. "
                    f"Attempted: {step['current_size']} → {step['target_size']}. "
                    f"Free up VRAM or reduce target resolution."
                )
                self.logger.error(error_msg)
                raise RuntimeError(error_msg) from e
            except Exception as e:
                error_msg = (
                    f"Progressive outpaint FAILED at step {step_num}/{total_steps}. "
                    f"Step: {step['description']}. Error: {type(e).__name__}: {str(e)}"
                )
                self.logger.error(error_msg)
                raise RuntimeError(error_msg) from e
        
        self.logger.info(f"Progressive adjustment complete: {total_steps} steps")
        return current_path
    
    def _execute_outpaint_step(self,
                              image_path: Path,
                              prompt: str,
                              step_info: Dict) -> Path:
        """
        Execute a single outpaint step with ZERO tolerance for errors.
        """
        # Load image
        image = Image.open(image_path)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        current_w, current_h = image.size
        target_w, target_h = step_info['target_size']
        
        # Validate expansion
        if target_w < current_w and target_h < current_h:
            raise ValueError(
                f"Invalid expansion: {current_w}x{current_h} → {target_w}x{target_h}. "
                f"Target must be larger in at least one dimension."
            )
        
        # Round to SDXL-compatible dimensions
        outpaint_w = ((target_w + 7) // 8) * 8
        outpaint_h = ((target_h + 7) // 8) * 8
        
        # Create canvas
        canvas = Image.new('RGB', (outpaint_w, outpaint_h), color='black')
        mask = Image.new('L', (outpaint_w, outpaint_h), color='white')
        
        # Calculate padding
        pad_left = (outpaint_w - current_w) // 2
        pad_top = (outpaint_h - current_h) // 2
        
        # Place image
        canvas.paste(image, (pad_left, pad_top))
        
        # Create mask (black = keep, white = generate)
        mask_draw = ImageDraw.Draw(mask)
        mask_draw.rectangle(
            [pad_left, pad_top, pad_left + current_w - 1, pad_top + current_h - 1],
            fill='black'
        )
        
        # Apply adaptive mask blur
        mask_blur = self._get_adaptive_blur(step_info)
        mask = mask.filter(ImageFilter.GaussianBlur(radius=mask_blur))
        
        # Intelligently fill empty areas with edge-extended colors
        # This provides coherent starting point for inpainting
        self.logger.info("Pre-filling canvas with edge-extended colors")

        canvas_array = np.array(canvas)
        img_array = np.array(image)

        # Analyze edges - FAIL LOUD on any error
        left_colors = self._analyze_edge_colors(image, 'left')
        right_colors = self._analyze_edge_colors(image, 'right')
        top_colors = self._analyze_edge_colors(image, 'top')
        bottom_colors = self._analyze_edge_colors(image, 'bottom')

        # Count pixels to fill
        pixels_to_fill = 0

        # Fill empty areas with gradient from nearest edge
        for y in range(canvas_array.shape[0]):
            for x in range(canvas_array.shape[1]):
                # Skip if pixel already has content
                if not np.all(canvas_array[y, x] == 0):
                    continue
                
                pixels_to_fill += 1
                
                # Calculate distance to original image
                dist_left = max(0, pad_left - x)
                dist_right = max(0, x - (pad_left + current_w - 1))
                dist_top = max(0, pad_top - y)
                dist_bottom = max(0, y - (pad_top + current_h - 1))
                
                # Determine which edge is closest
                if dist_left > 0 and dist_left >= max(dist_right, dist_top, dist_bottom):
                    # Use left edge colors with slight variation
                    base_color = np.array(left_colors['median_rgb'])
                    variation = np.random.normal(0, left_colors['color_variance'] * 0.2, 3)
                elif dist_right > 0 and dist_right >= max(dist_left, dist_top, dist_bottom):
                    # Use right edge colors
                    base_color = np.array(right_colors['median_rgb'])
                    variation = np.random.normal(0, right_colors['color_variance'] * 0.2, 3)
                elif dist_top > 0 and dist_top >= max(dist_left, dist_right, dist_bottom):
                    # Use top edge colors
                    base_color = np.array(top_colors['median_rgb'])
                    variation = np.random.normal(0, top_colors['color_variance'] * 0.2, 3)
                else:  # dist_bottom > 0
                    # Use bottom edge colors
                    base_color = np.array(bottom_colors['median_rgb'])
                    variation = np.random.normal(0, bottom_colors['color_variance'] * 0.2, 3)
                
                # Apply color with variation
                canvas_array[y, x] = np.clip(base_color + variation, 0, 255).astype(np.uint8)

        canvas = Image.fromarray(canvas_array)
        self.logger.info(f"Pre-filled {pixels_to_fill} pixels with edge colors")
        
        # Get adaptive parameters
        num_steps = self._get_adaptive_steps(step_info)
        guidance = self._get_adaptive_guidance(step_info)
        
        self.logger.info(
            f"Outpainting {current_w}x{current_h} → {outpaint_w}x{outpaint_h} "
            f"(blur: {mask_blur}, steps: {num_steps}, guidance: {guidance})"
        )
        
        # Run pipeline - NO ERROR CATCHING
        result = self.pipeline(
            prompt=prompt,
            image=canvas,
            mask_image=mask,
            strength=self.outpaint_strength,
            num_inference_steps=num_steps,
            guidance_scale=guidance,
            width=outpaint_w,
            height=outpaint_h
        ).images[0]
        
        # Validate result
        if result is None:
            raise RuntimeError("Pipeline returned None - generation failed")
        
        if result.size != (outpaint_w, outpaint_h):
            raise RuntimeError(
                f"Pipeline returned wrong size! "
                f"Expected {outpaint_w}x{outpaint_h}, got {result.size}"
            )
        
        # Crop to exact target
        if (outpaint_w, outpaint_h) != (target_w, target_h):
            left = (outpaint_w - target_w) // 2
            top = (outpaint_h - target_h) // 2
            result = result.crop((left, top, left + target_w, top + target_h))
        
        # Save result
        temp_dir = self.resolver.get_temp_dir() / 'ai-wallpaper' / 'progressive'
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        filename = f"prog_{current_w}x{current_h}_to_{target_w}x{target_h}_{timestamp}.png"
        save_path = temp_dir / filename
        
        # Use lossless save to ensure no quality loss
        from ..utils import save_lossless_png
        save_lossless_png(result, save_path)
        
        # Track boundaries for artifact detection - TRACK ACTUAL SEAMS
        if hasattr(self, 'model_metadata') and self.model_metadata:
            current_w, current_h = step_info['current_size']
            target_w, target_h = step_info['target_size']
            
            # CRITICAL: Track where new content meets old content
            # These are the ACTUAL seam locations that need blending
            if target_w > current_w:  # Horizontal expansion
                # Calculate padding (where original content is placed)
                pad_left = (target_w - current_w) // 2
                pad_right = pad_left + current_w
                
                # The seams are where old content meets new content
                # Left seam: between new content (0 to pad_left) and old content
                # Right seam: between old content and new content (pad_right to target_w)
                
                # Store exact seam positions with context
                seam_info = {
                    'left_seam': pad_left,      # Where left new content meets old
                    'right_seam': pad_right,     # Where old content meets right new
                    'expansion_size': target_w - current_w,
                    'step': step_info.get('step_type', 'unknown')
                }
                
                # Add individual seam positions for detection
                self.model_metadata['progressive_boundaries'].extend([pad_left, pad_right])
                self.model_metadata['seam_details'] = self.model_metadata.get('seam_details', [])
                self.model_metadata['seam_details'].append(seam_info)
                self.model_metadata['used_progressive'] = True
                
            elif target_h > current_h:  # Vertical expansion
                # Similar logic for vertical
                pad_top = (target_h - current_h) // 2
                pad_bottom = pad_top + current_h
                
                seam_info = {
                    'top_seam': pad_top,
                    'bottom_seam': pad_bottom,
                    'expansion_size': target_h - current_h,
                    'step': step_info.get('step_type', 'unknown')
                }
                
                # For vertical, track y-coordinates instead
                self.model_metadata['progressive_boundaries_vertical'] = self.model_metadata.get('progressive_boundaries_vertical', [])
                self.model_metadata['progressive_boundaries_vertical'].extend([pad_top, pad_bottom])
                self.model_metadata['seam_details'] = self.model_metadata.get('seam_details', [])
                self.model_metadata['seam_details'].append(seam_info)
                self.model_metadata['used_progressive'] = True
        
        return save_path
    
    def _execute_multi_pass_outpaint(self,
                                     image_path: Path,
                                     prompt: str,
                                     step_info: Dict,
                                     num_passes: int = 3) -> Path:
        """
        Execute multiple light outpainting passes for natural extension.
        Each pass uses progressively lighter denoising.
        NO ERROR TOLERANCE - FAIL LOUD
        """
        current_path = image_path
        base_strength = self.outpaint_strength
        
        # Validate input
        if num_passes < 1:
            raise ValueError(f"num_passes must be >= 1, got {num_passes}")
        
        if not image_path.exists():
            raise FileNotFoundError(f"Input image not found: {image_path}")
        
        for pass_num in range(num_passes):
            # Progressive strength reduction
            # Pass 1: base_strength (e.g., 0.35)
            # Pass 2: base_strength * 0.8 (e.g., 0.28)
            # Pass 3: base_strength * 0.6 (e.g., 0.21)
            # Pass 4: base_strength * 0.4 (e.g., 0.14)
            strength_multiplier = 1.0 - (pass_num * 0.2)
            current_strength = max(self.min_strength, base_strength * strength_multiplier)
            
            self.logger.info(
                f"Multi-pass outpaint {pass_num + 1}/{num_passes}, "
                f"strength: {current_strength:.3f}"
            )
            
            # Temporarily override strength - NO TRY/EXCEPT
            original_strength = self.outpaint_strength
            self.outpaint_strength = current_strength
            
            # Execute pass with enhanced prompt for later passes
            pass_prompt = prompt
            if pass_num > 0:
                pass_prompt += ", maintaining perfect continuity with existing content"
            
            # Execute the outpaint step - will fail loud if error
            current_path = self._execute_outpaint_step(
                current_path, pass_prompt, step_info
            )
            
            # Restore original strength
            self.outpaint_strength = original_strength
            
            # Validate result
            if not current_path.exists():
                raise RuntimeError(
                    f"Pass {pass_num + 1} failed to produce output at {current_path}"
                )
        
        return current_path
    
    def _enhance_prompt_progressive(self, 
                                   base_prompt: str,
                                   step_info: Dict,
                                   current_step: int,
                                   total_steps: int) -> str:
        """Enhance prompt with STRONG emphasis on seamless continuation"""
        direction = step_info.get('direction', 'horizontal')
        ratio = step_info.get('expansion_ratio', 1.5)
        
        # Core continuation phrases - CRITICAL for seamless extension
        continuation_emphasis = [
            "seamlessly extending the existing image",
            "naturally continuing all elements without interruption",
            "maintaining identical style, lighting, and perspective",
            "perfectly matched colors and textures",
            "coherent continuation of all visible elements"
        ]
        
        # Direction-specific phrases
        if direction == 'horizontal':
            spatial = "extending the scene naturally to the sides"
            specific = "continuing all horizontal elements smoothly"
        else:
            spatial = "extending the scene naturally vertically"  
            specific = "continuing all vertical elements smoothly"
        
        # Compose final prompt with HEAVY emphasis on continuation
        enhanced = (
            f"{base_prompt}, "
            f"{continuation_emphasis[0]}, "
            f"{spatial}, "
            f"{specific}, "
            f"{continuation_emphasis[1]}, "
            f"{continuation_emphasis[2]}"
        )
        
        # Add extra emphasis for larger expansions
        if ratio > 1.7:
            enhanced += f", {continuation_emphasis[3]}, {continuation_emphasis[4]}"
        
        # Critical: Add negative prompting guidance
        # Note: SDXL inpaint doesn't support negative prompts directly,
        # but we can emphasize what we want in the positive prompt
        enhanced += ", avoiding any discontinuities or style changes"
        
        return enhanced
    
    def _get_adaptive_blur(self, step_info: Dict) -> int:
        """Calculate MASSIVE mask blur for seamless transitions"""
        ratio = step_info.get('expansion_ratio', 1.5)
        base = self.adaptive_blur.get('base_radius', self.base_mask_blur)
        
        # Calculate new content dimensions
        current_size = step_info.get('current_size', [0, 0])
        target_size = step_info.get('target_size', [0, 0])
        
        new_width = target_size[0] - current_size[0]
        new_height = target_size[1] - current_size[1]
        max_new_dimension = max(new_width, new_height)
        
        if max_new_dimension > 0:
            # CRITICAL: Use 40% of new content dimension as blur radius
            # This creates VERY wide transition zones
            percentage_blur = int(max_new_dimension * 0.40)
            
            # Minimum blur should be substantial
            min_blur = max(100, int(base * 3.0))  # At least 100 pixels
            
            calculated_blur = max(min_blur, percentage_blur)
            
            # Cap at reasonable maximum to prevent memory issues
            final_blur = min(calculated_blur, 400)  # Increased cap
            
            self.logger.info(
                f"Massive blur for seamless transition: {final_blur}px "
                f"(40% of {max_new_dimension}px new content)"
            )
            
            return final_blur
        
        # Fallback - should never reach here
        raise ValueError(
            f"Invalid step_info: no dimension increase found. "
            f"Current: {current_size}, Target: {target_size}"
        )
    
    def _get_adaptive_steps(self, step_info: Dict) -> int:
        """Calculate adaptive inference steps"""
        ratio = step_info.get('expansion_ratio', 1.5)
        
        if ratio >= 1.8:
            return self.adaptive_steps.get('large_expansion', 80)
        elif ratio >= 1.5:
            return self.adaptive_steps.get('medium_expansion', 70)
        else:
            return self.adaptive_steps.get('base', self.base_steps)
    
    def _get_adaptive_guidance(self, step_info: Dict) -> float:
        """Calculate adaptive guidance scale"""
        ratio = step_info.get('expansion_ratio', 1.5)
        
        if ratio >= 2.0:
            return self.adaptive_guidance.get('large_expansion', 8.5)
        elif ratio >= 1.5:
            return self.adaptive_guidance.get('medium_expansion', 7.5)
        else:
            return self.adaptive_guidance.get('base', 7.0)
    
    def _single_step_adjust(self, 
                           image: Image.Image,
                           image_path: Path,
                           prompt: str,
                           target_aspect: float) -> Path:
        """Single-step adjustment for moderate aspect changes"""
        current_w, current_h = image.size
        
        # Calculate target dimensions
        if target_aspect > current_w / current_h:
            target_w = int(current_h * target_aspect)
            target_h = current_h
        else:
            target_w = current_w
            target_h = int(current_w / target_aspect)
        
        step_info = {
            'current_size': (current_w, current_h),
            'target_size': (target_w, target_h),
            'expansion_ratio': max(target_w/current_w, target_h/current_h),
            'direction': 'horizontal' if target_w > current_w else 'vertical',
            'step_type': 'single',
            'description': f"Single-step: {current_w}x{current_h} → {target_w}x{target_h}"
        }
        
        enhanced_prompt = f"{prompt}, seamlessly extending to fill the frame"
        
        return self._execute_outpaint_step(image_path, enhanced_prompt, step_info)
    
    def _save_intermediate(self, image_path: Path, step_num: int, size: Tuple[int, int]):
        """Save intermediate result"""
        intermediate_dir = self.resolver.get_temp_dir() / 'ai-wallpaper' / 'intermediates'
        intermediate_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"step_{step_num}_{size[0]}x{size[1]}_{timestamp}.png"
        
        import shutil
        shutil.copy2(image_path, intermediate_dir / filename)
        self.logger.info(f"Saved intermediate: {filename}")