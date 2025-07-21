#!/usr/bin/env python3
"""
SDXL Model Implementation
Generates images using Stable Diffusion XL with LoRA support
"""

import os
import gc
import sys
import time
import random
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List
from datetime import datetime

import torch
from diffusers import (
    StableDiffusionXLPipeline,
    StableDiffusionXLImg2ImgPipeline,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    DDIMScheduler,
    HeunDiscreteScheduler,
    KDPM2DiscreteScheduler
)
from PIL import Image

from .base_model import BaseImageModel
from ..utils import save_lossless_png
from ..core import get_logger, get_config
from ..core.exceptions import ModelNotFoundError, ModelLoadError, GenerationError, UpscalerError
from ..core.path_resolver import get_resolver
from .model_resolver import get_model_resolver
from ..utils import get_resource_manager
from ..processing.downsampler import HighQualityDownsampler
from ..processing.aspect_adjuster import AspectAdjuster
from ..processing.smart_refiner import SmartQualityRefiner

class SdxlModel(BaseImageModel):
    """SDXL implementation with LoRA support and 2x upscaling"""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize SDXL model
        
        Args:
            config: Model configuration from models.yaml
        """
        super().__init__(config)
        self.pipe = None
        self.refiner_pipe = None
        self.loaded_loras = {}
        self.available_loras = self._scan_available_loras()
        self.resource_manager = get_resource_manager()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    def initialize(self) -> bool:
        """Initialize SDXL model and verify requirements
        
        Returns:
            True if initialization successful
        """
        try:
            self.logger.info("Initializing SDXL Ultimate Photorealistic Pipeline...")
            
            # Check resources
            self.resource_manager.check_critical_resources(
                self.model_name,
                self.get_resource_requirements()
            )
            
            # Clear resources for this model
            self.resource_manager.prepare_for_model(self.model_name)
            
            # Check which model variant to use
            model_variant = self.config.get('model_variant', 'base').lower()
            model_resolver = get_model_resolver()
            
            if model_variant == 'juggernaut':
                # Try to load Juggernaut XL
                juggernaut_config = self.config.get('juggernaut_model', {})
                checkpoint_hints = juggernaut_config.get('checkpoint_hints', [])
                juggernaut_path = model_resolver.find_checkpoint(checkpoint_hints)
                
                if juggernaut_path:
                    self.logger.info("Loading Juggernaut XL v9...")
                    self.logger.warning("Note: Juggernaut XL has modified architecture - standard SDXL LoRAs may not be compatible")
                    self.pipe = self._load_from_single_file(str(juggernaut_path))
                else:
                    self.logger.warning("Juggernaut XL not found, falling back to base SDXL")
                    model_variant = 'base'
            
            if model_variant == 'base':
                # Load standard SDXL
                base_config = self.config.get('base_model', {})
                model_path = base_config.get('model_path', 'stabilityai/stable-diffusion-xl-base-1.0')
                checkpoint_hints = base_config.get('checkpoint_hints', [])
                
                # Try to find checkpoint file first (most reliable)
                checkpoint_path = model_resolver.find_checkpoint(checkpoint_hints)
                
                if checkpoint_path:
                    self.logger.info(f"Loading base SDXL from checkpoint: {checkpoint_path}")
                    self.pipe = self._load_from_single_file(str(checkpoint_path))
                else:
                    # Fallback to HuggingFace
                    self.logger.info(f"Loading SDXL pipeline from HuggingFace: {model_path}")
                    self.pipe = StableDiffusionXLPipeline.from_pretrained(
                        model_path,
                        torch_dtype=torch.float16,
                        use_safetensors=True
                    )
            
            # Move to GPU
            self.pipe = self.pipe.to(self.device)
            
            # Enable memory optimizations
            self.pipe.enable_vae_slicing()
            self.pipe.enable_vae_tiling()
            
            # Enable xformers if available
            try:
                self.pipe.enable_xformers_memory_efficient_attention()
                self.logger.info("xFormers enabled for memory efficiency")
            except ImportError as e:
                self.logger.info(f"xFormers not available: {e}. Using standard attention")
            except Exception as e:
                self.logger.warning(f"Failed to enable xFormers: {e}. Using standard attention")
                
            # Load SDXL Refiner model for ensemble of expert denoisers
            self._load_refiner_model()
                
            # Verify Real-ESRGAN is available
            self._find_realesrgan()
            
            # Validate pipeline stages before marking as initialized
            self.validate_pipeline_stages()
            
            # Only register with resource manager after all initialization succeeds
            # This prevents registration leaks if any initialization step fails
            self._initialized = True
            self.resource_manager.register_model(self.model_name, self)
            
            self.logger.info("SDXL model initialized successfully")
            self.logger.log_vram("After initialization")
            
            return True
            
        except Exception as e:
            raise ModelLoadError(self.name, e)
            
    def generate(self, prompt: str, seed: Optional[int] = None, **kwargs) -> Dict[str, Any]:
        """
        Generate image with new pipeline order:
        1. Generate at optimal size
        1.5. Progressive aspect adjustment (NEW!)
        2. Refine entire image
        2.5. Tiled ultra-refinement (optional)
        3. Upscale resolution only
        4. Final adjustments
        """
        self.ensure_initialized()
        
        # Clean prompt
        prompt = self.validate_prompt(prompt)
        
        # Get generation parameters
        params = self.get_generation_params(**kwargs)
        params['prompt'] = prompt
        params['seed'] = seed
        
        # Check disk space before starting generation
        self.check_disk_space_for_generation(no_upscale=params.get('no_upscale', False))
        
        # Use provided seed or generate random
        if seed is None:
            seed = random.randint(0, 2**32 - 1)
            
        # Log generation start
        self.log_generation_start(prompt, {**params, 'seed': seed})
        
        # Initialize generation metadata for artifact tracking
        self.generation_metadata = {
            'progressive_boundaries': [],  # X positions where expansions happened
            'tile_boundaries': [],         # (x,y) positions of tiles  
            'used_progressive': False,     # Flag if progressive was used
            'used_tiled': False,          # Flag if tiling was used
        }
        
        # Track timing
        start_time = time.time()
        
        # Track temp files for cleanup
        temp_files = []
        temp_dirs = []
        
        try:
            # Select and load LoRA if enabled
            lora_info = None
            if self.config['lora'].get('enabled') and self.config['lora'].get('auto_select_by_theme'):
                # Note: Theme information would need to be passed in kwargs
                theme = kwargs.get('theme', {})
                lora_info = self._select_and_load_lora(theme)
            
            # ============ STAGE 1: Initial Generation ============
            self.logger.log_stage("Stage 1", "Text-to-image generation")
            self.logger.info(f"Generating at optimal size: {params.get('generation_size', 'default')}")
            
            stage1_result = self._generate_stage1(prompt, seed, params, lora_info, temp_files)
            
            if self._should_save_stages(params):
                stage1_result['saved_path'] = self._save_intermediate(
                    stage1_result['image'], 'stage1_generated', prompt
                )
            
            # ============ STAGE 1.5: Progressive Aspect Adjustment ============
            # This is the KEY CHANGE - aspect adjustment BEFORE refinement!
            current_image_path = stage1_result['image_path']
            current_image = stage1_result['image']
            stage1_5_result = None
            
            # Check if aspect adjustment is needed
            if self._needs_aspect_adjustment(params):
                self.logger.log_stage("Stage 1.5", "Progressive aspect adjustment")
                
                # Calculate progressive strategy
                current_size = current_image.size
                target_aspect = params.get('target_aspect')
                
                # Check if SWPO is enabled (CLI override takes precedence)
                swpo_config = self.config.get('resolution', {}).get('progressive_outpainting', {}).get('sliding_window', {})
                # If swpo is None (not specified in CLI), use config value
                cli_swpo = params.get('swpo')
                use_swpo = cli_swpo if cli_swpo is not None else swpo_config.get('enabled', True)
                
                if use_swpo:
                    # Calculate sliding window steps
                    self.logger.info("Using Sliding Window Progressive Outpainting (SWPO)")
                    progressive_steps = self.resolution_manager.calculate_sliding_window_strategy(
                        current_size=(current_image.width, current_image.height),
                        target_size=(sdxl_params['width'], sdxl_params['height']),
                        window_size=params.get('window_size', swpo_config.get('window_size', 200)),
                        overlap_ratio=params.get('overlap_ratio', swpo_config.get('overlap_ratio', 0.8))
                    )
                else:
                    # Fall back to original progressive strategy
                    target_aspect = sdxl_params['width'] / sdxl_params['height']
                    progressive_steps = self.resolution_manager.calculate_progressive_outpaint_strategy(
                        current_size=(current_image.width, current_image.height),
                        target_aspect=target_aspect
                    )
                
                if progressive_steps:
                    self.logger.info(f"Aspect adjustment: {len(progressive_steps)} progressive steps")
                    
                    # Ensure we have inpaint pipeline
                    if not hasattr(self, 'inpaint_pipe') or self.inpaint_pipe is None:
                        self._create_inpaint_pipeline()
                    
                    # Create adjuster and perform adjustment
                    adjuster = AspectAdjuster(pipeline=self.inpaint_pipe)
                    
                    # Pass metadata reference for boundary tracking
                    adjuster.model_metadata = self.generation_metadata
                    
                    adjusted_path = adjuster.adjust_aspect_ratio(
                        image_path=current_image_path,
                        original_prompt=prompt,
                        target_aspect=target_aspect,
                        progressive_steps=progressive_steps,
                        save_intermediates=self._should_save_stages(params)
                    )
                    
                    # Update current image
                    current_image = Image.open(adjusted_path)
                    current_image_path = adjusted_path
                    temp_files.append(adjusted_path)
                    
                    stage1_5_result = {
                        'image': current_image,
                        'image_path': current_image_path,
                        'size': current_image.size,
                        'steps': len(progressive_steps)
                    }
                    
                    if self._should_save_stages(params):
                        stage1_5_result['saved_path'] = self._save_intermediate(
                            current_image, 'stage1_5_aspect_adjusted', prompt
                        )
                else:
                    self.logger.info("No aspect adjustment needed")
            
            # ============ STAGE 2: Smart Quality Refinement ============
            stage2_result = None
            initial_refinement = self.config['pipeline'].get('stages', {}).get('initial_refinement', {})

            if initial_refinement.get('enabled', True) and self.refiner_pipe:
                self.logger.log_stage("Stage 2", "Smart quality refinement")
                self.logger.info(f"Stage 2 WILL RUN - Refiner loaded: {self.refiner_pipe is not None}")
                
                # Use smart refiner if available and config has multi-pass settings
                if self.smart_refiner and initial_refinement.get('multi_pass_enabled', False):
                    self.logger.info("Using smart multi-pass refinement for maximum detail preservation")
                    
                    refinement_results = self.smart_refiner.refine_smart(
                        image_path=current_image_path,
                        prompt=prompt,
                        seed=seed,
                        params=params,
                        temp_files=temp_files
                    )
                    
                    # Update current image
                    current_image = refinement_results['image']
                    current_image_path = refinement_results['image_path']
                    
                    stage2_result = {
                        'image': current_image,
                        'image_path': current_image_path,
                        'size': refinement_results['size'],
                        'method': refinement_results['method'],
                        'passes': refinement_results['passes']
                    }
                    
                    self.logger.info(f"Smart refinement complete: {refinement_results['passes']} passes")
                else:
                    # Continue with existing VRAM strategy code for backward compatibility
                    # NEW: Intelligent refinement strategy selection
                    from ..core.vram_calculator import VRAMCalculator
                    vram_calc = VRAMCalculator()
                    
                    w, h = current_image.size
                    strategy = vram_calc.determine_refinement_strategy(w, h)
                    
                    self.logger.info(
                        f"Refinement strategy for {w}x{h}: {strategy['strategy']} "
                        f"(Required: {strategy['vram_required_mb']:.0f}MB, "
                        f"Available: {strategy['vram_available_mb']:.0f}MB)"
                    )
                    
                    # Determine refinement strength based on aspect adjustment
                    # More aggressive refinement after extreme aspect changes
                    had_extreme_aspect = (stage1_5_result and 
                                        stage1_5_result.get('steps', 0) >= 2)
                    refinement_strength = 0.5 if had_extreme_aspect else 0.3
                    
                    if had_extreme_aspect:
                        self.logger.info(
                            f"Using higher refinement strength ({refinement_strength}) "
                            f"after {stage1_5_result.get('steps', 0)}-step aspect adjustment"
                        )
                    
                    # Execute appropriate strategy
                    if strategy['strategy'] == 'full':
                        # Standard full refinement
                        stage2_result = self._refine_stage2_full(
                            current_image,
                            current_image_path,
                            prompt,
                            seed,
                            params,
                            temp_files,
                            strength_override=refinement_strength
                        )
                    elif strategy['strategy'] == 'tiled':
                        # Automatic tiled refinement
                        self.logger.info(f"Using tiled refinement: {strategy['details']['message']}")
                        
                        from ..processing.tiled_refiner import TiledRefiner
                        refiner = TiledRefiner(
                            pipeline=self.refiner_pipe,
                            vram_calculator=vram_calc
                        )
                        
                        # Override tile size from strategy
                        refiner.tile_size = strategy['details']['tile_size']
                        refiner.overlap = strategy['details']['overlap']
                        
                        refined_path = refiner.refine_tiled(
                            image_path=current_image_path,
                            prompt=prompt,
                            base_strength=refinement_strength,
                            base_steps=50,
                            seed=seed
                        )
                        
                        refined_image = Image.open(refined_path)
                        temp_files.append(refined_path)
                        
                        stage2_result = {
                            'image': refined_image,
                            'image_path': refined_path,
                            'size': refined_image.size,
                            'method': 'tiled',
                            'tile_size': strategy['details']['tile_size']
                        }
                    else:  # cpu_offload
                        # Last resort - CPU offload
                        self.logger.warning(f"Using CPU offload: {strategy['details']['warning']}")
                        
                        from ..processing.cpu_offload_refiner import CPUOffloadRefiner
                        cpu_refiner = CPUOffloadRefiner(pipeline=self.refiner_pipe)
                        
                        refined_path = cpu_refiner.refine_with_offload(
                            image_path=current_image_path,
                            prompt=prompt,
                            strength=refinement_strength,
                            steps=50,
                            seed=seed
                        )
                        
                        refined_image = Image.open(refined_path)
                        temp_files.append(refined_path)
                        
                        stage2_result = {
                            'image': refined_image,
                            'image_path': refined_path,
                            'size': refined_image.size,
                            'method': 'cpu_offload',
                            'warning': strategy['details']['warning']
                        }
                    
                    # Update current image
                    current_image = stage2_result['image']
                    current_image_path = stage2_result['image_path']
                
                if self._should_save_stages(params):
                    stage2_result['saved_path'] = self._save_intermediate(
                        current_image, f"stage2_refined_{stage2_result.get('method', 'full')}", prompt
                    )
            else:
                # FAIL LOUD - THIS SHOULD NOT HAPPEN
                raise RuntimeError(
                    f"Stage 2 CANNOT RUN! Enabled: {initial_refinement.get('enabled', True)}, "
                    f"Refiner exists: {self.refiner_pipe is not None}. "
                    f"This is a CRITICAL ERROR - refinement is required for quality!"
                )
            
            # ============ STAGE 2.5: Tiled Ultra-Refinement (Optional) ============
            stage2_5_result = None
            tiled_refinement = self.config.get('resolution', {}).get('tiled_refinement', {})
            
            if (tiled_refinement.get('enabled', False) and 
                params.get('quality_mode') == 'ultimate' and
                current_image.size[0] * current_image.size[1] > 1024 * 1024):
                
                self.logger.log_stage("Stage 2.5", "Tiled ultra-refinement")
                
                stage2_5_result = self._tiled_ultra_refine(
                    current_image,
                    current_image_path,
                    prompt,
                    params,
                    temp_files
                )
                
                if stage2_5_result and not stage2_5_result.get('skipped'):
                    current_image = stage2_5_result['image']
                    current_image_path = stage2_5_result['image_path']
                    
                    if self._should_save_stages(params):
                        stage2_5_result['saved_path'] = self._save_intermediate(
                            current_image, 'stage2_5_tiled_refined', prompt
                        )
            
            # ============ STAGE 3: Resolution Upscaling Only ============
            stage3_result = None
            
            if not params.get('no_upscale', False):
                # Calculate upscale-only strategy (no aspect adjustment)
                current_size = current_image.size
                target_resolution = params.get('target_resolution')
                
                if target_resolution and (
                    target_resolution[0] > current_size[0] or 
                    target_resolution[1] > current_size[1]
                ):
                    self.logger.log_stage("Stage 3", "Resolution upscaling")
                    
                    upscale_strategy = self.resolution_manager.calculate_upscale_strategy(
                        current_size,
                        target_resolution,
                        current_size[0] / current_size[1],  # Current aspect
                        target_resolution[0] / target_resolution[1]  # Target aspect (should match)
                    )
                    
                    if upscale_strategy:
                        stage3_result = self._upscale_stage3_simple(
                            current_image_path,
                            temp_dirs,
                            upscale_strategy
                        )
                        
                        current_image_path = stage3_result['image_path']
                        
                        if self._should_save_stages(params):
                            with Image.open(current_image_path) as img:
                                stage3_result['saved_path'] = self._save_intermediate(
                                    img.copy(), 'stage3_upscaled', prompt
                                )
                else:
                    self.logger.info("No upscaling needed")
            
            # ============ STAGE 4: Final Adjustments ============
            final_path = current_image_path
            
            if params.get('target_resolution'):
                # Ensure exact target size
                with Image.open(final_path) as img:
                    if img.size != tuple(params['target_resolution']):
                        self.logger.log_stage("Stage 4", "Final size adjustment")
                        final_path = self._ensure_exact_size(
                            final_path,
                            params['target_resolution']
                        )
                        temp_files.append(final_path)
            
            # Ensure final image is in the proper output directory
            config = get_config()
            if not str(final_path).startswith(str(config.paths.get('images_dir', ''))):
                # Image is still in temp directory, move it to output directory
                resolver = get_resolver()
                output_dir = Path(config.paths.get('images_dir', str(resolver.get_data_dir() / 'wallpapers')))
                output_dir.mkdir(exist_ok=True)
                
                # Create appropriate filename
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                if params.get('target_resolution'):
                    res_str = f"{params['target_resolution'][0]}x{params['target_resolution'][1]}"
                else:
                    res_str = "generated"
                filename = f"sdxl_{res_str}_{timestamp}.png"
                
                new_final_path = output_dir / filename
                
                # Copy the image to the output directory
                import shutil
                shutil.copy2(final_path, new_final_path)
                self.logger.info(f"Moved final image to: {new_final_path}")
                final_path = new_final_path
            
            # Unload LoRA if loaded
            if lora_info:
                self._unload_lora()
                
            # Load final image for metadata
            final_image = Image.open(final_path)
            
            # Save metadata
            self.save_metadata(final_path, {
                'prompt': prompt,
                'seed': seed,
                'model': self.model_name,
                'pipeline_version': 'v2.0-progressive',
                'generation_size': params.get('generation_size'),
                'final_size': final_image.size,
                'stages': {
                    'stage1': stage1_result.get('size') if stage1_result else None,
                    'stage1_5': stage1_5_result.get('size') if stage1_5_result else None,
                    'stage2': 'refined' if stage2_result and not stage2_result.get('skipped') else 'skipped',
                    'stage2_5': 'tiled' if stage2_5_result and not stage2_5_result.get('skipped') else 'skipped',
                    'stage3': 'upscaled' if stage3_result else 'skipped',
                },
                'parameters': params
            })
            
            # Build comprehensive result
            duration = time.time() - start_time
            result = {
                'success': True,
                'image_path': final_path,
                'seed': seed,
                'size': final_image.size,
                'stages': {
                    'generation': stage1_result,
                    'aspect_adjustment': stage1_5_result,
                    'refinement': stage2_result,
                    'tiled_refinement': stage2_5_result,
                    'upscaling': stage3_result
                },
                'metadata': {
                    'prompt': prompt,
                    'seed': seed,
                    'model': self.model_name,
                    'generation_time': duration
                }
            }
            
            # Log completion
            self.log_generation_complete(Path(final_path), duration)
            
            return result
            
        except torch.cuda.OutOfMemoryError as e:
            self.logger.error(f"CUDA OUT OF MEMORY: {str(e)}")
            raise
        except Exception as e:
            self.logger.error(f"Generation FAILED: {type(e).__name__}: {str(e)}")
            raise
        finally:
            # Clean up all temp files created during generation
            for temp_file in temp_files:
                try:
                    if temp_file.exists():
                        temp_file.unlink()
                        self.logger.debug(f"Cleaned up temp file: {temp_file}")
                except Exception as cleanup_e:
                    self.logger.warning(f"Failed to clean up {temp_file}: {cleanup_e}")
                    
            for temp_dir in temp_dirs:
                try:
                    if temp_dir.exists():
                        import shutil
                        shutil.rmtree(temp_dir)
                        self.logger.debug(f"Cleaned up temp dir: {temp_dir}")
                except Exception as cleanup_e:
                    self.logger.warning(f"Failed to clean up {temp_dir}: {cleanup_e}")
            
    def _generate_stage1(
        self, 
        prompt: str, 
        seed: int, 
        params: Dict[str, Any],
        lora_info: Optional[Dict[str, Any]],
        temp_files: List[Path]
    ) -> Dict[str, Any]:
        """Stage 1: Generate base image
        
        Args:
            prompt: Text prompt
            seed: Random seed
            params: Generation parameters
            lora_info: LoRA information if loaded
            
        Returns:
            Stage results
        """
        self.logger.log_stage("Stage 1", "SDXL generation")
        
        # Set up generator
        generator = torch.Generator(device=self.device).manual_seed(seed)
        
        # Get dimensions from generation parameters or use default
        if 'generation_size' in params:
            width, height = params['generation_size']
            self.logger.info(f"Using calculated generation size: {width}x{height}")
        else:
            # Fallback to default 16:9 for SDXL
            width, height = 1344, 768
            self.logger.info(f"Using default generation size: {width}x{height}")
        
        # Select scheduler
        scheduler_class = self._get_scheduler_class(params.get('scheduler'))
        
        # Get scheduler kwargs from config
        scheduler_kwargs = self.config['generation'].get('scheduler_kwargs', {})
        
        # Initialize scheduler with kwargs
        self.pipe.scheduler = scheduler_class.from_config(
            self.pipe.scheduler.config,
            **scheduler_kwargs
        )
        
        self.logger.info(f"Using scheduler: {scheduler_class.__name__}")
        if scheduler_kwargs:
            self.logger.debug(f"Scheduler kwargs: {scheduler_kwargs}")
        
        # Enhance prompt for photorealism
        photo_prefixes = [
            "RAW photo, ",
            "photograph, ",
            "DSLR photo, ",
            "professional photography, "
        ]
        photo_suffixes = [
            ", 8k uhd, dslr, high quality, film grain, Fujifilm XT3",
            ", cinematic, professional, 4k, highly detailed",
            ", shot on Canon EOS R5, 85mm lens, f/1.4, ISO 100",
            ", photorealistic, hyperrealistic, professional lighting"
        ]
        
        # Add random photorealistic modifiers
        enhanced_prompt = f"{random.choice(photo_prefixes)}{prompt}{random.choice(photo_suffixes)}"
        
        # Prepare parameters
        sdxl_params = {
            'prompt': enhanced_prompt,
            'height': height,
            'width': width,
            'generator': generator,
            'num_inference_steps': params.get('steps', 80),  # Increased for quality
            'guidance_scale': params.get('guidance_scale', 8.0),  # Optimal for photorealism
        }
        
        # Enhanced negative prompt targeting watercolor effects
        sdxl_params['negative_prompt'] = (
            "watercolor, painting, illustration, drawing, sketch, cartoon, anime, "
            "artistic, painted, brush strokes, canvas texture, paper texture, "
            "impressionism, expressionism, abstract, stylized, "
            "oil painting, acrylic, pastel, charcoal, "
            "(worst quality:1.4), (bad quality:1.4), (poor quality:1.4), "
            "blurry, soft focus, out of focus, bokeh, "
            "low resolution, low detail, pixelated, aliasing, "
            "jpeg artifacts, compression artifacts, "
            "oversaturated, undersaturated, overexposed, underexposed, "
            "grainy, noisy, film grain, sensor noise, "
            "bad anatomy, deformed, mutated, disfigured, "
            "extra limbs, missing limbs, floating limbs, "
            "bad hands, missing fingers, extra fingers, "
            "bad eyes, missing eyes, extra eyes, "
            "low quality skin, plastic skin, doll skin, "
            "bad teeth, ugly"
        )
        
        self.logger.info(
            f"Generating at {width}x{height} with {sdxl_params['num_inference_steps']} steps, "
            f"guidance {sdxl_params['guidance_scale']}"
        )
        
        if lora_info:
            if lora_info['count'] == 1:
                lora = lora_info['stack'][0]
                self.logger.info(f"Using LoRA: {lora['name']} (weight: {lora['weight']:.2f})")
            else:
                lora_names = [lora['name'] for lora in lora_info['stack']]
                self.logger.info(f"Using {lora_info['count']} LoRAs: {', '.join(lora_names)} (total weight: {lora_info['total_weight']:.2f})")
            
        # DISABLED: Ensemble mode was causing partial denoising and quality issues
        # The refiner is now used as a separate stage (Stage 2) with full passes
        use_ensemble = False  # Force single-pass generation for consistent quality
        
        if use_ensemble:
            # Ensemble of expert denoisers approach
            total_steps = sdxl_params['num_inference_steps']
            switch_at = 0.8  # 80% base, 20% refiner for quality
            base_steps = int(total_steps * switch_at)
            
            self.logger.info(f"Using ensemble of expert denoisers: base for {base_steps} steps, refiner for final {total_steps - base_steps}")
            
            # Generate base image with specified denoising steps
            sdxl_params['denoising_end'] = switch_at
            output = self.pipe(**sdxl_params)
            base_image = output.images[0]
            
            # Continue with refiner for final steps
            refiner_params = {
                'prompt': prompt,
                'negative_prompt': sdxl_params['negative_prompt'],
                'image': base_image,
                'num_inference_steps': total_steps,
                'denoising_start': switch_at,
                'guidance_scale': params.get('guidance_scale', 7.5),
                'generator': generator,
            }
            
            # Apply refiner scheduler if different
            if hasattr(self.refiner_pipe, 'scheduler'):
                self.refiner_pipe.scheduler = scheduler_class.from_config(
                    self.refiner_pipe.scheduler.config,
                    **scheduler_kwargs
                )
            
            output = self.refiner_pipe(**refiner_params)
            image = output.images[0]
            
            self.logger.info("Ensemble generation complete")
        else:
            # Standard generation
            output = self.pipe(**sdxl_params)
            image = output.images[0]
        
        # Save stage 1 output
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        resolver = get_resolver()
        temp_dir = resolver.get_temp_dir() / 'ai-wallpaper'
        temp_dir.mkdir(parents=True, exist_ok=True)
        stage1_path = temp_dir / f"sdxl_stage1_{timestamp}.png"
        # Save with lossless PNG
        save_lossless_png(image, stage1_path)
        # Track for cleanup
        temp_files.append(stage1_path)
        
        self.logger.info(f"Stage 1 complete: Generated at {image.size}")
        self.logger.log_vram("After Stage 1")
        
        return self._standardize_stage_result(
            image_path=stage1_path,
            image=image,
            size=image.size,
            scheduler=scheduler_class.__name__,
            lora=lora_info
        )
        
    def _refine_stage2(
        self,
        input_image: Image.Image,
        prompt: str,
        seed: int,
        params: Dict[str, Any],
        temp_files: List[Path]
    ) -> Dict[str, Any]:
        """Stage 2: Img2img refinement
        
        Args:
            input_image: Input image
            prompt: Text prompt
            seed: Random seed
            params: Generation parameters
            
        Returns:
            Stage results
        """
        self.logger.log_stage("Stage 2", "SDXL img2img refinement")
        
        # Set up generator
        generator = torch.Generator(device=self.device).manual_seed(seed + 1)  # Different seed for variation
        
        # Get refinement settings from new config structure
        refinement_config = self.config['pipeline'].get('stages', {}).get('initial_refinement', {})
        strength = refinement_config.get('denoising_strength', 0.35)
        steps = refinement_config.get('steps', 50)
        use_refiner = refinement_config.get('use_refiner_model', True)
        
        self.logger.info(f"Refining with strength {strength:.2f}, {steps} steps")
        
        # Select scheduler for refinement
        scheduler_class = self._get_scheduler_class(params.get('scheduler'))
        scheduler_kwargs = self.config['generation'].get('scheduler_kwargs', {})
        
        # Apply scheduler to refiner pipe
        if hasattr(self.refiner_pipe, 'scheduler'):
            self.refiner_pipe.scheduler = scheduler_class.from_config(
                self.refiner_pipe.scheduler.config,
                **scheduler_kwargs
            )
        
        # Use the same enhanced negative prompt from stage 1
        negative_prompt = (
            "watercolor, painting, illustration, drawing, sketch, cartoon, anime, "
            "artistic, painted, brush strokes, canvas texture, paper texture, "
            "impressionism, expressionism, abstract, stylized, "
            "oil painting, acrylic, pastel, charcoal, "
            "(worst quality:1.4), (bad quality:1.4), (poor quality:1.4), "
            "blurry, soft focus, out of focus, bokeh, "
            "low resolution, low detail, pixelated, aliasing, "
            "jpeg artifacts, compression artifacts, "
            "oversaturated, undersaturated, overexposed, underexposed, "
            "grainy, noisy, film grain, sensor noise, "
            "bad anatomy, deformed, mutated, disfigured, "
            "extra limbs, missing limbs, floating limbs, "
            "bad hands, missing fingers, extra fingers, "
            "bad eyes, missing eyes, extra eyes, "
            "low quality skin, plastic skin, doll skin, "
            "bad teeth, ugly"
        )
        
        # Refine image with SDXL refiner
        refined = self.refiner_pipe(
            prompt=prompt,
            image=input_image,
            strength=strength,
            guidance_scale=params.get('guidance_scale', 7.5),
            num_inference_steps=steps,
            generator=generator,
            negative_prompt=negative_prompt
        ).images[0]
        
        # Save refined image
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        resolver = get_resolver()
        temp_dir = resolver.get_temp_dir() / 'ai-wallpaper'
        temp_dir.mkdir(parents=True, exist_ok=True)
        stage2_path = temp_dir / f"sdxl_stage2_{timestamp}.png"
        # Save refined image with lossless PNG
        save_lossless_png(refined, stage2_path)
        # Track for cleanup
        temp_files.append(stage2_path)
        
        self.logger.info(f"Stage 2 complete: Refined at {refined.size}")
        
        return self._standardize_stage_result(
            image_path=stage2_path,
            image=refined,
            size=refined.size,
            strength=strength
        )
        
    def _upscale_stage3(self, input_path: Path, temp_dirs: List[Path], params: Dict[str, Any]) -> Dict[str, Any]:
        """Stage 3: Upscale using pre-calculated strategy
        
        Args:
            input_path: Path to input image
            temp_dirs: List of temporary directories to track
            params: Generation parameters with upscale strategy
            
        Returns:
            Stage results
        """
        if 'upscale_strategy' in params:
            # Use pre-calculated strategy
            strategy = params['upscale_strategy']
            self.logger.info(f"Using upscale strategy with {len(strategy)} steps")
            
            current_image_path = input_path
            
            for i, step in enumerate(strategy):
                self.logger.log_stage(f"Stage 3.{i+1}", f"Apply {step['method']}")
                
                if step['method'] == 'realesrgan':
                    current_image_path = self._apply_realesrgan(
                        current_image_path,
                        step['scale'],
                        step['model'],
                        temp_dirs
                    )
                elif step['method'] == 'aspect_adjust_img2img':
                    # AI-based aspect adjustment
                    target_aspect = step['output_size'][0] / step['output_size'][1]
                    current_image_path = self._apply_aspect_adjustment(
                        current_image_path,
                        params.get('prompt', ''),
                        target_aspect,
                        temp_dirs
                    )
                elif step['method'] == 'lanczos_downsample':
                    # High-quality downsampling
                    current_image_path = self._apply_downsample(
                        current_image_path,
                        step['output_size']
                    )
                elif step['method'] == 'center_crop':
                    # Deprecated - should not be used, but kept for backward compatibility
                    self.logger.warning("center_crop is deprecated - use lanczos_downsample instead")
                    current_image_path = self._apply_center_crop(
                        current_image_path,
                        step['output_size']
                    )
            
            return self._standardize_stage_result(
                image_path=current_image_path,
                image=None,
                size=strategy[-1]['output_size'] if strategy else None
            )
        else:
            # Fallback to old logic for backward compatibility
            current_width, current_height = self.config['generation']['dimensions']
            target_width = 3840
            scale_factor = target_width / current_width  # 3840 / 1344 = 2.857
            
            # Round to nearest supported scale (3x)
            scale_factor = 3
            
            self.logger.log_stage("Stage 3", f"Real-ESRGAN {scale_factor}x upscaling")
        
        # Find Real-ESRGAN
        realesrgan_script = self._find_realesrgan()
        
        # Prepare paths
        resolver = get_resolver()
        temp_dir = resolver.get_temp_dir() / 'ai-wallpaper'
        temp_dir.mkdir(parents=True, exist_ok=True)
        temp_output_dir = temp_dir / f"sdxl_upscaled_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        temp_output_dir.mkdir(exist_ok=True)
        # Track for cleanup
        temp_dirs.append(temp_output_dir)
        
        # Build command for 2x upscale
        if str(realesrgan_script).endswith('.py'):
            cmd = [
                sys.executable,
                str(realesrgan_script),
                "-n", "RealESRGAN_x4plus",  # Use best model even for 2x
                "-i", str(input_path),
                "-o", str(temp_output_dir),
                "--outscale", str(scale_factor),  # Dynamic scale
                "-t", "256",  # Smaller tile size to avoid Real-ESRGAN bug
                "--fp32"
            ]
        else:
            cmd = [
                str(realesrgan_script),
                "-i", str(input_path),
                "-o", str(temp_output_dir),
                "-s", str(scale_factor),
                "-n", "realesrgan-x4plus",
                "-t", "256"
            ]
            
        self.logger.debug(f"Executing: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                timeout=300
            )
            
            if result.stdout:
                self.logger.debug(f"Real-ESRGAN output: {result.stdout}")
                
        except subprocess.CalledProcessError as e:
            error_msg = f"Command failed with exit code {e.returncode}"
            if e.stderr:
                error_msg += f"\nError output: {e.stderr}"
            raise UpscalerError(str(input_path), Exception(error_msg))
        except subprocess.TimeoutExpired:
            raise UpscalerError(str(input_path), Exception("Upscaling timed out"))
            
        # Find output file
        output_files = list(temp_output_dir.glob("*.png"))
        if not output_files:
            raise UpscalerError(str(input_path), FileNotFoundError("No output from Real-ESRGAN"))
            
        output_path = output_files[0]
        
        # Load and verify
        with Image.open(output_path) as upscaled_image:
            upscaled_size = upscaled_image.size
            
        # Expected size after upscaling
        expected_size = (
            self.config['generation']['dimensions'][0] * scale_factor,
            self.config['generation']['dimensions'][1] * scale_factor
        )
        
        if upscaled_size != expected_size:
            self.logger.warning(f"Unexpected size: {upscaled_size}, expected {expected_size}")
            
        self.logger.info(f"Stage 3 complete: Upscaled to {upscaled_size}")
        
        # Note: Not loading image here since it's not needed by caller
        # Caller only uses image_path to pass to next stage
        return self._standardize_stage_result(
            image_path=output_path,
            image=None,  # Not loading to save memory
            size=upscaled_size,
            scale_factor=scale_factor
        )
        
    def _finalize_stage4(self, input_path: Path, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Stage 4: Final adjustment to target resolution
        
        Args:
            input_path: Path to upscaled image
            params: Generation parameters including target resolution
            
        Returns:
            Stage results
        """
        # Check if user wants to skip default 4K upscaling
        if params and params.get('skip_default_upscale'):
            self.logger.log_stage("Stage 4", "Skipping default 4K finalization - using user resolution")
            # Just return the input as-is
            return self._standardize_stage_result(
                image_path=input_path,
                image=None
            )
            
        self.logger.log_stage("Stage 4", "Finalizing to 4K")
        
        # Load upscaled image
        with Image.open(input_path) as temp_image:
            image = temp_image.copy()  # Create a copy that persists after context exit
        
        # Target resolution - use from params if available, otherwise default to 4K
        if params and 'target_resolution' in params:
            target_width, target_height = params['target_resolution']
            self.logger.info(f"Using target resolution from params: {target_width}x{target_height}")
        else:
            target_width, target_height = 3840, 2160
        
        # Handle different upscaled sizes
        if image.size != (target_width, target_height):
            self.logger.info(f"Resizing from {image.size} to {target_width}x{target_height}")
            
            # Use high-quality Lanczos resampling
            image = image.resize(
                (target_width, target_height),
                Image.Resampling.LANCZOS
            )
            
            self.logger.info(f"Resized to {target_width}x{target_height} using Lanczos resampling")
            
        # Save final image
        config = get_config()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"sdxl_4k_{timestamp}.png"
        
        resolver = get_resolver()
        output_dir = Path(config.paths.get('images_dir', str(resolver.get_data_dir() / 'wallpapers')))
        output_dir.mkdir(exist_ok=True)
        
        final_path = output_dir / filename
        # Save final image with lossless PNG
        save_lossless_png(image, final_path)
        
        self.logger.info(f"Stage 4 complete: Final 4K image at {image.size}")
        self.logger.info(f"Saved to: {final_path}")
        
        # Cleanup now handled in finally block
        
        return self._standardize_stage_result(
            image_path=final_path,
            image=image,
            size=image.size
        )
        
    def _get_scheduler_class(self, scheduler_name: Optional[str]):
        """Get scheduler class from name
        
        Args:
            scheduler_name: Name of scheduler
            
        Returns:
            Scheduler class
        """
        scheduler_map = {
            'DPMSolverMultistepScheduler': DPMSolverMultistepScheduler,
            'EulerAncestralDiscreteScheduler': EulerAncestralDiscreteScheduler,
            'DDIMScheduler': DDIMScheduler,
            'HeunDiscreteScheduler': HeunDiscreteScheduler,
            'KDPM2DiscreteScheduler': KDPM2DiscreteScheduler
        }
        
        if scheduler_name and scheduler_name in scheduler_map:
            scheduler_class = scheduler_map[scheduler_name]
            # Validate compatibility with current model
            if self._validate_scheduler_compatibility(scheduler_class):
                return scheduler_class
            else:
                self.logger.warning(f"Scheduler {scheduler_name} not compatible with current model, using default")
                return DPMSolverMultistepScheduler
            
        # Random selection from available
        if 'scheduler_options' in self.config['generation']:
            options = self.config['generation']['scheduler_options']
            # Try each option until we find a compatible one
            random.shuffle(options)
            for option in options:
                scheduler_class = scheduler_map.get(option)
                if scheduler_class and self._validate_scheduler_compatibility(scheduler_class):
                    return scheduler_class
            self.logger.warning("No compatible schedulers found in options, using default")
            
        return DPMSolverMultistepScheduler
    
    def _validate_scheduler_compatibility(self, scheduler_class) -> bool:
        """Validate if scheduler is compatible with current model
        
        Args:
            scheduler_class: Scheduler class to validate
            
        Returns:
            True if compatible, False otherwise
        """
        try:
            # Try to initialize scheduler with current pipeline config
            test_scheduler = scheduler_class.from_config(self.pipe.scheduler.config)
            # Check if scheduler has required methods
            required_methods = ['set_timesteps', 'step', 'scale_model_input']
            for method in required_methods:
                if not hasattr(test_scheduler, method):
                    return False
            return True
        except Exception as e:
            self.logger.debug(f"Scheduler {scheduler_class.__name__} compatibility check failed: {e}")
            return False
        
    def _select_and_load_lora(self, theme: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Select and load multiple LoRAs based on theme for photorealistic enhancement
        
        Args:
            theme: Theme information
            
        Returns:
            LoRA stack info if loaded
        """
        if not self.config['lora'].get('enabled') or not self.available_loras:
            return None
            
        loaded_loras = []
        category = theme.get('category', 'UNKNOWN')
        
        # Determine if this theme category typically contains people/faces
        face_likely_categories = {
            'LOCAL_MEDIA',      # Star Trek, Marvel, etc. - characters/people
            'GENRE_FUSION',     # Often includes characters
            'ANIME_MANGA',      # Characters are central
        }
        
        # Theme elements that suggest people/faces
        face_likely_elements = [
            'character', 'hero', 'villain', 'person', 'people', 'portrait',
            'face', 'crew', 'team', 'warrior', 'soldier', 'human', 'man', 'woman',
            'doctor', 'captain', 'commander', 'pilot', 'detective', 'wizard'
        ]
        
        # Check if theme likely contains faces
        theme_has_faces = category in face_likely_categories
        
        # Also check theme elements if available
        if not theme_has_faces and 'elements' in theme:
            theme_text = ' '.join(str(e).lower() for e in theme.get('elements', []))
            theme_has_faces = any(element in theme_text for element in face_likely_elements)
            
        if theme_has_faces:
            self.logger.info(f"Theme category '{category}' likely contains faces - face helper will be included")
        else:
            self.logger.info(f"Theme category '{category}' unlikely to contain faces - skipping face helper")
        
        # Theme-specific LoRA presets using SDXL-compatible LoRAs
        theme_presets = {
            "NATURE_EXPANDED": {
                "required": ["extremely_detailed_sdxl"],
                "optional": ["photorealistic_slider_sdxl"],
                "weights": {"extremely_detailed_sdxl": 0.9, "photorealistic_slider_sdxl": 0.8}
            },
            "LOCAL_MEDIA": {
                "required": ["face_helper_sdxl", "extremely_detailed_sdxl"],
                "optional": ["fantasy_slider_sdxl"],
                "weights": {"face_helper_sdxl": 0.8, "extremely_detailed_sdxl": 0.9, "fantasy_slider_sdxl": 1.0}
            },
            "URBAN_CITYSCAPE": {
                "required": ["extremely_detailed_sdxl"],
                "optional": ["photorealistic_slider_sdxl"],
                "weights": {"extremely_detailed_sdxl": 0.8, "photorealistic_slider_sdxl": 1.0}
            },
            "GENRE_FUSION": {
                "required": ["extremely_detailed_sdxl"],
                "optional": ["cyberpunk_sdxl", "fantasy_slider_sdxl"],
                "weights": {"extremely_detailed_sdxl": 0.8, "cyberpunk_sdxl": 1.0, "fantasy_slider_sdxl": 1.0}
            },
            "SPACE_COSMIC": {
                "required": ["extremely_detailed_sdxl"],
                "optional": ["scifi_70s_sdxl"],
                "weights": {"extremely_detailed_sdxl": 0.8, "scifi_70s_sdxl": 0.9}
            },
            "TEMPORAL": {
                "required": ["extremely_detailed_sdxl"],
                "optional": ["scifi_70s_sdxl", "fantasy_slider_sdxl"],
                "weights": {"extremely_detailed_sdxl": 0.8, "scifi_70s_sdxl": 0.9, "fantasy_slider_sdxl": 1.0}
            },
            "ANIME_MANGA": {
                "required": ["anime_slider_sdxl"],
                "optional": ["extremely_detailed_sdxl"],
                "weights": {"anime_slider_sdxl": 1.8, "extremely_detailed_sdxl": 0.7}
            },
            "DIGITAL_PROGRAMMING": {
                "required": ["cyberpunk_sdxl"],
                "optional": ["extremely_detailed_sdxl"],
                "weights": {"cyberpunk_sdxl": 1.1, "extremely_detailed_sdxl": 0.8}
            },
            "DEFAULT": {
                "required": ["extremely_detailed_sdxl"],
                "optional": ["photorealistic_slider_sdxl"],
                "optional_if_faces": ["face_helper_sdxl"],
                "weights": {"extremely_detailed_sdxl": 0.8, "photorealistic_slider_sdxl": 1.0, "face_helper_sdxl": 0.8}
            }
        }
        
        # Get preset for theme or use default
        preset = theme_presets.get(category, theme_presets["DEFAULT"])
        
        # Load required LoRAs
        for lora_name in preset["required"]:
            if lora_name in self.available_loras:
                lora_info = self.available_loras[lora_name]
                weight = preset["weights"].get(lora_name, random.uniform(*lora_info["weight_range"]))
                loaded_loras.append({
                    "name": lora_name,
                    "path": lora_info["path"],
                    "weight": weight,
                    "category": lora_info["category"]
                })
                
        # Load optional LoRAs with probability
        for lora_name in preset.get("optional", []):
            if lora_name in self.available_loras and random.random() > 0.5:
                lora_info = self.available_loras[lora_name]
                weight = preset["weights"].get(lora_name, random.uniform(*lora_info["weight_range"]))
                loaded_loras.append({
                    "name": lora_name,
                    "path": lora_info["path"],
                    "weight": weight,
                    "category": lora_info["category"]
                })
                
        # Load face-specific LoRAs only if theme likely contains faces
        if theme_has_faces:
            for lora_name in preset.get("optional_if_faces", []):
                if lora_name in self.available_loras:
                    lora_info = self.available_loras[lora_name]
                    weight = preset["weights"].get(lora_name, random.uniform(*lora_info["weight_range"]))
                    loaded_loras.append({
                        "name": lora_name,
                        "path": lora_info["path"],
                        "weight": weight,
                        "category": lora_info["category"]
                    })
                    self.logger.debug(f"Added face-specific LoRA: {lora_name}")
                
        # Apply LoRA stack
        if loaded_loras:
            # Check total weight doesn't exceed 4.0
            total_weight = sum(l["weight"] for l in loaded_loras)
            if total_weight > 4.0:
                # Scale down proportionally
                scale_factor = 4.0 / total_weight
                for lora in loaded_loras:
                    lora["weight"] *= scale_factor
                    
            # Load all LoRAs using new multi-LoRA approach
            self.logger.info(f"Loading {len(loaded_loras)} LoRAs for theme {category}")
            
            try:
                # Load each LoRA
                adapter_names = []
                adapter_weights = []
                
                for i, lora in enumerate(loaded_loras):
                    adapter_name = f"adapter_{i}_{lora['name']}"
                    self.logger.info(f"Loading LoRA: {lora['name']} @ {lora['weight']:.2f}")
                    
                    self.pipe.load_lora_weights(
                        lora['path'],
                        adapter_name=adapter_name
                    )
                    adapter_names.append(adapter_name)
                    adapter_weights.append(lora['weight'])
                    
                # Set all adapters with their weights
                self.pipe.set_adapters(adapter_names, adapter_weights=adapter_weights)
                
                return {
                    "stack": loaded_loras,
                    "total_weight": sum(l["weight"] for l in loaded_loras),
                    "count": len(loaded_loras)
                }
                
            except Exception as e:
                error_msg = f"Failed to load LoRA stack: {e}"
                
                # For compatibility errors, provide detailed explanation
                if "size mismatch" in str(e).lower():
                    raise ModelError(
                        f"{error_msg}\n"
                        f"LoRA architecture incompatibility detected!\n"
                        f"The LoRA models are not compatible with the current SDXL model.\n"
                        f"This usually means the LoRA was trained on a different model version.\n"
                        f"Please use compatible LoRAs or disable LoRA loading."
                    )
                
                # For other errors, try fallback to single LoRA
                if loaded_loras:
                    self.logger.info("Attempting fallback to single LoRA...")
                    try:
                        lora = loaded_loras[0]
                        self.logger.info(f"Trying single LoRA: {lora['name']} @ {lora['weight']:.2f}")
                        self.pipe.load_lora_weights(lora['path'])
                        self.pipe.fuse_lora(lora_scale=lora['weight'])
                        self.logger.info("Single LoRA fallback successful")
                        return {
                            "stack": [lora],
                            "total_weight": lora['weight'],
                            "count": 1
                        }
                    except Exception as fallback_e:
                        self.logger.warning(f"Single LoRA fallback also failed: {fallback_e}")
                        self.logger.info("Generation will continue without LoRAs")
                        
        return None
        
    def _unload_lora(self) -> None:
        """Unload current LoRA"""
        try:
            self.pipe.unfuse_lora()
            self.pipe.unload_lora_weights()
            self.logger.info("LoRA unloaded")
        except Exception as e:
            self.logger.warning(f"Failed to unload LoRA: {e}")
            
    def _scan_available_loras(self) -> Dict[str, Dict[str, Any]]:
        """Scan and catalog all available LoRAs"""
        loras = {}
        model_resolver = get_model_resolver()
        # Look for SDXL-specific LoRA directory in model search paths
        lora_base_dir = None
        for search_path in model_resolver.search_paths:
            potential_lora_dir = search_path / 'loras-sdxl'
            if potential_lora_dir.exists():
                lora_base_dir = potential_lora_dir
                break
        
        if not lora_base_dir:
            # Use default location with SDXL-specific directory
            resolver = get_resolver()
            lora_base_dir = resolver.get_data_dir() / 'models' / 'loras-sdxl'
        
        if not lora_base_dir.exists():
            return loras
            
        # Define LoRA metadata for SDXL-compatible LoRAs
        lora_metadata = {
            # General Enhancement LoRAs (for all themes)
            "extremely-detailed-sdxl.safetensors": {
                "name": "extremely_detailed_sdxl",
                "category": "detail",
                "weight_range": [0.7, 1.0],
                "purpose": "Enhanced detail generation",
                "compatible_themes": ["all"]
            },
            "face-helper-sdxl.safetensors": {
                "name": "face_helper_sdxl",
                "category": "detail",
                "weight_range": [0.6, 0.9],
                "purpose": "Improved facial features",
                "compatible_themes": ["all"]
            },
            "photorealistic-slider-sdxl.safetensors": {
                "name": "photorealistic_slider_sdxl",
                "category": "photorealism",
                "weight_range": [0.8, 1.2],
                "purpose": "Adjustable photorealism enhancement",
                "compatible_themes": ["all"]
            },
            # Theme-Specific LoRAs
            "anime-slider-sdxl.safetensors": {
                "name": "anime_slider_sdxl",
                "category": "style",
                "weight_range": [1.5, 2.0],
                "purpose": "Anime style enhancement",
                "trigger": "anime",
                "compatible_themes": ["ANIME_MANGA"]
            },
            "cyberpunk-sdxl.safetensors": {
                "name": "cyberpunk_sdxl",
                "category": "style",
                "weight_range": [0.8, 1.2],
                "purpose": "Cyberpunk tech noir aesthetics",
                "trigger": "a cityscape in szn style",
                "compatible_themes": ["GENRE_FUSION", "DIGITAL_PROGRAMMING"]
            },
            "scifi-70s-sdxl.safetensors": {
                "name": "scifi_70s_sdxl",
                "category": "style",
                "weight_range": [0.7, 1.0],
                "purpose": "Retro sci-fi aesthetics",
                "trigger": "<s0><s1>",
                "compatible_themes": ["SPACE_COSMIC", "TEMPORAL"]
            },
            "fantasy-slider-sdxl.safetensors": {
                "name": "fantasy_slider_sdxl",
                "category": "style",
                "weight_range": [0.8, 1.2],
                "purpose": "Fantasy and magical elements",
                "trigger": "fantasy",
                "compatible_themes": ["LOCAL_MEDIA", "GENRE_FUSION", "TEMPORAL"]
            }
        }
        
        # Scan all LoRA files
        for lora_file in lora_base_dir.rglob("*.safetensors"):
            if lora_file.stat().st_size > 0:  # Skip empty files
                filename = lora_file.name
                if filename in lora_metadata:
                    metadata = lora_metadata[filename].copy()
                    metadata["path"] = str(lora_file)
                    metadata["size_mb"] = lora_file.stat().st_size / (1024 * 1024)
                    loras[metadata["name"]] = metadata
                    self.logger.debug(f"Found LoRA: {metadata['name']} ({metadata['size_mb']:.1f}MB)")
                    
        self.logger.info(f"Scanned and found {len(loras)} valid LoRAs")
        return loras
            
    def _load_refiner_model(self) -> None:
        """Load SDXL Refiner model for ensemble of expert denoisers
        
        The refiner is specifically trained for the final denoising steps
        and produces higher quality results when used properly.
        """
        refiner_path = self.config.get('refiner_model_path', 'stabilityai/stable-diffusion-xl-refiner-1.0')
        refiner_checkpoint = self.config.get('refiner_checkpoint_path')
        
        try:
            if refiner_checkpoint and Path(refiner_checkpoint).exists():
                self.logger.info(f"Loading SDXL Refiner from checkpoint: {refiner_checkpoint}")
                # Load refiner from single file
                self.refiner_pipe = StableDiffusionXLImg2ImgPipeline.from_single_file(
                    refiner_checkpoint,
                    torch_dtype=torch.float16,
                    use_safetensors=True,
                    variant="fp16",
                    add_watermarker=False
                )
            else:
                self.logger.info(f"Loading SDXL Refiner from: {refiner_path}")
                # Load from HuggingFace
                self.refiner_pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
                    refiner_path,
                    torch_dtype=torch.float16,
                    use_safetensors=True,
                    variant="fp16"
                )
            
            # Move to GPU and enable optimizations
            self.refiner_pipe = self.refiner_pipe.to(self.device)
            self.refiner_pipe.enable_vae_slicing()
            self.refiner_pipe.enable_vae_tiling()
            
            # Try to enable xformers
            try:
                self.refiner_pipe.enable_xformers_memory_efficient_attention()
                self.logger.info("xFormers enabled for refiner")
            except:
                pass
                
            self.logger.info("SDXL Refiner loaded successfully")
            
            # Initialize smart quality refiner - MUST SUCCEED
            self.smart_refiner = None
            if self.refiner_pipe:
                self.smart_refiner = SmartQualityRefiner(self)
                self.logger.info("Smart quality refiner initialized")
            else:
                self.logger.warning("No refiner pipe - smart refiner not initialized")
            
        except Exception as e:
            self.logger.warning(f"Failed to load SDXL Refiner: {e}")
            self.logger.warning("Continuing without refiner - quality may be reduced")
            self.refiner_pipe = None
    
    def _load_from_single_file(self, checkpoint_path: str) -> StableDiffusionXLPipeline:
        """Load SDXL from single safetensors checkpoint
        
        Args:
            checkpoint_path: Path to checkpoint file
            
        Returns:
            Loaded pipeline
            
        Raises:
            ModelLoadError: If loading fails
        """
        try:
            # Validate checkpoint exists
            checkpoint = Path(checkpoint_path)
            if not checkpoint.exists():
                raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
                
            if not checkpoint.suffix in ['.safetensors', '.ckpt']:
                raise ValueError(f"Unsupported checkpoint format: {checkpoint.suffix}")
                
            self.logger.info("Loading SDXL from single checkpoint file...")
            
            # Import method for loading from single file
            from diffusers.loaders import FromSingleFileMixin
            
            # Load pipeline from single file
            # CRITICAL: Must use from_single_file method
            pipe = StableDiffusionXLPipeline.from_single_file(
                checkpoint_path,
                torch_dtype=torch.float16,
                use_safetensors=True,
                variant="fp16",
                add_watermarker=False  # Disable watermarker
                # Note: SDXL doesn't have safety_checker
            )
            
            self.logger.info("Successfully loaded SDXL from checkpoint")
            return pipe
            
        except Exception as e:
            self.logger.error(f"Failed to load checkpoint: {e}")
            raise ModelLoadError(f"SDXL checkpoint loading failed: {checkpoint_path}", e)
    
    def _find_realesrgan(self) -> Path:
        """Find Real-ESRGAN installation
        
        Returns:
            Path to Real-ESRGAN script
            
        Raises:
            UpscalerError: If not found
        """
        config = get_config()
        
        # Get configured paths
        realesrgan_paths = config.paths.get('models', {}).get('real_esrgan', [])
        
        # Add common locations using resolver
        resolver = get_resolver()
        common_paths = [
            resolver.project_root / "Real-ESRGAN/inference_realesrgan.py",
            Path.home() / "Real-ESRGAN/inference_realesrgan.py",
            Path("/opt/Real-ESRGAN/inference_realesrgan.py"),
            Path("/usr/local/bin/realesrgan-ncnn-vulkan")
        ]
        
        all_paths = realesrgan_paths + [str(p) for p in common_paths]
        
        for path in all_paths:
            path = Path(path).expanduser()
            if path.exists():
                self.logger.info(f"Found Real-ESRGAN at: {path}")
                return path
                
        # Not found
        error_msg = (
            "Real-ESRGAN not found! Cannot proceed with upscaling.\n"
            "Real-ESRGAN is REQUIRED for 4K wallpapers.\n"
            "Please install Real-ESRGAN"
        )
        
        raise UpscalerError("Real-ESRGAN", Exception(error_msg))
        
    def _pre_cleanup(self) -> None:
        """Pre-cleanup hook: clean up all pipelines and LoRA"""
        # Unload LoRA if loaded
        self._unload_lora()
        
        # Clean up inpaint pipeline if created
        if hasattr(self, 'inpaint_pipe'):
            self.logger.debug("Cleaning up inpaint pipeline")
            del self.inpaint_pipe
        
        # Unregister from resource manager
        self.resource_manager.unregister_model(self.model_name)
        
    def get_optimal_prompt(self, theme: Dict, weather: Dict, context: Dict) -> str:
        """Get SDXL-optimized prompt
        
        Args:
            theme: Theme dictionary
            weather: Weather context
            context: Additional context
            
        Returns:
            Optimized prompt for SDXL
        """
        # Let DeepSeek handle it, but we'll enhance in generate
        return ""
        
    def get_pipeline_stages(self) -> List[str]:
        """Return pipeline stages for SDXL
        
        Returns:
            List of stage names
        """
        stages = ["sdxl_generation"]  # 1920x1024
        
        if self.config['pipeline'].get('enable_img2img_refine'):
            stages.append("sdxl_img2img_refine")
            
        stages.extend([
            "realesrgan_2x",  # 2x to 3840x2048
            "finalize_4k"     # Adjust to 3840x2160
        ])
        
        return stages
        
    def validate_environment(self) -> Tuple[bool, str]:
        """Validate SDXL can run in current environment
        
        Returns:
            Tuple of (is_valid, message)
        """
        # Check CUDA
        if not torch.cuda.is_available():
            return False, "CUDA is required for SDXL model"
            
        # Check VRAM
        device_props = torch.cuda.get_device_properties(0)
        vram_gb = device_props.total_memory / 1024**3
        
        if vram_gb < 8:
            return False, f"Insufficient VRAM: {vram_gb:.1f}GB, need at least 8GB"
            
        # Check Real-ESRGAN
        try:
            self._find_realesrgan()
        except UpscalerError:
            return False, "Real-ESRGAN is required but not found"
            
        return True, "Environment validated for SDXL"
        
    def supports_feature(self, feature: str) -> bool:
        """Check if SDXL supports a feature
        
        Args:
            feature: Feature name
            
        Returns:
            True if supported
        """
        features = {
            'lora': True,
            'img2img': True,
            'scheduler_selection': True,
            'custom_dimensions': True,
            '8k_pipeline': False,
            'controlnet': False  # Could be added
        }
        return features.get(feature, super().supports_feature(feature))
        
    def get_resource_requirements(self) -> Dict[str, Any]:
        """Return SDXL resource requirements
        
        Returns:
            Resource requirements
        """
        return {
            'vram_gb': 10,  # SDXL needs ~8-10GB
            'disk_gb': 15,  # Model is ~6.5GB + LoRAs
            'time_minutes': 5
        }
    
    def _needs_aspect_adjustment(self, params: Dict[str, Any]) -> bool:
        """Check if aspect adjustment is needed"""
        gen_aspect = params.get('generation_aspect')
        target_aspect = params.get('target_aspect')
        
        if not gen_aspect or not target_aspect:
            return False
        
        # Check if aspects are significantly different
        return abs(gen_aspect - target_aspect) > 0.05
    
    def _create_inpaint_pipeline(self):
        """Create inpaint pipeline from existing pipeline"""
        if not hasattr(self, 'pipe') or self.pipe is None:
            raise RuntimeError("Cannot create inpaint pipeline - base pipeline not initialized")
        
        self.logger.info("Creating SDXL inpaint pipeline for aspect adjustment")
        
        from diffusers import StableDiffusionXLInpaintPipeline
        
        # Create inpaint pipeline sharing components
        self.inpaint_pipe = StableDiffusionXLInpaintPipeline(
            vae=self.pipe.vae,
            text_encoder=self.pipe.text_encoder,
            text_encoder_2=self.pipe.text_encoder_2,
            tokenizer=self.pipe.tokenizer,
            tokenizer_2=self.pipe.tokenizer_2,
            unet=self.pipe.unet,
            scheduler=self.pipe.scheduler
        ).to(self.device)
        
        # Apply same optimizations
        if hasattr(self.pipe, 'enable_model_cpu_offload'):
            self.inpaint_pipe.enable_model_cpu_offload()
        
        self.logger.info("Inpaint pipeline created successfully")
    
    def _refine_stage2_full(self,
                           image: Image.Image,
                           image_path: Path,
                           prompt: str,
                           seed: int,
                           params: Dict,
                           temp_files: List,
                           strength_override: Optional[float] = None) -> Dict[str, Any]:
        """
        Refine the full image (after aspect adjustment).
        FAILS if image is too large - no fallbacks!
        """
        w, h = image.size
        
        # Ensure SDXL-compatible dimensions
        if w % 8 != 0 or h % 8 != 0:
            w = ((w + 7) // 8) * 8
            h = ((h + 7) // 8) * 8
            image = image.resize((w, h), Image.Resampling.LANCZOS)
            self.logger.info(f"Resized for refinement: {w}x{h}")
        
        # Get refinement parameters
        refine_config = self.config['pipeline']['stages'].get('initial_refinement', {})
        refine_strength = strength_override or refine_config.get('denoising_strength', 0.3)
        refine_steps = refine_config.get('steps', 50)
        
        self.logger.info(f"Refining {w}x{h} image (strength: {refine_strength}, steps: {refine_steps})")
        
        # Run refinement
        generator = torch.Generator(device=self.device).manual_seed(seed + 1)
        
        refined = self.refiner_pipe(
            prompt=prompt,
            image=image,
            strength=refine_strength,
            num_inference_steps=refine_steps,
            generator=generator,
            width=w,
            height=h
        ).images[0]
        
        # Save refined image
        temp_dir = Path(tempfile.gettempdir()) / 'ai-wallpaper'
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        refined_path = temp_dir / f"refined_{w}x{h}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.png"
        # Save refined image with lossless PNG
        save_lossless_png(refined, refined_path)
        temp_files.append(refined_path)
        
        return {
            'image': refined,
            'image_path': refined_path,
            'size': refined.size
        }
    
    def _upscale_stage3_simple(self,
                              image_path: Path,
                              temp_dirs: List[Path],
                              strategy: List[Dict]) -> Dict[str, Any]:
        """
        Simple upscaling without aspect adjustment.
        Only Real-ESRGAN and downsampling operations.
        """
        if not strategy:
            return {'image_path': image_path, 'skipped': True}
        
        current_path = image_path
        
        for i, step in enumerate(strategy):
            self.logger.log_stage(f"Stage 3.{i+1}", step['description'])
            
            if step['method'] == 'realesrgan':
                current_path = self._apply_realesrgan(
                    current_path,
                    step['scale'],
                    step['model'],
                    temp_dirs
                )
            elif step['method'] == 'lanczos_downsample':
                current_path = self._apply_downsample(
                    current_path,
                    step['output_size']
                )
            else:
                raise ValueError(f"Unknown upscale method: {step['method']}")
        
        # Get final size
        with Image.open(current_path) as img:
            final_size = img.size
        
        return {
            'image_path': current_path,
            'size': final_size,
            'steps_applied': len(strategy)
        }
    
    def _ensure_exact_size(self, image_path: Path, target_size: Tuple[int, int]) -> Path:
        """Ensure image is exactly the target size"""
        with Image.open(image_path) as img:
            if img.size == target_size:
                return image_path
            
            # Use high-quality downsampling
            downsampler = HighQualityDownsampler()
            return downsampler.downsample(
                image_path=image_path,
                target_size=target_size
            )
    
    def _tiled_ultra_refine(self,
                           image: Image.Image,
                           image_path: Path,
                           prompt: str,
                           params: Dict,
                           temp_files: List) -> Optional[Dict[str, Any]]:
        """
        Stage 2.5: Tiled ultra-refinement for maximum quality.
        Only runs in ultimate quality mode on large images.
        """
        # Check if we should run
        if params.get('quality_mode') != 'ultimate':
            return {'skipped': True, 'reason': 'Not in ultimate quality mode'}
        
        w, h = image.size
        if w * h < 1024 * 1024:
            return {'skipped': True, 'reason': 'Image too small for tiled refinement'}
        
        # Check VRAM availability
        if torch.cuda.is_available():
            free_vram = torch.cuda.mem_get_info()[0] / 1024**3  # GB
            if free_vram < 8.0:
                raise RuntimeError(
                    f"Insufficient VRAM for tiled refinement. "
                    f"Need 8GB+, have {free_vram:.1f}GB free"
                )
        
        try:
            # Import here to avoid circular dependencies
            from ..processing.tiled_refiner import TiledRefiner
            
            # Create refiner with img2img pipeline
            # Use refiner if available, otherwise main pipeline
            refine_pipe = self.refiner_pipe if self.refiner_pipe else self.pipe
            
            if not refine_pipe:
                raise RuntimeError("No pipeline available for tiled refinement")
            
            refiner = TiledRefiner(pipeline=refine_pipe)
            
            # Run tiled refinement
            refined_path = refiner.refine_tiled(
                image_path=image_path,
                prompt=prompt,
                base_strength=0.25,  # Lower strength for tiled
                base_steps=40
            )
            
            # Load result
            refined_image = Image.open(refined_path)
            temp_files.append(refined_path)
            
            return {
                'image': refined_image,
                'image_path': refined_path,
                'size': refined_image.size,
                'tiles_processed': True
            }
            
        except Exception as e:
            error_msg = f"Tiled refinement FAILED: {type(e).__name__}: {str(e)}"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg) from e
    
    def _apply_downsample(self, image_path: Path, target_size: Tuple[int, int]) -> Path:
        """Apply high-quality downsampling"""
        downsampler = HighQualityDownsampler()
        return downsampler.downsample(
            image_path=image_path,
            target_size=target_size
        )
    
    def _apply_realesrgan(self, input_path: Path, scale: int, model: str, temp_dirs: List[Path]) -> Path:
        """Apply Real-ESRGAN upscaling
        
        Args:
            input_path: Path to input image
            scale: Scale factor (2, 4, etc)
            model: Real-ESRGAN model name
            temp_dirs: List of temporary directories to track
            
        Returns:
            Path to upscaled image
        """
        realesrgan_script = self._find_realesrgan()
        
        # Prepare output directory
        resolver = get_resolver()
        temp_dir = resolver.get_temp_dir() / 'ai-wallpaper'
        temp_dir.mkdir(parents=True, exist_ok=True)
        temp_output_dir = temp_dir / f"upscale_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        temp_output_dir.mkdir(exist_ok=True)
        temp_dirs.append(temp_output_dir)
        
        # Build command
        if str(realesrgan_script).endswith('.py'):
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
        else:
            # Binary version
            cmd = [
                str(realesrgan_script),
                "-i", str(input_path),
                "-o", str(temp_output_dir),
                "-s", str(scale),
                "-n", model.lower().replace('_', '-'),
                "-t", "512"
            ]
        
        # Log full command for debugging
        self.logger.debug(f"Real-ESRGAN command: {' '.join(cmd)}")
        
        # Check input image exists and log its properties
        if not input_path.exists():
            raise UpscalerError(str(input_path), FileNotFoundError("Input image not found"))
        
        with Image.open(input_path) as img:
            w, h = img.size
            self.logger.info(f"Input image size: {w}x{h}, aspect ratio: {w/h:.2f}")
        
        # Execute with better error capture
        self.logger.info(f"Running Real-ESRGAN {scale}x with model {model}")
        
        try:
            # Use Popen for better control over stdout/stderr
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            stdout, stderr = process.communicate()
            
            if process.returncode != 0:
                self.logger.error(f"Real-ESRGAN failed with exit code: {process.returncode}")
                self.logger.error(f"Command: {' '.join(cmd)}")
                self.logger.error(f"STDOUT:\n{stdout}")
                self.logger.error(f"STDERR:\n{stderr}")
                
                # Extract meaningful error message
                error_msg = stderr.strip() or stdout.strip() or f"Exit code {process.returncode}"
                raise UpscalerError(str(input_path), Exception(f"Real-ESRGAN error: {error_msg}"))
            
            self.logger.debug(f"Real-ESRGAN stdout: {stdout}")
            if stderr:
                self.logger.debug(f"Real-ESRGAN stderr: {stderr}")
                
        except FileNotFoundError:
            raise UpscalerError(str(input_path), FileNotFoundError("Real-ESRGAN script not found"))
        except Exception as e:
            self.logger.error(f"Unexpected error running Real-ESRGAN: {type(e).__name__}: {e}")
            raise UpscalerError(str(input_path), e)
        
        # Find output
        output_files = list(temp_output_dir.glob("*.png"))
        if not output_files:
            raise UpscalerError(str(input_path), FileNotFoundError("No output from Real-ESRGAN"))
        
        return output_files[0]
    
    def _apply_center_crop(self, input_path: Path, target_size: Tuple[int, int]) -> Path:
        """Apply center cropping to exact dimensions
        
        Args:
            input_path: Path to input image
            target_size: Target dimensions (width, height)
            
        Returns:
            Path to cropped image
        """
        target_w, target_h = target_size
        
        with Image.open(input_path) as img:
            current_w, current_h = img.size
            
            self.logger.info(f"Center cropping from {current_w}x{current_h} to {target_w}x{target_h}")
            
            # Calculate crop box
            left = (current_w - target_w) // 2
            top = (current_h - target_h) // 2
            right = left + target_w
            bottom = top + target_h
            
            # Validate crop dimensions
            if left < 0 or top < 0 or right > current_w or bottom > current_h:
                raise ValueError(
                    f"Cannot crop {current_w}x{current_h} to {target_w}x{target_h} - target is larger"
                )
            
            # Crop
            cropped = img.crop((left, top, right, bottom))
            
            # Save with unique filename
            resolver = get_resolver()
            temp_dir = resolver.get_temp_dir() / 'ai-wallpaper'
            temp_dir.mkdir(parents=True, exist_ok=True)
            temp_path = temp_dir / f"cropped_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.png"
            # Save cropped image with lossless PNG
            save_lossless_png(cropped, temp_path)
            
            return temp_path
    
    def _apply_aspect_adjustment(self, input_path: Path, prompt: str, 
                               target_aspect: float, temp_dirs: List[Path]) -> Path:
        """Apply AI-based aspect ratio adjustment using img2img
        
        Args:
            input_path: Path to input image
            prompt: Original generation prompt
            target_aspect: Target aspect ratio (width/height)
            temp_dirs: List of temporary directories
            
        Returns:
            Path to adjusted image
        """
        # Initialize aspect adjuster with current pipeline
        # Note: We need to create an img2img pipeline from the existing pipeline
        if hasattr(self, 'pipe') and self.pipe is not None:
            # Create inpaint pipeline if needed (for mask-based outpainting)
            if not hasattr(self, 'inpaint_pipe'):
                self.logger.info("Creating inpaint pipeline for aspect adjustment")
                from diffusers import StableDiffusionXLInpaintPipeline
                self.inpaint_pipe = StableDiffusionXLInpaintPipeline(
                    vae=self.pipe.vae,
                    text_encoder=self.pipe.text_encoder,
                    text_encoder_2=self.pipe.text_encoder_2,
                    tokenizer=self.pipe.tokenizer,
                    tokenizer_2=self.pipe.tokenizer_2,
                    unet=self.pipe.unet,
                    scheduler=self.pipe.scheduler
                ).to(self.pipe.device)
            
            adjuster = AspectAdjuster(pipeline=self.inpaint_pipe)
        else:
            # No pipeline available, AspectAdjuster will fall back to reflect method
            self.logger.warning("No SDXL pipeline available for outpainting, will use reflect method")
            adjuster = AspectAdjuster(pipeline=None)
        
        # Perform adjustment
        adjusted_path = adjuster.adjust_aspect_ratio(
            image_path=input_path,
            original_prompt=prompt,
            target_aspect=target_aspect
        )
        
        return adjusted_path
    
    def _apply_downsample(self, input_path: Path, target_size: Tuple[int, int]) -> Path:
        """Apply high-quality downsampling to exact size
        
        Args:
            input_path: Path to input image
            target_size: Target dimensions (width, height)
            
        Returns:
            Path to downsampled image
        """
        # Use the new HighQualityDownsampler class
        downsampler = HighQualityDownsampler()
        
        # Delegate to the downsampler
        return downsampler.downsample(
            image_path=input_path,
            target_size=target_size
        )