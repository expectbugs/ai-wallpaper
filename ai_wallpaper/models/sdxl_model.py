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
from ..core import get_logger, get_config
from ..core.exceptions import ModelNotFoundError, ModelLoadError, GenerationError, UpscalerError
from ..utils import get_resource_manager

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
            
            # Check for Juggernaut XL first
            juggernaut_path = Path("/home/user/ai-wallpaper/models/checkpoints/Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors")
            if juggernaut_path.exists():
                self.logger.info("Loading Juggernaut XL v9 for superior photorealism...")
                self.pipe = self._load_from_single_file(str(juggernaut_path))
            else:
                # Fallback to standard SDXL
                model_path = self.config.get('model_path', 'stabilityai/stable-diffusion-xl-base-1.0')
                checkpoint_path = self.config.get('checkpoint_path')
                
                # Determine loading method
                if checkpoint_path and Path(checkpoint_path).exists():
                    # Load from single checkpoint file
                    self.logger.info(f"Loading SDXL from checkpoint: {checkpoint_path}")
                    self.pipe = self._load_from_single_file(checkpoint_path)
                else:
                    # Load from HuggingFace repo or directory
                    self.logger.info(f"Loading SDXL pipeline from: {model_path}")
                    self.pipe = StableDiffusionXLPipeline.from_pretrained(
                        model_path,
                        torch_dtype=torch.float16,
                        use_safetensors=True,
                        variant="fp16"
                    )
            
            # Move to GPU
            self.pipe = self.pipe.to("cuda")
            
            # Enable memory optimizations
            self.pipe.enable_vae_slicing()
            self.pipe.enable_vae_tiling()
            
            # Enable xformers if available
            try:
                self.pipe.enable_xformers_memory_efficient_attention()
                self.logger.info("xFormers enabled for memory efficiency")
            except:
                self.logger.info("xFormers not available, using standard attention")
                
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
        """Generate image using SDXL pipeline
        
        Args:
            prompt: Text prompt
            seed: Random seed
            **kwargs: Additional parameters
            
        Returns:
            Generation results dictionary
        """
        self.ensure_initialized()
        
        # Clean prompt
        prompt = self.validate_prompt(prompt)
        
        # Get generation parameters
        params = self.get_generation_params(**kwargs)
        
        # Check disk space before starting generation
        self.check_disk_space_for_generation(no_upscale=params.get('no_upscale', False))
        
        # Use provided seed or generate random
        if seed is None:
            seed = random.randint(0, 2**32 - 1)
            
        # Log generation start
        self.log_generation_start(prompt, {**params, 'seed': seed})
        
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
                
            # Stage 1: Generate base image
            stage1_result = self._generate_stage1(prompt, seed, params, lora_info, temp_files)
            
            # Save intermediate if requested
            if self._should_save_stages(params):
                stage1_result['saved_path'] = self._save_intermediate(
                    stage1_result['image'], 'stage1_base', prompt
                )
            
            # Stage 2: Optional img2img refinement
            initial_refinement = self.config['pipeline'].get('stages', {}).get('initial_refinement', {})
            if initial_refinement.get('enabled', True) and self.refiner_pipe:
                stage2_result = self._refine_stage2(
                    stage1_result['image'],
                    prompt,
                    seed,
                    params,
                    temp_files
                )
                
                # Save intermediate if requested
                if self._should_save_stages(params):
                    stage2_result['saved_path'] = self._save_intermediate(
                        stage2_result['image'], 'stage2_refined', prompt
                    )
            else:
                stage2_result = stage1_result
                
            # Check if upscaling is requested
            if not params.get('no_upscale', False):
                # Stage 3: Upscale 2x
                stage3_result = self._upscale_stage3(stage2_result['image_path'], temp_dirs)
                
                # Save intermediate if requested
                if self._should_save_stages(params):
                    # Open the upscaled image to save intermediate
                    with Image.open(stage3_result['image_path']) as img:
                        stage3_result['saved_path'] = self._save_intermediate(
                            img.copy(), 'stage3_upscaled', prompt
                        )
                
                # Stage 4: Final crop/adjustment to 4K
                stage4_result = self._finalize_stage4(stage3_result['image_path'])
                
                # Save intermediate if requested
                if self._should_save_stages(params):
                    with Image.open(stage4_result['image_path']) as img:
                        stage4_result['saved_path'] = self._save_intermediate(
                            img.copy(), 'stage4_final', prompt
                        )
                final_path = stage4_result['image_path']
            else:
                # Skip upscaling, use stage2 result as final
                self.logger.info("Skipping upscaling as requested")
                stage3_result = None
                stage4_result = None
                final_path = stage2_result['image_path']
            
            # Unload LoRA if loaded
            if lora_info:
                self._unload_lora()
                
            # Prepare final results
            duration = time.time() - start_time
            
            results = {
                'image_path': final_path,
                'metadata': {
                    'prompt': prompt,
                    'seed': seed,
                    'model': 'SDXL',
                    'lora': lora_info,
                    'parameters': params,
                    'duration': duration,
                    'timestamp': datetime.now().isoformat()
                },
                'stages': {
                    'stage1_generation': stage1_result,
                    'stage2_refinement': stage2_result if self.config['pipeline'].get('enable_img2img_refine') else None,
                    'stage3_upscale': stage3_result,
                    'stage4_finalize': stage4_result
                }
            }
            
            # Save metadata
            self.save_metadata(Path(results['image_path']), results['metadata'])
            
            # Log completion
            self.log_generation_complete(Path(results['image_path']), duration)
            
            return results
            
        except Exception as e:
            raise GenerationError(self.name, "pipeline execution", e)
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
        generator = torch.Generator(device="cuda").manual_seed(seed)
        
        # Get dimensions from config (16:9 for SDXL)
        width, height = 1344, 768  # Native SDXL 16:9
        
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
            self.logger.info(f"Using LoRA: {lora_info['name']} (weight: {lora_info['weight']:.2f})")
            
        # Check if we should use ensemble of expert denoisers
        use_ensemble = (
            self.refiner_pipe is not None and 
            self.config['pipeline'].get('stages', {}).get('base_generation', {}).get('enable_refiner', True)
        )
        
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
        stage1_path = Path(tempfile.gettempdir()) / f"sdxl_stage1_{timestamp}.png"
        image.save(stage1_path, "PNG", quality=100)
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
        generator = torch.Generator(device="cuda").manual_seed(seed + 1)  # Different seed for variation
        
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
        stage2_path = Path(tempfile.gettempdir()) / f"sdxl_stage2_{timestamp}.png"
        refined.save(stage2_path, "PNG", quality=100)
        # Track for cleanup
        temp_files.append(stage2_path)
        
        self.logger.info(f"Stage 2 complete: Refined at {refined.size}")
        
        return self._standardize_stage_result(
            image_path=stage2_path,
            image=refined,
            size=refined.size,
            strength=strength
        )
        
    def _upscale_stage3(self, input_path: Path, temp_dirs: List[Path]) -> Dict[str, Any]:
        """Stage 3: Upscale using Real-ESRGAN to reach 4K
        
        Args:
            input_path: Path to input image
            
        Returns:
            Stage results
        """
        # Calculate required scale factor
        current_width, current_height = self.config['generation']['dimensions']
        target_width = 3840
        scale_factor = target_width / current_width  # 3840 / 1344 = 2.857
        
        # Round to nearest supported scale (3x)
        scale_factor = 3
        
        self.logger.log_stage("Stage 3", f"Real-ESRGAN {scale_factor}x upscaling")
        
        # Find Real-ESRGAN
        realesrgan_script = self._find_realesrgan()
        
        # Prepare paths
        temp_output_dir = Path(tempfile.gettempdir()) / f"sdxl_upscaled_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
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
        
    def _finalize_stage4(self, input_path: Path) -> Dict[str, Any]:
        """Stage 4: Final adjustment to 4K
        
        Args:
            input_path: Path to upscaled image
            
        Returns:
            Stage results
        """
        self.logger.log_stage("Stage 4", "Finalizing to 4K")
        
        # Load upscaled image
        with Image.open(input_path) as temp_image:
            image = temp_image.copy()  # Create a copy that persists after context exit
        
        # Target 4K resolution
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
        
        output_dir = Path(config.paths.get('images_dir', tempfile.gettempdir()))
        output_dir.mkdir(exist_ok=True)
        
        final_path = output_dir / filename
        image.save(final_path, "PNG", quality=100)
        
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
        
        # Theme-specific LoRA presets
        theme_presets = {
            "NATURE_EXPANDED": {
                "required": ["better_picture_more_details"],
                "optional": ["sdxl_film_photography"],
                "weights": {"better_picture_more_details": 0.9, "sdxl_film_photography": 0.4}
            },
            "LOCAL_MEDIA": {
                "required": ["skin_realism_sdxl", "better_picture_more_details"],
                "optional": [],
                "weights": {"skin_realism_sdxl": 0.8, "better_picture_more_details": 0.9}
            },
            "URBAN_CITYSCAPE": {
                "required": ["better_picture_more_details"],
                "optional": ["sdxl_film_photography"],
                "weights": {"better_picture_more_details": 0.8, "sdxl_film_photography": 0.3}
            },
            "GENRE_FUSION": {
                "required": ["better_picture_more_details"],
                "optional": ["skin_realism_sdxl", "sdxl_film_photography"],
                "weights": {"better_picture_more_details": 0.8, "skin_realism_sdxl": 0.6, "sdxl_film_photography": 0.4}
            },
            "DEFAULT": {
                "required": ["better_picture_more_details"],
                "optional": ["skin_realism_sdxl"],
                "weights": {"better_picture_more_details": 0.8, "skin_realism_sdxl": 0.6}
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
        for lora_name in preset["optional"]:
            if lora_name in self.available_loras and random.random() > 0.5:
                lora_info = self.available_loras[lora_name]
                weight = preset["weights"].get(lora_name, random.uniform(*lora_info["weight_range"]))
                loaded_loras.append({
                    "name": lora_name,
                    "path": lora_info["path"],
                    "weight": weight,
                    "category": lora_info["category"]
                })
                
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
                self.logger.error(f"Failed to load LoRA stack: {e}")
                # Try fallback to single LoRA
                if loaded_loras:
                    try:
                        lora = loaded_loras[0]
                        self.pipe.load_lora_weights(lora['path'])
                        self.pipe.fuse_lora(lora_scale=lora['weight'])
                        return {
                            "stack": [lora],
                            "total_weight": lora['weight'],
                            "count": 1
                        }
                    except:
                        pass
                        
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
        lora_base_dir = Path("/home/user/ai-wallpaper/models/loras")
        
        if not lora_base_dir.exists():
            return loras
            
        # Define LoRA metadata
        lora_metadata = {
            # Skip detail_tweaker_xl - it's 4GB which is too large for a LoRA
            "better_picture_more_details.safetensors": {
                "name": "better_picture_more_details",
                "category": "detail",
                "weight_range": [0.6, 1.0],
                "purpose": "Eye, skin, hair detail",
                "compatible_themes": ["all"]
            },
            "skin_realism_sdxl.safetensors": {
                "name": "skin_realism_sdxl",
                "category": "photorealism",
                "weight_range": [0.5, 0.8],
                "purpose": "Natural skin imperfections",
                "trigger": "Detailed natural skin and blemishes",
                "compatible_themes": ["LOCAL_MEDIA", "GENRE_FUSION"]
            },
            "sdxl_film_photography.safetensors": {
                "name": "sdxl_film_photography",
                "category": "effects",
                "weight_range": [0.3, 0.6],
                "purpose": "Film grain, cinematic look",
                "compatible_themes": ["ATMOSPHERIC", "SPACE_COSMIC", "TEMPORAL"]
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
            self.refiner_pipe = self.refiner_pipe.to("cuda")
            self.refiner_pipe.enable_vae_slicing()
            self.refiner_pipe.enable_vae_tiling()
            
            # Try to enable xformers
            try:
                self.refiner_pipe.enable_xformers_memory_efficient_attention()
                self.logger.info("xFormers enabled for refiner")
            except:
                pass
                
            self.logger.info("SDXL Refiner loaded successfully")
            
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
        
        # Add common locations
        common_paths = [
            "/home/user/ai-wallpaper/Real-ESRGAN/inference_realesrgan.py",
            "/home/user/Real-ESRGAN/inference_realesrgan.py",
            Path.home() / "Real-ESRGAN/inference_realesrgan.py",
            "/usr/local/bin/realesrgan-ncnn-vulkan"
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
        """Pre-cleanup hook: unload LoRA and unregister from resource manager"""
        # Unload LoRA if loaded
        self._unload_lora()
        
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