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
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List
from datetime import datetime

import torch
from diffusers import (
    StableDiffusionXLPipeline,
    StableDiffusionXLImg2ImgPipeline,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    DDIMScheduler
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
        self.resource_manager = get_resource_manager()
        
    def initialize(self) -> bool:
        """Initialize SDXL model and verify requirements
        
        Returns:
            True if initialization successful
        """
        try:
            self.logger.info("Initializing SDXL model...")
            
            # Check resources
            self.resource_manager.check_critical_resources(
                self.model_name,
                self.get_resource_requirements()
            )
            
            # Clear resources for this model
            self.resource_manager.prepare_for_model(self.model_name)
            
            # Load pipeline
            model_path = self.config.get('model_path', 'stabilityai/stable-diffusion-xl-base-1.0')
            
            self.logger.info(f"Loading SDXL pipeline from: {model_path}")
            
            # Load with optimizations
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
                
            # Load img2img pipeline if refinement enabled
            if self.config['pipeline'].get('enable_img2img_refine', True):
                self.logger.info("Loading img2img pipeline for refinement...")
                self.refiner_pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
                    model_path,
                    torch_dtype=torch.float16,
                    use_safetensors=True,
                    variant="fp16"
                )
                self.refiner_pipe = self.refiner_pipe.to("cuda")
                self.refiner_pipe.enable_vae_slicing()
                self.refiner_pipe.enable_vae_tiling()
                
            # Verify Real-ESRGAN is available
            self._find_realesrgan()
            
            # Register with resource manager
            self.resource_manager.register_model(self.model_name, self)
            
            self._initialized = True
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
        
        # Use provided seed or generate random
        if seed is None:
            seed = random.randint(0, 2**32 - 1)
            
        # Log generation start
        self.log_generation_start(prompt, {**params, 'seed': seed})
        
        # Track timing
        start_time = time.time()
        
        try:
            # Select and load LoRA if enabled
            lora_info = None
            if self.config['lora'].get('enabled') and self.config['lora'].get('auto_select_by_theme'):
                # Note: Theme information would need to be passed in kwargs
                theme = kwargs.get('theme', {})
                lora_info = self._select_and_load_lora(theme)
                
            # Stage 1: Generate base image
            stage1_result = self._generate_stage1(prompt, seed, params, lora_info)
            
            # Stage 2: Optional img2img refinement
            if self.config['pipeline'].get('enable_img2img_refine') and self.refiner_pipe:
                stage2_result = self._refine_stage2(
                    stage1_result['image'],
                    prompt,
                    seed,
                    params
                )
            else:
                stage2_result = stage1_result
                
            # Stage 3: Upscale 2x
            stage3_result = self._upscale_stage3(stage2_result['image_path'])
            
            # Stage 4: Final crop/adjustment to 4K
            stage4_result = self._finalize_stage4(stage3_result['image_path'])
            
            # Unload LoRA if loaded
            if lora_info:
                self._unload_lora()
                
            # Prepare final results
            duration = time.time() - start_time
            
            results = {
                'image_path': stage4_result['image_path'],
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
            
    def _generate_stage1(
        self, 
        prompt: str, 
        seed: int, 
        params: Dict[str, Any],
        lora_info: Optional[Dict[str, Any]]
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
        
        # Get dimensions from config
        width, height = self.config['generation']['dimensions']
        
        # Select scheduler
        scheduler_class = self._get_scheduler_class(params.get('scheduler'))
        self.pipe.scheduler = scheduler_class.from_config(self.pipe.scheduler.config)
        self.logger.info(f"Using scheduler: {scheduler_class.__name__}")
        
        # Prepare parameters
        sdxl_params = {
            'prompt': prompt,
            'height': height,
            'width': width,
            'generator': generator,
            'num_inference_steps': params.get('steps', 50),
            'guidance_scale': params.get('guidance_scale', 7.5),
        }
        
        # Add negative prompt for better quality
        sdxl_params['negative_prompt'] = (
            "low quality, blurry, pixelated, noisy, oversaturated, "
            "underexposed, overexposed, bad anatomy, bad proportions, "
            "watermark, signature, text, logo"
        )
        
        self.logger.info(
            f"Generating at {width}x{height} with {sdxl_params['num_inference_steps']} steps, "
            f"guidance {sdxl_params['guidance_scale']}"
        )
        
        if lora_info:
            self.logger.info(f"Using LoRA: {lora_info['name']} (weight: {lora_info['weight']:.2f})")
            
        # Generate image
        output = self.pipe(**sdxl_params)
        image = output.images[0]
        
        # Save stage 1 output
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        stage1_path = Path(f"/tmp/sdxl_stage1_{timestamp}.png")
        image.save(stage1_path, "PNG", quality=100)
        
        self.logger.info(f"Stage 1 complete: Generated at {image.size}")
        self.logger.log_vram("After Stage 1")
        
        return {
            'image': image,
            'image_path': stage1_path,
            'size': image.size,
            'scheduler': scheduler_class.__name__,
            'lora': lora_info
        }
        
    def _refine_stage2(
        self,
        input_image: Image.Image,
        prompt: str,
        seed: int,
        params: Dict[str, Any]
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
        
        # Get refinement strength
        strength_range = self.config['pipeline'].get('refine_strength_range', [0.2, 0.4])
        strength = random.uniform(*strength_range)
        
        self.logger.info(f"Refining with strength {strength:.2f}")
        
        # Refine image
        refined = self.refiner_pipe(
            prompt=prompt,
            image=input_image,
            strength=strength,
            guidance_scale=params.get('guidance_scale', 7.5),
            num_inference_steps=int(params.get('steps', 50) * 0.5),  # Fewer steps for refinement
            generator=generator
        ).images[0]
        
        # Save refined image
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        stage2_path = Path(f"/tmp/sdxl_stage2_{timestamp}.png")
        refined.save(stage2_path, "PNG", quality=100)
        
        self.logger.info(f"Stage 2 complete: Refined at {refined.size}")
        
        return {
            'image': refined,
            'image_path': stage2_path,
            'size': refined.size,
            'strength': strength
        }
        
    def _upscale_stage3(self, input_path: Path) -> Dict[str, Any]:
        """Stage 3: Upscale 2x using Real-ESRGAN
        
        Args:
            input_path: Path to input image
            
        Returns:
            Stage results
        """
        self.logger.log_stage("Stage 3", "Real-ESRGAN 2x upscaling")
        
        # Find Real-ESRGAN
        realesrgan_script = self._find_realesrgan()
        
        # Prepare paths
        temp_output_dir = Path(f"/tmp/sdxl_upscaled_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        temp_output_dir.mkdir(exist_ok=True)
        
        # Build command for 2x upscale
        if str(realesrgan_script).endswith('.py'):
            cmd = [
                sys.executable,
                str(realesrgan_script),
                "-n", "RealESRGAN_x4plus",  # Use best model even for 2x
                "-i", str(input_path),
                "-o", str(temp_output_dir),
                "--outscale", "2",  # 2x upscale
                "-t", "512",  # Smaller tile size for 2x
                "--fp32"
            ]
        else:
            cmd = [
                str(realesrgan_script),
                "-i", str(input_path),
                "-o", str(temp_output_dir),
                "-s", "2",
                "-n", "realesrgan-x4plus",
                "-t", "512"
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
            raise UpscalerError(str(input_path), e)
        except subprocess.TimeoutExpired:
            raise UpscalerError(str(input_path), Exception("Upscaling timed out"))
            
        # Find output file
        output_files = list(temp_output_dir.glob("*.png"))
        if not output_files:
            raise UpscalerError(str(input_path), FileNotFoundError("No output from Real-ESRGAN"))
            
        output_path = output_files[0]
        
        # Load and verify
        upscaled_image = Image.open(output_path)
        
        # Expected: 1920x1024 * 2 = 3840x2048
        expected_size = (
            self.config['generation']['dimensions'][0] * 2,
            self.config['generation']['dimensions'][1] * 2
        )
        
        if upscaled_image.size != expected_size:
            self.logger.warning(f"Unexpected size: {upscaled_image.size}, expected {expected_size}")
            
        self.logger.info(f"Stage 3 complete: Upscaled to {upscaled_image.size}")
        
        return {
            'image': upscaled_image,
            'image_path': output_path,
            'size': upscaled_image.size,
            'scale_factor': 2
        }
        
    def _finalize_stage4(self, input_path: Path) -> Dict[str, Any]:
        """Stage 4: Final adjustment to 4K
        
        Args:
            input_path: Path to upscaled image
            
        Returns:
            Stage results
        """
        self.logger.log_stage("Stage 4", "Finalizing to 4K")
        
        # Load upscaled image
        image = Image.open(input_path)
        
        # SDXL upscales to 3840x2048, need to crop to 3840x2160 for 16:9
        if image.size == (3840, 2048):
            # Add padding to reach 2160 height
            new_image = Image.new('RGB', (3840, 2160), (0, 0, 0))
            # Center the image vertically
            y_offset = (2160 - 2048) // 2
            new_image.paste(image, (0, y_offset))
            image = new_image
            self.logger.info("Added padding to reach 3840x2160")
        elif image.size[0] == 3840 and image.size[1] > 2160:
            # Crop if too tall
            y_offset = (image.size[1] - 2160) // 2
            image = image.crop((0, y_offset, 3840, y_offset + 2160))
            self.logger.info("Cropped to 3840x2160")
            
        # Save final image
        config = get_config()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"sdxl_4k_{timestamp}.png"
        
        output_dir = Path(config.paths.get('images_dir', '/tmp'))
        output_dir.mkdir(exist_ok=True)
        
        final_path = output_dir / filename
        image.save(final_path, "PNG", quality=100)
        
        self.logger.info(f"Stage 4 complete: Final 4K image at {image.size}")
        self.logger.info(f"Saved to: {final_path}")
        
        # Clean up temp files
        try:
            if input_path.parent.name.startswith("sdxl_upscaled_"):
                import shutil
                shutil.rmtree(input_path.parent)
        except Exception as e:
            self.logger.debug(f"Failed to clean up temp files: {e}")
            
        return {
            'image': image,
            'image_path': final_path,
            'size': image.size
        }
        
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
            'DDIMScheduler': DDIMScheduler
        }
        
        if scheduler_name and scheduler_name in scheduler_map:
            return scheduler_map[scheduler_name]
            
        # Random selection from available
        if 'scheduler_options' in self.config['generation']:
            options = self.config['generation']['scheduler_options']
            scheduler_name = random.choice(options)
            return scheduler_map.get(scheduler_name, DPMSolverMultistepScheduler)
            
        return DPMSolverMultistepScheduler
        
    def _select_and_load_lora(self, theme: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Select and load LoRA based on theme
        
        Args:
            theme: Theme information
            
        Returns:
            LoRA info if loaded
        """
        if not self.config['lora'].get('enabled'):
            return None
            
        available_loras = self.config['lora'].get('available_loras', {})
        if not available_loras:
            return None
            
        # Get theme category
        category = theme.get('category', '')
        
        # Find matching LoRA
        for lora_name, lora_data in available_loras.items():
            if category in lora_data.get('categories', []):
                # Load LoRA
                lora_path = Path(lora_data['path'])
                
                if not lora_path.exists():
                    self.logger.warning(f"LoRA file not found: {lora_path}")
                    continue
                    
                try:
                    # Get weight
                    weight_range = lora_data.get('weight_range', [0.5, 1.0])
                    weight = random.uniform(*weight_range)
                    
                    self.logger.info(f"Loading LoRA: {lora_name} with weight {weight:.2f}")
                    
                    # Load LoRA weights
                    self.pipe.load_lora_weights(str(lora_path))
                    self.pipe.fuse_lora(lora_scale=weight)
                    
                    return {
                        'name': lora_name,
                        'path': str(lora_path),
                        'weight': weight,
                        'category': category
                    }
                    
                except Exception as e:
                    self.logger.error(f"Failed to load LoRA {lora_name}: {e}")
                    
        return None
        
    def _unload_lora(self) -> None:
        """Unload current LoRA"""
        try:
            self.pipe.unfuse_lora()
            self.pipe.unload_lora_weights()
            self.logger.info("LoRA unloaded")
        except Exception as e:
            self.logger.warning(f"Failed to unload LoRA: {e}")
            
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
        
    def cleanup(self) -> None:
        """Clean up resources"""
        # Unload LoRA if loaded
        self._unload_lora()
        
        # Unregister from resource manager
        self.resource_manager.unregister_model(self.model_name)
        
        # Call parent cleanup
        super().cleanup()
        
    def get_optimal_prompt(self, theme: Dict, weather: Dict, context: Dict) -> str:
        """Get SDXL-optimized prompt
        
        Args:
            theme: Theme dictionary
            weather: Weather context
            context: Additional context
            
        Returns:
            Optimized prompt for SDXL
        """
        # SDXL works well with detailed prompts
        return ""  # Will be implemented with prompt generation
        
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