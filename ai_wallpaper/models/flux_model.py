#!/usr/bin/env python3
"""
FLUX.1-dev Model Implementation
Generates images using FLUX with 3-stage pipeline: generate → 8K → 4K
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
from diffusers import FluxPipeline, FlowMatchEulerDiscreteScheduler
from PIL import Image

from .base_model import BaseImageModel
from ..core import get_logger, get_config
from ..core.exceptions import ModelNotFoundError, ModelLoadError, GenerationError, UpscalerError
from ..core.path_resolver import get_resolver
from ..processing import get_upscaler
from ..utils import get_file_manager
from .model_resolver import get_model_resolver

class FluxModel(BaseImageModel):
    """FLUX.1-dev implementation with 8K→4K supersampling pipeline"""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize FLUX model
        
        Args:
            config: Model configuration from models.yaml
        """
        super().__init__(config)
        self.pipe = None
        self.generator = None
        
    def initialize(self) -> bool:
        """Initialize FLUX model and verify requirements
        
        Returns:
            True if initialization successful
        """
        try:
            self.logger.info("Initializing FLUX.1-dev model...")
            
            # Clear GPU cache before starting
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                # Set memory fraction to prevent using all VRAM
                torch.cuda.set_per_process_memory_fraction(0.90)
                
            # Find model path
            model_path = self._find_model_path()
            
            # Load pipeline with optimizations
            self.logger.info(f"Loading FLUX pipeline from: {model_path}")
            
            self.pipe = FluxPipeline.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True
            )
            
            # CRITICAL: Ensure FlowMatchEulerDiscreteScheduler
            self.pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(
                self.pipe.scheduler.config
            )
            self.logger.info("Using FlowMatchEulerDiscreteScheduler (required for FLUX)")
            
            # Enable memory optimizations
            self.logger.info("Enabling memory optimizations...")
            self.pipe.enable_sequential_cpu_offload()
            
            # VAE optimizations
            self.pipe.vae.enable_tiling()
            self.pipe.vae.enable_slicing()
            
            # Attention slicing
            self.pipe.enable_attention_slicing(1)
            
            # Note: xFormers disabled for FLUX compatibility
            self.logger.debug("xFormers disabled for FLUX compatibility")
            
            # Clean up after loading
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Validate pipeline stages before marking as initialized
            self.validate_pipeline_stages()
                
            self._initialized = True
            self.logger.info("FLUX model initialized successfully")
            self.logger.info("Note: FLUX uses CPU offloading to minimize VRAM usage")
            self.logger.log_vram("After initialization")
            
            return True
            
        except Exception as e:
            raise ModelLoadError(self.name, e)
            
    def _find_model_path(self) -> str:
        """Find FLUX model path from configured locations
        
        Returns:
            Model path
            
        Raises:
            ModelNotFoundError: If model not found
        """
        # Get model hints from config
        model_hints = self.config.get('model_hints', [])
        if not model_hints:
            # Fallback to old config key for compatibility
            model_hints = self.config.get('model_path_priority', [])
        
        # Use ModelResolver to find the model
        resolver = get_model_resolver()
        model_path = resolver.find_model(model_hints)
        
        if model_path:
            # Verify it's a valid FLUX model directory
            if model_path.is_dir():
                # Check for required FLUX directories
                required_dirs = ['transformer', 'text_encoder', 'tokenizer']
                
                # Check if model_index.json exists
                if not (model_path / 'model_index.json').exists():
                    self.logger.debug(f"No model_index.json found at {model_path}")
                else:
                    # Check for required directories
                    missing_dirs = [d for d in required_dirs if not (model_path / d).exists()]
                    if missing_dirs:
                        self.logger.debug(f"Missing required directories: {missing_dirs}")
                    else:
                        # Check if transformer has any model files
                        transformer_dir = model_path / 'transformer'
                        if transformer_dir.exists():
                            has_model_files = any(
                                f.name.endswith('.safetensors') or f.name.endswith('.bin')
                                for f in transformer_dir.iterdir()
                                if f.is_file() or f.is_symlink()
                            )
                            
                            if has_model_files:
                                self.logger.info(f"Found valid FLUX model at: {model_path}")
                                return str(model_path)
                            else:
                                self.logger.debug(f"No model files found in transformer directory at {model_path}")
        
        # If no local model found, try HuggingFace ID
        if model_hints:
            # The first hint is usually the HF repo ID
            hf_id = model_hints[0]
            if '/' in hf_id:  # Looks like a HF repo ID
                self.logger.warning("No local model found, will download from HuggingFace")
                self.logger.warning("This may take 15-30 minutes on first run!")
                return hf_id
                
        # If we get here, no model was found
        searched_paths = []
        if resolver.search_paths:
            for search_path in resolver.search_paths:
                searched_paths.append(str(search_path))
                
        raise ModelNotFoundError("FLUX.1-dev", searched_paths)
        
    def generate(self, prompt: str, seed: Optional[int] = None, **kwargs) -> Dict[str, Any]:
        """Generate image using FLUX pipeline
        
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
        
        # Check disk space before generation
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
            # Stage 1: Generate base image
            stage1_result = self._generate_stage1(prompt, seed, params, temp_files)
            
            # Save stage 1 if requested
            if self._should_save_stages(params):
                stage1_result['saved_path'] = self._save_intermediate(
                    stage1_result['image'], 
                    "stage1_generation",
                    prompt
                )
            
            # Stage 2: Upscale to 8K
            stage2_result = self._upscale_stage2(stage1_result['image_path'], temp_dirs)
            
            # Save stage 2 if requested
            if self._should_save_stages(params):
                stage2_result['saved_path'] = self._save_intermediate(
                    stage2_result['image'],
                    "stage2_8k_upscaled",
                    prompt
                )
            
            # Stage 3: Downsample to 4K
            stage3_result = self._downsample_stage3(stage2_result['image_path'])
            
            # Prepare final results
            duration = time.time() - start_time
            
            results = {
                'image_path': stage3_result['image_path'],
                'metadata': {
                    'prompt': prompt,
                    'seed': seed,
                    'model': 'FLUX.1-dev',
                    'parameters': params,
                    'duration': duration,
                    'timestamp': datetime.now().isoformat()
                },
                'stages': {
                    'stage1_generation': stage1_result,
                    'stage2_upscale': stage2_result,
                    'stage3_downsample': stage3_result
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
            
    def _generate_stage1(self, prompt: str, seed: int, params: Dict[str, Any], temp_files: List[Path]) -> Dict[str, Any]:
        """Stage 1: Generate base image at 1920x1088
        
        Args:
            prompt: Text prompt
            seed: Random seed
            params: Generation parameters
            
        Returns:
            Stage results
        """
        self.logger.log_stage("Stage 1", "FLUX generation at 1920x1088")
        
        # Set up generator
        generator = torch.Generator(device="cpu").manual_seed(seed)
        
        # Get dimensions from config
        width, height = self.config['generation']['dimensions']
        
        # Extract FLUX-specific parameters
        flux_params = {
            'prompt': prompt,
            'height': height,
            'width': width,
            'generator': generator,
            'guidance_scale': params.get('guidance_scale', 3.5),
            'num_inference_steps': params.get('steps', 100),
            'max_sequence_length': params.get('max_sequence_length', 512)
        }
        
        # Clear cache before generation
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        self.logger.info(f"Generating with seed: {seed}")
        self.logger.debug(f"Parameters: steps={flux_params['num_inference_steps']}, "
                         f"guidance={flux_params['guidance_scale']}")
        
        # Generate image
        with torch.inference_mode():
            output = self.pipe(**flux_params)
            image = output.images[0]
            
        # Verify dimensions
        if image.size != (width, height):
            raise GenerationError(
                self.name, 
                "Stage 1", 
                ValueError(f"Wrong dimensions {image.size}, expected ({width}, {height})")
            )
            
        # Save stage 1 output
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        resolver = get_resolver()
        temp_dir = resolver.get_temp_dir() / 'ai-wallpaper'
        temp_dir.mkdir(parents=True, exist_ok=True)
        stage1_path = temp_dir / f"flux_stage1_{timestamp}.png"
        image.save(stage1_path, "PNG", quality=100)
        # Track for cleanup
        temp_files.append(stage1_path)
        
        self.logger.info(f"Stage 1 complete: Generated at {image.size}")
        self.logger.log_vram("After Stage 1")
        
        return self._standardize_stage_result(
            image_path=stage1_path,
            image=image,
            size=image.size
        )
        
    def _upscale_stage2(self, input_path: Path, temp_dirs: List[Path]) -> Dict[str, Any]:
        """Stage 2: Upscale to 8K using Real-ESRGAN
        
        Args:
            input_path: Path to input image
            
        Returns:
            Stage results
        """
        self.logger.log_stage("Stage 2", "Real-ESRGAN upscaling to 8K")
        
        # Use centralized upscaler
        upscaler = get_upscaler()
        
        # Get upscale settings from config
        pipeline_config = self.config.get('pipeline', {})
        upscale_config = pipeline_config.get('stages', {}).get('upscale', {})
        
        # Use config values with defaults
        scale = upscale_config.get('scale', 4)
        model_name = upscale_config.get('model', 'RealESRGAN_x4plus')
        tile_size = upscale_config.get('tile_size', 1024)
        fp32 = upscale_config.get('fp32', True)
        
        # Perform upscaling
        result = upscaler.upscale(
            input_path,
            scale=scale,
            model_name=model_name,
            tile_size=tile_size,
            fp32=fp32
        )
        
        # Track the upscaled directory for cleanup
        output_path = Path(result['output_path'])
        if output_path.parent.name.startswith('upscaled_'):
            temp_dirs.append(output_path.parent)
        
        # Load image for consistency with rest of pipeline
        with Image.open(result['output_path']) as temp_image:
            upscaled_image = temp_image.copy()  # Create a copy that persists
        
        self.logger.info(f"Stage 2 complete: Upscaled to {result['output_size']}")
        
        return self._standardize_stage_result(
            image_path=result['output_path'],
            image=upscaled_image,
            size=result['output_size'],
            scale_factor=result['scale_factor']
        )
        
    def _downsample_stage3(self, input_path: Path) -> Dict[str, Any]:
        """Stage 3: Downsample to 4K using Lanczos
        
        Args:
            input_path: Path to 8K image
            
        Returns:
            Stage results
        """
        self.logger.log_stage("Stage 3", "Lanczos supersampling to 4K")
        
        # Load 8K image
        with Image.open(input_path) as temp_image:
            # Create a copy for processing
            image_8k = temp_image.copy()
        
        # Target 4K dimensions
        target_size = tuple(self.config['pipeline']['stage3_downsample'])
        
        # High-quality downsample
        image_4k = image_8k.resize(target_size, Image.Resampling.LANCZOS)
        
        # Save final image
        config = get_config()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"flux_4k_{timestamp}.png"
        
        # Use configured output path
        resolver = get_resolver()
        output_dir = Path(config.paths.get('images_dir', str(resolver.get_data_dir() / 'wallpapers')))
        output_dir.mkdir(exist_ok=True)
        
        final_path = output_dir / filename
        image_4k.save(final_path, "PNG", quality=100)
        
        self.logger.info(f"Stage 3 complete: Final 4K image at {image_4k.size}")
        self.logger.info(f"Saved to: {final_path}")
        
        # Cleanup now handled in finally block
        
        return self._standardize_stage_result(
            image_path=final_path,
            image=image_4k,
            size=image_4k.size
        )
        
        
    def get_optimal_prompt(self, theme: Dict, weather: Dict, context: Dict) -> str:
        """Get FLUX-optimized prompt
        
        Args:
            theme: Theme dictionary
            weather: Weather context
            context: Additional context
            
        Returns:
            Optimized prompt for FLUX
        """
        # FLUX handles quality internally, so we don't add extra quality keywords
        # Return empty string to use DeepSeek prompter
        return ""
        
    def get_pipeline_stages(self) -> List[str]:
        """Return pipeline stages for FLUX
        
        Returns:
            List of stage names
        """
        return [
            "flux_generation",      # 1920x1088
            "flux_upscale",        # 4x to 7680x4352  
            "flux_downsample"      # Downsample to 3840x2160
        ]
        
    def validate_environment(self) -> Tuple[bool, str]:
        """Validate FLUX can run in current environment
        
        Returns:
            Tuple of (is_valid, message)
        """
        # Check CUDA availability
        if not torch.cuda.is_available():
            return False, "CUDA is required for FLUX model"
            
        # Check VRAM
        device_props = torch.cuda.get_device_properties(0)
        vram_gb = device_props.total_memory / 1024**3
        
        if vram_gb < 20:
            return False, f"Insufficient VRAM: {vram_gb:.1f}GB, need at least 20GB"
            
        # Check Real-ESRGAN availability
        try:
            upscaler = get_upscaler()
        except UpscalerError:
            return False, "Real-ESRGAN is required but not found"
            
        return True, "Environment validated for FLUX"
        
    def supports_feature(self, feature: str) -> bool:
        """Check if FLUX supports a feature
        
        Args:
            feature: Feature name
            
        Returns:
            True if supported
        """
        features = {
            '8k_pipeline': True,
            'scheduler_selection': False,  # Must use FlowMatchEuler
            'custom_dimensions': False,    # Fixed at 1920x1088
            'lora': False,
            'img2img': False,
            'controlnet': False
        }
        return features.get(feature, super().supports_feature(feature))
        
    def get_resource_requirements(self) -> Dict[str, Any]:
        """Return FLUX resource requirements
        
        Returns:
            Resource requirements
        """
        return {
            'vram_gb': 24,  # RTX 3090 minimum
            'disk_gb': 30,  # Model is ~25GB
            'time_minutes': 15
        }
        
        
