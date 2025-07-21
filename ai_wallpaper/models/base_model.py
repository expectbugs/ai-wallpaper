#!/usr/bin/env python3
"""
Base Model Abstraction for AI Image Generation
All image generation models inherit from this base class
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional, List
from pathlib import Path
import torch
import gc
import os
from datetime import datetime

from ..core.logger import get_logger
from ..core.exceptions import ModelError, GenerationError, ResourceError
from ..core.config_manager import get_config
from ..utils.file_manager import get_file_manager
from ..core.resolution_manager import ResolutionManager, ResolutionConfig
from PIL import Image

class BaseImageModel(ABC):
    """Abstract base class for all image generation models"""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize base model with resolution support
        
        Args:
            config: Model configuration from models.yaml
        """
        self.config = config
        self.name = config.get('display_name', self.__class__.__name__)
        self.model_name = config.get('name', self.name.lower().replace(' ', '_'))
        self._initialized = False
        self.logger = get_logger(model=self.name)
        
        # Initialize resolution manager
        self.resolution_manager = ResolutionManager()
        self.resolution_manager.logger = self.logger
        
        # Track resource usage
        self._vram_usage = 0
        self._load_time = None
        
    @abstractmethod
    def initialize(self) -> bool:
        """Initialize model and verify requirements
        
        Returns:
            True if initialization successful
            
        Raises:
            ModelError: If initialization fails
        """
        pass
        
    @abstractmethod
    def generate(self, prompt: str, seed: Optional[int] = None, **kwargs) -> Dict[str, Any]:
        """Generate image from prompt
        
        Args:
            prompt: Text prompt for generation
            seed: Random seed for reproducibility
            **kwargs: Model-specific parameters
            
        Returns:
            Dictionary containing:
                - image_path: Path to generated image
                - metadata: Generation metadata
                - stages: Pipeline stage results
                
        Raises:
            GenerationError: If generation fails
        """
        pass
        
    @abstractmethod
    def get_optimal_prompt(self, theme: Dict, weather: Dict, context: Dict) -> str:
        """Get model-optimized prompt from theme and context
        
        Args:
            theme: Selected theme dictionary
            weather: Weather context
            context: Additional context (date, time, etc.)
            
        Returns:
            Optimized prompt string for this model
        """
        pass
        
    @abstractmethod
    def get_pipeline_stages(self) -> List[str]:
        """Return list of pipeline stages for this model
        
        Returns:
            List of stage names in order
        """
        pass
        
    @abstractmethod
    def validate_environment(self) -> Tuple[bool, str]:
        """Validate model can run in current environment
        
        Returns:
            Tuple of (is_valid, message)
        """
        pass
        
    def cleanup(self) -> None:
        """Clean up resources (VRAM, etc) in optimal order"""
        self.logger.info("Cleaning up model resources...")
        
        # Step 1: Call pre-cleanup hook for subclasses
        self._pre_cleanup()
        
        # Step 2: Delete pipeline and model objects
        self._delete_models()
        
        # Step 3: Force garbage collection (must be after deletions)
        gc.collect()
        
        # Step 4: Clear CUDA cache (must be after gc.collect)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            self.logger.log_vram("After cleanup")
            
        # Step 5: Mark as uninitialized
        self._initialized = False
        self.logger.info("Cleanup complete")
        
    def _pre_cleanup(self) -> None:
        """Hook for subclasses to perform cleanup before main cleanup
        
        Override this to unload LoRA, unregister from managers, etc.
        """
        pass
        
    def _delete_models(self) -> None:
        """Delete all model objects to free memory"""
        # Delete pipeline
        if hasattr(self, 'pipe'):
            self.logger.debug("Deleting pipeline...")
            del self.pipe
            
        # Delete any other model attributes (for subclasses)
        model_attrs = ['refiner_pipe', 'vae', 'text_encoder', 'text_encoder_2', 
                      'tokenizer', 'tokenizer_2', 'scheduler']
        for attr in model_attrs:
            if hasattr(self, attr):
                self.logger.debug(f"Deleting {attr}...")
                delattr(self, attr)
        
    def validate_pipeline_stages(self) -> None:
        """Validate that all declared pipeline stages have implementations
        
        This should be called during model initialization to ensure
        all stages returned by get_pipeline_stages() can be executed.
        
        Raises:
            ModelError: If any pipeline stage lacks implementation
        """
        stages = self.get_pipeline_stages()
        if not stages:
            return  # No stages is valid for some models
            
        self.logger.debug(f"Validating {len(stages)} pipeline stages")
        
        # Map of stage patterns to check
        stage_patterns = {
            # FLUX patterns
            'flux_generation': ['_generate_stage1', 'generate'],
            'flux_upscale': ['_upscale_stage2', '_upscale'],
            'flux_downsample': ['_downsample_stage3', '_downsample'],
            'flux_refine': ['_refine_stage3', '_refine'],
            'flux_finalize': ['_finalize_stage4', '_finalize'],
            'realesrgan_8k': ['_upscale_stage2', '_upscale'],
            'lanczos_4k': ['_downsample_stage3', '_downsample', '_finalize'],
            
            # SDXL patterns  
            'sdxl_generation': ['_generate_stage1', 'generate'],
            'sdxl_img2img_refine': ['_refine_stage2', '_refine'],
            'realesrgan_2x': ['_upscale_stage3', '_upscale'],
            'finalize_4k': ['_finalize_stage4', '_finalize'],
            
            # DALLE/GPT patterns
            'dalle_generation': ['_generate_stage1', 'generate', '_generate'],
            'gpt_generation': ['_generate_responses_api', '_generate_direct_api', 'generate', '_generate'],
            'crop_to_16_9': ['_crop_stage2', '_crop'],
            'realesrgan_4x': ['_upscale_stage3', '_upscale'],
            'lanczos_4k': ['_downsample_stage4', '_finalize_stage4', '_downsample'],
            'upscale_4k': ['_upscale', 'upscale'],
            
            # Generic patterns
            'generate': ['generate', '_generate'],
            'upscale': ['_upscale', 'upscale'],
            'refine': ['_refine', 'refine'],
            'finalize': ['_finalize', 'finalize']
        }
        
        missing_stages = []
        
        for stage in stages:
            # Check if stage has known implementation patterns
            patterns = stage_patterns.get(stage, [])
            
            # Also check for direct method match
            direct_method = f"_execute_{stage}"
            if hasattr(self, direct_method):
                continue
                
            # Check known patterns
            found = False
            for pattern in patterns:
                if hasattr(self, pattern):
                    found = True
                    break
                    
            # Check for any method containing the stage name
            if not found:
                stage_snake = stage.replace('-', '_').replace(' ', '_')
                for attr in dir(self):
                    if attr.startswith('_') and stage_snake in attr:
                        found = True
                        break
                    # Also check partial matches for stages like "lanczos_4k" -> "_downsample"
                    if attr.startswith('_') and any(part in attr for part in stage_snake.split('_')):
                        found = True
                        break
                        
            if not found:
                missing_stages.append(stage)
                
        if missing_stages:
            error_msg = (
                f"Pipeline validation failed! The following stages are declared "
                f"but have no implementation: {missing_stages}. "
                f"Each stage in get_pipeline_stages() must have a corresponding method."
            )
            self.logger.error(error_msg)
            raise ModelError(self.name, error_msg)
            
        self.logger.debug("All pipeline stages validated successfully")
        
    def get_resource_requirements(self) -> Dict[str, Any]:
        """Return estimated resource requirements
        
        Returns:
            Dictionary of resource requirements
        """
        return {
            'vram_gb': 24,  # Default, override in subclasses
            'disk_gb': 20,
            'time_minutes': 15
        }
        
    def check_resources(self) -> Tuple[bool, str]:
        """Check if sufficient resources are available
        
        Returns:
            Tuple of (has_resources, message)
        """
        requirements = self.get_resource_requirements()
        
        # Check VRAM if using CUDA
        if torch.cuda.is_available():
            device_props = torch.cuda.get_device_properties(0)
            total_vram_gb = device_props.total_memory / 1024**3
            used_vram_gb = torch.cuda.memory_allocated() / 1024**3
            free_vram_gb = total_vram_gb - used_vram_gb
            
            required_vram = requirements['vram_gb']
            if free_vram_gb < required_vram * 1.1:  # 10% buffer
                return False, (
                    f"Insufficient VRAM: {free_vram_gb:.1f}GB free, "
                    f"{required_vram}GB required"
                )
                
        # Check disk space
        stat = os.statvfs(os.path.expanduser("~"))
        free_disk_gb = (stat.f_bavail * stat.f_frsize) / 1024**3
        
        required_disk = requirements['disk_gb']
        if free_disk_gb < required_disk:
            return False, (
                f"Insufficient disk space: {free_disk_gb:.1f}GB free, "
                f"{required_disk}GB required"
            )
            
        return True, "Resources available"
    
    def check_disk_space_for_generation(self, no_upscale: bool = False) -> None:
        """Check if sufficient disk space is available for generation
        
        Args:
            no_upscale: Whether upscaling will be skipped
            
        Raises:
            ModelError: If insufficient disk space
        """
        # Calculate required space based on generation pipeline
        # Base image: ~5MB, upscaled stages: ~100MB each, final 4K: ~50MB
        required_mb = 200  # Base requirement
        
        if not no_upscale:
            # Add space for upscaling stages
            required_mb += 500  # Intermediate upscaled files can be large
            
        # Add buffer for temp files and safety margin
        required_mb = int(required_mb * 1.5)  # 50% safety margin
        
        # Check available space in temp directory
        import tempfile
        temp_dir = tempfile.gettempdir()
        stat = os.statvfs(temp_dir)
        free_mb = (stat.f_bavail * stat.f_frsize) / (1024 * 1024)
        
        if free_mb < required_mb:
            error_msg = (
                f"Insufficient disk space in {temp_dir}: "
                f"{free_mb:.0f}MB free, {required_mb}MB required. "
                f"Please free up disk space before generating images."
            )
            self.logger.error(error_msg)
            raise ModelError(self.name, error_msg)
            
        # Also check output directory
        config = get_config()
        output_dir = Path(config.paths.get('images_dir', temp_dir))
        
        # If output dir is on a different filesystem, check it too
        try:
            output_stat = os.statvfs(output_dir)
            output_free_mb = (output_stat.f_bavail * output_stat.f_frsize) / (1024 * 1024)
            
            # Only need space for final image in output dir
            if output_free_mb < 100:  # 100MB for final 4K image
                error_msg = (
                    f"Insufficient disk space in output directory {output_dir}: "
                    f"{output_free_mb:.0f}MB free, 100MB required for final image."
                )
                self.logger.error(error_msg)
                raise ModelError(self.name, error_msg)
        except Exception as e:
            # Output dir might not exist yet, which is OK - it will be created later
            self.logger.debug(f"Could not check output directory: {e}. Will be created if needed.")
            
        self.logger.debug(
            f"Disk space check passed: {free_mb:.0f}MB free in temp, "
            f"{required_mb}MB required"
        )
        
    def supports_feature(self, feature: str) -> bool:
        """Check if model supports a feature
        
        Args:
            feature: Feature name to check
            
        Returns:
            True if feature is supported
        """
        features = {
            'lora': False,
            'img2img': False,
            'controlnet': False,
            'scheduler_selection': True,
            'custom_dimensions': False,
            '8k_pipeline': False
        }
        return features.get(feature, False)
        
    def get_supported_schedulers(self) -> List[str]:
        """Get list of supported schedulers
        
        Returns:
            List of scheduler class names
        """
        return []  # Override in subclasses
        
    def get_generation_params(self, target_resolution: Optional[Tuple[int, int]] = None, **kwargs) -> Dict[str, Any]:
        """Get final generation parameters with defaults and resolution support
        
        Args:
            target_resolution: Target resolution tuple (width, height)
            **kwargs: User-provided parameters
            
        Returns:
            Merged parameters dictionary with resolution support
        """
        # Start with model defaults
        params = self.config.get('generation', {}).copy()
        
        # Handle range parameters (random selection)
        if 'steps_range' in params and 'steps' not in kwargs:
            import random
            params['steps'] = random.randint(*params['steps_range'])
            del params['steps_range']
            
        if 'guidance_range' in params and 'guidance_scale' not in kwargs:
            import random
            params['guidance_scale'] = random.uniform(*params['guidance_range'])
            del params['guidance_range']
            
        # Merge with user parameters
        params.update(kwargs)
        
        # Handle resolution parameters if target_resolution is provided
        if target_resolution:
            # Check if user wants exact resolution (not upscaled to default)
            config = get_config()
            default_final = tuple(config.settings.get('output', {}).get('final_resolution', [3840, 2160]))
            
            # If user specifies a resolution smaller than or equal to default, respect it exactly
            if (target_resolution[0] <= default_final[0] and 
                target_resolution[1] <= default_final[1]):
                # User wants THIS resolution, not upscaled to 4K
                params['skip_default_upscale'] = True
                self.logger.info(f"User requested specific resolution {target_resolution} - will not upscale to default {default_final}")
            
            # Calculate optimal generation size
            optimal_size = self.resolution_manager.get_optimal_generation_size(
                target_resolution, 
                self.model_name
            )
            
            # Calculate aspect ratios for strategy planning
            generation_aspect = optimal_size[0] / optimal_size[1]
            target_aspect = target_resolution[0] / target_resolution[1]
            
            params['generation_size'] = optimal_size
            params['target_resolution'] = target_resolution
            params['generation_aspect'] = generation_aspect
            params['target_aspect'] = target_aspect
            
            # Calculate complete processing strategy
            params['upscale_strategy'] = self.resolution_manager.calculate_upscale_strategy(
                optimal_size,
                target_resolution,
                generation_aspect,
                target_aspect
            )
            
            # Log the complete strategy
            self.logger.info(f"Generation strategy for {target_resolution[0]}x{target_resolution[1]}:")
            self.logger.info(f"  1. Generate at: {optimal_size}")
            for i, step in enumerate(params['upscale_strategy']):
                self.logger.info(f"  {i+2}. {step['description']}")
        
        return params
        
    def save_metadata(self, image_path: Path, metadata: Dict[str, Any]) -> None:
        """Save generation metadata alongside image
        
        Args:
            image_path: Path to generated image
            metadata: Generation metadata
        """
        import json
        
        metadata_path = image_path.with_suffix('.json')
        
        # Add base metadata
        metadata.update({
            'model': self.name,
            'model_config': self.config.get('display_name'),
            'timestamp': datetime.now().isoformat(),
            'version': get_config().settings.get('app', {}).get('version', 'unknown')
        })
        
        try:
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            self.logger.debug(f"Saved metadata to {metadata_path}")
        except Exception as e:
            # Following "fail loud" philosophy - metadata is important for tracking
            error_msg = f"Failed to save metadata to {metadata_path}: {e}"
            self.logger.error(error_msg)
            raise ModelError(self.name, f"Metadata save failed: {e}")
            
    def log_generation_start(self, prompt: str, params: Dict[str, Any]) -> None:
        """Log generation start with parameters
        
        Args:
            prompt: Generation prompt
            params: Generation parameters
        """
        self.logger.log_separator()
        self.logger.info(f"Starting {self.name} generation")
        self.logger.info(f"Prompt: {prompt}")
        self.logger.debug(f"Parameters: {params}")
        self.logger.log_vram("Before generation")
        
    def log_generation_complete(self, output_path: Path, duration: float) -> None:
        """Log generation completion
        
        Args:
            output_path: Path to generated image
            duration: Generation duration in seconds
        """
        self.logger.info(f"Generation complete in {duration:.1f} seconds")
        self.logger.info(f"Output saved to: {output_path}")
        self.logger.log_vram("After generation")
        self.logger.log_separator()
        
    def ensure_initialized(self) -> None:
        """Ensure model is initialized before use
        
        Raises:
            ModelError: If not initialized
        """
        if not self._initialized:
            self.logger.info(f"Initializing {self.name}...")
            success = self.initialize()
            if not success:
                raise ModelError(f"Failed to initialize {self.name}")
            self._initialized = True
            
    def validate_prompt(self, prompt: str) -> str:
        """Validate and clean prompt
        
        Args:
            prompt: Input prompt
            
        Returns:
            Cleaned prompt
            
        Raises:
            ValueError: If prompt is invalid
        """
        if not prompt or not isinstance(prompt, str):
            raise ValueError("Prompt must be a non-empty string")
            
        # Clean prompt
        prompt = prompt.strip()
        
        # Validate for problematic characters that can break generation
        problematic_chars = {
            '\n': 'newline',
            '\r': 'carriage return',
            '\t': 'tab',
            '\x00': 'null character',
            '\x0b': 'vertical tab',
            '\x0c': 'form feed'
        }
        
        for char, name in problematic_chars.items():
            if char in prompt:
                # Replace problematic whitespace with regular spaces
                prompt = prompt.replace(char, ' ')
                self.logger.warning(f"Replaced {name} characters in prompt with spaces")
        
        # Remove multiple consecutive spaces
        import re
        prompt = re.sub(r'\s+', ' ', prompt)
        
        # Escape backslashes if not already escaped
        # This prevents issues with string interpretation
        if '\\' in prompt and '\\\\' not in prompt:
            prompt = prompt.replace('\\', '\\\\')
            self.logger.warning("Escaped backslashes in prompt")
            
        # Validate quotes are balanced
        single_quotes = prompt.count("'")
        double_quotes = prompt.count('"')
        
        if single_quotes % 2 != 0:
            self.logger.warning(f"Unbalanced single quotes in prompt ({single_quotes} quotes found)")
            self.logger.info("Note: Unbalanced quotes are allowed but may affect generation")
            
        if double_quotes % 2 != 0:
            self.logger.warning(f"Unbalanced double quotes in prompt ({double_quotes} quotes found)")
            self.logger.info("Note: Unbalanced quotes are allowed but may affect generation")
            
        # Check for other potentially problematic patterns
        if prompt.count('(') != prompt.count(')'):
            self.logger.warning("Unbalanced parentheses in prompt - this may affect generation")
            
        if prompt.count('[') != prompt.count(']'):
            self.logger.warning("Unbalanced square brackets in prompt - this may affect generation")
            
        # Remove any remaining control characters
        # Control characters are in the range 0x00-0x1F and 0x7F-0x9F
        cleaned_prompt = ''.join(char for char in prompt if ord(char) >= 32 or char == ' ')
        
        if cleaned_prompt != prompt:
            self.logger.warning("Removed control characters from prompt")
            prompt = cleaned_prompt
        
        # Final strip to remove any leading/trailing spaces from cleaning
        prompt = prompt.strip()
        
        # Ensure prompt is not empty after cleaning
        if not prompt:
            raise ValueError("Prompt became empty after cleaning special characters")
        
        # Do NOT check or truncate length - word limits are just guidelines
        # Models like FLUX use T5 which can handle long prompts
            
        return prompt
        
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information for display
        
        Returns:
            Model information dictionary
        """
        return {
            'name': self.name,
            'class': self.__class__.__name__,
            'enabled': self.config.get('enabled', False),
            'initialized': self._initialized,
            'supports': {
                feature: self.supports_feature(feature)
                for feature in ['lora', 'img2img', 'controlnet', '8k_pipeline']
            },
            'resources': self.get_resource_requirements(),
            'pipeline_stages': self.get_pipeline_stages()
        }
        
    def _should_save_stages(self, params: Dict[str, Any]) -> bool:
        """Check if intermediate stages should be saved
        
        Args:
            params: Generation parameters
            
        Returns:
            True if stages should be saved
        """
        # Check parameter override first
        if 'save_stages' in params:
            return params['save_stages']
            
        # Check model config
        pipeline_config = self.config.get('pipeline', {})
        if 'save_intermediates' in pipeline_config:
            return pipeline_config['save_intermediates']
            
        # Check system config
        config = get_config()
        if config.system and 'generation' in config.system:
            return config.system['generation'].get('save_intermediate_stages', False)
            
        return False
        
    def _save_intermediate(self, image: Image.Image, stage_name: str, prompt: str) -> Path:
        """Save intermediate stage image
        
        Args:
            image: PIL Image to save
            stage_name: Name of the stage
            prompt: Generation prompt for filename
            
        Returns:
            Path where image was saved
        """
        config = get_config()
        file_manager = get_file_manager()
        
        # Get intermediate directory from config
        if config.system and 'generation' in config.system:
            intermediate_dir = config.system['generation'].get('intermediate_dir', 'stages')
        else:
            intermediate_dir = 'stages'
            
        # Create stage directory
        stage_dir = file_manager.get_images_dir() / intermediate_dir
        stage_dir.mkdir(parents=True, exist_ok=True)
        
        # Create filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prompt_excerpt = file_manager.sanitize_filename(prompt[:50])
        filename = f"{self.model_name}_{stage_name}_{timestamp}_{prompt_excerpt}.png"
        
        path = stage_dir / filename
        # Save intermediate with lossless PNG
        from ..utils import save_lossless_png
        save_lossless_png(image, path)
        
        self.logger.info(f"Saved intermediate stage: {path}")
        return path
        
    def _standardize_stage_result(self, 
                                 image_path: Path,
                                 image: Optional[Image.Image] = None,
                                 size: Optional[Tuple[int, int]] = None,
                                 **extras) -> Dict[str, Any]:
        """Standardize stage result dictionary format
        
        Args:
            image_path: Path to the stage output image
            image: PIL Image object (optional, will be loaded if not provided)
            size: Image dimensions (optional, will be determined if not provided)
            **extras: Additional stage-specific data
            
        Returns:
            Standardized stage result dictionary with consistent keys
        """
        # Load image if not provided
        if image is None and image_path.exists():
            with Image.open(image_path) as img:
                image = img.copy()
                
        # Get size if not provided
        if size is None:
            if image:
                size = image.size
            elif image_path.exists():
                with Image.open(image_path) as img:
                    size = img.size
            else:
                size = (0, 0)  # Unknown size
                
        # Build standardized result
        result = {
            'image_path': image_path,
            'size': size,
            'has_image_object': image is not None
        }
        
        # Include image object if available
        if image is not None:
            result['image'] = image
            
        # Add any extra stage-specific data
        if extras:
            result['extras'] = extras
            
        return result