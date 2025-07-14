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

class BaseImageModel(ABC):
    """Abstract base class for all image generation models"""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize base model
        
        Args:
            config: Model configuration from models.yaml
        """
        self.config = config
        self.name = config.get('display_name', self.__class__.__name__)
        self.model_name = config.get('name', self.name.lower().replace(' ', '_'))
        self._initialized = False
        self.logger = get_logger(model=self.name)
        
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
        """Clean up resources (VRAM, etc)"""
        self.logger.info("Cleaning up model resources...")
        
        # Clean up any PyTorch models
        if hasattr(self, 'pipe'):
            self.logger.debug("Deleting pipeline...")
            del self.pipe
            
        # Force garbage collection
        gc.collect()
        
        # Clear CUDA cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            self.logger.log_vram("After cleanup")
            
        self._initialized = False
        self.logger.info("Cleanup complete")
        
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
        
    def get_generation_params(self, **kwargs) -> Dict[str, Any]:
        """Get final generation parameters with defaults
        
        Args:
            **kwargs: User-provided parameters
            
        Returns:
            Merged parameters dictionary
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
            self.logger.warning(f"Failed to save metadata: {e}")
            
    def log_generation_start(self, prompt: str, params: Dict[str, Any]) -> None:
        """Log generation start with parameters
        
        Args:
            prompt: Generation prompt
            params: Generation parameters
        """
        self.logger.log_separator()
        self.logger.info(f"Starting {self.name} generation")
        self.logger.info(f"Prompt: {prompt[:100]}..." if len(prompt) > 100 else f"Prompt: {prompt}")
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