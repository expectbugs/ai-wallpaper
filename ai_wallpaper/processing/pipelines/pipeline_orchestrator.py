#!/usr/bin/env python3
"""
Pipeline Orchestrator for AI Wallpaper System
Manages model-specific pipelines at maximum quality
"""

import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Type
from abc import ABC, abstractmethod
from datetime import datetime

from ...core import get_logger, get_config
from ...core.exceptions import PipelineError, UpscalerError
from ..upscaler import get_upscaler


class PipelineStage(ABC):
    """Abstract base class for pipeline stages"""
    
    @abstractmethod
    def process(self, input_data: Any, **kwargs) -> Dict[str, Any]:
        """Process the stage
        
        Args:
            input_data: Input data for this stage
            **kwargs: Additional stage-specific parameters
            
        Returns:
            Stage result dictionary
        """
        pass
        
    @abstractmethod
    def get_name(self) -> str:
        """Get stage name"""
        pass


class FluxGenerationStage(PipelineStage):
    """FLUX generation stage"""
    
    def __init__(self, model_instance):
        self.model = model_instance
        self.logger = get_logger(model="Pipeline")
        
    def process(self, prompt: str, seed: int, params: Dict[str, Any]) -> Dict[str, Any]:
        """Run FLUX generation"""
        # The model's _generate_stage1 already handles this
        return self.model._generate_stage1(prompt, seed, params)
        
    def get_name(self) -> str:
        return "flux_generation"


class RealESRGAN8KStage(PipelineStage):
    """Real-ESRGAN 8K upscaling stage"""
    
    def __init__(self):
        self.upscaler = get_upscaler()
        self.logger = get_logger(model="Pipeline")
        
    def process(self, input_path: Path, **kwargs) -> Dict[str, Any]:
        """Upscale to 8K"""
        result = self.upscaler.upscale(
            input_path,
            scale=4,
            model_name="RealESRGAN_x4plus",
            tile_size=1024,
            fp32=True
        )
        
        return {
            'image_path': result['output_path'],
            'size': result['output_size'],
            'scale_factor': result['scale_factor']
        }
        
    def get_name(self) -> str:
        return "realesrgan_8k"


class Lanczos4KStage(PipelineStage):
    """Lanczos downsampling to 4K stage"""
    
    def __init__(self, target_size=(3840, 2160)):
        self.target_size = target_size
        self.logger = get_logger(model="Pipeline")
        
    def process(self, input_path: Path, **kwargs) -> Dict[str, Any]:
        """Downsample to 4K using Lanczos"""
        from PIL import Image
        
        # Load image
        image = Image.open(input_path)
        
        # High-quality downsample
        image_4k = image.resize(self.target_size, Image.Resampling.LANCZOS)
        
        # Save with maximum quality
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        config = get_config()
        output_dir = Path(config.paths.get('images_dir', '/home/user/ai-wallpaper/images'))
        output_dir.mkdir(exist_ok=True)
        
        output_path = output_dir / f"final_4k_{timestamp}.png"
        image_4k.save(output_path, "PNG", quality=100, optimize=False)
        
        return {
            'image_path': output_path,
            'size': image_4k.size
        }
        
    def get_name(self) -> str:
        return "lanczos_4k"


class CropStage(PipelineStage):
    """Image cropping stage for 16:9 conversion"""
    
    def __init__(self, crop_pixels: List[int]):
        self.crop_pixels = crop_pixels  # [top, bottom]
        self.logger = get_logger(model="Pipeline")
        
    def process(self, input_path: Path, **kwargs) -> Dict[str, Any]:
        """Crop image to 16:9"""
        from PIL import Image
        
        image = Image.open(input_path)
        width, height = image.size
        
        # Crop top and bottom
        top_crop = self.crop_pixels[0]
        bottom_crop = self.crop_pixels[1]
        
        cropped = image.crop((0, top_crop, width, height - bottom_crop))
        
        # Save cropped image
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        cropped_path = input_path.parent / f"stage_cropped_{timestamp}.png"
        cropped.save(cropped_path, "PNG", quality=100)
        
        return {
            'image_path': cropped_path,
            'size': cropped.size,
            'crop_applied': self.crop_pixels
        }
        
    def get_name(self) -> str:
        return "crop_16_9"


class PipelineOrchestrator:
    """Orchestrate model-specific pipelines at maximum quality"""
    
    def __init__(self, model):
        """Initialize orchestrator
        
        Args:
            model: Model instance with pipeline configuration
        """
        self.model = model
        self.logger = get_logger(model="Pipeline")
        self.stages = self._build_stages()
        
    def _build_stages(self) -> List[PipelineStage]:
        """Build pipeline stages based on model configuration
        
        Returns:
            List of pipeline stages
        """
        stages = []
        stage_names = self.model.get_pipeline_stages()
        
        for stage_name in stage_names:
            if stage_name == "flux_generation":
                stages.append(FluxGenerationStage(self.model))
            elif stage_name == "realesrgan_8k":
                stages.append(RealESRGAN8KStage())
            elif stage_name == "lanczos_4k":
                target_size = tuple(self.model.config['pipeline']['stage3_downsample'])
                stages.append(Lanczos4KStage(target_size))
            elif stage_name == "crop_16_9":
                crop_pixels = self.model.config['pipeline']['crop_pixels']
                stages.append(CropStage(crop_pixels))
            elif stage_name == "dalle_generation":
                # Model handles this internally
                pass
            elif stage_name == "dalle_crop":
                crop_pixels = self.model.config['pipeline']['crop_pixels']
                stages.append(CropStage(crop_pixels))
            elif stage_name == "dalle_upscale":
                stages.append(RealESRGAN8KStage())  # Reuse for 4x upscale
            elif stage_name == "dalle_downsample":
                target_size = tuple(self.model.config['pipeline']['final_resolution'])
                stages.append(Lanczos4KStage(target_size))
            else:
                self.logger.warning(f"Unknown stage: {stage_name}")
                
        return stages
        
    def execute(
        self,
        initial_data: Any,
        save_stages: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """Execute full pipeline
        
        Args:
            initial_data: Initial data (prompt for generation, image path for processing)
            save_stages: Whether to save intermediate images
            **kwargs: Additional parameters for stages
            
        Returns:
            Pipeline execution results
        """
        results = {
            'stages': {},
            'final_image': None,
            'metadata': {},
            'duration': 0
        }
        
        start_time = time.time()
        current_data = initial_data
        
        try:
            for i, stage in enumerate(self.stages):
                stage_name = stage.get_name()
                self.logger.log_stage(f"Stage {i+1}/{len(self.stages)}", stage_name)
                
                # Execute stage
                stage_start = time.time()
                
                if isinstance(stage, FluxGenerationStage):
                    # Special handling for generation stage
                    stage_result = stage.process(
                        initial_data['prompt'],
                        initial_data['seed'],
                        initial_data['params']
                    )
                else:
                    # Processing stages work on image paths
                    if isinstance(current_data, dict) and 'image_path' in current_data:
                        input_path = Path(current_data['image_path'])
                    else:
                        input_path = Path(current_data)
                        
                    stage_result = stage.process(input_path, **kwargs)
                    
                stage_duration = time.time() - stage_start
                
                # Store stage results
                stage_result['duration'] = stage_duration
                results['stages'][stage_name] = stage_result
                
                # Update current data for next stage
                current_data = stage_result
                
                # Save intermediate if requested
                if save_stages and 'image_path' in stage_result:
                    self._save_stage_output(stage_name, stage_result)
                    
            # Set final result
            if 'image_path' in current_data:
                results['final_image'] = current_data['image_path']
            
            results['duration'] = time.time() - start_time
            
            self.logger.info(f"Pipeline completed in {results['duration']:.1f} seconds")
            
            return results
            
        except Exception as e:
            raise PipelineError(
                self.model.name,
                f"Stage execution failed",
                e
            )
            
    def _save_stage_output(self, stage_name: str, stage_result: Dict[str, Any]) -> None:
        """Save stage output for debugging
        
        Args:
            stage_name: Name of the stage
            stage_result: Stage execution results
        """
        if 'image_path' not in stage_result:
            return
            
        from shutil import copy2
        
        source = Path(stage_result['image_path'])
        if not source.exists():
            return
            
        # Create stage output directory
        config = get_config()
        stage_dir = Path(config.paths.get('images_dir', '/tmp')) / "stages" / datetime.now().strftime("%Y%m%d_%H%M%S")
        stage_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy with stage name
        dest = stage_dir / f"{stage_name}_{source.name}"
        copy2(source, dest)
        
        self.logger.debug(f"Saved stage output: {dest}")
        
    def get_stage_count(self) -> int:
        """Get number of stages in pipeline
        
        Returns:
            Number of stages
        """
        return len(self.stages)
        
    def get_stage_names(self) -> List[str]:
        """Get list of stage names
        
        Returns:
            List of stage names
        """
        return [stage.get_name() for stage in self.stages]