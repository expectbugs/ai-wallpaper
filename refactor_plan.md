# Ultimate AI Wallpaper System Refactor Plan v2.0

## Executive Summary

This plan transforms the current monolithic AI wallpaper system (5 scripts, ~5000 lines) into a professional, modular framework with:
- **90% code reduction** through elimination of duplication
- **Unified CLI** with intelligent model selection
- **Maximum quality always** - no compromises
- **Centralized configuration** with validation
- **Extensible architecture** for easy model addition
- **Clean break from legacy** - fresh start

## Architecture Overview

### Directory Structure
```
ai_wallpaper/
├── config/
│   ├── models.yaml           # Model configurations with maximum quality settings
│   ├── paths.yaml           # System paths (no more hardcoding!)
│   ├── themes.yaml          # Theme database (converted from .txt)
│   ├── weather.yaml         # Weather API configuration
│   └── schemas/            # JSON schemas for validation
│       ├── model_schema.json
│       └── theme_schema.json
├── core/
│   ├── __init__.py
│   ├── config_manager.py    # Configuration loading with validation
│   ├── logger.py           # Unified logging (fail loud!)
│   ├── weather.py          # Weather context (deduplicated)
│   ├── wallpaper.py        # XFCE4 wallpaper management
│   └── exceptions.py       # Custom exceptions for loud failures
├── models/
│   ├── __init__.py
│   ├── base_model.py       # Abstract base with resource management
│   ├── flux_model.py       # FLUX-Dev (1920x1088→8K→4K)
│   ├── dalle_model.py      # DALL-E 3 (1792x1024→crop→4x→4K)
│   ├── gpt_image_model.py  # GPT-Image-1 (both variants)
│   └── sdxl_model.py       # SDXL+LoRA (1920x1080→2x→4K)
├── prompt/
│   ├── __init__.py
│   ├── base_prompter.py    # Abstract prompt interface
│   ├── deepseek_prompter.py # DeepSeek-r1:14b integration
│   ├── gpt_prompter.py     # GPT-4o direct prompting
│   ├── theme_selector.py   # Enhanced theme selection
│   └── prompt_optimizer.py # Model-specific prompt formatting
├── processing/
│   ├── __init__.py
│   ├── pipelines/          # Model-specific quality pipelines
│   │   ├── flux_pipeline.py    # 3-stage: generate→8K→4K
│   │   ├── dalle_pipeline.py   # Crop→upscale→downsample
│   │   ├── gpt_pipeline.py     # Similar to DALL-E
│   │   └── sdxl_pipeline.py    # 2-stage: generate→2x
│   └── upscaler.py         # Real-ESRGAN integration
├── cli/
│   ├── __init__.py
│   ├── main.py            # Click-based CLI entry point
│   ├── commands/          # Command implementations
│   │   ├── generate.py    # Main generation command
│   │   ├── test.py       # Testing commands
│   │   ├── config.py     # Configuration management
│   │   └── models.py     # Model information/testing
│   └── validators.py      # Input validation
├── utils/
│   ├── __init__.py
│   ├── resource_manager.py # VRAM/memory management
│   ├── file_manager.py    # Safe file operations
│   └── random_selector.py # Intelligent random selection
└── setup.py              # Package installation
```

## Core Components

### 1. Configuration System

**models.yaml** (with improved settings):
```yaml
models:
  flux:
    class: FluxModel
    enabled: true
    display_name: "FLUX.1-dev"
    model_path_priority:  # Check in order
      - "/home/user/.cache/huggingface/hub/models--black-forest-labs--FLUX.1-dev/snapshots/0ef5fff789c832c5c7f4e127f94c8b54bbcced44"
      - "/home/user/.cache/huggingface/hub/models--black-forest-labs--FLUX.1-dev"
      - "black-forest-labs/FLUX.1-dev"  # Fallback to download
    
    generation:
      dimensions: [1920, 1088]  # Must be divisible by 16
      scheduler: FlowMatchEulerDiscreteScheduler  # REQUIRED for FLUX
      torch_dtype: bfloat16
      steps_range: [50, 100]  # For random selection
      guidance_range: [2.0, 4.0]
      max_sequence_length: 512
      
    pipeline:
      type: "flux_3stage"  # generate→8K→4K
      stage1_resolution: [1920, 1088]
      stage2_upscale: 4  # to 7680x4352
      stage3_downsample: [3840, 2160]
      
    prompt_requirements:
      max_words: 65
      style: "photorealistic_technical"
      prompter: "deepseek"
    
    memory:
      enable_cpu_offload: true
      attention_slicing: 1
      vae_tiling: true
      clear_cache_after: true

  dalle3:
    class: DalleModel
    enabled: true
    display_name: "DALL-E 3"
    api_endpoint: "https://api.openai.com/v1/images/generations"
    
    generation:
      dimensions: [1792, 1024]
      quality: "hd"
      style: "vivid"
      timeout: 60
      
    pipeline:
      type: "dalle_crop_upscale"
      crop_to_16_9: true
      crop_pixels: [80, 80]  # top, bottom
      upscale_factor: 4
      final_resolution: [3840, 2160]
      
    prompt_requirements:
      max_words: 75
      style: "concrete_visual"
      prompter: "deepseek"
      
  gpt_image_1:
    class: GptImageModel
    enabled: true
    display_name: "GPT-Image-1"
    
    variants:
      direct_api:
        endpoint: "https://api.openai.com/v1/images/generations"
        timeout: 180
      responses_api:
        model: "gpt-4o"
        tool: "image_generation"
        
    generation:
      dimensions: [1536, 1024]
      quality: "high"
      
    pipeline:
      type: "gpt_crop_upscale"
      crop_to_16_9: true
      crop_pixels: [80, 80]
      upscale_factor: 4
      final_resolution: [3840, 2160]
      
    prompt_requirements:
      max_words: 80
      style: "descriptive_artistic"
      prompter: "gpt4o"  # For responses_api variant

  sdxl:
    class: SdxlModel
    enabled: true
    display_name: "SDXL + LoRA"
    model_path: "stabilityai/stable-diffusion-xl-base-1.0"
    
    generation:
      dimensions: [1920, 1080]  # Native 16:9!
      scheduler_options:  # For random selection
        - DPMSolverMultistepScheduler
        - EulerAncestralDiscreteScheduler
        - DDIMScheduler
      torch_dtype: float16
      steps_range: [30, 75]
      guidance_range: [5.0, 12.0]
      
    pipeline:
      type: "sdxl_2x"  # Simple 2x upscale to 4K
      upscale_factor: 2
      final_resolution: [3840, 2160]
      enable_img2img_refine: true
      refine_strength_range: [0.2, 0.4]
      
    lora:
      enabled: true
      auto_select_by_theme: true
      available_loras:
        photorealism:
          path: "/home/user/ai-wallpaper/loras/photorealism.safetensors"
          weight_range: [0.6, 0.9]
          categories: ["LOCAL_MEDIA", "NATURE_EXPANDED", "URBAN_CITYSCAPE"]
        anime:
          path: "/home/user/ai-wallpaper/loras/anime.safetensors"
          weight_range: [0.7, 1.0]
          categories: ["ANIME_MANGA", "GENRE_FUSION"]
        architectural:
          path: "/home/user/ai-wallpaper/loras/architectural.safetensors"
          weight_range: [0.5, 0.8]
          categories: ["ARCHITECTURAL", "URBAN_CITYSCAPE"]
          
    prompt_requirements:
      max_words: 100
      style: "detailed_artistic"
      prompter: "deepseek"
      
random_selection:
  enabled: true
  model_weights:
    flux: 35
    dalle3: 25
    gpt_image_1: 25
    sdxl: 15
    
  # Prevent invalid combinations
  exclusions:
    - model: "flux"
      scheduler: "!FlowMatchEulerDiscreteScheduler"
      
  parameter_randomization:
    steps: "range"  # Use model's steps_range
    guidance: "range"  # Use model's guidance_range
    scheduler: "choice"  # Pick from scheduler_options
    lora_weight: "range"  # Use LoRA's weight_range
```

### 2. Model Base Class (Improved)

```python
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional, List
from pathlib import Path
import torch
import gc

class BaseImageModel(ABC):
    """Abstract base class for all image generation models"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.name = config.get('display_name', self.__class__.__name__)
        self._initialized = False
        
    @abstractmethod
    def initialize(self) -> bool:
        """Initialize model and verify requirements"""
        pass
        
    @abstractmethod
    def generate(self, prompt: str, seed: Optional[int] = None, **kwargs) -> Dict[str, Any]:
        """Generate image, return paths and metadata"""
        pass
        
    @abstractmethod
    def get_optimal_prompt(self, theme: Dict, weather: Dict, context: Dict) -> str:
        """Get model-optimized prompt"""
        pass
        
    @abstractmethod
    def get_pipeline_stages(self) -> List[str]:
        """Return list of pipeline stages for this model"""
        pass
        
    @abstractmethod
    def validate_environment(self) -> Tuple[bool, str]:
        """Validate model can run in current environment"""
        pass
        
    def cleanup(self) -> None:
        """Clean up resources (VRAM, etc)"""
        if hasattr(self, 'pipe'):
            del self.pipe
        gc.collect()
        torch.cuda.empty_cache()
        
    def get_resource_requirements(self) -> Dict[str, Any]:
        """Return estimated resource requirements"""
        return {
            'vram_gb': 24,  # Default, override in subclasses
            'disk_gb': 20,
            'time_minutes': 15
        }
        
    def supports_feature(self, feature: str) -> bool:
        """Check if model supports a feature"""
        features = {
            'lora': False,
            'img2img': False,
            'controlnet': False,
            'scheduler_selection': True,
            'custom_dimensions': False,
            '8k_pipeline': False
        }
        return features.get(feature, False)
```

### 3. Unified CLI Interface

```bash
# Main executable: ai-wallpaper
ai-wallpaper [OPTIONS] COMMAND [ARGS]

# Global options
--config PATH          # Custom config directory
--model MODEL         # Force specific model
--seed INT           # Reproducible generation
--verbose           # Detailed output
--dry-run          # Show plan without executing

# Commands
generate           # Generate wallpaper
  --prompt TEXT     # Custom prompt (bypasses theme system)
  --theme THEME    # Force specific theme
  --random-model   # Use weighted random model selection
  --random-params  # Randomize valid parameters
  --no-upscale    # Skip upscaling stage
  --no-wallpaper  # Generate only, don't set
  --save-stages   # Save intermediate images
  
test              # Test components
  --model MODEL    # Test specific model
  --component      # Component to test (prompt/image/wallpaper/theme)
  --quick         # Fast test mode
  
config            # Configuration management
  --show          # Display current configuration
  --validate      # Validate all config files
  --set KEY=VAL   # Set configuration value
  --reset         # Reset to defaults
  
models            # Model information
  --list          # List available models
  --info MODEL    # Detailed model information
  --check MODEL   # Check if model is ready
  --install MODEL # Download/install model
```

### 4. Model-Specific Implementations

**FLUX Model** (flux_model.py):
```python
class FluxModel(BaseImageModel):
    """FLUX.1-dev with 8K→4K supersampling pipeline"""
    
    def get_pipeline_stages(self) -> List[str]:
        return [
            "flux_generation",      # 1920x1088
            "realesrgan_8k",       # 4x to 7680x4352
            "lanczos_4k"          # Downsample to 3840x2160
        ]
        
    def supports_feature(self, feature: str) -> bool:
        features = {
            '8k_pipeline': True,
            'scheduler_selection': False,  # Must use FlowMatchEuler
        }
        return features.get(feature, super().supports_feature(feature))
```

**SDXL Model** (sdxl_model.py):
```python
class SdxlModel(BaseImageModel):
    """SDXL with LoRA support and efficient 2x upscaling"""
    
    def get_pipeline_stages(self) -> List[str]:
        stages = ["sdxl_generation"]  # 1920x1080
        
        if self.config['pipeline']['enable_img2img_refine']:
            stages.append("sdxl_img2img_refine")
            
        stages.append("realesrgan_2x")  # 2x to 3840x2160
        return stages
        
    def supports_feature(self, feature: str) -> bool:
        features = {
            'lora': True,
            'img2img': True,
            'scheduler_selection': True,
            'custom_dimensions': True
        }
        return features.get(feature, super().supports_feature(feature))
        
    def auto_select_lora(self, theme: Dict) -> Optional[Dict]:
        """Intelligently select LoRA based on theme"""
        category = theme.get('category', '')
        lora_config = self.config.get('lora', {})
        
        for lora_name, lora_data in lora_config.get('available_loras', {}).items():
            if category in lora_data.get('categories', []):
                return {
                    'name': lora_name,
                    'path': lora_data['path'],
                    'weight': random.uniform(*lora_data['weight_range'])
                }
        return None
```

### 5. Resource Management

```python
class ResourceManager:
    """Manage VRAM and system resources"""
    
    def __init__(self):
        self.allocated_models = {}
        
    def can_load_model(self, model_name: str, requirements: Dict) -> bool:
        """Check if model can be loaded"""
        available_vram = torch.cuda.get_device_properties(0).total_memory
        used_vram = torch.cuda.memory_allocated()
        free_vram = (available_vram - used_vram) / 1024**3
        
        required_vram = requirements.get('vram_gb', 24)
        return free_vram >= required_vram * 1.1  # 10% buffer
        
    def prepare_for_model(self, model_name: str) -> None:
        """Clean up resources before loading model"""
        # Unload all other models
        for name, model in self.allocated_models.items():
            if name != model_name:
                model.cleanup()
                
        # Force garbage collection
        gc.collect()
        torch.cuda.empty_cache()
        
    def register_model(self, model_name: str, model_instance: BaseImageModel) -> None:
        """Register loaded model"""
        self.allocated_models[model_name] = model_instance
```

### 6. Pipeline Orchestration

```python
class PipelineOrchestrator:
    """Orchestrate model-specific pipelines at maximum quality"""
    
    def __init__(self, model: BaseImageModel):
        self.model = model
        self.stages = model.get_pipeline_stages()
        
    def execute(self, initial_image_path: Path) -> Dict[str, Any]:
        """Execute full pipeline"""
        results = {
            'stages': {},
            'final_image': None,
            'metadata': {}
        }
        
        current_image = initial_image_path
        
        for stage in self.stages:
            stage_handler = self.get_stage_handler(stage)
            stage_result = stage_handler.process(current_image)
            
            results['stages'][stage] = stage_result
            current_image = stage_result['output_path']
            
            # Save intermediate if requested
            if self.should_save_stage(stage):
                self.save_stage_output(stage, stage_result)
                
        results['final_image'] = current_image
        return results
```

## Implementation Phases

### Phase 1: Core Infrastructure (Week 1)
1. Set up package structure
2. Implement configuration system with validation
3. Create base model abstraction
4. Implement unified logging (fail loud!)
5. Migrate weather and wallpaper modules

### Phase 2: Model Implementation
1. Implement FLUX model with existing pipeline
2. Migrate DALL-E 3 implementation
3. Migrate GPT-Image-1 variants
4. Implement each model's maximum quality settings

### Phase 3: CLI and Pipeline System
1. Implement Click-based CLI
2. Implement pipeline orchestration
3. Add random selection with validation
4. Ensure all pipelines use maximum quality

### Phase 4: Advanced Features
1. Implement SDXL with LoRA support
2. Add resource management
3. Add intelligent parameter randomization
4. Finalize all model integrations

### Phase 5: Deployment
1. Performance optimization
2. Documentation
3. Update cron integration
4. Final validation of all models

## Key Improvements

1. **Model-Specific Pipelines**: Each model has its optimal pipeline
   - FLUX: 3-stage 8K→4K (maximum quality)
   - SDXL: 2-stage with 2x upscale (efficient)
   - DALL-E/GPT: Crop and upscale (API-optimized)

2. **Maximum Quality Always**: No compromises, always use highest settings

3. **Resource Management**: Proper VRAM management between models

4. **Validation**: JSON schemas for configuration validation

5. **LoRA Intelligence**: Auto-select LoRAs based on themes

6. **Fail Loud Philosophy**: Maintained throughout with custom exceptions

7. **Clean Architecture**: Modular design for easy extension

8. **Unified CLI**: Single entry point for all models and features

This refactor creates a professional, extensible system focused on maximum quality output with a clean break from the legacy monolithic scripts.