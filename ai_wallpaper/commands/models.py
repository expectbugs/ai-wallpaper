#!/usr/bin/env python3
"""
Models Command Implementation
Model management and information
"""

from typing import Optional
from pathlib import Path

from ..core import get_logger, get_config
from ..core.exceptions import ModelError
from ..models.flux_model import FluxModel

class ModelsCommand:
    """Handles model management"""
    
    def __init__(self, config_dir: Optional[str] = None, verbose: bool = False):
        """Initialize models command
        
        Args:
            config_dir: Custom config directory
            verbose: Enable verbose output
        """
        self.config_dir = config_dir
        self.verbose = verbose
        self.logger = get_logger()
        self.config = get_config()
        
    def list_models(self) -> None:
        """List all available models"""
        self.logger.info("=== AVAILABLE MODELS ===")
        
        models = self.config.models.get('models', {})
        
        for model_name, model_config in models.items():
            enabled = model_config.get('enabled', False)
            display_name = model_config.get('display_name', model_name)
            
            status = "✓ Enabled" if enabled else "✗ Disabled"
            self.logger.info(f"\n{display_name} ({model_name})")
            self.logger.info(f"  Status: {status}")
            self.logger.info(f"  Class: {model_config.get('class')}")
            
            # Show key features
            if 'pipeline' in model_config:
                pipeline = model_config['pipeline']
                self.logger.info(f"  Pipeline: {pipeline.get('type')}")
                
            if 'generation' in model_config:
                gen = model_config['generation']
                dims = gen.get('dimensions', [])
                if dims:
                    self.logger.info(f"  Resolution: {dims[0]}x{dims[1]}")
                    
    def show_model_info(self, model_name: str) -> None:
        """Show detailed model information
        
        Args:
            model_name: Model to show info for
        """
        try:
            model_config = self.config.get_model_config(model_name)
        except Exception as e:
            raise ModelError(
                f"Model not found: {model_name}!\n"
                f"Error: {str(e)}\n"
                f"Use 'list' command to see available models.\n"
                f"Model must exist to proceed!"
            )
            
        self.logger.info(f"=== {model_config.get('display_name', model_name)} ===")
        
        # Basic info
        self.logger.info(f"\nClass: {model_config.get('class')}")
        self.logger.info(f"Enabled: {model_config.get('enabled', False)}")
        
        # Generation settings
        if 'generation' in model_config:
            gen = model_config['generation']
            self.logger.info("\nGeneration Settings:")
            self.logger.info(f"  Dimensions: {gen.get('dimensions')}")
            self.logger.info(f"  Steps: {gen.get('steps_range', 'fixed')}")
            self.logger.info(f"  Guidance: {gen.get('guidance_range', 'fixed')}")
            
        # Pipeline info
        if 'pipeline' in model_config:
            pipeline = model_config['pipeline']
            self.logger.info(f"\nPipeline: {pipeline.get('type')}")
            
            # Show pipeline stages
            if model_name == 'flux':
                self.logger.info("  Stages:")
                self.logger.info("    1. Generate at 1920x1088")
                self.logger.info("    2. Upscale 4x to 7680x4352 (8K)")
                self.logger.info("    3. Downsample to 3840x2160 (4K)")
                
        # Requirements
        if model_name == 'flux':
            self.logger.info("\nRequirements:")
            self.logger.info("  VRAM: 24GB (RTX 3090 or better)")
            self.logger.info("  Disk: ~30GB for model files")
            self.logger.info("  Dependencies: Real-ESRGAN")
            
        if self.verbose:
            import yaml
            self.logger.info("\nFull Configuration:")
            print(yaml.dump(model_config, default_flow_style=False))
            
    def check_model(self, model_name: str) -> None:
        """Check if model is ready to use
        
        Args:
            model_name: Model to check
        """
        try:
            model_config = self.config.get_model_config(model_name)
        except Exception as e:
            raise ModelError(
                f"Model not found: {model_name}!\n"
                f"Error: {str(e)}\n"
                f"Use 'list' command to see available models.\n"
                f"Model must exist to proceed!"
            )
            
        self.logger.info(f"Checking {model_config.get('display_name', model_name)}...")
        
        # Check if enabled
        if not model_config.get('enabled', False):
            self.logger.warning("Model is disabled in configuration")
            return
            
        # Model-specific checks
        if model_name == 'flux':
            model = FluxModel(model_config)
            valid, message = model.validate_environment()
            
            if valid:
                self.logger.info(f"✓ {message}")
                
                # Check model files
                try:
                    model_path = model._find_model_path()
                    self.logger.info(f"✓ Model files found at: {model_path}")
                except FileNotFoundError:
                    self.logger.warning("✗ Model files not found locally")
                    self.logger.info("  Model will be downloaded on first use")
                except Exception as e:
                    self.logger.warning(f"✗ Error checking model files: {e}")
                    self.logger.info("  Model status unknown")
                    
            else:
                self.logger.error(f"✗ {message}")
                
        else:
            self.logger.warning(f"Model '{model_name}' not yet implemented")
            
    def install_model(self, model_name: str) -> None:
        """Install/download a model
        
        Args:
            model_name: Model to install
        """
        self.logger.warning("Model installation not yet implemented")
        
        if model_name == 'flux':
            self.logger.info("\nTo use FLUX, it will be automatically downloaded on first use")
            self.logger.info("Or you can manually download from HuggingFace:")
            self.logger.info("  https://huggingface.co/black-forest-labs/FLUX.1-dev")
            
        self.logger.info("\nPlease run 'ai-wallpaper generate' to trigger automatic download")