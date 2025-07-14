#!/usr/bin/env python3
"""
Generate Command Implementation
Main wallpaper generation workflow
"""

import random
import time
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

from ..core import get_logger, get_config, get_weather_context, set_wallpaper
from ..core.exceptions import AIWallpaperError, GenerationError
from ..prompt import DeepSeekPrompter, get_random_theme_with_weather

class GenerateCommand:
    """Handles the wallpaper generation workflow"""
    
    def __init__(self, config_dir: Optional[str] = None, verbose: bool = False, dry_run: bool = False):
        """Initialize generate command
        
        Args:
            config_dir: Custom config directory
            verbose: Enable verbose output
            dry_run: Show plan without executing
        """
        self.config_dir = config_dir
        self.verbose = verbose
        self.dry_run = dry_run
        self.logger = get_logger()
        self.config = get_config()
        
    def execute(
        self,
        prompt: Optional[str] = None,
        theme: Optional[str] = None,
        model: Optional[str] = None,
        random_model: bool = False,
        random_params: bool = False,
        seed: Optional[int] = None,
        no_upscale: bool = False,
        no_wallpaper: bool = False,
        save_stages: bool = False,
        output_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """Execute wallpaper generation
        
        Args:
            prompt: Custom prompt (bypasses theme system)
            theme: Force specific theme
            model: Force specific model
            random_model: Use weighted random selection
            random_params: Randomize parameters
            seed: Random seed
            no_upscale: Skip upscaling
            no_wallpaper: Don't set as wallpaper
            save_stages: Save intermediate images
            output_path: Custom output path
            
        Returns:
            Generation results
        """
        start_time = time.time()
        
        # Show plan if dry-run
        if self.dry_run:
            self._show_plan(locals())
            return None
            
        try:
            # Step 1: Select model
            self.logger.log_stage("Step 1/6", "Selecting model")
            selected_model = self._select_model(model, random_model)
            
            # Step 2: Initialize model
            self.logger.log_stage("Step 2/6", f"Initializing {selected_model}")
            model_instance = self._initialize_model(selected_model)
            
            # Step 3: Generate or get prompt
            self.logger.log_stage("Step 3/6", "Preparing prompt")
            if not prompt:
                prompt = self._generate_prompt(theme, model_instance)
                
            # Step 4: Set generation parameters
            self.logger.log_stage("Step 4/6", "Setting parameters")
            params = self._prepare_parameters(
                model_instance,
                random_params,
                no_upscale,
                save_stages
            )
            
            # Step 5: Generate image
            self.logger.log_stage("Step 5/6", "Generating image")
            result = model_instance.generate(prompt, seed=seed, **params)
            
            # Step 6: Set wallpaper if requested
            if not no_wallpaper:
                self.logger.log_stage("Step 6/6", "Setting wallpaper")
                set_wallpaper(Path(result['image_path']))
            else:
                self.logger.log_stage("Step 6/6", "Skipping wallpaper setting")
                
            # Clean up
            model_instance.cleanup()
            
            # Calculate duration
            duration = time.time() - start_time
            
            # Prepare final result
            final_result = {
                'image_path': result['image_path'],
                'model': selected_model,
                'prompt': prompt,
                'seed': result['metadata']['seed'],
                'duration': duration,
                'stages': result.get('stages', {})
            }
            
            # Save prompt to history
            self._save_prompt_history(prompt, final_result)
            
            # Show prominent seed display if configured
            config = get_config()
            if config.system and config.system.get('generation', {}).get('show_seed_prominently', True):
                # Use the actual seed from the result, not the input parameter
                actual_seed = final_result.get('seed', result['metadata'].get('seed'))
                if actual_seed:
                    self._display_seed_prominently(actual_seed)
            
            return final_result
            
        except Exception as e:
            raise GenerationError("generation", "workflow", e)
            
    def _show_plan(self, options: Dict[str, Any]) -> None:
        """Show execution plan for dry-run
        
        Args:
            options: Command options
        """
        self.logger.info("=== DRY RUN - EXECUTION PLAN ===")
        
        # Model selection
        if options['model']:
            self.logger.info(f"Model: {options['model']} (forced)")
        elif options['random_model']:
            self.logger.info("Model: Will select randomly based on weights")
        else:
            self.logger.info("Model: flux (default)")
            
        # Prompt
        if options['prompt']:
            self.logger.info(f"Prompt: Custom - '{options['prompt'][:50]}...'")
        else:
            self.logger.info("Prompt: Will generate based on theme and weather")
            
        # Parameters
        self.logger.info(f"Random parameters: {options['random_params']}")
        self.logger.info(f"Seed: {options['seed'] or 'random'}")
        self.logger.info(f"Upscaling: {'disabled' if options['no_upscale'] else 'enabled'}")
        self.logger.info(f"Set wallpaper: {'no' if options['no_wallpaper'] else 'yes'}")
        self.logger.info(f"Save stages: {options['save_stages']}")
        
        self.logger.info("=== END DRY RUN ===")
        
    def _select_model(self, model: Optional[str], random_model: bool) -> str:
        """Select which model to use
        
        Args:
            model: Forced model selection
            random_model: Use random selection
            
        Returns:
            Selected model name
        """
        if model:
            self.logger.info(f"Using forced model: {model}")
            return model
            
        if random_model:
            # Get model weights from config
            random_config = self.config.models.get('random_selection', {})
            if not random_config.get('enabled'):
                self.logger.warning("Random selection disabled in config, using flux")
                return 'flux'
                
            weights = random_config.get('model_weights', {})
            models = list(weights.keys())
            model_weights = list(weights.values())
            
            selected = random.choices(models, weights=model_weights, k=1)[0]
            self.logger.info(f"Randomly selected model: {selected}")
            return selected
            
        # Default to flux
        self.logger.info("Using default model: flux")
        return 'flux'
        
    def _initialize_model(self, model_name: str):
        """Initialize the selected model
        
        Args:
            model_name: Name of model to initialize
            
        Returns:
            Initialized model instance
        """
        # Get model config
        model_config = self.config.get_model_config(model_name)
        
        # Add name to config for consistency
        model_config['name'] = model_name
        
        # Create model instance based on class
        model_class = model_config.get('class')
        
        if model_class == 'FluxModel':
            from ..models import FluxModel
            model = FluxModel(model_config)
        elif model_class == 'DalleModel':
            from ..models import DalleModel
            model = DalleModel(model_config)
        elif model_class == 'GptImageModel':
            from ..models import GptImageModel
            model = GptImageModel(model_config)
        elif model_class == 'SdxlModel':
            from ..models import SdxlModel
            model = SdxlModel(model_config)
        else:
            raise ValueError(f"Unknown model class: {model_class}")
            
        # Initialize
        model.initialize()
        
        return model
        
    def _generate_prompt(self, theme: Optional[str], model_instance) -> str:
        """Generate prompt using theme system
        
        Args:
            theme: Forced theme name
            model_instance: Model instance for requirements
            
        Returns:
            Generated prompt
        """
        # Get weather context
        weather = get_weather_context()
        
        # Get theme
        theme_result = get_random_theme_with_weather(weather)
        
        if theme:
            self.logger.info(f"Overriding with forced theme: {theme}")
            # TODO: Implement theme override
            
        # Try model's optimal prompt method first
        prompt = model_instance.get_optimal_prompt(
            theme_result['theme'],
            weather,
            self._get_context()
        )
        
        # If model doesn't have specific prompt optimization, use DeepSeek
        if not prompt:
            # Initialize prompter
            prompter = DeepSeekPrompter()
            prompter.initialize()
            
            # Get model requirements
            model_requirements = model_instance.config.get('prompt_requirements', {})
            
            # Generate prompt
            context = self._get_context()
            history = prompter.load_prompt_history()
            
            prompt = prompter.generate_prompt(
                theme=theme_result['theme'],
                weather=weather,
                context=context,
                history=history,
                model_requirements=model_requirements
            )
            
            # Clean up prompter
            prompter.cleanup()
        
        return prompt
        
    def _get_context(self) -> Dict[str, Any]:
        """Get current context information
        
        Returns:
            Context dictionary
        """
        now = datetime.now()
        
        # Determine season
        month = now.month
        if month in [12, 1, 2]:
            season = "winter"
        elif month in [3, 4, 5]:
            season = "spring"
        elif month in [6, 7, 8]:
            season = "summer"
        else:
            season = "autumn"
            
        return {
            'date': now.strftime("%Y-%m-%d"),
            'time': now.strftime("%H:%M"),
            'day_of_week': now.strftime("%A"),
            'month': now.strftime("%B"),
            'year': now.year,
            'season': season
        }
        
    def _prepare_parameters(
        self,
        model_instance,
        random_params: bool,
        no_upscale: bool,
        save_stages: bool
    ) -> Dict[str, Any]:
        """Prepare generation parameters
        
        Args:
            model_instance: Model instance
            random_params: Randomize parameters
            no_upscale: Skip upscaling
            save_stages: Save stages
            
        Returns:
            Parameters dictionary
        """
        params = {}
        
        if random_params:
            # Get random parameters from model config
            generation_config = model_instance.config.get('generation', {})
            
            # Random steps
            if 'steps_range' in generation_config:
                params['steps'] = random.randint(*generation_config['steps_range'])
                
            # Random guidance
            if 'guidance_range' in generation_config:
                params['guidance_scale'] = random.uniform(*generation_config['guidance_range'])
                
            # Random scheduler (if supported)
            if 'scheduler_options' in generation_config and model_instance.supports_feature('scheduler_selection'):
                params['scheduler'] = random.choice(generation_config['scheduler_options'])
                
            self.logger.info(f"Randomized parameters: {params}")
            
        # Add other parameters
        params['no_upscale'] = no_upscale
        params['save_stages'] = save_stages
        
        return params
        
    def _save_prompt_history(self, prompt: str, result: Dict[str, Any]) -> None:
        """Save prompt to history
        
        Args:
            prompt: Generated prompt
            result: Generation result
        """
        prompter = DeepSeekPrompter()
        prompter.save_prompt(prompt, {
            'seed': result['seed'],
            'model': result['model'],
            'timestamp': datetime.now().isoformat()
        })
        
    def _display_seed_prominently(self, seed: int) -> None:
        """Display seed prominently for reproducibility
        
        Args:
            seed: Generation seed
        """
        self.logger.critical(f"""
╔═══════════════════════════════════════════╗
║       GENERATION SUCCESSFUL!              ║
║                                           ║
║  SEED: {seed:10d}                      ║
║  (Save this to reproduce the image!)      ║
╚═══════════════════════════════════════════╝
""")