#!/usr/bin/env python3
"""
Random Selector for AI Wallpaper System
Intelligent random selection of models and parameters
"""

import random
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path

from ..core import get_logger, get_config
from ..core.exceptions import ConfigurationError


class RandomSelector:
    """Manages random selection with validation"""
    
    def __init__(self):
        """Initialize random selector"""
        self.logger = get_logger(model="RandomSelector")
        self.config = get_config()
        self._load_random_config()
        
    def _load_random_config(self) -> None:
        """Load random selection configuration"""
        self.random_config = self.config.models.get('random_selection', {})
        
        if not self.random_config.get('enabled', False):
            self.logger.warning("Random selection is disabled in configuration")
            
        self.model_weights = self.random_config.get('model_weights', {})
        self.exclusions = self.random_config.get('exclusions', [])
        self.param_randomization = self.random_config.get('parameter_randomization', {})
        
    def select_random_model(self) -> str:
        """Select a random model based on weights
        
        Returns:
            Selected model name
            
        Raises:
            ConfigurationError: If no models configured
        """
        if not self.model_weights:
            raise ConfigurationError("No model weights configured for random selection")
            
        # Filter to enabled models only
        enabled_models = {}
        for model_name, weight in self.model_weights.items():
            model_config = self.config.get_model_config(model_name)
            if model_config and model_config.get('enabled', False):
                enabled_models[model_name] = weight
                
        if not enabled_models:
            raise ConfigurationError("No enabled models for random selection")
            
        # Weighted random selection
        models = list(enabled_models.keys())
        weights = list(enabled_models.values())
        
        selected = random.choices(models, weights=weights, k=1)[0]
        
        self.logger.info(f"Randomly selected model: {selected} (from {len(models)} options)")
        
        return selected
        
    def get_random_parameters(
        self,
        model_name: str,
        model_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Get randomized parameters for model
        
        Args:
            model_name: Model name
            model_config: Model configuration
            
        Returns:
            Randomized parameters
        """
        params = {}
        generation_config = model_config.get('generation', {})
        
        # Random steps
        if self.param_randomization.get('steps') == 'range':
            if 'steps_range' in generation_config:
                min_steps, max_steps = generation_config['steps_range']
                params['steps'] = random.randint(min_steps, max_steps)
                self.logger.debug(f"Random steps: {params['steps']} (range: {min_steps}-{max_steps})")
                
        # Random guidance scale
        if self.param_randomization.get('guidance') == 'range':
            if 'guidance_range' in generation_config:
                min_guidance, max_guidance = generation_config['guidance_range']
                params['guidance_scale'] = round(random.uniform(min_guidance, max_guidance), 1)
                self.logger.debug(f"Random guidance: {params['guidance_scale']} (range: {min_guidance}-{max_guidance})")
                
        # Random scheduler
        if self.param_randomization.get('scheduler') == 'choice':
            if 'scheduler_options' in generation_config:
                # Check exclusions
                valid_schedulers = self._filter_schedulers(
                    model_name,
                    generation_config['scheduler_options']
                )
                if valid_schedulers:
                    params['scheduler'] = random.choice(valid_schedulers)
                    self.logger.debug(f"Random scheduler: {params['scheduler']}")
                    
        # Random LoRA weight (for SDXL)
        if self.param_randomization.get('lora_weight') == 'range':
            lora_config = model_config.get('lora', {})
            if lora_config.get('enabled') and lora_config.get('auto_select_by_theme'):
                # This will be handled by the model based on theme
                pass
                
        return params
        
    def _filter_schedulers(
        self,
        model_name: str,
        schedulers: List[str]
    ) -> List[str]:
        """Filter schedulers based on exclusions
        
        Args:
            model_name: Model name
            schedulers: List of scheduler options
            
        Returns:
            Valid schedulers
        """
        valid_schedulers = schedulers.copy()
        
        for exclusion in self.exclusions:
            if exclusion.get('model') == model_name:
                excluded = exclusion.get('scheduler', '')
                if excluded.startswith('!'):
                    # Keep only this scheduler
                    required = excluded[1:]
                    valid_schedulers = [s for s in valid_schedulers if s == required]
                else:
                    # Remove this scheduler
                    valid_schedulers = [s for s in valid_schedulers if s != excluded]
                    
        return valid_schedulers
        
    def should_use_random_theme(self) -> bool:
        """Determine if random theme selection should be used
        
        Returns:
            True if random theme should be selected
        """
        # 95% chance of random theme
        return random.random() < 0.95
        
    def get_theme_chaos_chance(self) -> float:
        """Get chance of chaos mode activation
        
        Returns:
            Chaos mode probability
        """
        # Could be configurable
        return 0.05  # 5% chance
        
    def select_lora_weight(self, weight_range: List[float]) -> float:
        """Select random LoRA weight from range
        
        Args:
            weight_range: [min, max] weight range
            
        Returns:
            Selected weight
        """
        if len(weight_range) != 2:
            return 0.8  # Default
            
        min_weight, max_weight = weight_range
        weight = round(random.uniform(min_weight, max_weight), 2)
        
        self.logger.debug(f"Random LoRA weight: {weight} (range: {min_weight}-{max_weight})")
        
        return weight
        
    def select_from_list(
        self,
        items: List[Any],
        weights: Optional[List[float]] = None
    ) -> Any:
        """Select random item from list
        
        Args:
            items: List of items
            weights: Optional weights for selection
            
        Returns:
            Selected item
        """
        if not items:
            raise ValueError("No items to select from")
            
        if weights:
            return random.choices(items, weights=weights, k=1)[0]
        else:
            return random.choice(items)
            
    def get_random_seed(self) -> int:
        """Generate random seed for reproducibility
        
        Returns:
            Random seed
        """
        seed = random.randint(0, 2**32 - 1)
        self.logger.info(f"Generated seed: {seed} (save this to reproduce the image!)")
        return seed
        
    def should_save_stages(self) -> bool:
        """Randomly decide if stages should be saved
        
        Returns:
            True if stages should be saved
        """
        # 10% chance to save stages for debugging
        return random.random() < 0.1
        
    def get_quality_variance(self) -> Dict[str, Any]:
        """Get random quality variations (always maximum)
        
        Returns:
            Quality parameters
        """
        # Always use maximum quality, but can vary some settings
        return {
            'jpeg_quality': 100,  # Always maximum
            'png_compression': 0,  # No compression
            'save_lossless': True,
            'optimize': False  # Don't optimize to maintain quality
        }
        
    def validate_random_config(self) -> Tuple[bool, List[str]]:
        """Validate random selection configuration
        
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        
        if not self.random_config.get('enabled'):
            errors.append("Random selection is disabled")
            
        if not self.model_weights:
            errors.append("No model weights configured")
        else:
            # Check weights sum to reasonable value
            total_weight = sum(self.model_weights.values())
            if total_weight == 0:
                errors.append("Model weights sum to zero")
                
        # Check for invalid exclusions
        for exclusion in self.exclusions:
            if 'model' not in exclusion:
                errors.append(f"Invalid exclusion: missing 'model' key")
                
        is_valid = len(errors) == 0
        
        if not is_valid:
            self.logger.warning(f"Random config validation failed: {errors}")
            
        return is_valid, errors


# Global instance
_random_selector: Optional[RandomSelector] = None


def get_random_selector() -> RandomSelector:
    """Get global random selector instance
    
    Returns:
        RandomSelector instance
    """
    global _random_selector
    if _random_selector is None:
        _random_selector = RandomSelector()
    return _random_selector


def select_random_model() -> str:
    """Convenience function to select random model
    
    Returns:
        Selected model name
    """
    selector = get_random_selector()
    return selector.select_random_model()


def get_random_parameters(model_name: str, model_config: Dict[str, Any]) -> Dict[str, Any]:
    """Convenience function to get random parameters
    
    Args:
        model_name: Model name
        model_config: Model configuration
        
    Returns:
        Random parameters
    """
    selector = get_random_selector()
    return selector.get_random_parameters(model_name, model_config)