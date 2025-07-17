#!/usr/bin/env python3
"""
Prompt Optimizer Module
Optimizes prompts for specific models based on their requirements
"""

import re
from typing import Dict, Any, List, Optional
from pathlib import Path

from ..core import get_logger

class PromptOptimizer:
    """Optimizes prompts for different AI models"""
    
    def __init__(self):
        """Initialize prompt optimizer"""
        self.logger = get_logger(model="PromptOptimizer")
        
        # Model-specific optimization strategies
        self.strategies = {
            'flux': self._optimize_for_flux,
            'dalle3': self._optimize_for_dalle,
            'gpt-image-1': self._optimize_for_gpt,
            'sdxl': self._optimize_for_sdxl
        }
        
    def optimize(self, prompt: str, model_name: str, requirements: Dict[str, Any]) -> str:
        """Optimize prompt for specific model
        
        Args:
            prompt: Original prompt
            model_name: Target model name
            requirements: Model requirements
            
        Returns:
            Optimized prompt
        """
        # Clean base prompt
        prompt = self._clean_prompt(prompt)
        
        # Apply model-specific optimization
        if model_name in self.strategies:
            prompt = self.strategies[model_name](prompt, requirements)
        else:
            self.logger.warning(f"No optimization strategy for model: {model_name}")
            
        # Ensure length requirements
        prompt = self._enforce_length(prompt, requirements)
        
        return prompt
        
    def _clean_prompt(self, prompt: str) -> str:
        """Clean and normalize prompt
        
        Args:
            prompt: Original prompt
            
        Returns:
            Cleaned prompt
        """
        # Remove extra whitespace
        prompt = ' '.join(prompt.split())
        
        # Remove duplicate punctuation
        prompt = re.sub(r'([.!?])\1+', r'\1', prompt)
        
        # Ensure single space after punctuation
        prompt = re.sub(r'([.!?,])(\w)', r'\1 \2', prompt)
        
        return prompt.strip()
        
    def _enforce_length(self, prompt: str, requirements: Dict[str, Any]) -> str:
        """Check word count but do NOT truncate
        
        Args:
            prompt: Prompt to check
            requirements: Model requirements
            
        Returns:
            Original prompt (unmodified)
        """
        max_words = requirements.get('max_words', 100)
        words = prompt.split()
        
        if len(words) > max_words:
            self.logger.debug(f"Prompt has {len(words)} words (guideline: {max_words}) - keeping full prompt")
            
        return prompt
        
    def _optimize_for_flux(self, prompt: str, requirements: Dict[str, Any]) -> str:
        """Optimize prompt for FLUX model
        
        FLUX works best with:
        - Technical, detailed descriptions
        - Photorealistic style mentions
        - Clear composition structure
        - No extra quality keywords (handles internally)
        """
        # FLUX handles quality internally, remove redundant quality words
        quality_words = [
            'ultra high quality', 'masterpiece', 'best quality',
            '8k', '4k', 'hd', 'uhd', 'high resolution'
        ]
        
        prompt_lower = prompt.lower()
        for word in quality_words:
            if word in prompt_lower:
                prompt = re.sub(rf'\b{re.escape(word)}\b', '', prompt, flags=re.IGNORECASE)
                
        # Add technical photography terms if missing
        photo_terms = ['photorealistic', 'detailed', 'sharp focus']
        has_photo_term = any(term in prompt.lower() for term in photo_terms)
        
        if not has_photo_term and 'photorealistic' not in prompt.lower():
            prompt = f"Photorealistic {prompt}"
            
        # Ensure composition structure
        if 'foreground' not in prompt.lower() and 'background' not in prompt.lower():
            prompt += " with clear foreground and background elements"
            
        return prompt
        
    def _optimize_for_dalle(self, prompt: str, requirements: Dict[str, Any]) -> str:
        """Optimize prompt for DALL-E 3
        
        DALL-E 3 works best with:
        - Natural language descriptions
        - Specific visual details
        - Artistic style mentions
        - Concrete rather than abstract
        """
        # DALL-E likes more descriptive, story-like prompts
        
        # Add "digital art" or "painting" if no style mentioned
        style_words = ['style', 'art', 'painting', 'illustration', 'render']
        has_style = any(word in prompt.lower() for word in style_words)
        
        if not has_style:
            prompt += ", digital art style"
            
        # DALL-E responds well to emotional/atmospheric descriptions
        if 'atmosphere' not in prompt.lower() and 'mood' not in prompt.lower():
            # Add based on content
            if any(word in prompt.lower() for word in ['dark', 'night', 'shadow']):
                prompt += ", mysterious atmosphere"
            elif any(word in prompt.lower() for word in ['bright', 'sunny', 'day']):
                prompt += ", vibrant atmosphere"
                
        return prompt
        
    def _optimize_for_gpt(self, prompt: str, requirements: Dict[str, Any]) -> str:
        """Optimize prompt for GPT-Image-1
        
        GPT-Image works best with:
        - Comprehensive scene descriptions
        - Multiple detail layers
        - Clear artistic direction
        - Emphasis on composition
        """
        # GPT models benefit from structured descriptions
        
        # Add composition guidance if missing
        if 'composition' not in prompt.lower():
            prompt += " with balanced composition"
            
        # Add detail emphasis
        detail_words = ['detailed', 'intricate', 'rich']
        has_detail = any(word in prompt.lower() for word in detail_words)
        
        if not has_detail:
            prompt = f"Highly detailed {prompt}"
            
        # GPT responds well to cinematic descriptions
        if 'cinematic' not in prompt.lower() and 'dramatic' not in prompt.lower():
            if any(word in prompt.lower() for word in ['landscape', 'scene', 'view']):
                prompt = prompt.replace('scene', 'cinematic scene')
                prompt = prompt.replace('view', 'dramatic view')
                
        return prompt
        
    def _optimize_for_sdxl(self, prompt: str, requirements: Dict[str, Any]) -> str:
        """Optimize prompt for SDXL
        
        SDXL works best with:
        - Structured tag-like descriptions
        - Quality modifiers at the beginning
        - Style descriptors
        - Negative prompt awareness
        """
        # SDXL benefits from quality tags at the start
        if not prompt.lower().startswith(('masterpiece', 'best quality', 'high quality')):
            prompt = f"High quality, {prompt}"
            
        # Add style tags if missing
        if 'style' not in prompt.lower():
            # Determine appropriate style based on content
            if any(word in prompt.lower() for word in ['photo', 'realistic', 'real']):
                prompt += ", photographic style"
            elif any(word in prompt.lower() for word in ['anime', 'manga', 'cartoon']):
                prompt += ", anime style"
            else:
                prompt += ", artistic style"
                
        # SDXL responds well to technical descriptors
        tech_words = ['8k', 'detailed', 'sharp', 'professional']
        has_tech = any(word in prompt.lower() for word in tech_words)
        
        if not has_tech:
            prompt += ", highly detailed"
            
        # Add rendering style if applicable
        if any(word in prompt.lower() for word in ['3d', 'render', 'cgi']):
            if 'octane' not in prompt.lower() and 'unreal' not in prompt.lower():
                prompt += ", octane render"
                
        return prompt
        
    def get_negative_prompt(self, model_name: str, theme: Optional[Dict] = None) -> str:
        """Get appropriate negative prompt for model
        
        Args:
            model_name: Model name
            theme: Optional theme context
            
        Returns:
            Negative prompt string
        """
        base_negative = (
            "low quality, blurry, pixelated, noisy, oversaturated, "
            "underexposed, overexposed, bad anatomy, bad proportions"
        )
        
        model_specific = {
            'flux': "",  # FLUX doesn't use negative prompts
            'dalle3': "",  # DALL-E 3 doesn't use negative prompts
            'gpt-image-1': "",  # GPT doesn't use negative prompts
            'sdxl': f"{base_negative}, watermark, signature, text, logo, "
                   "duplicate, morbid, mutilated, extra fingers, mutated hands, "
                   "poorly drawn hands, mutation, deformed, bad proportions"
        }
        
        return model_specific.get(model_name, base_negative)
        
    def enhance_with_context(
        self, 
        prompt: str, 
        weather: Dict[str, Any],
        time_of_day: str
    ) -> str:
        """Enhance prompt with contextual information
        
        Args:
            prompt: Base prompt
            weather: Weather context
            time_of_day: Time context
            
        Returns:
            Enhanced prompt
        """
        # Add weather atmosphere if not already present
        weather_mood = weather.get('mood', 'neutral')
        
        weather_enhancements = {
            'stormy': 'dramatic storm lighting',
            'rainy': 'wet surfaces reflecting light',
            'sunny': 'bright natural sunlight',
            'cloudy': 'soft diffused lighting',
            'foggy': 'atmospheric fog and mist',
            'snowy': 'pristine snow coverage'
        }
        
        if weather_mood in weather_enhancements:
            enhancement = weather_enhancements[weather_mood]
            if enhancement.split()[0] not in prompt.lower():
                prompt += f", {enhancement}"
                
        # Add time-based lighting if missing
        time_enhancements = {
            'morning': 'golden morning light',
            'afternoon': 'bright afternoon sun',
            'evening': 'warm sunset glow',
            'night': 'moonlight and stars'
        }
        
        # Determine time of day from context
        hour = int(time_of_day.split(':')[0]) if ':' in time_of_day else 12
        
        if hour < 10:
            time_period = 'morning'
        elif hour < 17:
            time_period = 'afternoon'
        elif hour < 20:
            time_period = 'evening'
        else:
            time_period = 'night'
            
        if time_period in time_enhancements:
            if not any(word in prompt.lower() for word in ['light', 'lighting', 'sun', 'moon']):
                prompt += f", {time_enhancements[time_period]}"
                
        return prompt