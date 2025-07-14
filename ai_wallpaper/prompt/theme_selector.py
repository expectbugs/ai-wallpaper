#!/usr/bin/env python3
"""
Theme Selection System for AI Wallpaper
Weighted random theme selection with weather integration
"""

import random
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path

from ..core import get_logger, get_config
from ..core.exceptions import ConfigurationError

class ThemeSelector:
    """Manages theme selection with weather-based weighting"""
    
    def __init__(self):
        """Initialize theme selector"""
        self.logger = get_logger(model="Theme")
        self.config = get_config()
        self.themes = self._load_themes()
        
    def _load_themes(self) -> Dict[str, Any]:
        """Load themes from configuration
        
        Returns:
            Theme database dictionary
        """
        themes = self.config.themes.get('categories', {})
        
        if not themes:
            raise ConfigurationError("No theme categories found in configuration")
            
        # Validate and calculate total weights
        total_weight = 0
        total_entries = 0
        
        for category_name, category in themes.items():
            if 'weight' not in category:
                raise ConfigurationError(f"Category '{category_name}' missing weight")
                
            if 'themes' not in category or not category['themes']:
                raise ConfigurationError(f"Category '{category_name}' has no themes")
                
            total_weight += category['weight']
            total_entries += len(category['themes'])
            
        self.logger.info(
            f"Loaded {len(themes)} categories with {total_entries} themes "
            f"(total weight: {total_weight})"
        )
        
        return themes
        
    def get_random_theme_with_weather(self, weather_context: Dict[str, Any]) -> Dict[str, Any]:
        """Select a random theme with weather influence
        
        Args:
            weather_context: Current weather information
            
        Returns:
            Dictionary with theme info and formatting
        """
        self.logger.info("Selecting random theme with weather context...")
        
        # Select category based on weights
        selected_category = self._select_category()
        
        # Select theme within category
        selected_theme = self._select_theme_from_category(
            selected_category,
            weather_context
        )
        
        # Apply optional chaos mode
        if self._should_apply_chaos():
            selected_theme = self._apply_chaos_mode(selected_theme)
            
        # Format theme instruction
        instruction = self._format_theme_instruction(selected_theme, weather_context)
        
        result = {
            'category': selected_category['name'],
            'theme': selected_theme,
            'instruction': instruction,
            'weather_influence': weather_context.get('mood', 'neutral')
        }
        
        self.logger.info(
            f"Selected: {selected_theme['name']} from {selected_category['name']} "
            f"(weather: {weather_context.get('mood', 'neutral')})"
        )
        
        return result
        
    def _select_category(self) -> Dict[str, Any]:
        """Select a category based on weights
        
        Returns:
            Selected category dictionary
        """
        categories = []
        weights = []
        
        for cat_name, cat_data in self.themes.items():
            categories.append({
                'name': cat_name,
                'data': cat_data
            })
            weights.append(cat_data['weight'])
            
        selected = self._weighted_random_choice(categories, weights)
        return selected
        
    def _select_theme_from_category(
        self, 
        category: Dict[str, Any], 
        weather_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Select theme from category with optional weather influence
        
        Args:
            category: Category dictionary
            weather_context: Weather information
            
        Returns:
            Selected theme
        """
        themes = []
        weights = []
        
        for theme_name, theme_data in category['data']['themes'].items():
            theme = {
                'name': theme_data.get('name', theme_name),
                'elements': theme_data.get('elements', []),
                'styles': theme_data.get('styles', []),
                'colors': theme_data.get('colors', []),
                'weight': theme_data.get('weight', 1)
            }
            
            # Apply weather influence to weights
            weight = theme['weight']
            weather_mood = weather_context.get('mood', 'neutral')
            
            # Boost certain themes based on weather
            if weather_mood == 'stormy' and 'storm' in theme_name.lower():
                weight *= 1.5
            elif weather_mood == 'serene' and 'peaceful' in theme_name.lower():
                weight *= 1.3
            elif weather_mood == 'mysterious' and 'mystery' in theme_name.lower():
                weight *= 1.4
                
            themes.append(theme)
            weights.append(weight)
            
        return self._weighted_random_choice(themes, weights)
        
    def _weighted_random_choice(self, items: List[Any], weights: List[float]) -> Any:
        """Select random item based on weights
        
        Args:
            items: List of items to choose from
            weights: Corresponding weights
            
        Returns:
            Selected item
        """
        if not items:
            raise ValueError("No items to choose from")
            
        if len(items) != len(weights):
            raise ValueError("Items and weights length mismatch")
            
        # Use random.choices for weighted selection
        return random.choices(items, weights=weights, k=1)[0]
        
    def _should_apply_chaos(self) -> bool:
        """Determine if chaos mode should be applied
        
        Returns:
            True if chaos mode should be applied
        """
        # 5% chance of chaos mode
        return random.random() < 0.05
        
    def _apply_chaos_mode(self, theme: Dict[str, Any]) -> Dict[str, Any]:
        """Apply chaos mode to mix random elements
        
        Args:
            theme: Original theme
            
        Returns:
            Chaos-modified theme
        """
        self.logger.info("CHAOS MODE ACTIVATED! Mixing random elements...")
        
        # Collect all possible elements from all themes
        all_elements = []
        all_styles = []
        all_colors = []
        
        for category in self.themes.values():
            for theme_data in category['themes'].values():
                all_elements.extend(theme_data.get('elements', []))
                all_styles.extend(theme_data.get('styles', []))
                all_colors.extend(theme_data.get('colors', []))
                
        # Create chaos theme
        chaos_theme = {
            'name': f"CHAOS: {theme['name']}",
            'elements': random.sample(all_elements, min(4, len(all_elements))),
            'styles': random.sample(all_styles, min(2, len(all_styles))),
            'colors': random.sample(all_colors, min(3, len(all_colors))),
            'weight': 1,
            'chaos_applied': True
        }
        
        return chaos_theme
        
    def _format_theme_instruction(
        self, 
        theme: Dict[str, Any], 
        weather_context: Dict[str, Any]
    ) -> str:
        """Format theme into instruction text
        
        Args:
            theme: Theme dictionary
            weather_context: Weather information
            
        Returns:
            Formatted instruction
        """
        elements = ', '.join(theme['elements'])
        styles = ', '.join(theme['styles'])
        colors = ', '.join(theme['colors'])
        
        instruction = f"""
THEME: {theme['name']}
CORE ELEMENTS: {elements}
ARTISTIC STYLE: {styles}
COLOR PALETTE: {colors}
WEATHER INTEGRATION: Current {weather_context.get('condition', 'clear')} conditions with {weather_context.get('mood', 'neutral')} atmosphere

Create a desktop wallpaper that combines these elements in an unexpected, creative way.
The image should tell a visual story incorporating the weather naturally into the theme.
Focus on photorealistic quality with rich detail and perfect composition.
"""
        
        if theme.get('chaos_applied'):
            instruction += "\nCHAOS MODE: Blend these wildly different elements into a cohesive, surreal masterpiece!"
            
        return instruction.strip()

# Global instance
_theme_selector: Optional[ThemeSelector] = None

def get_random_theme_with_weather(weather_context: Dict[str, Any]) -> Dict[str, Any]:
    """Get random theme with weather influence
    
    Args:
        weather_context: Weather information
        
    Returns:
        Theme selection result
    """
    global _theme_selector
    if _theme_selector is None:
        _theme_selector = ThemeSelector()
    return _theme_selector.get_random_theme_with_weather(weather_context)