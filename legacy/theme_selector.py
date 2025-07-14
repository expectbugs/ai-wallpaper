#!/usr/bin/env python3
"""
AI Wallpaper Theme Selection System
Weighted random theme selection with NO FALLBACKS, NO SILENT ERRORS
"""

import random
import os
import sys
from datetime import datetime

# CRITICAL: Set theme database path
THEME_DATABASE_PATH = "/home/user/ai-wallpaper/themes_database.txt"

def log(message):
    """Print timestamped message - must match main script logging"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    print(f"[{timestamp}] [THEME] {message}")

def load_theme_database():
    """Load and parse theme database - FAIL LOUDLY if any issues"""
    log("Loading theme database...")
    
    if not os.path.exists(THEME_DATABASE_PATH):
        log(f"ERROR: Theme database not found at {THEME_DATABASE_PATH}")
        sys.exit(1)
    
    themes = {}
    current_category = None
    
    with open(THEME_DATABASE_PATH, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            
            # Skip comments and empty lines
            if not line or line.startswith('#'):
                continue
                
            # Category header
            if line.startswith('[') and line.endswith(']'):
                category_parts = line[1:-1].split(':')
                if len(category_parts) != 2:
                    log(f"ERROR: Invalid category format at line {line_num}: {line}")
                    sys.exit(1)
                    
                current_category = category_parts[0]
                try:
                    category_weight = int(category_parts[1])
                except ValueError:
                    log(f"ERROR: Invalid weight at line {line_num}: {category_parts[1]}")
                    sys.exit(1)
                    
                themes[current_category] = {
                    'weight': category_weight,
                    'entries': []
                }
                continue
            
            # Theme entry
            if current_category:
                parts = line.split('|')
                if len(parts) != 5:
                    log(f"ERROR: Invalid theme format at line {line_num}: {line}")
                    log("Expected format: theme_name|core_elements|style_descriptors|color_palette|weight")
                    sys.exit(1)
                
                try:
                    theme_weight = int(parts[4])
                except ValueError:
                    log(f"ERROR: Invalid theme weight at line {line_num}: {parts[4]}")
                    sys.exit(1)
                
                theme_entry = {
                    'name': parts[0],
                    'elements': parts[1].split(','),
                    'styles': parts[2].split(','),
                    'colors': parts[3].split(','),
                    'weight': theme_weight
                }
                
                themes[current_category]['entries'].append(theme_entry)
    
    # Validate we have themes
    if not themes:
        log("ERROR: No themes loaded from database!")
        sys.exit(1)
        
    total_entries = sum(len(cat['entries']) for cat in themes.values())
    log(f"Loaded {len(themes)} categories with {total_entries} total theme entries")
    
    return themes

def weighted_random_choice(items, weights):
    """Select random item based on weights - FAIL LOUDLY if invalid"""
    if not items:
        log("ERROR: No items provided for weighted random choice")
        sys.exit(1)
        
    if len(items) != len(weights):
        log(f"ERROR: Items ({len(items)}) and weights ({len(weights)}) length mismatch")
        sys.exit(1)
    
    # Validate individual weights
    for i, weight in enumerate(weights):
        if not isinstance(weight, (int, float)):
            log(f"ERROR: Weight at index {i} is not numeric: {weight} (type: {type(weight).__name__})")
            sys.exit(1)
        if weight < 0:
            log(f"ERROR: Weight at index {i} is negative: {weight}")
            sys.exit(1)
        
    total_weight = sum(weights)
    if total_weight <= 0:
        log(f"ERROR: Total weight is {total_weight}, must be positive")
        sys.exit(1)
    
    r = random.uniform(0, total_weight)
    cumulative = 0
    
    for item, weight in zip(items, weights):
        cumulative += weight
        if r <= cumulative:
            return item
    
    # Should never reach here, but fail loudly if we do
    log("ERROR: Weighted random choice failed to select item")
    sys.exit(1)

def select_theme(themes, chaos_factor=0.1):
    """Select a theme using multi-layer randomization"""
    log("Starting theme selection process...")
    
    # Layer 1: Select category
    categories = list(themes.keys())
    category_weights = [themes[cat]['weight'] for cat in categories]
    
    selected_category = weighted_random_choice(categories, category_weights)
    log(f"Selected category: {selected_category}")
    
    # Layer 2: Select theme within category
    category_themes = themes[selected_category]['entries']
    if not category_themes:
        log(f"ERROR: Category {selected_category} has no themes")
        sys.exit(1)
        
    theme_weights = [theme['weight'] for theme in category_themes]
    selected_theme = weighted_random_choice(category_themes, theme_weights)
    log(f"Selected theme: {selected_theme['name']}")
    
    # Layer 3: Chaos injection (optional)
    if random.random() < chaos_factor:
        log("CHAOS MODE ACTIVATED! Adding random secondary theme...")
        # Pick a random category (could be same or different)
        chaos_category = random.choice(categories)
        if themes[chaos_category]['entries']:
            chaos_theme = random.choice(themes[chaos_category]['entries'])
            log(f"Chaos theme: {chaos_theme['name']} from category {chaos_category}")
            
            # Merge elements
            selected_theme = {
                'name': f"{selected_theme['name']}_{chaos_theme['name']}_fusion",
                'elements': selected_theme['elements'] + chaos_theme['elements'],
                'styles': selected_theme['styles'] + chaos_theme['styles'],
                'colors': selected_theme['colors'] + chaos_theme['colors'],
                'weight': selected_theme['weight'],
                'chaos_applied': True
            }
    
    return selected_theme, selected_category

def format_theme_for_deepseek(theme, weather_context):
    """Format selected theme into instructions for deepseek"""
    log("Formatting theme for deepseek prompt generation...")
    
    # Extract theme components
    elements = ', '.join(theme['elements'])
    styles = ', '.join(theme['styles'])
    colors = ', '.join(theme['colors'])
    
    # Build enhanced instruction
    instruction = f"""
THEME: {theme['name']}
CORE ELEMENTS: {elements}
ARTISTIC STYLE: {styles}
COLOR PALETTE: {colors}
WEATHER INTEGRATION: Current {weather_context['condition']} conditions with {weather_context['mood']} atmosphere

Create a desktop wallpaper that combines these elements in an unexpected, creative way.
The image should tell a visual story incorporating the weather naturally into the theme.
Focus on photorealistic quality with rich detail and perfect composition.
"""
    
    if theme.get('chaos_applied'):
        instruction += "\nCHAOS MODE: Blend the merged themes creatively!"
    
    return instruction

def get_random_theme_with_weather(weather_context):
    """Main function to get a theme selection with weather integration"""
    log("=== THEME SELECTION INITIATED ===")
    
    # Load themes
    themes = load_theme_database()
    
    # Select theme
    theme, category = select_theme(themes)
    
    # Format for deepseek
    theme_instruction = format_theme_for_deepseek(theme, weather_context)
    
    log(f"Theme selection complete: {theme['name']} from {category}")
    log("=== THEME SELECTION COMPLETE ===")
    
    return {
        'theme': theme,
        'category': category,
        'instruction': theme_instruction
    }

# Test function
if __name__ == "__main__":
    log("Running theme selector test...")
    
    # Mock weather context
    test_weather = {
        'condition': 'Partly Cloudy',
        'temperature': '72°F',
        'wind': '10 mph NW',
        'mood': 'pleasant'
    }
    
    # Test selection
    result = get_random_theme_with_weather(test_weather)
    
    log("Test result:")
    log(f"Category: {result['category']}")
    log(f"Theme: {result['theme']['name']}")
    log("Instruction preview:")
    print(result['instruction'][:200] + "...")