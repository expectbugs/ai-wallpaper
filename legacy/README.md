# Legacy Scripts

These are the original monolithic scripts that were used before the v4.0.0 modular refactor. They are preserved for reference and backward compatibility.

## Scripts

### `daily_wallpaper.py`
- **Original FLUX implementation** (v1.0.0 - v3.0.0)
- Complete pipeline: DeepSeek → FLUX-Dev → Real-ESRGAN → Wallpaper
- 8K supersampling workflow
- Weather integration and theme selection
- Direct execution: `python daily_wallpaper.py --run-now`

### `daily_wallpaper_dalle.py`
- **DALL-E 3 implementation** (v3.0.0+)
- OpenAI API integration
- Crop → upscale workflow
- High-quality HD generation

### `daily_wallpaper_gpt.py` / `daily_wallpaper_gpt2.py`
- **GPT-Image-1 implementations** (v3.0.0+)
- Two variants: Responses API and Direct API
- GPT-4o intelligent prompting
- Organization verification handling

### `theme_selector.py`
- **Theme selection system**
- Weather-based mood mapping
- Chaos mode implementation
- Used by all legacy scripts

## Migration Notes

These scripts work exactly as they did in their respective versions, but the new modular system in v4.0.0+ provides:

- Unified CLI interface
- Configuration-driven design
- Better error handling
- Resource management
- Modular architecture

For new installations, use the main `ai-wallpaper` executable instead of these legacy scripts.

## Usage (Legacy)

```bash
# FLUX generation
python daily_wallpaper.py --run-now

# DALL-E 3 generation  
python daily_wallpaper_dalle.py --run-now

# GPT-Image-1 generation
python daily_wallpaper_gpt.py --run-now
python daily_wallpaper_gpt2.py --run-now
```

## Requirements (Legacy)

- Same as main project
- All hardcoded paths and settings
- No configuration system
- Direct file modifications needed for customization