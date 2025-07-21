# AI Wallpaper Generator v4.5.4 - Usage Guide

High-quality wallpaper generation using AI models with weather integration.

## 🚀 Quick Start

```bash
# Generate with default settings (FLUX, 4K, random theme)
./ai-wallpaper generate

# Generate at specific resolution
./ai-wallpaper generate --resolution 8K

# Use different model
./ai-wallpaper generate --model sdxl --resolution ultrawide_4K
```

## 📋 Generate Command - All Options

```bash
./ai-wallpaper generate [OPTIONS]
```

### Core Options
- `--model [flux|dalle3|gpt-image-1|sdxl]` - Force specific model
- `--prompt TEXT` - Custom prompt (bypasses theme system)
- `--theme TEXT` - Force specific theme from themes.yaml
- `--seed INTEGER` - Seed for reproducible generation

### Resolution Control
- `--resolution TEXT` - Target resolution as WIDTHxHEIGHT or preset name
  - **Presets**: `1080p`, `1440p`, `4K`, `5K`, `8K`, `ultrawide_4K`, `ultrawide_1440p`, `portrait_4K`, `square_4K`
  - **Custom**: `3840x2160`, `5760x1080`, `2160x3840`, etc.
  - **Extreme Support**: 16K+, extreme ultrawide, any aspect ratio
  - **Examples**: `5140x600`, `7680x1080`, `15360x8640`

### SWPO Options (Sliding Window Progressive Outpainting)
- `--swpo/--no-swpo` - Enable/disable SWPO for extreme aspect ratios
- `--window-size INTEGER` - Window size for SWPO (default: 200 pixels)
- `--overlap-ratio FLOAT` - Overlap ratio for SWPO windows (default: 0.8)

### Quality Settings
- `--quality-mode [fast|balanced|ultimate]` - Quality mode (default: balanced)
- `--no-tiled-refinement` - Disable tiled refinement pass
- `--no-upscale` - Skip upscaling stage (generate at base resolution only)

### Generation Behavior
- `--random-model` - Use weighted random model selection
- `--random-params` - Randomize valid parameters within ranges
- `--no-wallpaper` - Generate only, don't set as desktop wallpaper
- `--save-stages` - Save intermediate stage images
- `--output PATH` - Custom output path

### Global Options
- `--config PATH` - Custom config directory
- `--verbose` - Enable verbose output
- `--dry-run` - Show execution plan without running

## 🎯 Models & Capabilities

### FLUX.1-dev (Default, Best Quality)
- **Resolution**: Any resolution with 1MP optimal generation
- **Time**: 5-15 minutes depending on target resolution
- **Requirements**: 24GB VRAM, Real-ESRGAN

### SDXL + LoRA (AI Art Focus)
- **Resolution**: SDXL optimal dimensions (1024-1536px) with intelligent upscaling
- **LoRA System**: 8 LoRAs with theme-based auto-selection
- **Time**: 8-20 minutes depending on resolution and quality mode
- **Requirements**: 16GB VRAM

### DALL-E 3 (Fastest, API-based)
- **Resolution**: Any resolution with intelligent cropping
- **Time**: 2-5 minutes depending on target resolution
- **Requirements**: OpenAI API key

### GPT-Image-1 (Latest OpenAI)
- **Resolution**: Flexible resolution with intelligent processing
- **Time**: 3-6 minutes depending on target resolution
- **Requirements**: OpenAI API key

## 📖 Usage Examples

### Basic Generation
```bash
# Default 4K generation
./ai-wallpaper generate

# Specific model and theme
./ai-wallpaper generate --model sdxl --theme "cyberpunk cityscape"

# Custom prompt at 8K
./ai-wallpaper generate --prompt "Mountain landscape at sunrise" --resolution 8K
```

### Resolution Examples
```bash
# Standard resolutions
./ai-wallpaper generate --resolution 1080p
./ai-wallpaper generate --resolution 4K
./ai-wallpaper generate --resolution 8K

# Ultrawide monitors
./ai-wallpaper generate --resolution ultrawide_4K
./ai-wallpaper generate --resolution 5760x1080

# Portrait orientation
./ai-wallpaper generate --resolution portrait_4K
./ai-wallpaper generate --resolution 2160x3840

# Custom resolutions
./ai-wallpaper generate --resolution 3440x1440
```

### Quality Control
```bash
# Ultimate quality (slowest, best results)
./ai-wallpaper generate --quality-mode ultimate

# Fast mode (no tiled refinement)
./ai-wallpaper generate --quality-mode fast

# Skip upscaling entirely
./ai-wallpaper generate --no-upscale

# Generate without setting wallpaper
./ai-wallpaper generate --no-wallpaper
```

### SWPO Examples (Extreme Aspect Ratios)
```bash
# Enable SWPO for extreme ultrawide (10:1 aspect ratio)
./ai-wallpaper generate --resolution 21600x2160 --swpo

# Custom SWPO settings for faster generation
./ai-wallpaper generate --resolution 10240x1080 --swpo --window-size 300 --overlap-ratio 0.7

# Disable SWPO to use original method
./ai-wallpaper generate --resolution 5376x768 --no-swpo
```

### Advanced Options
```bash
# Random model with random parameters
./ai-wallpaper generate --random-model --random-params

# Reproducible generation
./ai-wallpaper generate --seed 42 --model flux

# Save all intermediate stages
./ai-wallpaper generate --save-stages --output /custom/path/

# Verbose output with dry run
./ai-wallpaper --verbose --dry-run generate --model sdxl
```

## 🔧 Other Commands

### Test System
```bash
./ai-wallpaper test                           # Test everything
./ai-wallpaper test --model flux --quick      # Quick model test
./ai-wallpaper test --component prompt        # Test specific component
```

### Configuration
```bash
./ai-wallpaper config --show                  # View current config
./ai-wallpaper config --validate              # Validate config files
```

### Model Management
```bash
./ai-wallpaper models --list                  # List available models
./ai-wallpaper models --info sdxl             # Show model details
./ai-wallpaper models --check flux            # Check model readiness
```

## ⚙️ Configuration Files

Located in `ai_wallpaper/config/`:

- **`models.yaml`** - Model settings and pipeline configurations
- **`resolution.yaml`** - Resolution presets, quality modes, tiled refinement
- **`system.yaml`** - Python venv, weather coordinates, paths
- **`themes.yaml`** - Theme categories and definitions (60+ themes)
- **`weather.yaml`** - Weather API settings
- **`paths.yaml`** - Directory paths
- **`settings.yaml`** - Desktop environment settings

## 🔄 Automated Scheduling

```bash
# Set up daily generation at 6 AM
./setup_cron.sh

# Manual cron setup
crontab -e
# Add: 0 6 * * * /home/user/ai-wallpaper/run_ai_wallpaper.sh
```

## 🎨 Theme System

60+ curated themes across 10 categories:
- **LOCAL_MEDIA** (30%): Star Trek, Marvel, Doctor Who, Final Fantasy
- **GENRE_FUSION** (25%): Cross-universe mashups
- **ATMOSPHERIC** (20%): Weather-enhanced themes
- **SPACE_COSMIC** (12%): Cosmic horror and space phenomena
- **And 6 more categories** with hundreds of theme variations

## 🚨 Troubleshooting

```bash
# Verbose mode for debugging
./ai-wallpaper --verbose generate

# Test specific model
./ai-wallpaper test --model sdxl

# Validate configuration
./ai-wallpaper config --validate

# Check environment
./ai-wallpaper test --quick
```

## 📊 Quality Modes

- **Fast**: Basic generation, no tiled refinement (~5-8 minutes)
- **Balanced**: Standard quality with moderate refinement (~8-15 minutes)  
- **Ultimate**: Maximum quality with full tiled refinement (~15-25 minutes)

Time varies by model and target resolution. Higher resolutions take longer but produce superior results.

## 🚀 Advanced Features (v4.5.4)

### Extreme Resolution Support
- **Extreme Resolutions**: Generate at 16K+, ultrawide, any aspect ratio
- **SWPO**: Sliding Window Progressive Outpainting for seamless extreme expansions
  - 200px windows with 80% overlap (configurable)
  - Maintains context throughout expansion
  - No visible seams even at extreme ratios
- **VRAM Management**: Automatic strategy selection:
  - Full refinement for images that fit in VRAM
  - Tiled refinement with adaptive tile sizing
  - Smart memory management for large images

### Quality Enhancements
- **Consistent Denoising**: All stages use 60-80 step generation
- **Adaptive Refinement**: Stronger refinement after extreme aspect adjustments
- **Smart Blur Scaling**: Transition zones scale with expansion size
- **Pipeline Optimization**: Generate → Aspect Adjust → Refine → Upscale order