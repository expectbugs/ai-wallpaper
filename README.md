# AI Wallpaper Generator

Ultra-high-quality wallpaper generation at any resolution using AI models with weather integration and automated scheduling.

## Features

- **Multiple AI Models**: FLUX.1-dev, DALL-E 3, GPT-Image-1, SDXL with Juggernaut XL v9 + 8 LoRAs
- **NO RESOLUTION LIMITS**: Generate at ANY resolution - 16K+, extreme ultrawide, any aspect ratio
- **Dynamic Resolution Support**: Intelligent VRAM-based strategy selection with automatic fallbacks
- **Resolution Presets**: 1080p to 8K, ultrawide, portrait, and unlimited custom dimensions
- **Quality Modes**: Fast, balanced, and ultimate quality with seamless tiled refinement
- **Weather Integration**: Real-time weather data influences artistic themes and moods
- **Theme System**: 60+ curated themes across 10 categories with chaos mode
- **Smart Prompting**: DeepSeek-r1:14b generates creative, contextual prompts
- **Automated Scheduling**: Cron integration for daily wallpaper changes
- **Desktop Integration**: XFCE4 multi-monitor/workspace support

## Quick Start

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Set Up Real-ESRGAN** (required for FLUX):
   ```bash
   git clone https://github.com/xinntao/Real-ESRGAN.git
   cd Real-ESRGAN
   pip install basicsr facexlib gfpgan -r requirements.txt
   python setup.py develop
   wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth -P weights
   ```

3. **Configure API Keys** (for DALL-E/GPT models):
   ```bash
   export OPENAI_API_KEY="your-key-here"
   ```

4. **Generate Wallpaper**:
   ```bash
   ./ai-wallpaper generate
   ```

## Usage

### Basic Generation
```bash
# Generate with default model (FLUX) at 4K
./ai-wallpaper generate

# Use specific model
./ai-wallpaper generate --model dalle3

# Random model selection
./ai-wallpaper generate --random-model
```

### Resolution Control
```bash
# Generate at specific resolution
./ai-wallpaper generate --resolution 3840x2160

# Use resolution presets
./ai-wallpaper generate --resolution 4K
./ai-wallpaper generate --resolution ultrawide_4K
./ai-wallpaper generate --resolution 8K

# Portrait orientation for vertical monitors
./ai-wallpaper generate --resolution portrait_4K

# Custom ultrawide setup
./ai-wallpaper generate --resolution 5760x1080
```

### Quality Settings
```bash
# Ultimate quality mode (slower but best results)
./ai-wallpaper generate --quality-mode ultimate

# Balanced quality (default)
./ai-wallpaper generate --quality-mode balanced

# Fast mode (no tiled refinement)
./ai-wallpaper generate --quality-mode fast

# Disable tiled refinement specifically
./ai-wallpaper generate --no-tiled-refinement
```

### Other Options
```bash
# Save intermediate stages
./ai-wallpaper generate --save-stages

# Dry run (show plan)
./ai-wallpaper --dry-run generate

# List available models
./ai-wallpaper models --list

# Show configuration
./ai-wallpaper config --show

# Test system
./ai-wallpaper test
```

## Configuration

All settings are in YAML files under `ai_wallpaper/config/`:

- `models.yaml` - Model settings and pipeline configurations
- `resolution.yaml` - Resolution presets, quality modes, tiled refinement settings
- `system.yaml` - Python venv, weather coordinates, paths
- `themes.yaml` - Theme categories and definitions
- `weather.yaml` - Weather API settings
- `paths.yaml` - Directory paths
- `settings.yaml` - Desktop environment settings

## Models

### FLUX.1-dev (Default)
- **Pipeline**: Generate at optimal size → Real-ESRGAN upscaling → target resolution
- **Quality**: Maximum (100 steps, bfloat16)
- **Resolution**: Any resolution with intelligent dimension calculation
- **Requirements**: 24GB VRAM, Real-ESRGAN
- **Time**: ~5-15 minutes depending on target resolution

### DALL-E 3
- **Pipeline**: API generation → crop → Real-ESRGAN upscaling → target resolution
- **Quality**: HD, vivid style
- **Resolution**: Supports dynamic resolution with intelligent cropping
- **Requirements**: OpenAI API key
- **Time**: ~2-5 minutes

### GPT-Image-1
- **Pipeline**: Direct API or Responses API → crop → upscale → target resolution
- **Quality**: High quality
- **Resolution**: Flexible resolution support
- **Requirements**: OpenAI API key
- **Time**: ~3-6 minutes

### SDXL + LoRA
- **Pipeline**: Generate at optimal size → Real-ESRGAN multi-step → target resolution
- **Quality**: 8 LoRAs with theme-based auto-selection (anime, cyberpunk, sci-fi, fantasy + 4 general)
- **LoRA System**: Up to 5 simultaneous LoRAs, 4.0 total weight limit
- **Resolution**: Dynamic generation sizes with intelligent upscaling strategy
- **Requirements**: 16GB VRAM
- **Time**: ~8-20 minutes depending on target resolution and quality mode

## Scheduling

Set up daily wallpaper generation:

```bash
./setup_cron.sh
```

Runs at 6 AM daily by default. Edit crontab to customize timing.

## Architecture

```
ai_wallpaper/
├── cli/          # Click-based command interface
├── core/         # Configuration, logging, weather, wallpaper
├── models/       # Model implementations (FLUX, DALL-E, etc.)
├── prompt/       # DeepSeek prompt generation and theme selection
├── processing/   # Real-ESRGAN upscaling integration
├── utils/        # Resource management, file operations
└── config/       # YAML configuration files
```

## Requirements

- **GPU**: NVIDIA RTX 3090 (24GB) recommended for FLUX
- **Storage**: 50GB free space
- **OS**: Linux (XFCE4 desktop environment)
- **Python**: 3.10+ with virtual environment

## Legacy Scripts

The original monolithic scripts are preserved in `legacy/` for reference:
- `daily_wallpaper.py` - FLUX implementation
- `daily_wallpaper_dalle.py` - DALL-E 3 implementation  
- `daily_wallpaper_gpt.py` - GPT-Image-1 implementation

## License

Open source - see individual model licenses for restrictions.