# AI Wallpaper Generator

Ultra-high-quality 4K wallpaper generation using AI models with weather integration and automated scheduling.

## Features

- **Multiple AI Models**: FLUX.1-dev, DALL-E 3, GPT-Image-1, SDXL with LoRA
- **8K→4K Pipeline**: Generate at base resolution, upscale to 8K, downsample to perfect 4K
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

```bash
# Generate with default model (FLUX)
./ai-wallpaper generate

# Use specific model
./ai-wallpaper generate --model dalle3

# Random model selection
./ai-wallpaper generate --random-model

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
- `system.yaml` - Python venv, weather coordinates, paths
- `themes.yaml` - Theme categories and definitions
- `weather.yaml` - Weather API settings
- `paths.yaml` - Directory paths
- `settings.yaml` - Desktop environment settings

## Models

### FLUX.1-dev (Default)
- **Pipeline**: Generate 1920x1088 → Real-ESRGAN 8K → Lanczos 4K
- **Quality**: Maximum (100 steps, bfloat16)
- **Requirements**: 24GB VRAM, Real-ESRGAN
- **Time**: ~11 minutes

### DALL-E 3
- **Pipeline**: API generation → crop → Real-ESRGAN 4x → 4K
- **Quality**: HD, vivid style
- **Requirements**: OpenAI API key
- **Time**: ~2 minutes

### GPT-Image-1
- **Pipeline**: Direct API or Responses API → crop → upscale
- **Quality**: High quality
- **Requirements**: OpenAI API key
- **Time**: ~3 minutes

### SDXL + LoRA
- **Pipeline**: Generate 1920x1024 → Real-ESRGAN 2x → 4K
- **Quality**: LoRA auto-selection by theme
- **Requirements**: 16GB VRAM
- **Time**: ~8 minutes

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