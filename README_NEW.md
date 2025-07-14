# AI Wallpaper Generator v3.0

Ultra-high-quality 4K wallpaper generation using multiple AI models with automatic prompt generation and weather integration.

## Features

- **Multiple AI Models**: FLUX-Dev, DALL-E 3, GPT-Image-1, SDXL (with LoRA support)
- **Maximum Quality Always**: Each model uses its optimal pipeline for 4K output
- **Smart Theme System**: Weather-aware theme selection from 100+ themes
- **Unified CLI**: Single command interface for all functionality
- **Automatic Prompt Generation**: Uses DeepSeek-r1:14b for creative prompts
- **Multi-Desktop Support**: Works with XFCE, GNOME, KDE, and more

## Quick Start

```bash
# Generate wallpaper with random model
./ai-wallpaper generate --random-model

# Generate with specific model
./ai-wallpaper generate --model flux

# Generate with custom prompt
./ai-wallpaper generate --prompt "A serene mountain landscape at sunset"

# Test system components
./ai-wallpaper test --quick

# View configuration
./ai-wallpaper config --show

# List available models
./ai-wallpaper models --list
```

## Installation

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up API keys** (for DALL-E and GPT models):
   ```bash
   export OPENAI_API_KEY='your-key-here'
   ```

3. **Install Real-ESRGAN** (required for all models):
   ```bash
   cd /home/user/ai-wallpaper
   git clone https://github.com/xinntao/Real-ESRGAN.git
   cd Real-ESRGAN
   pip install basicsr facexlib gfpgan
   pip install -r requirements.txt
   python setup.py develop
   wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth -P weights
   ```

4. **Set up daily generation** (optional):
   ```bash
   ./setup_cron.sh
   ```

## Configuration

Configuration files are in `ai_wallpaper/config/`:

- `models.yaml` - Model settings and parameters
- `themes.yaml` - Theme database
- `weather.yaml` - Weather location and API settings  
- `settings.yaml` - General settings and desktop configuration
- `paths.yaml` - File paths and directories

## Model Pipelines

Each model has an optimized pipeline for maximum quality:

### FLUX-Dev
1. Generate at 1920x1088
2. Upscale to 8K (7680x4352) 
3. Downsample to 4K with Lanczos

### DALL-E 3
1. Generate at 1792x1024 (HD quality)
2. Crop to 16:9 (1792x1008)
3. Upscale 4x to 7168x4032
4. Downsample to 4K

### GPT-Image-1
1. Generate at 1536x1024
2. Crop to 16:9 (1536x864)
3. Upscale 4x to 6144x3456
4. Downsample to 4K

### SDXL
1. Generate at 1920x1024
2. Optional img2img refinement
3. Upscale 2x to 3840x2048
4. Adjust to 4K (3840x2160)

## Architecture

The refactored system features:

- **Modular Design**: Clean separation of concerns
- **Configuration-Driven**: No hardcoded values
- **Fail Loud**: Comprehensive error reporting
- **Resource Management**: Automatic VRAM management
- **Extensible**: Easy to add new models

## Troubleshooting

### Model not found
- FLUX will auto-download on first use
- Check `models.yaml` for model paths

### Out of memory
- Close other GPU applications
- The system manages VRAM automatically between models

### API errors
- Check your OPENAI_API_KEY is set
- Verify API quota and billing

### Desktop wallpaper not changing
- Check `settings.yaml` for desktop environment configuration
- Ensure `auto_set_wallpaper: true`

## Development

To add a new model:

1. Create model class inheriting from `BaseImageModel`
2. Implement required methods
3. Add to `models.yaml` configuration
4. Update `generate.py` to support the model

## License

MIT License - See LICENSE file for details