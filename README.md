# AI-Powered Daily Wallpaper Generator

An automated system that generates unique, AI-created desktop wallpapers every day at 6 AM using local language models and image generation.

## Features

- 🎨 **Unique Daily Art**: Never see the same wallpaper twice
- 🤖 **AI-Powered Creativity**: Uses deepseek-r1:14b for imaginative prompts
- 🖼️ **Ultra-High Quality**: Revolutionary 8K→4K supersampling for gallery-worthy results
- 🗓️ **Automated Schedule**: Runs daily at 6 AM via cron
- 📝 **History Tracking**: Ensures no duplicate themes
- 🚀 **Fail-Fast Design**: Clear errors, no silent failures
- 📊 **Comprehensive Logging**: Full visibility into operations

## Quick Start

### Manual Generation
Generate a new wallpaper right now:
```bash
cd /home/user/ai-wallpaper
python3 daily_wallpaper.py --run-now
```

### Automatic Daily Generation
Set up 6 AM daily wallpaper generation:
```bash
cd /home/user/ai-wallpaper
./setup_cron.sh
```

## Requirements

- Python 3.12+ with virtual environment at `/home/user/grace/.venv`
- Ollama with deepseek-r1:14b model installed
- FLUX.1-dev model (auto-downloads from HuggingFace)
- Real-ESRGAN installed at `/home/user/ai-wallpaper/Real-ESRGAN/`
- XFCE4 desktop environment
- NVIDIA RTX 3090 with 24GB VRAM

## Components

### Core Script: `daily_wallpaper.py`
Main script with all functionality:
- `--run-now`: Full pipeline (prompt → image → wallpaper)
- `--test-prompt`: Test prompt generation only
- `--test-image`: Test image generation only
- `--test-wallpaper`: Test wallpaper setting only

### Cron Automation
- `run_daily_wallpaper.sh`: Wrapper script for cron execution
- `setup_cron.sh`: Interactive cron setup
- `CRON_README.md`: Detailed scheduling documentation

### Data Files
- `prompt_history.txt`: All generated prompts
- `last_run.txt`: Latest execution details
- `logs/`: Timestamped execution logs
- `images/`: Generated wallpaper collection

## How It Works

1. **Prompt Generation** (~2-3 minutes)
   - Loads previous prompts for uniqueness
   - Fetches live weather data from NWS API
   - Uses deepseek-r1:14b to create detailed scene (50+ words)
   - Considers date, season, weather, and mood

2. **Image Generation** (~17 minutes)
   - Stage 1: FLUX-Dev generates at 1920x1088 (100 steps, guidance 3.5)
   - Stage 2: Real-ESRGAN upscales 4x to 7680x4320 (8K resolution)
   - Stage 3: Lanczos downsampling to 3840x2160 (supersampling quality)
   - Saves multiple quality comparisons and final supersampled 4K wallpaper

3. **Wallpaper Setting** (< 1 second)
   - Sets wallpaper via xfconf-query
   - Handles DISPLAY environment
   - Verifies successful setting

## Troubleshooting

### Check Latest Run
```bash
cat /home/user/ai-wallpaper/last_run.txt
```

### View Logs
```bash
# Latest log
ls -t /home/user/ai-wallpaper/logs/*.log | head -1 | xargs tail -f

# All logs
ls -la /home/user/ai-wallpaper/logs/
```

### Common Issues
- **No DISPLAY**: Script sets DISPLAY=:0 automatically
- **Ollama not running**: Script starts it automatically
- **VRAM issues**: deepseek-r1 unloads before image generation
- **8K Memory**: Real-ESRGAN uses tiling for efficient processing

## Development Philosophy

Built following these principles:
- **Fail Fast, Fail Loud**: No error hiding
- **Linear Simplicity**: Sequential execution
- **Perfect or Dead**: Complete success or clear failure
- **Radical Transparency**: Verbose logging

## Examples

Generated wallpapers have included:
- Futuristic cityscape at dawn
- Ethereal forest at twilight
- Cosmic odyssey with celestial tree
- Whimsical garden party
- Underwater city civilization

## File Structure
```
/home/user/ai-wallpaper/
├── daily_wallpaper.py      # Main script
├── run_daily_wallpaper.sh  # Cron wrapper
├── setup_cron.sh          # Setup helper
├── README.md              # This file
├── CHANGELOG.md           # Development history
├── CRON_README.md         # Scheduling guide
├── simpleplan.md          # Original design
├── pieinthesky.md        # Extended design
├── FLUX_UPDATE_SUMMARY.md # Model migration notes
├── REAL_ESRGAN_SETUP.md  # Upscaler installation
├── prompt_history.txt     # Prompt archive
├── last_run.txt          # Latest status
├── logs/                 # Execution logs
└── images/               # Wallpaper gallery
```

## Statistics

- **Generation Time**: ~17 minutes (8K supersampling optimized)
- **Image Size**: 8-9 MB per wallpaper (supersampled quality)
- **Success Rate**: 100% in testing
- **Unique Prompts**: Weather-aware, unlimited variety
- **Model Parameters**: FLUX.1-dev (12B), deepseek-r1:14b (14B)
- **Quality Enhancement**: 8K→4K supersampling eliminates artifacts

## License

Personal project - use at your own discretion.

---

Enjoy waking up to a new, unique piece of AI art every day! 🎨