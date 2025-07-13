# AI Wallpaper Generator

Automated system that generates desktop wallpapers daily at 6 AM using local language models and image generation.

## Features

- Generates unique wallpapers daily
- Uses deepseek-r1:14b for prompt generation
- 8K to 4K supersampling for upscaling
- Runs daily at 6 AM via cron
- Tracks history to prevent duplicates
- Fails immediately on errors
- Logs all operations

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
- FLUX.1-dev model
- Real-ESRGAN installed at `/home/user/ai-wallpaper/Real-ESRGAN/`
- XFCE4 desktop environment
- GPU with sufficient VRAM

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

## Process

1. **Prompt Generation** (~2-3 minutes)
   - Loads previous prompts for uniqueness
   - Fetches weather data from NWS API
   - Uses deepseek-r1:14b to create scene description
   - Considers date, season, weather, and mood

2. **Image Generation** (~17 minutes)
   - Stage 1: FLUX-Dev generates at 1920x1088 (100 steps, guidance 3.5)
   - Stage 2: Real-ESRGAN upscales 4x to 7680x4320
   - Stage 3: Lanczos downsampling to 3840x2160
   - Saves intermediate files and final wallpaper

3. **Wallpaper Setting** (< 1 second)
   - Sets wallpaper via xfconf-query
   - Handles DISPLAY environment
   - Verifies setting

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
- **Memory**: Real-ESRGAN uses tiling for processing

## File Structure
```
/home/user/ai-wallpaper/
├── daily_wallpaper.py      # Main script
├── run_daily_wallpaper.sh  # Cron wrapper
├── setup_cron.sh          # Setup helper
├── README.md              # This file
├── CHANGELOG.md           # Version history
├── CRON_README.md         # Scheduling guide
├── pieinthesky.md        # Design documentation
├── REAL_ESRGAN_SETUP.md  # Upscaler installation
├── prompt_history.txt     # Prompt archive
├── last_run.txt          # Latest status
├── logs/                 # Execution logs
└── images/               # Wallpaper gallery
```

## Technical Details

- **Generation Time**: ~17 minutes
- **Image Size**: 8-9 MB per wallpaper
- **Model Parameters**: FLUX.1-dev (12B), deepseek-r1:14b (14B)
- **Output Resolution**: 3840x2160 (4K)