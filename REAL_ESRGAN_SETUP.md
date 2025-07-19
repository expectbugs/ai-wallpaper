# Real-ESRGAN Installation

Real-ESRGAN is required for high-quality 4K upscaling across all AI models.

## Installation Steps

### Install from GitHub (Required)

```bash
cd /home/user/ai-wallpaper
git clone https://github.com/xinntao/Real-ESRGAN.git
cd Real-ESRGAN

# Install dependencies (ensure virtual environment is activated)
pip install basicsr facexlib gfpgan
pip install -r requirements.txt
python setup.py develop

# Download the RealESRGAN_x4plus model (REQUIRED)
wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth -P weights
```

## Verification

After installation, verify Real-ESRGAN is working:
```bash
cd /home/user/ai-wallpaper/Real-ESRGAN
python inference_realesrgan.py --help
```

Test with the AI wallpaper system:
```bash
cd /home/user/ai-wallpaper
./ai-wallpaper test --component upscaler
```

## Expected Locations

The upscaler component automatically searches for Real-ESRGAN in:
- `/home/user/ai-wallpaper/Real-ESRGAN/inference_realesrgan.py` (recommended)
- `/home/user/Real-ESRGAN/inference_realesrgan.py`
- `~/Real-ESRGAN/inference_realesrgan.py`

Path is configurable in `ai_wallpaper/config/paths.yaml`

## GPU Settings

Default settings (configurable in `ai_wallpaper/config/models.yaml`):
- Model: RealESRGAN_x4plus (downloaded automatically)
- Tile size: VRAM-aware auto-detection (typically 1024 pixels)
- Precision: fp32
- Scale: Model-dependent (2x for SDXL, 4x for FLUX/DALL-E)

The system automatically adjusts tile size based on available VRAM to prevent out-of-memory errors.