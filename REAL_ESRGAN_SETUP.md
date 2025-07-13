# Real-ESRGAN Installation Guide

Real-ESRGAN is REQUIRED for ultra-high-quality 4K upscaling. The script will NOT work without it.

## Installation Steps

### Install from GitHub (Required)

```bash
cd /home/user/ai-wallpaper
git clone https://github.com/xinntao/Real-ESRGAN.git
cd Real-ESRGAN

# Install dependencies
pip install basicsr
pip install facexlib
pip install gfpgan
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

## Expected Locations

The script looks for Real-ESRGAN in these locations:
- `/home/user/ai-wallpaper/Real-ESRGAN/inference_realesrgan.py`
- `/home/user/Real-ESRGAN/inference_realesrgan.py`
- `~/Real-ESRGAN/inference_realesrgan.py`
- `/usr/local/bin/realesrgan-ncnn-vulkan`

## NO FALLBACK

If Real-ESRGAN is not found, the script will FAIL with clear error messages.
This is intentional - we want maximum quality or nothing.

## RTX 3090 Settings

The script uses these optimal settings for your RTX 3090:
- Model: RealESRGAN_x4plus (highest quality)
- Tile size: 1024 pixels (maximum for 24GB VRAM)
- Precision: fp32 (maximum quality)
- Scale: 2x (1920x1080 → 3840x2160)
- Face enhancement: Enabled