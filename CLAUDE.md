This is a project that generates a beautiful, ultra-high quality 4K desktop background every morning using advanced supersampling techniques.

It retrieves the weather and then uses deepseek-r1:14b via ollama to come up with a unique image prompt tailored to FLUX-Dev, incorporating local weather and date contexts for a beautiful, sometimes surreal, imaginative, amazing background.

The script uses a 3-stage quality pipeline:
1. **FLUX-Dev Generation**: Creates base image at 1920x1088 with 100 steps for maximum detail
2. **Real-ESRGAN 8K Upscaling**: AI upscales to 7680x4320 (8K) using RealESRGAN_x4plus model
3. **Lanczos Supersampling**: High-quality downsample to 3840x2160 (4K) for anti-aliasing perfection

My hardware is as follows:

CPU: Core i7 13700KF
RAM: 32GB DDR4
GPU: NVidia RTX 3090 with 24GB VRAM
SSD: 4TB m.2 NVMe 4th gen
SSD2: 1TB m.2 NVMe 4th gen
HDD: 3TB 7200rpm HDD
Display:0.0 - Sony Bravia 55" 4k OLED TV

Remember, NO SILENT FAILURES.  All errors should be loud and proud.  Every part of every function should either work perfectly or the script should fail completely with verbose errors.  We must keep it as easy to fix as possible.