# AI Wallpaper Project - Critical Context for Claude

This is a project that generates a beautiful, ultra-high quality 4K desktop background every morning using advanced supersampling techniques.

It retrieves the weather and a random theme or themes and then uses deepseek-r1:14b via ollama to come up with a unique image prompt tailored to the chosen image model, incorporating local weather theme and date contexts for a beautiful, sometimes surreal, imaginative, amazing background.

## Hardware Specifications

CPU: Core i7 13700KF
RAM: 32GB DDR4
GPU: NVidia RTX 3090 with 24GB VRAM
SSD: 4TB m.2 NVMe 4th gen
SSD2: 1TB m.2 NVMe 4th gen
HDD: 3TB 7200rpm HDD
Display:0.0 - Sony Bravia 55" 4k OLED TV

## 🚨 CRITICAL: Error Philosophy

Remember, NO SILENT FAILURES. All errors should be loud and proud. Every part of every function should either work perfectly or the script should fail completely with verbose errors. We must keep it as easy to fix as possible.

Also remember, it does not matter how long it takes, or how much space it uses, the only priority in this project is amazing, incredibly detailed, ultimate high quality images.

## 🔧 Current Pipeline Architecture (v4.5.2)

### Stage 1: Base Generation
- SDXL at optimal resolution (typically 1344x768)
- 80 inference steps, single-pass (ensemble mode disabled)
- LoRA support with proper weight stacking

### Stage 1.5: Progressive Outpainting
- Handles extreme aspect ratios (up to 8x)
- Multi-pass approach with strength decay (0.95 → 0.76 → 0.57)
- Edge color analysis and pre-filling
- Massive blur radii (40% of new content) for seamless blending
- Centers original, expands outward

### Stage 2: Smart Quality Refinement
- Aggressive seam detection (color/gradient/frequency)
- Targeted inpainting on artifacts only
- 3x boundary width masks

### Stage 3: Upscaling
- Progressive Real-ESRGAN 2x steps
- Lossless PNG throughout (compress_level=0)

## ⚠️ Current Challenge: Seam Visibility

**Problem**: Visible seams in extreme aspect ratio expansions
**Root Cause**: Large expansion steps (2x, 1.5x) lose too much context
**Current Mitigation**: 
- Multi-pass outpainting with decreasing strength
- Edge color pre-filling
- Massive blur zones (120+ pixels)
- Denoising strength: 0.95 (critical - lower values only produce colors)

**Planned Solution**: Sliding Window Progressive Outpainting (SWPO)
- 200px windows with 80% overlap
- Maintains context throughout
- See plan1.md for implementation details

## 📁 Key Files & Their Roles

- `models/sdxl_model.py` - Pipeline orchestration, stage management
- `processing/aspect_adjuster.py` - Progressive outpainting implementation
- `processing/smart_detector.py` - Seam/artifact detection
- `config/resolution.yaml` - Quality/resolution settings
- `generation_metadata` dict - Tracks boundaries, seams, processing info

## 💡 Critical Technical Insights

1. **Denoising Strength**: Must be 0.8+ for content generation (0.35 only makes colors)
2. **Context Loss**: Each expansion step loses original context → seams
3. **Boundary Tracking**: Track where NEW content meets OLD (not just original position)
4. **Mask Blur**: Bigger is better - 40% of new content dimension minimum
5. **Multi-Pass**: Each pass refines the transition zone

## 🔍 Debugging Commands

```bash
# Visualize all stages
./generate.py --save-stages

# Check seam locations
# Look for generation_metadata['progressive_boundaries']
# and generation_metadata['seam_details']

# Force specific resolution
./generate.py --resolution 5376x768
```

## 🎯 Remember the Goals

1. **QUALITY OVER ALL** - No speed/space limits
2. **NO HIDDEN ERRORS** - Fail loud and clear
3. **ALL OR NOTHING** - Perfect or explode

When in doubt, choose quality. When facing errors, fail loudly. When implementing features, make them perfect or don't merge them.