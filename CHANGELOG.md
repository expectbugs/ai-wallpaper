# Changelog

## [0.1.0] - 2025-07-09

### Added
- Basic prompt generation with qwq-32b
- Image generation with SDXL/BigASP2
- XFCE4 wallpaper setting
- Command-line interface
- Cron scheduling
- Logging system

---

## [0.7.0] - 2025-07-11

### Changed
- Switched from qwq-32b to deepseek-r1:14b for prompt generation
- Simplified prompt format for deepseek-r1 model

### Fixed
- Issue with qwq-32b generating extra commentary/text

---

## [0.8.0] - 2025-07-11

### Added
- Weather context integration from NWS API
- Weather mood mapping for prompt enhancement
- Weather-based creative guidance
- Weather data caching system

### Enhanced
- Context system includes weather conditions
- Prompt generation uses weather context

## [0.8.1] - 2025-07-11

### Added
- Multi-step upscaling system
- Contrast and sharpness enhancement

### Fixed
- Upscaling quality issues
- Memory management

## [0.8.2] - 2025-07-11

### Fixed
- Wallpaper not changing on desktop
- Multi-monitor/workspace support

## [0.8.3] - 2025-07-11

### Fixed
- xfdesktop not refreshing wallpaper display
- Added `xfdesktop --reload` command

## [0.9.0] - 2025-07-12

### Added
- High-quality generation mode
- Two-stage generation process
- Native high resolution support
- Enhanced prompting

### Enhanced 
- Image quality
- Generation time
- Color grading

## [1.0.0] - 2025-07-12

### Changed
- Replaced SDXL/BigASP2 with FLUX-Dev
- Real-ESRGAN now required
- Resolution changed to 1920x1080
- Extended prompt generation to 300-400 words
- Using bfloat16 for FLUX optimization

### Added
- FLUX-Dev pipeline (12B parameters)
- Full 512 token support
- Two-stage generation
- Real-ESRGAN integration
- Enhanced prompting system

## [1.1.0] - 2025-07-13

### Fixed
- Wrong image dimensions (1980x1080 → 1920x1080)
- Model path to point to FLUX
- Weather cache resource leak
- Function order issues
- Logging consistency
- xFormers error handling

### Enhanced
- Quality settings
- Memory safety
- Prompt requirements

### Removed
- Face enhancement flag
- Silent fallbacks
- Try-except blocks

## [1.1.1] - 2025-07-13

### Fixed
- FLUX dimension requirements (1080 → 1088)
- Real-ESRGAN directory output handling
- Disabled FP8 quantization
- Disabled xFormers

### Verified
- 77 token warning is normal for FLUX
- DeepSeek integration working
- Quality pipeline functioning
- Architecture components validated

## [2.0.0] - 2025-07-13

### Changed
- Implemented 8K to 4K supersampling workflow
- Removed img2img refinement step
- 3-stage workflow: FLUX → Real-ESRGAN 8K → Lanczos 4K
- Processing time reduced from ~23 to ~17 minutes
- Enhanced prompting for creativity

### Added
- 8K supersampling
- Stage artifacts saved for comparison
- Perfect 16:9 aspect ratios

### Removed
- img2img refinement pass
- Complex multi-stage processing

## [2.0.1] - 2025-07-13

### Fixed
- FLUX scheduler compatibility (explicit FlowMatchEulerDiscreteScheduler)
- Weather cache corruption recovery (automatic recreation)
- Exponential backoff without max attempts (5 attempt limit)
- Ollama startup timeout (removed 30-second limit)
- Weight validation in theme selection
- Crontab installation validation
- Cron service check in setup script

### Enhanced
- Weather cache self-healing on corruption
- Unlimited Ollama startup wait time
- Robust cron setup with proper validation

---

## [3.0.0] - 2025-07-13

### Changed
- Replaced FLUX-Dev with OpenAI GPT-Image-1
- Two-branch architecture: Responses API and Direct API
- Eliminated DeepSeek from Responses API branch
- Direct GPT-4o integration for intelligent image generation

### Added
- `daily_wallpaper_gpt.py` - GPT-4o → GPT-Image-1 via Responses API
- `daily_wallpaper_gpt2.py` - Direct GPT-Image-1 API calls
- Organization verification handling
- Base64 image data processing
- Comprehensive context feeding to GPT-4o
- 3-minute API timeouts for high-quality generation

### Removed
- FLUX-Dev pipeline
- Ollama dependencies (Responses API branch)
- DeepSeek prompt generation (Responses API branch)
- Complex local model management

### Enhanced
- Generation speed (minutes vs 17+ minutes)
- Image quality via OpenAI's latest model
- Context intelligence through GPT-4o
- Theme and weather integration

---

## [4.0.0] - 2025-07-14

### Refactored
- Complete modular architecture with unified CLI
- Configuration-driven design (YAML files)
- Centralized resource management
- Model-agnostic pipeline system

### Added
- `ai-wallpaper` executable with Click CLI
- Support for all models: FLUX, DALL-E 3, GPT-Image-1, SDXL
- System configuration (venv paths, coordinates, settings)
- Intermediate stage saving capability
- Prominent seed display for reproducibility
- Environment variable expansion in configs
- Centralized upscaler integration

### Enhanced
- Weather coordinates now configurable
- Ollama path configurable 
- Prompt history path configurable
- All hardcoded paths moved to configuration
- Unified logging with VRAM monitoring
- Fail-loud error handling throughout

### Fixed
- Prompt word limit enforcement (removed truncation)
- Token limit warnings (FLUX uses T5, not CLIP)
- Full pipeline execution without failures
- Proper seed handling and display

### Maintained
- Original script functionality preserved
- No fallbacks, fail-loud philosophy
- All quality settings at maximum