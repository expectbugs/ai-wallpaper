# Changelog

## [4.3.0]

### Added
- Downloaded and integrated 8 SDXL-compatible LoRAs: 4 general enhancement + 4 theme-specific
- Theme-specific LoRAs: anime slider, cyberpunk style, 70s sci-fi, fantasy slider
- General enhancement LoRAs: photorealistic slider, extremely detailed, face helper
- LoRA mapping system linking specific LoRAs to theme categories
- Comprehensive LoRA compatibility testing and validation
- Cross-platform compatibility system with dynamic path resolution
- PathResolver for OS-agnostic file system navigation
- ConfigLoader with environment variable overrides
- EnvironmentValidator for system compatibility checks
- WallpaperSetter with multi-desktop environment support

### Enhanced
- SDXL model configuration with proper 2048-dimension LoRA support
- Theme-based automatic LoRA selection for 6/10 theme categories
- LoRA weight ranges optimized for each style and purpose
- Trigger word integration for style-specific LoRAs
- Virtual environment auto-detection with multiple fallbacks
- Configuration system now fully dynamic and portable
- Model path discovery with hint-based resolution
- Shell scripts updated for cross-platform compatibility

### Fixed
- Removed corrupted SDXL detail enhancer LoRA file
- Updated models.yaml with verified working LoRA configurations
- Eliminated hardcoded paths throughout the system
- Virtual environment detection across different setups

---

## [4.2.0]

### Changed
- Upgraded SDXL pipeline with Juggernaut XL v9 for superior photorealistic output
- Implemented multi-LoRA stacking system for enhanced image quality
- Added comprehensive negative prompts to eliminate watercolor/artistic effects
- Enhanced theme-based automatic LoRA selection
- Optimized settings: 80+ steps, CFG 8.0, ensemble base/refiner pipeline
- Added photorealistic prompt enhancement with DSLR camera modifiers

### Added
- Downloaded and integrated photorealistic LoRAs: Better Picture v3, Photo Enhance v2, Photorealistic v1, Real Skin Slider
- Multi-LoRA adapter system with weight balancing (4.0 total weight limit)
- Advanced negative prompt targeting to prevent artistic/painted styles

### Fixed
- Resolved SDXL watercolor painting output issue - now produces photorealistic images
- Size mismatch warnings for LoRA compatibility (non-fatal, generation continues)

---

## [4.1.0]

### Fixed
- 33 critical bugs across all system components
- VRAM detection now shows accurate system usage via nvidia-smi
- FLUX model detection improved for various file structures
- Virtual environment activation for Real-ESRGAN execution
- Prompt history file locking and corruption prevention
- Memory cleanup order standardized across models
- Resource manager registration leaks
- Scheduler compatibility validation
- Pipeline stage validation for all models
- Quote validation messaging enhanced
- Log rotation implementation (100MB, 7 backups)
- Temporary file cleanup in all error paths
- Desktop environment detection race conditions
- Disk space validation before generation
- API key format validation
- Weather cache corruption recovery
- Thread safety for configuration singleton
- Subprocess zombie process prevention
- Float precision in theme weight selection
- XFCE multi-monitor property validation

### Enhanced
- Removed all prompt truncation (display and internal)
- Comprehensive error handling with detailed messages
- Proper resource cleanup and VRAM management
- Fail-loud philosophy consistently applied
- File handle management using context managers
- Upscaler tile size considers available VRAM
- Weather API backoff with maximum delay caps
- LoRA file validation before loading
- Metadata save failures now raise exceptions
- Special character handling in prompts
- Hardcoded /tmp paths replaced with tempfile

### Added
- Save intermediate stages functionality
- Standardized stage result dictionaries
- Pipeline stage validation system
- Concurrent write protection with file locking
- VRAM-aware processing optimizations
- Comprehensive testing suite verification

---

## [4.0.1]

### Fixed
- Theme override functionality (--theme flag now works correctly)
- Seed display using wrong logging level (critical → info)
- Theme selection logic properly respects forced theme parameter

### Enhanced
- Theme lookup system with find_theme_by_name() function
- Theme override validation with proper error handling

---

## [4.0.0]

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

---

## [3.0.0]

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

## [2.0.1]

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

## [2.0.0]

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

---

## [1.1.1]

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

---

## [1.1.0]

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

---

## [1.0.0]

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

---

## [0.9.0]

### Added
- High-quality generation mode
- Two-stage generation process
- Native high resolution support
- Enhanced prompting

### Enhanced 
- Image quality
- Generation time
- Color grading

---

## [0.8.3]

### Fixed
- xfdesktop not refreshing wallpaper display
- Added `xfdesktop --reload` command

---

## [0.8.2]

### Fixed
- Wallpaper not changing on desktop
- Multi-monitor/workspace support

---

## [0.8.1]

### Added
- Multi-step upscaling system
- Contrast and sharpness enhancement

### Fixed
- Upscaling quality issues
- Memory management

---

## [0.8.0]

### Added
- Weather context integration from NWS API
- Weather mood mapping for prompt enhancement
- Weather-based creative guidance
- Weather data caching system

### Enhanced
- Context system includes weather conditions
- Prompt generation uses weather context

---

## [0.7.0]

### Changed
- Switched from qwq-32b to deepseek-r1:14b for prompt generation
- Simplified prompt format for deepseek-r1 model

### Fixed
- Issue with qwq-32b generating extra commentary/text

---

## [0.1.0]

### Added
- Basic prompt generation with qwq-32b
- Image generation with SDXL/BigASP2
- XFCE4 wallpaper setting
- Command-line interface
- Cron scheduling
- Logging system