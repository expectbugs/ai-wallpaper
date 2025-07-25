# Changelog

## [4.5.4] - 2025-07-21 - Sliding Window Progressive Outpainting (SWPO)

### Major Feature - SWPO Implementation
- **Revolutionary Outpainting Method**: Replaced large expansion steps with sliding windows
  - 200px windows with 80% overlap for maximum context preservation
  - Eliminates visible seams in extreme aspect ratio expansions
  - Each window sees mostly existing content, ensuring perfect continuity
  - Supports expansions up to 8x aspect ratio change

### SWPO Features
- **Configurable Parameters**:
  - Window size: 100-300px (default: 200px)
  - Overlap ratio: 0.0-1.0 (default: 0.8)
  - Automatic CUDA cache clearing every 5 windows
  - Optional final unification pass for seamless results

- **CLI Integration**:
  - `--swpo/--no-swpo`: Enable/disable SWPO
  - `--window-size`: Set window size in pixels
  - `--overlap-ratio`: Set overlap between windows
  - CLI parameters override configuration file settings

### Technical Implementation
- **Smart Strategy Calculator**: Automatically plans optimal window progression
- **Gradient Masking**: Smooth transitions between windows with edge blur
- **Memory Management**: Periodic CUDA cache clearing prevents OOM errors
- **Stage Saving**: Full support for `--save-stages` to visualize each window
- **Automatic Fallback**: Seamlessly falls back to original method when disabled

### Performance Impact
- **Old Method**: 1344x768 → 5376x768 in 3 large steps (visible seams)
- **SWPO Method**: Same expansion in ~20 small windows (seamless result)
- **Generation Time**: Linear with expansion size (more windows = more time)
- **Quality**: Dramatically improved for extreme aspect ratios

### Bug Fixes
- Fixed config access using dictionary notation instead of attribute access
- Ensured CLI parameters properly flow through to SDXL model
- Added proper None checking for CLI override logic

## [4.5.3] - 2025-07-20 - Comprehensive Fail-Loud Refactor

### Major Refactoring - Fail-Loud Philosophy Implementation
- **Complete Error Handling Overhaul**: Transformed entire codebase to follow strict fail-loud principles
  - Replaced 50+ instances of functions returning None/False with proper exceptions
  - Eliminated 15+ empty except blocks
  - Removed all silent fallback behaviors (except intentional CPU fallback support)
  - Every error now raises specific exceptions with verbose, actionable messages

### Critical Fixes
- **Version Consistency**: Updated all version strings from 3.0.0 → 4.5.2
- **Wallpaper Setter**: Complete rewrite - all methods now raise WallpaperError on failure
- **Logger**: Fixed silent fallbacks and improved error context
- **Path Resolver**: Added proper error handling for directory creation
- **File Manager**: All operations now raise FileManagerError instead of returning None/True

### Infrastructure Improvements
- **Dynamic Path Configuration**: Replaced hardcoded paths with environment variables
  - `${AI_WALLPAPER_ROOT}` for project root
  - `${AI_WALLPAPER_VENV}` for virtual environment
  - `${HF_HOME}` for Hugging Face cache
- **Missing Module Exports**: Added AspectAdjuster, HighQualityDownsampler, TiledRefiner to processing/__init__.py
- **Exception Messages**: Updated VRAMError to provide actionable guidance

### 📋 Code Quality
- **85+ Issues Fixed**: Comprehensive code review identified and fixed all silent failures
- **Consistent Error Patterns**: Established uniform exception handling across all modules
- **Better Debugging**: Every exception includes context, values, and resolution guidance

### ✅ Validation
- Successfully tested with ultra-wide 7680x2160 generation
- All pipeline stages working correctly with proper error propagation
- No silent failures possible - system either succeeds perfectly or fails loudly

## [4.5.2] - 2025-07-20 - Enhanced Seam Detection & Multi-Pass Outpainting

### Quality Improvements
- **Multi-Pass Outpainting**: Implemented progressive strength reduction for natural transitions
  - 3-5 passes per expansion step with decreasing denoising strength
  - Each pass refines the transition zone for seamless blending
  - Adaptive pass count based on expansion ratio

- **Edge Color Pre-filling**: Canvas now pre-filled with analyzed edge colors
  - Analyzes dominant colors from image edges before outpainting
  - Provides coherent starting point for inpainting
  - Reduces color discontinuities at boundaries

- **Aggressive Seam Detection**: Complete rewrite of SmartDetector
  - Multiple detection methods: color difference, gradient discontinuity, frequency analysis
  - Lower detection thresholds for catching subtle seams
  - Wider refinement masks (3x boundary width) for better blending

### Technical Changes
- **Parameter Fixes**: 
  - Fixed refiner using wrong parameter name ('strength' → 'denoising_strength')
  - Added proper metadata initialization for boundary tracking
- **Denoising Strength Optimization**: 
  - Adjusted from 1.0 → 0.95 for better content generation
  - Previous 0.35 setting only produced solid colors
- **Boundary Tracking**: Now tracks actual seam locations where new content meets old
  - Previously only tracked original content position
  - Critical for accurate seam detection and refinement

### Known Issues
- Seams still visible in some extreme aspect ratio expansions
- Working on new Sliding Window Progressive Outpainting (SWPO) approach

---

## [4.5.1] - 2025-07-20 - Zero Quality Loss Update

### Critical Fix
- **Lossless PNG Saves**: Fixed catastrophic quality loss where 20K+ images were only 0.5MB
  - TiledRefiner was missing PNG format specification
  - Created centralized `save_lossless_png()` utility
  - All saves now use `compress_level=0` for zero compression
  - File sizes now correct: 20K image ~30-80MB as expected

### Technical Changes
- Removed all JPEG-style `quality` parameters from PNG saves
- Added explicit `'PNG'` format specification everywhere
- Implemented file size validation to catch quality issues
- Standardized all save operations through lossless utility

---

## [4.5.0] - 2025-07-20 - Ultimate Quality System: No Limits Edition

### Major Features
- **NO SIZE LIMITS**: Completely removed all arbitrary resolution restrictions
  - Dynamic VRAM-based strategy selection (full → tiled → CPU offload)
  - Processes ANY resolution with automatic resource management
  - 16K+ images now possible with CPU offloading
- **Pipeline Reordering**: Fixed critical quality issue with seams
  - New order: Generate → Aspect Adjust → Refine → Upscale
  - Prevents visible seams in extreme aspect ratios
  - Aspect adjustment now happens BEFORE refinement (Stage 1.5)
- **Progressive Outpainting**: Seamless extreme aspect ratio support
  - Handles up to 8x aspect ratio difference
  - Adaptive blur scaling (25% of new content dimension)
  - Multiple expansion steps with intelligent blending
- **Tiled Refinement System**: Ultra-quality mode for large images
  - VRAM-aware tile size calculation
  - Overlapping tiles with seamless blending
  - Automatic fallback to smaller tiles when needed

### Critical Fixes
- **Ensemble Mode Disabled**: Fixed partial denoising issue
  - Was causing noise/artifacts in base generation
  - Now uses single-pass generation with full 80 steps
  - Consistent quality across all image areas
- **Seam Elimination**: Fixed visible seams in progressive expansion
  - Enhanced adaptive blur calculation
  - Increased refinement strength after extreme adjustments
  - Better transition zones between expansion steps
- **Quality Consistency**: All stages now use unified parameters
  - Base generation: 80 steps (was interrupted at ~64)
  - Outpainting: 80 steps with full denoising
  - Refinement: Adaptive strength based on aspect changes

### Enhanced Features
- **VRAMCalculator**: Accurate memory requirement predictions
- **CPUOffloadRefiner**: Last-resort processing for extreme sizes
- **Adaptive Parameters**: Blur, steps, and guidance scale with expansion
- **Progressive Strategy**: Intelligent multi-step expansion planning
- **Error Handling**: New VRAMError exception with helpful messages

### 📋 Configuration Updates
- Removed `max_refinement_pixels` limit entirely
- Added VRAM management settings
- Enhanced progressive outpainting parameters
- Increased base blur (32→64) and steps (60→80)
- Added adaptive parameter multipliers

### Technical Details
- Refactored `AspectAdjuster` with strict pipeline validation
- Updated `ResolutionManager` with progressive strategies
- Modified `SdxlModel` with new pipeline stages
- Integrated dynamic refinement strategy selection

---

## [4.4.1] - Critical Resolution and Quality Fixes

### Fixed
- **Real-ESRGAN Error Reporting**: Now shows actual stderr/stdout instead of generic subprocess errors
- **SDXL Generation Quality**: Always uses trained dimensions (1344x768, 1536x640, etc.) instead of scaled-up sizes
- **LoRA Selection Intelligence**: Face helper LoRA only loads for themes likely to contain people/faces
- **Resolution Intent Respect**: User-specified resolutions honored without forced 4K upscaling
- **Output Directory Bug**: Final images now properly saved to configured images directory
- **Extreme Aspect Ratio Support**: Fixed upscaling strategy for ultrawide resolutions (3.5:1+)

### Enhanced
- **Quality-First Architecture**: SDXL generates at exact trained resolutions for maximum quality
- **Content-Aware LoRAs**: Face enhancement only applied to LOCAL_MEDIA, GENRE_FUSION, ANIME_MANGA themes
- **Better Error Visibility**: Real-ESRGAN failures now show actual error messages with input dimensions
- **Intelligent Upscaling**: Strategy ensures both dimensions exceed target before cropping
- **Filename Conventions**: Output files include resolution in name (e.g., sdxl_2560x1440_timestamp.png)

---

## [4.4.0] - Phase 2: Dynamic Resolution Support

### Added
- **Resolution Management System**: Complete infrastructure for configurable resolutions
  - ResolutionManager class with model-specific optimal dimension calculations
  - Support for SDXL (1024-1536px optimal) and FLUX (1MP optimal) constraints
  - Integer-only upscaling strategy to preserve maximum quality
  - 10 resolution presets from 1080p to 8K (including ultrawide and portrait)
- **CLI Resolution Parameters**: New command-line options for resolution control
  - `--resolution` parameter accepts WIDTHxHEIGHT or preset names (4K, ultrawide_4K, etc.)
  - `--quality-mode` parameter with fast/balanced/ultimate options
  - `--no-tiled-refinement` flag for faster processing
- **Dynamic Model Integration**: Models now support configurable generation sizes
  - BaseImageModel updated with ResolutionManager integration
  - Automatic optimal generation size calculation based on target resolution
  - Pre-calculated upscaling strategies with Real-ESRGAN integration
- **Resolution Configuration**: New resolution.yaml config file
  - Quality modes configuration with tiled refinement settings
  - Upscaling preferences and maximum resolution limits
  - Tiled refinement parameters (tile size, overlap, passes)

### Enhanced
- **SDXL Model Upgrades**: Dynamic resolution support in SDXL pipeline
  - Generation stage now uses calculated optimal dimensions instead of hardcoded 1344x768
  - Upscaling stage uses pre-calculated strategy with multiple Real-ESRGAN steps
  - Added _apply_realesrgan and _apply_center_crop methods for precise processing
  - Maintains backward compatibility with fallback to original behavior
- **Intelligent Upscaling**: Multi-step upscaling with quality preservation
  - Uses 2x Real-ESRGAN steps where possible for integer scaling
  - Center cropping for exact target dimensions
  - No stretching or squashing - maintains aspect ratios intelligently

### Technical Implementation
- Resolution calculations preserve aspect ratios and optimize for model capabilities
- SDXL always generates at exact trained dimensions for maximum quality
- FLUX calculations ensure 16-divisible dimensions and stay within 2048px limits
- Upscaling strategy handles extreme aspect ratios with multiple 2x steps
- Full integration with existing CLI and generation workflow

---

## [4.3.1]

### Fixed
- SDXL LoRA architecture incompatibility between Juggernaut XL v9 and standard LoRAs
- LoRA scanning now uses `/loras-sdxl/` directory for SDXL-compatible LoRAs
- Updated LoRA metadata to match available SDXL-specific LoRA files
- Theme presets now reference correct SDXL-compatible LoRA names
- Size mismatch errors eliminated - LoRAs now load successfully

### Enhanced
- Expanded theme-specific LoRA mappings for better style coverage
- Added support for 8 additional theme categories in LoRA selection
- Multi-LoRA loading with proper weight distribution (2.6/4.0 total weight)

---

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