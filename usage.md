# AI Wallpaper Generator - Comprehensive Usage Overview

Based on my thorough analysis of your AI wallpaper project, here's a complete overview of all the ways you can use it, including every setting, option, and configuration available.

## 🚀 Main CLI Commands

### 1. **Generate Command** - Core Wallpaper Creation

```bash
./ai-wallpaper generate [OPTIONS]
```

**All Generate Options:**
- `--prompt TEXT` - Custom prompt (bypasses theme system)
- `--theme TEXT` - Force specific theme from themes.yaml
- `--model [flux|dalle3|gpt-image-1|sdxl]` - Force specific model
- `--random-model` - Use weighted random model selection
- `--random-params` - Randomize valid parameters within ranges
- `--seed INTEGER` - Seed for reproducible generation
- `--no-upscale` - Skip upscaling stage (generate base resolution only)
- `--no-wallpaper` - Generate only, don't set as desktop wallpaper
- `--save-stages` - Save intermediate stage images
- `--output PATH` - Custom output path

**Global Options (work with any command):**
- `--config PATH` - Custom config directory
- `--verbose` - Enable verbose output
- `--dry-run` - Show execution plan without running

### 2. **Test Command** - System Verification

```bash
./ai-wallpaper test [OPTIONS]
```

**Test Options:**
- `--model [flux|dalle3|gpt-image-1|sdxl]` - Test specific model
- `--component [prompt|image|wallpaper|theme]` - Test specific component
- `--quick` - Fast test mode (environment validation only)

### 3. **Config Command** - Configuration Management

```bash
./ai-wallpaper config [OPTIONS]
```

**Config Options:**
- `--show` - Display current configuration
- `--validate` - Validate all config files
- `--set KEY=VALUE` - Set configuration value (not yet implemented)
- `--reset` - Reset to defaults (not yet implemented)

### 4. **Models Command** - Model Management

```bash
./ai-wallpaper models [OPTIONS]
```

**Model Options:**
- `--list` - List available models with status
- `--info MODEL_NAME` - Show detailed info for specific model
- `--check MODEL_NAME` - Check if model is ready to use
- `--install MODEL_NAME` - Download/install specific model (not yet implemented)

## 🎯 Model Configurations & Pipelines

### **FLUX.1-dev** (Default, Best Quality)
- **Resolution Pipeline**: 1920x1088 → Real-ESRGAN 8K → Lanczos 4K
- **Quality Settings**: 100 steps, bfloat16, guidance 3.5
- **Requirements**: 24GB VRAM (RTX 3090), Real-ESRGAN
- **Generation Time**: ~11 minutes
- **Configurable Parameters**:
  - `steps_range: [50, 100]` - For random parameter selection
  - `guidance_range: [2.0, 4.0]` - Guidance scale randomization
  - `torch_dtype: bfloat16` - Precision setting
  - `scheduler: FlowMatchEulerDiscreteScheduler` - Required for FLUX

### **DALL-E 3** (Fastest, API-based)
- **Resolution Pipeline**: 1792x1024 → crop to 16:9 → Real-ESRGAN 4x → 4K
- **Quality Settings**: HD quality, vivid style
- **Requirements**: OpenAI API key
- **Generation Time**: ~2 minutes
- **Configurable Parameters**:
  - `quality: "hd"` - Image quality setting
  - `style: "vivid"` - Style preference
  - `timeout: 60` - API timeout

### **GPT-Image-1** (Latest OpenAI Model)
- **Two Variants**:
  - Direct API: Direct GPT-Image-1 generation
  - Responses API: GPT-4o → GPT-Image-1 workflow
- **Resolution Pipeline**: 1536x1024 → crop → upscale → 4K
- **Requirements**: OpenAI API key
- **Generation Time**: ~3 minutes

### **SDXL + LoRA** (AI Art Focus)
- **Resolution Pipeline**: 1920x1024 → optional img2img → Real-ESRGAN 2x → 4K
- **LoRA Features**: Auto-selection by theme category
- **Requirements**: 16GB VRAM
- **Generation Time**: ~8 minutes
- **Configurable Parameters**:
  - `scheduler_options`: Multiple scheduler choices for randomization
  - `steps_range: [30, 75]`
  - `guidance_range: [5.0, 12.0]`
  - LoRA weights and auto-selection by theme

## ⚙️ Configuration Files Deep Dive

### **models.yaml** - Model & Pipeline Settings
- **Model Definitions**: All 4 models with complete configuration
- **Pipeline Settings**: 3-stage quality pipelines for each model
- **Random Selection**: Weighted model selection (flux: 35, dalle3: 25, gpt_image_1: 25, sdxl: 15)
- **Quality Settings**: Always maximum (jpeg_quality: 100, png_compression: 0)
- **API Configuration**: Environment variable expansion for keys

### **settings.yaml** - Application Behavior
- **Wallpaper Settings**: 
  - `auto_set_wallpaper: true/false`
  - Desktop environment auto-detection and commands
  - Support for: XFCE, GNOME, KDE, MATE, Cinnamon, LXDE, LXQT, i3, Sway, Hyprland, macOS, Windows
- **Output Settings**:
  - `final_resolution: [3840, 2160]` - 4K output
  - `format: "png"` - File format
  - `save_stages: false` - Intermediate stage saving
  - Archive settings with automatic cleanup
- **Performance Settings**:
  - GPU memory management (`memory_fraction: 0.95`)
  - CPU thread control
  - Aggressive cleanup options
- **Logging Configuration**: Full logging control with file rotation

### **paths.yaml** - File System Locations
- **Directory Paths**: All directories with absolute paths
- **Model Paths**: Priority-ordered model file locations
- **Cache Directories**: Weather cache, general cache
- **Output Paths**: Default and custom output locations

### **system.yaml** - System Integration
- **Python Environment**: Virtual environment path configuration
- **Ollama Integration**: Ollama executable path
- **Weather Configuration**: GPS coordinates for weather API
- **Generation Settings**: Seed display, prompt history management

### **themes.yaml** - Theme Database (100+ Themes)
- **10 Theme Categories** with weights:
  - LOCAL_MEDIA (30% weight): Star Trek, Marvel, Doctor Who, Final Fantasy, etc.
  - GENRE_FUSION (25%): Cross-universe mashups
  - ATMOSPHERIC (20%): Weather-enhanced themes
  - SPACE_COSMIC (12%): Cosmic horror and space phenomena
  - ARCHITECTURAL (10%): Impossible architecture
  - ANIME_MANGA (15%): Anime-inspired themes
  - TEMPORAL (15%): Time period fusions
  - ABSTRACT_CHAOS (12%): Surreal combinations
  - EXPERIMENTAL (10%): Pure chaos mode
  - DIGITAL_PROGRAMMING (10%): Tech and code themes

### **weather.yaml** - Weather Integration
- **Location Settings**: GPS coordinates for weather data
- **API Configuration**: National Weather Service API setup
- **Weather Mapping**: Conditions to artistic moods
- **Cache Settings**: Weather data caching with expiration times

## 🔄 Usage Workflows

### **Basic Generation**
```bash
# Default generation (FLUX model, random theme)
./ai-wallpaper generate

# Quick generation with DALL-E 3
./ai-wallpaper generate --model dalle3

# Custom prompt
./ai-wallpaper generate --prompt "A serene mountain landscape at golden hour"
```

### **Advanced Generation**
```bash
# Random model with random parameters
./ai-wallpaper generate --random-model --random-params

# Reproducible generation
./ai-wallpaper generate --seed 42 --model flux

# Save all intermediate stages
./ai-wallpaper generate --save-stages --output /custom/path/

# Generate without setting wallpaper
./ai-wallpaper generate --no-wallpaper --model sdxl
```

### **Development & Testing**
```bash
# Test everything
./ai-wallpaper test

# Test specific model quickly
./ai-wallpaper test --model flux --quick

# Test individual components
./ai-wallpaper test --component prompt
./ai-wallpaper test --component wallpaper

# Dry run to see execution plan
./ai-wallpaper --dry-run generate --random-model
```

### **Configuration Management**
```bash
# View current configuration
./ai-wallpaper config --show

# Verbose configuration dump
./ai-wallpaper --verbose config --show

# Validate configuration files
./ai-wallpaper config --validate

# Check model status
./ai-wallpaper models --check flux
./ai-wallpaper models --info dalle3
```

### **Automated Scheduling**
```bash
# Set up daily generation at 6 AM
./setup_cron.sh

# Manual cron setup
crontab -e
# Add: 0 6 * * * /home/user/ai-wallpaper/run_ai_wallpaper.sh

# Run manually with logging
./run_ai_wallpaper.sh
```

## 🛠️ Customization Options

### **Model Behavior Customization**
- **FLUX**: Adjust steps (50-100), guidance (2.0-4.0), memory optimizations
- **DALL-E**: Quality settings, style preferences, timeout values
- **SDXL**: Scheduler selection, LoRA weights, img2img refinement
- **All Models**: Custom prompt requirements, pipeline modifications

### **Theme System Customization**
- **Category Weights**: Adjust probability of theme categories
- **Individual Theme Weights**: Fine-tune specific theme selection
- **Weather Integration**: Customize weather-to-mood mappings
- **Custom Themes**: Add new themes with elements, styles, colors

### **Output Customization**
- **Quality Settings**: JPEG/PNG compression levels
- **Resolution**: Final output resolution (default 4K)
- **File Naming**: Timestamp and model-based naming patterns
- **Archive Management**: Automatic cleanup by age and size

### **Desktop Integration**
- **Multi-Environment Support**: Custom commands for any desktop environment
- **Multi-Monitor Support**: XFCE specialized multi-monitor handling
- **Wallpaper Verification**: Automatic verification of wallpaper setting

## 📊 Advanced Features

### **Random Parameter System**
When using `--random-params`:
- **Steps**: Randomized within model's `steps_range`
- **Guidance**: Randomized within model's `guidance_range`
- **Scheduler**: Random selection from `scheduler_options`
- **LoRA Weights**: Randomized within LoRA's `weight_range`

### **Weather Integration**
- **Real-time Weather**: NWS API integration with your GPS coordinates
- **Mood Mapping**: Weather conditions influence artistic themes
- **Cache System**: Intelligent caching with configurable expiration
- **Context Integration**: Weather data fed to prompt generation

### **Quality Pipeline System**
- **3-Stage Processing**: Generate → Upscale → Downsample for maximum quality
- **Real-ESRGAN Integration**: AI upscaling with configurable tile sizes
- **Memory Management**: Automatic VRAM management and cleanup
- **Stage Saving**: Optional intermediate stage preservation

### **Logging & Monitoring**
- **Comprehensive Logging**: VRAM usage, performance metrics, errors
- **Fail-Loud Philosophy**: Verbose error reporting, no silent failures
- **Daily Log Rotation**: Automatic log management
- **Cron Integration**: Specialized cron logging and error handling

## 🚨 Error Handling & Debugging

### **Verbose Mode**
```bash
./ai-wallpaper --verbose generate --model flux
```
Provides detailed output including:
- VRAM usage monitoring
- Model loading progress
- Pipeline stage execution
- Configuration validation results

### **Test Framework**
```bash
# Complete system test
./ai-wallpaper test --verbose

# Quick environment validation
./ai-wallpaper test --quick --model flux
```

### **Configuration Validation**
```bash
# Validate all YAML files
./ai-wallpaper config --validate

# Check specific model readiness
./ai-wallpaper models --check flux
```

This system is incredibly comprehensive and flexible - you can use it for everything from simple daily wallpaper generation to complex automated workflows with custom themes, models, and quality settings. Every aspect is configurable through the YAML files, and the CLI provides full control over generation parameters.