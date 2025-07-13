#!/usr/bin/env python3
"""
AI-Powered Daily Wallpaper Generator
Phase 1: Basic prompt generation with deepseek-r1:14b
Phase 2: Image generation with FLUX-Dev
Phase 3: Wallpaper setting with XFCE4
"""

import subprocess
import sys
import os
import time
from datetime import datetime
import argparse
import random
import re
import textwrap
import json
import requests
import shelve
import atexit

# Import theme selector - FAIL LOUDLY if not found
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from theme_selector import get_random_theme_with_weather
except ImportError as e:
    print(f"ERROR: Failed to import theme_selector module: {e}")
    print("Ensure theme_selector.py exists in the same directory")
    sys.exit(1)

# Configuration constants
OLLAMA_PATH = "/usr/local/bin/ollama"
LOG_DIR = "/home/user/ai-wallpaper/logs"
HISTORY_FILE = "/home/user/ai-wallpaper/prompt_history.txt"
LAST_RUN_FILE = "/home/user/ai-wallpaper/last_run.txt"
IMAGES_DIR = "/home/user/ai-wallpaper/images"
VENV_PYTHON = "/home/user/grace/.venv/bin/python3"
MODEL_PATH = "/home/user/.cache/huggingface/hub/models--black-forest-labs--FLUX.1-dev"

# Weather configuration constants
ELKHORN_WI_LAT = 42.6728
ELKHORN_WI_LON = -88.5443
WEATHER_CACHE_DIR = "/home/user/ai-wallpaper/.weather_cache"
CACHE_GRID_EXPIRATION = 24 * 3600  # 1 day
CACHE_FORECAST_EXPIRATION = 15 * 60  # 15 minutes
TOO_MANY_API_CALLS_DELAY = 60  # 1 minute
WEATHER_AGENT = '(ai-wallpaper weather, ai-wallpaper@example.com)'
WEATHER_ACCEPT = 'application/geo+json'

def log(message):
    """Print timestamped message to console and log file"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    full_message = f"[{timestamp}] {message}"
    print(full_message)
    
    # Ensure log directory exists
    os.makedirs(LOG_DIR, exist_ok=True)
    
    # Write to daily log file
    log_file = os.path.join(LOG_DIR, f"{datetime.now().strftime('%Y-%m-%d')}.log")
    with open(log_file, 'a') as f:
        f.write(full_message + '\n')

def recreate_weather_cache():
    """Recreate weather cache when corruption is detected"""
    global weather_cache
    import shutil
    log("Weather cache corruption detected - recreating cache automatically...")
    
    # Close any existing cache handles
    try:
        if 'weather_cache' in globals():
            weather_cache.close()
    except:
        pass
    
    # Remove corrupted cache directory
    if os.path.exists(WEATHER_CACHE_DIR):
        shutil.rmtree(WEATHER_CACHE_DIR)
        log(f"Removed corrupted cache directory: {WEATHER_CACHE_DIR}")
    
    # Recreate directory and cache
    os.makedirs(WEATHER_CACHE_DIR, exist_ok=True)
    
    # Create new cache
    try:
        weather_cache = shelve.open(os.path.join(WEATHER_CACHE_DIR, 'nws_api.cache'))
        atexit.register(weather_cache.close)
        log("Weather cache recreated successfully")
        return weather_cache
    except Exception as e:
        log(f"FATAL ERROR: Cannot recreate weather cache: {e}")
        sys.exit(1)

# Initialize weather cache with proper cleanup and corruption detection
os.makedirs(WEATHER_CACHE_DIR, exist_ok=True)
try:
    weather_cache = shelve.open(os.path.join(WEATHER_CACHE_DIR, 'nws_api.cache'))
    atexit.register(weather_cache.close)  # Ensure cache is closed on exit
except Exception as e:
    log(f"ERROR: Weather cache corrupted or inaccessible: {e}")
    weather_cache = recreate_weather_cache()

def get_weather_url_with_cache(url, cache_expiration=CACHE_FORECAST_EXPIRATION):
    """Get URL content with caching - NO FALLBACKS, FAIL LOUDLY"""
    tim = time.time()
    
    # Try to access cache with corruption detection
    try:
        cache_hit = url in weather_cache
        if cache_hit:
            cached_time = weather_cache[url][0]
            needs_refresh = cached_time < tim - cache_expiration
        else:
            needs_refresh = True
    except Exception as e:
        log(f"ERROR: Weather cache corrupted when reading: {e}")
        log(f"Failed to read cache for URL: {url}")
        recreate_weather_cache()
        # After recreation, we need to fetch fresh data
        needs_refresh = True
        cache_hit = False
    
    if not cache_hit or needs_refresh:
        delay = TOO_MANY_API_CALLS_DELAY
        headers = {'User-Agent': WEATHER_AGENT, 'Accept': WEATHER_ACCEPT}
        max_attempts = 5  # Maximum retry attempts for rate limiting
        attempt = 0
        
        while True:
            response = requests.get(url, headers=headers, timeout=30)
            if response.status_code == requests.codes.too_many:
                attempt += 1
                if attempt >= max_attempts:
                    log(f"ERROR: Weather API rate limited after {max_attempts} attempts")
                    log("Maximum retry attempts exceeded - API may be down or heavily rate limited")
                    sys.exit(1)
                log(f"Weather API rate limited, waiting {delay} seconds... (attempt {attempt}/{max_attempts})")
                time.sleep(delay)
                delay *= 2
                tim = time.time()
            else:
                break
        
        if response.status_code != requests.codes.ok:
            log(f"ERROR: Weather API failed with status {response.status_code} for url '{url}'")
            log(f"Response: {response.text}")
            sys.exit(1)
        
        # Try to write to cache with corruption detection
        try:
            weather_cache[url] = (tim, response.json())
        except Exception as e:
            log(f"ERROR: Failed to write to weather cache: {e}")
            log("Cache may be corrupted - recreating...")
            recreate_weather_cache()
            # Try to write again after recreation
            try:
                weather_cache[url] = (tim, response.json())
            except Exception as e2:
                log(f"FATAL ERROR: Cannot write to recreated cache: {e2}")
                sys.exit(1)
    
    # Try to read from cache with corruption detection
    try:
        return weather_cache[url][1]
    except Exception as e:
        log(f"ERROR: Failed to read from weather cache: {e}")
        log("Cache may be corrupted - this should not happen after write")
        log("There may be a deeper issue with cache consistency")
        recreate_weather_cache()
        log(f"FATAL ERROR: Cache corruption occurred immediately after write operation: {e}")
        sys.exit(1)

def get_weather_grid(latitude, longitude):
    """Get the NWS grid for a particular location"""
    url = f'https://api.weather.gov/points/{latitude},{longitude}'
    return get_weather_url_with_cache(url, CACHE_GRID_EXPIRATION)

def map_weather_condition_to_mood(condition):
    """Map weather conditions to creative moods for image generation"""
    condition_lower = condition.lower()
    
    # Rain and storms
    if any(word in condition_lower for word in ['rain', 'shower', 'drizzle', 'storm', 'thunderstorm']):
        return 'rainy'
    
    # Snow and winter weather
    if any(word in condition_lower for word in ['snow', 'blizzard', 'flurries', 'sleet', 'ice']):
        return 'snowy'
    
    # Fog and mist
    if any(word in condition_lower for word in ['fog', 'mist', 'haze']):
        return 'misty'
    
    # Clear and sunny
    if any(word in condition_lower for word in ['clear', 'sunny', 'fair']):
        return 'sunny'
    
    # Cloudy but no precipitation
    if any(word in condition_lower for word in ['cloud', 'overcast', 'partly']):
        return 'cloudy'
    
    # Wind
    if any(word in condition_lower for word in ['wind', 'breezy', 'gusty']):
        return 'windy'
    
    # Default fallback
    return 'neutral'

def get_weather_context():
    """Get current weather context for Elkhorn, WI - NO FALLBACKS"""
    log("Fetching weather context for Elkhorn, WI...")
    
    # Get NWS grid for Elkhorn, WI
    grid = get_weather_grid(ELKHORN_WI_LAT, ELKHORN_WI_LON)
    
    if not grid or 'properties' not in grid:
        log("ERROR: Failed to get valid weather grid data")
        log(f"Grid response: {grid}")
        sys.exit(1)
    
    # Get hourly forecast URL
    if 'forecastHourly' not in grid['properties']:
        log("ERROR: No forecastHourly URL in grid properties")
        log(f"Grid properties: {grid['properties']}")
        sys.exit(1)
    
    hourly_forecast_url = grid['properties']['forecastHourly']
    
    # Get hourly forecast
    hourly_forecast = get_weather_url_with_cache(hourly_forecast_url)
    
    if not hourly_forecast or 'properties' not in hourly_forecast:
        log("ERROR: Failed to get valid hourly forecast data")
        log(f"Forecast response: {hourly_forecast}")
        sys.exit(1)
    
    if 'periods' not in hourly_forecast['properties'] or not hourly_forecast['properties']['periods']:
        log("ERROR: No periods in hourly forecast")
        log(f"Forecast properties: {hourly_forecast['properties']}")
        sys.exit(1)
    
    # Get current hour's forecast (first entry)
    current_hour = hourly_forecast['properties']['periods'][0]
    
    # Validate all required fields exist
    required_fields = ['shortForecast', 'temperature', 'temperatureUnit', 'windSpeed', 'windDirection']
    for field in required_fields:
        if field not in current_hour:
            log(f"ERROR: Missing required field '{field}' in weather data")
            log(f"Current hour data: {current_hour}")
            sys.exit(1)
    
    condition = current_hour['shortForecast']
    temperature = current_hour['temperature']
    temp_unit = current_hour['temperatureUnit']
    wind_speed = current_hour['windSpeed']
    wind_direction = current_hour['windDirection']
    
    # Create weather context
    weather_context = {
        'condition': condition,
        'temperature': f"{temperature}°{temp_unit}",
        'wind': f"{wind_speed} {wind_direction}",
        'mood': map_weather_condition_to_mood(condition)
    }
    
    log(f"Weather context: {condition}, {temperature}°{temp_unit}, Wind {wind_speed} {wind_direction}, Mood: {weather_context['mood']}")
    
    return weather_context

def start_ollama_server():
    """Start Ollama server if not already running - NO FALLBACKS"""
    log("Checking if Ollama server is running...")
    
    # Check if Ollama is already running
    result = subprocess.run([OLLAMA_PATH, "list"], capture_output=True, text=True)
    
    if result.returncode != 0:
        log("Ollama server not running, starting it now...")
        # Start Ollama server in background
        subprocess.Popen([OLLAMA_PATH, "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # Wait for server to be ready
        log("Waiting for Ollama server to be ready...")
        wait_seconds = 0
        while True:
            time.sleep(1)
            wait_seconds += 1
            result = subprocess.run([OLLAMA_PATH, "list"], capture_output=True, text=True)
            if result.returncode == 0:
                log(f"Ollama server is ready after {wait_seconds} seconds!")
                break
            # Log progress every 10 seconds
            if wait_seconds % 10 == 0:
                log(f"Still waiting for Ollama server... ({wait_seconds} seconds elapsed)")
    else:
        log("Ollama server is already running")

def ensure_deepseek_model():
    """Ensure deepseek-r1:14b model is available - NO FALLBACKS"""
    log("Checking if deepseek-r1:14b model is available...")
    
    result = subprocess.run([OLLAMA_PATH, "list"], capture_output=True, text=True, check=True)
    
    if "deepseek-r1:14b" not in result.stdout:
        log("deepseek-r1:14b model not found, pulling it now...")
        log("This may take a while on first run...")
        subprocess.run([OLLAMA_PATH, "pull", "deepseek-r1:14b"], check=True)
        log("deepseek-r1:14b model pulled successfully")
    else:
        log("deepseek-r1:14b model is already available")

def load_prompt_history():
    """Load previous prompts from history file"""
    log("Loading prompt history...")
    
    if not os.path.exists(HISTORY_FILE):
        log("No history file found, starting fresh")
        return []
    
    with open(HISTORY_FILE, 'r') as f:
        history = [line.strip() for line in f.readlines() if line.strip()]
    
    log(f"Loaded {len(history)} previous prompts")
    return history

def get_context_info():
    """Get current context information for prompt generation"""
    now = datetime.now()
    
    # Determine season
    month = now.month
    if month in [12, 1, 2]:
        season = "winter"
    elif month in [3, 4, 5]:
        season = "spring"
    elif month in [6, 7, 8]:
        season = "summer"
    else:
        season = "autumn"
    
    # Get weather context
    weather = get_weather_context()
    
    context = {
        'date': now.strftime("%Y-%m-%d"),
        'time': now.strftime("%H:%M"),
        'day_of_week': now.strftime("%A"),
        'month': now.strftime("%B"),
        'season': season,
        'year': now.year,
        'weather': weather
    }
    
    log(f"Context: {context['day_of_week']}, {context['month']} {now.day}, {context['year']} ({context['season']})")
    log(f"Weather: {weather['condition']}, {weather['temperature']}, {weather['wind']}, Mood: {weather['mood']}")
    
    return context

def generate_prompt_with_deepseek(history):
    """Generate a unique, creative image prompt using deepseek-r1:14b with themed guidance"""
    log("Generating creative prompt with deepseek-r1:14b...")

    context = get_context_info()

    # Get themed instruction from theme selector
    log("Selecting theme for prompt generation...")
    theme_result = get_random_theme_with_weather(context['weather'])
    
    # Log the selected theme
    log(f"Using theme: {theme_result['theme']['name']} from category {theme_result['category']}")

    # DeepSeek instruction combining theme with requirements
    deepseek_instruction = f"""
Generate a single, richly detailed image prompt for a desktop wallpaper, optimized for the FLUX-Dev model.

{theme_result['instruction']}

Context: It's {context['day_of_week']} in {context['season']}.

Requirements:
- The prompt MUST be under 65 words. Do NOT go over.
- The prompt MUST be the **only** thing in your output. Absolutely no extra text, no commentary, no quotes, no labels, no headers.
- Combine the theme elements creatively and unexpectedly.
- Describe a vivid scene with clear composition: foreground, midground, and background.
- Specify lighting and atmospheric conditions that enhance the theme.
- Include the color palette and artistic style mentioned above.
- Add rich texture and material details.
- Make it photorealistic, ultra-detailed, and gallery-worthy.

ONLY return the image prompt. No other text.

Image prompt:
""".strip()

    log("Sending instruction to deepseek-r1:14b...")

    cmd = [OLLAMA_PATH, "run", "deepseek-r1:14b", deepseek_instruction]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)

    raw_output = result.stdout
    log(f"Raw output from model: {raw_output[:200]}...")

    generated_prompt = raw_output.strip()

    log("Stopping deepseek-r1:14b model to free VRAM...")
    stop_cmd = [OLLAMA_PATH, "stop", "deepseek-r1:14b"]
    subprocess.run(stop_cmd, capture_output=True, text=True, check=True)
    log("Model stopped successfully")

    # Clean up formatting
    if "<think>" in generated_prompt and "</think>" in generated_prompt:
        end_tag_index = generated_prompt.find("</think>")
        if end_tag_index != -1:
            generated_prompt = generated_prompt[end_tag_index + len("</think>"):].strip()

    if "```" in generated_prompt:
        parts = generated_prompt.split("```")
        if len(parts) >= 2:
            generated_prompt = parts[1].strip()

    if generated_prompt.startswith('"') and generated_prompt.endswith('"'):
        generated_prompt = generated_prompt[1:-1].strip()

    generated_prompt = generated_prompt.replace('\n', ' ').replace('  ', ' ').strip()
    generated_prompt = generated_prompt.replace('**', '')

    unwanted_prefixes = ["Here's", "Here is", "This is", "I'll create", "I'll generate", 
                         "Let me", "Sure", "Certainly", "Of course", "Image prompt:", 
                         "Image description:", "Description:", "**Image Description:**", 
                         "Image Description:"]
    for prefix in unwanted_prefixes:
        if generated_prompt.startswith(prefix):
            for punct in ['.', ':', '\n']:
                idx = generated_prompt.find(punct)
                if idx > 0:
                    generated_prompt = generated_prompt[idx+1:].strip()
                    break

    log(f"Generated prompt: {generated_prompt[:100]}..." if len(generated_prompt) > 100 else f"Generated prompt: {generated_prompt}")

    if not generated_prompt or len(generated_prompt.strip()) < 10:
        log("ERROR: Generated prompt is too short or empty")
        sys.exit(1)

    return generated_prompt


def save_prompt(prompt, seed=None):
    """Save the generated prompt to history and last run file"""
    log("Saving prompt to history...")
    
    # Ensure history file ends with newline before appending
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, 'rb') as f:
            f.seek(-1, 2)  # Go to last byte
            last_char = f.read(1)
            needs_newline = last_char != b'\n'
    else:
        needs_newline = False
    
    # Append to history file
    with open(HISTORY_FILE, 'a') as f:
        if needs_newline:
            f.write('\n')
        f.write(prompt + '\n')
    
    log("Saving last run information...")
    
    # Save detailed last run info
    with open(LAST_RUN_FILE, 'w') as f:
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Prompt: {prompt}\n")
        if seed is not None:
            f.write(f"Seed: {seed}\n")
        f.write(f"Status: Success\n")
    
    log("Prompt saved successfully")

def sanitize_filename(text, max_length=50):
    """Convert text to a safe filename"""
    # Remove non-alphanumeric characters, keep spaces
    safe = re.sub(r'[^a-zA-Z0-9\s\-]', '', text)
    # Replace spaces with underscores
    safe = safe.replace(' ', '_')
    # Truncate to max length
    safe = safe[:max_length]
    # Remove trailing underscores
    safe = safe.rstrip('_')
    return safe.lower()

def generate_image(prompt):
    """Generate image using FLUX-Dev via venv - NO FALLBACKS"""
    log("Starting image generation...")
    
    # Ensure images directory exists
    os.makedirs(IMAGES_DIR, exist_ok=True)
    
    # Generate seed for reproducibility
    seed = random.randint(0, 2**32-1)
    log(f"Generated seed: {seed} (save this to reproduce the image!)")
    
    # Create filename
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    prompt_excerpt = sanitize_filename(prompt)
    filename = f"{timestamp}_{prompt_excerpt}.png"
    filepath = os.path.join(IMAGES_DIR, filename)
    
    log(f"Image will be saved to: {filepath}")
    
    # Create the Python script to run in venv
    generation_script = f'''
import torch
from diffusers import FluxPipeline, FluxTransformer2DModel, FlowMatchEulerDiscreteScheduler
from transformers import T5EncoderModel
from PIL import Image
import gc
import numpy as np
import os
import sys

print("Loading FLUX-Dev pipeline with advanced memory optimizations...")
# Clear GPU cache before starting
torch.cuda.empty_cache()
gc.collect()

# FP8 quantization disabled due to compatibility issues with CPU offload
# The quantized weights cause TypeError in accelerate library
use_fp8 = False
print("Using standard precision (FP8 disabled for compatibility)")

# Set memory fraction to prevent PyTorch from using all available memory
torch.cuda.set_per_process_memory_fraction(0.90)  # Use only 90% of VRAM

# Try local model paths first (in order of preference)
flux_model_paths = [
    "/home/user/.cache/huggingface/hub/models--black-forest-labs--FLUX.1-dev/snapshots/0ef5fff789c832c5c7f4e127f94c8b54bbcced44",
    "/home/user/.cache/huggingface/hub/models--black-forest-labs--FLUX.1-dev",  # Try base directory
    "/home/user/Pictures/ai/models/flux",  # Backup location
    # Note: /home/user/vidgen/ComfyUI/models/checkpoints/FLUX1 is FP8 format - not compatible
]

model_path = None
for path in flux_model_paths:
    if os.path.exists(path):
        # Verify it's a valid model directory (should contain model files)
        if os.path.isdir(path):
            # Check for key model files
            has_model_files = any(os.path.exists(os.path.join(path, f)) for f in 
                                ['model_index.json', 'unet/diffusion_pytorch_model.safetensors', 
                                 'transformer/diffusion_pytorch_model.safetensors'])
            if has_model_files:
                model_path = path
                print(f"Found valid FLUX model at: {{model_path}}")
                break
            else:
                print(f"Path exists but appears incomplete: {{path}}")
        else:
            print(f"Path exists but is not a directory: {{path}}")

if not model_path:
    model_path = "black-forest-labs/FLUX.1-dev"
    print("No local model found, will download from HuggingFace...")
    print("This may take 15-30 minutes on first run!")

# Load components separately for better memory control
print("Loading FLUX components with memory optimizations...")

# Load standard pipeline first
# CRITICAL: Load both CLIP and T5 encoders for full 512 token support
pipe = FluxPipeline.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True
    # Do NOT set text_encoder_2=None - that removes T5!
)

# FLUX uses its default FlowMatchEulerDiscreteScheduler which is optimal
# Other schedulers like DPM++ and Euler Ancestral are incompatible with FLUX's custom sigmas
# Explicitly ensure we're using the correct scheduler
pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(pipe.scheduler.config)
print("Using FlowMatchEulerDiscreteScheduler (explicit set for FLUX compatibility)")

# FP8 is disabled, using standard precision
fp8_active = False

# Enable memory-saving features based on whether FP8 is active
if not fp8_active:
    # Standard path: use sequential CPU offload
    print("Enabling sequential CPU offload...")
    pipe.enable_sequential_cpu_offload()
else:
    # FP8 path: models are already on GPU, no CPU offload needed
    print("FP8 models loaded on GPU (CPU offload disabled for compatibility)")

# Enhanced VAE optimizations
pipe.vae.enable_tiling()
pipe.vae.enable_slicing()

# xFormers disabled due to FLUX compatibility issues
# Causes UnboundLocalError in FLUX transformer
print("xFormers disabled for FLUX compatibility")

# Enable attention slicing with aggressive settings
pipe.enable_attention_slicing(1)  # Most aggressive slicing

# Force garbage collection after model loading
gc.collect()
torch.cuda.empty_cache()

# RTX 3090 Ultra-Quality Settings for FLUX-Dev with Memory Optimizations
# With 24GB VRAM, we need careful memory management at 1920x1080

# Use the full prompt without additional keywords (FLUX handles quality internally)
original_prompt = {repr(prompt)}

print("=== RTX 3090 Memory-Optimized FLUX Generation ===")
print("GPU: NVIDIA RTX 3090 (24GB VRAM)")
print("Model: FLUX.1-dev (12B parameters)")
print("Resolution: 1920x1088 (16:9, Full HD base for 4x upscale to 8K, then downsample to 4K)")
print("Steps: 100 (high quality)")
print("Guidance: 3.5 (optimal for FLUX)")
print("Max tokens: 512")
print("Scheduler: FlowMatchEulerDiscrete (optimal for FLUX)")
print("Memory optimizations: CPU offload, attention slicing")

# Clear cache before generation
torch.cuda.empty_cache()
gc.collect()

# Stage 1: Generate at 1920x1088 with memory-optimized settings
seed = {seed}  # Use seed from parent script
generator = torch.Generator(device="cpu").manual_seed(seed)  # CPU generator for memory

print(f"Using seed: {{seed}}")
print("Starting FLUX generation at 1920x1088...")

# Monitor memory usage
if torch.cuda.is_available():
    print(f"VRAM before generation: {{torch.cuda.memory_allocated() / 1024**3:.2f}} GB")

# Generate with memory monitoring
with torch.inference_mode():  # Disable gradient computation
    image = pipe(
        prompt=original_prompt,
        height=1088,  # Divisible by 16 (68*16=1088, close to 1080)
        width=1920,   # Divisible by 16 (120*16=1920)
        guidance_scale=3.5,           # Optimal for FLUX-Dev
        num_inference_steps=100,      # High quality from other scripts
        max_sequence_length=512,      # Full token length
        generator=generator
    ).images[0]

if torch.cuda.is_available():
    print(f"VRAM after generation: {{torch.cuda.memory_allocated() / 1024**3:.2f}} GB")

print(f"Stage 1 complete: Generated at {{image.size}}")

# Verify dimensions
assert image.size == (1920, 1088), f"ERROR: Wrong dimensions {{image.size}}, expected (1920, 1088)"

# Save Stage 1 output for comparison
stage1_path = "{filepath}".replace(".png", "_stage1_flux.png")
print(f"Saving Stage 1 output to: {{stage1_path}}")
image.save(stage1_path, "PNG", quality=100)

# Clean up initial pipeline to free memory
print("Cleaning up Stage 1 pipeline...")
del pipe
torch.cuda.empty_cache()
gc.collect()
print(f"VRAM after cleanup: {{torch.cuda.memory_allocated() / 1024**3:.2f}} GB" if torch.cuda.is_available() else "")

# Stage 2: Real-ESRGAN upscaling to 8K for supersampling
print("Stage 2: Real-ESRGAN ultra-quality upscaling to 8K")
print("Running garbage collection before upscaling...")
gc.collect()

print("Starting Real-ESRGAN ultra-quality upscaling to 8K...")
print(f"Input image size: {{image.size}}")

# Save intermediate image for Real-ESRGAN
temp_input = "/tmp/flux_temp_input.png"
temp_output_dir = "/tmp/flux_temp_output"  # Real-ESRGAN creates a directory
print(f"Saving intermediate image to {{temp_input}}")
image.save(temp_input, "PNG", quality=100)

# Check if Real-ESRGAN is available
import subprocess
import shutil

# Try to find Real-ESRGAN installation
realesrgan_paths = [
    "/home/user/ai-wallpaper/Real-ESRGAN/inference_realesrgan.py",
    "/home/user/Real-ESRGAN/inference_realesrgan.py",
    os.path.expanduser("~/Real-ESRGAN/inference_realesrgan.py"),
    "/usr/local/bin/realesrgan-ncnn-vulkan"  # Alternative binary
]

realesrgan_script = None
for path in realesrgan_paths:
    if os.path.exists(path):
        realesrgan_script = path
        print(f"Found Real-ESRGAN at: {{realesrgan_script}}")
        break

if not realesrgan_script:
    print("ERROR: Real-ESRGAN not found! Cannot proceed with 4K upscaling.")
    print("Real-ESRGAN is REQUIRED for ultra-high-quality 4K wallpapers.")
    print("Please install Real-ESRGAN:")
    print("  1. cd /home/user/ai-wallpaper")
    print("  2. git clone https://github.com/xinntao/Real-ESRGAN.git")
    print("  3. cd Real-ESRGAN")
    print("  4. pip install basicsr facexlib gfpgan")
    print("  5. pip install -r requirements.txt")
    print("  6. python setup.py develop")
    print("  7. wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth -P weights")
    raise RuntimeError("Real-ESRGAN is required but not found. Aborting.")
else:
    # Use Real-ESRGAN with maximum quality settings
    print("Running Real-ESRGAN with ultra-quality settings for 8K...")
    
    # 4x upscale from 1920x1088 to 7680x4352
    scale_factor = 4
    
    if realesrgan_script.endswith('.py'):
        # Python script version - use same interpreter as parent
        cmd = [
            sys.executable, realesrgan_script,  # Use venv Python
            "-n", "RealESRGAN_x4plus",      # Best quality model
            "-i", temp_input,
            "-o", temp_output_dir,
            "--outscale", str(scale_factor),
            "-t", "1024",                    # Large tile size for RTX 3090
            "--fp32"                         # Maximum precision, NO face enhance
        ]
    else:
        # Binary version
        cmd = [
            realesrgan_script,
            "-i", temp_input,
            "-o", temp_output_dir,
            "-s", str(scale_factor),
            "-n", "realesrgan-x4plus",
            "-t", "1024"
        ]
    
    print(f"Executing: {{' '.join(cmd)}}")
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    print("Real-ESRGAN output: " + result.stdout)
    if result.stderr:
        print("Real-ESRGAN warnings: " + result.stderr)
    
    # Load upscaled image (Real-ESRGAN puts it in a directory)
    # Look for the output file in the directory
    output_files = []
    if os.path.exists(temp_output_dir):
        if os.path.isdir(temp_output_dir):
            output_files = [f for f in os.listdir(temp_output_dir) if f.endswith('.png')]
            if output_files:
                actual_output = os.path.join(temp_output_dir, output_files[0])
            else:
                print("ERROR: No PNG files found in Real-ESRGAN output directory!")
                raise FileNotFoundError(f"No output files in {{temp_output_dir}}")
        else:
            # It's a file, not a directory
            actual_output = temp_output_dir
    else:
        print("ERROR: Real-ESRGAN output not found!")
        raise FileNotFoundError(f"Expected output at {{temp_output_dir}}")
    
    print(f"Loading upscaled image from {{actual_output}}")
    image_8k = Image.open(actual_output)
    print(f"Upscaled image size: {{image_8k.size}}")
    
    # Save pre-crop upscaled image for comparison
    stage2_precrop_path = "{filepath}".replace(".png", "_stage2_8k_upscaled_precrop.png")
    print(f"Saving pre-crop 8K upscaled image to: {{stage2_precrop_path}}")
    image_8k.save(stage2_precrop_path, "PNG", quality=100)
    
    # With 4x upscale from 1920x1088, we get 7680x4352 (then crop to 8K standard)
    # Center crop to exact 8K resolution - no resampling for maximum quality
    if image_8k.size != (7680, 4320):
        print(f"Cropping from {{image_8k.size}} to 8K (7680x4320)...")
        width, height = image_8k.size
        if width == 7680 and height == 4352:
            # Perfect 4x upscale - crop 32 pixels (16 top, 16 bottom)
            image_8k = image_8k.crop((0, 16, 7680, 4336))
        else:
            # Unexpected size - center crop to 8K
            left = (width - 7680) // 2
            top = (height - 4320) // 2
            right = left + 7680
            bottom = top + 4320
            image_8k = image_8k.crop((left, top, right, bottom))
    
    # Clean up temporary files
    if os.path.exists(temp_input):
        os.remove(temp_input)
        print(f"Cleaned up: {{temp_input}}")
    if os.path.exists(temp_output_dir):
        if os.path.isdir(temp_output_dir):
            shutil.rmtree(temp_output_dir)
        else:
            os.remove(temp_output_dir)
        print(f"Cleaned up: {{temp_output_dir}}")

# Verify size after cropping
assert image_8k.size == (7680, 4320), f"Expected 7680x4320, got {{image_8k.size}}"
print("8K dimensions verified after cropping")

# Save Stage 2 cropped 8K output
stage2_cropped_8k_path = "{filepath}".replace(".png", "_stage2_cropped_8k.png")
print(f"Saving Stage 2 cropped 8K output to: {{stage2_cropped_8k_path}}")
image_8k.save(stage2_cropped_8k_path, "PNG", quality=100)

print("Stage 2 complete: Upscaled and cropped to 8K")

# Stage 3: High-quality downsample from 8K to 4K using Lanczos
print("Stage 3: Downsampling from 8K to 4K using Lanczos for supersampling quality...")
print(f"Input size: {{image_8k.size}}")

# Downsample to 4K using high-quality Lanczos filter
image_4k = image_8k.resize((3840, 2160), Image.Resampling.LANCZOS)
print(f"Downsampled to: {{image_4k.size}}")

# Verify final size
assert image_4k.size == (3840, 2160), f"Expected 3840x2160, got {{image_4k.size}}"
print("Final 4K wallpaper dimensions verified")

print("Saving final supersampled 4K image...")
image_4k.save("{filepath}", "PNG", quality=100)
print("High-quality supersampled 4K image saved successfully")

print("Image generation with 8K→4K supersampling complete!")
'''
    
    log("Executing image generation in venv...")
    
    # Run the script with venv Python
    cmd = [VENV_PYTHON, "-c", generation_script]
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    
    # Stream output
    for line in iter(process.stdout.readline, ''):
        if line:
            log(f"  [IMG] {line.strip()}")
    
    # Wait for completion
    return_code = process.wait()
    
    if return_code != 0:
        log(f"ERROR: Image generation failed with return code {return_code}")
        sys.exit(1)
    
    # Verify file exists
    if not os.path.exists(filepath):
        log(f"ERROR: Generated image not found at {filepath}")
        sys.exit(1)
    
    # Get file size
    file_size = os.path.getsize(filepath) / (1024 * 1024)  # MB
    log(f"Image saved successfully: {filename} ({file_size:.2f} MB)")
    
    return filepath, seed

def get_all_xfce4_backdrop_properties():
    """Get ALL XFCE4 backdrop property paths for all monitors and workspaces"""
    log("Detecting ALL XFCE4 backdrop properties...")
    
    # Ensure DISPLAY is set for xfconf-query
    env = os.environ.copy()
    if 'DISPLAY' not in env or not env['DISPLAY']:
        env['DISPLAY'] = ':0'
        log("Setting DISPLAY=:0 for xfconf-query")
    
    # List all properties in xfce4-desktop channel to find backdrop
    cmd = ["xfconf-query", "-c", "xfce4-desktop", "-l"]
    result = subprocess.run(cmd, capture_output=True, text=True, env=env, check=True)
    
    # Find ALL backdrop properties that end with "last-image"
    properties = result.stdout.strip().split('\n')
    backdrop_props = [p for p in properties if p.endswith('/last-image')]
    
    if not backdrop_props:
        log("ERROR: No backdrop properties found in XFCE4")
        sys.exit(1)
    
    log(f"Found {len(backdrop_props)} backdrop properties across all monitors/workspaces:")
    for prop in backdrop_props:
        log(f"  - {prop}")
    
    return backdrop_props

def set_wallpaper(image_path):
    """Set the wallpaper using xfconf-query for ALL monitors and workspaces"""
    log("Setting wallpaper with XFCE4 on ALL monitors and workspaces...")
    
    # Ensure image path is absolute
    if not os.path.isabs(image_path):
        image_path = os.path.abspath(image_path)
    
    # Verify image exists
    if not os.path.exists(image_path):
        log(f"ERROR: Image not found: {image_path}")
        sys.exit(1)
    
    log(f"Setting wallpaper to: {image_path}")
    
    # Get ALL backdrop property paths
    property_paths = get_all_xfce4_backdrop_properties()
    
    # Ensure DISPLAY is set
    env = os.environ.copy()
    if 'DISPLAY' not in env or not env['DISPLAY']:
        env['DISPLAY'] = ':0'
    
    # Set the wallpaper on ALL properties (all monitors and workspaces)
    for property_path in property_paths:
        log(f"Setting wallpaper for: {property_path}")
        
        # Set the image path
        cmd = [
            "xfconf-query",
            "-c", "xfce4-desktop",
            "-p", property_path,
            "-s", image_path
        ]
        subprocess.run(cmd, check=True, env=env)
        
        # Also set the image style to ensure it's scaled properly
        # 0 = None, 1 = Centered, 2 = Tiled, 3 = Stretched, 4 = Scaled, 5 = Zoomed
        style_property = property_path.replace("/last-image", "/image-style")
        cmd_style = [
            "xfconf-query",
            "-c", "xfce4-desktop",
            "-p", style_property,
            "-s", "4"  # Scaled (fits screen maintaining aspect ratio)
        ]
        subprocess.run(cmd_style, check=True, env=env)
    
    log(f"Wallpaper set successfully on {len(property_paths)} monitor/workspace combinations")
    log("Wallpaper style set to 'Scaled' for proper 4K display on all combinations")
    
    # CRITICAL: Force xfdesktop to reload and refresh wallpaper display
    log("Forcing xfdesktop to reload configuration...")
    
    # Use xfdesktop --reload command to safely refresh wallpaper
    # This reloads settings without killing the xfdesktop process
    reload_cmd = ["xfdesktop", "--reload"]
    
    # Execute reload command with proper DISPLAY environment
    subprocess.run(reload_cmd, check=True, env=env)
    log("xfdesktop configuration reloaded - wallpaper should now be visible")

def verify_wallpaper(expected_path):
    """Verify the wallpaper was set correctly on ALL monitor/workspace combinations"""
    log("Verifying wallpaper setting on ALL monitor/workspace combinations...")
    
    # Ensure DISPLAY is set
    env = os.environ.copy()
    if 'DISPLAY' not in env or not env['DISPLAY']:
        env['DISPLAY'] = ':0'
    
    # Get ALL property paths
    property_paths = get_all_xfce4_backdrop_properties()
    
    # Ensure both paths are absolute for comparison
    expected_abs = os.path.abspath(expected_path)
    
    verification_failed = False
    failed_properties = []
    
    # Check each property
    for property_path in property_paths:
        cmd = [
            "xfconf-query",
            "-c", "xfce4-desktop",
            "-p", property_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, env=env)
        current_wallpaper = result.stdout.strip()
        
        if current_wallpaper == expected_abs:
            log(f"✓ Verified: {property_path}")
        else:
            log(f"✗ FAILED: {property_path}")
            log(f"  Expected: {expected_abs}")
            log(f"  Current: {current_wallpaper}")
            verification_failed = True
            failed_properties.append(property_path)
    
    if verification_failed:
        log(f"ERROR: Wallpaper verification failed for {len(failed_properties)} properties")
        sys.exit(1)
    
    log(f"Wallpaper verified successfully on all {len(property_paths)} monitor/workspace combinations")
    return True

def test_wallpaper_setting(image_path=None):
    """Test function for wallpaper setting only"""
    log("=== STARTING WALLPAPER SETTING TEST ===")
    
    if not image_path:
        # Use the most recent image from our images directory
        if os.path.exists(IMAGES_DIR):
            images = [f for f in os.listdir(IMAGES_DIR) if f.endswith('.png')]
            if images:
                # Sort by modification time, newest first
                images.sort(key=lambda x: os.path.getmtime(os.path.join(IMAGES_DIR, x)), reverse=True)
                image_path = os.path.join(IMAGES_DIR, images[0])
                log(f"Using most recent image: {images[0]}")
            else:
                log("ERROR: No images found in images directory")
                sys.exit(1)
        else:
            log(f"ERROR: Images directory not found: {IMAGES_DIR}")
            sys.exit(1)
    
    # Set wallpaper
    set_wallpaper(image_path)
    
    # Verify it was set
    verify_wallpaper(image_path)
    
    log("=== WALLPAPER SETTING TEST COMPLETE ===")
    
    return image_path

def test_prompt_generation():
    """Test function for prompt generation only"""
    log("=== STARTING PROMPT GENERATION TEST ===")
    
    # Start Ollama server
    start_ollama_server()
    
    # Ensure model is available
    ensure_deepseek_model()
    
    # Load history
    history = load_prompt_history()
    
    # Generate prompt
    prompt = generate_prompt_with_deepseek(history)
    
    # Save prompt
    save_prompt(prompt)
    
    log("=== PROMPT GENERATION TEST COMPLETE ===")
    log(f"Generated prompt: {prompt}")
    
    return prompt

def test_image_generation(prompt=None):
    """Test function for image generation only"""
    log("=== STARTING IMAGE GENERATION TEST ===")
    
    if not prompt:
        log("No prompt provided, using test prompt")
        prompt = "A stunning cyberpunk cityscape at sunset with neon lights reflecting on wet streets, flying cars above, dramatic clouds, ultra detailed, photorealistic style"
    
    log(f"Test prompt: {prompt}")
    
    # Generate image
    image_path, seed = generate_image(prompt)
    
    log("=== IMAGE GENERATION TEST COMPLETE ===")
    log(f"Image saved to: {image_path}")
    log(f"Seed used: {seed}")
    
    return image_path

def full_generation():
    """Complete wallpaper generation from prompt to desktop"""
    log("=== STARTING DAILY WALLPAPER GENERATION ===")
    log("Phase 5: Full Integration - Prompt → Image → Wallpaper")
    
    start_time = datetime.now()
    
    # Step 1: Start Ollama server
    log("Step 1/8: Starting Ollama server...")
    start_ollama_server()
    
    # Step 2: Ensure model is available
    log("Step 2/8: Ensuring deepseek-r1:14b model is available...")
    ensure_deepseek_model()
    
    # Step 3: Load history
    log("Step 3/8: Loading prompt history...")
    history = load_prompt_history()
    
    # Step 4: Generate unique prompt
    log("Step 4/8: Generating unique creative prompt...")
    prompt = generate_prompt_with_deepseek(history)
    log(f"Generated prompt: {prompt[:100]}..." if len(prompt) > 100 else f"Generated prompt: {prompt}")
    
    # Step 5: Generate image (moved before save_prompt to get seed)
    log("Step 5/8: Generating 4K image from prompt...")
    image_path, seed = generate_image(prompt)
    log(f"Image saved: {os.path.basename(image_path)}")
    
    # Step 6: Save prompt and seed to history
    log("Step 6/8: Saving prompt and seed to history...")
    save_prompt(prompt, seed)
    
    # Step 7: Set wallpaper
    log("Step 7/8: Setting desktop wallpaper...")
    set_wallpaper(image_path)
    
    # Step 8: Verify wallpaper
    log("Step 8/8: Verifying wallpaper was set...")
    verify_wallpaper(image_path)
    
    # Calculate total time
    end_time = datetime.now()
    duration = end_time - start_time
    minutes = int(duration.total_seconds() / 60)
    seconds = int(duration.total_seconds() % 60)
    
    log(f"Total generation time: {minutes}m {seconds}s")
    log(f"Wallpaper path: {image_path}")
    log("=== DAILY WALLPAPER GENERATION COMPLETE ===")
    
    return image_path

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="AI-Powered Daily Wallpaper Generator")
    parser.add_argument("--test-prompt", action="store_true", help="Test prompt generation only")
    parser.add_argument("--test-image", action="store_true", help="Test image generation only")
    parser.add_argument("--test-wallpaper", action="store_true", help="Test wallpaper setting only")
    parser.add_argument("--run-now", action="store_true", help="Run full generation: prompt → image → wallpaper")
    parser.add_argument("--prompt", type=str, help="Prompt to use for image generation")
    parser.add_argument("--image", type=str, help="Image path to use for wallpaper setting")
    
    args = parser.parse_args()
    
    if args.test_prompt:
        test_prompt_generation()
    elif args.test_image:
        test_image_generation(args.prompt)
    elif args.test_wallpaper:
        test_wallpaper_setting(args.image)
    elif args.run_now:
        full_generation()
    else:
        log("Usage: daily_wallpaper.py [--run-now | --test-prompt | --test-image | --test-wallpaper]")
        log("  --run-now: Generate new wallpaper (full pipeline)")
        log("  --test-prompt: Test prompt generation only")
        log("  --test-image: Test image generation only")
        log("  --test-wallpaper: Test wallpaper setting only")

if __name__ == "__main__":
    main()