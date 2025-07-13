#!/usr/bin/env python3
"""
AI-Powered Daily Wallpaper Generator - GPT-Image-1 Edition
Phase 1: Basic prompt generation with deepseek-r1:14b
Phase 2: Image generation with GPT-Image-1 (OpenAI's latest model)
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
import base64
import requests
import shelve
import atexit
from PIL import Image
import shutil

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

# Weather configuration constants
ELKHORN_WI_LAT = 42.6728
ELKHORN_WI_LON = -88.5443
WEATHER_CACHE_DIR = "/home/user/ai-wallpaper/.weather_cache"
CACHE_GRID_EXPIRATION = 24 * 3600  # 1 day
CACHE_FORECAST_EXPIRATION = 15 * 60  # 15 minutes
TOO_MANY_API_CALLS_DELAY = 60  # 1 minute
WEATHER_AGENT = '(ai-wallpaper weather, ai-wallpaper@example.com)'
WEATHER_ACCEPT = 'application/geo+json'

# GPT-Image-1 configuration
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
if not OPENAI_API_KEY:
    print("ERROR: OPENAI_API_KEY environment variable not set")
    sys.exit(1)

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
    """Generate a unique, creative image prompt using deepseek-r1:14b with themed guidance for GPT-Image-1"""
    log("Generating creative prompt with deepseek-r1:14b for GPT-Image-1...")

    context = get_context_info()

    # Get themed instruction from theme selector
    log("Selecting theme for prompt generation...")
    theme_result = get_random_theme_with_weather(context['weather'])
    
    # Log the selected theme
    log(f"Using theme: {theme_result['theme']['name']} from category {theme_result['category']}")

    # DeepSeek instruction optimized for GPT-Image-1 (shorter, clearer prompts)
    deepseek_instruction = f"""
Generate a single, vividly descriptive image prompt for an amazing image, optimized for GPT-Image-1.

{theme_result['instruction']}

Context: It's {context['day_of_week']} in {context['season']}.

Requirements:
- The prompt MUST be between 50-75 words. Do NOT exceed 80 words.
- The prompt MUST be the **only** thing in your output. No extra text, no commentary.
- Use clear, concrete visual descriptions that GPT-Image-1 can interpret.
- Describe a complete scene with foreground, midground, and background elements.
- Include specific lighting and atmospheric details.
- The prompt MUST ensure photorealistic hyperrealistic extremely detailed and high quality images.
- Be creative and surprising, make sure prompt is unique.
- Use the provided theme AND weather AND date/time context in the image.
- Focus on photorealistic, ultra-detail.

ONLY return the image prompt. Nothing else.

Image prompt:
""".strip()

    log("Sending instruction to deepseek-r1:14b...")

    cmd = [OLLAMA_PATH, "run", "deepseek-r1:14b", deepseek_instruction]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)

    raw_output = result.stdout
    log(f"Raw output from model: {raw_output[:200]}...")

    generated_prompt = raw_output.strip()

    log("Stopping deepseek-r1:14b model to free memory...")
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

def save_prompt(prompt):
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
    """Generate image using GPT-Image-1 direct API - NO FALLBACKS"""
    log("Starting image generation with GPT-Image-1 direct API...")
    
    # Ensure images directory exists
    os.makedirs(IMAGES_DIR, exist_ok=True)
    
    # Create filename
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    prompt_excerpt = sanitize_filename(prompt)
    filename = f"{timestamp}_{prompt_excerpt}.png"
    filepath = os.path.join(IMAGES_DIR, filename)
    
    log(f"Image will be saved to: {filepath}")
    
    # GPT-Image-1 direct API request (highest quality settings)
    log("Calling GPT-Image-1 direct API with highest quality settings...")
    log("Note: High-quality image generation can take 2-3 minutes...")
    
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": "gpt-image-1",
        "prompt": prompt,
        "n": 1,
        "size": "1536x1024",        # Landscape format perfect for wallpapers
        "quality": "high"           # Highest quality available
    }
    
    # Make API request with retry logic
    max_attempts = 3
    for attempt in range(max_attempts):
        try:
            response = requests.post(
                "https://api.openai.com/v1/images/generations",
                headers=headers,
                json=data,
                timeout=180  # 3 minutes timeout for GPT-Image-1 generation
            )
            
            if response.status_code == 200:
                break
            elif response.status_code == 403:
                # Organization verification required for GPT-Image-1
                log("ERROR: Organization verification required for GPT-Image-1")
                log("Please verify your organization in the OpenAI console to use GPT-Image-1")
                log("Visit: https://platform.openai.com/settings/organization")
                log("Note: Verification can take up to 15 minutes to propagate")
                sys.exit(1)
            elif response.status_code == 429:
                # Rate limited
                wait_time = 30 * (attempt + 1)
                log(f"Rate limited, waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
            else:
                log(f"API error: {response.status_code} - {response.text}")
                if attempt == max_attempts - 1:
                    sys.exit(1)
        except requests.exceptions.Timeout as e:
            log(f"Request timeout (attempt {attempt + 1}): {e}")
            log("GPT-Image-1 generation can take several minutes - this is normal")
            if attempt == max_attempts - 1:
                log("ERROR: All timeout attempts exhausted")
                sys.exit(1)
            wait_time = 30 * (attempt + 1)
            log(f"Waiting {wait_time} seconds before retry...")
            time.sleep(wait_time)
        except Exception as e:
            log(f"Request failed: {e}")
            if attempt == max_attempts - 1:
                sys.exit(1)
            time.sleep(10)
    
    # Parse response - Debug the actual structure first
    try:
        result = response.json()
        log(f"Response structure - Top level keys: {list(result.keys())}")
        
        # Check if response contains image data
        if 'data' in result and len(result['data']) > 0:
            image_data = result['data'][0]
            log(f"Image data keys: {list(image_data.keys())}")
            
            # Check for different possible base64 fields
            if 'b64_json' in image_data:
                image_base64 = image_data['b64_json']
                log(f"Found b64_json field, extracting base64 data...")
            elif 'image' in image_data:
                image_base64 = image_data['image']
                log(f"Found image field, extracting base64 data...")
            elif 'data' in image_data:
                image_base64 = image_data['data']
                log(f"Found data field, extracting base64 data...")
            elif 'url' in image_data:
                # URL format fallback
                image_url = image_data['url']
                log(f"Got URL format, downloading from: {image_url}")
                image_response = requests.get(image_url, timeout=30)
                if image_response.status_code != 200:
                    log(f"ERROR: Failed to download image: {image_response.status_code}")
                    sys.exit(1)
                image_bytes = image_response.content
            else:
                log(f"ERROR: No recognized image field. Available keys: {list(image_data.keys())}")
                log(f"First 200 chars of response: {str(result)[:200]}...")
                sys.exit(1)
        else:
            log(f"ERROR: No data array in response")
            log(f"Response keys: {list(result.keys())}")
            log(f"First 200 chars of response: {str(result)[:200]}...")
            sys.exit(1)
            
    except Exception as e:
        log(f"ERROR: Failed to parse API response: {e}")
        log(f"Response: {response.text[:500]}...")
        sys.exit(1)
    
    # Decode and save image (if base64)
    try:
        if 'image_base64' in locals():
            # Decode base64 image data
            image_bytes = base64.b64decode(image_base64)
            log("Successfully decoded base64 image data")
        
        # Save image
        with open(filepath, 'wb') as f:
            f.write(image_bytes)
        log(f"Image decoded and saved: {filename}")
        
    except Exception as e:
        log(f"ERROR: Failed to decode/save image: {e}")
        sys.exit(1)
    
    # Verify and get dimensions
    try:
        image = Image.open(filepath)
        log(f"Image dimensions: {image.size}")
        image.close()
    except Exception as e:
        log(f"ERROR: Failed to verify image: {e}")
        sys.exit(1)
    
    return filepath

def process_image_to_4k(image_path):
    """Process GPT-Image-1 image to 4K with crop, upscale, and downsample"""
    log("Starting image processing pipeline...")
    
    # Stage 1: Load and crop to 16:9
    log("Stage 1: Loading and cropping to 16:9...")
    image = Image.open(image_path)
    log(f"Original dimensions: {image.size}")
    
    # Verify expected dimensions
    if image.size != (1536, 1024):
        log(f"WARNING: Unexpected dimensions {image.size}, expected (1536, 1024)")
    
    # Crop height from 1024 to 864 (remove 80px top, 80px bottom) for 16:9 ratio
    image_cropped = image.crop((0, 80, 1536, 944))  # 944 = 1024 - 80
    log(f"Cropped dimensions: {image_cropped.size}")
    
    # Verify 16:9 ratio
    ratio = image_cropped.size[0] / image_cropped.size[1]
    expected_ratio = 16 / 9
    log(f"Aspect ratio: {ratio:.4f} (expected: {expected_ratio:.4f})")
    
    # Save cropped image for Real-ESRGAN
    temp_cropped = "/tmp/gptimage_cropped.png"
    image_cropped.save(temp_cropped, "PNG", quality=100)
    log(f"Saved cropped image for upscaling: {temp_cropped}")
    
    # Save Stage 1 output
    stage1_path = image_path.replace(".png", "_stage1_cropped.png")
    image_cropped.save(stage1_path, "PNG", quality=100)
    log(f"Stage 1 complete: {stage1_path}")
    
    # Stage 2: Real-ESRGAN upscaling to near-8K
    log("Stage 2: Real-ESRGAN ultra-quality upscaling...")
    
    # Find Real-ESRGAN
    realesrgan_paths = [
        "/home/user/ai-wallpaper/Real-ESRGAN/inference_realesrgan.py",
        "/home/user/Real-ESRGAN/inference_realesrgan.py",
        os.path.expanduser("~/Real-ESRGAN/inference_realesrgan.py"),
        "/usr/local/bin/realesrgan-ncnn-vulkan"
    ]
    
    realesrgan_script = None
    for path in realesrgan_paths:
        if os.path.exists(path):
            realesrgan_script = path
            log(f"Found Real-ESRGAN at: {realesrgan_script}")
            break
    
    if not realesrgan_script:
        log("ERROR: Real-ESRGAN not found! Cannot proceed with 4K upscaling.")
        log("Real-ESRGAN is REQUIRED for ultra-high-quality 4K wallpapers.")
        sys.exit(1)
    
    # Run Real-ESRGAN
    temp_output_dir = "/tmp/gptimage_upscaled"
    
    if realesrgan_script.endswith('.py'):
        # Python script version
        cmd = [
            sys.executable, realesrgan_script,
            "-n", "RealESRGAN_x4plus",      # Best quality model
            "-i", temp_cropped,
            "-o", temp_output_dir,
            "--outscale", "4",               # 4x upscale
            "-t", "1024",                    # Large tile size for RTX 3090
            "--fp32"                         # Maximum precision
        ]
    else:
        # Binary version
        cmd = [
            realesrgan_script,
            "-i", temp_cropped,
            "-o", temp_output_dir,
            "-s", "4",
            "-n", "realesrgan-x4plus",
            "-t", "1024"
        ]
    
    log(f"Executing: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    log("Real-ESRGAN output: " + result.stdout)
    if result.stderr:
        log("Real-ESRGAN warnings: " + result.stderr)
    
    # Find upscaled image
    if os.path.exists(temp_output_dir):
        if os.path.isdir(temp_output_dir):
            output_files = [f for f in os.listdir(temp_output_dir) if f.endswith('.png')]
            if output_files:
                actual_output = os.path.join(temp_output_dir, output_files[0])
            else:
                log("ERROR: No PNG files found in Real-ESRGAN output directory!")
                sys.exit(1)
        else:
            actual_output = temp_output_dir
    else:
        log("ERROR: Real-ESRGAN output not found!")
        sys.exit(1)
    
    log(f"Loading upscaled image from {actual_output}")
    image_upscaled = Image.open(actual_output)
    log(f"Upscaled dimensions: {image_upscaled.size}")
    
    # Expected: 1536x864 * 4 = 6144x3456
    if image_upscaled.size != (6144, 3456):
        log(f"WARNING: Unexpected upscaled size {image_upscaled.size}, expected (6144, 3456)")
    
    # Save Stage 2 output
    stage2_path = image_path.replace(".png", "_stage2_upscaled.png")
    image_upscaled.save(stage2_path, "PNG", quality=100)
    log(f"Stage 2 complete: {stage2_path}")
    
    # Stage 3: High-quality downsample to 4K
    log("Stage 3: Downsampling to 4K using Lanczos...")
    log(f"Input size: {image_upscaled.size}")
    
    # Downsample to 4K using high-quality Lanczos filter
    image_4k = image_upscaled.resize((3840, 2160), Image.Resampling.LANCZOS)
    log(f"Downsampled to: {image_4k.size}")
    
    # Verify final size
    if image_4k.size != (3840, 2160):
        log(f"ERROR: Wrong final dimensions {image_4k.size}, expected (3840, 2160)")
        sys.exit(1)
    
    # Save final 4K image
    final_path = image_path.replace(".png", "_final_4k.png")
    image_4k.save(final_path, "PNG", quality=100)
    log(f"Final 4K wallpaper saved: {final_path}")
    
    # Cleanup temporary files
    if os.path.exists(temp_cropped):
        os.remove(temp_cropped)
    if os.path.exists(temp_output_dir):
        if os.path.isdir(temp_output_dir):
            shutil.rmtree(temp_output_dir)
        else:
            os.remove(temp_output_dir)
    
    log("Image processing complete!")
    return final_path

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
    image_path = generate_image(prompt)
    
    # Process to 4K
    final_path = process_image_to_4k(image_path)
    
    log("=== IMAGE GENERATION TEST COMPLETE ===")
    log(f"Original saved to: {image_path}")
    log(f"4K version saved to: {final_path}")
    
    return final_path

def full_generation():
    """Complete wallpaper generation from prompt to desktop"""
    log("=== STARTING DAILY WALLPAPER GENERATION (GPT-IMAGE-1 EDITION) ===")
    log("Pipeline: Prompt → GPT-Image-1 → 4K Processing → Wallpaper")
    
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
    
    # Step 5: Generate image with GPT-Image-1
    log("Step 5/8: Generating image with GPT-Image-1...")
    image_path = generate_image(prompt)
    log(f"Image saved: {os.path.basename(image_path)}")
    
    # Step 6: Process to 4K
    log("Step 6/8: Processing image to 4K...")
    final_image_path = process_image_to_4k(image_path)
    
    # Step 7: Save prompt to history
    log("Step 7/8: Saving prompt to history...")
    save_prompt(prompt)
    
    # Step 8: Set wallpaper
    log("Step 8/8: Setting desktop wallpaper...")
    set_wallpaper(final_image_path)
    
    # Verify wallpaper
    verify_wallpaper(final_image_path)
    
    # Calculate total time
    end_time = datetime.now()
    duration = end_time - start_time
    minutes = int(duration.total_seconds() / 60)
    seconds = int(duration.total_seconds() % 60)
    
    log(f"Total generation time: {minutes}m {seconds}s")
    log(f"Wallpaper path: {final_image_path}")
    log("=== DAILY WALLPAPER GENERATION COMPLETE ===")
    
    return final_image_path

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="AI-Powered Daily Wallpaper Generator - GPT-Image-1 Edition")
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
        log("Usage: daily_wallpaper_gpt2.py [--run-now | --test-prompt | --test-image | --test-wallpaper]")
        log("  --run-now: Generate new wallpaper (full pipeline)")
        log("  --test-prompt: Test prompt generation only")
        log("  --test-image: Test image generation only")
        log("  --test-wallpaper: Test wallpaper setting only")

if __name__ == "__main__":
    main()