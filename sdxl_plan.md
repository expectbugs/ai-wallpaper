# SDXL Maximum Quality Pipeline Implementation Plan

## Executive Summary

This plan outlines the implementation of an uncompromising SDXL pipeline that achieves maximum possible image quality through:
1. Native SDXL generation
2. 8K upscaling via Real-ESRGAN
3. 8K detail enhancement using tiled img2img
4. 4K downsampling for perfect anti-aliasing
5. Multi-LoRA stacking with theme integration

**Core Philosophy**: No fallbacks, no compromises. Every component must work perfectly or fail loudly.

## Architecture Overview

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Theme System  │────▶│ LoRA Selector    │────▶│ SDXL Generator  │
└─────────────────┘     └──────────────────┘     └────────┬────────┘
                                                            │ 1920x1024
                        ┌──────────────────┐                ▼
                        │ Detail Enhancer  │     ┌─────────────────┐
                        │ (Tiled Img2Img) │◀────│ Real-ESRGAN 4x  │
                        └────────┬─────────┘     └─────────────────┘
                                 │ 7680x4096                 
                                 ▼
                        ┌──────────────────┐
                        │ Lanczos Resizer  │
                        └────────┬─────────┘
                                 │ 3840x2160
                                 ▼
                        ┌──────────────────┐
                        │ Final 4K Output  │
                        └──────────────────┘
```

## Phase 1: SDXL Base Implementation (Week 1)

### 1.1 Fix SDXL Model Loading

**Current Issue**: SDXL expects HuggingFace repo or diffusers directory, not single checkpoint.

**Solution**: Download and use the official SDXL base model, then implement checkpoint conversion.

```python
# ai_wallpaper/models/sdxl_model.py modifications
def _load_from_single_file(self, checkpoint_path: str):
    """Load SDXL from single safetensors checkpoint"""
    from diffusers import StableDiffusionXLPipeline
    
    # CRITICAL: Must convert checkpoint to pipeline format
    pipe = StableDiffusionXLPipeline.from_single_file(
        checkpoint_path,
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16"
    )
    return pipe
```

### 1.2 Configure SDXL Pipeline

```yaml
# models.yaml update
sdxl:
  class: SdxlModel
  enabled: true
  display_name: "SDXL Ultimate"
  model_path: "/home/user/ai-wallpaper/models/sdxl-base"  # Official SDXL base
  checkpoint_path: "/home/user/vidgen/ComfyUI/models/checkpoints/SDXL/sd_xl_base_1.0.safetensors"  # Future
  
  generation:
    dimensions: [1920, 1024]  # Optimal for 16:9 at SDXL scale
    scheduler: "DPMSolverMultistepScheduler"
    torch_dtype: float16
    steps: 50  # Base quality
    guidance_scale: 7.5
    
  pipeline:
    type: "sdxl_ultimate"
    stages:
      generation:
        resolution: [1920, 1024]
        steps: 50
      upscale:
        model: "RealESRGAN_x4plus"
        scale: 4
        target: [7680, 4096]
      enhancement:
        method: "tiled_img2img"
        tile_size: 1024
        overlap: 256
        strength: 0.35
        steps: 30
      downsample:
        resolution: [3840, 2160]
        method: "lanczos"
```

### 1.3 Test Basic SDXL Generation

```bash
# Test command
./ai-wallpaper generate --model sdxl --no-upscale --save-stages
```

**Expected Result**: 1920x1024 SDXL image saved successfully.

## Phase 2: 8K Enhancement Pipeline (Week 2-3)

### 2.1 Implement Tiled Img2Img

**Critical Component**: Process 8K images in tiles to manage VRAM.

```python
# ai_wallpaper/processing/enhancer.py (NEW FILE)
class TiledImageEnhancer:
    """8K image enhancement using tiled img2img"""
    
    def __init__(self, pipe, tile_size=1024, overlap=256):
        self.pipe = pipe
        self.tile_size = tile_size
        self.overlap = overlap
        
    def enhance_image(self, image_8k: Image, prompt: str, strength: float = 0.35) -> Image:
        """Enhance 8K image using tiled processing"""
        width, height = image_8k.size
        
        # CRITICAL: Validate 8K dimensions
        if width != 7680 or height != 4096:
            raise ValueError(f"Image must be exactly 7680x4096, got {width}x{height}")
            
        # Create tile grid
        tiles = self._create_tile_grid(width, height)
        enhanced_tiles = []
        
        for tile_info in tiles:
            # Extract tile with overlap
            tile = self._extract_tile(image_8k, tile_info)
            
            # Enhance tile
            enhanced = self._enhance_tile(tile, prompt, strength)
            
            # Store with position info
            enhanced_tiles.append((enhanced, tile_info))
            
        # Blend tiles back together
        return self._blend_tiles(enhanced_tiles, width, height)
```

### 2.2 Memory Management

**VRAM Requirements**:
- 8K image in memory: ~768MB
- SDXL model: ~6GB
- Tile processing: ~4GB
- Total: ~11GB (safe for RTX 3090)

```python
def _enhance_tile(self, tile: Image, prompt: str, strength: float) -> Image:
    """Enhance single tile with aggressive VRAM management"""
    # Clear cache before processing
    torch.cuda.empty_cache()
    
    # Process tile
    enhanced = self.pipe(
        prompt=prompt,
        image=tile,
        strength=strength,
        num_inference_steps=30,
        guidance_scale=7.5
    ).images[0]
    
    # Clear cache after processing
    torch.cuda.empty_cache()
    
    return enhanced
```

### 2.3 Tile Blending Algorithm

```python
def _blend_tiles(self, tiles: List[Tuple[Image, TileInfo]], width: int, height: int) -> Image:
    """Blend tiles with overlap using gradient masks"""
    canvas = Image.new('RGB', (width, height))
    
    for tile_image, tile_info in tiles:
        # Create gradient mask for seamless blending
        mask = self._create_blend_mask(tile_info)
        
        # Paste with blending
        canvas.paste(tile_image, (tile_info.x, tile_info.y), mask)
        
    return canvas
```

## Phase 3: Pipeline Integration (Week 3)

### 3.1 Update SDXL Model Class

```python
# Add to sdxl_model.py
def _run_generation_pipeline(self, prompt: str, seed: int, **params) -> Dict[str, Any]:
    """Run complete SDXL ultimate pipeline"""
    
    # Stage 1: Base generation
    base_image = self._generate_base(prompt, seed, **params)
    
    # Stage 2: Upscale to 8K
    image_8k = self._upscale_to_8k(base_image)
    
    # Stage 3: Enhance at 8K
    enhanced_8k = self._enhance_8k_details(image_8k, prompt)
    
    # Stage 4: Downsample to 4K
    final_4k = self._downsample_to_4k(enhanced_8k)
    
    return {
        'base': base_image,
        'upscaled_8k': image_8k,
        'enhanced_8k': enhanced_8k,
        'final_4k': final_4k
    }
```

### 3.2 Testing Protocol

```python
# Test each stage independently
def test_sdxl_pipeline():
    # Test 1: Base generation
    assert base_image.size == (1920, 1024)
    
    # Test 2: 8K upscaling
    assert upscaled.size == (7680, 4096)
    
    # Test 3: Enhancement (check VRAM usage)
    assert torch.cuda.max_memory_allocated() < 23 * 1024**3  # Under 23GB
    
    # Test 4: Final quality
    assert final.size == (3840, 2160)
```

## Phase 4: LoRA System Architecture (Week 4-5)

### 4.1 LoRA Management System

```yaml
# New file: ai_wallpaper/config/loras.yaml
lora_library:
  # Style LoRAs
  style_photorealistic:
    path: "/home/user/ai-wallpaper/loras/photorealism_v2.safetensors"
    type: "style"
    weight_range: [0.6, 0.9]
    compatible_themes: ["PHOTOREALISTIC", "NATURE", "URBAN"]
    
  style_anime:
    path: "/home/user/ai-wallpaper/loras/anime_style_v3.safetensors"
    type: "style"
    weight_range: [0.7, 1.0]
    compatible_themes: ["ANIME_MANGA", "GENRE_FUSION"]
    
  # Detail LoRAs
  detail_enhancer:
    path: "/home/user/ai-wallpaper/loras/detail_tweaker.safetensors"
    type: "detail"
    weight_range: [0.3, 0.6]
    stackable: true
    
  # Character LoRAs
  char_yuffie:
    path: "/home/user/ai-wallpaper/loras/yuffie_kisaragi.safetensors"
    type: "character"
    weight_range: [0.8, 1.0]
    trigger_words: ["yuffie kisaragi", "ff7 yuffie"]
    specific_themes: ["final_fantasy_vii"]
    
# Stacking rules
stacking_rules:
  max_loras: 5
  type_limits:
    style: 1      # Only one style LoRA
    detail: 2     # Up to 2 detail LoRAs
    character: 2  # Up to 2 characters
  weight_sum_max: 3.5  # Total weights shouldn't exceed this
```

### 4.2 LoRA Selector Engine

```python
# ai_wallpaper/lora/selector.py (NEW FILE)
class LoRASelector:
    """Intelligent LoRA selection based on theme"""
    
    def select_loras_for_theme(self, theme: Dict, weather: Dict) -> List[LoRAConfig]:
        """Select optimal LoRAs for given theme"""
        selected = []
        
        # 1. Select primary style LoRA
        style_lora = self._select_style_lora(theme)
        if style_lora:
            selected.append(style_lora)
            
        # 2. Add detail enhancers based on weather/time
        if weather.get('mood') == 'vibrant':
            detail_lora = self._get_lora('detail_enhancer')
            detail_lora.weight = random.uniform(0.4, 0.6)
            selected.append(detail_lora)
            
        # 3. Add character LoRAs for specific themes
        char_loras = self._select_character_loras(theme)
        selected.extend(char_loras)
        
        # 4. Validate stack compatibility
        self._validate_lora_stack(selected)
        
        return selected
```

### 4.3 LoRA Loading and Application

```python
# Integration with SDXL pipeline
def _apply_loras(self, lora_configs: List[LoRAConfig]):
    """Apply multiple LoRAs with proper weighting"""
    
    # Clear any existing LoRAs
    self.pipe.unload_lora_weights()
    
    # Load each LoRA
    adapter_names = []
    adapter_weights = []
    
    for lora in lora_configs:
        # Load LoRA weights
        self.pipe.load_lora_weights(
            lora.path,
            adapter_name=lora.name
        )
        adapter_names.append(lora.name)
        adapter_weights.append(lora.weight)
        
        self.logger.info(f"Loaded LoRA: {lora.name} @ {lora.weight}")
        
    # Set all adapters with weights
    if adapter_names:
        self.pipe.set_adapters(adapter_names, adapter_weights=adapter_weights)
```

## Phase 5: Theme Integration (Week 5)

### 5.1 Extended Theme Configuration

```yaml
# Update themes.yaml
themes:
  final_fantasy_vii:
    name: "Final Fantasy VII"
    lora_hints:
      required: ["char_yuffie"]  # Always use this LoRA
      preferred: ["style_anime", "detail_enhancer"]
      excluded: ["style_photorealistic"]  # Never use these
    prompt_modifiers:
      prepend: "in the style of final fantasy vii, "
      append: ", highly detailed, game art style"
```

### 5.2 Prompt Engineering for LoRAs

```python
def _enhance_prompt_for_loras(self, base_prompt: str, loras: List[LoRAConfig]) -> str:
    """Add LoRA trigger words and modifiers"""
    prompt_parts = [base_prompt]
    
    # Add trigger words
    for lora in loras:
        if lora.trigger_words:
            # Intelligently insert trigger words
            prompt_parts.append(random.choice(lora.trigger_words))
            
    # Add quality modifiers for detail LoRAs
    if any(l.type == 'detail' for l in loras):
        prompt_parts.append("intricate details, sharp focus, masterpiece")
        
    return ", ".join(prompt_parts)
```

## Phase 6: Full Integration Testing (Week 6)

### 6.1 Performance Benchmarks

```python
# Expected timings
BENCHMARK_TARGETS = {
    'base_generation': 180,      # 3 minutes
    'upscale_to_8k': 480,       # 8 minutes  
    'enhance_8k': 1200,         # 20 minutes
    'downsample_to_4k': 120,    # 2 minutes
    'total_pipeline': 2000      # ~33 minutes
}
```

### 6.2 Quality Validation

```python
def validate_output_quality(image_path: str) -> Dict[str, float]:
    """Validate final image meets quality standards"""
    img = Image.open(image_path)
    
    # Check resolution
    assert img.size == (3840, 2160), "Must be exactly 4K"
    
    # Check sharpness (using Laplacian variance)
    sharpness = calculate_sharpness(img)
    assert sharpness > MINIMUM_SHARPNESS_THRESHOLD
    
    # Check detail density
    detail_score = calculate_detail_density(img)
    assert detail_score > MINIMUM_DETAIL_THRESHOLD
    
    return {
        'sharpness': sharpness,
        'detail_score': detail_score
    }
```

## Phase 7: Contextual Information Integration (Week 7)

### 7.1 Intelligent Text Placement System

**Revolutionary Feature**: Automatically embed weather, date, and contextual information directly into the generated image in theme-appropriate ways.

```python
# ai_wallpaper/contextual/text_integrator.py (NEW FILE)
class ContextualTextIntegrator:
    """Intelligently place weather/date info within generated images"""
    
    def __init__(self):
        self.object_detector = self._load_object_detection_model()
        self.text_placer = self._load_text_placement_engine()
        self.style_matcher = self._load_style_matching_model()
        
    def integrate_contextual_info(self, image: Image, weather: Dict, date: datetime, theme: Dict) -> Image:
        """Add weather and date info naturally into the image"""
        
        # 1. Detect optimal placement locations
        placement_candidates = self._detect_text_surfaces(image, theme)
        
        # 2. Generate theme-appropriate text content
        weather_text = self._format_weather_for_theme(weather, theme)
        date_text = self._format_date_for_theme(date, theme)
        
        # 3. Select best placement location
        optimal_spot = self._select_optimal_placement(placement_candidates, theme)
        
        # 4. Render text with proper perspective and lighting
        enhanced_image = self._render_contextual_text(
            image, weather_text, date_text, optimal_spot, theme
        )
        
        return enhanced_image
```

### 7.2 Theme-Appropriate Text Styling

```yaml
# New file: ai_wallpaper/config/text_styles.yaml
text_integration:
  enabled: true
  placement_confidence_threshold: 0.8
  
  theme_styles:
    cyberpunk:
      preferred_surfaces: ["holographic_display", "neon_sign", "computer_screen", "hud_interface"]
      font_style: "futuristic_mono"
      text_format: "SYSTEM: {date} | WEATHER.EXE: {temp}°F {condition}"
      effects: ["glow", "scan_lines", "digital_noise"]
      colors: ["cyan", "green", "orange"]
      
    fantasy:
      preferred_surfaces: ["scroll", "book_page", "stone_tablet", "banner", "parchment"]
      font_style: "medieval_script"
      text_format: "On the {ordinal_day} of {fantasy_month}, the realm sees {fantasy_weather}"
      effects: ["aged_paper", "ink_bleed", "magical_glow"]
      colors: ["sepia", "gold", "deep_blue"]
      
    office_modern:
      preferred_surfaces: ["computer_monitor", "notepad", "calendar", "whiteboard", "sticky_note"]
      font_style: "clean_sans"
      text_format: "{weekday}, {month} {day}\nForecast: {condition}, {temp}°F"
      effects: ["clean", "shadow"]
      colors: ["black", "blue", "gray"]
      
    nature:
      preferred_surfaces: ["tree_bark", "rock_face", "sky_writing", "sand", "snow"]
      font_style: "natural_carved"
      text_format: "{season_name} {day}: {natural_weather_description}"
      effects: ["carved", "weathered", "organic"]
      colors: ["brown", "green", "earth_tones"]
```

### 7.3 Object Detection and Placement Logic

```python
def _detect_text_surfaces(self, image: Image, theme: Dict) -> List[PlacementCandidate]:
    """Detect surfaces suitable for text placement"""
    
    # Run object detection for text-friendly surfaces
    detected_objects = self.object_detector.detect(image, classes=[
        'monitor', 'screen', 'paper', 'sign', 'book', 'tablet', 
        'notepad', 'whiteboard', 'banner', 'poster'
    ])
    
    # Score each detection based on theme compatibility
    candidates = []
    for obj in detected_objects:
        compatibility_score = self._score_theme_compatibility(obj, theme)
        size_score = self._score_size_appropriateness(obj)
        occlusion_score = self._score_occlusion_level(obj)
        
        overall_score = (compatibility_score * 0.5 + 
                        size_score * 0.3 + 
                        occlusion_score * 0.2)
        
        if overall_score > self.placement_confidence_threshold:
            candidates.append(PlacementCandidate(
                bbox=obj.bbox,
                surface_type=obj.class_name,
                confidence=overall_score,
                perspective_matrix=self._calculate_perspective(obj)
            ))
    
    return sorted(candidates, key=lambda x: x.confidence, reverse=True)
```

### 7.4 Natural Text Rendering

```python
def _render_contextual_text(self, image: Image, weather_text: str, date_text: str, 
                           placement: PlacementCandidate, theme: Dict) -> Image:
    """Render text naturally integrated into the scene"""
    
    # Get theme-specific styling
    style = self.get_theme_style(theme.name)
    
    # Create text image with proper perspective
    text_content = f"{date_text}\n{weather_text}"
    text_image = self._create_styled_text(text_content, style, placement)
    
    # Apply perspective transformation
    transformed_text = self._apply_perspective_transform(text_image, placement.perspective_matrix)
    
    # Match scene lighting
    lit_text = self._match_scene_lighting(transformed_text, image, placement.bbox)
    
    # Apply theme-specific effects
    effects_applied = self._apply_theme_effects(lit_text, style.effects)
    
    # Blend with original image
    blended = self._blend_with_scene(image, effects_applied, placement.bbox)
    
    return blended
```

### 7.5 Weather and Date Formatting

```python
def _format_weather_for_theme(self, weather: Dict, theme: Dict) -> str:
    """Format weather data according to theme style"""
    
    formatters = {
        'cyberpunk': lambda w: f"TEMP: {w.temp}°F | STATUS: {w.condition.upper()} | HUMIDITY: {w.humidity}%",
        'fantasy': lambda w: f"The {self._weather_to_fantasy(w.condition)} brings {self._temp_to_fantasy(w.temp)}",
        'office': lambda w: f"Today: {w.condition}, {w.temp}°F",
        'nature': lambda w: f"{w.condition} skies, {w.temp}° - {self._nature_advice(w)}"
    }
    
    formatter = formatters.get(theme.name, formatters['office'])
    return formatter(weather)

def _weather_to_fantasy(self, condition: str) -> str:
    """Convert weather to fantasy terminology"""
    mapping = {
        'Clear': 'crystal heavens',
        'Cloudy': 'grey wandering mists',
        'Rain': 'tears of the sky spirits',
        'Snow': 'frozen starlight dancing',
        'Thunderstorm': 'the fury of the storm gods'
    }
    return mapping.get(condition, 'mystical atmospheric energies')
```

### 7.6 Pipeline Integration

```python
# Add to SDXL model pipeline
def _run_generation_pipeline(self, prompt: str, seed: int, **params) -> Dict[str, Any]:
    """Extended pipeline with contextual text integration"""
    
    # ... existing pipeline stages ...
    
    # Stage 5: Contextual information integration
    if self.config.get('text_integration', {}).get('enabled', False):
        weather_data = get_weather_context()
        current_date = datetime.now()
        theme_data = params.get('theme', {})
        
        self.logger.info("Stage 5: Adding contextual information")
        final_with_context = self.text_integrator.integrate_contextual_info(
            final_4k, weather_data, current_date, theme_data
        )
        
        if params.get('save_stages'):
            save_path = f"{base_filename}_stage5_contextual.png"
            final_with_context.save(save_path)
            stages['contextual'] = save_path
            
        return {
            'image_path': save_path,
            'stages': stages,
            'metadata': {
                'weather': weather_data,
                'date': current_date.isoformat(),
                'text_placement': 'successful'
            }
        }
```

## Phase 8: Steam Integration and Holiday Themes (Week 8)

### 8.1 Steam Web API Integration

**Purpose**: Weight theme selection based on recently played games and incorporate game achievements into wallpapers.

```python
# ai_wallpaper/steam/steam_client.py (NEW FILE)
import requests
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional

class SteamClient:
    """Steam Web API client for game data and achievements"""
    
    def __init__(self, api_key: str, steam_id: str):
        self.api_key = api_key
        self.steam_id = steam_id
        self.base_url = "https://api.steampowered.com"
        
    def get_recent_games(self, days: int = 14) -> List[Dict]:
        """Get recently played games with playtime data"""
        url = f"{self.base_url}/IPlayerService/GetRecentlyPlayedGames/v0001/"
        params = {
            'key': self.api_key,
            'steamid': self.steam_id,
            'format': 'json',
            'count': 50
        }
        
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            games = data.get('response', {}).get('games', [])
            
            # Filter to games played in specified timeframe
            cutoff_date = datetime.now() - timedelta(days=days)
            recent_games = []
            
            for game in games:
                # Convert last_played to datetime for comparison
                last_played = datetime.fromtimestamp(game.get('last_played', 0))
                if last_played >= cutoff_date:
                    recent_games.append(game)
                    
            return recent_games
            
        except Exception as e:
            self.logger.error(f"Failed to fetch recent games: {e}")
            return []
    
    def get_game_achievements(self, app_id: int) -> List[Dict]:
        """Get player achievements for a specific game"""
        url = f"{self.base_url}/ISteamUserStats/GetPlayerAchievements/v0001/"
        params = {
            'key': self.api_key,
            'steamid': self.steam_id,
            'appid': app_id,
            'format': 'json'
        }
        
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            achievements = data.get('playerstats', {}).get('achievements', [])
            
            # Filter to recent achievements (last 30 days)
            recent_achievements = []
            cutoff_date = datetime.now() - timedelta(days=30)
            
            for achievement in achievements:
                if achievement.get('achieved') == 1:
                    unlock_time = datetime.fromtimestamp(achievement.get('unlocktime', 0))
                    if unlock_time >= cutoff_date:
                        recent_achievements.append(achievement)
                        
            return recent_achievements
            
        except Exception as e:
            self.logger.error(f"Failed to fetch achievements for app {app_id}: {e}")
            return []
```

### 8.2 Game-Based Theme Weighting System

```python
# ai_wallpaper/steam/theme_weigher.py (NEW FILE)
class GameThemeWeighter:
    """Weight theme selection based on Steam game data"""
    
    def __init__(self, steam_client: SteamClient):
        self.steam_client = steam_client
        self.game_theme_mapping = self._load_game_theme_mapping()
        
    def _load_game_theme_mapping(self) -> Dict:
        """Load mapping of Steam games to AI wallpaper themes"""
        return {
            # Popular games to theme mappings
            730: 'MODERN_WARFARE',      # CS2
            570: 'FANTASY_MEDIEVAL',    # Dota 2
            1172470: 'CYBERPUNK',       # Cyberpunk 2077
            292030: 'HORROR_GOTHIC',    # The Witcher 3
            1174180: 'SAMURAI_FEUDAL',  # Red Dead Redemption 2
            271590: 'HORROR_GOTHIC',    # Grand Theft Auto V
            1091500: 'CYBERPUNK',       # Cyberpunk 2077
            39210: 'FANTASY_MEDIEVAL',  # Final Fantasy XIV
            # Add more mappings as needed
        }
    
    def calculate_theme_weights(self) -> Dict[str, float]:
        """Calculate theme weights based on recent Steam activity"""
        recent_games = self.steam_client.get_recent_games(days=14)
        theme_weights = {}
        
        for game in recent_games:
            app_id = game.get('appid')
            playtime_2weeks = game.get('playtime_2weeks', 0)  # Minutes
            
            # Map game to theme
            theme = self.game_theme_mapping.get(app_id)
            if not theme:
                continue
                
            # Calculate weight based on playtime and recency
            recency_factor = self._calculate_recency_factor(game.get('last_played', 0))
            playtime_factor = min(playtime_2weeks / 600, 2.0)  # Cap at 10 hours = 2.0 factor
            
            game_weight = recency_factor * playtime_factor
            
            # Accumulate theme weights
            if theme not in theme_weights:
                theme_weights[theme] = 0
            theme_weights[theme] += game_weight
            
        # Normalize weights to sum to 1.0
        total_weight = sum(theme_weights.values())
        if total_weight > 0:
            theme_weights = {k: v/total_weight for k, v in theme_weights.items()}
            
        return theme_weights
    
    def _calculate_recency_factor(self, last_played_timestamp: int) -> float:
        """Calculate weight factor based on how recently game was played"""
        if last_played_timestamp == 0:
            return 0.1
            
        last_played = datetime.fromtimestamp(last_played_timestamp)
        days_ago = (datetime.now() - last_played).days
        
        if days_ago <= 1:
            return 2.0
        elif days_ago <= 3:
            return 1.5
        elif days_ago <= 7:
            return 1.0
        elif days_ago <= 14:
            return 0.5
        else:
            return 0.1
```

### 8.3 Achievement Integration for Game Themes

```python
# ai_wallpaper/steam/achievement_integrator.py (NEW FILE)
class AchievementIntegrator:
    """Integrate Steam achievements into game-themed wallpapers"""
    
    def __init__(self, steam_client: SteamClient):
        self.steam_client = steam_client
        
    def enhance_game_theme_prompt(self, theme: str, selected_game_id: Optional[int], 
                                 base_prompt: str) -> str:
        """Enhance prompt with achievement data if game theme was selected"""
        
        if not selected_game_id or theme not in ['GAME_INSPIRED']:
            return base_prompt
            
        # Get recent achievements for the selected game
        achievements = self.steam_client.get_game_achievements(selected_game_id)
        
        if not achievements:
            return base_prompt
            
        # Select 1-2 recent achievements to incorporate
        selected_achievements = achievements[:2]
        
        # Generate achievement-inspired prompt additions
        achievement_prompts = []
        for achievement in selected_achievements:
            name = achievement.get('name', '')
            description = achievement.get('description', '')
            
            # Convert achievement to visual elements
            visual_element = self._achievement_to_visual(name, description)
            if visual_element:
                achievement_prompts.append(visual_element)
        
        # Integrate into base prompt
        if achievement_prompts:
            enhanced_prompt = f"{base_prompt}, featuring {', '.join(achievement_prompts)}"
            return enhanced_prompt
            
        return base_prompt
    
    def _achievement_to_visual(self, name: str, description: str) -> str:
        """Convert achievement name/description to visual prompt element"""
        # Simple keyword mapping - can be expanded
        keyword_mappings = {
            'victory': 'triumphant victory celebration',
            'defeat': 'epic battle aftermath',
            'treasure': 'glowing treasure chest',
            'level': 'character progression aura',
            'boss': 'legendary boss encounter',
            'completion': 'quest completion celebration',
            'unlock': 'mystical unlock effect',
            'master': 'mastery achievement glow'
        }
        
        text = f"{name} {description}".lower()
        
        for keyword, visual in keyword_mappings.items():
            if keyword in text:
                return visual
                
        return None
```

### 8.4 Holiday Detection System

```python
# ai_wallpaper/holidays/holiday_detector.py (NEW FILE)
from datetime import datetime, date
import calendar

class HolidayDetector:
    """Detect nationally-recognized holidays and add to theme"""
    
    def __init__(self):
        self.holidays = self._initialize_holidays()
        
    def _initialize_holidays(self) -> Dict[str, Dict]:
        """Initialize static holiday definitions"""
        return {
            # Fixed date holidays
            'new_years': {
                'date': (1, 1),
                'name': 'New Year\'s Day',
                'theme_modifier': 'new year celebration with fireworks and champagne'
            },
            'independence_day': {
                'date': (7, 4),
                'name': 'Independence Day',
                'theme_modifier': 'patriotic celebration with red white and blue, fireworks'
            },
            'christmas': {
                'date': (12, 25),
                'name': 'Christmas',
                'theme_modifier': 'Christmas holiday with snow, decorated trees, warm lights'
            },
            'halloween': {
                'date': (10, 31),
                'name': 'Halloween',
                'theme_modifier': 'Halloween atmosphere with pumpkins, autumn leaves, spooky elements'
            },
            'valentines': {
                'date': (2, 14),
                'name': 'Valentine\'s Day',
                'theme_modifier': 'romantic Valentine\'s theme with hearts, roses, warm colors'
            },
            'st_patricks': {
                'date': (3, 17),
                'name': 'St. Patrick\'s Day',
                'theme_modifier': 'Irish celebration with green themes, shamrocks, festive atmosphere'
            },
            
            # Variable date holidays (calculated)
            'memorial_day': {
                'calculation': 'last_monday_may',
                'name': 'Memorial Day',
                'theme_modifier': 'patriotic memorial theme with American flags, solemn respect'
            },
            'labor_day': {
                'calculation': 'first_monday_september',
                'name': 'Labor Day',
                'theme_modifier': 'end of summer celebration, outdoor activities'
            },
            'thanksgiving': {
                'calculation': 'fourth_thursday_november',
                'name': 'Thanksgiving',
                'theme_modifier': 'autumn harvest theme with warm colors, gratitude, family gathering'
            }
        }
    
    def get_current_holiday(self, check_date: date = None) -> Optional[Dict]:
        """Check if current date is a nationally-recognized holiday"""
        if check_date is None:
            check_date = date.today()
            
        current_month = check_date.month
        current_day = check_date.day
        current_year = check_date.year
        
        # Check fixed date holidays
        for holiday_key, holiday_data in self.holidays.items():
            if 'date' in holiday_data:
                holiday_month, holiday_day = holiday_data['date']
                if current_month == holiday_month and current_day == holiday_day:
                    return holiday_data
                    
        # Check calculated holidays
        calculated_holidays = self._calculate_variable_holidays(current_year)
        for holiday_key, holiday_date in calculated_holidays.items():
            if check_date == holiday_date:
                return self.holidays[holiday_key]
                
        return None
    
    def _calculate_variable_holidays(self, year: int) -> Dict[str, date]:
        """Calculate variable date holidays for given year"""
        holidays = {}
        
        # Memorial Day - Last Monday in May
        last_monday_may = self._get_last_monday_of_month(year, 5)
        holidays['memorial_day'] = last_monday_may
        
        # Labor Day - First Monday in September
        first_monday_sept = self._get_first_monday_of_month(year, 9)
        holidays['labor_day'] = first_monday_sept
        
        # Thanksgiving - Fourth Thursday in November
        fourth_thursday_nov = self._get_fourth_thursday_of_month(year, 11)
        holidays['thanksgiving'] = fourth_thursday_nov
        
        return holidays
    
    def _get_last_monday_of_month(self, year: int, month: int) -> date:
        """Get the last Monday of a given month"""
        # Get the last day of the month
        last_day = calendar.monthrange(year, month)[1]
        last_date = date(year, month, last_day)
        
        # Find the last Monday
        days_to_subtract = (last_date.weekday() - 0) % 7
        last_monday = last_date - timedelta(days=days_to_subtract)
        
        return last_monday
    
    def _get_first_monday_of_month(self, year: int, month: int) -> date:
        """Get the first Monday of a given month"""
        first_date = date(year, month, 1)
        days_to_add = (7 - first_date.weekday()) % 7
        first_monday = first_date + timedelta(days=days_to_add)
        
        return first_monday
    
    def _get_fourth_thursday_of_month(self, year: int, month: int) -> date:
        """Get the fourth Thursday of a given month"""
        first_date = date(year, month, 1)
        first_thursday = first_date + timedelta(days=(3 - first_date.weekday()) % 7)
        fourth_thursday = first_thursday + timedelta(days=21)  # 3 weeks later
        
        return fourth_thursday
```

### 8.5 Integration with Theme Selection

```python
# ai_wallpaper/themes/enhanced_theme_selector.py (MODIFY EXISTING)
class EnhancedThemeSelector(ThemeSelector):
    """Extended theme selector with Steam and holiday integration"""
    
    def __init__(self, steam_client: Optional[SteamClient] = None):
        super().__init__()
        self.steam_client = steam_client
        self.game_weigher = GameThemeWeighter(steam_client) if steam_client else None
        self.holiday_detector = HolidayDetector()
        self.achievement_integrator = AchievementIntegrator(steam_client) if steam_client else None
        
    def select_theme(self, weather: Dict, time_context: Dict) -> Dict:
        """Enhanced theme selection with Steam weighting and holiday detection"""
        
        # Start with base theme selection
        base_theme = super().select_theme(weather, time_context)
        
        # Check for holidays first (takes precedence)
        current_holiday = self.holiday_detector.get_current_holiday()
        if current_holiday:
            holiday_theme = self._apply_holiday_theme(base_theme, current_holiday)
            return holiday_theme
            
        # Apply Steam-based weighting if available
        if self.game_weigher:
            steam_weights = self.game_weigher.calculate_theme_weights()
            if steam_weights:
                # Use weighted random selection
                selected_theme = self._weighted_theme_selection(base_theme, steam_weights)
                return selected_theme
                
        return base_theme
    
    def _apply_holiday_theme(self, base_theme: Dict, holiday: Dict) -> Dict:
        """Apply holiday theming to base theme"""
        holiday_theme = base_theme.copy()
        
        # Add holiday modifier to prompt
        holiday_modifier = holiday['theme_modifier']
        if 'prompt_modifiers' not in holiday_theme:
            holiday_theme['prompt_modifiers'] = {}
            
        # Prepend holiday theme to existing modifiers
        existing_prepend = holiday_theme['prompt_modifiers'].get('prepend', '')
        holiday_theme['prompt_modifiers']['prepend'] = f"{holiday_modifier}, {existing_prepend}"
        
        # Add holiday metadata
        holiday_theme['holiday'] = {
            'name': holiday['name'],
            'modifier': holiday_modifier
        }
        
        return holiday_theme
    
    def _weighted_theme_selection(self, base_theme: Dict, steam_weights: Dict) -> Dict:
        """Select theme using Steam game weights"""
        import random
        
        # Use weighted random selection
        themes = list(steam_weights.keys())
        weights = list(steam_weights.values())
        
        if themes and weights:
            selected_theme_name = random.choices(themes, weights=weights)[0]
            
            # Get the actual theme configuration
            selected_theme = self._get_theme_by_name(selected_theme_name)
            if selected_theme:
                return selected_theme
                
        return base_theme
    
    def enhance_prompt_with_achievements(self, theme: Dict, prompt: str) -> str:
        """Enhance prompt with Steam achievements if game theme selected"""
        if not self.achievement_integrator:
            return prompt
            
        # Check if this is a game-inspired theme
        game_id = theme.get('steam_game_id')
        if game_id:
            enhanced_prompt = self.achievement_integrator.enhance_game_theme_prompt(
                theme['name'], game_id, prompt
            )
            return enhanced_prompt
            
        return prompt
```

### 8.6 Configuration Integration

```yaml
# Add to ai_wallpaper/config/config.yaml
steam_integration:
  enabled: true
  api_key: "${STEAM_API_KEY}"  # Set in environment variables
  steam_id: "${STEAM_ID}"      # Set in environment variables
  theme_weighting: true
  achievement_integration: true
  cache_duration: 3600  # 1 hour cache for API calls

holiday_themes:
  enabled: true
  override_base_theme: true  # Holiday themes take precedence
  
  # Holiday-specific configurations
  christmas:
    lora_preferences: ["winter_scenes", "warm_lighting"]
    color_bias: ["red", "green", "gold"]
    
  halloween:
    lora_preferences: ["dark_atmosphere", "spooky_elements"]
    color_bias: ["orange", "black", "purple"]
    
  fourth_of_july:
    lora_preferences: ["patriotic_themes", "fireworks"]
    color_bias: ["red", "white", "blue"]
```

### 8.7 Environment Variables Setup

```bash
# Add to .env file
STEAM_API_KEY=your_steam_api_key_here
STEAM_ID=your_steam_id_here
```

### 8.8 Pipeline Integration

```python
# Modify main generation pipeline in sdxl_model.py
def _run_generation_pipeline(self, prompt: str, seed: int, **params) -> Dict[str, Any]:
    """Extended pipeline with Steam and holiday integration"""
    
    # Initialize Steam client if configured
    steam_client = None
    if self.config.get('steam_integration', {}).get('enabled', False):
        steam_client = SteamClient(
            api_key=os.getenv('STEAM_API_KEY'),
            steam_id=os.getenv('STEAM_ID')
        )
    
    # Enhanced theme selection
    theme_selector = EnhancedThemeSelector(steam_client)
    weather_data = get_weather_context()
    time_context = get_time_context()
    
    selected_theme = theme_selector.select_theme(weather_data, time_context)
    
    # Enhance prompt with achievements if applicable
    enhanced_prompt = theme_selector.enhance_prompt_with_achievements(selected_theme, prompt)
    
    # Log theme selection reasoning
    self.logger.info(f"Selected theme: {selected_theme.get('name', 'Unknown')}")
    if 'holiday' in selected_theme:
        self.logger.info(f"Holiday detected: {selected_theme['holiday']['name']}")
    
    # Continue with existing pipeline using enhanced_prompt and selected_theme
    # ... rest of pipeline remains the same ...
```

## Configuration Summary

### Final models.yaml for SDXL

```yaml
sdxl:
  class: SdxlModel
  enabled: true
  display_name: "SDXL Ultimate"
  model_path: "/home/user/ai-wallpaper/models/sdxl-base"
  
  generation:
    dimensions: [1920, 1024]
    scheduler: "DPMSolverMultistepScheduler"
    torch_dtype: float16
    steps_range: [40, 60]
    guidance_range: [6.0, 9.0]
    
  pipeline:
    type: "sdxl_ultimate"
    save_intermediates: true
    stages:
      generation:
        resolution: [1920, 1024]
        steps: 50
        cfg_scale: 7.5
      upscale:
        model: "RealESRGAN_x4plus"
        scale: 4
        tile_size: 1024
        fp32: true
      enhancement:
        method: "tiled_img2img"
        tile_size: 1024
        overlap: 256
        strength_range: [0.25, 0.45]
        steps: 30
        cfg_scale: 7.5
      downsample:
        resolution: [3840, 2160]
        method: "lanczos"
        
  lora:
    enabled: true
    max_count: 5
    auto_select: true
    weight_sum_max: 3.5
    
  memory:
    enable_model_cpu_offload: false  # Keep on GPU
    enable_sequential_cpu_offload: false  # No CPU offload
    enable_attention_slicing: true
    vae_tiling: true
    clear_cache_after_stage: true
```

## Implementation Checklist

### Week 1: Foundation
- [ ] Update SDXL model to handle both directory and checkpoint paths
- [ ] Implement basic SDXL generation
- [ ] Test with SDXL base model
- [ ] Verify 1920x1024 output quality

### Week 2-3: 8K Pipeline
- [ ] Implement TiledImageEnhancer class
- [ ] Add tile extraction and blending logic
- [ ] Integrate with SDXL pipeline
- [ ] Test memory usage stays under 23GB

### Week 4-5: LoRA System
- [ ] Create loras.yaml configuration
- [ ] Implement LoRASelector class
- [ ] Add LoRA loading to SDXL model
- [ ] Test multi-LoRA stacking

### Week 6: Integration
- [ ] Full pipeline testing
- [ ] Performance optimization
- [ ] Quality validation
- [ ] Documentation

### Week 7: Contextual Information
- [ ] Implement ContextualTextIntegrator class
- [ ] Add object detection for text placement
- [ ] Create theme-appropriate text styling system
- [ ] Integrate with SDXL pipeline
- [ ] Test natural text integration across all themes

### Week 8: Steam Integration and Holiday Themes
- [ ] Implement SteamClient class for Web API integration
- [ ] Create GameThemeWeighter for theme selection weighting
- [ ] Add AchievementIntegrator for game-themed wallpapers
- [ ] Implement HolidayDetector for nationally-recognized holidays
- [ ] Create EnhancedThemeSelector with Steam and holiday support
- [ ] Add Steam integration configuration to config.yaml
- [ ] Set up environment variables for Steam API
- [ ] Integrate Steam and holiday features into main pipeline
- [ ] Test theme weighting based on recent game activity
- [ ] Validate holiday detection and theme modification

## Critical Success Factors

1. **Memory Management**: Never exceed 23GB VRAM
2. **Tile Blending**: Seamless 8K processing
3. **LoRA Compatibility**: Careful weight balancing
4. **Performance**: Keep under 45 minutes total
5. **Quality**: Measurable improvement over base SDXL

## No-Compromise Requirements

1. **FAIL LOUD**: Every error must halt execution with detailed diagnostics
2. **MAXIMUM QUALITY**: No speed optimizations that reduce quality
3. **FULL PIPELINE**: Every stage must complete successfully
4. **VALIDATION**: Built-in quality checks at each stage
5. **DETERMINISTIC**: Same seed must produce identical results

This plan represents the absolute maximum quality achievable with current technology while remaining implementable and maintainable.