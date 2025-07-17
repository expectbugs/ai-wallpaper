#!/usr/bin/env python3
"""
Weather Context Module for AI Wallpaper System
Fetches weather data from NWS API with caching and fail-loud error handling
"""

import os
import sys
import time
import json
import shelve
import shutil
import atexit
import requests
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, Tuple

from .logger import get_logger
from .exceptions import WeatherError
from .config_manager import get_config

class WeatherCache:
    """Manages weather API caching with corruption detection"""
    
    def __init__(self, cache_dir: Path):
        """Initialize weather cache
        
        Args:
            cache_dir: Directory for cache storage
        """
        self.cache_dir = Path(cache_dir)
        self.cache_file = self.cache_dir / "nws_api.cache"
        self.logger = get_logger(model="Weather")
        self.cache = None
        
        # Initialize cache
        self._initialize_cache()
        
    def _initialize_cache(self):
        """Initialize or recreate cache with corruption detection"""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            self.cache = shelve.open(str(self.cache_file))
            atexit.register(self.close)
            self.logger.info("Weather cache initialized successfully")
        except Exception as e:
            self.logger.error(f"Weather cache corrupted: {e}")
            self._recreate_cache()
            
    def _recreate_cache(self):
        """Recreate cache when corruption is detected"""
        self.logger.warning("Recreating weather cache due to corruption...")
        
        # Close existing cache if open
        if self.cache:
            try:
                self.cache.close()
            except:
                pass
                
        # Remove corrupted cache and all shelve-related files
        # Shelve can create .bak, .dat, .dir, .db files
        cache_patterns = [
            self.cache_file,
            self.cache_file.with_suffix('.bak'),
            self.cache_file.with_suffix('.dat'),
            self.cache_file.with_suffix('.dir'),
            self.cache_file.with_suffix('.db'),
            # Also check for files with the base name + extensions
            self.cache_dir / f"{self.cache_file.name}.bak",
            self.cache_dir / f"{self.cache_file.name}.dat",
            self.cache_dir / f"{self.cache_file.name}.dir",
            self.cache_dir / f"{self.cache_file.name}.db"
        ]
        
        for cache_path in cache_patterns:
            if cache_path.exists():
                try:
                    cache_path.unlink()
                    self.logger.debug(f"Removed cache file: {cache_path}")
                except Exception as e:
                    self.logger.warning(f"Failed to remove {cache_path}: {e}")
                
        # Now try to remove the directory if empty
        if self.cache_dir.exists():
            try:
                # Only remove if directory is empty
                if not any(self.cache_dir.iterdir()):
                    self.cache_dir.rmdir()
                else:
                    self.logger.warning(f"Cache directory not empty, keeping: {self.cache_dir}")
            except Exception as e:
                self.logger.warning(f"Failed to remove cache directory: {e}")
                
        # Create fresh cache
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            self.cache = shelve.open(str(self.cache_file))
            atexit.register(self.close)
            self.logger.info("Weather cache recreated successfully")
        except Exception as e:
            raise WeatherError("weather cache", e)
            
    def get(self, url: str, expiration: int) -> Optional[Any]:
        """Get cached data if not expired
        
        Args:
            url: URL to check cache for
            expiration: Cache expiration in seconds
            
        Returns:
            Cached data or None if expired/not found
        """
        current_time = time.time()
        
        try:
            if url in self.cache:
                cached_time, cached_data = self.cache[url]
                if cached_time >= current_time - expiration:
                    self.logger.debug(f"Cache hit for: {url}")
                    return cached_data
            return None
        except Exception as e:
            self.logger.error(f"Cache read error: {e}")
            self._recreate_cache()
            return None
            
    def set(self, url: str, data: Any):
        """Store data in cache
        
        Args:
            url: URL key
            data: Data to cache
        """
        try:
            self.cache[url] = (time.time(), data)
            self.cache.sync()  # Force write to disk
        except Exception as e:
            self.logger.error(f"Cache write error: {e}")
            self._recreate_cache()
            # Try once more after recreation
            try:
                self.cache[url] = (time.time(), data)
            except Exception as e2:
                raise WeatherError("weather cache write", e2)
                
    def close(self):
        """Close cache properly"""
        if self.cache:
            try:
                self.cache.close()
            except:
                pass

class WeatherClient:
    """Client for fetching weather data from NWS API"""
    
    def __init__(self):
        """Initialize weather client"""
        self.config = get_config()
        self.logger = get_logger(model="Weather")
        
        # Load configuration
        weather_config = self.config.weather
        
        # Get location from top-level or location dict
        if 'latitude' in weather_config and 'longitude' in weather_config:
            # Use top-level coordinates (from system.yaml or validated weather.yaml)
            self.location = {
                'latitude': weather_config['latitude'],
                'longitude': weather_config['longitude'],
                'name': weather_config.get('name', 'Current Location')
            }
        else:
            self.location = weather_config['location']
            
        self.api_config = weather_config['api']
        self.cache_config = weather_config['cache']
        self.weather_moods = weather_config.get('weather_moods', {})
        self.weather_keywords = weather_config.get('weather_keywords', {})
        
        # Initialize cache
        cache_dir = Path(self.cache_config['directory'])
        self.cache = WeatherCache(cache_dir) if self.cache_config['enabled'] else None
        
        # API settings
        self.headers = {
            'User-Agent': self.api_config.get('user_agent', 'AI-Wallpaper/3.0'),
            'Accept': 'application/geo+json'
        }
        self.timeout = self.api_config.get('timeout', 30)
        self.max_retries = self.api_config.get('retries', 3)
        
    def _fetch_url(self, url: str, cache_expiration: Optional[int] = None) -> Dict[str, Any]:
        """Fetch URL with caching and rate limit handling
        
        Args:
            url: URL to fetch
            cache_expiration: Cache expiration in seconds
            
        Returns:
            JSON response data
            
        Raises:
            WeatherError: On API failure
        """
        # Check cache first
        if self.cache and cache_expiration:
            cached_data = self.cache.get(url, cache_expiration)
            if cached_data:
                return cached_data
                
        # Fetch from API
        delay = 60  # Initial rate limit delay
        for attempt in range(self.max_retries):
            try:
                response = requests.get(url, headers=self.headers, timeout=self.timeout)
                
                if response.status_code == 429:  # Too Many Requests
                    if attempt >= self.max_retries - 1:
                        raise WeatherError(
                            self.location['name'],
                            Exception(f"Rate limited after {self.max_retries} attempts")
                        )
                    self.logger.warning(
                        f"Rate limited, waiting {delay}s (attempt {attempt + 1}/{self.max_retries})"
                    )
                    time.sleep(delay)
                    # Double the delay but cap at 5 minutes
                    delay = min(delay * 2, 300)
                    continue
                    
                response.raise_for_status()
                data = response.json()
                
                # Cache successful response
                if self.cache and cache_expiration:
                    self.cache.set(url, data)
                    
                return data
                
            except requests.exceptions.RequestException as e:
                if attempt >= self.max_retries - 1:
                    raise WeatherError(self.location['name'], e)
                self.logger.warning(f"Request failed, retrying: {e}")
                time.sleep(2 ** attempt)  # Exponential backoff
                
        raise WeatherError(
            self.location['name'],
            Exception(f"Failed after {self.max_retries} attempts")
        )
        
    def get_weather_grid(self) -> Dict[str, Any]:
        """Get NWS grid for configured location
        
        Returns:
            Grid data
        """
        lat = self.location['latitude']
        lon = self.location['longitude']
        url = f"{self.api_config['base_url']}/points/{lat},{lon}"
        
        return self._fetch_url(url, self.cache_config['expiration']['grid'])
        
    def get_hourly_forecast(self, grid_data: Dict[str, Any]) -> Dict[str, Any]:
        """Get hourly forecast from grid data
        
        Args:
            grid_data: Grid response data
            
        Returns:
            Hourly forecast data
        """
        if 'properties' not in grid_data or 'forecastHourly' not in grid_data['properties']:
            raise WeatherError(
                self.location['name'],
                Exception("No forecastHourly URL in grid data")
            )
            
        forecast_url = grid_data['properties']['forecastHourly']
        return self._fetch_url(forecast_url, self.cache_config['expiration']['hourly'])
        
    def map_condition_to_mood(self, condition: str) -> str:
        """Map weather condition to creative mood
        
        Args:
            condition: Weather condition text
            
        Returns:
            Mood string
        """
        condition_lower = condition.lower()
        
        # Check mood mappings
        mood_mappings = [
            (['rain', 'shower', 'drizzle'], 'rain'),
            (['storm', 'thunderstorm'], 'thunderstorm'),
            (['snow', 'blizzard', 'flurries', 'sleet', 'ice'], 'snow'),
            (['fog', 'mist', 'haze'], 'fog'),
            (['clear', 'sunny', 'fair'], 'clear_day'),
            (['cloud', 'overcast', 'partly'], 'partly_cloudy'),
            (['wind', 'breezy', 'gusty'], 'windy')
        ]
        
        for keywords, mood_key in mood_mappings:
            if any(word in condition_lower for word in keywords):
                return self.weather_moods.get(mood_key, 'neutral')
                
        return self.weather_moods.get('default', 'neutral')
        
    def get_weather_context(self) -> Dict[str, Any]:
        """Get current weather context
        
        Returns:
            Weather context dictionary
        """
        self.logger.info(f"Fetching weather for {self.location['name']}...")
        
        # Get grid data
        grid = self.get_weather_grid()
        
        # Get hourly forecast
        forecast = self.get_hourly_forecast(grid)
        
        if 'properties' not in forecast or 'periods' not in forecast['properties']:
            raise WeatherError(
                self.location['name'],
                Exception("Invalid forecast data structure")
            )
            
        periods = forecast['properties']['periods']
        if not periods:
            raise WeatherError(
                self.location['name'],
                Exception("No forecast periods available")
            )
            
        # Get current hour data
        current = periods[0]
        
        # Validate required fields
        required = ['shortForecast', 'temperature', 'temperatureUnit', 'windSpeed', 'windDirection']
        missing = [field for field in required if field not in current]
        if missing:
            raise WeatherError(
                self.location['name'],
                Exception(f"Missing required fields: {missing}")
            )
            
        # Build weather context
        condition = current['shortForecast']
        temperature = current['temperature']
        temp_unit = current['temperatureUnit']
        wind_speed = current['windSpeed']
        wind_direction = current['windDirection']
        
        # Determine temperature category
        temp_f = temperature if temp_unit == 'F' else (temperature * 9/5) + 32
        if temp_f < 32:
            temp_category = 'cold'
        elif temp_f < 50:
            temp_category = 'cool'
        elif temp_f < 70:
            temp_category = 'mild'
        elif temp_f < 85:
            temp_category = 'warm'
        else:
            temp_category = 'hot'
            
        context = {
            'condition': condition,
            'temperature': f"{temperature}°{temp_unit}",
            'temperature_value': temperature,
            'temperature_unit': temp_unit,
            'temperature_category': temp_category,
            'wind': f"{wind_speed} {wind_direction}",
            'wind_speed': wind_speed,
            'wind_direction': wind_direction,
            'mood': self.map_condition_to_mood(condition),
            'location': self.location['name'],
            'timestamp': datetime.now().isoformat()
        }
        
        # Add weather keywords for prompt generation
        context['keywords'] = {
            'temperature': self.weather_keywords.get('temperature', {}).get(temp_category, []),
            'conditions': self.weather_keywords.get('conditions', {}).get(
                self.map_condition_to_mood(condition).replace('_', ''), 
                []
            )
        }
        
        self.logger.info(
            f"Weather: {condition}, {temperature}°{temp_unit}, "
            f"Wind {wind_speed} {wind_direction}, Mood: {context['mood']}"
        )
        
        return context

# Global weather client instance
_weather_client: Optional[WeatherClient] = None

def get_weather_context() -> Dict[str, Any]:
    """Get current weather context
    
    Returns:
        Weather context dictionary
    """
    global _weather_client
    if _weather_client is None:
        _weather_client = WeatherClient()
    return _weather_client.get_weather_context()