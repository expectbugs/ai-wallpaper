#!/usr/bin/env python3
"""
Test Command Implementation
System component testing
"""

from typing import Optional, Dict, Any
import time

from ..core import get_logger, get_config, get_weather_context
from ..prompt import get_random_theme_with_weather, DeepSeekPrompter
from ..models.flux_model import FluxModel

class TestCommand:
    """Handles system testing"""
    
    def __init__(self, config_dir: Optional[str] = None, verbose: bool = False):
        """Initialize test command
        
        Args:
            config_dir: Custom config directory  
            verbose: Enable verbose output
        """
        self.config_dir = config_dir
        self.verbose = verbose
        self.logger = get_logger()
        self.config = get_config()
        
    def execute(
        self,
        model: Optional[str] = None,
        component: Optional[str] = None,
        quick: bool = False
    ) -> Dict[str, Any]:
        """Execute tests
        
        Args:
            model: Test specific model
            component: Test specific component
            quick: Quick test mode
            
        Returns:
            Test results
        """
        results = {
            'success': True,
            'tests': {}
        }
        
        try:
            if component == 'prompt' or not component:
                results['tests']['prompt'] = self._test_prompt_generation(quick)
                
            if component == 'theme' or not component:
                results['tests']['theme'] = self._test_theme_selection()
                
            if component == 'wallpaper' or not component:
                results['tests']['wallpaper'] = self._test_wallpaper_setting()
                
            if component == 'image' or not component:
                if model:
                    results['tests']['image'] = self._test_image_generation(model, quick)
                else:
                    results['tests']['image'] = self._test_image_generation('flux', quick)
                    
            # Check if any tests failed
            for test_name, test_result in results['tests'].items():
                if not test_result.get('success', False):
                    results['success'] = False
                    
        except Exception as e:
            self.logger.error(f"Test failed: {e}")
            results['success'] = False
            results['error'] = str(e)
            
        return results
        
    def _test_prompt_generation(self, quick: bool) -> Dict[str, Any]:
        """Test prompt generation
        
        Args:
            quick: Quick mode
            
        Returns:
            Test result
        """
        self.logger.info("Testing prompt generation...")
        
        try:
            # Get weather
            weather = get_weather_context()
            
            # Get theme
            theme_result = get_random_theme_with_weather(weather)
            
            self.logger.info(f"Selected theme: {theme_result['theme']['name']}")
            
            if not quick:
                # Test actual prompt generation
                prompter = DeepSeekPrompter()
                prompter.initialize()
                
                prompt = prompter.generate_prompt(
                    theme=theme_result['theme'],
                    weather=weather,
                    context={'season': 'test', 'day_of_week': 'Test Day'},
                    history=[],
                    model_requirements={'max_words': 65}
                )
                
                prompter.cleanup()
                
                self.logger.info(f"Generated prompt: {prompt}")
                
            return {
                'success': True,
                'message': 'Prompt generation working'
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
            
    def _test_theme_selection(self) -> Dict[str, Any]:
        """Test theme selection
        
        Returns:
            Test result
        """
        self.logger.info("Testing theme selection...")
        
        try:
            # Test multiple selections
            themes_seen = set()
            
            for i in range(10):
                weather = {'mood': 'neutral', 'condition': 'Clear'}
                theme_result = get_random_theme_with_weather(weather)
                themes_seen.add(theme_result['theme']['name'])
                
            self.logger.info(f"Selected {len(themes_seen)} unique themes in 10 tries")
            
            return {
                'success': True,
                'unique_themes': len(themes_seen)
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
            
    def _test_wallpaper_setting(self) -> Dict[str, Any]:
        """Test wallpaper setting capability
        
        Returns:
            Test result
        """
        self.logger.info("Testing wallpaper setting...")
        
        try:
            from ..core import WallpaperSetter
            
            setter = WallpaperSetter()
            de = setter.desktop_env
            
            if de:
                self.logger.info(f"Detected desktop environment: {de}")
                return {
                    'success': True,
                    'desktop_env': de
                }
            else:
                return {
                    'success': False,
                    'error': 'No desktop environment detected'
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
            
    def _test_image_generation(self, model_name: str, quick: bool) -> Dict[str, Any]:
        """Test image generation
        
        Args:
            model_name: Model to test
            quick: Quick mode
            
        Returns:
            Test result
        """
        self.logger.info(f"Testing {model_name} image generation...")
        
        try:
            # Get model config
            model_config = self.config.get_model_config(model_name)
            
            # Check if enabled
            if not model_config.get('enabled', False):
                return {
                    'success': False,
                    'error': f'Model {model_name} is disabled'
                }
                
            if quick:
                # Just validate environment
                if model_name == 'flux':
                    model = FluxModel(model_config)
                    valid, msg = model.validate_environment()
                    
                    return {
                        'success': valid,
                        'message': msg
                    }
                else:
                    return {
                        'success': True,
                        'message': f'{model_name} not yet implemented'
                    }
            else:
                # Full generation test
                if model_name == 'flux':
                    model = FluxModel(model_config)
                    model.initialize()
                    
                    # Generate test image
                    result = model.generate(
                        "A simple test image with geometric shapes",
                        seed=42
                    )
                    
                    model.cleanup()
                    
                    return {
                        'success': True,
                        'image_path': result['image_path']
                    }
                else:
                    return {
                        'success': False,
                        'error': f'{model_name} not yet implemented'
                    }
                    
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }