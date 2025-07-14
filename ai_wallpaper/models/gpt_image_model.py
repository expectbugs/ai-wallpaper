#!/usr/bin/env python3
"""
GPT-Image-1 Model Implementation
Generates images using OpenAI's GPT-Image-1 API with two variants:
1. Direct API (gpt-image-1 model)
2. Responses API (gpt-4o with image_generation tool)
"""

import os
import sys
import time
import base64
import random
import requests
import subprocess
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List
from datetime import datetime
from PIL import Image

from .base_model import BaseImageModel
from ..core import get_logger, get_config
from ..core.exceptions import ModelError, GenerationError, UpscalerError, APIError

class GptImageModel(BaseImageModel):
    """GPT-Image-1 implementation with two API variants"""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize GPT-Image-1 model
        
        Args:
            config: Model configuration from models.yaml
        """
        super().__init__(config)
        self.api_key = os.environ.get('OPENAI_API_KEY')
        
        # Determine which variant to use
        self.use_responses_api = self._should_use_responses_api()
        
        if self.use_responses_api:
            self.logger.info("Using Responses API variant (gpt-4o)")
        else:
            self.logger.info("Using Direct API variant (gpt-image-1)")
            
    def _should_use_responses_api(self) -> bool:
        """Determine which API variant to use
        
        Returns:
            True to use Responses API, False for Direct API
        """
        # Check environment variable override
        variant = os.environ.get('GPT_IMAGE_VARIANT', '').lower()
        if variant == 'responses':
            return True
        elif variant == 'direct':
            return False
            
        # Default to Responses API (gpt-4o) as it's more reliable
        return True
        
    def initialize(self) -> bool:
        """Initialize GPT-Image-1 model and verify requirements
        
        Returns:
            True if initialization successful
        """
        try:
            self.logger.info("Initializing GPT-Image-1 model...")
            
            # Check API key
            if not self.api_key:
                raise ModelError(
                    "GPT-Image-1 requires OPENAI_API_KEY environment variable.\n"
                    "Please set: export OPENAI_API_KEY='your-api-key-here'"
                )
                
            # Verify Real-ESRGAN is available
            self._find_realesrgan()
            
            # Initialize OpenAI client if using Responses API
            if self.use_responses_api:
                try:
                    from openai import OpenAI
                    self.openai_client = OpenAI(api_key=self.api_key)
                    self.logger.debug("OpenAI client initialized for Responses API")
                except ImportError:
                    raise ModelError(
                        "OpenAI Python package required for Responses API.\n"
                        "Please install: pip install openai>=1.0"
                    )
                    
            self._initialized = True
            self.logger.info("GPT-Image-1 model initialized successfully")
            
            return True
            
        except Exception as e:
            raise ModelLoadError(self.name, e)
            
    def generate(self, prompt: str, seed: Optional[int] = None, **kwargs) -> Dict[str, Any]:
        """Generate image using GPT-Image-1 pipeline
        
        Args:
            prompt: Text prompt
            seed: Random seed (not used by API)
            **kwargs: Additional parameters
            
        Returns:
            Generation results dictionary
        """
        self.ensure_initialized()
        
        # Get generation parameters
        params = self.get_generation_params(**kwargs)
        
        # Use provided seed for filename generation
        if seed is None:
            seed = random.randint(0, 2**32 - 1)
            
        # Log generation start
        self.log_generation_start(prompt, {**params, 'seed': seed})
        
        # Track timing
        start_time = time.time()
        
        try:
            # Stage 1: Generate via API
            if self.use_responses_api:
                stage1_result = self._generate_responses_api(prompt, params)
            else:
                stage1_result = self._generate_direct_api(prompt, params)
                
            # Stage 2: Crop to 16:9
            stage2_result = self._crop_stage2(stage1_result['image_path'])
            
            # Stage 3: Upscale 4x
            stage3_result = self._upscale_stage3(stage2_result['image_path'])
            
            # Stage 4: Downsample to 4K
            stage4_result = self._downsample_stage4(stage3_result['image_path'])
            
            # Prepare final results
            duration = time.time() - start_time
            
            results = {
                'image_path': stage4_result['image_path'],
                'metadata': {
                    'prompt': prompt,
                    'seed': seed,
                    'model': 'GPT-Image-1',
                    'variant': 'responses' if self.use_responses_api else 'direct',
                    'parameters': params,
                    'duration': duration,
                    'timestamp': datetime.now().isoformat()
                },
                'stages': {
                    'stage1_generation': stage1_result,
                    'stage2_crop': stage2_result,
                    'stage3_upscale': stage3_result,
                    'stage4_downsample': stage4_result
                }
            }
            
            # Save metadata
            self.save_metadata(Path(results['image_path']), results['metadata'])
            
            # Log completion
            self.log_generation_complete(Path(results['image_path']), duration)
            
            return results
            
        except Exception as e:
            raise GenerationError(self.name, "pipeline execution", e)
            
    def _generate_responses_api(self, prompt: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Generate using Responses API (gpt-4o)
        
        Args:
            prompt: Text prompt (full description)
            params: Generation parameters
            
        Returns:
            Stage results
        """
        self.logger.log_stage("Stage 1", "GPT-4o Responses API generation")
        
        # Get dimensions from config
        width, height = self.config['generation']['dimensions']
        
        # Create comprehensive description
        image_description = prompt  # Prompt is already comprehensive from theme system
        
        self.logger.info(f"Calling GPT-4o with size {width}x{height}, quality 'high'")
        
        # Retry logic
        max_attempts = 3
        response = None
        
        for attempt in range(max_attempts):
            try:
                response = self.openai_client.responses.create(
                    model="gpt-4o",
                    input=image_description,
                    tools=[{
                        "type": "image_generation",
                        "size": f"{width}x{height}",
                        "quality": params.get('quality', 'high')
                    }]
                )
                self.logger.info("API call successful")
                break
                
            except Exception as e:
                self.logger.warning(f"API call attempt {attempt + 1} failed: {e}")
                if attempt == max_attempts - 1:
                    raise APIError("GPT-4o", "responses API", None, str(e))
                wait_time = 10 * (attempt + 1)
                self.logger.info(f"Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
                
        # Extract image data
        try:
            image_generation_calls = [
                output for output in response.output
                if output.type == "image_generation_call"
            ]
            
            if not image_generation_calls:
                raise APIError(
                    "GPT-4o", 
                    "responses API", 
                    None, 
                    "No image generation call found in response"
                )
                
            image_call = image_generation_calls[0]
            
            if hasattr(image_call, 'revised_prompt') and image_call.revised_prompt:
                self.logger.info(f"GPT-4o revised description: {image_call.revised_prompt[:100]}...")
                
            if image_call.status != "completed":
                raise APIError(
                    "GPT-4o",
                    "responses API",
                    None,
                    f"Image generation failed with status: {image_call.status}"
                )
                
            image_base64 = image_call.result
            self.logger.info("Successfully extracted base64 image data")
            
        except Exception as e:
            if not isinstance(e, APIError):
                raise APIError("GPT-4o", "responses API", None, f"Failed to parse response: {e}")
            raise
            
        # Decode and save image
        try:
            image_bytes = base64.b64decode(image_base64)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            stage1_path = Path(f"/tmp/gpt_stage1_{timestamp}.png")
            
            with open(stage1_path, 'wb') as f:
                f.write(image_bytes)
                
            self.logger.info(f"Image decoded and saved to: {stage1_path}")
            
        except Exception as e:
            raise GenerationError(self.name, "Stage 1", e)
            
        # Verify image
        image = Image.open(stage1_path)
        self.logger.info(f"Stage 1 complete: Generated at {image.size}")
        
        return {
            'image': image,
            'image_path': stage1_path,
            'size': image.size,
            'api_variant': 'responses'
        }
        
    def _generate_direct_api(self, prompt: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Generate using Direct API (gpt-image-1)
        
        Args:
            prompt: Text prompt
            params: Generation parameters
            
        Returns:
            Stage results
        """
        self.logger.log_stage("Stage 1", "GPT-Image-1 Direct API generation")
        
        # Get configuration
        direct_config = self.config['variants']['direct_api']
        endpoint = direct_config['endpoint']
        
        # Prepare API request
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # Get dimensions from config
        width, height = self.config['generation']['dimensions']
        
        data = {
            "model": "gpt-image-1",
            "prompt": prompt,
            "n": 1,
            "size": f"{width}x{height}",
            "quality": params.get('quality', 'high')
        }
        
        self.logger.info(f"Calling GPT-Image-1 API with size {width}x{height}")
        
        # Make request with retry logic
        max_attempts = 3
        response = None
        
        for attempt in range(max_attempts):
            try:
                response = requests.post(
                    endpoint,
                    headers=headers,
                    json=data,
                    timeout=direct_config.get('timeout', 180)
                )
                
                if response.status_code == 200:
                    break
                elif response.status_code == 403:
                    error_msg = (
                        "Organization verification required for GPT-Image-1.\n"
                        "Please verify your organization in the OpenAI console.\n"
                        "Visit: https://platform.openai.com/settings/organization"
                    )
                    raise APIError("GPT-Image-1", endpoint, 403, error_msg)
                elif response.status_code == 429:
                    wait_time = 30 * (attempt + 1)
                    self.logger.warning(f"Rate limited, waiting {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    error_msg = f"API error: {response.status_code} - {response.text}"
                    if attempt == max_attempts - 1:
                        raise APIError("GPT-Image-1", endpoint, response.status_code, error_msg)
                    self.logger.warning(error_msg)
                    
            except requests.exceptions.Timeout:
                self.logger.warning(f"Request timeout (attempt {attempt + 1})")
                self.logger.info("GPT-Image-1 generation can take several minutes")
                if attempt == max_attempts - 1:
                    raise APIError("GPT-Image-1", endpoint, None, "Request timeout")
                time.sleep(30 * (attempt + 1))
                
            except requests.exceptions.RequestException as e:
                if attempt == max_attempts - 1:
                    raise APIError("GPT-Image-1", endpoint, None, str(e))
                time.sleep(10)
                
        # Parse response
        try:
            result = response.json()
            
            # Extract image (handles multiple response formats)
            image_data = result['data'][0]
            
            if 'b64_json' in image_data:
                image_base64 = image_data['b64_json']
                image_bytes = base64.b64decode(image_base64)
            elif 'url' in image_data:
                # URL format
                image_url = image_data['url']
                self.logger.info(f"Downloading image from URL...")
                image_response = requests.get(image_url, timeout=30)
                if image_response.status_code != 200:
                    raise APIError("GPT-Image-1", image_url, image_response.status_code, "Failed to download")
                image_bytes = image_response.content
            else:
                raise APIError("GPT-Image-1", endpoint, None, f"Unknown response format: {list(image_data.keys())}")
                
        except Exception as e:
            if not isinstance(e, APIError):
                raise APIError("GPT-Image-1", endpoint, response.status_code, f"Failed to parse response: {e}")
            raise
            
        # Save image
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            stage1_path = Path(f"/tmp/gpt_stage1_{timestamp}.png")
            
            with open(stage1_path, 'wb') as f:
                f.write(image_bytes)
                
            self.logger.info(f"Image saved to: {stage1_path}")
            
        except Exception as e:
            raise GenerationError(self.name, "Stage 1", e)
            
        # Verify image
        image = Image.open(stage1_path)
        self.logger.info(f"Stage 1 complete: Generated at {image.size}")
        
        return {
            'image': image,
            'image_path': stage1_path,
            'size': image.size,
            'api_variant': 'direct'
        }
        
    def _crop_stage2(self, input_path: Path) -> Dict[str, Any]:
        """Stage 2: Crop to 16:9 ratio
        
        Args:
            input_path: Path to input image
            
        Returns:
            Stage results
        """
        self.logger.log_stage("Stage 2", "Cropping to 16:9 aspect ratio")
        
        # Load image
        image = Image.open(input_path)
        original_size = image.size
        
        # Get crop settings from config
        crop_pixels = self.config['pipeline'].get('crop_pixels', [80, 80])
        
        # Crop height (1536x1024 -> 1536x864)
        new_height = original_size[1] - sum(crop_pixels)
        image_cropped = image.crop((0, crop_pixels[0], original_size[0], original_size[1] - crop_pixels[1]))
        
        self.logger.info(f"Cropped from {original_size} to {image_cropped.size}")
        
        # Verify 16:9 ratio
        ratio = image_cropped.size[0] / image_cropped.size[1]
        expected_ratio = 16 / 9
        self.logger.info(f"Aspect ratio: {ratio:.4f} (expected: {expected_ratio:.4f})")
        
        # Save cropped image
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        stage2_path = Path(f"/tmp/gpt_stage2_{timestamp}.png")
        image_cropped.save(stage2_path, "PNG", quality=100)
        
        self.logger.info(f"Stage 2 complete: Cropped to {image_cropped.size}")
        
        return {
            'image': image_cropped,
            'image_path': stage2_path,
            'size': image_cropped.size,
            'crop_pixels': crop_pixels
        }
        
    def _upscale_stage3(self, input_path: Path) -> Dict[str, Any]:
        """Stage 3: Upscale 4x using Real-ESRGAN
        
        Args:
            input_path: Path to cropped image
            
        Returns:
            Stage results
        """
        self.logger.log_stage("Stage 3", "Real-ESRGAN 4x upscaling")
        
        # Find Real-ESRGAN
        realesrgan_script = self._find_realesrgan()
        
        # Prepare paths
        temp_output_dir = Path(f"/tmp/gpt_upscaled_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        temp_output_dir.mkdir(exist_ok=True)
        
        # Build command
        if str(realesrgan_script).endswith('.py'):
            cmd = [
                sys.executable,
                str(realesrgan_script),
                "-n", "RealESRGAN_x4plus",
                "-i", str(input_path),
                "-o", str(temp_output_dir),
                "--outscale", "4",
                "-t", "1024",
                "--fp32"
            ]
        else:
            cmd = [
                str(realesrgan_script),
                "-i", str(input_path),
                "-o", str(temp_output_dir),
                "-s", "4",
                "-n", "realesrgan-x4plus",
                "-t", "1024"
            ]
            
        self.logger.debug(f"Executing: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                timeout=300
            )
            
            if result.stdout:
                self.logger.debug(f"Real-ESRGAN output: {result.stdout}")
                
        except subprocess.CalledProcessError as e:
            raise UpscalerError(str(input_path), e)
        except subprocess.TimeoutExpired:
            raise UpscalerError(str(input_path), Exception("Upscaling timed out"))
            
        # Find output file
        output_files = list(temp_output_dir.glob("*.png"))
        if not output_files:
            raise UpscalerError(str(input_path), FileNotFoundError("No output from Real-ESRGAN"))
            
        output_path = output_files[0]
        
        # Load and verify
        upscaled_image = Image.open(output_path)
        
        # Expected: 1536x864 * 4 = 6144x3456
        expected_size = (
            Image.open(input_path).size[0] * 4,
            Image.open(input_path).size[1] * 4
        )
        
        if upscaled_image.size != expected_size:
            self.logger.warning(f"Unexpected size: {upscaled_image.size}, expected {expected_size}")
            
        self.logger.info(f"Stage 3 complete: Upscaled to {upscaled_image.size}")
        
        return {
            'image': upscaled_image,
            'image_path': output_path,
            'size': upscaled_image.size,
            'scale_factor': 4
        }
        
    def _downsample_stage4(self, input_path: Path) -> Dict[str, Any]:
        """Stage 4: Downsample to 4K using Lanczos
        
        Args:
            input_path: Path to upscaled image
            
        Returns:
            Stage results
        """
        self.logger.log_stage("Stage 4", "Lanczos downsampling to 4K")
        
        # Load upscaled image
        image_upscaled = Image.open(input_path)
        
        # Target 4K dimensions
        target_size = tuple(self.config['pipeline']['final_resolution'])
        
        # High-quality downsample
        image_4k = image_upscaled.resize(target_size, Image.Resampling.LANCZOS)
        
        # Save final image
        config = get_config()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"gpt_4k_{timestamp}.png"
        
        # Use configured output path
        output_dir = Path(config.paths.get('images_dir', '/tmp'))
        output_dir.mkdir(exist_ok=True)
        
        final_path = output_dir / filename
        image_4k.save(final_path, "PNG", quality=100)
        
        self.logger.info(f"Stage 4 complete: Final 4K image at {image_4k.size}")
        self.logger.info(f"Saved to: {final_path}")
        
        # Clean up temp files
        try:
            if input_path.parent.name.startswith("gpt_upscaled_"):
                import shutil
                shutil.rmtree(input_path.parent)
        except Exception as e:
            self.logger.debug(f"Failed to clean up temp files: {e}")
            
        return {
            'image': image_4k,
            'image_path': final_path,
            'size': image_4k.size
        }
        
    def _find_realesrgan(self) -> Path:
        """Find Real-ESRGAN installation
        
        Returns:
            Path to Real-ESRGAN script
            
        Raises:
            UpscalerError: If not found
        """
        config = get_config()
        
        # Get configured paths
        realesrgan_paths = config.paths.get('models', {}).get('real_esrgan', [])
        
        # Add common locations
        common_paths = [
            "/home/user/ai-wallpaper/Real-ESRGAN/inference_realesrgan.py",
            "/home/user/Real-ESRGAN/inference_realesrgan.py",
            Path.home() / "Real-ESRGAN/inference_realesrgan.py",
            "/usr/local/bin/realesrgan-ncnn-vulkan"
        ]
        
        all_paths = realesrgan_paths + [str(p) for p in common_paths]
        
        for path in all_paths:
            path = Path(path).expanduser()
            if path.exists():
                self.logger.info(f"Found Real-ESRGAN at: {path}")
                return path
                
        # Not found
        error_msg = (
            "Real-ESRGAN not found! Cannot proceed with 4K upscaling.\n"
            "Real-ESRGAN is REQUIRED for ultra-high-quality 4K wallpapers.\n"
            "Please install Real-ESRGAN:\n"
            "  1. cd /home/user/ai-wallpaper\n"
            "  2. git clone https://github.com/xinntao/Real-ESRGAN.git\n"
            "  3. cd Real-ESRGAN\n"
            "  4. pip install basicsr facexlib gfpgan\n"
            "  5. pip install -r requirements.txt\n"
            "  6. python setup.py develop\n"
            "  7. wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth -P weights"
        )
        
        raise UpscalerError("Real-ESRGAN", Exception(error_msg))
        
    def get_optimal_prompt(self, theme: Dict, weather: Dict, context: Dict) -> str:
        """Get GPT-optimized prompt
        
        Args:
            theme: Theme dictionary
            weather: Weather context
            context: Additional context
            
        Returns:
            Optimized prompt for GPT
        """
        # GPT models work well with descriptive prompts
        return ""  # Will be implemented with prompt generation
        
    def get_pipeline_stages(self) -> List[str]:
        """Return pipeline stages for GPT-Image-1
        
        Returns:
            List of stage names
        """
        return [
            "gpt_generation",      # 1536x1024
            "crop_to_16_9",       # 1536x864
            "realesrgan_4x",      # 6144x3456
            "lanczos_4k"         # 3840x2160
        ]
        
    def validate_environment(self) -> Tuple[bool, str]:
        """Validate GPT-Image-1 can run in current environment
        
        Returns:
            Tuple of (is_valid, message)
        """
        # Check API key
        if not os.environ.get('OPENAI_API_KEY'):
            return False, "OPENAI_API_KEY environment variable not set"
            
        # Check OpenAI package if using Responses API
        if self.use_responses_api:
            try:
                import openai
            except ImportError:
                return False, "OpenAI Python package required: pip install openai>=1.0"
                
        # Check Real-ESRGAN availability
        try:
            self._find_realesrgan()
        except UpscalerError:
            return False, "Real-ESRGAN is required but not found"
            
        return True, "Environment validated for GPT-Image-1"
        
    def supports_feature(self, feature: str) -> bool:
        """Check if GPT-Image-1 supports a feature
        
        Args:
            feature: Feature name
            
        Returns:
            True if supported
        """
        features = {
            '8k_pipeline': False,
            'scheduler_selection': False,
            'custom_dimensions': False,
            'lora': False,
            'img2img': False,
            'controlnet': False
        }
        return features.get(feature, super().supports_feature(feature))
        
    def get_resource_requirements(self) -> Dict[str, Any]:
        """Return GPT-Image-1 resource requirements
        
        Returns:
            Resource requirements
        """
        return {
            'vram_gb': 8,   # For Real-ESRGAN only
            'disk_gb': 2,   # Minimal disk usage
            'time_minutes': 3  # API can be slower than DALL-E
        }