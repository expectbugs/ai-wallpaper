#!/usr/bin/env python3
"""
Real-ESRGAN Upscaler Integration for AI Wallpaper System
Provides high-quality image upscaling using Real-ESRGAN models
"""

import sys
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from datetime import datetime
from PIL import Image

try:
    import torch
except ImportError:
    torch = None

from ..core import get_logger, get_config
from ..core.exceptions import UpscalerError
from ..core.path_resolver import get_resolver


class RealESRGANUpscaler:
    """Manages Real-ESRGAN upscaling operations"""
    
    def __init__(self):
        """Initialize upscaler"""
        self.logger = get_logger(model="Upscaler")
        self.config = get_config()
        self.realesrgan_path = None
        self._validate_installation()
        
    def _validate_installation(self) -> None:
        """Validate Real-ESRGAN is installed and accessible"""
        self.realesrgan_path = self._find_realesrgan()
        
        if not self.realesrgan_path:
            resolver = get_resolver()
            error_msg = (
                "Real-ESRGAN not found! Cannot proceed with 4K upscaling.\n"
                "Real-ESRGAN is REQUIRED for ultra-high-quality 4K wallpapers.\n"
                "Please install Real-ESRGAN:\n"
                f"  1. cd {resolver.project_root}\n"
                "  2. git clone https://github.com/xinntao/Real-ESRGAN.git\n"
                "  3. cd Real-ESRGAN\n"
                "  4. pip install basicsr facexlib gfpgan\n"
                "  5. pip install -r requirements.txt\n"
                "  6. python setup.py develop\n"
                "  7. wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth -P weights"
            )
            raise UpscalerError("Real-ESRGAN", Exception(error_msg))
            
    def _find_realesrgan(self) -> Optional[Path]:
        """Find Real-ESRGAN installation
        
        Returns:
            Path to Real-ESRGAN script or None
        """
        resolver = get_resolver()
        
        # Check environment variable first
        if env_path := resolver.find_executable('realesrgan-ncnn-vulkan'):
            return env_path
            
        # Get configured paths from config
        realesrgan_paths = self.config.paths.get('models', {}).get('real_esrgan', [])
        
        # Build search paths using resolver
        search_paths = [
            resolver.project_root / "Real-ESRGAN/inference_realesrgan.py",
            Path.home() / "Real-ESRGAN/inference_realesrgan.py",
            Path("/opt/Real-ESRGAN/inference_realesrgan.py"),
            resolver.get_data_dir() / "Real-ESRGAN/inference_realesrgan.py",
        ]
        
        # Also check for the ncnn-vulkan executable
        ncnn_paths = [
            resolver.project_root / "Real-ESRGAN/realesrgan-ncnn-vulkan",
            Path.home() / "Real-ESRGAN/realesrgan-ncnn-vulkan",
            Path("/usr/local/bin/realesrgan-ncnn-vulkan"),
        ]
        
        all_paths = realesrgan_paths + [str(p) for p in search_paths + ncnn_paths]
        
        for path in all_paths:
            path = Path(path).expanduser()
            if path.exists():
                self.logger.info(f"Found Real-ESRGAN at: {path}")
                return path
                
        return None
        
    def upscale(
        self, 
        input_path: Path, 
        scale: int = 4,
        model_name: str = "RealESRGAN_x4plus",
        tile_size: int = 1024,
        fp32: bool = True
    ) -> Dict[str, Any]:
        """Upscale image using Real-ESRGAN
        
        Args:
            input_path: Path to input image
            scale: Upscale factor (default 4)
            model_name: Real-ESRGAN model name
            tile_size: Tile size for processing
            fp32: Use FP32 precision
            
        Returns:
            Dictionary with upscaling results
            
        Raises:
            UpscalerError: If upscaling fails
        """
        self.logger.info(f"Upscaling {input_path} by {scale}x using {model_name}")
        
        # Verify input exists
        if not input_path.exists():
            raise UpscalerError(str(input_path), FileNotFoundError("Input file not found"))
            
        # Get input dimensions
        with Image.open(input_path) as input_image:
            input_size = input_image.size
        
        # Prepare unique output path with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        output_dir = input_path.parent / f"upscaled_{timestamp}"
        output_dir.mkdir(exist_ok=True)
        
        # Build command based on script type
        if str(self.realesrgan_path).endswith('.py'):
            # Python script
            cmd = [
                sys.executable,
                str(self.realesrgan_path),
                "-n", model_name,
                "-i", str(input_path),
                "-o", str(output_dir),
                "--outscale", str(scale),
                "-t", str(tile_size)
            ]
            
            if fp32:
                cmd.append("--fp32")
                
        else:
            # Binary executable
            cmd = [
                str(self.realesrgan_path),
                "-i", str(input_path),
                "-o", str(output_dir),
                "-s", str(scale),
                "-n", model_name.lower().replace('_', '-'),
                "-t", str(tile_size)
            ]
            
        self.logger.debug(f"Executing: {' '.join(cmd)}")
        
        try:
            # Run upscaling
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                timeout=300  # 5 minute timeout
            )
            
            if result.stdout:
                self.logger.debug(f"Real-ESRGAN output: {result.stdout}")
                
        except subprocess.CalledProcessError as e:
            error_msg = f"Command failed: {e.stderr if e.stderr else 'Unknown error'}"
            raise UpscalerError(str(input_path), Exception(error_msg))
        except subprocess.TimeoutExpired:
            raise UpscalerError(str(input_path), Exception("Upscaling timed out after 5 minutes"))
            
        # Find output file
        output_files = list(output_dir.glob("*.png"))
        if not output_files:
            raise UpscalerError(str(input_path), FileNotFoundError("No output from Real-ESRGAN"))
            
        output_path = output_files[0]
        
        # Verify output
        with Image.open(output_path) as output_image:
            output_size = output_image.size
        
        expected_size = (input_size[0] * scale, input_size[1] * scale)
        
        if output_size != expected_size:
            self.logger.warning(
                f"Unexpected output size: {output_size}, expected {expected_size}"
            )
            
        self.logger.info(f"Upscaling complete: {input_size} → {output_size}")
        
        return {
            'output_path': output_path,
            'input_size': input_size,
            'output_size': output_size,
            'scale_factor': scale,
            'model': model_name
        }
        
    def upscale_to_target(
        self,
        input_path: Path,
        target_width: int,
        target_height: int,
        model_name: str = "RealESRGAN_x4plus"
    ) -> Dict[str, Any]:
        """Upscale image to at least target dimensions
        
        Args:
            input_path: Path to input image
            target_width: Target width
            target_height: Target height
            model_name: Real-ESRGAN model name
            
        Returns:
            Upscaling results
        """
        # Get input dimensions
        with Image.open(input_path) as input_image:
            input_width, input_height = input_image.size
        
        # Calculate required scale
        scale_w = (target_width + input_width - 1) // input_width  # Ceiling division
        scale_h = (target_height + input_height - 1) // input_height
        scale = max(scale_w, scale_h, 2)  # At least 2x
        
        # Limit scale to reasonable values
        if scale > 8:
            self.logger.warning(f"Limiting scale from {scale}x to 8x")
            scale = 8
            
        self.logger.info(
            f"Upscaling from {input_width}x{input_height} to "
            f"~{input_width * scale}x{input_height * scale} "
            f"(target: {target_width}x{target_height})"
        )
        
        return self.upscale(input_path, scale, model_name)
        
    def check_models(self) -> Dict[str, bool]:
        """Check which Real-ESRGAN models are available
        
        Returns:
            Dictionary of model availability
        """
        models = {
            "RealESRGAN_x4plus": False,
            "RealESRGAN_x4plus_anime_6B": False,
            "RealESRGAN_x2plus": False,
            "RealESRNet_x4plus": False
        }
        
        if not self.realesrgan_path:
            return models
            
        # Check weights directory
        weights_dir = self.realesrgan_path.parent / "weights"
        if weights_dir.exists():
            for model_name in models:
                weight_file = weights_dir / f"{model_name}.pth"
                if weight_file.exists():
                    models[model_name] = True
                    
        return models
        
    def get_optimal_settings(self, image_size: Tuple[int, int]) -> Dict[str, Any]:
        """Get optimal settings based on image size and available VRAM
        
        Args:
            image_size: Input image dimensions
            
        Returns:
            Optimal settings dictionary
        """
        width, height = image_size
        pixels = width * height
        
        # Check available VRAM if torch is available
        available_vram_gb = 0
        if torch and torch.cuda.is_available():
            try:
                # Get free VRAM in GB
                free_vram = torch.cuda.mem_get_info()[0]
                available_vram_gb = free_vram / (1024**3)
                self.logger.debug(f"Available VRAM: {available_vram_gb:.1f}GB")
            except Exception as e:
                # FAIL LOUD - VRAM info is critical for proper tile sizing
                raise RuntimeError(
                    f"Failed to get VRAM information: {e}\n"
                    f"Cannot determine optimal tile size without VRAM info!\n"
                    f"This should not happen with properly configured CUDA."
                )
                
        # Determine tile size based on both image size and VRAM
        # Larger tiles are faster but use more VRAM
        if available_vram_gb > 8:  # High VRAM (>8GB)
            # Can use larger tiles
            if pixels > 4000000:  # > 4MP
                tile_size = 768
            elif pixels > 2000000:  # > 2MP
                tile_size = 1024
            else:
                tile_size = 1536
        elif available_vram_gb > 4:  # Medium VRAM (4-8GB)
            # Use moderate tiles
            if pixels > 4000000:  # > 4MP
                tile_size = 512
            elif pixels > 2000000:  # > 2MP
                tile_size = 768
            else:
                tile_size = 1024
        else:  # Low VRAM (<4GB) or CPU
            # Use smaller tiles to avoid OOM
            if pixels > 4000000:  # > 4MP
                tile_size = 256
            elif pixels > 2000000:  # > 2MP
                tile_size = 384
            else:
                tile_size = 512
                
        # Use FP32 for quality if we have enough VRAM, FP16 otherwise
        use_fp32 = available_vram_gb > 6
        
        settings = {
            'tile_size': tile_size,
            'fp32': use_fp32,
            'model': 'RealESRGAN_x4plus'  # Best general-purpose model
        }
        
        self.logger.debug(
            f"Optimal settings for {width}x{height} with {available_vram_gb:.1f}GB VRAM: "
            f"tile_size={tile_size}, fp32={use_fp32}"
        )
        
        return settings


# Global instance
_upscaler: Optional[RealESRGANUpscaler] = None


def get_upscaler() -> RealESRGANUpscaler:
    """Get global upscaler instance
    
    Returns:
        RealESRGANUpscaler instance
    """
    global _upscaler
    if _upscaler is None:
        _upscaler = RealESRGANUpscaler()
    return _upscaler


def upscale_image(
    input_path: Path,
    scale: int = 4,
    model_name: str = "RealESRGAN_x4plus"
) -> Path:
    """Convenience function to upscale an image
    
    Args:
        input_path: Path to input image
        scale: Upscale factor
        model_name: Model to use
        
    Returns:
        Path to upscaled image
    """
    upscaler = get_upscaler()
    result = upscaler.upscale(input_path, scale, model_name)
    return result['output_path']