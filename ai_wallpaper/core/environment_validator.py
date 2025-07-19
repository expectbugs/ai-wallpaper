"""
Environment validation to ensure system is properly configured.
Checks for required dependencies, disk space, and system capabilities.
"""
import os
import sys
import shutil
import platform
import subprocess
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from .path_resolver import get_resolver


logger = logging.getLogger(__name__)


class EnvironmentValidator:
    """Validate system environment for AI wallpaper generation."""
    
    def __init__(self):
        self.resolver = get_resolver()
        self.issues = []
        self.warnings = []
        
    def validate(self) -> Tuple[List[str], List[str]]:
        """
        Validate environment and return issues and warnings.
        
        Returns:
            Tuple of (critical_issues, warnings)
        """
        self.issues = []
        self.warnings = []
        
        # Run all validation checks
        self._check_python_version()
        self._check_gpu()
        self._check_disk_space()
        self._check_required_executables()
        self._check_permissions()
        self._check_network()
        self._check_dependencies()
        
        return self.issues, self.warnings
        
    def _check_python_version(self) -> None:
        """Check Python version compatibility."""
        if sys.version_info < (3, 8):
            self.issues.append(
                f"Python {sys.version.split()[0]} is too old. AI Wallpaper requires Python 3.8 or higher."
            )
            
    def _check_gpu(self) -> None:
        """Check GPU availability and CUDA support."""
        try:
            import torch
            
            if not torch.cuda.is_available():
                self.warnings.append(
                    "No CUDA GPU detected. Image generation will be significantly slower on CPU."
                )
            else:
                # Check GPU memory
                gpu_count = torch.cuda.device_count()
                for i in range(gpu_count):
                    props = torch.cuda.get_device_properties(i)
                    memory_gb = props.total_memory / (1024**3)
                    
                    if memory_gb < 8:
                        self.warnings.append(
                            f"GPU {i} ({props.name}) has only {memory_gb:.1f}GB VRAM. "
                            "Some models may require 8GB+ for optimal performance."
                        )
                        
                    logger.info(f"GPU {i}: {props.name} with {memory_gb:.1f}GB VRAM")
                    
        except ImportError:
            self.issues.append(
                "PyTorch not installed. Run: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118"
            )
            
    def _check_disk_space(self) -> None:
        """Check available disk space in critical directories."""
        min_space_gb = 30  # Minimum required space
        
        dirs_to_check = [
            (self.resolver.get_cache_dir(), "cache", 50),  # Models need more space
            (self.resolver.get_data_dir(), "data", 20),
            (self.resolver.get_temp_dir(), "temp", 10),
        ]
        
        for directory, name, recommended_gb in dirs_to_check:
            try:
                # Ensure directory exists before checking
                directory.mkdir(parents=True, exist_ok=True)
                
                # Get disk usage
                stat = shutil.disk_usage(directory)
                free_gb = stat.free / (1024**3)
                
                if free_gb < min_space_gb:
                    self.issues.append(
                        f"Low disk space in {name} directory ({directory}): "
                        f"only {free_gb:.1f}GB free, need at least {min_space_gb}GB"
                    )
                elif free_gb < recommended_gb:
                    self.warnings.append(
                        f"Limited disk space in {name} directory ({directory}): "
                        f"{free_gb:.1f}GB free, recommend {recommended_gb}GB+"
                    )
                    
            except Exception as e:
                self.warnings.append(f"Could not check disk space for {directory}: {e}")
                
    def _check_required_executables(self) -> None:
        """Check for required external executables."""
        # Check Ollama
        if not self.resolver.find_executable('ollama'):
            self.issues.append(
                "Ollama not found. Install from https://ollama.ai or set AI_WALLPAPER_OLLAMA_PATH"
            )
        else:
            # Check if Ollama is running and has required model
            try:
                result = subprocess.run(
                    ['ollama', 'list'],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if result.returncode != 0:
                    self.warnings.append(
                        "Ollama is installed but not running. Start it with: ollama serve"
                    )
                elif 'deepseek-r1:14b' not in result.stdout:
                    self.warnings.append(
                        "DeepSeek R1:14b model not found. Install with: ollama pull deepseek-r1:14b"
                    )
            except Exception as e:
                self.warnings.append(f"Could not check Ollama status: {e}")
                
        # Check Real-ESRGAN (optional but recommended)
        realesrgan_paths = [
            str(self.resolver.project_root / 'Real-ESRGAN'),
            str(Path.home() / 'Real-ESRGAN'),
            '/opt/Real-ESRGAN',
        ]
        
        found_realesrgan = False
        for path in realesrgan_paths:
            exe_name = 'realesrgan-ncnn-vulkan'
            if platform.system() == 'Windows':
                exe_name += '.exe'
                
            if (Path(path) / exe_name).exists():
                found_realesrgan = True
                break
                
        if not found_realesrgan:
            self.warnings.append(
                "Real-ESRGAN not found. Upscaling will be disabled. "
                "Install from: https://github.com/xinntao/Real-ESRGAN/releases"
            )
            
    def _check_permissions(self) -> None:
        """Check write permissions in required directories."""
        dirs_to_check = [
            self.resolver.get_config_dir(),
            self.resolver.get_cache_dir(),
            self.resolver.get_data_dir(),
            self.resolver.get_log_dir(),
        ]
        
        for directory in dirs_to_check:
            try:
                # Try to create directory
                directory.mkdir(parents=True, exist_ok=True)
                
                # Try to write a test file
                test_file = directory / '.permission_test'
                test_file.write_text('test')
                test_file.unlink()
                
            except Exception as e:
                self.issues.append(
                    f"No write permission in {directory}: {e}"
                )
                
    def _check_network(self) -> None:
        """Check network connectivity for API access."""
        try:
            import requests
            
            # Check weather API
            response = requests.get(
                'https://api.weather.gov',
                timeout=5,
                headers={'User-Agent': 'AI-Wallpaper-Validator/1.0'}
            )
            
            if response.status_code != 200:
                self.warnings.append(
                    "Cannot reach weather.gov API. Weather integration may not work."
                )
                
        except Exception as e:
            self.warnings.append(
                f"Network connectivity issue: {e}. Some features may not work."
            )
            
    def _check_dependencies(self) -> None:
        """Check Python package dependencies."""
        required_packages = [
            'torch',
            'transformers',
            'diffusers',
            'pillow',
            'requests',
            'pyyaml',
            'click',
        ]
        
        missing = []
        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                missing.append(package)
                
        if missing:
            self.issues.append(
                f"Missing required Python packages: {', '.join(missing)}. "
                f"Install with: pip install {' '.join(missing)}"
            )
            
    def print_report(self) -> None:
        """Print validation report to console."""
        print("\n" + "="*60)
        print("AI WALLPAPER ENVIRONMENT VALIDATION REPORT")
        print("="*60 + "\n")
        
        if not self.issues and not self.warnings:
            print("✅ All checks passed! System is ready for AI wallpaper generation.")
            return
            
        if self.issues:
            print("❌ CRITICAL ISSUES (must be fixed):")
            for issue in self.issues:
                print(f"  • {issue}")
            print()
            
        if self.warnings:
            print("⚠️  WARNINGS (recommended to fix):")
            for warning in self.warnings:
                print(f"  • {warning}")
            print()
            
        print("="*60 + "\n")


def validate_environment() -> bool:
    """
    Run environment validation and return success status.
    
    Returns:
        True if no critical issues, False otherwise
    """
    validator = EnvironmentValidator()
    issues, warnings = validator.validate()
    
    # Print report
    validator.print_report()
    
    # Return False if there are critical issues
    return len(issues) == 0