#!/usr/bin/env python3
"""
Resource Manager for AI Wallpaper System
Manages VRAM and system resources between model switches
"""

import gc
import os
import psutil
from typing import Dict, Any, Optional
from pathlib import Path

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from ..core import get_logger
from ..core.exceptions import ResourceError

class ResourceManager:
    """Manage VRAM and system resources"""
    
    def __init__(self):
        """Initialize resource manager"""
        self.logger = get_logger(model="Resources")
        self.allocated_models = {}
        self._last_cleanup = None
        
    def can_load_model(self, model_name: str, requirements: Dict[str, Any]) -> bool:
        """Check if model can be loaded
        
        Args:
            model_name: Name of model to load
            requirements: Resource requirements
            
        Returns:
            True if resources are available
        """
        # Check VRAM if using CUDA
        if TORCH_AVAILABLE and torch.cuda.is_available():
            device_props = torch.cuda.get_device_properties(0)
            total_vram_gb = device_props.total_memory / 1024**3
            used_vram_gb = torch.cuda.memory_allocated() / 1024**3
            free_vram_gb = total_vram_gb - used_vram_gb
            
            required_vram = requirements.get('vram_gb', 24)
            
            self.logger.info(f"VRAM check: {free_vram_gb:.1f}GB free, {required_vram}GB required")
            
            if free_vram_gb < required_vram * 1.1:  # 10% buffer
                return False
                
        # Check disk space
        stat = os.statvfs(os.path.expanduser("~"))
        free_disk_gb = (stat.f_bavail * stat.f_frsize) / 1024**3
        
        required_disk = requirements.get('disk_gb', 20)
        
        self.logger.info(f"Disk check: {free_disk_gb:.1f}GB free, {required_disk}GB required")
        
        if free_disk_gb < required_disk:
            return False
            
        # Check system RAM
        memory = psutil.virtual_memory()
        free_ram_gb = memory.available / 1024**3
        
        # Assume we need at least 8GB free RAM
        if free_ram_gb < 8:
            self.logger.warning(f"Low RAM: {free_ram_gb:.1f}GB available")
            
        return True
        
    def prepare_for_model(self, model_name: str) -> None:
        """Clean up resources before loading model
        
        Args:
            model_name: Model to prepare for
        """
        self.logger.info(f"Preparing resources for {model_name}...")
        
        # Unload all other models
        for name, model in list(self.allocated_models.items()):
            if name != model_name:
                self.logger.info(f"Unloading {name} to free resources")
                # FAIL LOUD - cleanup must succeed
                model.cleanup()
                del self.allocated_models[name]
                
        # Force garbage collection
        self._aggressive_cleanup()
        
    def register_model(self, model_name: str, model_instance: Any) -> None:
        """Register a loaded model
        
        Args:
            model_name: Name of the model
            model_instance: Model instance
        """
        self.allocated_models[model_name] = model_instance
        self.logger.info(f"Registered model: {model_name}")
        
    def unregister_model(self, model_name: str) -> None:
        """Unregister a model
        
        Args:
            model_name: Name of the model
        """
        if model_name in self.allocated_models:
            del self.allocated_models[model_name]
            self.logger.info(f"Unregistered model: {model_name}")
            
    def get_resource_usage(self) -> Dict[str, Any]:
        """Get current resource usage
        
        Returns:
            Resource usage statistics
        """
        usage = {
            'models_loaded': list(self.allocated_models.keys())
        }
        
        # System memory
        memory = psutil.virtual_memory()
        usage['ram'] = {
            'total_gb': memory.total / 1024**3,
            'used_gb': memory.used / 1024**3,
            'free_gb': memory.available / 1024**3,
            'percent': memory.percent
        }
        
        # Disk
        disk = psutil.disk_usage('/')
        usage['disk'] = {
            'total_gb': disk.total / 1024**3,
            'used_gb': disk.used / 1024**3,
            'free_gb': disk.free / 1024**3,
            'percent': disk.percent
        }
        
        # GPU if available
        if TORCH_AVAILABLE and torch.cuda.is_available():
            device_props = torch.cuda.get_device_properties(0)
            total_vram = device_props.total_memory
            allocated = torch.cuda.memory_allocated()
            reserved = torch.cuda.memory_reserved()
            
            usage['gpu'] = {
                'device': device_props.name,
                'vram_total_gb': total_vram / 1024**3,
                'vram_allocated_gb': allocated / 1024**3,
                'vram_reserved_gb': reserved / 1024**3,
                'vram_free_gb': (total_vram - allocated) / 1024**3
            }
        else:
            usage['gpu'] = None
            
        # CPU
        usage['cpu'] = {
            'count': psutil.cpu_count(),
            'percent': psutil.cpu_percent(interval=0.1)
        }
        
        return usage
        
    def check_critical_resources(self, model_name: str, requirements: Dict[str, Any]) -> None:
        """Check resources and raise error if insufficient
        
        Args:
            model_name: Model name
            requirements: Resource requirements
            
        Raises:
            ResourceError: If resources insufficient
        """
        # Check VRAM
        if TORCH_AVAILABLE and torch.cuda.is_available():
            device_props = torch.cuda.get_device_properties(0)
            total_vram_gb = device_props.total_memory / 1024**3
            used_vram_gb = torch.cuda.memory_allocated() / 1024**3
            free_vram_gb = total_vram_gb - used_vram_gb
            
            required_vram = requirements.get('vram_gb', 24)
            
            if free_vram_gb < required_vram:
                # Try cleanup first
                self._aggressive_cleanup()
                
                # Check again
                used_vram_gb = torch.cuda.memory_allocated() / 1024**3
                free_vram_gb = total_vram_gb - used_vram_gb
                
                if free_vram_gb < required_vram:
                    raise ResourceError(
                        "VRAM",
                        f"{required_vram}GB",
                        f"{free_vram_gb:.1f}GB"
                    )
                    
        # Check disk
        stat = os.statvfs(os.path.expanduser("~"))
        free_disk_gb = (stat.f_bavail * stat.f_frsize) / 1024**3
        required_disk = requirements.get('disk_gb', 20)
        
        if free_disk_gb < required_disk:
            raise ResourceError(
                "Disk space",
                f"{required_disk}GB",
                f"{free_disk_gb:.1f}GB"
            )
            
    def _aggressive_cleanup(self) -> None:
        """Perform aggressive cleanup of resources"""
        self.logger.info("Performing aggressive resource cleanup...")
        
        # Python garbage collection
        gc.collect()
        gc.collect()  # Run twice to ensure cleanup
        
        # PyTorch specific cleanup
        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
        # Log new state
        self.log_resource_state()
        
    def log_resource_state(self) -> None:
        """Log current resource state"""
        usage = self.get_resource_usage()
        
        self.logger.info("=== Resource State ===")
        self.logger.info(f"Models loaded: {', '.join(usage['models_loaded']) or 'None'}")
        self.logger.info(f"RAM: {usage['ram']['used_gb']:.1f}/{usage['ram']['total_gb']:.1f}GB ({usage['ram']['percent']:.1f}%)")
        
        if usage['gpu']:
            gpu = usage['gpu']
            self.logger.info(f"GPU: {gpu['device']}")
            self.logger.info(f"VRAM: {gpu['vram_allocated_gb']:.1f}/{gpu['vram_total_gb']:.1f}GB allocated")
            self.logger.info(f"VRAM Free: {gpu['vram_free_gb']:.1f}GB")

# Global resource manager instance
_resource_manager: Optional[ResourceManager] = None

def get_resource_manager() -> ResourceManager:
    """Get global resource manager instance
    
    Returns:
        ResourceManager instance
    """
    global _resource_manager
    if _resource_manager is None:
        _resource_manager = ResourceManager()
    return _resource_manager