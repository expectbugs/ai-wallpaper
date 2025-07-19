#!/usr/bin/env python3
"""
Wallpaper Setting Module for AI Wallpaper System
Handles desktop wallpaper setting across different desktop environments
"""

import os
import sys
import subprocess
import time
from pathlib import Path
from typing import List, Optional, Dict, Any

from .logger import get_logger
from .exceptions import WallpaperError
from .config_manager import get_config
from .wallpaper_setter import WallpaperSetter as CrossPlatformSetter

class WallpaperSetter:
    """Manages desktop wallpaper setting across different environments"""
    
    def __init__(self):
        """Initialize wallpaper setter"""
        self.config = get_config()
        self.logger = get_logger(model="Wallpaper")
        
        # Load wallpaper settings
        self.wallpaper_config = self.config.settings.get('wallpaper', {})
        self.auto_set = self.wallpaper_config.get('auto_set_wallpaper', True)
        self.de_config = self.wallpaper_config.get('desktop_environment', {})
        
        # Use the new cross-platform wallpaper setter
        self.cross_platform_setter = CrossPlatformSetter()
        
        # Desktop environment (from cross-platform setter)
        self.desktop_env = self.cross_platform_setter.desktop
        
    def _detect_desktop_environment(self) -> Optional[str]:
        """Auto-detect the current desktop environment
        
        Returns:
            Desktop environment name or None
        """
        de_type = self.de_config.get('type', 'auto')
        
        if de_type != 'auto':
            self.logger.info(f"Using configured desktop environment: {de_type}")
            return de_type
            
        # Try to detect desktop environment
        commands = self.de_config.get('commands', {})
        
        # Ensure DISPLAY is set before any detection
        env = os.environ.copy()
        if 'DISPLAY' not in env:
            # Try common display values
            for display in [':0', ':0.0', ':1', ':1.0']:
                if os.path.exists(f'/tmp/.X11-unix/X{display.strip(":")}'):
                    env['DISPLAY'] = display
                    self.logger.debug(f"Set DISPLAY to {display}")
                    break
            else:
                # Default to :0 if no X11 socket found
                env['DISPLAY'] = ':0'
                self.logger.warning("No X11 socket found, defaulting DISPLAY to :0")
        
        # Add synchronization delay between detection attempts
        detection_delay = 0.1  # 100ms between attempts
        
        for de_name, de_info in commands.items():
            detect_cmd = de_info.get('detect')
            if not detect_cmd:
                continue
                
            try:
                # Run detection command with proper environment
                result = subprocess.run(
                    detect_cmd,
                    shell=True,
                    capture_output=True,
                    text=True,
                    env=env,
                    timeout=5
                )
                
                if result.returncode == 0:
                    self.logger.info(f"Detected desktop environment: {de_name}")
                    return de_name
                    
                # Add delay to prevent race conditions
                time.sleep(detection_delay)
                    
            except subprocess.TimeoutExpired:
                self.logger.debug(f"Detection timeout for {de_name}")
            except Exception as e:
                self.logger.debug(f"Detection failed for {de_name}: {e}")
                
        self.logger.warning("Could not auto-detect desktop environment")
        return None
        
    def set_wallpaper(self, image_path: Path, desktop_env: Optional[str] = None) -> bool:
        """Set desktop wallpaper
        
        Args:
            image_path: Path to wallpaper image
            desktop_env: Override desktop environment detection
            
        Returns:
            True if successful
            
        Raises:
            WallpaperError: If setting fails
        """
        # Check if auto-set is enabled
        if not self.auto_set:
            self.logger.info("Auto-set wallpaper is disabled in configuration")
            return False
            
        # Ensure absolute path
        image_path = Path(image_path).absolute()
        
        # Verify image exists
        if not image_path.exists():
            raise WallpaperError(
                self.desktop_env or "unknown",
                "N/A",
                FileNotFoundError(f"Image not found: {image_path}")
            )
            
        # Use provided desktop environment or detected one
        de = desktop_env or self.desktop_env
        
        if not de:
            raise WallpaperError(
                "unknown",
                "N/A",
                Exception("No desktop environment detected or configured")
            )
            
        self.logger.info(f"Setting wallpaper on {de}: {image_path}")
        
        # Use the cross-platform setter
        try:
            success = self.cross_platform_setter.set_wallpaper(image_path)
            if success:
                self.logger.info("Successfully set wallpaper")
            else:
                self.logger.error("Failed to set wallpaper")
            return success
        except Exception as e:
            raise WallpaperError(de, str(image_path), e)
            
    def _set_generic_wallpaper(self, desktop_env: str, image_path: Path) -> bool:
        """Generic wallpaper setter using configured commands
        
        Args:
            desktop_env: Desktop environment name
            image_path: Path to wallpaper image
            
        Returns:
            True if successful
        """
        de_info = self.de_config.get('commands', {}).get(desktop_env)
        if not de_info:
            raise WallpaperError(
                desktop_env,
                "N/A",
                Exception(f"No configuration for desktop environment: {desktop_env}")
            )
            
        set_cmd = de_info.get('set')
        if not set_cmd:
            raise WallpaperError(
                desktop_env,
                "N/A",
                Exception(f"No set command configured for: {desktop_env}")
            )
            
        # Check dependencies
        deps = de_info.get('dependencies', [])
        for dep in deps:
            if not self._check_command_exists(dep):
                raise WallpaperError(
                    desktop_env,
                    set_cmd,
                    Exception(f"Required dependency not found: {dep}")
                )
                
        # Format command with image path
        command = set_cmd.format(image_path=str(image_path))
        
        # Set up environment
        env = os.environ.copy()
        if desktop_env in ['xfce', 'gnome', 'kde', 'mate', 'cinnamon', 'i3', 'sway', 'hyprland']:
            if 'DISPLAY' not in env:
                env['DISPLAY'] = ':0'
                
        try:
            # Execute command
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                env=env,
                timeout=10
            )
            
            if result.returncode != 0:
                raise WallpaperError(
                    desktop_env,
                    command,
                    Exception(f"Command failed: {result.stderr}")
                )
                
            self.logger.info(f"Wallpaper set successfully on {desktop_env}")
            return True
            
        except subprocess.TimeoutExpired:
            raise WallpaperError(
                desktop_env,
                command,
                Exception("Command timed out")
            )
        except Exception as e:
            raise WallpaperError(desktop_env, command, e)
            
    def _set_xfce_wallpaper(self, image_path: Path) -> bool:
        """Specialized XFCE wallpaper setter with multi-monitor support
        
        Args:
            image_path: Path to wallpaper image
            
        Returns:
            True if successful
        """
        self.logger.info("Setting wallpaper on XFCE with full monitor/workspace support")
        
        # Get all backdrop properties
        backdrop_props = self._get_xfce_backdrop_properties()
        
        if not backdrop_props:
            raise WallpaperError(
                "xfce",
                "xfconf-query",
                Exception("No XFCE backdrop properties found")
            )
            
        # Set up environment
        env = os.environ.copy()
        if 'DISPLAY' not in env:
            env['DISPLAY'] = ':0'
            
        self.logger.info(f"Found {len(backdrop_props)} backdrop properties")
        
        # Validate and set wallpaper on all properties
        valid_props_set = 0
        
        for prop in backdrop_props:
            try:
                # Validate property corresponds to actual monitor/workspace
                if not self._validate_xfce_property(prop, env):
                    self.logger.warning(f"Skipping invalid property: {prop}")
                    continue
                
                # Set image path
                cmd = [
                    "xfconf-query",
                    "-c", "xfce4-desktop",
                    "-p", prop,
                    "-s", str(image_path)
                ]
                
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    env=env,
                    timeout=5
                )
                
                if result.returncode != 0:
                    self.logger.warning(f"Failed to set {prop}: {result.stderr}")
                    continue
                    
                # Check if style property exists before setting
                style_prop = prop.replace("/last-image", "/image-style")
                if self._xfce_property_exists(style_prop, env):
                    # Set image style (4 = Scaled)
                    style_cmd = [
                        "xfconf-query",
                        "-c", "xfce4-desktop",
                        "-p", style_prop,
                        "-s", "4"
                    ]
                    
                    subprocess.run(style_cmd, env=env, timeout=5)
                else:
                    self.logger.debug(f"Style property {style_prop} does not exist, skipping")
                    
                self.logger.debug(f"Set wallpaper for {prop}")
                valid_props_set += 1
                
            except Exception as e:
                self.logger.warning(f"Error setting {prop}: {e}")
                
        # Ensure at least one property was set
        if valid_props_set == 0:
            raise WallpaperError(
                "xfce",
                "xfconf-query",
                Exception("No valid XFCE properties could be set")
            )
                
        # Reload xfdesktop
        try:
            reload_cmd = ["xfdesktop", "--reload"]
            subprocess.run(reload_cmd, env=env, timeout=5)
            self.logger.info("Reloaded xfdesktop configuration")
        except Exception as e:
            self.logger.warning(f"Failed to reload xfdesktop: {e}")
            
        # Verify at least one property was set
        if not self._verify_xfce_wallpaper(image_path, backdrop_props[:1]):
            raise WallpaperError(
                "xfce",
                "xfconf-query",
                Exception("Wallpaper verification failed")
            )
            
        self.logger.info(f"XFCE wallpaper set successfully on {len(backdrop_props)} properties")
        return True
        
    def _get_xfce_backdrop_properties(self) -> List[str]:
        """Get all XFCE backdrop property paths
        
        Returns:
            List of property paths
        """
        env = os.environ.copy()
        if 'DISPLAY' not in env:
            env['DISPLAY'] = ':0'
            
        try:
            # List all properties
            cmd = ["xfconf-query", "-c", "xfce4-desktop", "-l"]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                env=env,
                timeout=5
            )
            
            if result.returncode != 0:
                return []
                
            # Find backdrop properties
            properties = result.stdout.strip().split('\n')
            backdrop_props = [p for p in properties if p.endswith('/last-image')]
            
            return backdrop_props
            
        except Exception as e:
            self.logger.error(f"Failed to get XFCE properties: {e}")
            return []
            
    def _verify_xfce_wallpaper(self, expected_path: Path, properties: List[str]) -> bool:
        """Verify XFCE wallpaper was set
        
        Args:
            expected_path: Expected wallpaper path
            properties: Properties to check
            
        Returns:
            True if at least one property matches
        """
        env = os.environ.copy()
        if 'DISPLAY' not in env:
            env['DISPLAY'] = ':0'
            
        expected_str = str(expected_path.absolute())
        
        for prop in properties:
            try:
                cmd = [
                    "xfconf-query",
                    "-c", "xfce4-desktop",
                    "-p", prop
                ]
                
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    env=env,
                    timeout=5
                )
                
                if result.returncode == 0:
                    current = result.stdout.strip()
                    if current == expected_str:
                        return True
                        
            except Exception:
                pass
                
        return False
        
    def _check_command_exists(self, command: str) -> bool:
        """Check if a command exists in PATH
        
        Args:
            command: Command to check
            
        Returns:
            True if command exists
        """
        try:
            subprocess.run(
                ["which", command],
                capture_output=True,
                check=True,
                timeout=2
            )
            return True
        except:
            return False
            
    def verify_wallpaper(self, image_path: Path) -> bool:
        """Verify wallpaper was set correctly
        
        Args:
            image_path: Expected wallpaper path
            
        Returns:
            True if verified
        """
        if not self.desktop_env:
            self.logger.warning("Cannot verify - no desktop environment detected")
            return False
            
        # Use specialized verifier if available
        verifier_method = getattr(self, f"_verify_{self.desktop_env}_wallpaper", None)
        if verifier_method:
            # For XFCE, we need to pass the properties
            if self.desktop_env == "xfce":
                props = self._get_xfce_backdrop_properties()
                return verifier_method(image_path, props)
            else:
                return verifier_method(image_path)
                
        # No specific verifier available
        self.logger.info("No verification method available for this desktop environment")
        return True
        
    def _validate_xfce_property(self, prop: str, env: Dict[str, str]) -> bool:
        """Validate that an XFCE property corresponds to a real monitor/workspace
        
        Args:
            prop: Property path to validate
            env: Environment variables
            
        Returns:
            True if property is valid
        """
        # Extract monitor and workspace from property path
        # Format: /backdrop/screen0/monitorXXX/workspace0/last-image
        parts = prop.split('/')
        if len(parts) < 5:
            return False
            
        # Check if property exists by querying it
        try:
            cmd = [
                "xfconf-query",
                "-c", "xfce4-desktop",
                "-p", prop
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                env=env,
                timeout=2
            )
            
            # Property exists if query returns 0 (even if value is empty)
            return result.returncode == 0
            
        except Exception:
            return False
            
    def _xfce_property_exists(self, prop: str, env: Dict[str, str]) -> bool:
        """Check if an XFCE property exists
        
        Args:
            prop: Property path to check
            env: Environment variables
            
        Returns:
            True if property exists
        """
        try:
            cmd = [
                "xfconf-query",
                "-c", "xfce4-desktop",
                "-p", prop
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                env=env,
                timeout=2
            )
            
            # Property exists if return code is 0
            return result.returncode == 0
            
        except Exception:
            return False

# Global wallpaper setter instance
_wallpaper_setter: Optional[WallpaperSetter] = None

def set_wallpaper(image_path: Path) -> bool:
    """Set desktop wallpaper
    
    Args:
        image_path: Path to wallpaper image
        
    Returns:
        True if successful
    """
    global _wallpaper_setter
    if _wallpaper_setter is None:
        _wallpaper_setter = WallpaperSetter()
    return _wallpaper_setter.set_wallpaper(image_path)
    
def verify_wallpaper(image_path: Path) -> bool:
    """Verify wallpaper was set
    
    Args:
        image_path: Expected wallpaper path
        
    Returns:
        True if verified
    """
    global _wallpaper_setter
    if _wallpaper_setter is None:
        _wallpaper_setter = WallpaperSetter()
    return _wallpaper_setter.verify_wallpaper(image_path)