"""
Cross-platform wallpaper setting functionality.
Supports Windows, macOS, and various Linux desktop environments.
"""
import os
import sys
import platform
import subprocess
import logging
from pathlib import Path
from typing import Optional, List, Dict


logger = logging.getLogger(__name__)


class WallpaperSetter:
    """Set desktop wallpaper across different platforms and environments."""
    
    def __init__(self):
        self.platform = platform.system()
        self.desktop = self._detect_desktop()
        logger.info(f"Detected platform: {self.platform}, desktop: {self.desktop}")
        
    def _detect_desktop(self) -> str:
        """Detect desktop environment."""
        if self.platform == 'Windows':
            return 'windows'
        elif self.platform == 'Darwin':
            return 'macos'
        else:
            # Linux desktop detection
            desktop = os.environ.get('XDG_CURRENT_DESKTOP', '').lower()
            session = os.environ.get('DESKTOP_SESSION', '').lower()
            
            # Check various desktop environments
            if 'gnome' in desktop or 'gnome' in session or 'ubuntu' in desktop:
                return 'gnome'
            elif 'kde' in desktop or 'plasma' in desktop or 'kde' in session:
                return 'kde'
            elif 'xfce' in desktop or 'xfce' in session:
                return 'xfce'
            elif 'mate' in desktop or 'mate' in session:
                return 'mate'
            elif 'cinnamon' in desktop or 'cinnamon' in session:
                return 'cinnamon'
            elif 'lxde' in desktop or 'lxde' in session:
                return 'lxde'
            elif 'lxqt' in desktop or 'lxqt' in session:
                return 'lxqt'
            elif 'i3' in desktop or 'i3' in session:
                return 'i3'
            elif 'sway' in desktop or 'sway' in session:
                return 'sway'
            else:
                # Try to detect from running processes
                try:
                    ps_output = subprocess.check_output(['ps', '-e'], text=True)
                    if 'gnome-shell' in ps_output:
                        return 'gnome'
                    elif 'plasmashell' in ps_output:
                        return 'kde'
                    elif 'xfce4-session' in ps_output:
                        return 'xfce'
                except:
                    pass
                    
                return 'unknown'
                
    def set_wallpaper(self, image_path: Path) -> bool:
        """
        Set wallpaper using appropriate method for the platform.
        
        Args:
            image_path: Path to the wallpaper image
            
        Returns:
            True if successful, False otherwise
        """
        if not image_path.exists():
            logger.error(f"Wallpaper image not found: {image_path}")
            return False
            
        # Convert to absolute path
        image_path = image_path.resolve()
        
        # Set wallpaper based on platform/desktop
        setters = {
            'windows': self._set_windows,
            'macos': self._set_macos,
            'gnome': self._set_gnome,
            'kde': self._set_kde,
            'xfce': self._set_xfce,
            'mate': self._set_mate,
            'cinnamon': self._set_cinnamon,
            'lxde': self._set_lxde,
            'lxqt': self._set_lxqt,
            'i3': self._set_i3,
            'sway': self._set_sway,
        }
        
        if setter := setters.get(self.desktop):
            try:
                return setter(image_path)
            except Exception as e:
                logger.error(f"Failed to set wallpaper using {self.desktop} method: {e}")
                # Try generic method as fallback
                if self.desktop != 'unknown':
                    return self._set_generic(image_path)
        else:
            return self._set_generic(image_path)
            
    def _set_windows(self, image_path: Path) -> bool:
        """Set wallpaper on Windows."""
        try:
            import ctypes
            
            # Use Windows API
            SPI_SETDESKWALLPAPER = 0x0014
            result = ctypes.windll.user32.SystemParametersInfoW(
                SPI_SETDESKWALLPAPER, 0, str(image_path), 3
            )
            
            if result:
                logger.info(f"Successfully set Windows wallpaper: {image_path}")
                return True
            else:
                logger.error("Windows API call failed")
                return False
                
        except Exception as e:
            logger.error(f"Windows wallpaper setting failed: {e}")
            return False
            
    def _set_macos(self, image_path: Path) -> bool:
        """Set wallpaper on macOS."""
        try:
            # Use osascript to set wallpaper
            script = f'''
            tell application "System Events"
                tell every desktop
                    set picture to "{image_path}"
                end tell
            end tell
            '''
            
            result = subprocess.run(
                ['osascript', '-e', script],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                logger.info(f"Successfully set macOS wallpaper: {image_path}")
                return True
            else:
                logger.error(f"osascript failed: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"macOS wallpaper setting failed: {e}")
            return False
            
    def _set_gnome(self, image_path: Path) -> bool:
        """Set wallpaper on GNOME."""
        try:
            # Set for both light and dark themes
            schemas = [
                'org.gnome.desktop.background',
                'org.gnome.desktop.screensaver'
            ]
            
            for schema in schemas:
                subprocess.run([
                    'gsettings', 'set', schema,
                    'picture-uri', f'file://{image_path}'
                ], check=True)
                
                # Also set for dark theme (GNOME 42+)
                subprocess.run([
                    'gsettings', 'set', schema,
                    'picture-uri-dark', f'file://{image_path}'
                ], check=False)  # Don't fail if dark theme not supported
                
            logger.info(f"Successfully set GNOME wallpaper: {image_path}")
            return True
            
        except Exception as e:
            logger.error(f"GNOME wallpaper setting failed: {e}")
            return False
            
    def _set_kde(self, image_path: Path) -> bool:
        """Set wallpaper on KDE Plasma."""
        try:
            # Use qdbus to set wallpaper
            script = f'''
            var allDesktops = desktops();
            for (i=0;i<allDesktops.length;i++) {{
                d = allDesktops[i];
                d.wallpaperPlugin = "org.kde.image";
                d.currentConfigGroup = Array("Wallpaper", "org.kde.image", "General");
                d.writeConfig("Image", "file://{image_path}");
            }}
            '''
            
            subprocess.run([
                'qdbus',
                'org.kde.plasmashell',
                '/PlasmaShell',
                'org.kde.PlasmaShell.evaluateScript',
                script
            ], check=True)
            
            logger.info(f"Successfully set KDE wallpaper: {image_path}")
            return True
            
        except Exception as e:
            logger.error(f"KDE wallpaper setting failed: {e}")
            return False
            
    def _set_xfce(self, image_path: Path) -> bool:
        """Set wallpaper on XFCE."""
        try:
            # Get all monitors and workspaces
            monitors = subprocess.check_output([
                'xfconf-query', '-c', 'xfce4-desktop', '-l'
            ], text=True).splitlines()
            
            # Filter for last-image properties
            image_properties = [
                prop for prop in monitors 
                if 'last-image' in prop and 'workspace' in prop
            ]
            
            # Set wallpaper for each property
            for prop in image_properties:
                subprocess.run([
                    'xfconf-query', '-c', 'xfce4-desktop',
                    '-p', prop, '-s', str(image_path)
                ], check=True)
                
            # Force xfdesktop to reload
            subprocess.run(['xfdesktop', '--reload'], check=False)
            
            logger.info(f"Successfully set XFCE wallpaper: {image_path}")
            return True
            
        except Exception as e:
            logger.error(f"XFCE wallpaper setting failed: {e}")
            return False
            
    def _set_mate(self, image_path: Path) -> bool:
        """Set wallpaper on MATE."""
        try:
            subprocess.run([
                'gsettings', 'set', 'org.mate.background',
                'picture-filename', str(image_path)
            ], check=True)
            
            logger.info(f"Successfully set MATE wallpaper: {image_path}")
            return True
            
        except Exception as e:
            logger.error(f"MATE wallpaper setting failed: {e}")
            return False
            
    def _set_cinnamon(self, image_path: Path) -> bool:
        """Set wallpaper on Cinnamon."""
        try:
            subprocess.run([
                'gsettings', 'set', 'org.cinnamon.desktop.background',
                'picture-uri', f'file://{image_path}'
            ], check=True)
            
            logger.info(f"Successfully set Cinnamon wallpaper: {image_path}")
            return True
            
        except Exception as e:
            logger.error(f"Cinnamon wallpaper setting failed: {e}")
            return False
            
    def _set_lxde(self, image_path: Path) -> bool:
        """Set wallpaper on LXDE."""
        try:
            subprocess.run([
                'pcmanfm', '--set-wallpaper', str(image_path)
            ], check=True)
            
            logger.info(f"Successfully set LXDE wallpaper: {image_path}")
            return True
            
        except Exception as e:
            logger.error(f"LXDE wallpaper setting failed: {e}")
            return False
            
    def _set_lxqt(self, image_path: Path) -> bool:
        """Set wallpaper on LXQt."""
        try:
            subprocess.run([
                'pcmanfm-qt', '--set-wallpaper', str(image_path)
            ], check=True)
            
            logger.info(f"Successfully set LXQt wallpaper: {image_path}")
            return True
            
        except Exception as e:
            logger.error(f"LXQt wallpaper setting failed: {e}")
            return False
            
    def _set_i3(self, image_path: Path) -> bool:
        """Set wallpaper on i3."""
        try:
            # Try feh first
            if subprocess.run(['which', 'feh'], capture_output=True).returncode == 0:
                subprocess.run([
                    'feh', '--bg-scale', str(image_path)
                ], check=True)
                logger.info(f"Successfully set i3 wallpaper with feh: {image_path}")
                return True
                
            # Try nitrogen
            elif subprocess.run(['which', 'nitrogen'], capture_output=True).returncode == 0:
                subprocess.run([
                    'nitrogen', '--set-scaled', str(image_path)
                ], check=True)
                logger.info(f"Successfully set i3 wallpaper with nitrogen: {image_path}")
                return True
                
            else:
                logger.error("Neither feh nor nitrogen found for i3")
                return False
                
        except Exception as e:
            logger.error(f"i3 wallpaper setting failed: {e}")
            return False
            
    def _set_sway(self, image_path: Path) -> bool:
        """Set wallpaper on Sway."""
        try:
            subprocess.run([
                'swaymsg', 'output', '*', 'bg', str(image_path), 'fill'
            ], check=True)
            
            logger.info(f"Successfully set Sway wallpaper: {image_path}")
            return True
            
        except Exception as e:
            logger.error(f"Sway wallpaper setting failed: {e}")
            return False
            
    def _set_generic(self, image_path: Path) -> bool:
        """Generic wallpaper setting fallback."""
        logger.warning("Using generic wallpaper setting method")
        
        # Try common wallpaper setters
        commands = [
            ['feh', '--bg-scale', str(image_path)],
            ['nitrogen', '--set-scaled', str(image_path)],
            ['xwallpaper', '--zoom', str(image_path)],
            ['hsetroot', '-fill', str(image_path)],
        ]
        
        for cmd in commands:
            try:
                if subprocess.run(['which', cmd[0]], capture_output=True).returncode == 0:
                    subprocess.run(cmd, check=True)
                    logger.info(f"Successfully set wallpaper with {cmd[0]}")
                    return True
            except:
                continue
                
        logger.error("No suitable wallpaper setter found")
        return False