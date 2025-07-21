"""
Cross-platform wallpaper setting functionality.
Supports Windows, macOS, and various Linux desktop environments.
"""
import os
import sys
import platform
import subprocess
from pathlib import Path
from typing import Optional, List, Dict
from .logger import get_logger
from .exceptions import WallpaperError


logger = get_logger(__name__)


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
                except subprocess.CalledProcessError as e:
                    logger.warning(f"Failed to detect desktop environment via ps: {e}")
                except Exception as e:
                    logger.warning(f"Unexpected error detecting desktop environment: {e}")
                    
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
            raise WallpaperError(
                f"Wallpaper image not found: {image_path}\n"
                f"The generated wallpaper file is missing!\n"
                f"This is critical - cannot set non-existent wallpaper."
            )
            
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
        
        primary_error = None
        
        if setter := setters.get(self.desktop):
            try:
                return setter(image_path)
            except Exception as e:
                primary_error = e
                logger.warning(f"Failed with {self.desktop} method: {e}")
                # Try generic method as fallback
                if self.desktop != 'unknown':
                    try:
                        logger.info(f"Attempting generic fallback method...")
                        return self._set_generic(image_path)
                    except Exception as fallback_error:
                        raise WallpaperError(
                            f"All wallpaper setting methods failed!\n"
                            f"Primary method ({self.desktop}): {primary_error}\n"
                            f"Generic fallback: {fallback_error}\n"
                            f"Image path: {image_path}\n"
                            f"This is unacceptable - wallpaper MUST be set!"
                        )
        
        # No specific setter, try generic
        try:
            return self._set_generic(image_path)
        except Exception as e:
            raise WallpaperError(
                f"Failed to set wallpaper using generic method!\n"
                f"Desktop environment: {self.desktop}\n"
                f"Error: {e}\n"
                f"Image path: {image_path}"
            )
            
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
                raise WallpaperError(
                    "Windows API call failed to set wallpaper!\n"
                    f"Image path: {image_path}\n"
                    "The SystemParametersInfoW call returned False.\n"
                    "This usually means Windows rejected the wallpaper."
                )
                
        except ImportError as e:
            raise WallpaperError(
                f"Failed to import Windows ctypes module: {e}\n"
                "This is required for setting wallpaper on Windows."
            )
        except Exception as e:
            raise WallpaperError(
                f"Windows wallpaper setting failed: {e}\n"
                f"Image path: {image_path}\n"
                f"Error type: {type(e).__name__}"
            )
            
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
                raise WallpaperError(
                    f"osascript failed to set wallpaper!\n"
                    f"Return code: {result.returncode}\n"
                    f"Error output: {result.stderr}\n"
                    f"Image path: {image_path}\n"
                    "This usually means macOS rejected the AppleScript command."
                )
                
        except subprocess.SubprocessError as e:
            raise WallpaperError(
                f"Failed to run osascript command: {e}\n"
                f"Image path: {image_path}\n"
                "osascript is required for setting wallpaper on macOS."
            )
        except Exception as e:
            raise WallpaperError(
                f"macOS wallpaper setting failed: {e}\n"
                f"Image path: {image_path}\n"
                f"Error type: {type(e).__name__}"
            )
            
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
            
        except subprocess.CalledProcessError as e:
            raise WallpaperError(
                f"gsettings command failed!\n"
                f"Command: {e.cmd}\n"
                f"Return code: {e.returncode}\n"
                f"Image path: {image_path}\n"
                "This usually means GNOME is not running or gsettings is not available."
            )
        except Exception as e:
            raise WallpaperError(
                f"GNOME wallpaper setting failed: {e}\n"
                f"Image path: {image_path}\n"
                f"Error type: {type(e).__name__}"
            )
            
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
            
        except subprocess.CalledProcessError as e:
            raise WallpaperError(
                f"KDE wallpaper command failed!\n"
                f"Command: {e.cmd}\n"
                f"Return code: {e.returncode}\n"
                f"Image path: {image_path}\n"
                "This usually means KDE Plasma is not running or qdbus is not available."
            )
        except Exception as e:
            raise WallpaperError(
                f"KDE wallpaper setting failed: {e}\n"
                f"Image path: {image_path}\n"
                f"Error type: {type(e).__name__}"
            )
            
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
            
        except subprocess.CalledProcessError as e:
            raise WallpaperError(
                f"XFCE wallpaper command failed!\n"
                f"Command: {e.cmd}\n"
                f"Return code: {e.returncode}\n"
                f"Image path: {image_path}\n"
                "This usually means XFCE is not running or xfconf-query is not available."
            )
        except Exception as e:
            raise WallpaperError(
                f"XFCE wallpaper setting failed: {e}\n"
                f"Image path: {image_path}\n"
                f"Error type: {type(e).__name__}"
            )
            
    def _set_mate(self, image_path: Path) -> bool:
        """Set wallpaper on MATE."""
        try:
            subprocess.run([
                'gsettings', 'set', 'org.mate.background',
                'picture-filename', str(image_path)
            ], check=True)
            
            logger.info(f"Successfully set MATE wallpaper: {image_path}")
            return True
            
        except subprocess.CalledProcessError as e:
            raise WallpaperError(
                f"MATE wallpaper command failed!\n"
                f"Command: {e.cmd}\n"
                f"Return code: {e.returncode}\n"
                f"Image path: {image_path}\n"
                "This usually means MATE is not running or gsettings is not available."
            )
        except Exception as e:
            raise WallpaperError(
                f"MATE wallpaper setting failed: {e}\n"
                f"Image path: {image_path}\n"
                f"Error type: {type(e).__name__}"
            )
            
    def _set_cinnamon(self, image_path: Path) -> bool:
        """Set wallpaper on Cinnamon."""
        try:
            subprocess.run([
                'gsettings', 'set', 'org.cinnamon.desktop.background',
                'picture-uri', f'file://{image_path}'
            ], check=True)
            
            logger.info(f"Successfully set Cinnamon wallpaper: {image_path}")
            return True
            
        except subprocess.CalledProcessError as e:
            raise WallpaperError(
                f"Cinnamon wallpaper command failed!\n"
                f"Command: {e.cmd}\n"
                f"Return code: {e.returncode}\n"
                f"Image path: {image_path}\n"
                "This usually means Cinnamon is not running or gsettings is not available."
            )
        except Exception as e:
            raise WallpaperError(
                f"Cinnamon wallpaper setting failed: {e}\n"
                f"Image path: {image_path}\n"
                f"Error type: {type(e).__name__}"
            )
            
    def _set_lxde(self, image_path: Path) -> bool:
        """Set wallpaper on LXDE."""
        try:
            subprocess.run([
                'pcmanfm', '--set-wallpaper', str(image_path)
            ], check=True)
            
            logger.info(f"Successfully set LXDE wallpaper: {image_path}")
            return True
            
        except subprocess.CalledProcessError as e:
            raise WallpaperError(
                f"LXDE wallpaper command failed!\n"
                f"Command: {e.cmd}\n"
                f"Return code: {e.returncode}\n"
                f"Image path: {image_path}\n"
                "This usually means LXDE is not running or pcmanfm is not available."
            )
        except Exception as e:
            raise WallpaperError(
                f"LXDE wallpaper setting failed: {e}\n"
                f"Image path: {image_path}\n"
                f"Error type: {type(e).__name__}"
            )
            
    def _set_lxqt(self, image_path: Path) -> bool:
        """Set wallpaper on LXQt."""
        try:
            subprocess.run([
                'pcmanfm-qt', '--set-wallpaper', str(image_path)
            ], check=True)
            
            logger.info(f"Successfully set LXQt wallpaper: {image_path}")
            return True
            
        except subprocess.CalledProcessError as e:
            raise WallpaperError(
                f"LXQt wallpaper command failed!\n"
                f"Command: {e.cmd}\n"
                f"Return code: {e.returncode}\n"
                f"Image path: {image_path}\n"
                "This usually means LXQt is not running or pcmanfm-qt is not available."
            )
        except Exception as e:
            raise WallpaperError(
                f"LXQt wallpaper setting failed: {e}\n"
                f"Image path: {image_path}\n"
                f"Error type: {type(e).__name__}"
            )
            
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
                raise WallpaperError(
                    "Neither feh nor nitrogen found for i3!\n"
                    f"Image path: {image_path}\n"
                    "i3 requires either feh or nitrogen to set wallpaper.\n"
                    "Install with: sudo apt install feh\n"
                    "Or: sudo apt install nitrogen"
                )
                
        except subprocess.CalledProcessError as e:
            raise WallpaperError(
                f"i3 wallpaper command failed!\n"
                f"Command: {e.cmd}\n"
                f"Return code: {e.returncode}\n"
                f"Image path: {image_path}\n"
                "This usually means feh is not installed. Install with: sudo apt install feh"
            )
        except Exception as e:
            raise WallpaperError(
                f"i3 wallpaper setting failed: {e}\n"
                f"Image path: {image_path}\n"
                f"Error type: {type(e).__name__}"
            )
            
    def _set_sway(self, image_path: Path) -> bool:
        """Set wallpaper on Sway."""
        try:
            subprocess.run([
                'swaymsg', 'output', '*', 'bg', str(image_path), 'fill'
            ], check=True)
            
            logger.info(f"Successfully set Sway wallpaper: {image_path}")
            return True
            
        except subprocess.CalledProcessError as e:
            raise WallpaperError(
                f"Sway wallpaper command failed!\n"
                f"Command: {e.cmd}\n"
                f"Return code: {e.returncode}\n"
                f"Image path: {image_path}\n"
                "This usually means swaymsg is not available or Sway is not running."
            )
        except Exception as e:
            raise WallpaperError(
                f"Sway wallpaper setting failed: {e}\n"
                f"Image path: {image_path}\n"
                f"Error type: {type(e).__name__}"
            )
            
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
        
        errors = []
        for cmd in commands:
            try:
                if subprocess.run(['which', cmd[0]], capture_output=True).returncode == 0:
                    subprocess.run(cmd, check=True)
                    logger.info(f"Successfully set wallpaper with {cmd[0]}")
                    return True
            except subprocess.CalledProcessError as e:
                errors.append(f"{cmd[0]}: {e}")
                continue
            except Exception as e:
                errors.append(f"{cmd[0]}: {type(e).__name__}: {e}")
                continue
                
        raise WallpaperError(
            f"No wallpaper setter could set the image!\n"
            f"Image path: {image_path}\n"
            f"Tried commands: {', '.join(c[0] for c in commands)}\n"
            f"Errors: {'; '.join(errors) if errors else 'None found/installed'}\n"
            "Install one of: feh, nitrogen, xwallpaper, or hsetroot"
        )