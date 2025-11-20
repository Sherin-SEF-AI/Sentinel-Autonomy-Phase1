"""
Theme Manager

Manages application themes and accent colors.
"""

import logging
from pathlib import Path
from typing import Optional
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QSettings

logger = logging.getLogger(__name__)


class ThemeManager:
    """
    Manages application themes and styling.
    
    Features:
    - Dark and light themes
    - Configurable accent colors
    - Theme persistence
    - Dynamic theme switching
    """
    
    # Default accent colors
    DEFAULT_ACCENT_COLORS = {
        'blue': '#0078d4',
        'green': '#107c10',
        'red': '#e81123',
        'purple': '#881798',
        'orange': '#ff8c00',
        'teal': '#008080'
    }
    
    def __init__(self, app: QApplication):
        self.app = app
        self.settings = QSettings('SENTINEL', 'SentinelGUI')
        self.themes_dir = Path(__file__).parent
        
        # Load saved preferences
        self.current_theme = self.settings.value('theme', 'dark')
        self.current_accent = self.settings.value('accent_color', 'blue')
        
        logger.info(f"ThemeManager initialized with theme={self.current_theme}, accent={self.current_accent}")
    
    def apply_theme(self, theme_name: str = None, accent_color: str = None):
        """
        Apply a theme to the application.
        
        Args:
            theme_name: Theme name ('dark' or 'light'). If None, uses current theme.
            accent_color: Accent color name or hex code. If None, uses current accent.
        """
        if theme_name is not None:
            self.current_theme = theme_name
        
        if accent_color is not None:
            self.current_accent = accent_color
        
        # Load theme stylesheet
        theme_file = self.themes_dir / f"{self.current_theme}.qss"
        
        if not theme_file.exists():
            logger.error(f"Theme file not found: {theme_file}")
            return
        
        try:
            with open(theme_file, 'r') as f:
                stylesheet = f.read()
            
            # Replace accent color placeholders
            accent_hex = self._get_accent_hex(self.current_accent)
            accent_light = self._lighten_color(accent_hex)
            
            stylesheet = stylesheet.replace('{accent_color}', accent_hex)
            stylesheet = stylesheet.replace('{accent_color_light}', accent_light)
            
            # Apply stylesheet
            self.app.setStyleSheet(stylesheet)
            
            # Save preferences
            self.settings.setValue('theme', self.current_theme)
            self.settings.setValue('accent_color', self.current_accent)
            
            logger.info(f"Applied theme: {self.current_theme} with accent: {accent_hex}")
            
        except Exception as e:
            logger.error(f"Error applying theme: {e}")
    
    def set_theme(self, theme_name: str):
        """
        Set the theme.
        
        Args:
            theme_name: Theme name ('dark' or 'light')
        """
        if theme_name in ['dark', 'light']:
            self.apply_theme(theme_name=theme_name)
        else:
            logger.warning(f"Unknown theme: {theme_name}")
    
    def set_accent_color(self, accent_color: str):
        """
        Set the accent color.
        
        Args:
            accent_color: Accent color name or hex code
        """
        self.apply_theme(accent_color=accent_color)
    
    def toggle_theme(self):
        """Toggle between dark and light themes"""
        new_theme = 'light' if self.current_theme == 'dark' else 'dark'
        self.apply_theme(theme_name=new_theme)
        logger.info(f"Toggled theme to: {new_theme}")
    
    def get_current_theme(self) -> str:
        """Get the current theme name"""
        return self.current_theme
    
    def get_current_accent(self) -> str:
        """Get the current accent color"""
        return self.current_accent
    
    def get_available_themes(self) -> list:
        """Get list of available themes"""
        return ['dark', 'light']
    
    def get_available_accents(self) -> dict:
        """Get dictionary of available accent colors"""
        return self.DEFAULT_ACCENT_COLORS.copy()
    
    def _get_accent_hex(self, accent: str) -> str:
        """
        Get hex code for accent color.
        
        Args:
            accent: Accent color name or hex code
            
        Returns:
            Hex color code
        """
        # If already a hex code, return it
        if accent.startswith('#'):
            return accent
        
        # Look up in default colors
        return self.DEFAULT_ACCENT_COLORS.get(accent, self.DEFAULT_ACCENT_COLORS['blue'])
    
    def _lighten_color(self, hex_color: str, factor: float = 0.2) -> str:
        """
        Lighten a hex color.
        
        Args:
            hex_color: Hex color code (e.g., '#0078d4')
            factor: Lightening factor (0.0 to 1.0)
            
        Returns:
            Lightened hex color code
        """
        # Remove '#' if present
        hex_color = hex_color.lstrip('#')
        
        # Convert to RGB
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        
        # Lighten
        r = min(255, int(r + (255 - r) * factor))
        g = min(255, int(g + (255 - g) * factor))
        b = min(255, int(b + (255 - b) * factor))
        
        # Convert back to hex
        return f'#{r:02x}{g:02x}{b:02x}'
