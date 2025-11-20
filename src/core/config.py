"""Configuration management for SENTINEL system."""

import yaml
from pathlib import Path
from typing import Any, Dict, Optional
import logging


class ConfigManager:
    """Manages YAML configuration loading and access."""
    
    def __init__(self, config_path: str):
        """
        Initialize ConfigManager.
        
        Args:
            config_path: Path to YAML configuration file
        """
        self.config_path = Path(config_path)
        self.config: Dict[str, Any] = {}
        self.logger = logging.getLogger(__name__)
        
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        self.load()
    
    def load(self) -> None:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as f:
                self.config = yaml.safe_load(f)
            self.logger.info(f"Configuration loaded from {self.config_path}")
        except yaml.YAMLError as e:
            self.logger.error(f"Error parsing YAML configuration: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Error loading configuration: {e}")
            raise
    
    def reload(self) -> None:
        """Reload configuration from file."""
        self.logger.info("Reloading configuration...")
        self.load()
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by key.
        
        Args:
            key: Configuration key (supports dot notation, e.g., 'cameras.interior.device')
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """
        Get entire configuration section.
        
        Args:
            section: Section name (e.g., 'cameras', 'models')
            
        Returns:
            Configuration section as dictionary
        """
        return self.config.get(section, {})
    
    def set(self, key: str, value: Any) -> None:
        """
        Set configuration value (runtime only, not persisted).
        
        Args:
            key: Configuration key (supports dot notation)
            value: Value to set
        """
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
        self.logger.debug(f"Configuration updated: {key} = {value}")
    
    def save(self, output_path: Optional[str] = None) -> None:
        """
        Save current configuration to file.
        
        Args:
            output_path: Path to save configuration (defaults to original path)
        """
        save_path = Path(output_path) if output_path else self.config_path
        
        try:
            with open(save_path, 'w') as f:
                yaml.safe_dump(self.config, f, default_flow_style=False, sort_keys=False)
            self.logger.info(f"Configuration saved to {save_path}")
        except Exception as e:
            self.logger.error(f"Error saving configuration: {e}")
            raise
    
    def validate(self) -> bool:
        """
        Validate configuration has required fields.
        
        Returns:
            True if configuration is valid
        """
        required_sections = ['system', 'cameras', 'models']
        
        for section in required_sections:
            if section not in self.config:
                self.logger.error(f"Missing required configuration section: {section}")
                return False
        
        return True
    
    def __repr__(self) -> str:
        return f"ConfigManager(config_path='{self.config_path}')"
