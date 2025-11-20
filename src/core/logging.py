"""Logging infrastructure for SENTINEL system."""

import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime


class LoggerSetup:
    """Configures logging for the SENTINEL system."""
    
    @staticmethod
    def setup(
        log_level: str = "INFO",
        log_file: Optional[str] = None,
        log_dir: str = "logs",
        console_output: bool = True
    ) -> logging.Logger:
        """
        Set up logging configuration.
        
        Args:
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_file: Optional log file name (auto-generated if None)
            log_dir: Directory for log files
            console_output: Whether to output logs to console
            
        Returns:
            Configured root logger
        """
        # Create log directory if it doesn't exist
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        
        # Generate log file name if not provided
        if log_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = f"sentinel_{timestamp}.log"
        
        log_file_path = log_path / log_file
        
        # Configure root logger
        logger = logging.getLogger()
        logger.setLevel(getattr(logging, log_level.upper()))
        
        # Remove existing handlers
        logger.handlers.clear()
        
        # Create formatters
        detailed_formatter = logging.Formatter(
            fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        simple_formatter = logging.Formatter(
            fmt='%(levelname)s - %(name)s - %(message)s'
        )
        
        # File handler
        file_handler = logging.FileHandler(log_file_path, mode='a')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(detailed_formatter)
        logger.addHandler(file_handler)
        
        # Console handler
        if console_output:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(getattr(logging, log_level.upper()))
            console_handler.setFormatter(simple_formatter)
            logger.addHandler(console_handler)
        
        logger.info(f"Logging initialized - Level: {log_level}, File: {log_file_path}")
        
        return logger
    
    @staticmethod
    def get_logger(name: str) -> logging.Logger:
        """
        Get a logger instance for a specific module.
        
        Args:
            name: Logger name (typically __name__)
            
        Returns:
            Logger instance
        """
        return logging.getLogger(name)
    
    @staticmethod
    def set_level(level: str) -> None:
        """
        Change logging level at runtime.
        
        Args:
            level: New logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        """
        logger = logging.getLogger()
        logger.setLevel(getattr(logging, level.upper()))
        
        for handler in logger.handlers:
            if isinstance(handler, logging.StreamHandler):
                handler.setLevel(getattr(logging, level.upper()))
        
        logger.info(f"Logging level changed to {level}")
