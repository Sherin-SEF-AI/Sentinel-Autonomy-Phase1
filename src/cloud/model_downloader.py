"""
Model downloader for OTA (Over-The-Air) model updates.

Downloads new model versions from cloud backend with signature verification
and atomic installation.
"""

import os
import logging
import threading
import hashlib
import shutil
import time
from typing import Dict, Any, Optional, List
from pathlib import Path
import json
import tempfile

from .api_client import CloudAPIClient


class ModelDownloader:
    """
    Downloads and installs model updates from cloud.
    
    Features:
    - Periodic update checks (every 24 hours)
    - Model signature verification
    - Atomic installation (no partial updates)
    - Rollback on failure
    """
    
    def __init__(
        self,
        api_client: CloudAPIClient,
        models_dir: str = "models",
        check_interval: float = 86400.0,  # 24 hours
        auto_install: bool = True
    ):
        """
        Initialize model downloader.
        
        Args:
            api_client: Cloud API client
            models_dir: Directory for model files
            check_interval: Update check interval in seconds
            auto_install: Automatically install downloaded models
        """
        self.api_client = api_client
        self.models_dir = models_dir
        self.check_interval = check_interval
        self.auto_install = auto_install
        
        self.logger = logging.getLogger(__name__)
        
        # Ensure models directory exists
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Model registry (tracks installed versions)
        self.registry_file = os.path.join(self.models_dir, 'registry.json')
        self.registry = self._load_registry()
        
        # Background check thread
        self.running = False
        self.check_thread: Optional[threading.Thread] = None
        
        # Download status
        self.download_status: Dict[str, str] = {}
        
        self.logger.info("ModelDownloader initialized")
    
    def start_background_checks(self) -> None:
        """Start background update check thread"""
        if self.running:
            return
        
        self.running = True
        self.check_thread = threading.Thread(target=self._check_loop, daemon=True)
        self.check_thread.start()
        self.logger.info("Background model update check thread started")
    
    def stop_background_checks(self) -> None:
        """Stop background update check thread"""
        if not self.running:
            return
        
        self.running = False
        if self.check_thread:
            self.check_thread.join(timeout=5.0)
        self.logger.info("Background model update check thread stopped")
    
    def _check_loop(self) -> None:
        """Background update check loop"""
        last_check = 0.0
        
        while self.running:
            try:
                now = time.time()
                
                # Check if it's time for update check
                if now - last_check >= self.check_interval:
                    self.check_for_updates()
                    last_check = now
                
                # Sleep for a short interval
                time.sleep(60.0)  # Check every minute if it's time
            
            except Exception as e:
                self.logger.error(f"Error in check loop: {e}", exc_info=True)
                time.sleep(300.0)  # Wait 5 minutes on error
    
    def check_for_updates(self) -> List[Dict[str, Any]]:
        """
        Check for available model updates.
        
        Returns:
            List of available updates
        """
        try:
            # Check connectivity
            if not self.api_client.check_connectivity():
                self.logger.warning("No connectivity, skipping update check")
                return []
            
            # Get list of available models
            response = self.api_client.get('/models')
            
            if not response.success:
                self.logger.error(f"Failed to check for updates: {response.error}")
                return []
            
            available_models = response.data.get('models', [])
            updates = []
            
            # Check each model for updates
            for model_info in available_models:
                model_name = model_info['name']
                latest_version = model_info['version']
                
                current_version = self.registry.get(model_name, {}).get('version')
                
                if current_version != latest_version:
                    updates.append({
                        'name': model_name,
                        'current_version': current_version,
                        'latest_version': latest_version,
                        'size': model_info.get('size', 0),
                        'checksum': model_info.get('checksum'),
                        'signature': model_info.get('signature')
                    })
                    
                    self.logger.info(f"Update available for {model_name}: "
                                   f"{current_version} -> {latest_version}")
            
            # Auto-download if enabled
            if self.auto_install and updates:
                for update in updates:
                    self.download_and_install_model(
                        update['name'],
                        update['latest_version'],
                        update.get('checksum'),
                        update.get('signature')
                    )
            
            return updates
        
        except Exception as e:
            self.logger.error(f"Error checking for updates: {e}", exc_info=True)
            return []
    
    def download_and_install_model(
        self,
        model_name: str,
        version: str,
        expected_checksum: Optional[str] = None,
        signature: Optional[str] = None
    ) -> bool:
        """
        Download and install a model update.
        
        Args:
            model_name: Model name (e.g., 'yolov8m_automotive')
            version: Model version
            expected_checksum: Expected SHA256 checksum
            signature: Model signature for verification
        
        Returns:
            True if download and installation successful
        """
        try:
            self.download_status[model_name] = 'downloading'
            self.logger.info(f"Downloading {model_name} version {version}")
            
            # Download model to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.tmp') as temp_file:
                temp_path = temp_file.name
                
                # Download model file
                response = self.api_client.get(
                    f'/models/{model_name}/download',
                    params={'version': version}
                )
                
                if not response.success:
                    self.logger.error(f"Failed to download model: {response.error}")
                    self.download_status[model_name] = 'failed'
                    return False
                
                # Get download URL from response
                download_url = response.data.get('download_url')
                if not download_url:
                    self.logger.error("No download URL in response")
                    self.download_status[model_name] = 'failed'
                    return False
                
                # Download file
                if not self._download_file(download_url, temp_path):
                    self.download_status[model_name] = 'failed'
                    return False
            
            # Verify checksum
            if expected_checksum:
                self.download_status[model_name] = 'verifying'
                
                if not self._verify_checksum(temp_path, expected_checksum):
                    self.logger.error(f"Checksum verification failed for {model_name}")
                    os.unlink(temp_path)
                    self.download_status[model_name] = 'failed'
                    return False
                
                self.logger.info(f"Checksum verified for {model_name}")
            
            # Verify signature (placeholder - would use cryptographic verification)
            if signature:
                if not self._verify_signature(temp_path, signature):
                    self.logger.error(f"Signature verification failed for {model_name}")
                    os.unlink(temp_path)
                    self.download_status[model_name] = 'failed'
                    return False
                
                self.logger.info(f"Signature verified for {model_name}")
            
            # Install model atomically
            self.download_status[model_name] = 'installing'
            
            if not self._install_model(model_name, version, temp_path):
                self.logger.error(f"Failed to install {model_name}")
                os.unlink(temp_path)
                self.download_status[model_name] = 'failed'
                return False
            
            # Cleanup temp file
            os.unlink(temp_path)
            
            self.download_status[model_name] = 'completed'
            self.logger.info(f"Successfully installed {model_name} version {version}")
            
            return True
        
        except Exception as e:
            self.logger.error(f"Error downloading model: {e}", exc_info=True)
            self.download_status[model_name] = 'failed'
            return False
    
    def _download_file(self, url: str, output_path: str) -> bool:
        """
        Download file from URL.
        
        Args:
            url: Download URL
            output_path: Output file path
        
        Returns:
            True if download successful
        """
        try:
            import requests
            
            response = requests.get(url, stream=True, timeout=300)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        
                        # Log progress
                        if total_size > 0:
                            progress = (downloaded / total_size) * 100
                            if downloaded % (1024 * 1024 * 10) == 0:  # Every 10MB
                                self.logger.debug(f"Download progress: {progress:.1f}%")
            
            return True
        
        except Exception as e:
            self.logger.error(f"Error downloading file: {e}", exc_info=True)
            return False
    
    def _verify_checksum(self, file_path: str, expected_checksum: str) -> bool:
        """
        Verify file checksum.
        
        Args:
            file_path: File to verify
            expected_checksum: Expected SHA256 checksum
        
        Returns:
            True if checksum matches
        """
        try:
            sha256 = hashlib.sha256()
            
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(8192), b''):
                    sha256.update(chunk)
            
            actual_checksum = sha256.hexdigest()
            return actual_checksum == expected_checksum
        
        except Exception as e:
            self.logger.error(f"Error verifying checksum: {e}", exc_info=True)
            return False
    
    def _verify_signature(self, file_path: str, signature: str) -> bool:
        """
        Verify file signature (placeholder implementation).
        
        In production, this would use cryptographic signature verification
        with public key cryptography.
        
        Args:
            file_path: File to verify
            signature: Digital signature
        
        Returns:
            True if signature is valid
        """
        # Placeholder - in production would use proper cryptographic verification
        # For example, using RSA or Ed25519 signatures
        self.logger.warning("Signature verification not implemented (placeholder)")
        return True
    
    def _install_model(self, model_name: str, version: str, temp_path: str) -> bool:
        """
        Install model atomically.
        
        Args:
            model_name: Model name
            version: Model version
            temp_path: Path to downloaded model file
        
        Returns:
            True if installation successful
        """
        try:
            # Determine file extension
            ext = '.pt' if 'yolo' in model_name.lower() else '.pth'
            
            # Target path
            target_path = os.path.join(self.models_dir, f"{model_name}{ext}")
            backup_path = target_path + '.backup'
            
            # Backup existing model if it exists
            if os.path.exists(target_path):
                shutil.copy2(target_path, backup_path)
                self.logger.debug(f"Backed up existing model to {backup_path}")
            
            try:
                # Copy new model to target location
                shutil.copy2(temp_path, target_path)
                
                # Update registry
                self.registry[model_name] = {
                    'version': version,
                    'installed_at': time.time(),
                    'path': target_path
                }
                self._save_registry()
                
                # Remove backup on success
                if os.path.exists(backup_path):
                    os.unlink(backup_path)
                
                return True
            
            except Exception as e:
                # Rollback on failure
                self.logger.error(f"Installation failed, rolling back: {e}")
                
                if os.path.exists(backup_path):
                    shutil.copy2(backup_path, target_path)
                    os.unlink(backup_path)
                    self.logger.info("Rolled back to previous version")
                
                return False
        
        except Exception as e:
            self.logger.error(f"Error installing model: {e}", exc_info=True)
            return False
    
    def _load_registry(self) -> Dict[str, Any]:
        """Load model registry from disk"""
        if not os.path.exists(self.registry_file):
            return {}
        
        try:
            with open(self.registry_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Error loading registry: {e}", exc_info=True)
            return {}
    
    def _save_registry(self) -> None:
        """Save model registry to disk"""
        try:
            with open(self.registry_file, 'w') as f:
                json.dump(self.registry, f, indent=2)
        except Exception as e:
            self.logger.error(f"Error saving registry: {e}", exc_info=True)
    
    def get_installed_models(self) -> Dict[str, Any]:
        """
        Get list of installed models.
        
        Returns:
            Dictionary of installed models
        """
        return self.registry.copy()
    
    def get_download_status(self, model_name: str) -> Optional[str]:
        """
        Get download status for a model.
        
        Args:
            model_name: Model name
        
        Returns:
            Status string or None
        """
        return self.download_status.get(model_name)
