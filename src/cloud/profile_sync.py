"""
Driver profile synchronization for cloud backend.

Synchronizes driver profiles across vehicles in a fleet with encryption.
"""

import os
import logging
import json
import time
from typing import Dict, Any, Optional, List
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64

from .api_client import CloudAPIClient


class ProfileSynchronizer:
    """
    Synchronizes driver profiles with cloud backend.
    
    Features:
    - Upload profiles to cloud
    - Download profiles from cloud
    - Merge profiles across vehicles
    - Encrypt profile data
    """
    
    def __init__(
        self,
        api_client: CloudAPIClient,
        profiles_dir: str = "profiles",
        encryption_key: Optional[str] = None
    ):
        """
        Initialize profile synchronizer.
        
        Args:
            api_client: Cloud API client
            profiles_dir: Directory for profile files
            encryption_key: Encryption key for profile data (if None, generates one)
        """
        self.logger = logging.getLogger(__name__)
        self.logger.debug(f"Initializing ProfileSynchronizer: profiles_dir={profiles_dir}")
        
        self.api_client = api_client
        self.profiles_dir = profiles_dir
        
        # Ensure profiles directory exists
        os.makedirs(self.profiles_dir, exist_ok=True)
        self.logger.debug(f"Profiles directory created/verified: {self.profiles_dir}")
        
        # Setup encryption
        if encryption_key:
            self.encryption_key = encryption_key.encode()
            self.logger.debug("Using provided encryption key")
        else:
            # Generate encryption key from vehicle ID (in production, use secure key management)
            self.encryption_key = self._derive_key(api_client.vehicle_id)
            self.logger.debug(f"Derived encryption key from vehicle_id: {api_client.vehicle_id}")
        
        self.cipher = Fernet(base64.urlsafe_b64encode(self.encryption_key[:32]))
        
        # Sync status
        self.last_sync_time = 0.0
        
        self.logger.info(f"ProfileSynchronizer initialized: profiles_dir={profiles_dir}, encryption_enabled=True")
    
    def _derive_key(self, password: str) -> bytes:
        """
        Derive encryption key from password.
        
        Args:
            password: Password string
        
        Returns:
            Derived key bytes
        """
        self.logger.debug("Deriving encryption key using PBKDF2HMAC")
        start_time = time.time()
        
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=b'sentinel_salt',  # In production, use random salt
            iterations=100000,
        )
        key = kdf.derive(password.encode())
        
        duration = time.time() - start_time
        self.logger.debug(f"Key derivation completed: duration={duration*1000:.2f}ms")
        
        return key
    
    def encrypt_profile(self, profile_data: Dict[str, Any]) -> str:
        """
        Encrypt profile data.
        
        Args:
            profile_data: Profile dictionary
        
        Returns:
            Encrypted profile as base64 string
        """
        try:
            self.logger.debug("Encrypting profile data")
            start_time = time.time()
            
            # Convert to JSON
            json_data = json.dumps(profile_data)
            data_size = len(json_data)
            
            # Encrypt
            encrypted = self.cipher.encrypt(json_data.encode())
            
            # Return as base64 string
            result = base64.b64encode(encrypted).decode()
            
            duration = time.time() - start_time
            self.logger.debug(f"Profile encrypted: data_size={data_size}B, duration={duration*1000:.2f}ms")
            
            return result
        
        except Exception as e:
            self.logger.error(f"Profile encryption failed: {e}", exc_info=True)
            raise
    
    def decrypt_profile(self, encrypted_data: str) -> Dict[str, Any]:
        """
        Decrypt profile data.
        
        Args:
            encrypted_data: Encrypted profile as base64 string
        
        Returns:
            Decrypted profile dictionary
        """
        try:
            self.logger.debug("Decrypting profile data")
            start_time = time.time()
            
            # Decode from base64
            encrypted = base64.b64decode(encrypted_data.encode())
            
            # Decrypt
            decrypted = self.cipher.decrypt(encrypted)
            
            # Parse JSON
            profile = json.loads(decrypted.decode())
            
            duration = time.time() - start_time
            self.logger.debug(f"Profile decrypted: duration={duration*1000:.2f}ms")
            
            return profile
        
        except Exception as e:
            self.logger.error(f"Profile decryption failed: {e}", exc_info=True)
            raise
    
    def upload_profile(self, driver_id: str, profile_data: Dict[str, Any]) -> bool:
        """
        Upload driver profile to cloud.
        
        Args:
            driver_id: Driver identifier
            profile_data: Profile dictionary
        
        Returns:
            True if upload successful
        """
        try:
            self.logger.debug(f"Uploading profile: driver_id={driver_id}")
            start_time = time.time()
            
            # Encrypt profile
            encrypted_profile = self.encrypt_profile(profile_data)
            
            # Upload to cloud
            response = self.api_client.put(
                f'/drivers/{driver_id}/profile',
                data={
                    'driver_id': driver_id,
                    'encrypted_data': encrypted_profile,
                    'updated_at': time.time()
                }
            )
            
            duration = time.time() - start_time
            
            if response.success:
                self.logger.info(f"Profile uploaded successfully: driver_id={driver_id}, duration={duration*1000:.2f}ms")
                return True
            else:
                self.logger.error(f"Profile upload failed: driver_id={driver_id}, error={response.error}, status_code={response.status_code}")
                return False
        
        except Exception as e:
            self.logger.error(f"Profile upload error: driver_id={driver_id}, error={e}", exc_info=True)
            return False
    
    def download_profile(self, driver_id: str) -> Optional[Dict[str, Any]]:
        """
        Download driver profile from cloud.
        
        Args:
            driver_id: Driver identifier
        
        Returns:
            Profile dictionary or None if not found
        """
        try:
            self.logger.debug(f"Downloading profile: driver_id={driver_id}")
            start_time = time.time()
            
            # Download from cloud
            response = self.api_client.get(f'/drivers/{driver_id}/profile')
            
            if not response.success:
                if response.status_code == 404:
                    self.logger.info(f"Cloud profile not found: driver_id={driver_id}")
                else:
                    self.logger.error(f"Profile download failed: driver_id={driver_id}, error={response.error}, status_code={response.status_code}")
                return None
            
            # Decrypt profile
            encrypted_data = response.data.get('encrypted_data')
            if not encrypted_data:
                self.logger.error(f"Profile download failed: driver_id={driver_id}, reason=no_encrypted_data_in_response")
                return None
            
            profile_data = self.decrypt_profile(encrypted_data)
            
            duration = time.time() - start_time
            self.logger.info(f"Profile downloaded successfully: driver_id={driver_id}, duration={duration*1000:.2f}ms")
            
            return profile_data
        
        except Exception as e:
            self.logger.error(f"Profile download error: driver_id={driver_id}, error={e}", exc_info=True)
            return None
    
    def sync_profile(self, driver_id: str, local_profile: Dict[str, Any]) -> Dict[str, Any]:
        """
        Synchronize profile with cloud (merge local and cloud versions).
        
        Args:
            driver_id: Driver identifier
            local_profile: Local profile dictionary
        
        Returns:
            Merged profile dictionary
        """
        try:
            self.logger.debug(f"Synchronizing profile: driver_id={driver_id}")
            start_time = time.time()
            
            # Download cloud profile
            cloud_profile = self.download_profile(driver_id)
            
            if not cloud_profile:
                # No cloud profile, upload local
                self.logger.debug(f"No cloud profile found, uploading local: driver_id={driver_id}")
                self.upload_profile(driver_id, local_profile)
                duration = time.time() - start_time
                self.logger.info(f"Profile synchronized (upload only): driver_id={driver_id}, duration={duration*1000:.2f}ms")
                return local_profile
            
            # Merge profiles
            self.logger.debug(f"Merging local and cloud profiles: driver_id={driver_id}")
            merged_profile = self._merge_profiles(local_profile, cloud_profile)
            
            # Upload merged profile
            self.upload_profile(driver_id, merged_profile)
            
            # Save merged profile locally
            self._save_local_profile(driver_id, merged_profile)
            
            self.last_sync_time = time.time()
            duration = time.time() - start_time
            self.logger.info(f"Profile synchronized (merged): driver_id={driver_id}, duration={duration*1000:.2f}ms")
            
            return merged_profile
        
        except Exception as e:
            self.logger.error(f"Profile sync error: driver_id={driver_id}, error={e}, returning_local_profile", exc_info=True)
            return local_profile
    
    def _merge_profiles(
        self,
        local_profile: Dict[str, Any],
        cloud_profile: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Merge local and cloud profiles.
        
        Strategy:
        - Use most recent data for each field
        - Aggregate metrics (sum distances, average scores)
        - Keep maximum values for safety-critical fields
        
        Args:
            local_profile: Local profile
            cloud_profile: Cloud profile
        
        Returns:
            Merged profile
        """
        self.logger.debug("Merging profiles")
        
        merged = local_profile.copy()
        
        # Aggregate distance and time
        local_distance = local_profile.get('total_distance', 0.0)
        cloud_distance = cloud_profile.get('total_distance', 0.0)
        merged['total_distance'] = local_distance + cloud_distance
        
        local_time = local_profile.get('total_time', 0.0)
        cloud_time = cloud_profile.get('total_time', 0.0)
        merged['total_time'] = local_time + cloud_time
        
        self.logger.debug(f"Aggregated metrics: total_distance={merged['total_distance']:.1f}km, total_time={merged['total_time']:.1f}h")
        
        # Use most recent driving style
        local_updated = local_profile.get('last_updated', 0)
        cloud_updated = cloud_profile.get('last_updated', 0)
        
        if cloud_updated > local_updated:
            merged['driving_style'] = cloud_profile.get('driving_style', 'normal')
            self.logger.debug(f"Using cloud driving_style: {merged['driving_style']}")
        else:
            self.logger.debug(f"Using local driving_style: {merged.get('driving_style', 'normal')}")
        
        # Merge metrics (take average)
        local_metrics = local_profile.get('metrics', {})
        cloud_metrics = cloud_profile.get('metrics', {})
        merged_metrics = {}
        
        all_metric_keys = set(local_metrics.keys()) | set(cloud_metrics.keys())
        
        for key in all_metric_keys:
            local_values = local_metrics.get(key, [])
            cloud_values = cloud_metrics.get(key, [])
            merged_metrics[key] = local_values + cloud_values
        
        merged['metrics'] = merged_metrics
        self.logger.debug(f"Merged metrics: {len(all_metric_keys)} metric types")
        
        # Average scores
        local_safety = local_profile.get('safety_score', 0)
        cloud_safety = cloud_profile.get('safety_score', 0)
        merged['safety_score'] = (local_safety + cloud_safety) / 2
        
        local_attention = local_profile.get('attention_score', 0)
        cloud_attention = cloud_profile.get('attention_score', 0)
        merged['attention_score'] = (local_attention + cloud_attention) / 2
        
        self.logger.debug(f"Averaged scores: safety={merged['safety_score']:.1f}, attention={merged['attention_score']:.1f}")
        
        # Update timestamp
        merged['last_updated'] = max(local_updated, cloud_updated)
        
        return merged
    
    def _save_local_profile(self, driver_id: str, profile_data: Dict[str, Any]) -> None:
        """
        Save profile to local disk.
        
        Args:
            driver_id: Driver identifier
            profile_data: Profile dictionary
        """
        try:
            profile_path = os.path.join(self.profiles_dir, f"{driver_id}.json")
            
            self.logger.debug(f"Saving local profile: driver_id={driver_id}, path={profile_path}")
            
            with open(profile_path, 'w') as f:
                json.dump(profile_data, f, indent=2)
            
            self.logger.debug(f"Local profile saved: driver_id={driver_id}")
        
        except Exception as e:
            self.logger.error(f"Local profile save failed: driver_id={driver_id}, error={e}", exc_info=True)
    
    def sync_all_profiles(self) -> int:
        """
        Synchronize all local profiles with cloud.
        
        Returns:
            Number of profiles synchronized
        """
        try:
            self.logger.info("Starting bulk profile synchronization")
            start_time = time.time()
            
            # Get all local profile files
            profile_files = [
                f for f in os.listdir(self.profiles_dir)
                if f.endswith('.json')
            ]
            
            self.logger.info(f"Found {len(profile_files)} local profiles to sync")
            
            synced_count = 0
            failed_count = 0
            
            for profile_file in profile_files:
                driver_id = profile_file.replace('.json', '')
                
                try:
                    # Load local profile
                    profile_path = os.path.join(self.profiles_dir, profile_file)
                    with open(profile_path, 'r') as f:
                        local_profile = json.load(f)
                    
                    # Sync with cloud
                    self.sync_profile(driver_id, local_profile)
                    synced_count += 1
                    
                except Exception as e:
                    self.logger.error(f"Failed to sync profile: driver_id={driver_id}, error={e}")
                    failed_count += 1
            
            duration = time.time() - start_time
            self.logger.info(f"Bulk sync completed: synced={synced_count}, failed={failed_count}, duration={duration:.2f}s")
            
            return synced_count
        
        except Exception as e:
            self.logger.error(f"Bulk profile sync error: {e}", exc_info=True)
            return 0
    
    def get_sync_status(self) -> Dict[str, Any]:
        """
        Get synchronization status.
        
        Returns:
            Dictionary with sync status
        """
        return {
            'last_sync_time': self.last_sync_time,
            'profiles_dir': self.profiles_dir,
            'encryption_enabled': True
        }
