"""
Offline operation manager for cloud synchronization.

Handles offline queueing, connectivity detection, and automatic sync
when connection is restored.
"""

import logging
import threading
import time
from typing import Callable, Optional, List
from enum import Enum
import socket

from .api_client import CloudAPIClient


class ConnectivityStatus(Enum):
    """Connectivity status"""
    ONLINE = "online"
    OFFLINE = "offline"
    UNKNOWN = "unknown"


class OfflineManager:
    """
    Manages offline operation and automatic synchronization.
    
    Features:
    - Connectivity monitoring
    - Offline operation queueing
    - Automatic sync when online
    - Status notifications
    """
    
    def __init__(
        self,
        api_client: CloudAPIClient,
        check_interval: float = 30.0,
        connectivity_callback: Optional[Callable[[ConnectivityStatus], None]] = None
    ):
        """
        Initialize offline manager.
        
        Args:
            api_client: Cloud API client
            check_interval: Connectivity check interval in seconds
            connectivity_callback: Callback function for connectivity changes
        """
        self.api_client = api_client
        self.check_interval = check_interval
        self.connectivity_callback = connectivity_callback
        
        self.logger = logging.getLogger(__name__)
        
        # Current connectivity status
        self.status = ConnectivityStatus.UNKNOWN
        self.last_online_time = 0.0
        
        # Background monitoring thread
        self.running = False
        self.monitor_thread: Optional[threading.Thread] = None
        
        # Sync callbacks (called when connectivity is restored)
        self.sync_callbacks: List[Callable[[], None]] = []
        
        self.logger.info("OfflineManager initialized")
    
    def start_monitoring(self) -> None:
        """Start background connectivity monitoring"""
        if self.running:
            return
        
        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        self.logger.info("Background connectivity monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop background connectivity monitoring"""
        if not self.running:
            return
        
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
        self.logger.info("Background connectivity monitoring stopped")
    
    def _monitor_loop(self) -> None:
        """Background connectivity monitoring loop"""
        while self.running:
            try:
                # Check connectivity
                previous_status = self.status
                current_status = self._check_connectivity()
                
                # Update status
                if current_status != previous_status:
                    self.status = current_status
                    self.logger.info(f"Connectivity status changed: {previous_status.value} -> {current_status.value}")
                    
                    # Notify callback
                    if self.connectivity_callback:
                        try:
                            self.connectivity_callback(current_status)
                        except Exception as e:
                            self.logger.error(f"Error in connectivity callback: {e}", exc_info=True)
                    
                    # Trigger sync if coming back online
                    if current_status == ConnectivityStatus.ONLINE and previous_status == ConnectivityStatus.OFFLINE:
                        self._trigger_sync()
                
                # Update last online time
                if current_status == ConnectivityStatus.ONLINE:
                    self.last_online_time = time.time()
                
                # Sleep until next check
                time.sleep(self.check_interval)
            
            except Exception as e:
                self.logger.error(f"Error in monitor loop: {e}", exc_info=True)
                time.sleep(self.check_interval)
    
    def _check_connectivity(self) -> ConnectivityStatus:
        """
        Check connectivity to cloud backend.
        
        Returns:
            Current connectivity status
        """
        try:
            # First check internet connectivity
            if not self._check_internet():
                return ConnectivityStatus.OFFLINE
            
            # Then check API connectivity
            if self.api_client.check_connectivity():
                return ConnectivityStatus.ONLINE
            else:
                return ConnectivityStatus.OFFLINE
        
        except Exception as e:
            self.logger.error(f"Error checking connectivity: {e}", exc_info=True)
            return ConnectivityStatus.UNKNOWN
    
    def _check_internet(self) -> bool:
        """
        Check basic internet connectivity.
        
        Returns:
            True if internet is available
        """
        try:
            # Try to resolve a well-known hostname
            socket.create_connection(("8.8.8.8", 53), timeout=3)
            return True
        except OSError:
            return False
    
    def _trigger_sync(self) -> None:
        """Trigger synchronization callbacks"""
        self.logger.info("Triggering synchronization after coming online")
        
        for callback in self.sync_callbacks:
            try:
                callback()
            except Exception as e:
                self.logger.error(f"Error in sync callback: {e}", exc_info=True)
    
    def register_sync_callback(self, callback: Callable[[], None]) -> None:
        """
        Register a callback to be called when connectivity is restored.
        
        Args:
            callback: Function to call on sync
        """
        self.sync_callbacks.append(callback)
        callback_name = getattr(callback, '__name__', 'unknown')
        self.logger.debug(f"Registered sync callback: {callback_name}")
    
    def is_online(self) -> bool:
        """
        Check if currently online.
        
        Returns:
            True if online
        """
        return self.status == ConnectivityStatus.ONLINE
    
    def is_offline(self) -> bool:
        """
        Check if currently offline.
        
        Returns:
            True if offline
        """
        return self.status == ConnectivityStatus.OFFLINE
    
    def get_status(self) -> ConnectivityStatus:
        """
        Get current connectivity status.
        
        Returns:
            Current status
        """
        return self.status
    
    def get_offline_duration(self) -> float:
        """
        Get duration of current offline period.
        
        Returns:
            Offline duration in seconds (0 if online)
        """
        if self.status == ConnectivityStatus.ONLINE:
            return 0.0
        
        if self.last_online_time == 0.0:
            return 0.0
        
        return time.time() - self.last_online_time
    
    def force_sync(self) -> bool:
        """
        Force synchronization attempt.
        
        Returns:
            True if sync was triggered
        """
        if not self.is_online():
            self.logger.warning("Cannot force sync while offline")
            return False
        
        self._trigger_sync()
        return True
    
    def get_sync_status_message(self) -> str:
        """
        Get human-readable sync status message.
        
        Returns:
            Status message
        """
        if self.status == ConnectivityStatus.ONLINE:
            return "Connected to cloud"
        elif self.status == ConnectivityStatus.OFFLINE:
            offline_duration = self.get_offline_duration()
            if offline_duration > 0:
                minutes = int(offline_duration / 60)
                if minutes > 0:
                    return f"Offline for {minutes} minutes"
                else:
                    return f"Offline for {int(offline_duration)} seconds"
            else:
                return "Offline"
        else:
            return "Connectivity unknown"
