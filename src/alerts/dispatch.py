"""Multi-modal alert dispatch for SENTINEL system."""

import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

from ..core.data_structures import Alert


class AlertDispatcher:
    """Dispatches alerts through multiple modalities (visual, audio, haptic)."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize alert dispatcher.
        
        Args:
            config: Alert configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger('sentinel.alerts.dispatch')
        
        # Visual settings
        self.visual_config = config.get('modalities', {}).get('visual', {})
        self.display_duration = self.visual_config.get('display_duration', 3.0)
        self.flash_rate = self.visual_config.get('flash_rate', 2)
        
        # Audio settings
        self.audio_config = config.get('modalities', {}).get('audio', {})
        self.volume = self.audio_config.get('volume', 0.8)
        self.critical_sound = self.audio_config.get('critical_sound', 'sounds/alarm.wav')
        self.warning_sound = self.audio_config.get('warning_sound', 'sounds/beep.wav')
        
        # Haptic settings
        self.haptic_config = config.get('modalities', {}).get('haptic', {})
        self.haptic_enabled = self.haptic_config.get('enabled', False)
        
        # Track active visual alerts for rendering
        self.active_visual_alerts: List[Dict[str, Any]] = []
    
    def dispatch_alerts(self, alerts: List[Alert]) -> None:
        """
        Dispatch alerts through appropriate modalities.
        
        Args:
            alerts: List of alerts to dispatch
        """
        for alert in alerts:
            self._dispatch_alert(alert)
    
    def _dispatch_alert(self, alert: Alert) -> None:
        """
        Dispatch a single alert through its specified modalities.
        
        Args:
            alert: Alert to dispatch
        """
        self.logger.info(f"Dispatching {alert.urgency} alert: {alert.message}")
        
        for modality in alert.modalities:
            if modality == 'visual':
                self._dispatch_visual(alert)
            elif modality == 'audio':
                self._dispatch_audio(alert)
            elif modality == 'haptic':
                self._dispatch_haptic(alert)
            else:
                self.logger.warning(f"Unknown modality: {modality}")
    
    def _dispatch_visual(self, alert: Alert) -> None:
        """
        Dispatch visual alert.
        
        Args:
            alert: Alert to display visually
        """
        # Create visual alert data
        visual_data = {
            'alert': alert,
            'display_duration': self.display_duration,
            'flash_rate': self.flash_rate if alert.urgency == 'critical' else 0,
            'color': self._get_alert_color(alert.urgency),
            'position': self._get_alert_position(alert)
        }
        
        # Add to active visual alerts
        self.active_visual_alerts.append(visual_data)
        
        self.logger.debug(f"Visual alert added: {alert.message}")
    
    def _dispatch_audio(self, alert: Alert) -> None:
        """
        Dispatch audio alert.
        
        Args:
            alert: Alert to play audio for
        """
        # Determine sound file based on urgency
        if alert.urgency == 'critical':
            sound_file = self.critical_sound
        elif alert.urgency == 'warning':
            sound_file = self.warning_sound
        else:
            sound_file = None  # No audio for info alerts by default
        
        if sound_file:
            # In a real implementation, this would play the audio file
            # For now, we just log it
            self.logger.info(f"Audio alert: {sound_file} at volume {self.volume}")
            
            # Placeholder for actual audio playback
            # self._play_audio(sound_file, self.volume)
        else:
            self.logger.debug(f"No audio configured for {alert.urgency} alert")
    
    def _dispatch_haptic(self, alert: Alert) -> None:
        """
        Dispatch haptic alert.
        
        Args:
            alert: Alert to provide haptic feedback for
        """
        if not self.haptic_enabled:
            self.logger.debug("Haptic feedback disabled")
            return
        
        # Determine haptic pattern based on urgency
        if alert.urgency == 'critical':
            pattern = 'pulse'  # Strong pulsing
            intensity = 1.0
        elif alert.urgency == 'warning':
            pattern = 'tap'  # Single tap
            intensity = 0.7
        else:
            pattern = 'gentle'  # Gentle vibration
            intensity = 0.5
        
        # In a real implementation, this would trigger haptic hardware
        # For now, we just log it
        self.logger.info(f"Haptic alert: {pattern} at intensity {intensity}")
        
        # Placeholder for actual haptic feedback
        # self._trigger_haptic(pattern, intensity)
    
    def _get_alert_color(self, urgency: str) -> str:
        """
        Get color for alert based on urgency.
        
        Args:
            urgency: Alert urgency level
            
        Returns:
            Color code (hex or name)
        """
        color_map = {
            'critical': '#FF0000',  # Red
            'warning': '#FFA500',   # Orange
            'info': '#00BFFF'       # Blue
        }
        return color_map.get(urgency, '#FFFFFF')
    
    def _get_alert_position(self, alert: Alert) -> str:
        """
        Determine screen position for alert based on context.
        
        Args:
            alert: Alert object
            
        Returns:
            Position identifier
        """
        # Position based on urgency
        if alert.urgency == 'critical':
            return 'center'  # Center of screen for critical
        elif alert.urgency == 'warning':
            return 'top'  # Top of screen for warnings
        else:
            return 'bottom'  # Bottom for info
    
    def get_active_visual_alerts(self) -> List[Dict[str, Any]]:
        """
        Get currently active visual alerts for rendering.
        
        Returns:
            List of active visual alert data
        """
        return self.active_visual_alerts.copy()
    
    def clear_visual_alerts(self) -> None:
        """Clear all active visual alerts."""
        self.active_visual_alerts.clear()
    
    def remove_expired_visual_alerts(self, current_time: float) -> None:
        """
        Remove visual alerts that have exceeded their display duration.
        
        Args:
            current_time: Current timestamp
        """
        self.active_visual_alerts = [
            alert_data for alert_data in self.active_visual_alerts
            if current_time - alert_data['alert'].timestamp < alert_data['display_duration']
        ]
