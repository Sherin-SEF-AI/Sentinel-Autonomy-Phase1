"""
Warning Animations

Utilities for creating warning animations and effects.
"""

import logging
from typing import Optional
from PyQt6.QtWidgets import QWidget, QGraphicsOpacityEffect
from PyQt6.QtCore import QTimer, QPropertyAnimation, QEasingCurve, QSequentialAnimationGroup, pyqtProperty
from PyQt6.QtGui import QColor
from PyQt6.QtMultimedia import QSoundEffect
from PyQt6.QtCore import QUrl

logger = logging.getLogger(__name__)


class WarningAnimationManager:
    """
    Manager for warning animations and effects.
    
    Features:
    - Pulsing effects for threshold crossings
    - Flash indicator colors
    - Sound effect triggers
    """
    
    def __init__(self):
        # Sound effects
        self._sound_effects = {}
        self._sounds_enabled = True
        
        logger.debug("WarningAnimationManager created")
    
    def load_sound(self, name: str, file_path: str):
        """
        Load a sound effect.
        
        Args:
            name: Identifier for the sound
            file_path: Path to the sound file
        """
        try:
            sound = QSoundEffect()
            sound.setSource(QUrl.fromLocalFile(file_path))
            sound.setVolume(0.5)
            self._sound_effects[name] = sound
            logger.debug(f"Loaded sound: {name} from {file_path}")
        except Exception as e:
            logger.error(f"Failed to load sound {name}: {e}")
    
    def play_sound(self, name: str):
        """
        Play a sound effect.
        
        Args:
            name: Identifier of the sound to play
        """
        if not self._sounds_enabled:
            return
        
        if name in self._sound_effects:
            try:
                self._sound_effects[name].play()
                logger.debug(f"Playing sound: {name}")
            except Exception as e:
                logger.error(f"Failed to play sound {name}: {e}")
        else:
            logger.warning(f"Sound not found: {name}")
    
    def set_sounds_enabled(self, enabled: bool):
        """Enable or disable sound effects"""
        self._sounds_enabled = enabled
        logger.debug(f"Sounds {'enabled' if enabled else 'disabled'}")
    
    def create_pulse_animation(
        self,
        widget: QWidget,
        duration: int = 1000,
        min_opacity: float = 0.3,
        max_opacity: float = 1.0,
        loop_count: int = -1
    ) -> QPropertyAnimation:
        """
        Create a pulsing opacity animation for a widget.
        
        Args:
            widget: Widget to animate
            duration: Animation duration in milliseconds
            min_opacity: Minimum opacity value
            max_opacity: Maximum opacity value
            loop_count: Number of loops (-1 for infinite)
        
        Returns:
            QPropertyAnimation object
        """
        # Create opacity effect if not exists
        if not widget.graphicsEffect():
            opacity_effect = QGraphicsOpacityEffect()
            widget.setGraphicsEffect(opacity_effect)
        
        opacity_effect = widget.graphicsEffect()
        
        # Create animation
        animation = QPropertyAnimation(opacity_effect, b"opacity")
        animation.setDuration(duration)
        animation.setStartValue(max_opacity)
        animation.setEndValue(min_opacity)
        animation.setEasingCurve(QEasingCurve.Type.InOutSine)
        animation.setLoopCount(loop_count)
        
        logger.debug(f"Created pulse animation for widget")
        
        return animation
    
    def create_flash_animation(
        self,
        widget: QWidget,
        flash_count: int = 3,
        flash_duration: int = 200
    ) -> QSequentialAnimationGroup:
        """
        Create a flashing animation for a widget.
        
        Args:
            widget: Widget to animate
            flash_count: Number of flashes
            flash_duration: Duration of each flash in milliseconds
        
        Returns:
            QSequentialAnimationGroup object
        """
        # Create opacity effect if not exists
        if not widget.graphicsEffect():
            opacity_effect = QGraphicsOpacityEffect()
            widget.setGraphicsEffect(opacity_effect)
        
        opacity_effect = widget.graphicsEffect()
        
        # Create animation group
        animation_group = QSequentialAnimationGroup()
        
        for _ in range(flash_count):
            # Flash off
            fade_out = QPropertyAnimation(opacity_effect, b"opacity")
            fade_out.setDuration(flash_duration // 2)
            fade_out.setStartValue(1.0)
            fade_out.setEndValue(0.2)
            animation_group.addAnimation(fade_out)
            
            # Flash on
            fade_in = QPropertyAnimation(opacity_effect, b"opacity")
            fade_in.setDuration(flash_duration // 2)
            fade_in.setStartValue(0.2)
            fade_in.setEndValue(1.0)
            animation_group.addAnimation(fade_in)
        
        logger.debug(f"Created flash animation for widget")
        
        return animation_group
    
    def trigger_warning_animation(
        self,
        widget: QWidget,
        severity: str = 'warning',
        play_sound: bool = True
    ):
        """
        Trigger a warning animation on a widget.
        
        Args:
            widget: Widget to animate
            severity: 'warning' or 'critical'
            play_sound: Whether to play a sound effect
        """
        if severity == 'critical':
            # Critical: flash animation + sound
            animation = self.create_flash_animation(widget, flash_count=5, flash_duration=150)
            animation.start()
            
            if play_sound:
                self.play_sound('critical_alert')
        
        elif severity == 'warning':
            # Warning: pulse animation + sound
            animation = self.create_pulse_animation(widget, duration=800, loop_count=3)
            animation.start()
            
            if play_sound:
                self.play_sound('warning_alert')
        
        logger.debug(f"Triggered {severity} animation")


class ThresholdMonitor:
    """
    Monitor values for threshold crossings and trigger animations.
    
    Monitors metric values and triggers warning animations when
    thresholds are crossed.
    """
    
    def __init__(self, animation_manager: WarningAnimationManager):
        self._animation_manager = animation_manager
        self._previous_values = {}
        self._threshold_states = {}
        
        logger.debug("ThresholdMonitor created")
    
    def register_metric(
        self,
        key: str,
        warning_threshold: float,
        critical_threshold: float,
        reverse: bool = False
    ):
        """
        Register a metric to monitor.
        
        Args:
            key: Unique identifier for the metric
            warning_threshold: Threshold for warning state
            critical_threshold: Threshold for critical state
            reverse: True if lower values are worse
        """
        self._threshold_states[key] = {
            'warning_threshold': warning_threshold,
            'critical_threshold': critical_threshold,
            'reverse': reverse,
            'current_state': 'ok'
        }
        self._previous_values[key] = None
        
        logger.debug(f"Registered metric: {key}")
    
    def update_metric(
        self,
        key: str,
        value: float,
        widget: Optional[QWidget] = None
    ) -> str:
        """
        Update a metric value and check for threshold crossings.
        
        Args:
            key: Metric identifier
            value: New value
            widget: Optional widget to animate on threshold crossing
        
        Returns:
            Current state: 'ok', 'warning', or 'critical'
        """
        if key not in self._threshold_states:
            logger.warning(f"Metric not registered: {key}")
            return 'ok'
        
        config = self._threshold_states[key]
        previous_state = config['current_state']
        
        # Determine new state
        if config['reverse']:
            # Lower is better
            if value >= config['critical_threshold']:
                new_state = 'critical'
            elif value >= config['warning_threshold']:
                new_state = 'warning'
            else:
                new_state = 'ok'
        else:
            # Higher is better
            if value <= config['critical_threshold']:
                new_state = 'critical'
            elif value <= config['warning_threshold']:
                new_state = 'warning'
            else:
                new_state = 'ok'
        
        # Check for state change
        if new_state != previous_state:
            logger.info(f"Metric {key} changed from {previous_state} to {new_state}")
            
            # Trigger animation if widget provided and state worsened
            if widget and new_state in ['warning', 'critical']:
                if previous_state == 'ok' or (previous_state == 'warning' and new_state == 'critical'):
                    self._animation_manager.trigger_warning_animation(
                        widget,
                        severity=new_state,
                        play_sound=True
                    )
            
            config['current_state'] = new_state
        
        self._previous_values[key] = value
        
        return new_state
    
    def get_state(self, key: str) -> str:
        """Get current state for a metric"""
        if key in self._threshold_states:
            return self._threshold_states[key]['current_state']
        return 'ok'
    
    def reset_metric(self, key: str):
        """Reset a metric to OK state"""
        if key in self._threshold_states:
            self._threshold_states[key]['current_state'] = 'ok'
            self._previous_values[key] = None
    
    def reset_all(self):
        """Reset all metrics to OK state"""
        for key in self._threshold_states:
            self.reset_metric(key)
