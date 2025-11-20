"""
Alerts Panel Widget

Displays real-time alerts with color coding, timestamps, and icons.
Provides controls for managing alert history and settings.
"""

import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTextEdit, QPushButton,
    QLabel, QComboBox, QSpinBox, QCheckBox, QGroupBox, QFileDialog,
    QMessageBox
)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer
from PyQt6.QtGui import QTextCursor, QFont, QColor
from PyQt6.QtMultimedia import QSoundEffect, QAudioOutput, QMediaPlayer
from PyQt6.QtCore import QUrl

from core.data_structures import Alert

logger = logging.getLogger(__name__)


class AlertsPanel(QWidget):
    """
    Panel for displaying and managing alerts.
    
    Features:
    - Color-coded alert display with HTML formatting
    - Timestamps and urgency icons
    - Auto-scroll to latest alert
    - Audio alerts with mute control
    - Alert history management
    - Statistics display
    - Critical alert effects
    """
    
    # Signals
    alert_dismissed = pyqtSignal(int)  # Alert ID
    false_positive_marked = pyqtSignal(int)  # Alert ID
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.logger = logging.getLogger('sentinel.gui.alerts_panel')
        self.logger.debug("AlertsPanel initialization started")
        
        # Alert history
        self.alert_history: List[Dict[str, Any]] = []
        self.alert_counter = 0
        
        # Statistics
        self.total_alerts = 0
        self.critical_alerts = 0
        self.warning_alerts = 0
        self.info_alerts = 0
        self.false_positives = 0
        
        # Audio settings
        self.audio_enabled = True
        self.audio_volume = 0.8
        self.logger.debug(f"Audio settings initialized: enabled={self.audio_enabled}, volume={self.audio_volume}")
        
        # Audio players
        self.audio_players: Dict[str, QMediaPlayer] = {}
        self._init_audio()
        
        # Critical alert effects
        self.flash_timer = QTimer()
        self.flash_timer.timeout.connect(self._flash_effect)
        self.flash_state = False
        self.logger.debug("Flash timer initialized for critical alerts")
        
        # Initialize UI
        self._init_ui()
        
        self.logger.info("AlertsPanel initialized successfully")
    
    def _init_audio(self):
        """Initialize audio players for different alert types"""
        self.logger.debug("Initializing audio players for alert types")
        
        # Create audio players for each urgency level
        for urgency in ['critical', 'warning', 'info']:
            try:
                player = QMediaPlayer()
                audio_output = QAudioOutput()
                player.setAudioOutput(audio_output)
                audio_output.setVolume(self.audio_volume)
                self.audio_players[urgency] = player
                self.logger.debug(f"Audio player created for urgency level: {urgency}")
            except Exception as e:
                self.logger.error(f"Failed to create audio player for {urgency}: {e}")
        
        self.logger.info(f"Audio players initialized: {len(self.audio_players)} players created")
    
    def _init_ui(self):
        """Initialize the user interface"""
        self.logger.debug("Initializing AlertsPanel UI components")
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(5)
        
        # Title
        title_label = QLabel("Alerts")
        title_font = QFont()
        title_font.setPointSize(12)
        title_font.setBold(True)
        title_label.setFont(title_font)
        layout.addWidget(title_label)
        self.logger.debug("Title label created")
        
        # Statistics panel
        self.stats_group = self._create_statistics_panel()
        layout.addWidget(self.stats_group)
        self.logger.debug("Statistics panel created")
        
        # Alert display
        self.alert_display = QTextEdit()
        self.alert_display.setReadOnly(True)
        self.alert_display.setMinimumHeight(300)
        self.alert_display.setHtml(self._get_welcome_message())
        layout.addWidget(self.alert_display, stretch=1)
        self.logger.debug("Alert display widget created with minimum height 300px")
        
        # Controls panel
        controls_layout = self._create_controls_panel()
        layout.addLayout(controls_layout)
        self.logger.debug("Controls panel created")
        
        self.logger.info("AlertsPanel UI initialization completed")
    
    def _create_statistics_panel(self) -> QGroupBox:
        """Create statistics display panel"""
        group = QGroupBox("Statistics")
        layout = QHBoxLayout(group)
        
        # Total alerts
        self.total_label = QLabel("Total: 0")
        layout.addWidget(self.total_label)
        
        # Critical alerts
        self.critical_label = QLabel("Critical: 0")
        self.critical_label.setStyleSheet("color: #ff4444;")
        layout.addWidget(self.critical_label)
        
        # Warning alerts
        self.warning_label = QLabel("Warning: 0")
        self.warning_label.setStyleSheet("color: #ffaa00;")
        layout.addWidget(self.warning_label)
        
        # Info alerts
        self.info_label = QLabel("Info: 0")
        self.info_label.setStyleSheet("color: #4488ff;")
        layout.addWidget(self.info_label)
        
        # False positives
        self.fp_label = QLabel("False Positives: 0")
        self.fp_label.setStyleSheet("color: #888888;")
        layout.addWidget(self.fp_label)
        
        layout.addStretch()
        
        return group
    
    def _create_controls_panel(self) -> QHBoxLayout:
        """Create controls panel"""
        layout = QHBoxLayout()
        
        # Audio mute button
        self.mute_button = QPushButton("🔊 Mute")
        self.mute_button.setCheckable(True)
        self.mute_button.setMaximumWidth(100)
        self.mute_button.clicked.connect(self._on_mute_toggle)
        layout.addWidget(self.mute_button)
        
        # Volume control
        volume_label = QLabel("Volume:")
        layout.addWidget(volume_label)
        
        self.volume_spin = QSpinBox()
        self.volume_spin.setRange(0, 100)
        self.volume_spin.setValue(int(self.audio_volume * 100))
        self.volume_spin.setSuffix("%")
        self.volume_spin.setMaximumWidth(80)
        self.volume_spin.valueChanged.connect(self._on_volume_changed)
        layout.addWidget(self.volume_spin)
        
        layout.addStretch()
        
        # Filter combo
        filter_label = QLabel("Filter:")
        layout.addWidget(filter_label)
        
        self.filter_combo = QComboBox()
        self.filter_combo.addItems(["All", "Critical", "Warning", "Info"])
        self.filter_combo.currentTextChanged.connect(self._on_filter_changed)
        layout.addWidget(self.filter_combo)
        
        # Clear button
        self.clear_button = QPushButton("Clear History")
        self.clear_button.clicked.connect(self._on_clear_history)
        layout.addWidget(self.clear_button)
        
        # Export button
        self.export_button = QPushButton("Export Log")
        self.export_button.clicked.connect(self._on_export_log)
        layout.addWidget(self.export_button)
        
        return layout
    
    def _get_welcome_message(self) -> str:
        """Get welcome message HTML"""
        return """
        <div style='text-align: center; padding: 20px; color: #888;'>
            <h3>Alert Monitor</h3>
            <p>No alerts yet. System is monitoring for hazards.</p>
        </div>
        """
    
    def add_alert(self, alert: Alert):
        """
        Add a new alert to the display.
        
        Args:
            alert: Alert object to display
        """
        self.logger.debug(f"Adding alert: urgency={alert.urgency}, hazard_id={alert.hazard_id}, modalities={alert.modalities}")
        
        # Update statistics
        self.total_alerts += 1
        if alert.urgency == 'critical':
            self.critical_alerts += 1
        elif alert.urgency == 'warning':
            self.warning_alerts += 1
        elif alert.urgency == 'info':
            self.info_alerts += 1
        
        self._update_statistics()
        self.logger.debug(f"Statistics updated: total={self.total_alerts}, critical={self.critical_alerts}, warning={self.warning_alerts}, info={self.info_alerts}")
        
        # Create alert entry
        alert_id = self.alert_counter
        self.alert_counter += 1
        
        alert_entry = {
            'id': alert_id,
            'alert': alert,
            'timestamp': datetime.fromtimestamp(alert.timestamp),
            'false_positive': False
        }
        
        self.alert_history.append(alert_entry)
        self.logger.debug(f"Alert entry created: id={alert_id}, history_size={len(self.alert_history)}")
        
        # Add to display
        self._add_alert_to_display(alert_entry)
        
        # Play audio
        if self.audio_enabled and 'audio' in alert.modalities:
            self._play_alert_sound(alert.urgency)
            self.logger.debug(f"Audio alert triggered for urgency: {alert.urgency}")
        elif not self.audio_enabled:
            self.logger.debug("Audio alert skipped: audio disabled")
        elif 'audio' not in alert.modalities:
            self.logger.debug("Audio alert skipped: not in modalities")
        
        # Critical alert effects
        if alert.urgency == 'critical':
            self._trigger_critical_effects()
            self.logger.debug("Critical alert effects triggered")
        
        self.logger.info(f"Alert added successfully: urgency={alert.urgency}, message='{alert.message}', hazard_id={alert.hazard_id}")
    
    def _add_alert_to_display(self, alert_entry: Dict[str, Any]):
        """Add alert entry to the text display"""
        alert = alert_entry['alert']
        timestamp = alert_entry['timestamp']
        alert_id = alert_entry['id']
        
        self.logger.debug(f"Adding alert to display: id={alert_id}, urgency={alert.urgency}")
        
        # Get color based on urgency
        if alert.urgency == 'critical':
            color = '#ff4444'
            icon = '🚨'
            bg_color = '#331111'
        elif alert.urgency == 'warning':
            color = '#ffaa00'
            icon = '⚠️'
            bg_color = '#332200'
        else:  # info
            color = '#4488ff'
            icon = 'ℹ️'
            bg_color = '#112233'
        
        # Format timestamp
        time_str = timestamp.strftime('%H:%M:%S')
        
        # Create HTML for alert
        html = f"""
        <div style='background-color: {bg_color}; padding: 8px; margin: 4px 0; border-left: 4px solid {color}; border-radius: 4px;'>
            <div style='display: flex; align-items: center;'>
                <span style='font-size: 20px; margin-right: 8px;'>{icon}</span>
                <div style='flex: 1;'>
                    <div style='color: {color}; font-weight: bold; font-size: 11pt;'>
                        {alert.urgency.upper()} - {time_str}
                    </div>
                    <div style='color: #ffffff; margin-top: 4px;'>
                        {alert.message}
                    </div>
                    <div style='color: #888888; font-size: 9pt; margin-top: 4px;'>
                        Hazard ID: {alert.hazard_id} | Modalities: {', '.join(alert.modalities)}
                    </div>
                </div>
            </div>
        </div>
        """
        
        # Append to display
        cursor = self.alert_display.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        self.alert_display.setTextCursor(cursor)
        self.alert_display.insertHtml(html)
        
        # Auto-scroll to bottom
        scrollbar = self.alert_display.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
        
        self.logger.debug(f"Alert displayed successfully: id={alert_id}, time={time_str}")
    
    def _update_statistics(self):
        """Update statistics labels"""
        self.total_label.setText(f"Total: {self.total_alerts}")
        self.critical_label.setText(f"Critical: {self.critical_alerts}")
        self.warning_label.setText(f"Warning: {self.warning_alerts}")
        self.info_label.setText(f"Info: {self.info_alerts}")
        self.fp_label.setText(f"False Positives: {self.false_positives}")
    
    def _play_alert_sound(self, urgency: str):
        """Play alert sound based on urgency"""
        self.logger.debug(f"Attempting to play alert sound: urgency={urgency}")
        
        if urgency in self.audio_players:
            player = self.audio_players[urgency]
            
            # Set sound file based on urgency
            # For now, we'll use system beep or placeholder
            # In production, load actual sound files
            sound_files = {
                'critical': 'sounds/alarm.wav',
                'warning': 'sounds/beep.wav',
                'info': 'sounds/notification.wav'
            }
            
            # Try to load and play sound file
            sound_file = sound_files.get(urgency)
            if sound_file:
                try:
                    url = QUrl.fromLocalFile(sound_file)
                    player.setSource(url)
                    player.play()
                    self.logger.info(f"Playing {urgency} alert sound from {sound_file}")
                except Exception as e:
                    self.logger.error(f"Failed to play sound file {sound_file}: {e}")
                    QApplication.beep()
            else:
                # Fallback: system beep
                self.logger.debug(f"No sound file for {urgency}, using system beep")
                QApplication.beep()
        else:
            self.logger.warning(f"No audio player found for urgency: {urgency}")
    
    def _trigger_critical_effects(self):
        """Trigger visual effects for critical alerts"""
        self.logger.debug("Triggering critical alert visual effects")
        
        # Flash window
        self.flash_state = False
        self.flash_timer.start(200)  # Flash every 200ms
        self.logger.debug("Flash timer started: interval=200ms")
        
        # Stop flashing after 2 seconds
        QTimer.singleShot(2000, self.flash_timer.stop)
        self.logger.debug("Flash timer scheduled to stop after 2000ms")
        
        # Bring window to front
        window = self.window()
        if window:
            window.raise_()
            window.activateWindow()
            self.logger.debug("Window brought to front and activated")
        else:
            self.logger.warning("Could not bring window to front: window is None")
        
        self.logger.info("Critical alert effects triggered successfully")
    
    def _flash_effect(self):
        """Flash effect for critical alerts"""
        if self.flash_state:
            # Normal background
            self.setStyleSheet("")
        else:
            # Red flash
            self.setStyleSheet("background-color: rgba(255, 0, 0, 0.2);")
        
        self.flash_state = not self.flash_state
    
    def _on_mute_toggle(self, checked: bool):
        """Handle mute button toggle"""
        self.audio_enabled = not checked
        
        if checked:
            self.mute_button.setText("🔇 Unmute")
            self.logger.info("Audio alerts muted")
        else:
            self.mute_button.setText("🔊 Mute")
            self.logger.info("Audio alerts unmuted")
    
    def _on_volume_changed(self, value: int):
        """Handle volume change"""
        self.audio_volume = value / 100.0
        
        # Update all audio players
        for player in self.audio_players.values():
            if player.audioOutput():
                player.audioOutput().setVolume(self.audio_volume)
        
        self.logger.debug(f"Volume changed to {value}%")
    
    def _on_filter_changed(self, filter_text: str):
        """Handle filter change"""
        self.logger.info(f"Alert filter changed: {filter_text}")
        self._refresh_display()
    
    def _refresh_display(self):
        """Refresh the alert display based on current filter"""
        filter_text = self.filter_combo.currentText().lower()
        self.logger.debug(f"Refreshing display with filter: {filter_text}")
        
        # Clear display
        self.alert_display.clear()
        
        if not self.alert_history:
            self.alert_display.setHtml(self._get_welcome_message())
            self.logger.debug("Display refreshed: no alerts in history")
            return
        
        # Re-add filtered alerts
        filtered_count = 0
        for alert_entry in self.alert_history:
            alert = alert_entry['alert']
            
            # Apply filter
            if filter_text == 'all' or alert.urgency == filter_text:
                self._add_alert_to_display(alert_entry)
                filtered_count += 1
        
        self.logger.debug(f"Display refreshed: {filtered_count}/{len(self.alert_history)} alerts shown")
    
    def _on_clear_history(self):
        """Handle clear history button"""
        reply = QMessageBox.question(
            self,
            'Clear History',
            'Are you sure you want to clear all alert history?',
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            self.clear_history()
            self.logger.info("Alert history cleared by user")
    
    def _on_export_log(self):
        """Handle export log button"""
        if not self.alert_history:
            QMessageBox.information(
                self,
                'Export Log',
                'No alerts to export.'
            )
            return
        
        # Open file dialog
        filename, _ = QFileDialog.getSaveFileName(
            self,
            'Export Alert Log',
            f'alert_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt',
            'Text Files (*.txt);;All Files (*)'
        )
        
        if filename:
            self._export_to_file(filename)
    
    def _export_to_file(self, filename: str):
        """Export alert history to text file"""
        self.logger.info(f"Exporting alert log to file: {filename}")
        
        try:
            with open(filename, 'w') as f:
                f.write("SENTINEL Alert Log\n")
                f.write("=" * 80 + "\n\n")
                
                for alert_entry in self.alert_history:
                    alert = alert_entry['alert']
                    timestamp = alert_entry['timestamp']
                    fp_marker = " [FALSE POSITIVE]" if alert_entry['false_positive'] else ""
                    
                    f.write(f"[{timestamp.strftime('%Y-%m-%d %H:%M:%S')}] ")
                    f.write(f"{alert.urgency.upper()}{fp_marker}\n")
                    f.write(f"  Message: {alert.message}\n")
                    f.write(f"  Hazard ID: {alert.hazard_id}\n")
                    f.write(f"  Modalities: {', '.join(alert.modalities)}\n")
                    f.write("\n")
                
                f.write("\n" + "=" * 80 + "\n")
                f.write("Statistics:\n")
                f.write(f"  Total Alerts: {self.total_alerts}\n")
                f.write(f"  Critical: {self.critical_alerts}\n")
                f.write(f"  Warning: {self.warning_alerts}\n")
                f.write(f"  Info: {self.info_alerts}\n")
                f.write(f"  False Positives: {self.false_positives}\n")
            
            QMessageBox.information(
                self,
                'Export Successful',
                f'Alert log exported to:\n{filename}'
            )
            
            self.logger.info(f"Alert log exported successfully: {filename}, {len(self.alert_history)} alerts")
            
        except Exception as e:
            QMessageBox.critical(
                self,
                'Export Failed',
                f'Failed to export alert log:\n{str(e)}'
            )
            self.logger.error(f"Failed to export alert log to {filename}: {e}", exc_info=True)
    
    def mark_false_positive(self, alert_id: int):
        """Mark an alert as false positive"""
        self.logger.debug(f"Marking alert as false positive: id={alert_id}")
        
        for alert_entry in self.alert_history:
            if alert_entry['id'] == alert_id:
                alert_entry['false_positive'] = True
                self.false_positives += 1
                self._update_statistics()
                self.false_positive_marked.emit(alert_id)
                self.logger.info(f"Alert marked as false positive: id={alert_id}, total_fp={self.false_positives}")
                return
        
        self.logger.warning(f"Alert not found for false positive marking: id={alert_id}")
    
    def clear_history(self):
        """Clear all alert history"""
        self.logger.info(f"Clearing alert history: {len(self.alert_history)} alerts")
        
        previous_total = self.total_alerts
        
        self.alert_history.clear()
        self.alert_counter = 0
        self.total_alerts = 0
        self.critical_alerts = 0
        self.warning_alerts = 0
        self.info_alerts = 0
        self.false_positives = 0
        
        self._update_statistics()
        self.alert_display.clear()
        self.alert_display.setHtml(self._get_welcome_message())
        
        self.logger.info(f"Alert history cleared: {previous_total} alerts removed")
    
    def get_statistics(self) -> Dict[str, int]:
        """Get alert statistics"""
        return {
            'total': self.total_alerts,
            'critical': self.critical_alerts,
            'warning': self.warning_alerts,
            'info': self.info_alerts,
            'false_positives': self.false_positives
        }
    
    def set_audio_enabled(self, enabled: bool):
        """Set audio alerts enabled/disabled"""
        self.logger.info(f"Audio alerts {'enabled' if enabled else 'disabled'}")
        self.audio_enabled = enabled
        self.mute_button.setChecked(not enabled)
    
    def set_volume(self, volume: float):
        """Set audio volume (0.0 to 1.0)"""
        clamped_volume = max(0.0, min(1.0, volume))
        self.logger.info(f"Audio volume changed: {self.audio_volume:.2f} -> {clamped_volume:.2f}")
        
        self.audio_volume = clamped_volume
        self.volume_spin.setValue(int(self.audio_volume * 100))
        
        for player in self.audio_players.values():
            if player.audioOutput():
                player.audioOutput().setVolume(self.audio_volume)
        
        self.logger.debug(f"Volume updated for {len(self.audio_players)} audio players")


# Import QApplication for beep
from PyQt6.QtWidgets import QApplication
