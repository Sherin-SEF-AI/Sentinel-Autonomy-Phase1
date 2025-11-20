"""
Cloud settings dock widget for SENTINEL GUI.

Provides interface for configuring cloud synchronization settings.
"""

import logging
from typing import Optional
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
    QLineEdit, QPushButton, QCheckBox, QSpinBox, QTextEdit,
    QFormLayout, QMessageBox
)
from PyQt6.QtCore import Qt, pyqtSignal


class CloudSettingsDock(QWidget):
    """
    Cloud settings configuration widget.
    
    Features:
    - Configure API endpoint and credentials
    - Enable/disable cloud sync
    - Set sync interval
    - Manage consent preferences
    - View sync status
    """
    
    # Signals
    settings_changed = pyqtSignal(dict)  # Emitted when settings are saved
    test_connection = pyqtSignal()  # Emitted when test connection is clicked
    
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.logger = logging.getLogger(__name__)
        
        self.init_ui()
        self.logger.info("CloudSettingsDock initialized")
    
    def init_ui(self) -> None:
        """Initialize user interface"""
        layout = QVBoxLayout()
        layout.setSpacing(10)
        
        # Connection settings
        connection_group = self._create_connection_group()
        layout.addWidget(connection_group)
        
        # Sync settings
        sync_group = self._create_sync_group()
        layout.addWidget(sync_group)
        
        # Consent settings
        consent_group = self._create_consent_group()
        layout.addWidget(consent_group)
        
        # Status display
        status_group = self._create_status_group()
        layout.addWidget(status_group)
        
        # Action buttons
        button_layout = self._create_button_layout()
        layout.addLayout(button_layout)
        
        layout.addStretch()
        self.setLayout(layout)
    
    def _create_connection_group(self) -> QGroupBox:
        """Create connection settings group"""
        group = QGroupBox("Connection Settings")
        layout = QFormLayout()
        
        # API URL
        self.api_url_edit = QLineEdit()
        self.api_url_edit.setPlaceholderText("https://api.sentinel-fleet.com")
        layout.addRow("API URL:", self.api_url_edit)
        
        # API Key
        self.api_key_edit = QLineEdit()
        self.api_key_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.api_key_edit.setPlaceholderText("Enter API key")
        layout.addRow("API Key:", self.api_key_edit)
        
        # Vehicle ID
        self.vehicle_id_edit = QLineEdit()
        self.vehicle_id_edit.setPlaceholderText("vehicle_001")
        layout.addRow("Vehicle ID:", self.vehicle_id_edit)
        
        # Fleet ID
        self.fleet_id_edit = QLineEdit()
        self.fleet_id_edit.setPlaceholderText("fleet_alpha")
        layout.addRow("Fleet ID:", self.fleet_id_edit)
        
        # Test connection button
        test_btn = QPushButton("Test Connection")
        test_btn.clicked.connect(self.test_connection.emit)
        layout.addRow("", test_btn)
        
        group.setLayout(layout)
        return group
    
    def _create_sync_group(self) -> QGroupBox:
        """Create sync settings group"""
        group = QGroupBox("Synchronization Settings")
        layout = QFormLayout()
        
        # Enable cloud sync
        self.enable_sync_checkbox = QCheckBox("Enable cloud synchronization")
        self.enable_sync_checkbox.setChecked(False)
        self.enable_sync_checkbox.stateChanged.connect(self._on_sync_enabled_changed)
        layout.addRow(self.enable_sync_checkbox)
        
        # Sync interval
        self.sync_interval_spin = QSpinBox()
        self.sync_interval_spin.setMinimum(1)
        self.sync_interval_spin.setMaximum(60)
        self.sync_interval_spin.setValue(5)
        self.sync_interval_spin.setSuffix(" minutes")
        layout.addRow("Sync Interval:", self.sync_interval_spin)
        
        # Auto-install models
        self.auto_install_checkbox = QCheckBox("Automatically install model updates")
        self.auto_install_checkbox.setChecked(True)
        layout.addRow(self.auto_install_checkbox)
        
        group.setLayout(layout)
        return group
    
    def _create_consent_group(self) -> QGroupBox:
        """Create consent settings group"""
        group = QGroupBox("Data Sharing Consent")
        layout = QVBoxLayout()
        
        # Trip data consent
        self.trip_consent_checkbox = QCheckBox("Upload trip summaries (anonymized)")
        self.trip_consent_checkbox.setChecked(False)
        layout.addWidget(self.trip_consent_checkbox)
        
        # Scenario consent
        self.scenario_consent_checkbox = QCheckBox("Upload high-risk scenarios")
        self.scenario_consent_checkbox.setChecked(False)
        layout.addWidget(self.scenario_consent_checkbox)
        
        # Profile sync consent
        self.profile_consent_checkbox = QCheckBox("Sync driver profiles across vehicles")
        self.profile_consent_checkbox.setChecked(False)
        layout.addWidget(self.profile_consent_checkbox)
        
        # Info label
        info_label = QLabel(
            "Note: All uploaded data is encrypted and anonymized. "
            "You can change these settings at any time."
        )
        info_label.setWordWrap(True)
        info_label.setStyleSheet("color: #888; font-size: 10px;")
        layout.addWidget(info_label)
        
        group.setLayout(layout)
        return group
    
    def _create_status_group(self) -> QGroupBox:
        """Create status display group"""
        group = QGroupBox("Sync Status")
        layout = QVBoxLayout()
        
        # Status text
        self.status_text = QTextEdit()
        self.status_text.setReadOnly(True)
        self.status_text.setMaximumHeight(100)
        self.status_text.setPlaceholderText("No sync status available")
        layout.addWidget(self.status_text)
        
        # Statistics
        stats_layout = QHBoxLayout()
        
        self.trips_label = QLabel("Trips: 0")
        stats_layout.addWidget(self.trips_label)
        
        self.scenarios_label = QLabel("Scenarios: 0")
        stats_layout.addWidget(self.scenarios_label)
        
        self.profiles_label = QLabel("Profiles: 0")
        stats_layout.addWidget(self.profiles_label)
        
        stats_layout.addStretch()
        layout.addLayout(stats_layout)
        
        group.setLayout(layout)
        return group
    
    def _create_button_layout(self) -> QHBoxLayout:
        """Create action buttons layout"""
        layout = QHBoxLayout()
        
        # Save button
        self.save_btn = QPushButton("Save Settings")
        self.save_btn.clicked.connect(self._on_save_clicked)
        layout.addWidget(self.save_btn)
        
        # Force sync button
        self.sync_now_btn = QPushButton("Sync Now")
        self.sync_now_btn.clicked.connect(self._on_sync_now_clicked)
        self.sync_now_btn.setEnabled(False)
        layout.addWidget(self.sync_now_btn)
        
        layout.addStretch()
        
        return layout
    
    def _on_sync_enabled_changed(self, state: int) -> None:
        """Handle sync enabled checkbox state change"""
        enabled = state == Qt.CheckState.Checked.value
        
        # Enable/disable sync-related controls
        self.sync_interval_spin.setEnabled(enabled)
        self.auto_install_checkbox.setEnabled(enabled)
        self.trip_consent_checkbox.setEnabled(enabled)
        self.scenario_consent_checkbox.setEnabled(enabled)
        self.profile_consent_checkbox.setEnabled(enabled)
        self.sync_now_btn.setEnabled(enabled)
    
    def _on_save_clicked(self) -> None:
        """Handle save button click"""
        # Validate settings
        if not self._validate_settings():
            return
        
        # Collect settings
        settings = self.get_settings()
        
        # Emit signal
        self.settings_changed.emit(settings)
        
        # Show confirmation
        QMessageBox.information(
            self,
            "Settings Saved",
            "Cloud settings have been saved successfully."
        )
        
        self.logger.info("Cloud settings saved")
    
    def _on_sync_now_clicked(self) -> None:
        """Handle sync now button click"""
        # This would trigger immediate sync
        self.logger.info("Manual sync triggered")
        self.update_status("Synchronizing...")
    
    def _validate_settings(self) -> bool:
        """
        Validate settings before saving.
        
        Returns:
            True if settings are valid
        """
        if self.enable_sync_checkbox.isChecked():
            # Check required fields
            if not self.api_url_edit.text():
                QMessageBox.warning(
                    self,
                    "Validation Error",
                    "API URL is required when cloud sync is enabled."
                )
                return False
            
            if not self.api_key_edit.text():
                QMessageBox.warning(
                    self,
                    "Validation Error",
                    "API Key is required when cloud sync is enabled."
                )
                return False
            
            if not self.vehicle_id_edit.text():
                QMessageBox.warning(
                    self,
                    "Validation Error",
                    "Vehicle ID is required when cloud sync is enabled."
                )
                return False
            
            if not self.fleet_id_edit.text():
                QMessageBox.warning(
                    self,
                    "Validation Error",
                    "Fleet ID is required when cloud sync is enabled."
                )
                return False
        
        return True
    
    def get_settings(self) -> dict:
        """
        Get current settings.
        
        Returns:
            Dictionary with settings
        """
        return {
            'enabled': self.enable_sync_checkbox.isChecked(),
            'api_url': self.api_url_edit.text(),
            'api_key': self.api_key_edit.text(),
            'vehicle_id': self.vehicle_id_edit.text(),
            'fleet_id': self.fleet_id_edit.text(),
            'sync_interval': self.sync_interval_spin.value() * 60,  # Convert to seconds
            'auto_install': self.auto_install_checkbox.isChecked(),
            'trip_consent': self.trip_consent_checkbox.isChecked(),
            'scenario_consent': self.scenario_consent_checkbox.isChecked(),
            'profile_consent': self.profile_consent_checkbox.isChecked()
        }
    
    def set_settings(self, settings: dict) -> None:
        """
        Set settings from dictionary.
        
        Args:
            settings: Dictionary with settings
        """
        self.enable_sync_checkbox.setChecked(settings.get('enabled', False))
        self.api_url_edit.setText(settings.get('api_url', ''))
        self.api_key_edit.setText(settings.get('api_key', ''))
        self.vehicle_id_edit.setText(settings.get('vehicle_id', ''))
        self.fleet_id_edit.setText(settings.get('fleet_id', ''))
        self.sync_interval_spin.setValue(settings.get('sync_interval', 300) // 60)
        self.auto_install_checkbox.setChecked(settings.get('auto_install', True))
        self.trip_consent_checkbox.setChecked(settings.get('trip_consent', False))
        self.scenario_consent_checkbox.setChecked(settings.get('scenario_consent', False))
        self.profile_consent_checkbox.setChecked(settings.get('profile_consent', False))
    
    def update_status(self, status: str) -> None:
        """
        Update status display.
        
        Args:
            status: Status message
        """
        self.status_text.append(f"[{self._get_timestamp()}] {status}")
        
        # Auto-scroll to bottom
        scrollbar = self.status_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
    
    def update_statistics(self, trips: int, scenarios: int, profiles: int) -> None:
        """
        Update statistics display.
        
        Args:
            trips: Number of queued trips
            scenarios: Number of queued scenarios
            profiles: Number of profiles
        """
        self.trips_label.setText(f"Trips: {trips}")
        self.scenarios_label.setText(f"Scenarios: {scenarios}")
        self.profiles_label.setText(f"Profiles: {profiles}")
    
    def _get_timestamp(self) -> str:
        """Get current timestamp string"""
        from datetime import datetime
        return datetime.now().strftime("%H:%M:%S")
