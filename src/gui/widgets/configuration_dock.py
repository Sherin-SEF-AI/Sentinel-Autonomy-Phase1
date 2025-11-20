"""
Configuration Dock Widget for SENTINEL GUI

Provides configuration interface with:
- Tabbed interface for different parameter categories
- Labeled sliders for parameter adjustment
- Real-time parameter updates
- Save and reset functionality
- Profile management (import/export)
"""

import os
import yaml
import logging
from typing import Dict, Any, Optional, Callable
from pathlib import Path
from datetime import datetime

from PyQt6.QtWidgets import (
    QDockWidget, QWidget, QVBoxLayout, QHBoxLayout, QTabWidget,
    QLabel, QPushButton, QSlider, QDoubleSpinBox, QSpinBox,
    QGroupBox, QScrollArea, QFileDialog, QMessageBox, QComboBox,
    QCheckBox, QLineEdit
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont

logger = logging.getLogger(__name__)


class LabeledSlider(QWidget):
    """
    Custom widget combining a slider with value display.
    
    Features:
    - Slider for visual adjustment
    - SpinBox for precise value entry
    - Min, max, current value display
    - Units label
    - Value change signal
    """
    
    value_changed = pyqtSignal(float)  # Emits new value
    
    def __init__(
        self,
        label: str,
        min_value: float,
        max_value: float,
        current_value: float,
        step: float = 0.1,
        decimals: int = 2,
        units: str = "",
        tooltip: str = ""
    ):
        super().__init__()
        
        self.min_value = min_value
        self.max_value = max_value
        self.step = step
        self.decimals = decimals
        self.units = units
        
        # Determine if we need float or int
        self.use_float = (decimals > 0) or (step < 1.0)
        
        # Initialize UI
        self._init_ui(label, current_value, tooltip)
        
        logger.debug(f"LabeledSlider created: label='{label}', range=[{min_value}, {max_value}], value={current_value}")
    
    def _init_ui(self, label: str, current_value: float, tooltip: str):
        """Initialize the UI components"""
        layout = QVBoxLayout()
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(5)
        
        # Label row
        label_layout = QHBoxLayout()
        
        param_label = QLabel(label)
        param_label.setStyleSheet("font-weight: bold;")
        label_layout.addWidget(param_label)
        
        label_layout.addStretch()
        
        # Range info
        range_label = QLabel(f"[{self.min_value} - {self.max_value}]")
        range_label.setStyleSheet("color: #888; font-size: 10px;")
        label_layout.addWidget(range_label)
        
        layout.addLayout(label_layout)
        
        # Slider and spinbox row
        control_layout = QHBoxLayout()
        
        # Slider
        self.slider = QSlider(Qt.Orientation.Horizontal)
        
        # Calculate slider range (use 1000 steps for smooth float values)
        if self.use_float:
            self.slider_steps = 1000
            self.slider.setMinimum(0)
            self.slider.setMaximum(self.slider_steps)
            # Set initial position
            slider_pos = int((current_value - self.min_value) / (self.max_value - self.min_value) * self.slider_steps)
            self.slider.setValue(slider_pos)
        else:
            self.slider.setMinimum(int(self.min_value))
            self.slider.setMaximum(int(self.max_value))
            self.slider.setValue(int(current_value))
        
        self.slider.valueChanged.connect(self._on_slider_changed)
        control_layout.addWidget(self.slider, 3)
        
        # SpinBox
        if self.use_float:
            self.spinbox = QDoubleSpinBox()
            self.spinbox.setDecimals(self.decimals)
            self.spinbox.setSingleStep(self.step)
        else:
            self.spinbox = QSpinBox()
            self.spinbox.setSingleStep(int(self.step))
        
        self.spinbox.setMinimum(self.min_value)
        self.spinbox.setMaximum(self.max_value)
        self.spinbox.setValue(current_value)
        
        if self.units:
            self.spinbox.setSuffix(f" {self.units}")
        
        self.spinbox.valueChanged.connect(self._on_spinbox_changed)
        control_layout.addWidget(self.spinbox, 1)
        
        layout.addLayout(control_layout)
        
        # Tooltip
        if tooltip:
            self.setToolTip(tooltip)
            param_label.setToolTip(tooltip)
            self.slider.setToolTip(tooltip)
            self.spinbox.setToolTip(tooltip)
        
        self.setLayout(layout)
    
    def _on_slider_changed(self, slider_value: int):
        """Handle slider value change"""
        if self.use_float:
            # Convert slider position to actual value
            ratio = slider_value / self.slider_steps
            actual_value = self.min_value + ratio * (self.max_value - self.min_value)
            # Round to step
            actual_value = round(actual_value / self.step) * self.step
        else:
            actual_value = slider_value
        
        # Update spinbox without triggering its signal
        self.spinbox.blockSignals(True)
        self.spinbox.setValue(actual_value)
        self.spinbox.blockSignals(False)
        
        # Emit value changed
        self.value_changed.emit(actual_value)
        logger.debug(f"Slider changed: value={actual_value}")
    
    def _on_spinbox_changed(self, value: float):
        """Handle spinbox value change"""
        # Update slider without triggering its signal
        self.slider.blockSignals(True)
        
        if self.use_float:
            # Convert value to slider position
            ratio = (value - self.min_value) / (self.max_value - self.min_value)
            slider_pos = int(ratio * self.slider_steps)
            self.slider.setValue(slider_pos)
        else:
            self.slider.setValue(int(value))
        
        self.slider.blockSignals(False)
        
        # Emit value changed
        self.value_changed.emit(value)
        logger.debug(f"SpinBox changed: value={value}")
    
    def get_value(self) -> float:
        """Get current value"""
        return self.spinbox.value()
    
    def set_value(self, value: float):
        """Set value programmatically"""
        self.spinbox.setValue(value)


class ConfigurationDockWidget(QDockWidget):
    """
    Dock widget for system configuration.
    
    Features:
    - Tabbed interface (Cameras, Detection, DMS, Risk, Alerts)
    - Parameter controls with sliders
    - Real-time updates for non-critical parameters
    - Save/reset functionality
    - Profile import/export
    """
    
    # Signals
    config_changed = pyqtSignal(dict)  # Emits updated config
    config_saved = pyqtSignal(str)  # Emits config file path
    
    def __init__(self, config_path: str = 'configs/default.yaml'):
        super().__init__("Configuration")
        
        self.config_path = config_path
        self.config: Dict[str, Any] = {}
        self.original_config: Dict[str, Any] = {}
        self.has_unsaved_changes = False
        
        # Parameter widgets storage
        self.param_widgets: Dict[str, LabeledSlider] = {}
        
        # Load configuration
        self._load_config()
        
        # Create main widget
        main_widget = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(5)
        
        # Create tab widget
        self.tab_widget = QTabWidget()
        self._create_tabs()
        layout.addWidget(self.tab_widget)
        
        # Unsaved changes indicator
        self.changes_label = QLabel("")
        self.changes_label.setStyleSheet("color: #ffaa00; font-weight: bold;")
        self.changes_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.changes_label)
        
        # Action buttons
        buttons_layout = QHBoxLayout()
        
        self.save_button = QPushButton("Save Configuration")
        self.save_button.setEnabled(False)
        self.save_button.clicked.connect(self._on_save_clicked)
        buttons_layout.addWidget(self.save_button)
        
        self.reset_button = QPushButton("Reset to Defaults")
        self.reset_button.clicked.connect(self._on_reset_clicked)
        buttons_layout.addWidget(self.reset_button)
        
        layout.addLayout(buttons_layout)
        
        # Profile management buttons
        profile_layout = QHBoxLayout()
        
        self.import_button = QPushButton("Import Profile")
        self.import_button.clicked.connect(self._on_import_clicked)
        profile_layout.addWidget(self.import_button)
        
        self.export_button = QPushButton("Export Profile")
        self.export_button.clicked.connect(self._on_export_clicked)
        profile_layout.addWidget(self.export_button)
        
        layout.addLayout(profile_layout)
        
        main_widget.setLayout(layout)
        self.setWidget(main_widget)
        
        logger.info(f"ConfigurationDockWidget initialized: config_path={config_path}")
    
    def _load_config(self):
        """Load configuration from YAML file"""
        logger.info(f"Loading configuration: path={self.config_path}")
        
        try:
            with open(self.config_path, 'r') as f:
                self.config = yaml.safe_load(f)
            
            # Store original for reset
            import copy
            self.original_config = copy.deepcopy(self.config)
            
            logger.info(f"Configuration loaded successfully: {len(self.config)} top-level keys")
            
        except FileNotFoundError:
            logger.error(f"Configuration file not found: {self.config_path}")
            self.config = {}
            self.original_config = {}
            QMessageBox.critical(
                self,
                "Configuration Error",
                f"Configuration file not found:\n{self.config_path}"
            )
        except yaml.YAMLError as e:
            logger.error(f"Invalid YAML in configuration: {e}")
            self.config = {}
            self.original_config = {}
            QMessageBox.critical(
                self,
                "Configuration Error",
                f"Invalid YAML in configuration file:\n{e}"
            )
    
    def _create_tabs(self):
        """Create all configuration tabs"""
        logger.debug("Creating configuration tabs")
        
        # Cameras tab
        cameras_tab = self._create_cameras_tab()
        self.tab_widget.addTab(cameras_tab, "Cameras")
        
        # Detection tab
        detection_tab = self._create_detection_tab()
        self.tab_widget.addTab(detection_tab, "Detection")
        
        # DMS tab
        dms_tab = self._create_dms_tab()
        self.tab_widget.addTab(dms_tab, "DMS")
        
        # Risk tab
        risk_tab = self._create_risk_tab()
        self.tab_widget.addTab(risk_tab, "Risk")
        
        # Alerts tab
        alerts_tab = self._create_alerts_tab()
        self.tab_widget.addTab(alerts_tab, "Alerts")
        
        logger.debug(f"Created {self.tab_widget.count()} configuration tabs")
    
    def _create_cameras_tab(self) -> QWidget:
        """Create cameras configuration tab"""
        tab = QWidget()
        layout = QVBoxLayout()
        
        # Create scroll area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout()
        
        cameras_config = self.config.get('cameras', {})
        
        # Interior camera group
        if 'interior' in cameras_config:
            interior_group = QGroupBox("Interior Camera")
            interior_layout = QVBoxLayout()
            
            interior = cameras_config['interior']
            
            # Device ID
            device_layout = QHBoxLayout()
            device_layout.addWidget(QLabel("Device ID:"))
            device_spin = QSpinBox()
            device_spin.setMinimum(0)
            device_spin.setMaximum(10)
            device_spin.setValue(interior.get('device', 0))
            device_spin.valueChanged.connect(
                lambda v: self._on_param_changed('cameras.interior.device', v)
            )
            device_layout.addWidget(device_spin)
            device_layout.addStretch()
            interior_layout.addLayout(device_layout)
            
            # FPS
            fps_slider = self._create_slider(
                'cameras.interior.fps',
                "Frame Rate",
                10, 60, interior.get('fps', 30),
                step=1, decimals=0, units="fps",
                tooltip="Camera capture frame rate"
            )
            interior_layout.addWidget(fps_slider)
            
            interior_group.setLayout(interior_layout)
            scroll_layout.addWidget(interior_group)
        
        # Front cameras
        for camera_name in ['front_left', 'front_right']:
            if camera_name in cameras_config:
                camera_group = QGroupBox(f"{camera_name.replace('_', ' ').title()} Camera")
                camera_layout = QVBoxLayout()
                
                camera = cameras_config[camera_name]
                
                # Device ID
                device_layout = QHBoxLayout()
                device_layout.addWidget(QLabel("Device ID:"))
                device_spin = QSpinBox()
                device_spin.setMinimum(0)
                device_spin.setMaximum(10)
                device_spin.setValue(camera.get('device', 0))
                device_spin.valueChanged.connect(
                    lambda v, cn=camera_name: self._on_param_changed(f'cameras.{cn}.device', v)
                )
                device_layout.addWidget(device_spin)
                device_layout.addStretch()
                camera_layout.addLayout(device_layout)
                
                # FPS
                fps_slider = self._create_slider(
                    f'cameras.{camera_name}.fps',
                    "Frame Rate",
                    10, 60, camera.get('fps', 30),
                    step=1, decimals=0, units="fps",
                    tooltip="Camera capture frame rate"
                )
                camera_layout.addWidget(fps_slider)
                
                camera_group.setLayout(camera_layout)
                scroll_layout.addWidget(camera_group)
        
        scroll_layout.addStretch()
        scroll_widget.setLayout(scroll_layout)
        scroll.setWidget(scroll_widget)
        
        layout.addWidget(scroll)
        tab.setLayout(layout)
        
        return tab
    
    def _create_detection_tab(self) -> QWidget:
        """Create detection configuration tab"""
        tab = QWidget()
        layout = QVBoxLayout()
        
        # Create scroll area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout()
        
        # Detection model parameters
        detection_group = QGroupBox("Detection Model")
        detection_layout = QVBoxLayout()
        
        models_config = self.config.get('models', {})
        detection_config = models_config.get('detection', {})
        
        # Confidence threshold
        conf_slider = self._create_slider(
            'models.detection.confidence_threshold',
            "Confidence Threshold",
            0.1, 0.9, detection_config.get('confidence_threshold', 0.5),
            step=0.05, decimals=2,
            tooltip="Minimum confidence for object detection"
        )
        detection_layout.addWidget(conf_slider)
        
        # NMS threshold
        nms_slider = self._create_slider(
            'models.detection.nms_threshold',
            "NMS Threshold",
            0.1, 0.9, detection_config.get('nms_threshold', 0.4),
            step=0.05, decimals=2,
            tooltip="Non-maximum suppression threshold"
        )
        detection_layout.addWidget(nms_slider)
        
        detection_group.setLayout(detection_layout)
        scroll_layout.addWidget(detection_group)
        
        # Fusion parameters
        fusion_group = QGroupBox("Multi-View Fusion")
        fusion_layout = QVBoxLayout()
        
        fusion_config = self.config.get('fusion', {})
        
        # IoU threshold
        iou_slider = self._create_slider(
            'fusion.iou_threshold_3d',
            "3D IoU Threshold",
            0.1, 0.7, fusion_config.get('iou_threshold_3d', 0.3),
            step=0.05, decimals=2,
            tooltip="IoU threshold for multi-view fusion"
        )
        fusion_layout.addWidget(iou_slider)
        
        fusion_group.setLayout(fusion_layout)
        scroll_layout.addWidget(fusion_group)
        
        # Tracking parameters
        tracking_group = QGroupBox("Object Tracking")
        tracking_layout = QVBoxLayout()
        
        tracking_config = self.config.get('tracking', {})
        
        # Max age
        max_age_slider = self._create_slider(
            'tracking.max_age',
            "Max Age",
            10, 60, tracking_config.get('max_age', 30),
            step=1, decimals=0, units="frames",
            tooltip="Maximum frames to keep track without detection"
        )
        tracking_layout.addWidget(max_age_slider)
        
        # Min hits
        min_hits_slider = self._create_slider(
            'tracking.min_hits',
            "Min Hits",
            1, 10, tracking_config.get('min_hits', 3),
            step=1, decimals=0, units="frames",
            tooltip="Minimum detections before track is confirmed"
        )
        tracking_layout.addWidget(min_hits_slider)
        
        # IoU threshold
        track_iou_slider = self._create_slider(
            'tracking.iou_threshold',
            "IoU Threshold",
            0.1, 0.7, tracking_config.get('iou_threshold', 0.3),
            step=0.05, decimals=2,
            tooltip="IoU threshold for track association"
        )
        tracking_layout.addWidget(track_iou_slider)
        
        tracking_group.setLayout(tracking_layout)
        scroll_layout.addWidget(tracking_group)
        
        scroll_layout.addStretch()
        scroll_widget.setLayout(scroll_layout)
        scroll.setWidget(scroll_widget)
        
        layout.addWidget(scroll)
        tab.setLayout(layout)
        
        return tab
    
    def _create_dms_tab(self) -> QWidget:
        """Create DMS configuration tab"""
        tab = QWidget()
        layout = QVBoxLayout()
        
        # Create scroll area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout()
        
        # Segmentation model parameters
        seg_group = QGroupBox("Segmentation Model")
        seg_layout = QVBoxLayout()
        
        models_config = self.config.get('models', {})
        seg_config = models_config.get('segmentation', {})
        
        # Temporal smoothing alpha
        alpha_slider = self._create_slider(
            'models.segmentation.smoothing_alpha',
            "Temporal Smoothing Alpha",
            0.1, 1.0, seg_config.get('smoothing_alpha', 0.7),
            step=0.05, decimals=2,
            tooltip="Alpha value for temporal smoothing (higher = more smoothing)"
        )
        seg_layout.addWidget(alpha_slider)
        
        seg_group.setLayout(seg_layout)
        scroll_layout.addWidget(seg_group)
        
        # Note about DMS parameters
        note_label = QLabel(
            "Note: DMS model parameters are primarily configured through model weights.\n"
            "Advanced DMS tuning will be available in future updates."
        )
        note_label.setStyleSheet("color: #888; font-style: italic; padding: 10px;")
        note_label.setWordWrap(True)
        scroll_layout.addWidget(note_label)
        
        scroll_layout.addStretch()
        scroll_widget.setLayout(scroll_layout)
        scroll.setWidget(scroll_widget)
        
        layout.addWidget(scroll)
        tab.setLayout(layout)
        
        return tab
    
    def _create_risk_tab(self) -> QWidget:
        """Create risk assessment configuration tab"""
        tab = QWidget()
        layout = QVBoxLayout()
        
        # Create scroll area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout()
        
        risk_config = self.config.get('risk_assessment', {})
        
        # TTC calculation
        ttc_group = QGroupBox("Time-To-Collision")
        ttc_layout = QVBoxLayout()
        
        ttc_calc = risk_config.get('ttc_calculation', {})
        
        # Safety margin
        margin_slider = self._create_slider(
            'risk_assessment.ttc_calculation.safety_margin',
            "Safety Margin",
            0.5, 3.0, ttc_calc.get('safety_margin', 1.5),
            step=0.1, decimals=1, units="m",
            tooltip="Safety margin added to TTC calculation"
        )
        ttc_layout.addWidget(margin_slider)
        
        ttc_group.setLayout(ttc_layout)
        scroll_layout.addWidget(ttc_group)
        
        # Trajectory prediction
        traj_group = QGroupBox("Trajectory Prediction")
        traj_layout = QVBoxLayout()
        
        traj_pred = risk_config.get('trajectory_prediction', {})
        
        # Horizon
        horizon_slider = self._create_slider(
            'risk_assessment.trajectory_prediction.horizon',
            "Prediction Horizon",
            1.0, 5.0, traj_pred.get('horizon', 3.0),
            step=0.5, decimals=1, units="s",
            tooltip="How far ahead to predict trajectories"
        )
        traj_layout.addWidget(horizon_slider)
        
        # Time step
        dt_slider = self._create_slider(
            'risk_assessment.trajectory_prediction.dt',
            "Time Step",
            0.05, 0.5, traj_pred.get('dt', 0.1),
            step=0.05, decimals=2, units="s",
            tooltip="Time step for trajectory prediction"
        )
        traj_layout.addWidget(dt_slider)
        
        traj_group.setLayout(traj_layout)
        scroll_layout.addWidget(traj_group)
        
        # Risk weights
        weights_group = QGroupBox("Base Risk Weights")
        weights_layout = QVBoxLayout()
        
        base_weights = risk_config.get('base_risk_weights', {})
        
        # TTC weight
        ttc_weight_slider = self._create_slider(
            'risk_assessment.base_risk_weights.ttc',
            "TTC Weight",
            0.0, 1.0, base_weights.get('ttc', 0.4),
            step=0.05, decimals=2,
            tooltip="Weight for TTC in risk calculation"
        )
        weights_layout.addWidget(ttc_weight_slider)
        
        # Trajectory conflict weight
        traj_weight_slider = self._create_slider(
            'risk_assessment.base_risk_weights.trajectory_conflict',
            "Trajectory Conflict Weight",
            0.0, 1.0, base_weights.get('trajectory_conflict', 0.3),
            step=0.05, decimals=2,
            tooltip="Weight for trajectory conflict in risk calculation"
        )
        weights_layout.addWidget(traj_weight_slider)
        
        # Vulnerability weight
        vuln_weight_slider = self._create_slider(
            'risk_assessment.base_risk_weights.vulnerability',
            "Vulnerability Weight",
            0.0, 1.0, base_weights.get('vulnerability', 0.2),
            step=0.05, decimals=2,
            tooltip="Weight for object vulnerability in risk calculation"
        )
        weights_layout.addWidget(vuln_weight_slider)
        
        # Relative speed weight
        speed_weight_slider = self._create_slider(
            'risk_assessment.base_risk_weights.relative_speed',
            "Relative Speed Weight",
            0.0, 1.0, base_weights.get('relative_speed', 0.1),
            step=0.05, decimals=2,
            tooltip="Weight for relative speed in risk calculation"
        )
        weights_layout.addWidget(speed_weight_slider)
        
        weights_group.setLayout(weights_layout)
        scroll_layout.addWidget(weights_group)
        
        # Risk thresholds
        thresholds_group = QGroupBox("Risk Thresholds")
        thresholds_layout = QVBoxLayout()
        
        thresholds = risk_config.get('thresholds', {})
        
        # Hazard detection
        hazard_slider = self._create_slider(
            'risk_assessment.thresholds.hazard_detection',
            "Hazard Detection",
            0.1, 0.9, thresholds.get('hazard_detection', 0.3),
            step=0.05, decimals=2,
            tooltip="Minimum risk score to detect hazard"
        )
        thresholds_layout.addWidget(hazard_slider)
        
        # Intervention
        intervention_slider = self._create_slider(
            'risk_assessment.thresholds.intervention',
            "Intervention",
            0.3, 0.9, thresholds.get('intervention', 0.7),
            step=0.05, decimals=2,
            tooltip="Risk score threshold for intervention"
        )
        thresholds_layout.addWidget(intervention_slider)
        
        # Critical
        critical_slider = self._create_slider(
            'risk_assessment.thresholds.critical',
            "Critical",
            0.5, 1.0, thresholds.get('critical', 0.9),
            step=0.05, decimals=2,
            tooltip="Risk score threshold for critical alerts"
        )
        thresholds_layout.addWidget(critical_slider)
        
        thresholds_group.setLayout(thresholds_layout)
        scroll_layout.addWidget(thresholds_group)
        
        scroll_layout.addStretch()
        scroll_widget.setLayout(scroll_layout)
        scroll.setWidget(scroll_widget)
        
        layout.addWidget(scroll)
        tab.setLayout(layout)
        
        return tab
    
    def _create_alerts_tab(self) -> QWidget:
        """Create alerts configuration tab"""
        tab = QWidget()
        layout = QVBoxLayout()
        
        # Create scroll area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout()
        
        alerts_config = self.config.get('alerts', {})
        
        # Suppression
        suppression_group = QGroupBox("Alert Suppression")
        suppression_layout = QVBoxLayout()
        
        suppression = alerts_config.get('suppression', {})
        
        # Duplicate window
        dup_window_slider = self._create_slider(
            'alerts.suppression.duplicate_window',
            "Duplicate Window",
            1.0, 10.0, suppression.get('duplicate_window', 5.0),
            step=0.5, decimals=1, units="s",
            tooltip="Time window for suppressing duplicate alerts"
        )
        suppression_layout.addWidget(dup_window_slider)
        
        # Max simultaneous
        max_sim_slider = self._create_slider(
            'alerts.suppression.max_simultaneous',
            "Max Simultaneous Alerts",
            1, 5, suppression.get('max_simultaneous', 2),
            step=1, decimals=0,
            tooltip="Maximum number of simultaneous alerts"
        )
        suppression_layout.addWidget(max_sim_slider)
        
        suppression_group.setLayout(suppression_layout)
        scroll_layout.addWidget(suppression_group)
        
        # Escalation thresholds
        escalation_group = QGroupBox("Alert Escalation")
        escalation_layout = QVBoxLayout()
        
        escalation = alerts_config.get('escalation', {})
        
        # Critical threshold
        critical_slider = self._create_slider(
            'alerts.escalation.critical_threshold',
            "Critical Threshold",
            0.5, 1.0, escalation.get('critical_threshold', 0.9),
            step=0.05, decimals=2,
            tooltip="Risk score for critical alerts"
        )
        escalation_layout.addWidget(critical_slider)
        
        # High threshold
        high_slider = self._create_slider(
            'alerts.escalation.high_threshold',
            "High Threshold",
            0.3, 0.9, escalation.get('high_threshold', 0.7),
            step=0.05, decimals=2,
            tooltip="Risk score for high priority alerts"
        )
        escalation_layout.addWidget(high_slider)
        
        # Medium threshold
        medium_slider = self._create_slider(
            'alerts.escalation.medium_threshold',
            "Medium Threshold",
            0.1, 0.7, escalation.get('medium_threshold', 0.5),
            step=0.05, decimals=2,
            tooltip="Risk score for medium priority alerts"
        )
        escalation_layout.addWidget(medium_slider)
        
        escalation_group.setLayout(escalation_layout)
        scroll_layout.addWidget(escalation_group)
        
        # Modalities
        modalities_group = QGroupBox("Alert Modalities")
        modalities_layout = QVBoxLayout()
        
        modalities = alerts_config.get('modalities', {})
        
        # Visual
        visual = modalities.get('visual', {})
        
        visual_duration_slider = self._create_slider(
            'alerts.modalities.visual.display_duration',
            "Visual Display Duration",
            1.0, 10.0, visual.get('display_duration', 3.0),
            step=0.5, decimals=1, units="s",
            tooltip="How long to display visual alerts"
        )
        modalities_layout.addWidget(visual_duration_slider)
        
        visual_flash_slider = self._create_slider(
            'alerts.modalities.visual.flash_rate',
            "Visual Flash Rate",
            1, 5, visual.get('flash_rate', 2),
            step=1, decimals=0, units="Hz",
            tooltip="Flash rate for critical visual alerts"
        )
        modalities_layout.addWidget(visual_flash_slider)
        
        # Audio
        audio = modalities.get('audio', {})
        
        audio_volume_slider = self._create_slider(
            'alerts.modalities.audio.volume',
            "Audio Volume",
            0.0, 1.0, audio.get('volume', 0.8),
            step=0.05, decimals=2,
            tooltip="Audio alert volume (0.0 = mute, 1.0 = max)"
        )
        modalities_layout.addWidget(audio_volume_slider)
        
        modalities_group.setLayout(modalities_layout)
        scroll_layout.addWidget(modalities_group)
        
        scroll_layout.addStretch()
        scroll_widget.setLayout(scroll_layout)
        scroll.setWidget(scroll_widget)
        
        layout.addWidget(scroll)
        tab.setLayout(layout)
        
        return tab
    
    def _create_slider(
        self,
        config_key: str,
        label: str,
        min_value: float,
        max_value: float,
        current_value: float,
        step: float = 0.1,
        decimals: int = 2,
        units: str = "",
        tooltip: str = ""
    ) -> LabeledSlider:
        """
        Create a labeled slider and register it.
        
        Args:
            config_key: Dot-notation key in config (e.g., 'models.detection.confidence_threshold')
            label: Display label
            min_value: Minimum value
            max_value: Maximum value
            current_value: Current value
            step: Step size
            decimals: Number of decimal places
            units: Units string
            tooltip: Tooltip text
            
        Returns:
            LabeledSlider widget
        """
        slider = LabeledSlider(
            label, min_value, max_value, current_value,
            step, decimals, units, tooltip
        )
        
        # Connect to parameter change handler
        slider.value_changed.connect(
            lambda value, key=config_key: self._on_param_changed(key, value)
        )
        
        # Store reference
        self.param_widgets[config_key] = slider
        
        return slider
    
    def _on_param_changed(self, config_key: str, value: Any):
        """
        Handle parameter value change.
        
        Args:
            config_key: Dot-notation key in config
            value: New value
        """
        logger.debug(f"Parameter changed: key={config_key}, value={value}")
        
        # Update config
        keys = config_key.split('.')
        config_ref = self.config
        
        for key in keys[:-1]:
            if key not in config_ref:
                config_ref[key] = {}
            config_ref = config_ref[key]
        
        config_ref[keys[-1]] = value
        
        # Mark as having unsaved changes
        self.has_unsaved_changes = True
        self._update_changes_indicator()
        
        # Emit config changed signal for real-time updates
        self.config_changed.emit(self.config)
        
        logger.info(f"Configuration updated: {config_key}={value}")
    
    def _update_changes_indicator(self):
        """Update the unsaved changes indicator"""
        if self.has_unsaved_changes:
            self.changes_label.setText("⚠ Unsaved Changes")
            self.save_button.setEnabled(True)
        else:
            self.changes_label.setText("")
            self.save_button.setEnabled(False)
    
    def _on_save_clicked(self):
        """Handle save button click"""
        logger.info("Save configuration requested")
        
        # Confirm save
        reply = QMessageBox.question(
            self,
            "Save Configuration",
            f"Save configuration to:\n{self.config_path}?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            self._save_config()
    
    def _save_config(self):
        """Save configuration to YAML file"""
        logger.info(f"Saving configuration: path={self.config_path}")
        
        try:
            # Create backup of existing config
            if os.path.exists(self.config_path):
                backup_path = f"{self.config_path}.backup"
                import shutil
                shutil.copy2(self.config_path, backup_path)
                logger.debug(f"Created backup: {backup_path}")
            
            # Write new config
            with open(self.config_path, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False, sort_keys=False)
            
            # Update original config
            import copy
            self.original_config = copy.deepcopy(self.config)
            
            # Clear unsaved changes
            self.has_unsaved_changes = False
            self._update_changes_indicator()
            
            # Emit saved signal
            self.config_saved.emit(self.config_path)
            
            QMessageBox.information(
                self,
                "Configuration Saved",
                f"Configuration saved successfully to:\n{self.config_path}"
            )
            
            logger.info("Configuration saved successfully")
            
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
            QMessageBox.critical(
                self,
                "Save Failed",
                f"Failed to save configuration:\n{e}"
            )
    
    def _on_reset_clicked(self):
        """Handle reset button click"""
        logger.info("Reset to defaults requested")
        
        # Confirm reset
        reply = QMessageBox.question(
            self,
            "Reset Configuration",
            "Reset all parameters to default values?\n\nThis will discard all unsaved changes.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            self._reset_config()
    
    def _reset_config(self):
        """Reset configuration to original values"""
        logger.info("Resetting configuration to defaults")
        
        # Restore original config
        import copy
        self.config = copy.deepcopy(self.original_config)
        
        # Update all widgets
        for config_key, widget in self.param_widgets.items():
            # Get value from config
            keys = config_key.split('.')
            value = self.config
            
            try:
                for key in keys:
                    value = value[key]
                
                # Update widget
                widget.set_value(value)
                logger.debug(f"Reset parameter: {config_key}={value}")
                
            except (KeyError, TypeError):
                logger.warning(f"Could not reset parameter: {config_key}")
        
        # Clear unsaved changes
        self.has_unsaved_changes = False
        self._update_changes_indicator()
        
        # Emit config changed
        self.config_changed.emit(self.config)
        
        QMessageBox.information(
            self,
            "Configuration Reset",
            "Configuration has been reset to default values."
        )
        
        logger.info("Configuration reset complete")
    
    def _on_import_clicked(self):
        """Handle import profile button click"""
        logger.info("Import profile requested")
        
        # Show file dialog
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Import Configuration Profile",
            "",
            "YAML Files (*.yaml *.yml);;All Files (*)"
        )
        
        if file_path:
            self._import_profile(file_path)
    
    def _import_profile(self, file_path: str):
        """Import configuration profile from file"""
        logger.info(f"Importing profile: path={file_path}")
        
        try:
            with open(file_path, 'r') as f:
                imported_config = yaml.safe_load(f)
            
            # Validate imported config (basic check)
            if not isinstance(imported_config, dict):
                raise ValueError("Invalid configuration format")
            
            # Update config
            self.config = imported_config
            
            # Update all widgets
            for config_key, widget in self.param_widgets.items():
                keys = config_key.split('.')
                value = self.config
                
                try:
                    for key in keys:
                        value = value[key]
                    widget.set_value(value)
                except (KeyError, TypeError):
                    logger.warning(f"Parameter not found in imported config: {config_key}")
            
            # Mark as having unsaved changes
            self.has_unsaved_changes = True
            self._update_changes_indicator()
            
            # Emit config changed
            self.config_changed.emit(self.config)
            
            QMessageBox.information(
                self,
                "Profile Imported",
                f"Configuration profile imported from:\n{file_path}\n\nRemember to save to apply changes."
            )
            
            logger.info("Profile imported successfully")
            
        except Exception as e:
            logger.error(f"Failed to import profile: {e}")
            QMessageBox.critical(
                self,
                "Import Failed",
                f"Failed to import configuration profile:\n{e}"
            )
    
    def _on_export_clicked(self):
        """Handle export profile button click"""
        logger.info("Export profile requested")
        
        # Generate default filename with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        default_name = f"sentinel_config_{timestamp}.yaml"
        
        # Show file dialog
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Configuration Profile",
            default_name,
            "YAML Files (*.yaml *.yml);;All Files (*)"
        )
        
        if file_path:
            self._export_profile(file_path)
    
    def _export_profile(self, file_path: str):
        """Export configuration profile to file"""
        logger.info(f"Exporting profile: path={file_path}")
        
        try:
            with open(file_path, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False, sort_keys=False)
            
            QMessageBox.information(
                self,
                "Profile Exported",
                f"Configuration profile exported to:\n{file_path}"
            )
            
            logger.info("Profile exported successfully")
            
        except Exception as e:
            logger.error(f"Failed to export profile: {e}")
            QMessageBox.critical(
                self,
                "Export Failed",
                f"Failed to export configuration profile:\n{e}"
            )
    
    def get_config(self) -> Dict[str, Any]:
        """Get current configuration"""
        return self.config
    
    def has_changes(self) -> bool:
        """Check if there are unsaved changes"""
        return self.has_unsaved_changes
