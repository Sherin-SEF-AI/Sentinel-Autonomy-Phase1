"""
SENTINEL Main Window

Implements the main application window with menu bar, toolbar, status bar,
and central monitoring widget.
"""

import logging
from typing import Optional
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QMenuBar, QMenu, QToolBar,
    QStatusBar, QMessageBox, QApplication, QDockWidget
)
from PyQt6.QtGui import QAction, QKeySequence, QIcon
from PyQt6.QtCore import Qt, QSettings, QTimer

from .widgets import LiveMonitorWidget
from .widgets.driver_state_panel import DriverStatePanel
from .widgets.risk_panel import RiskAssessmentPanel
from .widgets.alerts_panel import AlertsPanel
from .widgets.performance_dock import PerformanceDockWidget
from .themes import ThemeManager
from .workers import SentinelWorker
from src.core.config import ConfigManager

logger = logging.getLogger(__name__)


class SENTINELMainWindow(QMainWindow):
    """
    Main application window for SENTINEL system.
    
    Provides:
    - Menu bar with File, System, View, Tools, Analytics, Help menus
    - Toolbar with quick action buttons
    - Status bar with system status indicators
    - Central monitoring widget
    - Keyboard shortcuts
    """
    
    def __init__(self, theme_manager: ThemeManager, config: ConfigManager):
        super().__init__()
        
        self.settings = QSettings('SENTINEL', 'SentinelGUI')
        self.system_running = False
        self.theme_manager = theme_manager
        self.config = config
        
        # Worker thread
        self.worker: Optional[SentinelWorker] = None
        
        # Initialize UI components
        self._init_ui()
        self._create_dock_widgets()
        self._create_menus()
        self._create_toolbar()
        self._create_status_bar()
        self._setup_shortcuts()
        
        # Apply theme
        self.theme_manager.apply_theme()
        
        # Restore window state
        self._restore_state()
        
        logger.info("SENTINEL Main Window initialized")
    
    def _init_ui(self):
        """Initialize the main UI structure"""
        self.setWindowTitle("SENTINEL - Contextual Safety Intelligence Platform")
        self.setMinimumSize(1280, 720)
        
        # Create central monitoring widget
        self.live_monitor = LiveMonitorWidget()
        self.setCentralWidget(self.live_monitor)
        
        logger.debug("Main UI structure initialized")
    
    def _create_dock_widgets(self):
        """Create all dock widgets"""
        # Driver State Panel
        self.driver_state_dock = QDockWidget("Driver State", self)
        self.driver_state_panel = DriverStatePanel()
        self.driver_state_dock.setWidget(self.driver_state_panel)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.driver_state_dock)
        
        # Risk Assessment Panel
        self.risk_dock = QDockWidget("Risk Assessment", self)
        self.risk_panel = RiskAssessmentPanel()
        self.risk_dock.setWidget(self.risk_panel)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.risk_dock)
        
        # Alerts Panel
        self.alerts_dock = QDockWidget("Alerts", self)
        self.alerts_panel = AlertsPanel()
        self.alerts_dock.setWidget(self.alerts_panel)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.alerts_dock)
        
        # Performance Monitoring Dock
        self.performance_dock = PerformanceDockWidget()
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self.performance_dock)
        
        # Tabify right-side docks
        self.tabifyDockWidget(self.driver_state_dock, self.risk_dock)
        self.tabifyDockWidget(self.risk_dock, self.alerts_dock)
        
        # Show driver state panel by default
        self.driver_state_dock.raise_()
        
        logger.debug("Dock widgets created")
    
    def _create_menus(self):
        """Create menu bar with all menus"""
        menubar = self.menuBar()
        
        # File Menu
        file_menu = menubar.addMenu("&File")
        
        self.action_open_config = QAction("&Open Configuration...", self)
        self.action_open_config.setShortcut(QKeySequence.StandardKey.Open)
        self.action_open_config.triggered.connect(self._on_open_config)
        file_menu.addAction(self.action_open_config)
        
        self.action_save_config = QAction("&Save Configuration", self)
        self.action_save_config.setShortcut(QKeySequence.StandardKey.Save)
        self.action_save_config.triggered.connect(self._on_save_config)
        file_menu.addAction(self.action_save_config)
        
        file_menu.addSeparator()
        
        self.action_export_scenario = QAction("&Export Scenario...", self)
        self.action_export_scenario.triggered.connect(self._on_export_scenario)
        file_menu.addAction(self.action_export_scenario)
        
        file_menu.addSeparator()
        
        self.action_quit = QAction("&Quit", self)
        self.action_quit.setShortcut(QKeySequence("Ctrl+Q"))
        self.action_quit.triggered.connect(self.close)
        file_menu.addAction(self.action_quit)
        
        # System Menu
        system_menu = menubar.addMenu("&System")
        
        self.action_start = QAction("&Start System", self)
        self.action_start.setShortcut(QKeySequence("F5"))
        self.action_start.triggered.connect(self._on_start_system)
        system_menu.addAction(self.action_start)
        
        self.action_stop = QAction("S&top System", self)
        self.action_stop.setShortcut(QKeySequence("F6"))
        self.action_stop.setEnabled(False)
        self.action_stop.triggered.connect(self._on_stop_system)
        system_menu.addAction(self.action_stop)
        
        system_menu.addSeparator()
        
        self.action_restart = QAction("&Restart System", self)
        self.action_restart.triggered.connect(self._on_restart_system)
        system_menu.addAction(self.action_restart)
        
        system_menu.addSeparator()
        
        self.action_calibrate = QAction("&Calibrate Cameras...", self)
        self.action_calibrate.triggered.connect(self._on_calibrate_cameras)
        system_menu.addAction(self.action_calibrate)
        
        # View Menu
        view_menu = menubar.addMenu("&View")
        
        self.action_fullscreen = QAction("&Fullscreen", self)
        self.action_fullscreen.setShortcut(QKeySequence("F11"))
        self.action_fullscreen.setCheckable(True)
        self.action_fullscreen.triggered.connect(self._on_toggle_fullscreen)
        view_menu.addAction(self.action_fullscreen)
        
        view_menu.addSeparator()
        
        # Dock widgets submenu
        self.docks_menu = view_menu.addMenu("&Dock Widgets")
        self.docks_menu.addAction(self.driver_state_dock.toggleViewAction())
        self.docks_menu.addAction(self.risk_dock.toggleViewAction())
        self.docks_menu.addAction(self.alerts_dock.toggleViewAction())
        self.docks_menu.addAction(self.performance_dock.toggleViewAction())
        
        view_menu.addSeparator()
        
        # Theme submenu
        theme_menu = view_menu.addMenu("&Theme")
        
        self.action_dark_theme = QAction("&Dark Theme", self)
        self.action_dark_theme.setCheckable(True)
        self.action_dark_theme.triggered.connect(lambda: self._on_change_theme('dark'))
        theme_menu.addAction(self.action_dark_theme)
        
        self.action_light_theme = QAction("&Light Theme", self)
        self.action_light_theme.setCheckable(True)
        self.action_light_theme.triggered.connect(lambda: self._on_change_theme('light'))
        theme_menu.addAction(self.action_light_theme)
        
        # Set current theme checked
        if self.theme_manager.get_current_theme() == 'dark':
            self.action_dark_theme.setChecked(True)
        else:
            self.action_light_theme.setChecked(True)
        
        theme_menu.addSeparator()
        
        # Accent color submenu
        accent_menu = theme_menu.addMenu("&Accent Color")
        for color_name in self.theme_manager.get_available_accents().keys():
            action = QAction(color_name.capitalize(), self)
            action.triggered.connect(lambda checked, c=color_name: self._on_change_accent(c))
            accent_menu.addAction(action)
        
        view_menu.addSeparator()
        
        # Monitor selection submenu
        self.monitors_menu = view_menu.addMenu("&Move to Monitor")
        self._populate_monitors_menu()
        
        view_menu.addSeparator()
        
        self.action_reset_layout = QAction("&Reset Layout", self)
        self.action_reset_layout.triggered.connect(self._on_reset_layout)
        view_menu.addAction(self.action_reset_layout)
        
        # Tools Menu
        tools_menu = menubar.addMenu("&Tools")
        
        self.action_record = QAction("Start &Recording", self)
        self.action_record.setShortcut(QKeySequence("Ctrl+R"))
        self.action_record.setCheckable(True)
        self.action_record.triggered.connect(self._on_toggle_recording)
        tools_menu.addAction(self.action_record)
        
        self.action_screenshot = QAction("Take &Screenshot", self)
        self.action_screenshot.setShortcut(QKeySequence("Ctrl+S"))
        self.action_screenshot.triggered.connect(self._on_take_screenshot)
        tools_menu.addAction(self.action_screenshot)
        
        tools_menu.addSeparator()
        
        self.action_settings = QAction("&Settings...", self)
        self.action_settings.triggered.connect(self._on_open_settings)
        tools_menu.addAction(self.action_settings)
        
        # Analytics Menu
        analytics_menu = menubar.addMenu("&Analytics")
        
        self.action_trip_report = QAction("&Trip Report...", self)
        self.action_trip_report.triggered.connect(self._on_trip_report)
        analytics_menu.addAction(self.action_trip_report)
        
        self.action_driver_report = QAction("&Driver Report...", self)
        self.action_driver_report.triggered.connect(self._on_driver_report)
        analytics_menu.addAction(self.action_driver_report)
        
        analytics_menu.addSeparator()
        
        self.action_export_data = QAction("&Export Data...", self)
        self.action_export_data.triggered.connect(self._on_export_data)
        analytics_menu.addAction(self.action_export_data)
        
        # Help Menu
        help_menu = menubar.addMenu("&Help")
        
        self.action_documentation = QAction("&Documentation", self)
        self.action_documentation.setShortcut(QKeySequence.StandardKey.HelpContents)
        self.action_documentation.triggered.connect(self._on_documentation)
        help_menu.addAction(self.action_documentation)
        
        self.action_about = QAction("&About SENTINEL", self)
        self.action_about.triggered.connect(self._on_about)
        help_menu.addAction(self.action_about)
        
        logger.debug("Menu bar created")
    
    def _create_toolbar(self):
        """Create toolbar with quick action buttons"""
        toolbar = QToolBar("Main Toolbar")
        toolbar.setMovable(False)
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, toolbar)
        
        # Start button
        toolbar.addAction(self.action_start)
        
        # Stop button
        toolbar.addAction(self.action_stop)
        
        toolbar.addSeparator()
        
        # Record button
        toolbar.addAction(self.action_record)
        
        # Screenshot button
        toolbar.addAction(self.action_screenshot)
        
        toolbar.addSeparator()
        
        # Settings button
        toolbar.addAction(self.action_settings)
        
        logger.debug("Toolbar created")
    
    def _create_status_bar(self):
        """Create status bar with system status indicators"""
        statusbar = QStatusBar()
        self.setStatusBar(statusbar)
        
        # System status message
        self.status_label = "System: Idle"
        statusbar.showMessage(self.status_label)
        
        # Additional status indicators will be added in future subtasks
        
        logger.debug("Status bar created")
    
    def _setup_shortcuts(self):
        """Setup keyboard shortcuts"""
        # Shortcuts are already set in menu actions
        # F5 - Start system
        # F6 - Stop system
        # F11 - Fullscreen
        # Ctrl+Q - Quit
        # Ctrl+R - Record
        # Ctrl+S - Screenshot
        
        logger.debug("Keyboard shortcuts configured")
    
    def _restore_state(self):
        """Restore window state from settings"""
        # Detect available monitors
        self._detect_monitors()
        
        # Restore geometry
        geometry = self.settings.value('geometry')
        if geometry:
            self.restoreGeometry(geometry)
            # Verify window is on a valid screen
            self._ensure_on_valid_screen()
        else:
            # Default size and position
            self.resize(1920, 1080)
            self._center_window()
        
        # Restore window state (dock positions, etc.)
        state = self.settings.value('windowState')
        if state:
            self.restoreState(state)
        
        logger.debug("Window state restored")
    
    def _center_window(self):
        """Center window on screen"""
        screen = QApplication.primaryScreen().geometry()
        window_geometry = self.frameGeometry()
        center_point = screen.center()
        window_geometry.moveCenter(center_point)
        self.move(window_geometry.topLeft())
    
    def _detect_monitors(self):
        """Detect and log available monitors"""
        screens = QApplication.screens()
        logger.info(f"Detected {len(screens)} monitor(s):")
        for i, screen in enumerate(screens):
            geometry = screen.geometry()
            logger.info(f"  Monitor {i}: {screen.name()} - {geometry.width()}x{geometry.height()} at ({geometry.x()}, {geometry.y()})")
    
    def _ensure_on_valid_screen(self):
        """Ensure window is on a valid screen"""
        # Get all available screens
        screens = QApplication.screens()
        
        # Check if window is visible on any screen
        window_rect = self.frameGeometry()
        on_screen = False
        
        for screen in screens:
            screen_rect = screen.geometry()
            if screen_rect.intersects(window_rect):
                on_screen = True
                break
        
        # If not on any screen, move to primary screen
        if not on_screen:
            logger.warning("Window not on any valid screen, moving to primary screen")
            self._center_window()
    
    def _save_state(self):
        """Save window state to settings"""
        self.settings.setValue('geometry', self.saveGeometry())
        self.settings.setValue('windowState', self.saveState())
        
        # Save current screen information
        screen = self.screen()
        if screen:
            self.settings.setValue('screen_name', screen.name())
        
        logger.debug("Window state saved")
    
    def closeEvent(self, event):
        """Handle window close event"""
        # Stop system if running
        if self.system_running:
            reply = QMessageBox.question(
                self,
                'Confirm Exit',
                'System is running. Stop and exit?',
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )
            
            if reply == QMessageBox.StandardButton.Yes:
                self._on_stop_system()
                self._save_state()
                event.accept()
            else:
                event.ignore()
        else:
            self._save_state()
            event.accept()
    
    # Menu action handlers
    
    def _on_open_config(self):
        """Handle open configuration action"""
        logger.info("Open configuration requested")
        QMessageBox.information(self, "Open Configuration", "Configuration dialog will be implemented in future tasks")
    
    def _on_save_config(self):
        """Handle save configuration action"""
        logger.info("Save configuration requested")
        QMessageBox.information(self, "Save Configuration", "Configuration saving will be implemented in future tasks")
    
    def _on_export_scenario(self):
        """Handle export scenario action"""
        logger.info("Export scenario requested")
        QMessageBox.information(self, "Export Scenario", "Scenario export will be implemented in future tasks")
    
    def _on_start_system(self):
        """Handle start system action"""
        logger.info("Starting SENTINEL system")
        
        try:
            # Create and configure worker thread
            self.worker = SentinelWorker(self.config, self)
            
            # Connect all signals to GUI slots
            self._connect_worker_signals()
            
            # Start the worker thread
            self.worker.start()
            
            self.system_running = True
            self.action_start.setEnabled(False)
            self.action_stop.setEnabled(True)
            self.statusBar().showMessage("System: Starting...")
            
            logger.info("SENTINEL system worker thread started")
            
        except Exception as e:
            logger.error(f"Failed to start system: {e}")
            QMessageBox.critical(self, "System Start Error", f"Failed to start system:\n{str(e)}")
            self.system_running = False
            self.action_start.setEnabled(True)
            self.action_stop.setEnabled(False)
    
    def _on_stop_system(self):
        """Handle stop system action"""
        logger.info("Stopping SENTINEL system")
        
        if self.worker:
            # Request worker to stop
            self.worker.stop()
            
            # Wait for worker to finish (with timeout)
            if not self.worker.wait(5000):  # 5 second timeout
                logger.warning("Worker thread did not stop gracefully, terminating...")
                self.worker.terminate()
                self.worker.wait()
            
            self.worker = None
        
        self.system_running = False
        self.action_start.setEnabled(True)
        self.action_stop.setEnabled(False)
        self.statusBar().showMessage("System: Stopped")
        
        # Clear all displays
        self.live_monitor.clear_all_frames()
        
        logger.info("SENTINEL system stopped")
    
    def _on_restart_system(self):
        """Handle restart system action"""
        logger.info("Restarting SENTINEL system")
        if self.system_running:
            self._on_stop_system()
        QTimer.singleShot(500, self._on_start_system)
    
    def _on_calibrate_cameras(self):
        """Handle calibrate cameras action"""
        logger.info("Camera calibration requested")
        QMessageBox.information(self, "Camera Calibration", "Camera calibration dialog will be implemented in future tasks")
    
    def _on_toggle_fullscreen(self, checked):
        """Handle fullscreen toggle"""
        if checked:
            self.showFullScreen()
            logger.info("Entered fullscreen mode")
        else:
            self.showNormal()
            logger.info("Exited fullscreen mode")
    
    def _on_change_theme(self, theme_name: str):
        """Handle theme change"""
        logger.info(f"Changing theme to: {theme_name}")
        self.theme_manager.set_theme(theme_name)
        
        # Update checkboxes
        self.action_dark_theme.setChecked(theme_name == 'dark')
        self.action_light_theme.setChecked(theme_name == 'light')
    
    def _on_change_accent(self, accent_color: str):
        """Handle accent color change"""
        logger.info(f"Changing accent color to: {accent_color}")
        self.theme_manager.set_accent_color(accent_color)
    
    def _populate_monitors_menu(self):
        """Populate the monitors menu with available screens"""
        self.monitors_menu.clear()
        
        screens = QApplication.screens()
        for i, screen in enumerate(screens):
            geometry = screen.geometry()
            action = QAction(f"Monitor {i+1}: {screen.name()} ({geometry.width()}x{geometry.height()})", self)
            action.triggered.connect(lambda checked, idx=i: self._on_move_to_monitor(idx))
            self.monitors_menu.addAction(action)
        
        if len(screens) == 1:
            self.monitors_menu.setEnabled(False)
    
    def _on_move_to_monitor(self, monitor_index: int):
        """Move window to specified monitor"""
        screens = QApplication.screens()
        
        if 0 <= monitor_index < len(screens):
            target_screen = screens[monitor_index]
            screen_geometry = target_screen.geometry()
            
            # Move window to center of target screen
            window_geometry = self.frameGeometry()
            center_point = screen_geometry.center()
            window_geometry.moveCenter(center_point)
            self.move(window_geometry.topLeft())
            
            logger.info(f"Moved window to monitor {monitor_index}: {target_screen.name()}")
        else:
            logger.warning(f"Invalid monitor index: {monitor_index}")
    
    def _on_reset_layout(self):
        """Handle reset layout action"""
        logger.info("Resetting layout")
        # Clear saved state
        self.settings.remove('geometry')
        self.settings.remove('windowState')
        # Reset to default
        self.resize(1920, 1080)
        self._center_window()
        QMessageBox.information(self, "Reset Layout", "Layout has been reset to default")
    
    def _on_toggle_recording(self, checked):
        """Handle recording toggle"""
        if checked:
            logger.info("Starting recording")
            self.statusBar().showMessage("System: Running | Recording: Active")
        else:
            logger.info("Stopping recording")
            self.statusBar().showMessage("System: Running")
    
    def _on_take_screenshot(self):
        """Handle take screenshot action"""
        logger.info("Taking screenshot")
        QMessageBox.information(self, "Screenshot", "Screenshot functionality will be implemented in future tasks")
    
    def _on_open_settings(self):
        """Handle open settings action"""
        logger.info("Opening settings")
        QMessageBox.information(self, "Settings", "Settings dialog will be implemented in future tasks")
    
    def _on_trip_report(self):
        """Handle trip report action"""
        logger.info("Generating trip report")
        QMessageBox.information(self, "Trip Report", "Trip report will be implemented in future tasks")
    
    def _on_driver_report(self):
        """Handle driver report action"""
        logger.info("Generating driver report")
        QMessageBox.information(self, "Driver Report", "Driver report will be implemented in future tasks")
    
    def _on_export_data(self):
        """Handle export data action"""
        logger.info("Exporting data")
        QMessageBox.information(self, "Export Data", "Data export will be implemented in future tasks")
    
    def _on_documentation(self):
        """Handle documentation action"""
        logger.info("Opening documentation")
        QMessageBox.information(self, "Documentation", "Documentation will be available in future releases")
    
    def _on_about(self):
        """Handle about action"""
        about_text = """
        <h2>SENTINEL</h2>
        <p><b>Contextual Safety Intelligence Platform</b></p>
        <p>Version 1.0</p>
        <p>SENTINEL combines 360° Bird's Eye View perception, multi-camera sensor fusion, 
        and driver monitoring to create a holistic safety system that prevents accidents 
        by understanding both the environment and driver state in real-time.</p>
        <p>© 2024 SENTINEL Project</p>
        """
        QMessageBox.about(self, "About SENTINEL", about_text)
    
    # Worker signal connection methods
    
    def _connect_worker_signals(self):
        """Connect all worker thread signals to GUI slots"""
        if not self.worker:
            return
        
        logger.info("Connecting worker signals to GUI slots...")
        
        # Connect frame_ready signal to video displays
        self.worker.frame_ready.connect(self._on_frames_ready)
        
        # Connect bev_ready signal to BEV canvas
        self.worker.bev_ready.connect(self._on_bev_ready)
        
        # Connect detections_ready signal to overlays
        self.worker.detections_ready.connect(self._on_detections_ready)
        
        # Connect driver_state_ready signal to driver panel
        self.worker.driver_state_ready.connect(self._on_driver_state_ready)
        
        # Connect risks_ready signal to risk panel
        self.worker.risks_ready.connect(self._on_risks_ready)
        
        # Connect alerts_ready signal to alerts panel
        self.worker.alerts_ready.connect(self._on_alerts_ready)
        
        # Connect performance_ready signal to performance dock
        self.worker.performance_ready.connect(self._on_performance_ready)
        
        # Connect error and status signals
        self.worker.error_occurred.connect(self._on_worker_error)
        self.worker.status_changed.connect(self._on_worker_status_changed)
        
        logger.info("All worker signals connected")
    
    # Worker signal handler slots
    
    def _on_frames_ready(self, frames: dict):
        """
        Handle frame_ready signal from worker.
        
        Args:
            frames: Dictionary with camera frames
        """
        try:
            # Update video displays
            if 'interior' in frames and frames['interior'] is not None:
                self.live_monitor.update_camera_frame('interior', frames['interior'])
            
            if 'front_left' in frames and frames['front_left'] is not None:
                self.live_monitor.update_camera_frame('front_left', frames['front_left'])
            
            if 'front_right' in frames and frames['front_right'] is not None:
                self.live_monitor.update_camera_frame('front_right', frames['front_right'])
        
        except Exception as e:
            logger.error(f"Error updating camera frames: {e}")
    
    def _on_bev_ready(self, bev_image, bev_mask):
        """
        Handle bev_ready signal from worker.
        
        Args:
            bev_image: BEV image array
            bev_mask: BEV mask array
        """
        try:
            # Update BEV canvas
            self.live_monitor.update_camera_frame('bev', bev_image)
        
        except Exception as e:
            logger.error(f"Error updating BEV: {e}")
    
    def _on_detections_ready(self, detections: list):
        """
        Handle detections_ready signal from worker.
        
        Args:
            detections: List of Detection3D objects
        """
        try:
            # Convert detections to format expected by BEV canvas
            detection_dicts = []
            for det in detections:
                detection_dicts.append({
                    'bbox_3d': det.bbox_3d,
                    'class_name': det.class_name,
                    'confidence': det.confidence,
                    'velocity': det.velocity,
                    'track_id': det.track_id
                })
            
            # Update BEV canvas overlays
            bev_canvas = self.live_monitor.bev_display
            if hasattr(bev_canvas, 'update_objects'):
                bev_canvas.update_objects(detection_dicts)
        
        except Exception as e:
            logger.error(f"Error updating detections: {e}")
    
    def _on_driver_state_ready(self, driver_state):
        """
        Handle driver_state_ready signal from worker.
        
        Args:
            driver_state: DriverState object
        """
        try:
            # Convert driver state to dictionary format
            driver_state_dict = {
                'face_detected': driver_state.face_detected,
                'readiness_score': driver_state.readiness_score,
                'head_pose': driver_state.head_pose,
                'gaze': driver_state.gaze,
                'eye_state': driver_state.eye_state,
                'drowsiness': driver_state.drowsiness,
                'distraction': driver_state.distraction
            }
            
            # Update driver state panel
            self.driver_state_panel.update_driver_state(driver_state_dict)
            
            # Update attention zones on BEV canvas
            bev_canvas = self.live_monitor.bev_display
            if hasattr(bev_canvas, 'update_attention_zones'):
                zones_data = {
                    'current_zone': driver_state.gaze.get('attention_zone', 'front'),
                    'gaze_pitch': driver_state.gaze.get('pitch', 0),
                    'gaze_yaw': driver_state.gaze.get('yaw', 0)
                }
                bev_canvas.update_attention_zones(zones_data)
        
        except Exception as e:
            logger.error(f"Error updating driver state: {e}")
    
    def _on_risks_ready(self, risk_assessment):
        """
        Handle risks_ready signal from worker.
        
        Args:
            risk_assessment: RiskAssessment object
        """
        try:
            # Calculate overall risk score
            overall_risk = 0.0
            if risk_assessment.top_risks:
                overall_risk = max(r.contextual_score for r in risk_assessment.top_risks)
            
            # Update risk panel
            self.risk_panel.update_risk_score(overall_risk)
            
            # Convert hazards to dictionary format
            hazards_list = []
            for risk in risk_assessment.top_risks[:3]:  # Top 3 hazards
                hazard = risk.hazard
                hazards_list.append({
                    'type': hazard.type,
                    'zone': hazard.zone,
                    'ttc': hazard.ttc,
                    'risk_score': risk.contextual_score,
                    'attended': risk.driver_aware
                })
            
            self.risk_panel.update_hazards(hazards_list)
            
            # Update minimum TTC
            if risk_assessment.top_risks:
                min_ttc = min(r.hazard.ttc for r in risk_assessment.top_risks)
                self.risk_panel.update_ttc(min_ttc)
            
            # Update zone risks (8 zones)
            zone_risks = [0.0] * 8
            # TODO: Calculate zone-specific risks from scene graph
            self.risk_panel.update_zone_risks(zone_risks)
        
        except Exception as e:
            logger.error(f"Error updating risks: {e}")
    
    def _on_alerts_ready(self, alerts: list):
        """
        Handle alerts_ready signal from worker.
        
        Args:
            alerts: List of Alert objects
        """
        try:
            # Add alerts to alerts panel
            for alert in alerts:
                alert_dict = {
                    'timestamp': alert.timestamp,
                    'urgency': alert.urgency,
                    'message': alert.message,
                    'modalities': alert.modalities
                }
                self.alerts_panel.add_alert(alert_dict)
        
        except Exception as e:
            logger.error(f"Error updating alerts: {e}")
    
    def _on_performance_ready(self, metrics: dict):
        """
        Handle performance_ready signal from worker.
        
        Args:
            metrics: Dictionary with performance metrics
        """
        try:
            # Update performance dock
            fps = metrics.get('fps', 0.0)
            total_latency = metrics.get('total_latency_ms', 0.0)
            module_latencies = metrics.get('module_latencies', {})
            
            self.performance_dock.update_fps(fps)
            self.performance_dock.update_latency(total_latency)
            self.performance_dock.update_module_timings(module_latencies)
            
            # TODO: Get actual GPU and CPU usage from system
            # For now, use placeholder values
            gpu_memory = 0.0
            cpu_usage = 0.0
            self.performance_dock.update_resources(gpu_memory, cpu_usage)
        
        except Exception as e:
            logger.error(f"Error updating performance: {e}")
    
    def _on_worker_error(self, error_type: str, error_message: str):
        """
        Handle error_occurred signal from worker.
        
        Args:
            error_type: Type of error
            error_message: Error message
        """
        logger.error(f"Worker error ({error_type}): {error_message}")
        
        # Show error in status bar
        self.statusBar().showMessage(f"Error: {error_type} - {error_message}", 5000)
        
        # For fatal errors, show dialog
        if error_type in ['Fatal', 'Initialization']:
            QMessageBox.critical(
                self,
                f"System Error: {error_type}",
                f"A critical error occurred:\n\n{error_message}\n\nThe system will be stopped."
            )
            self._on_stop_system()
    
    def _on_worker_status_changed(self, status: str):
        """
        Handle status_changed signal from worker.
        
        Args:
            status: Status message
        """
        logger.info(f"Worker status: {status}")
        self.statusBar().showMessage(f"System: {status}")
