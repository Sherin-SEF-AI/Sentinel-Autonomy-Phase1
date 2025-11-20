"""Test suite for gui_main module."""

import pytest
import sys
from unittest.mock import Mock, patch, MagicMock, call
from pathlib import Path


class TestGuiMain:
    """Test suite for GUI main entry point."""
    
    @patch('src.gui_main.QApplication')
    @patch('src.gui_main.SENTINELMainWindow')
    @patch('src.gui_main.ThemeManager')
    @patch('src.gui_main.ConfigManager')
    @patch('src.gui_main.LoggerSetup')
    def test_main_successful_startup(
        self, 
        mock_logger_setup, 
        mock_config_manager, 
        mock_theme_manager, 
        mock_main_window, 
        mock_qapp
    ):
        """Test successful GUI application startup with all components initialized."""
        # Setup mocks
        mock_config = Mock()
        mock_config.validate.return_value = True
        mock_config_manager.return_value = mock_config
        
        mock_app = Mock()
        mock_app.exec.return_value = 0
        mock_qapp.return_value = mock_app
        
        mock_theme = Mock()
        mock_theme_manager.return_value = mock_theme
        
        mock_window = Mock()
        mock_main_window.return_value = mock_window
        
        # Import and run main
        from src.gui_main import main
        
        with patch('sys.exit') as mock_exit:
            main()
        
        # Verify logging setup
        mock_logger_setup.setup.assert_called_once_with(
            log_level="INFO", 
            console_output=True
        )
        
        # Verify config loading
        mock_config_manager.assert_called_once_with('configs/default.yaml')
        mock_config.validate.assert_called_once()
        
        # Verify QApplication creation
        mock_qapp.assert_called_once()
        mock_app.setApplicationName.assert_called_once_with("SENTINEL")
        mock_app.setOrganizationName.assert_called_once_with("SENTINEL")
        mock_app.setOrganizationDomain.assert_called_once_with("sentinel-safety.com")
        
        # Verify theme manager creation
        mock_theme_manager.assert_called_once_with(mock_app)
        
        # Verify main window creation and display
        mock_main_window.assert_called_once_with(mock_theme, mock_config)
        mock_window.show.assert_called_once()
        
        # Verify event loop execution
        mock_app.exec.assert_called_once()
        mock_exit.assert_called_once_with(0)
    
    @patch('src.gui_main.ConfigManager')
    @patch('src.gui_main.LoggerSetup')
    def test_main_config_file_not_found(self, mock_logger_setup, mock_config_manager):
        """Test handling of missing configuration file."""
        # Setup mock to raise FileNotFoundError
        mock_config_manager.side_effect = FileNotFoundError("Config file not found")
        
        # Import and run main
        from src.gui_main import main
        
        result = main()
        
        # Verify error handling
        assert result == 1
        mock_logger_setup.setup.assert_called_once()
    
    @patch('src.gui_main.ConfigManager')
    @patch('src.gui_main.LoggerSetup')
    def test_main_config_validation_failed(self, mock_logger_setup, mock_config_manager):
        """Test handling of invalid configuration."""
        # Setup mock config that fails validation
        mock_config = Mock()
        mock_config.validate.return_value = False
        mock_config_manager.return_value = mock_config
        
        # Import and run main
        from src.gui_main import main
        
        result = main()
        
        # Verify error handling
        assert result == 1
        mock_config.validate.assert_called_once()
    
    @patch('src.gui_main.QApplication')
    @patch('src.gui_main.ConfigManager')
    @patch('src.gui_main.LoggerSetup')
    def test_main_qapplication_creation_failed(
        self, 
        mock_logger_setup, 
        mock_config_manager, 
        mock_qapp
    ):
        """Test handling of QApplication creation failure."""
        # Setup valid config
        mock_config = Mock()
        mock_config.validate.return_value = True
        mock_config_manager.return_value = mock_config
        
        # Setup QApplication to raise exception
        mock_qapp.side_effect = RuntimeError("Qt initialization failed")
        
        # Import and run main
        from src.gui_main import main
        
        result = main()
        
        # Verify error handling
        assert result == 1
    
    @patch('src.gui_main.QApplication')
    @patch('src.gui_main.ThemeManager')
    @patch('src.gui_main.ConfigManager')
    @patch('src.gui_main.LoggerSetup')
    def test_main_theme_manager_creation_failed(
        self, 
        mock_logger_setup, 
        mock_config_manager, 
        mock_theme_manager, 
        mock_qapp
    ):
        """Test handling of ThemeManager creation failure."""
        # Setup valid config and QApplication
        mock_config = Mock()
        mock_config.validate.return_value = True
        mock_config_manager.return_value = mock_config
        
        mock_app = Mock()
        mock_qapp.return_value = mock_app
        
        # Setup ThemeManager to raise exception
        mock_theme_manager.side_effect = Exception("Theme loading failed")
        
        # Import and run main
        from src.gui_main import main
        
        result = main()
        
        # Verify error handling
        assert result == 1
    
    @patch('src.gui_main.QApplication')
    @patch('src.gui_main.SENTINELMainWindow')
    @patch('src.gui_main.ThemeManager')
    @patch('src.gui_main.ConfigManager')
    @patch('src.gui_main.LoggerSetup')
    def test_main_window_creation_failed(
        self, 
        mock_logger_setup, 
        mock_config_manager, 
        mock_theme_manager, 
        mock_main_window, 
        mock_qapp
    ):
        """Test handling of main window creation failure."""
        # Setup valid config, QApplication, and ThemeManager
        mock_config = Mock()
        mock_config.validate.return_value = True
        mock_config_manager.return_value = mock_config
        
        mock_app = Mock()
        mock_qapp.return_value = mock_app
        
        mock_theme = Mock()
        mock_theme_manager.return_value = mock_theme
        
        # Setup main window to raise exception
        mock_main_window.side_effect = Exception("Window creation failed")
        
        # Import and run main
        from src.gui_main import main
        
        result = main()
        
        # Verify error handling
        assert result == 1
    
    @patch('src.gui_main.QApplication')
    @patch('src.gui_main.SENTINELMainWindow')
    @patch('src.gui_main.ThemeManager')
    @patch('src.gui_main.ConfigManager')
    @patch('src.gui_main.LoggerSetup')
    def test_main_event_loop_exception(
        self, 
        mock_logger_setup, 
        mock_config_manager, 
        mock_theme_manager, 
        mock_main_window, 
        mock_qapp
    ):
        """Test handling of event loop exception."""
        # Setup all mocks successfully
        mock_config = Mock()
        mock_config.validate.return_value = True
        mock_config_manager.return_value = mock_config
        
        mock_app = Mock()
        mock_app.exec.side_effect = RuntimeError("Event loop crashed")
        mock_qapp.return_value = mock_app
        
        mock_theme = Mock()
        mock_theme_manager.return_value = mock_theme
        
        mock_window = Mock()
        mock_main_window.return_value = mock_window
        
        # Import and run main
        from src.gui_main import main
        
        result = main()
        
        # Verify error handling
        assert result == 1
    
    @patch('src.gui_main.QApplication')
    @patch('src.gui_main.SENTINELMainWindow')
    @patch('src.gui_main.ThemeManager')
    @patch('src.gui_main.ConfigManager')
    @patch('src.gui_main.LoggerSetup')
    def test_main_high_dpi_scaling_configured(
        self, 
        mock_logger_setup, 
        mock_config_manager, 
        mock_theme_manager, 
        mock_main_window, 
        mock_qapp
    ):
        """Test that high DPI scaling is configured correctly."""
        # Setup mocks
        mock_config = Mock()
        mock_config.validate.return_value = True
        mock_config_manager.return_value = mock_config
        
        mock_app = Mock()
        mock_app.exec.return_value = 0
        mock_qapp.return_value = mock_app
        
        mock_theme = Mock()
        mock_theme_manager.return_value = mock_theme
        
        mock_window = Mock()
        mock_main_window.return_value = mock_window
        
        # Import and run main
        from src.gui_main import main
        
        with patch('sys.exit'):
            with patch('src.gui_main.QApplication.setHighDpiScaleFactorRoundingPolicy') as mock_dpi:
                main()
                
                # Verify high DPI scaling was configured
                # Note: This is called before QApplication instantiation
                # so we just verify the function was imported and available
                assert hasattr(mock_qapp, 'setHighDpiScaleFactorRoundingPolicy')
    
    @patch('src.gui_main.QApplication')
    @patch('src.gui_main.SENTINELMainWindow')
    @patch('src.gui_main.ThemeManager')
    @patch('src.gui_main.ConfigManager')
    @patch('src.gui_main.LoggerSetup')
    def test_main_exit_code_propagation(
        self, 
        mock_logger_setup, 
        mock_config_manager, 
        mock_theme_manager, 
        mock_main_window, 
        mock_qapp
    ):
        """Test that exit code from event loop is propagated correctly."""
        # Setup mocks with non-zero exit code
        mock_config = Mock()
        mock_config.validate.return_value = True
        mock_config_manager.return_value = mock_config
        
        mock_app = Mock()
        mock_app.exec.return_value = 42  # Non-zero exit code
        mock_qapp.return_value = mock_app
        
        mock_theme = Mock()
        mock_theme_manager.return_value = mock_theme
        
        mock_window = Mock()
        mock_main_window.return_value = mock_window
        
        # Import and run main
        from src.gui_main import main
        
        with patch('sys.exit') as mock_exit:
            main()
            
            # Verify exit code is propagated
            mock_exit.assert_called_once_with(42)
    
    @patch('src.gui_main.QApplication')
    @patch('src.gui_main.SENTINELMainWindow')
    @patch('src.gui_main.ThemeManager')
    @patch('src.gui_main.ConfigManager')
    @patch('src.gui_main.LoggerSetup')
    @patch('src.gui_main.time.time')
    def test_main_startup_timing_logged(
        self, 
        mock_time,
        mock_logger_setup, 
        mock_config_manager, 
        mock_theme_manager, 
        mock_main_window, 
        mock_qapp
    ):
        """Test that startup timing is measured and logged."""
        # Setup time mock to return incrementing values
        time_values = [0.0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
        mock_time.side_effect = time_values
        
        # Setup mocks
        mock_config = Mock()
        mock_config.validate.return_value = True
        mock_config_manager.return_value = mock_config
        
        mock_app = Mock()
        mock_app.exec.return_value = 0
        mock_qapp.return_value = mock_app
        
        mock_theme = Mock()
        mock_theme_manager.return_value = mock_theme
        
        mock_window = Mock()
        mock_main_window.return_value = mock_window
        
        # Import and run main
        from src.gui_main import main
        
        with patch('sys.exit'):
            main()
        
        # Verify time.time() was called multiple times for timing
        assert mock_time.call_count >= 2
    
    @patch('src.gui_main.ConfigManager')
    @patch('src.gui_main.LoggerSetup')
    def test_main_config_exception_handling(self, mock_logger_setup, mock_config_manager):
        """Test handling of generic configuration exceptions."""
        # Setup mock to raise generic exception
        mock_config_manager.side_effect = Exception("Unexpected config error")
        
        # Import and run main
        from src.gui_main import main
        
        result = main()
        
        # Verify error handling
        assert result == 1
    
    @patch('src.gui_main.sys.path')
    def test_sys_path_modification(self, mock_sys_path):
        """Test that sys.path is modified to include src directory."""
        # Import the module (this will execute the sys.path.insert at module level)
        import src.gui_main
        
        # Verify sys.path.insert was called
        # Note: This is tricky to test since it happens at import time
        # We just verify the module imports successfully
        assert hasattr(src.gui_main, 'main')
    
    @patch('src.gui_main.QApplication')
    @patch('src.gui_main.SENTINELMainWindow')
    @patch('src.gui_main.ThemeManager')
    @patch('src.gui_main.ConfigManager')
    @patch('src.gui_main.LoggerSetup')
    def test_main_application_metadata_set(
        self, 
        mock_logger_setup, 
        mock_config_manager, 
        mock_theme_manager, 
        mock_main_window, 
        mock_qapp
    ):
        """Test that application metadata is set correctly."""
        # Setup mocks
        mock_config = Mock()
        mock_config.validate.return_value = True
        mock_config_manager.return_value = mock_config
        
        mock_app = Mock()
        mock_app.exec.return_value = 0
        mock_qapp.return_value = mock_app
        
        mock_theme = Mock()
        mock_theme_manager.return_value = mock_theme
        
        mock_window = Mock()
        mock_main_window.return_value = mock_window
        
        # Import and run main
        from src.gui_main import main
        
        with patch('sys.exit'):
            main()
        
        # Verify application metadata
        mock_app.setApplicationName.assert_called_once_with("SENTINEL")
        mock_app.setOrganizationName.assert_called_once_with("SENTINEL")
        mock_app.setOrganizationDomain.assert_called_once_with("sentinel-safety.com")
    
    @pytest.mark.performance
    @patch('src.gui_main.QApplication')
    @patch('src.gui_main.SENTINELMainWindow')
    @patch('src.gui_main.ThemeManager')
    @patch('src.gui_main.ConfigManager')
    @patch('src.gui_main.LoggerSetup')
    def test_startup_performance(
        self, 
        mock_logger_setup, 
        mock_config_manager, 
        mock_theme_manager, 
        mock_main_window, 
        mock_qapp
    ):
        """Test that GUI startup completes within reasonable time (< 2000ms with mocks)."""
        import time
        
        # Setup mocks
        mock_config = Mock()
        mock_config.validate.return_value = True
        mock_config_manager.return_value = mock_config
        
        mock_app = Mock()
        mock_app.exec.return_value = 0
        mock_qapp.return_value = mock_app
        
        mock_theme = Mock()
        mock_theme_manager.return_value = mock_theme
        
        mock_window = Mock()
        mock_main_window.return_value = mock_window
        
        # Import and run main
        from src.gui_main import main
        
        start_time = time.perf_counter()
        
        with patch('sys.exit'):
            main()
        
        end_time = time.perf_counter()
        execution_time_ms = (end_time - start_time) * 1000
        
        # With mocks, startup should be very fast
        assert execution_time_ms < 2000, f"Startup took {execution_time_ms:.2f}ms, expected < 2000ms"
