"""Test suite for Vehicle Telemetry Dock widget module."""

import pytest
from unittest.mock import Mock, patch, MagicMock
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor

from src.gui.widgets.vehicle_telemetry_dock import (
    VehicleTelemetryDock,
    SteeringIndicator,
    BarIndicator,
    GearIndicator,
    TurnSignalIndicator
)
from src.core.data_structures import VehicleTelemetry


@pytest.fixture(scope='module')
def qapp():
    """Create QApplication instance for testing."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


@pytest.fixture
def mock_telemetry():
    """Fixture providing mock vehicle telemetry data."""
    return VehicleTelemetry(
        timestamp=1234567890.0,
        speed=15.0,  # m/s
        steering_angle=0.2,  # radians
        brake_pressure=2.5,  # bar
        throttle_position=0.6,  # 0-1
        gear=3,
        turn_signal='left'
    )


@pytest.fixture
def telemetry_dock(qapp):
    """Fixture creating an instance of VehicleTelemetryDock for testing."""
    dock = VehicleTelemetryDock()
    yield dock
    dock.deleteLater()


@pytest.fixture
def steering_indicator(qapp):
    """Fixture creating a SteeringIndicator for testing."""
    indicator = SteeringIndicator()
    yield indicator
    indicator.deleteLater()


@pytest.fixture
def bar_indicator(qapp):
    """Fixture creating a BarIndicator for testing."""
    indicator = BarIndicator("Test", QColor(255, 0, 0))
    yield indicator
    indicator.deleteLater()


@pytest.fixture
def gear_indicator(qapp):
    """Fixture creating a GearIndicator for testing."""
    indicator = GearIndicator()
    yield indicator
    indicator.deleteLater()


@pytest.fixture
def turn_signal_indicator(qapp):
    """Fixture creating a TurnSignalIndicator for testing."""
    indicator = TurnSignalIndicator()
    yield indicator
    indicator.deleteLater()


class TestSteeringIndicator:
    """Test suite for SteeringIndicator class."""
    
    def test_initialization(self, steering_indicator):
        """Test that SteeringIndicator initializes correctly."""
        assert steering_indicator is not None
        assert steering_indicator.steering_angle == 0.0
        assert steering_indicator.minimumSize().width() == 200
        assert steering_indicator.minimumSize().height() == 100
    
    def test_set_steering_angle(self, steering_indicator):
        """Test setting steering angle."""
        steering_indicator.set_steering_angle(0.5)
        assert steering_indicator.steering_angle == 0.5
        
        steering_indicator.set_steering_angle(-0.3)
        assert steering_indicator.steering_angle == -0.3
    
    def test_set_steering_angle_extreme_values(self, steering_indicator):
        """Test setting extreme steering angles."""
        # Large positive angle
        steering_indicator.set_steering_angle(1.5)
        assert steering_indicator.steering_angle == 1.5
        
        # Large negative angle
        steering_indicator.set_steering_angle(-1.5)
        assert steering_indicator.steering_angle == -1.5
    
    def test_paint_event_executes(self, steering_indicator, qapp):
        """Test that paint event executes without errors."""
        # Trigger paint event
        steering_indicator.show()
        qapp.processEvents()
        steering_indicator.hide()


class TestBarIndicator:
    """Test suite for BarIndicator class."""
    
    def test_initialization(self, bar_indicator):
        """Test that BarIndicator initializes correctly."""
        assert bar_indicator is not None
        assert bar_indicator.label == "Test"
        assert bar_indicator.value == 0.0
        assert bar_indicator.minimumHeight() == 40
    
    def test_set_value_normal_range(self, bar_indicator):
        """Test setting value in normal 0-1 range."""
        bar_indicator.set_value(0.5)
        assert bar_indicator.value == 0.5
        
        bar_indicator.set_value(0.0)
        assert bar_indicator.value == 0.0
        
        bar_indicator.set_value(1.0)
        assert bar_indicator.value == 1.0
    
    def test_set_value_clamping(self, bar_indicator):
        """Test that values are clamped to 0-1 range."""
        # Value above 1.0 should be clamped
        bar_indicator.set_value(1.5)
        assert bar_indicator.value == 1.0
        
        # Value below 0.0 should be clamped
        bar_indicator.set_value(-0.5)
        assert bar_indicator.value == 0.0
    
    def test_paint_event_executes(self, bar_indicator, qapp):
        """Test that paint event executes without errors."""
        bar_indicator.set_value(0.7)
        bar_indicator.show()
        qapp.processEvents()
        bar_indicator.hide()


class TestGearIndicator:
    """Test suite for GearIndicator class."""
    
    def test_initialization(self, gear_indicator):
        """Test that GearIndicator initializes correctly."""
        assert gear_indicator is not None
        assert gear_indicator.gear == 0
        assert gear_indicator.minimumSize().width() == 80
        assert gear_indicator.minimumSize().height() == 80
    
    def test_set_gear_forward(self, gear_indicator):
        """Test setting forward gears."""
        for gear in range(1, 7):
            gear_indicator.set_gear(gear)
            assert gear_indicator.gear == gear
    
    def test_set_gear_neutral(self, gear_indicator):
        """Test setting neutral gear."""
        gear_indicator.set_gear(0)
        assert gear_indicator.gear == 0
    
    def test_set_gear_reverse(self, gear_indicator):
        """Test setting reverse gear."""
        gear_indicator.set_gear(-1)
        assert gear_indicator.gear == -1
    
    def test_paint_event_executes(self, gear_indicator, qapp):
        """Test that paint event executes for different gears."""
        for gear in [-1, 0, 1, 3, 6]:
            gear_indicator.set_gear(gear)
            gear_indicator.show()
            qapp.processEvents()
        gear_indicator.hide()


class TestTurnSignalIndicator:
    """Test suite for TurnSignalIndicator class."""
    
    def test_initialization(self, turn_signal_indicator):
        """Test that TurnSignalIndicator initializes correctly."""
        assert turn_signal_indicator is not None
        assert turn_signal_indicator.turn_signal == 'none'
        assert turn_signal_indicator.minimumSize().width() == 200
        assert turn_signal_indicator.minimumSize().height() == 60
    
    def test_set_turn_signal_left(self, turn_signal_indicator):
        """Test setting left turn signal."""
        turn_signal_indicator.set_turn_signal('left')
        assert turn_signal_indicator.turn_signal == 'left'
    
    def test_set_turn_signal_right(self, turn_signal_indicator):
        """Test setting right turn signal."""
        turn_signal_indicator.set_turn_signal('right')
        assert turn_signal_indicator.turn_signal == 'right'
    
    def test_set_turn_signal_none(self, turn_signal_indicator):
        """Test setting no turn signal."""
        turn_signal_indicator.set_turn_signal('none')
        assert turn_signal_indicator.turn_signal == 'none'
    
    def test_paint_event_executes(self, turn_signal_indicator, qapp):
        """Test that paint event executes for different signals."""
        for signal in ['none', 'left', 'right']:
            turn_signal_indicator.set_turn_signal(signal)
            turn_signal_indicator.show()
            qapp.processEvents()
        turn_signal_indicator.hide()


class TestVehicleTelemetryDock:
    """Test suite for VehicleTelemetryDock class."""
    
    def test_initialization(self, telemetry_dock):
        """Test that VehicleTelemetryDock initializes correctly with valid configuration."""
        assert telemetry_dock is not None
        assert hasattr(telemetry_dock, 'speedometer')
        assert hasattr(telemetry_dock, 'steering_indicator')
        assert hasattr(telemetry_dock, 'brake_bar')
        assert hasattr(telemetry_dock, 'throttle_bar')
        assert hasattr(telemetry_dock, 'gear_indicator')
        assert hasattr(telemetry_dock, 'turn_signal_indicator')
    
    def test_ui_components_created(self, telemetry_dock):
        """Test that all UI components are properly created."""
        # Check speedometer
        assert telemetry_dock.speedometer is not None
        assert telemetry_dock.speedometer.min_value == 0
        assert telemetry_dock.speedometer.max_value == 50
        
        # Check indicators
        assert telemetry_dock.steering_indicator is not None
        assert telemetry_dock.brake_bar is not None
        assert telemetry_dock.throttle_bar is not None
        assert telemetry_dock.gear_indicator is not None
        assert telemetry_dock.turn_signal_indicator is not None
    
    def test_update_telemetry_happy_path(self, telemetry_dock, mock_telemetry):
        """Test updating telemetry with valid data."""
        # Should not raise any exceptions
        telemetry_dock.update_telemetry(mock_telemetry)
        
        # Verify speedometer updated
        assert telemetry_dock.speedometer.value == mock_telemetry.speed
        
        # Verify steering updated
        assert telemetry_dock.steering_indicator.steering_angle == mock_telemetry.steering_angle
        
        # Verify brake updated (normalized)
        expected_brake = min(1.0, mock_telemetry.brake_pressure / 10.0)
        assert telemetry_dock.brake_bar.value == expected_brake
        
        # Verify throttle updated
        assert telemetry_dock.throttle_bar.value == mock_telemetry.throttle_position
        
        # Verify gear updated
        assert telemetry_dock.gear_indicator.gear == mock_telemetry.gear
        
        # Verify turn signal updated
        assert telemetry_dock.turn_signal_indicator.turn_signal == mock_telemetry.turn_signal
    
    def test_update_telemetry_zero_values(self, telemetry_dock):
        """Test updating telemetry with zero values."""
        zero_telemetry = VehicleTelemetry(
            timestamp=0.0,
            speed=0.0,
            steering_angle=0.0,
            brake_pressure=0.0,
            throttle_position=0.0,
            gear=0,
            turn_signal='none'
        )
        
        telemetry_dock.update_telemetry(zero_telemetry)
        
        assert telemetry_dock.speedometer.value == 0.0
        assert telemetry_dock.steering_indicator.steering_angle == 0.0
        assert telemetry_dock.brake_bar.value == 0.0
        assert telemetry_dock.throttle_bar.value == 0.0
        assert telemetry_dock.gear_indicator.gear == 0
    
    def test_update_telemetry_extreme_values(self, telemetry_dock):
        """Test updating telemetry with extreme values."""
        extreme_telemetry = VehicleTelemetry(
            timestamp=9999999999.0,
            speed=100.0,  # Very high speed
            steering_angle=1.57,  # ~90 degrees
            brake_pressure=15.0,  # High brake pressure
            throttle_position=1.0,  # Full throttle
            gear=6,
            turn_signal='right'
        )
        
        telemetry_dock.update_telemetry(extreme_telemetry)
        
        # Speed should be set (even if beyond gauge max)
        assert telemetry_dock.speedometer.value == 100.0
        
        # Brake should be clamped to 1.0
        assert telemetry_dock.brake_bar.value == 1.0
        
        # Throttle at max
        assert telemetry_dock.throttle_bar.value == 1.0
    
    def test_update_telemetry_negative_steering(self, telemetry_dock):
        """Test updating telemetry with negative steering angle."""
        telemetry = VehicleTelemetry(
            timestamp=1234567890.0,
            speed=10.0,
            steering_angle=-0.5,  # Right turn
            brake_pressure=0.0,
            throttle_position=0.5,
            gear=2,
            turn_signal='right'
        )
        
        telemetry_dock.update_telemetry(telemetry)
        assert telemetry_dock.steering_indicator.steering_angle == -0.5
    
    def test_update_telemetry_reverse_gear(self, telemetry_dock):
        """Test updating telemetry with reverse gear."""
        telemetry = VehicleTelemetry(
            timestamp=1234567890.0,
            speed=2.0,
            steering_angle=0.0,
            brake_pressure=0.0,
            throttle_position=0.3,
            gear=-1,  # Reverse
            turn_signal='none'
        )
        
        telemetry_dock.update_telemetry(telemetry)
        assert telemetry_dock.gear_indicator.gear == -1
    
    def test_update_telemetry_error_handling(self, telemetry_dock):
        """Test that update_telemetry handles errors gracefully."""
        # Create invalid telemetry (missing attributes)
        invalid_telemetry = Mock()
        invalid_telemetry.speed = None  # Invalid value
        
        # Should not raise exception, should log error
        try:
            telemetry_dock.update_telemetry(invalid_telemetry)
        except Exception as e:
            pytest.fail(f"update_telemetry raised exception: {e}")
    
    def test_brake_pressure_normalization(self, telemetry_dock):
        """Test brake pressure normalization to 0-1 range."""
        # Test various brake pressures
        test_cases = [
            (0.0, 0.0),
            (5.0, 0.5),
            (10.0, 1.0),
            (15.0, 1.0),  # Should be clamped to 1.0
        ]
        
        for brake_pressure, expected_normalized in test_cases:
            telemetry = VehicleTelemetry(
                timestamp=1234567890.0,
                speed=10.0,
                steering_angle=0.0,
                brake_pressure=brake_pressure,
                throttle_position=0.0,
                gear=1,
                turn_signal='none'
            )
            
            telemetry_dock.update_telemetry(telemetry)
            assert telemetry_dock.brake_bar.value == expected_normalized
    
    def test_multiple_telemetry_updates(self, telemetry_dock):
        """Test multiple consecutive telemetry updates."""
        for i in range(10):
            telemetry = VehicleTelemetry(
                timestamp=1234567890.0 + i,
                speed=float(i * 5),
                steering_angle=float(i * 0.1),
                brake_pressure=float(i * 0.5),
                throttle_position=float(i * 0.1),
                gear=(i % 6) + 1,
                turn_signal=['none', 'left', 'right'][i % 3]
            )
            
            telemetry_dock.update_telemetry(telemetry)
            
            # Verify last update is reflected
            assert telemetry_dock.speedometer.value == float(i * 5)
    
    @pytest.mark.performance
    def test_update_performance(self, telemetry_dock, mock_telemetry):
        """Test that telemetry update completes within performance requirements."""
        import time
        
        # Warm up
        for _ in range(5):
            telemetry_dock.update_telemetry(mock_telemetry)
        
        # Measure performance
        iterations = 100
        start_time = time.perf_counter()
        
        for _ in range(iterations):
            telemetry_dock.update_telemetry(mock_telemetry)
        
        end_time = time.perf_counter()
        
        avg_time_ms = ((end_time - start_time) / iterations) * 1000
        
        # Should be very fast (< 5ms per update)
        assert avg_time_ms < 5.0, f"Average update took {avg_time_ms:.2f}ms, expected < 5ms"
    
    def test_widget_visibility(self, telemetry_dock, qapp):
        """Test that widget can be shown and hidden."""
        telemetry_dock.show()
        qapp.processEvents()
        assert telemetry_dock.isVisible()
        
        telemetry_dock.hide()
        qapp.processEvents()
        assert not telemetry_dock.isVisible()
    
    def test_logging_on_initialization(self, telemetry_dock):
        """Test that initialization is logged."""
        # Logger should be set up
        assert telemetry_dock.logger is not None
        assert telemetry_dock.logger.name == 'src.gui.widgets.vehicle_telemetry_dock'
    
    def test_all_turn_signal_states(self, telemetry_dock):
        """Test all turn signal states."""
        for signal in ['none', 'left', 'right']:
            telemetry = VehicleTelemetry(
                timestamp=1234567890.0,
                speed=10.0,
                steering_angle=0.0,
                brake_pressure=0.0,
                throttle_position=0.5,
                gear=3,
                turn_signal=signal
            )
            
            telemetry_dock.update_telemetry(telemetry)
            assert telemetry_dock.turn_signal_indicator.turn_signal == signal


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
