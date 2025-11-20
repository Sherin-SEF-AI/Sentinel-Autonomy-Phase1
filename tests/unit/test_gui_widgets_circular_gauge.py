"""Test suite for CircularGaugeWidget module."""

import pytest
import time
from unittest.mock import Mock, patch, MagicMock
from PyQt6.QtWidgets import QApplication, QWidget
from PyQt6.QtCore import Qt, QPointF
from PyQt6.QtGui import QPainter, QColor

# Ensure QApplication exists for widget testing
@pytest.fixture(scope="session")
def qapp():
    """Create QApplication instance for testing."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


@pytest.fixture
def circular_gauge(qapp):
    """Fixture creating a CircularGaugeWidget instance for testing."""
    from src.gui.widgets.circular_gauge import CircularGaugeWidget
    
    widget = CircularGaugeWidget(
        min_value=0.0,
        max_value=100.0,
        value=50.0,
        title="Test Gauge",
        unit="%",
        green_zone=(70.0, 100.0),
        yellow_zone=(50.0, 70.0),
        red_zone=(0.0, 50.0)
    )
    return widget


class TestCircularGaugeWidget:
    """Test suite for CircularGaugeWidget class."""
    
    def test_initialization(self, circular_gauge):
        """Test that CircularGaugeWidget initializes correctly with valid configuration."""
        assert circular_gauge is not None
        assert circular_gauge._min_value == 0.0
        assert circular_gauge._max_value == 100.0
        assert circular_gauge._value == 50.0
        assert circular_gauge._animated_value == 50.0
        assert circular_gauge._title == "Test Gauge"
        assert circular_gauge._unit == "%"
        assert circular_gauge._green_zone == (70.0, 100.0)
        assert circular_gauge._yellow_zone == (50.0, 70.0)
        assert circular_gauge._red_zone == (0.0, 50.0)
        assert circular_gauge._start_angle == 135
        assert circular_gauge._span_angle == 270
        assert circular_gauge.minimumSize().width() == 150
        assert circular_gauge.minimumSize().height() == 150
    
    def test_initialization_with_defaults(self, qapp):
        """Test initialization with default parameters."""
        from src.gui.widgets.circular_gauge import CircularGaugeWidget
        
        widget = CircularGaugeWidget()
        assert widget._min_value == 0.0
        assert widget._max_value == 100.0
        assert widget._value == 0.0
        assert widget._title == ""
        assert widget._unit == ""
    
    def test_set_value_without_animation(self, circular_gauge):
        """Test setting value without animation."""
        circular_gauge.set_value(75.0, animate=False)
        
        assert circular_gauge._value == 75.0
        assert circular_gauge._animated_value == 75.0
    
    def test_set_value_with_animation(self, circular_gauge, qapp):
        """Test setting value with animation."""
        initial_value = circular_gauge._value
        circular_gauge.set_value(80.0, animate=True)
        
        assert circular_gauge._value == 80.0
        # Animation should be started
        assert circular_gauge._animation.state() != 0  # Not stopped
    
    def test_set_value_clamping_max(self, circular_gauge):
        """Test that values above max are clamped."""
        circular_gauge.set_value(150.0, animate=False)
        
        assert circular_gauge._value == 100.0
        assert circular_gauge._animated_value == 100.0
    
    def test_set_value_clamping_min(self, circular_gauge):
        """Test that values below min are clamped."""
        circular_gauge.set_value(-50.0, animate=False)
        
        assert circular_gauge._value == 0.0
        assert circular_gauge._animated_value == 0.0
    
    def test_set_value_no_change(self, circular_gauge):
        """Test setting the same value doesn't trigger animation."""
        circular_gauge.set_value(50.0, animate=True)
        
        # Value unchanged, animation should not start
        assert circular_gauge._value == 50.0
    
    def test_get_value(self, circular_gauge):
        """Test getting current gauge value."""
        circular_gauge.set_value(65.0, animate=False)
        
        assert circular_gauge.get_value() == 65.0
    
    def test_set_title(self, circular_gauge):
        """Test setting gauge title."""
        circular_gauge.set_title("New Title")
        
        assert circular_gauge._title == "New Title"
    
    def test_set_unit(self, circular_gauge):
        """Test setting value unit."""
        circular_gauge.set_unit("km/h")
        
        assert circular_gauge._unit == "km/h"
    
    def test_set_color_zones(self, circular_gauge):
        """Test setting color zone ranges."""
        new_green = (80.0, 100.0)
        new_yellow = (60.0, 80.0)
        new_red = (0.0, 60.0)
        
        circular_gauge.set_color_zones(new_green, new_yellow, new_red)
        
        assert circular_gauge._green_zone == new_green
        assert circular_gauge._yellow_zone == new_yellow
        assert circular_gauge._red_zone == new_red
    
    def test_get_color_for_value_green(self, circular_gauge):
        """Test color selection for value in green zone."""
        color = circular_gauge._get_color_for_value(85.0)
        
        assert color == QColor(76, 175, 80)  # Green
    
    def test_get_color_for_value_yellow(self, circular_gauge):
        """Test color selection for value in yellow zone."""
        color = circular_gauge._get_color_for_value(60.0)
        
        assert color == QColor(255, 193, 7)  # Yellow
    
    def test_get_color_for_value_red(self, circular_gauge):
        """Test color selection for value in red zone."""
        color = circular_gauge._get_color_for_value(30.0)
        
        assert color == QColor(244, 67, 54)  # Red
    
    def test_get_color_for_value_outside_zones(self, circular_gauge):
        """Test color selection for value outside all zones."""
        # Set zones that don't cover full range
        circular_gauge.set_color_zones((80.0, 90.0), (60.0, 70.0), (40.0, 50.0))
        
        color = circular_gauge._get_color_for_value(95.0)
        
        assert color == QColor(158, 158, 158)  # Gray default
    
    def test_value_to_angle_min(self, circular_gauge):
        """Test angle conversion for minimum value."""
        angle = circular_gauge._value_to_angle(0.0)
        
        assert angle == 135.0  # Start angle
    
    def test_value_to_angle_max(self, circular_gauge):
        """Test angle conversion for maximum value."""
        angle = circular_gauge._value_to_angle(100.0)
        
        assert angle == 405.0  # Start angle + span angle (135 + 270)
    
    def test_value_to_angle_mid(self, circular_gauge):
        """Test angle conversion for middle value."""
        angle = circular_gauge._value_to_angle(50.0)
        
        expected = 135.0 + (0.5 * 270.0)  # 270 degrees
        assert angle == expected
    
    def test_animated_value_property_getter(self, circular_gauge):
        """Test animatedValue property getter."""
        circular_gauge._animated_value = 42.0
        
        assert circular_gauge.animatedValue == 42.0
    
    def test_animated_value_property_setter(self, circular_gauge):
        """Test animatedValue property setter triggers update."""
        with patch.object(circular_gauge, 'update') as mock_update:
            circular_gauge.animatedValue = 55.0
            
            assert circular_gauge._animated_value == 55.0
            mock_update.assert_called_once()
    
    def test_paint_event_executes(self, circular_gauge, qapp):
        """Test that paintEvent executes without errors."""
        # Create a mock paint event
        from PyQt6.QtGui import QPaintEvent
        from PyQt6.QtCore import QRect
        
        circular_gauge.resize(200, 200)
        event = QPaintEvent(QRect(0, 0, 200, 200))
        
        # Should not raise any exceptions
        try:
            circular_gauge.paintEvent(event)
        except Exception as e:
            pytest.fail(f"paintEvent raised exception: {e}")
    
    def test_paint_event_with_different_sizes(self, circular_gauge, qapp):
        """Test painting with different widget sizes."""
        from PyQt6.QtGui import QPaintEvent
        from PyQt6.QtCore import QRect
        
        sizes = [(150, 150), (200, 200), (300, 300), (200, 300)]
        
        for width, height in sizes:
            circular_gauge.resize(width, height)
            event = QPaintEvent(QRect(0, 0, width, height))
            
            try:
                circular_gauge.paintEvent(event)
            except Exception as e:
                pytest.fail(f"paintEvent failed for size {width}x{height}: {e}")
    
    def test_paint_event_with_different_values(self, circular_gauge, qapp):
        """Test painting with different gauge values."""
        from PyQt6.QtGui import QPaintEvent
        from PyQt6.QtCore import QRect
        
        circular_gauge.resize(200, 200)
        event = QPaintEvent(QRect(0, 0, 200, 200))
        
        test_values = [0.0, 25.0, 50.0, 75.0, 100.0]
        
        for value in test_values:
            circular_gauge.set_value(value, animate=False)
            
            try:
                circular_gauge.paintEvent(event)
            except Exception as e:
                pytest.fail(f"paintEvent failed for value {value}: {e}")
    
    def test_paint_event_without_title(self, qapp):
        """Test painting when title is empty."""
        from src.gui.widgets.circular_gauge import CircularGaugeWidget
        from PyQt6.QtGui import QPaintEvent
        from PyQt6.QtCore import QRect
        
        widget = CircularGaugeWidget(title="")
        widget.resize(200, 200)
        event = QPaintEvent(QRect(0, 0, 200, 200))
        
        try:
            widget.paintEvent(event)
        except Exception as e:
            pytest.fail(f"paintEvent failed without title: {e}")
    
    def test_paint_event_without_unit(self, qapp):
        """Test painting when unit is empty."""
        from src.gui.widgets.circular_gauge import CircularGaugeWidget
        from PyQt6.QtGui import QPaintEvent
        from PyQt6.QtCore import QRect
        
        widget = CircularGaugeWidget(unit="")
        widget.resize(200, 200)
        event = QPaintEvent(QRect(0, 0, 200, 200))
        
        try:
            widget.paintEvent(event)
        except Exception as e:
            pytest.fail(f"paintEvent failed without unit: {e}")
    
    def test_animation_duration(self, circular_gauge):
        """Test that animation has correct duration."""
        assert circular_gauge._animation.duration() == 500  # 500ms
    
    def test_animation_easing_curve(self, circular_gauge):
        """Test that animation uses correct easing curve."""
        from PyQt6.QtCore import QEasingCurve
        
        assert circular_gauge._animation.easingCurve().type() == QEasingCurve.Type.OutCubic
    
    def test_custom_range_gauge(self, qapp):
        """Test gauge with custom min/max range."""
        from src.gui.widgets.circular_gauge import CircularGaugeWidget
        
        widget = CircularGaugeWidget(
            min_value=-50.0,
            max_value=50.0,
            value=0.0
        )
        
        assert widget._min_value == -50.0
        assert widget._max_value == 50.0
        assert widget._value == 0.0
        
        # Test value to angle conversion
        angle_min = widget._value_to_angle(-50.0)
        angle_max = widget._value_to_angle(50.0)
        angle_mid = widget._value_to_angle(0.0)
        
        assert angle_min == 135.0
        assert angle_max == 405.0
        assert angle_mid == 270.0
    
    def test_integer_value_formatting(self, circular_gauge):
        """Test that integer values are formatted correctly."""
        circular_gauge.set_value(42, animate=False)
        
        # Value should be stored as float but formatted as int in display
        assert circular_gauge._value == 42
    
    def test_float_value_formatting(self, circular_gauge):
        """Test that float values are formatted correctly."""
        circular_gauge.set_value(42.7, animate=False)
        
        assert circular_gauge._value == 42.7
    
    @pytest.mark.performance
    def test_paint_performance(self, circular_gauge, qapp):
        """Test that painting completes within performance requirements."""
        from PyQt6.QtGui import QPaintEvent
        from PyQt6.QtCore import QRect
        
        circular_gauge.resize(200, 200)
        event = QPaintEvent(QRect(0, 0, 200, 200))
        
        # Warm up
        circular_gauge.paintEvent(event)
        
        # Measure performance
        iterations = 100
        start_time = time.perf_counter()
        
        for _ in range(iterations):
            circular_gauge.paintEvent(event)
        
        end_time = time.perf_counter()
        avg_time_ms = ((end_time - start_time) / iterations) * 1000
        
        # Should be fast enough for 30 FPS (< 33ms per frame)
        assert avg_time_ms < 33, f"Average paint time {avg_time_ms:.2f}ms exceeds 33ms target"
    
    @pytest.mark.performance
    def test_value_update_performance(self, circular_gauge):
        """Test that value updates are fast."""
        iterations = 1000
        start_time = time.perf_counter()
        
        for i in range(iterations):
            circular_gauge.set_value(float(i % 100), animate=False)
        
        end_time = time.perf_counter()
        avg_time_ms = ((end_time - start_time) / iterations) * 1000
        
        assert avg_time_ms < 1, f"Average update time {avg_time_ms:.2f}ms exceeds 1ms target"
    
    def test_multiple_gauges_independent(self, qapp):
        """Test that multiple gauge instances are independent."""
        from src.gui.widgets.circular_gauge import CircularGaugeWidget
        
        gauge1 = CircularGaugeWidget(value=30.0, title="Gauge 1")
        gauge2 = CircularGaugeWidget(value=70.0, title="Gauge 2")
        
        assert gauge1._value == 30.0
        assert gauge2._value == 70.0
        assert gauge1._title == "Gauge 1"
        assert gauge2._title == "Gauge 2"
        
        # Modify one gauge
        gauge1.set_value(90.0, animate=False)
        
        # Other gauge should be unaffected
        assert gauge1._value == 90.0
        assert gauge2._value == 70.0
    
    def test_zone_boundaries(self, circular_gauge):
        """Test color selection at zone boundaries."""
        # Test exact boundary values
        assert circular_gauge._get_color_for_value(70.0) in [
            QColor(76, 175, 80),   # Green
            QColor(255, 193, 7)    # Yellow
        ]
        
        assert circular_gauge._get_color_for_value(50.0) in [
            QColor(255, 193, 7),   # Yellow
            QColor(244, 67, 54)    # Red
        ]
    
    def test_widget_visibility(self, circular_gauge):
        """Test widget visibility and show/hide."""
        assert not circular_gauge.isVisible()  # Not shown yet
        
        circular_gauge.show()
        assert circular_gauge.isVisible()
        
        circular_gauge.hide()
        assert not circular_gauge.isVisible()
