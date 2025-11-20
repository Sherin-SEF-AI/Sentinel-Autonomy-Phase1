"""
Simple unit tests for Risk Assessment Panel components

Tests individual components without full module dependencies.
"""

import sys
import pytest

# Test that the module can be imported
def test_risk_panel_module_exists():
    """Test that risk_panel module exists"""
    import os
    assert os.path.exists('src/gui/widgets/risk_panel.py')


def test_risk_panel_has_required_classes():
    """Test that risk_panel has all required classes"""
    import ast
    
    with open('src/gui/widgets/risk_panel.py', 'r') as f:
        tree = ast.parse(f.read())
    
    class_names = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
    
    required_classes = [
        'RiskAssessmentPanel',
        'TTCDisplayWidget',
        'HazardListItem',
        'ZoneRiskRadarChart'
    ]
    
    for cls in required_classes:
        assert cls in class_names, f"Missing class: {cls}"


def test_risk_panel_has_required_methods():
    """Test that RiskAssessmentPanel has required methods"""
    import ast
    
    with open('src/gui/widgets/risk_panel.py', 'r') as f:
        tree = ast.parse(f.read())
    
    # Find RiskAssessmentPanel class
    risk_panel_class = None
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == 'RiskAssessmentPanel':
            risk_panel_class = node
            break
    
    assert risk_panel_class is not None
    
    # Get method names
    method_names = [n.name for n in risk_panel_class.body if isinstance(n, ast.FunctionDef)]
    
    required_methods = [
        '__init__',
        '_init_ui',
        'update_risk_score',
        'update_hazards',
        'update_zone_risks',
        'update_ttc',
        'add_alert_event',
        '_update_timeline'
    ]
    
    for method in required_methods:
        assert method in method_names, f"Missing method: {method}"


def test_ttc_widget_has_required_methods():
    """Test that TTCDisplayWidget has required methods"""
    import ast
    
    with open('src/gui/widgets/risk_panel.py', 'r') as f:
        tree = ast.parse(f.read())
    
    # Find TTCDisplayWidget class
    ttc_class = None
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == 'TTCDisplayWidget':
            ttc_class = node
            break
    
    assert ttc_class is not None
    
    # Get method names
    method_names = [n.name for n in ttc_class.body if isinstance(n, ast.FunctionDef)]
    
    required_methods = [
        '__init__',
        'set_ttc',
        '_animate',
        '_get_color_for_ttc',
        'paintEvent'
    ]
    
    for method in required_methods:
        assert method in method_names, f"Missing method: {method}"


def test_hazard_list_item_has_required_methods():
    """Test that HazardListItem has required methods"""
    import ast
    
    with open('src/gui/widgets/risk_panel.py', 'r') as f:
        tree = ast.parse(f.read())
    
    # Find HazardListItem class
    hazard_class = None
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == 'HazardListItem':
            hazard_class = node
            break
    
    assert hazard_class is not None
    
    # Get method names
    method_names = [n.name for n in hazard_class.body if isinstance(n, ast.FunctionDef)]
    
    required_methods = [
        '__init__',
        '_init_ui',
        '_get_icon'
    ]
    
    for method in required_methods:
        assert method in method_names, f"Missing method: {method}"


def test_zone_radar_has_required_methods():
    """Test that ZoneRiskRadarChart has required methods"""
    import ast
    
    with open('src/gui/widgets/risk_panel.py', 'r') as f:
        tree = ast.parse(f.read())
    
    # Find ZoneRiskRadarChart class
    radar_class = None
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == 'ZoneRiskRadarChart':
            radar_class = node
            break
    
    assert radar_class is not None
    
    # Get method names
    method_names = [n.name for n in radar_class.body if isinstance(n, ast.FunctionDef)]
    
    required_methods = [
        '__init__',
        'set_zone_risks',
        'paintEvent'
    ]
    
    for method in required_methods:
        assert method in method_names, f"Missing method: {method}"


def test_risk_panel_imports():
    """Test that risk_panel has correct imports"""
    with open('src/gui/widgets/risk_panel.py', 'r') as f:
        content = f.read()
    
    # Check for required imports
    assert 'from PyQt6.QtWidgets import' in content
    assert 'from PyQt6.QtCore import' in content
    assert 'from PyQt6.QtGui import' in content
    assert 'import pyqtgraph as pg' in content
    assert 'from .circular_gauge import CircularGaugeWidget' in content


def test_risk_panel_exports_in_init():
    """Test that risk_panel classes are exported in __init__.py"""
    with open('src/gui/widgets/__init__.py', 'r') as f:
        content = f.read()
    
    assert 'RiskAssessmentPanel' in content
    assert 'TTCDisplayWidget' in content
    assert 'HazardListItem' in content
    assert 'ZoneRiskRadarChart' in content


def test_risk_panel_docstrings():
    """Test that classes have docstrings"""
    import ast
    
    with open('src/gui/widgets/risk_panel.py', 'r') as f:
        tree = ast.parse(f.read())
    
    classes = ['RiskAssessmentPanel', 'TTCDisplayWidget', 'HazardListItem', 'ZoneRiskRadarChart']
    
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name in classes:
            docstring = ast.get_docstring(node)
            assert docstring is not None, f"Class {node.name} missing docstring"
            assert len(docstring) > 10, f"Class {node.name} has too short docstring"


def test_risk_panel_logging():
    """Test that risk_panel uses logging"""
    with open('src/gui/widgets/risk_panel.py', 'r') as f:
        content = f.read()
    
    assert 'import logging' in content
    assert 'logger = logging.getLogger(__name__)' in content
    assert 'logger.info' in content or 'logger.debug' in content


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
