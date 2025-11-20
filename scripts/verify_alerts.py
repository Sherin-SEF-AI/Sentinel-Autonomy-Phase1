#!/usr/bin/env python3
"""Verification script for Alert & Action System."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from src.alerts import AlertSystem
from src.core.data_structures import (
    RiskAssessment, Risk, Hazard, DriverState
)
from src.core.config import ConfigManager


def verify_alert_generation():
    """Verify alert generation logic."""
    print("Testing Alert Generation...")
    
    config_manager = ConfigManager('configs/default.yaml')
    alert_system = AlertSystem(config_manager.get_section('alerts'))
    
    # Test critical alert
    hazard = Hazard(
        object_id=1,
        type='pedestrian',
        position=(5.0, 0.0, 0.0),
        velocity=(0.0, 0.0, 0.0),
        trajectory=[],
        ttc=0.8,
        zone='front',
        base_risk=0.95
    )
    
    risk = Risk(
        hazard=hazard,
        contextual_score=0.95,
        driver_aware=True,
        urgency='critical',
        intervention_needed=True
    )
    
    risk_assessment = RiskAssessment(
        scene_graph={},
        hazards=[hazard],
        attention_map={},
        top_risks=[risk]
    )
    
    driver = DriverState(
        face_detected=True,
        landmarks=np.zeros((68, 2)),
        head_pose={'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0},
        gaze={'pitch': 0.0, 'yaw': 0.0, 'attention_zone': 'front'},
        eye_state={'left_ear': 0.3, 'right_ear': 0.3, 'perclos': 0.1},
        drowsiness={'score': 0.2, 'yawn_detected': False, 'micro_sleep': False, 'head_nod': False},
        distraction={'type': 'none', 'confidence': 0.0, 'duration': 0.0},
        readiness_score=80.0
    )
    
    alerts = alert_system.process(risk_assessment, driver)
    
    assert len(alerts) == 1, f"Expected 1 alert, got {len(alerts)}"
    assert alerts[0].urgency == 'critical', f"Expected critical, got {alerts[0].urgency}"
    assert 'visual' in alerts[0].modalities
    assert 'audio' in alerts[0].modalities
    assert 'haptic' in alerts[0].modalities
    
    print("✓ Critical alert generation working")
    return True


def verify_alert_suppression():
    """Verify alert suppression logic."""
    print("Testing Alert Suppression...")
    
    config_manager = ConfigManager('configs/default.yaml')
    alert_system = AlertSystem(config_manager.get_section('alerts'))
    
    hazard = Hazard(
        object_id=2,
        type='vehicle',
        position=(8.0, 0.0, 0.0),
        velocity=(10.0, 0.0, 0.0),
        trajectory=[],
        ttc=2.0,
        zone='front',
        base_risk=0.8
    )
    
    risk = Risk(
        hazard=hazard,
        contextual_score=0.85,
        driver_aware=False,
        urgency='high',
        intervention_needed=True
    )
    
    risk_assessment = RiskAssessment(
        scene_graph={},
        hazards=[hazard],
        attention_map={},
        top_risks=[risk]
    )
    
    driver = DriverState(
        face_detected=True,
        landmarks=np.zeros((68, 2)),
        head_pose={'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0},
        gaze={'pitch': 0.0, 'yaw': 0.0, 'attention_zone': 'right'},
        eye_state={'left_ear': 0.3, 'right_ear': 0.3, 'perclos': 0.1},
        drowsiness={'score': 0.3, 'yawn_detected': False, 'micro_sleep': False, 'head_nod': False},
        distraction={'type': 'none', 'confidence': 0.0, 'duration': 0.0},
        readiness_score=60.0
    )
    
    # First alert should pass
    alerts1 = alert_system.process(risk_assessment, driver)
    assert len(alerts1) == 1, f"Expected 1 alert, got {len(alerts1)}"
    
    # Second alert should be suppressed
    alerts2 = alert_system.process(risk_assessment, driver)
    assert len(alerts2) == 0, f"Expected 0 alerts (suppressed), got {len(alerts2)}"
    
    print("✓ Alert suppression working")
    return True


def verify_alert_logging():
    """Verify alert logging functionality."""
    print("Testing Alert Logging...")
    
    config_manager = ConfigManager('configs/default.yaml')
    alert_system = AlertSystem(config_manager.get_section('alerts'))
    
    # Generate some alerts
    for i in range(3):
        hazard = Hazard(
            object_id=100 + i,
            type='vehicle',
            position=(5.0 + i, 0.0, 0.0),
            velocity=(10.0, 0.0, 0.0),
            trajectory=[],
            ttc=1.0 + i * 0.5,
            zone='front',
            base_risk=0.9 - i * 0.1
        )
        
        risk = Risk(
            hazard=hazard,
            contextual_score=0.9 - i * 0.1,
            driver_aware=False,
            urgency='critical' if i == 0 else 'high',
            intervention_needed=True
        )
        
        risk_assessment = RiskAssessment(
            scene_graph={},
            hazards=[hazard],
            attention_map={},
            top_risks=[risk]
        )
        
        driver = DriverState(
            face_detected=True,
            landmarks=np.zeros((68, 2)),
            head_pose={'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0},
            gaze={'pitch': 0.0, 'yaw': 0.0, 'attention_zone': 'front'},
            eye_state={'left_ear': 0.3, 'right_ear': 0.3, 'perclos': 0.1},
            drowsiness={'score': 0.2, 'yawn_detected': False, 'micro_sleep': False, 'head_nod': False},
            distraction={'type': 'none', 'confidence': 0.0, 'duration': 0.0},
            readiness_score=70.0
        )
        
        alert_system.process(risk_assessment, driver)
    
    # Check statistics
    stats = alert_system.get_alert_statistics()
    assert stats['total'] >= 2, f"Expected at least 2 alerts logged, got {stats['total']}"
    
    # Check history
    history = alert_system.get_alert_history()
    assert len(history) >= 2, f"Expected at least 2 alerts in history, got {len(history)}"
    
    print("✓ Alert logging working")
    return True


def verify_multi_modal_dispatch():
    """Verify multi-modal alert dispatch."""
    print("Testing Multi-Modal Dispatch...")
    
    config_manager = ConfigManager('configs/default.yaml')
    alert_system = AlertSystem(config_manager.get_section('alerts'))
    
    hazard = Hazard(
        object_id=3,
        type='pedestrian',
        position=(5.0, 0.0, 0.0),
        velocity=(0.0, 0.0, 0.0),
        trajectory=[],
        ttc=1.0,
        zone='front',
        base_risk=0.95
    )
    
    risk = Risk(
        hazard=hazard,
        contextual_score=0.95,
        driver_aware=True,
        urgency='critical',
        intervention_needed=True
    )
    
    risk_assessment = RiskAssessment(
        scene_graph={},
        hazards=[hazard],
        attention_map={},
        top_risks=[risk]
    )
    
    driver = DriverState(
        face_detected=True,
        landmarks=np.zeros((68, 2)),
        head_pose={'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0},
        gaze={'pitch': 0.0, 'yaw': 0.0, 'attention_zone': 'front'},
        eye_state={'left_ear': 0.3, 'right_ear': 0.3, 'perclos': 0.1},
        drowsiness={'score': 0.2, 'yawn_detected': False, 'micro_sleep': False, 'head_nod': False},
        distraction={'type': 'none', 'confidence': 0.0, 'duration': 0.0},
        readiness_score=80.0
    )
    
    alerts = alert_system.process(risk_assessment, driver)
    
    # Check visual alerts are available
    visual_alerts = alert_system.get_active_visual_alerts()
    assert len(visual_alerts) > 0, "Expected visual alerts to be available"
    
    print("✓ Multi-modal dispatch working")
    return True


def main():
    """Run all verification tests."""
    print("\n" + "="*60)
    print("Alert & Action System Verification")
    print("="*60 + "\n")
    
    tests = [
        verify_alert_generation,
        verify_alert_suppression,
        verify_alert_logging,
        verify_multi_modal_dispatch
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
        except AssertionError as e:
            print(f"✗ {test.__name__} failed: {e}")
            failed += 1
        except Exception as e:
            print(f"✗ {test.__name__} error: {e}")
            failed += 1
    
    print("\n" + "="*60)
    print(f"Results: {passed} passed, {failed} failed")
    print("="*60 + "\n")
    
    if failed == 0:
        print("✓ All verification tests passed!")
        return 0
    else:
        print("✗ Some verification tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
