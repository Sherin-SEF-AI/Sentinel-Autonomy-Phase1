"""Example demonstrating the Alert & Action System."""

import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from src.alerts import AlertSystem
from src.core.data_structures import (
    RiskAssessment, Risk, Hazard, DriverState
)
from src.core.config import ConfigManager


def create_test_hazard(
    object_id: int,
    hazard_type: str,
    position: tuple,
    ttc: float,
    zone: str,
    base_risk: float
) -> Hazard:
    """Create a test hazard."""
    return Hazard(
        object_id=object_id,
        type=hazard_type,
        position=position,
        velocity=(0.0, 0.0, 0.0),
        trajectory=[],
        ttc=ttc,
        zone=zone,
        base_risk=base_risk
    )


def create_test_risk(
    hazard: Hazard,
    contextual_score: float,
    driver_aware: bool
) -> Risk:
    """Create a test risk."""
    # Determine urgency based on score
    if contextual_score > 0.9:
        urgency = 'critical'
    elif contextual_score > 0.7:
        urgency = 'high'
    elif contextual_score > 0.5:
        urgency = 'medium'
    else:
        urgency = 'low'
    
    return Risk(
        hazard=hazard,
        contextual_score=contextual_score,
        driver_aware=driver_aware,
        urgency=urgency,
        intervention_needed=contextual_score > 0.7
    )


def create_test_driver_state(
    readiness_score: float,
    attention_zone: str,
    face_detected: bool = True
) -> DriverState:
    """Create a test driver state."""
    return DriverState(
        face_detected=face_detected,
        landmarks=np.zeros((68, 2)),
        head_pose={'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0},
        gaze={
            'pitch': 0.0,
            'yaw': 0.0,
            'attention_zone': attention_zone
        },
        eye_state={'left_ear': 0.3, 'right_ear': 0.3, 'perclos': 0.1},
        drowsiness={
            'score': 0.2,
            'yawn_detected': False,
            'micro_sleep': False,
            'head_nod': False
        },
        distraction={
            'type': 'none',
            'confidence': 0.0,
            'duration': 0.0
        },
        readiness_score=readiness_score
    )


def demo_critical_alert():
    """Demonstrate critical alert generation."""
    print("\n" + "="*60)
    print("DEMO 1: Critical Alert (risk > 0.9)")
    print("="*60)
    
    # Load config
    config_manager = ConfigManager('configs/default.yaml')
    alert_system = AlertSystem(config_manager.get_section('alerts'))
    
    # Create critical hazard
    hazard = create_test_hazard(
        object_id=1,
        hazard_type='pedestrian',
        position=(5.0, 0.0, 0.0),
        ttc=0.8,
        zone='front',
        base_risk=0.95
    )
    
    risk = create_test_risk(hazard, contextual_score=0.95, driver_aware=True)
    
    risk_assessment = RiskAssessment(
        scene_graph={},
        hazards=[hazard],
        attention_map={},
        top_risks=[risk]
    )
    
    driver = create_test_driver_state(
        readiness_score=80.0,
        attention_zone='front'
    )
    
    # Process alerts
    alerts = alert_system.process(risk_assessment, driver)
    
    print(f"\nGenerated {len(alerts)} alert(s):")
    for alert in alerts:
        print(f"  - Urgency: {alert.urgency.upper()}")
        print(f"    Message: {alert.message}")
        print(f"    Modalities: {', '.join(alert.modalities)}")
        print(f"    Hazard ID: {alert.hazard_id}")


def demo_warning_alert():
    """Demonstrate warning alert when driver unaware."""
    print("\n" + "="*60)
    print("DEMO 2: Warning Alert (risk > 0.7, driver unaware)")
    print("="*60)
    
    config_manager = ConfigManager('configs/default.yaml')
    alert_system = AlertSystem(config_manager.get_section('alerts'))
    
    # Create hazard in left zone
    hazard = create_test_hazard(
        object_id=2,
        hazard_type='vehicle',
        position=(0.0, -3.0, 0.0),
        ttc=1.5,
        zone='left',
        base_risk=0.75
    )
    
    # Driver not aware (looking front, hazard on left)
    risk = create_test_risk(hazard, contextual_score=0.8, driver_aware=False)
    
    risk_assessment = RiskAssessment(
        scene_graph={},
        hazards=[hazard],
        attention_map={},
        top_risks=[risk]
    )
    
    # Driver looking front, not at hazard
    driver = create_test_driver_state(
        readiness_score=70.0,
        attention_zone='front'
    )
    
    alerts = alert_system.process(risk_assessment, driver)
    
    print(f"\nGenerated {len(alerts)} alert(s):")
    for alert in alerts:
        print(f"  - Urgency: {alert.urgency.upper()}")
        print(f"    Message: {alert.message}")
        print(f"    Modalities: {', '.join(alert.modalities)}")


def demo_info_alert():
    """Demonstrate info alert with low cognitive load."""
    print("\n" + "="*60)
    print("DEMO 3: Info Alert (risk > 0.5, low cognitive load)")
    print("="*60)
    
    config_manager = ConfigManager('configs/default.yaml')
    alert_system = AlertSystem(config_manager.get_section('alerts'))
    
    hazard = create_test_hazard(
        object_id=3,
        hazard_type='cyclist',
        position=(10.0, 2.0, 0.0),
        ttc=3.0,
        zone='front-right',
        base_risk=0.55
    )
    
    risk = create_test_risk(hazard, contextual_score=0.6, driver_aware=True)
    
    risk_assessment = RiskAssessment(
        scene_graph={},
        hazards=[hazard],
        attention_map={},
        top_risks=[risk]
    )
    
    # High readiness = low cognitive load
    driver = create_test_driver_state(
        readiness_score=85.0,
        attention_zone='front'
    )
    
    alerts = alert_system.process(risk_assessment, driver)
    
    print(f"\nGenerated {len(alerts)} alert(s):")
    for alert in alerts:
        print(f"  - Urgency: {alert.urgency.upper()}")
        print(f"    Message: {alert.message}")
        print(f"    Modalities: {', '.join(alert.modalities)}")


def demo_alert_suppression():
    """Demonstrate alert suppression."""
    print("\n" + "="*60)
    print("DEMO 4: Alert Suppression")
    print("="*60)
    
    config_manager = ConfigManager('configs/default.yaml')
    alert_system = AlertSystem(config_manager.get_section('alerts'))
    
    hazard = create_test_hazard(
        object_id=4,
        hazard_type='vehicle',
        position=(8.0, 0.0, 0.0),
        ttc=2.0,
        zone='front',
        base_risk=0.8
    )
    
    risk = create_test_risk(hazard, contextual_score=0.85, driver_aware=False)
    
    risk_assessment = RiskAssessment(
        scene_graph={},
        hazards=[hazard],
        attention_map={},
        top_risks=[risk]
    )
    
    driver = create_test_driver_state(
        readiness_score=60.0,
        attention_zone='right'
    )
    
    # First alert
    print("\nFirst alert:")
    alerts1 = alert_system.process(risk_assessment, driver)
    print(f"  Generated {len(alerts1)} alert(s)")
    
    # Immediate second alert (should be suppressed)
    print("\nImmediate second alert (should be suppressed):")
    alerts2 = alert_system.process(risk_assessment, driver)
    print(f"  Generated {len(alerts2)} alert(s)")
    
    # Wait and try again
    print("\nWaiting 2 seconds...")
    time.sleep(2)
    print("Third alert (still within 5s window, should be suppressed):")
    alerts3 = alert_system.process(risk_assessment, driver)
    print(f"  Generated {len(alerts3)} alert(s)")


def demo_multiple_risks():
    """Demonstrate handling multiple risks with max simultaneous limit."""
    print("\n" + "="*60)
    print("DEMO 5: Multiple Risks (max 2 simultaneous)")
    print("="*60)
    
    config_manager = ConfigManager('configs/default.yaml')
    alert_system = AlertSystem(config_manager.get_section('alerts'))
    
    # Create 3 hazards with different priorities
    hazards = [
        create_test_hazard(5, 'pedestrian', (5.0, 0.0, 0.0), 1.0, 'front', 0.9),
        create_test_hazard(6, 'vehicle', (8.0, -2.0, 0.0), 2.0, 'front-left', 0.75),
        create_test_hazard(7, 'cyclist', (12.0, 3.0, 0.0), 3.5, 'front-right', 0.6)
    ]
    
    risks = [
        create_test_risk(hazards[0], 0.92, False),  # Critical
        create_test_risk(hazards[1], 0.78, False),  # Warning
        create_test_risk(hazards[2], 0.62, True)    # Info
    ]
    
    risk_assessment = RiskAssessment(
        scene_graph={},
        hazards=hazards,
        attention_map={},
        top_risks=risks
    )
    
    driver = create_test_driver_state(
        readiness_score=65.0,
        attention_zone='front'
    )
    
    alerts = alert_system.process(risk_assessment, driver)
    
    print(f"\nGenerated {len(alerts)} alert(s) (max 2):")
    for i, alert in enumerate(alerts, 1):
        print(f"\n  Alert {i}:")
        print(f"    Urgency: {alert.urgency.upper()}")
        print(f"    Message: {alert.message}")
        print(f"    Hazard ID: {alert.hazard_id}")


def demo_alert_statistics():
    """Demonstrate alert statistics and history."""
    print("\n" + "="*60)
    print("DEMO 6: Alert Statistics and History")
    print("="*60)
    
    config_manager = ConfigManager('configs/default.yaml')
    alert_system = AlertSystem(config_manager.get_section('alerts'))
    
    # Generate various alerts
    scenarios = [
        (0.95, 'critical'),
        (0.85, 'warning'),
        (0.75, 'warning'),
        (0.6, 'info')
    ]
    
    for i, (score, expected_urgency) in enumerate(scenarios):
        hazard = create_test_hazard(
            object_id=100 + i,
            hazard_type='vehicle',
            position=(5.0 + i, 0.0, 0.0),
            ttc=1.0 + i * 0.5,
            zone='front',
            base_risk=score
        )
        
        risk = create_test_risk(hazard, score, driver_aware=(i % 2 == 0))
        
        risk_assessment = RiskAssessment(
            scene_graph={},
            hazards=[hazard],
            attention_map={},
            top_risks=[risk]
        )
        
        driver = create_test_driver_state(
            readiness_score=70.0 - i * 5,
            attention_zone='front'
        )
        
        alert_system.process(risk_assessment, driver)
        time.sleep(0.1)  # Small delay to avoid suppression
    
    # Get statistics
    stats = alert_system.get_alert_statistics()
    print("\nAlert Statistics:")
    print(f"  Total alerts: {stats['total']}")
    print(f"  By urgency: {stats['by_urgency']}")
    print(f"  Unique hazards: {stats['unique_hazards']}")
    
    # Get recent history
    history = alert_system.get_alert_history(limit=3)
    print(f"\nRecent alerts (last 3):")
    for alert in history:
        print(f"  - {alert['urgency'].upper()}: {alert['message']}")


def main():
    """Run all demos."""
    print("\n" + "="*60)
    print("SENTINEL Alert & Action System Demo")
    print("="*60)
    
    try:
        demo_critical_alert()
        demo_warning_alert()
        demo_info_alert()
        demo_alert_suppression()
        demo_multiple_risks()
        demo_alert_statistics()
        
        print("\n" + "="*60)
        print("All demos completed successfully!")
        print("="*60 + "\n")
        
    except Exception as e:
        print(f"\nError during demo: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
