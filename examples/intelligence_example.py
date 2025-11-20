"""Example demonstrating the Contextual Intelligence Engine."""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from src.intelligence import ContextualIntelligence
from src.core.data_structures import Detection3D, DriverState, SegmentationOutput
from src.core.config import ConfigManager


def create_sample_scenario():
    """Create a sample scenario with detections and driver state."""
    
    # Create detections
    detections = [
        # Vehicle approaching from front
        Detection3D(
            bbox_3d=(15.0, 0.0, 0.0, 2.0, 1.5, 4.5, 0.0),
            class_name='vehicle',
            confidence=0.92,
            velocity=(-8.0, 0.0, 0.0),  # Approaching at 8 m/s
            track_id=1
        ),
        # Pedestrian on the right side
        Detection3D(
            bbox_3d=(8.0, -4.0, 0.0, 0.5, 1.8, 0.5, 0.0),
            class_name='pedestrian',
            confidence=0.88,
            velocity=(0.0, 2.0, 0.0),  # Moving left (toward road)
            track_id=2
        ),
        # Cyclist on the left
        Detection3D(
            bbox_3d=(12.0, 3.0, 0.0, 0.6, 1.7, 1.8, 0.0),
            class_name='cyclist',
            confidence=0.85,
            velocity=(-3.0, -0.5, 0.0),  # Moving forward and slightly right
            track_id=3
        ),
        # Stationary vehicle far away
        Detection3D(
            bbox_3d=(30.0, 1.0, 0.0, 2.0, 1.5, 4.5, 0.0),
            class_name='vehicle',
            confidence=0.90,
            velocity=(0.0, 0.0, 0.0),  # Stationary
            track_id=4
        )
    ]
    
    # Driver looking forward (aware of front hazards)
    driver_state_aware = DriverState(
        face_detected=True,
        landmarks=np.zeros((68, 2)),
        head_pose={'roll': 0.0, 'pitch': 5.0, 'yaw': 0.0},
        gaze={'pitch': 5.0, 'yaw': 0.0, 'attention_zone': 'front'},
        eye_state={'left_ear': 0.28, 'right_ear': 0.30, 'perclos': 0.12},
        drowsiness={'score': 0.15, 'yawn_detected': False, 'micro_sleep': False, 'head_nod': False},
        distraction={'type': 'none', 'confidence': 0.0, 'duration': 0.0},
        readiness_score=82.0
    )
    
    # Driver looking at phone (distracted, not aware)
    driver_state_distracted = DriverState(
        face_detected=True,
        landmarks=np.zeros((68, 2)),
        head_pose={'roll': 0.0, 'pitch': -25.0, 'yaw': 0.0},
        gaze={'pitch': -30.0, 'yaw': 0.0, 'attention_zone': 'down'},
        eye_state={'left_ear': 0.32, 'right_ear': 0.31, 'perclos': 0.18},
        drowsiness={'score': 0.25, 'yawn_detected': False, 'micro_sleep': False, 'head_nod': False},
        distraction={'type': 'phone', 'confidence': 0.85, 'duration': 2.5},
        readiness_score=45.0
    )
    
    # BEV segmentation (simplified)
    bev_seg = SegmentationOutput(
        timestamp=0.0,
        class_map=np.zeros((640, 640), dtype=np.int8),
        confidence=np.ones((640, 640), dtype=np.float32) * 0.9
    )
    
    return detections, driver_state_aware, driver_state_distracted, bev_seg


def print_risk_assessment(assessment, scenario_name):
    """Print risk assessment results."""
    print(f"\n{'='*70}")
    print(f"SCENARIO: {scenario_name}")
    print(f"{'='*70}")
    
    # Scene graph
    print(f"\nScene Graph:")
    print(f"  Objects detected: {assessment.scene_graph['num_objects']}")
    print(f"  Relationships: {len(assessment.scene_graph['relationships'])}")
    
    # Attention map
    print(f"\nDriver Attention:")
    if assessment.attention_map['attention_valid']:
        print(f"  Primary zone: {assessment.attention_map['primary_zone']}")
        print(f"  Attended zones: {', '.join(assessment.attention_map['attended_zones'])}")
        print(f"  Gaze: yaw={assessment.attention_map['gaze_yaw']:.1f}°, "
              f"pitch={assessment.attention_map['gaze_pitch']:.1f}°")
        
        if assessment.attention_map.get('mismatches'):
            print(f"  ⚠️  Attention-risk mismatches: {len(assessment.attention_map['mismatches'])}")
            for mismatch in assessment.attention_map['mismatches']:
                print(f"     - {mismatch['hazard_type']} in {mismatch['hazard_zone']} "
                      f"(risk: {mismatch['contextual_score']:.2f})")
    else:
        print("  ⚠️  Attention data not valid")
    
    # Hazards
    print(f"\nHazards Detected: {len(assessment.hazards)}")
    for i, hazard in enumerate(assessment.hazards, 1):
        print(f"  {i}. {hazard.type} (ID: {hazard.object_id})")
        print(f"     Position: ({hazard.position[0]:.1f}, {hazard.position[1]:.1f}, {hazard.position[2]:.1f})m")
        print(f"     Zone: {hazard.zone}")
        print(f"     TTC: {hazard.ttc:.2f}s" if hazard.ttc != float('inf') else "     TTC: ∞")
        print(f"     Base risk: {hazard.base_risk:.3f}")
    
    # Top risks
    print(f"\nTop {len(assessment.top_risks)} Risks:")
    for i, risk in enumerate(assessment.top_risks, 1):
        print(f"  {i}. {risk.hazard.type} (ID: {risk.hazard.object_id})")
        print(f"     Contextual score: {risk.contextual_score:.3f}")
        print(f"     Urgency: {risk.urgency.upper()}")
        print(f"     Driver aware: {'✓' if risk.driver_aware else '✗'}")
        print(f"     Intervention needed: {'YES' if risk.intervention_needed else 'NO'}")


def main():
    """Run intelligence engine example."""
    print("Contextual Intelligence Engine Example")
    print("=" * 70)
    
    # Load configuration
    config_manager = ConfigManager('configs/default.yaml')
    config = config_manager.config
    
    # Initialize intelligence engine
    print("\nInitializing Contextual Intelligence Engine...")
    intelligence = ContextualIntelligence(config)
    print("✓ Engine initialized")
    
    # Create sample scenario
    detections, driver_aware, driver_distracted, bev_seg = create_sample_scenario()
    
    print(f"\nScenario setup:")
    print(f"  - {len(detections)} objects detected")
    print(f"  - Testing with 2 driver states: aware vs. distracted")
    
    # Scenario 1: Driver is aware (looking forward)
    print("\n" + "="*70)
    print("Running assessment with AWARE driver...")
    assessment_aware = intelligence.assess(detections, driver_aware, bev_seg)
    print_risk_assessment(assessment_aware, "Driver Aware (Looking Forward)")
    
    # Scenario 2: Driver is distracted (looking at phone)
    print("\n" + "="*70)
    print("Running assessment with DISTRACTED driver...")
    assessment_distracted = intelligence.assess(detections, driver_distracted, bev_seg)
    print_risk_assessment(assessment_distracted, "Driver Distracted (Looking at Phone)")
    
    # Comparison
    print(f"\n{'='*70}")
    print("COMPARISON")
    print(f"{'='*70}")
    
    if assessment_aware.top_risks and assessment_distracted.top_risks:
        max_risk_aware = max(r.contextual_score for r in assessment_aware.top_risks)
        max_risk_distracted = max(r.contextual_score for r in assessment_distracted.top_risks)
        
        print(f"\nMaximum contextual risk:")
        print(f"  Aware driver:      {max_risk_aware:.3f}")
        print(f"  Distracted driver: {max_risk_distracted:.3f}")
        print(f"  Risk increase:     {((max_risk_distracted / max_risk_aware - 1) * 100):.1f}%")
        
        print(f"\nKey insight:")
        print(f"  When the driver is distracted, the same environmental hazards")
        print(f"  result in significantly higher contextual risk scores because")
        print(f"  the system accounts for reduced driver awareness and readiness.")
    
    print(f"\n{'='*70}")
    print("Example completed successfully!")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
