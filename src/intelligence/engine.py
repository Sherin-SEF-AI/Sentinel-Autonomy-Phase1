"""Contextual Intelligence Engine - integrates all risk assessment components."""

import logging
import time
from typing import List, Dict, Any

from src.core.interfaces import IContextualIntelligence
from src.core.data_structures import (
    Detection3D, DriverState, SegmentationOutput, RiskAssessment, VehicleTelemetry
)
from .scene_graph import SceneGraphBuilder
from .attention import AttentionMapper
from .ttc import TTCCalculator
from .trajectory import TrajectoryPredictor
from .risk import RiskCalculator
from .prioritization import RiskPrioritizer
from .advanced_trajectory import AdvancedTrajectoryPredictor
from .advanced_risk import AdvancedRiskAssessor


class ContextualIntelligence(IContextualIntelligence):
    """
    Contextual Intelligence Engine that assesses risks by correlating
    environmental hazards with driver awareness state.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Contextual Intelligence Engine.
        
        Args:
            config: Configuration dictionary with risk_assessment settings
        """
        self.logger = logging.getLogger(__name__)

        # Extract risk assessment config
        risk_config = config.get('risk_assessment', {})
        trajectory_config = risk_config.get('trajectory_prediction', {})

        # Check if advanced features are enabled
        self.use_advanced_prediction = trajectory_config.get('enabled', True)

        # Initialize components
        self.scene_graph_builder = SceneGraphBuilder()
        self.attention_mapper = AttentionMapper(risk_config.get('zone_mapping', {}))
        self.ttc_calculator = TTCCalculator(risk_config.get('ttc_calculation', {}))

        # Use advanced or basic trajectory predictor based on configuration
        if self.use_advanced_prediction:
            self.logger.info("Initializing ADVANCED trajectory predictor")
            self.trajectory_predictor = AdvancedTrajectoryPredictor(trajectory_config)
            self.advanced_risk_assessor = AdvancedRiskAssessor(risk_config)
            self.use_advanced_risk = True
        else:
            self.logger.info("Initializing basic trajectory predictor")
            self.trajectory_predictor = TrajectoryPredictor(trajectory_config)
            self.use_advanced_risk = False

        self.risk_calculator = RiskCalculator(risk_config)
        self.risk_prioritizer = RiskPrioritizer()

        mode = "ADVANCED" if self.use_advanced_prediction else "BASIC"
        self.logger.info(f"Contextual Intelligence Engine initialized in {mode} mode")
    
    def assess(
        self,
        detections: List[Detection3D],
        driver_state: DriverState,
        bev_seg: SegmentationOutput,
        vehicle_telemetry: VehicleTelemetry = None
    ) -> RiskAssessment:
        """
        Assess contextual risks by correlating environmental threats with driver state.
        
        Args:
            detections: List of 3D detections
            driver_state: Current driver state
            bev_seg: BEV segmentation output
            vehicle_telemetry: Optional vehicle telemetry from CAN bus
            
        Returns:
            RiskAssessment with scene graph, hazards, attention map, and top risks
        """
        start_time = time.time()
        
        # Build scene graph
        scene_graph = self.scene_graph_builder.build(detections)
        
        # Add vehicle telemetry to scene graph if available
        if vehicle_telemetry:
            scene_graph['vehicle_telemetry'] = {
                'speed': vehicle_telemetry.speed,
                'steering_angle': vehicle_telemetry.steering_angle,
                'brake_pressure': vehicle_telemetry.brake_pressure,
                'throttle_position': vehicle_telemetry.throttle_position,
                'gear': vehicle_telemetry.gear,
                'turn_signal': vehicle_telemetry.turn_signal,
                'braking': vehicle_telemetry.brake_pressure > 0.1  # Detect braking event
            }
        
        # Map driver attention to spatial zones
        attention_map = self.attention_mapper.map_attention(driver_state)
        
        # Add turn signal information to attention map
        if vehicle_telemetry:
            attention_map['turn_signal'] = vehicle_telemetry.turn_signal
            # Adjust attention based on turn signal
            if vehicle_telemetry.turn_signal == 'left':
                attention_map['intended_zones'] = ['left', 'front_left']
            elif vehicle_telemetry.turn_signal == 'right':
                attention_map['intended_zones'] = ['right', 'front_right']
            else:
                attention_map['intended_zones'] = []
        
        # Extract ego vehicle speed from telemetry
        ego_speed = vehicle_telemetry.speed if vehicle_telemetry else 0.0

        # Process each detection to create hazards and assess risks
        hazards = []
        risks = []

        # Use advanced risk assessment if enabled
        if self.use_advanced_risk and len(detections) > 0:
            # Update trajectory history for better predictions
            self.trajectory_predictor.update_history(detections)

            # Use advanced risk assessor for enhanced hazard detection
            hazards, object_trajectories, collision_probs = \
                self.advanced_risk_assessor.assess_hazards_with_trajectories(
                    detections, ego_trajectory=None, driver_state=driver_state
                )

            # Create risks from hazards
            for hazard in hazards:
                # Check if driver is aware of this hazard
                driver_aware = self.attention_mapper.is_looking_at_zone(attention_map, hazard.zone)

                # Calculate contextual risk
                contextual_score = self.risk_calculator.calculate_contextual_risk(
                    hazard.base_risk, driver_aware, driver_state.readiness_score
                )

                # Create risk object
                risk = self.risk_calculator.create_risk(
                    hazard, contextual_score, driver_aware
                )
                risks.append(risk)

        else:
            # Use basic risk assessment
            for detection in detections:
                # Calculate TTC with ego vehicle speed
                ttc = self.ttc_calculator.calculate_ttc(detection, ego_speed)

                # Predict trajectory
                trajectory = self.trajectory_predictor.predict(detection)

                # Determine spatial zone
                zone = self.attention_mapper.get_zone_for_position(detection.bbox_3d[:3])

                # Calculate trajectory conflict with vehicle path
                # (Simplified: assume vehicle stays stationary or moves forward)
                vehicle_trajectory = [(0, 0, 0)] * len(trajectory)  # Vehicle at origin
                trajectory_conflict = self.trajectory_predictor.calculate_trajectory_conflict_score(
                    trajectory, vehicle_trajectory
                )

                # Calculate base risk
                base_risk = self.risk_calculator.calculate_base_risk(
                    detection, ttc, trajectory_conflict, zone
                )

                # Create hazard
                hazard = self.risk_calculator.create_hazard(
                    detection, ttc, trajectory, zone, base_risk
                )
                hazards.append(hazard)

                # Check if driver is aware of this hazard
                driver_aware = self.attention_mapper.is_looking_at_zone(attention_map, zone)

                # Calculate contextual risk
                contextual_score = self.risk_calculator.calculate_contextual_risk(
                    base_risk, driver_aware, driver_state.readiness_score
                )

                # Create risk object
                risk = self.risk_calculator.create_risk(
                    hazard, contextual_score, driver_aware
                )
                risks.append(risk)
        
        # Filter risks by threshold
        significant_risks = self.risk_prioritizer.filter_by_threshold(
            risks, self.risk_calculator.hazard_threshold
        )
        
        # Prioritize and get top risks
        top_risks = self.risk_prioritizer.prioritize(significant_risks)
        
        # Detect attention-risk mismatches
        mismatches = self.risk_prioritizer.detect_attention_mismatches(
            top_risks, attention_map
        )
        
        # Add mismatch information to attention map
        attention_map['mismatches'] = mismatches
        
        # Create risk assessment
        risk_assessment = RiskAssessment(
            scene_graph=scene_graph,
            hazards=hazards,
            attention_map=attention_map,
            top_risks=top_risks
        )
        
        # Log performance
        elapsed_time = (time.time() - start_time) * 1000  # ms
        self.logger.debug(
            f"Risk assessment completed in {elapsed_time:.1f}ms: "
            f"{len(detections)} detections, {len(hazards)} hazards, "
            f"{len(top_risks)} top risks"
        )
        
        # Check if we're meeting the 10ms target
        if elapsed_time > 10.0:
            self.logger.warning(
                f"Risk assessment exceeded 10ms target: {elapsed_time:.1f}ms"
            )
        
        return risk_assessment
