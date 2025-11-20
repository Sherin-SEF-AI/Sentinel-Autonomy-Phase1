"""Multi-object interaction prediction system."""

import logging
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

from src.core.data_structures import Detection3D


class InteractionType(Enum):
    """Types of predicted interactions."""
    NONE = "none"
    PEDESTRIAN_CROSSING = "pedestrian_crossing"
    VEHICLE_LANE_CHANGE = "vehicle_lane_change"
    VEHICLE_MERGE = "vehicle_merge"
    CYCLIST_TURN = "cyclist_turn"
    VEHICLE_OVERTAKE = "vehicle_overtake"
    PEDESTRIAN_JAYWALKING = "pedestrian_jaywalking"
    VEHICLE_CUT_IN = "vehicle_cut_in"
    COLLISION_COURSE = "collision_course"


@dataclass
class PredictedInteraction:
    """Predicted interaction between objects."""
    interaction_type: InteractionType
    primary_object_id: int
    secondary_object_id: Optional[int]  # None for ego vehicle
    confidence: float  # 0-1
    time_to_interaction: float  # seconds
    risk_level: str  # 'low', 'medium', 'high', 'critical'
    description: str


class MultiObjectInteractionPredictor:
    """
    Predicts interactions between detected objects and ego vehicle.

    Uses heuristics and geometric analysis to predict:
    - Pedestrians crossing path
    - Vehicles changing lanes
    - Cyclists turning
    - Potential collisions
    """

    def __init__(self, config: Dict = None):
        """
        Initialize interaction predictor.

        Args:
            config: Configuration dictionary
        """
        self.logger = logging.getLogger(__name__)
        config = config or {}

        # Prediction thresholds
        self.pedestrian_crossing_threshold = config.get('pedestrian_crossing_threshold', 5.0)  # meters
        self.lane_change_angle_threshold = config.get('lane_change_angle_threshold', 15.0)  # degrees
        self.min_confidence = config.get('min_confidence', 0.5)
        self.prediction_horizon = config.get('prediction_horizon', 5.0)  # seconds

        # Interaction history for tracking
        self.interaction_history: Dict[int, List[PredictedInteraction]] = {}

        self.logger.info("Multi-object interaction predictor initialized")

    def predict_interactions(
        self,
        detections: List[Detection3D],
        ego_speed: float = 0.0
    ) -> List[PredictedInteraction]:
        """
        Predict interactions between objects.

        Args:
            detections: List of detected objects
            ego_speed: Ego vehicle speed (m/s)

        Returns:
            List of predicted interactions
        """
        interactions = []

        # Analyze each object
        for detection in detections:
            # Pedestrian crossing prediction
            if detection.class_name == 'pedestrian':
                crossing_interaction = self._predict_pedestrian_crossing(detection, ego_speed)
                if crossing_interaction:
                    interactions.append(crossing_interaction)

            # Vehicle lane change prediction
            elif detection.class_name == 'vehicle':
                lane_change = self._predict_vehicle_lane_change(detection, ego_speed)
                if lane_change:
                    interactions.append(lane_change)

                # Vehicle merge prediction
                merge = self._predict_vehicle_merge(detection, ego_speed)
                if merge:
                    interactions.append(merge)

                # Overtaking prediction
                overtake = self._predict_vehicle_overtake(detection, ego_speed)
                if overtake:
                    interactions.append(overtake)

            # Cyclist turn prediction
            elif detection.class_name == 'cyclist':
                turn = self._predict_cyclist_turn(detection, ego_speed)
                if turn:
                    interactions.append(turn)

        # Predict vehicle-to-vehicle interactions
        interactions.extend(self._predict_vehicle_interactions(detections))

        # Filter by confidence
        interactions = [i for i in interactions if i.confidence >= self.min_confidence]

        # Store in history
        for interaction in interactions:
            obj_id = interaction.primary_object_id
            if obj_id not in self.interaction_history:
                self.interaction_history[obj_id] = []
            self.interaction_history[obj_id].append(interaction)

        return interactions

    def _predict_pedestrian_crossing(
        self,
        detection: Detection3D,
        ego_speed: float
    ) -> Optional[PredictedInteraction]:
        """Predict if pedestrian will cross vehicle path."""
        x, y, z, w, h, l, theta = detection.bbox_3d
        vx, vy, vz = detection.velocity

        # Check if pedestrian is near roadway
        if abs(y) > 5.0:  # Too far from center
            return None

        # Check if moving toward road center
        velocity_toward_center = -np.sign(y) * vy

        if velocity_toward_center > 0.3:  # Moving toward road at > 0.3 m/s
            # Calculate time to reach road center
            distance_to_center = abs(y)
            time_to_center = distance_to_center / max(abs(vy), 0.1)

            # Calculate if ego vehicle will be there
            ego_distance_traveled = ego_speed * time_to_center
            will_intersect = abs(x - ego_distance_traveled) < 3.0  # 3m tolerance

            if will_intersect and time_to_center < self.prediction_horizon:
                # High risk if time is short
                if time_to_center < 2.0:
                    risk_level = 'critical'
                    confidence = 0.9
                elif time_to_center < 3.5:
                    risk_level = 'high'
                    confidence = 0.8
                else:
                    risk_level = 'medium'
                    confidence = 0.6

                return PredictedInteraction(
                    interaction_type=InteractionType.PEDESTRIAN_CROSSING,
                    primary_object_id=detection.track_id,
                    secondary_object_id=None,  # Ego vehicle
                    confidence=confidence,
                    time_to_interaction=time_to_center,
                    risk_level=risk_level,
                    description=f"Pedestrian crossing path in {time_to_center:.1f}s"
                )

        return None

    def _predict_vehicle_lane_change(
        self,
        detection: Detection3D,
        ego_speed: float
    ) -> Optional[PredictedInteraction]:
        """Predict if vehicle is changing lanes."""
        x, y, z, w, h, l, theta = detection.bbox_3d
        vx, vy, vz = detection.velocity

        # Check for lateral movement
        if abs(vy) > 0.5:  # Significant lateral velocity
            # Estimate lane change direction
            if vy > 0:
                direction = "right to left"
                target_y = y + 3.5  # Assume 3.5m lane width
            else:
                direction = "left to right"
                target_y = y - 3.5

            # Check if lane change brings vehicle into ego path
            if abs(target_y) < 2.0:  # Within ego lane
                time_to_lane_change = abs(3.5) / abs(vy)

                # Check relative positioning
                if -5.0 < x < 30.0:  # Ahead of ego
                    if time_to_lane_change < 3.0:
                        risk_level = 'high'
                        confidence = 0.85
                    else:
                        risk_level = 'medium'
                        confidence = 0.7

                    return PredictedInteraction(
                        interaction_type=InteractionType.VEHICLE_LANE_CHANGE,
                        primary_object_id=detection.track_id,
                        secondary_object_id=None,
                        confidence=confidence,
                        time_to_interaction=time_to_lane_change,
                        risk_level=risk_level,
                        description=f"Vehicle changing lanes {direction} in {time_to_lane_change:.1f}s"
                    )

        return None

    def _predict_vehicle_merge(
        self,
        detection: Detection3D,
        ego_speed: float
    ) -> Optional[PredictedInteraction]:
        """Predict if vehicle is merging into traffic."""
        x, y, z, w, h, l, theta = detection.bbox_3d
        vx, vy, vz = detection.velocity

        # Check if vehicle is on side and accelerating forward
        if abs(y) > 2.5 and abs(y) < 6.0:  # In adjacent lane/shoulder
            if vx > ego_speed * 0.5 and vx < ego_speed * 1.5:  # Similar speed
                if abs(vy) > 0.3:  # Moving toward ego lane
                    time_to_merge = abs(y - 1.5) / max(abs(vy), 0.1)

                    if time_to_merge < 4.0:
                        risk_level = 'medium' if time_to_merge > 2.0 else 'high'
                        confidence = 0.75

                        return PredictedInteraction(
                            interaction_type=InteractionType.VEHICLE_MERGE,
                            primary_object_id=detection.track_id,
                            secondary_object_id=None,
                            confidence=confidence,
                            time_to_interaction=time_to_merge,
                            risk_level=risk_level,
                            description=f"Vehicle merging in {time_to_merge:.1f}s"
                        )

        return None

    def _predict_vehicle_overtake(
        self,
        detection: Detection3D,
        ego_speed: float
    ) -> Optional[PredictedInteraction]:
        """Predict if vehicle is overtaking."""
        x, y, z, w, h, l, theta = detection.bbox_3d
        vx, vy, vz = detection.velocity

        # Check if vehicle is behind and faster
        if x < -2.0 and vx > ego_speed * 1.2:  # Behind and faster
            relative_speed = vx - ego_speed
            time_to_overtake = abs(x) / max(relative_speed, 0.1)

            if time_to_overtake < 5.0:
                return PredictedInteraction(
                    interaction_type=InteractionType.VEHICLE_OVERTAKE,
                    primary_object_id=detection.track_id,
                    secondary_object_id=None,
                    confidence=0.7,
                    time_to_interaction=time_to_overtake,
                    risk_level='low',
                    description=f"Vehicle overtaking in {time_to_overtake:.1f}s"
                )

        return None

    def _predict_cyclist_turn(
        self,
        detection: Detection3D,
        ego_speed: float
    ) -> Optional[PredictedInteraction]:
        """Predict if cyclist is turning."""
        x, y, z, w, h, l, theta = detection.bbox_3d
        vx, vy, vz = detection.velocity

        # Check for turning motion (change in direction)
        speed = np.sqrt(vx**2 + vy**2)

        if speed > 2.0 and abs(vy) > 0.5:  # Moving with lateral component
            # Predict turn into path
            if 0 < x < 20.0 and abs(y) < 4.0:
                time_to_turn = 2.0  # Assume 2s for turn

                return PredictedInteraction(
                    interaction_type=InteractionType.CYCLIST_TURN,
                    primary_object_id=detection.track_id,
                    secondary_object_id=None,
                    confidence=0.65,
                    time_to_interaction=time_to_turn,
                    risk_level='medium',
                    description="Cyclist turning into path"
                )

        return None

    def _predict_vehicle_interactions(
        self,
        detections: List[Detection3D]
    ) -> List[PredictedInteraction]:
        """Predict vehicle-to-vehicle interactions."""
        interactions = []

        vehicles = [d for d in detections if d.class_name == 'vehicle']

        for i, v1 in enumerate(vehicles):
            for v2 in vehicles[i+1:]:
                interaction = self._check_collision_course(v1, v2)
                if interaction:
                    interactions.append(interaction)

        return interactions

    def _check_collision_course(
        self,
        v1: Detection3D,
        v2: Detection3D
    ) -> Optional[PredictedInteraction]:
        """Check if two vehicles are on collision course."""
        x1, y1, _, _, _, _, _ = v1.bbox_3d
        x2, y2, _, _, _, _, _ = v2.bbox_3d
        vx1, vy1, _ = v1.velocity
        vx2, vy2, _ = v2.velocity

        # Predict future positions (simple linear extrapolation)
        t = 2.0  # Look ahead 2 seconds

        future_x1 = x1 + vx1 * t
        future_y1 = y1 + vy1 * t
        future_x2 = x2 + vx2 * t
        future_y2 = y2 + vy2 * t

        # Check distance at future time
        future_distance = np.sqrt((future_x1 - future_x2)**2 + (future_y1 - future_y2)**2)

        if future_distance < 3.0:  # Within 3 meters
            return PredictedInteraction(
                interaction_type=InteractionType.COLLISION_COURSE,
                primary_object_id=v1.track_id,
                secondary_object_id=v2.track_id,
                confidence=0.8,
                time_to_interaction=t,
                risk_level='high',
                description=f"Vehicles on collision course"
            )

        return None

    def get_critical_interactions(
        self,
        interactions: List[PredictedInteraction]
    ) -> List[PredictedInteraction]:
        """Filter for only critical interactions."""
        return [i for i in interactions if i.risk_level in ['high', 'critical']]
