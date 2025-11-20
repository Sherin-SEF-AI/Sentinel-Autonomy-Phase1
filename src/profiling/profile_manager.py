"""
Profile Manager for Driver Behavior Profiles

Manages loading, saving, and updating driver profiles with persistence.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
from dataclasses import dataclass, asdict
import logging

from .face_recognition import FaceRecognitionSystem
from .metrics_tracker import MetricsTracker
from .style_classifier import DrivingStyle, DrivingStyleClassifier
from .threshold_adapter import ThresholdAdapter
from .report_generator import DriverReportGenerator

logger = logging.getLogger(__name__)


@dataclass
class DriverProfile:
    """Complete driver behavior profile."""
    driver_id: str
    face_embedding: List[float]  # Stored as list for JSON serialization
    total_distance: float = 0.0
    total_time: float = 0.0
    driving_style: str = "unknown"
    
    # Aggregated metrics
    avg_reaction_time: float = 0.0
    avg_following_distance: float = 0.0
    avg_lane_change_freq: float = 0.0
    avg_speed: float = 0.0
    risk_tolerance: float = 0.5
    
    # Scores
    safety_score: float = 0.0
    attention_score: float = 0.0
    eco_score: float = 0.0
    
    # Metadata
    session_count: int = 0
    last_updated: str = ""
    created_at: str = ""


class ProfileManager:
    """
    Manages driver profiles with persistence.
    
    Handles:
    - Loading profiles from disk
    - Saving profiles to disk
    - Updating profiles after sessions
    - Managing multiple driver profiles
    """
    
    def __init__(self, config: dict):
        """
        Initialize profile manager.
        
        Args:
            config: Configuration dictionary with:
                - profiles_dir: Directory for storing profiles
                - auto_save: Whether to auto-save after updates
        """
        self.config = config
        self.profiles_dir = Path(config.get('profiles_dir', 'profiles'))
        self.auto_save = config.get('auto_save', True)
        
        # Create profiles directory if it doesn't exist
        self.profiles_dir.mkdir(parents=True, exist_ok=True)
        
        # In-memory profile storage
        self.profiles: Dict[str, DriverProfile] = {}
        
        # Current active profile
        self.active_profile_id: Optional[str] = None
        
        # Initialize components
        self.face_recognition = FaceRecognitionSystem(config.get('face_recognition', {}))
        self.metrics_tracker = MetricsTracker(config.get('metrics_tracker', {}))
        self.style_classifier = DrivingStyleClassifier(config.get('style_classifier', {}))
        self.threshold_adapter = ThresholdAdapter(config.get('threshold_adapter', {}))
        self.report_generator = DriverReportGenerator(config.get('report_generator', {}))
        
        # Load existing profiles
        self._load_all_profiles()
        
        logger.info(f"ProfileManager initialized with {len(self.profiles)} profiles")
    
    def _load_all_profiles(self):
        """Load all profiles from disk."""
        if not self.profiles_dir.exists():
            return
        
        for profile_file in self.profiles_dir.glob('*.json'):
            try:
                profile = self._load_profile_from_file(profile_file)
                if profile:
                    self.profiles[profile.driver_id] = profile
                    logger.info(f"Loaded profile: {profile.driver_id}")
            except Exception as e:
                logger.error(f"Failed to load profile from {profile_file}: {e}")
    
    def _load_profile_from_file(self, filepath: Path) -> Optional[DriverProfile]:
        """Load a single profile from file."""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            # Convert to DriverProfile
            profile = DriverProfile(**data)
            return profile
        except Exception as e:
            logger.error(f"Error loading profile from {filepath}: {e}")
            return None
    
    def save_profile(self, driver_id: str):
        """
        Save a profile to disk.
        
        Args:
            driver_id: Driver identifier
        """
        if driver_id not in self.profiles:
            logger.warning(f"Profile {driver_id} not found, cannot save")
            return
        
        profile = self.profiles[driver_id]
        filepath = self.profiles_dir / f"{driver_id}.json"
        
        try:
            # Convert to dict
            profile_dict = asdict(profile)
            
            # Save to file
            with open(filepath, 'w') as f:
                json.dump(profile_dict, f, indent=2)
            
            logger.info(f"Profile saved: {driver_id}")
        except Exception as e:
            logger.error(f"Failed to save profile {driver_id}: {e}")
    
    def save_all_profiles(self):
        """Save all profiles to disk."""
        for driver_id in self.profiles:
            self.save_profile(driver_id)
    
    def identify_driver(self, frame: np.ndarray) -> Optional[str]:
        """
        Identify driver from face in frame.
        
        Args:
            frame: Camera frame with driver face
        
        Returns:
            Driver ID if identified, None otherwise
        """
        # Extract face embedding
        embedding = self.face_recognition.extract_face_embedding(frame)
        if embedding is None:
            logger.warning("No face detected for driver identification")
            return None
        
        # Get stored embeddings
        stored_embeddings = {
            driver_id: np.array(profile.face_embedding)
            for driver_id, profile in self.profiles.items()
        }
        
        # Match face
        driver_id, similarity = self.face_recognition.match_face(embedding, stored_embeddings)
        
        if driver_id is not None:
            self.active_profile_id = driver_id
            logger.info(f"Driver identified: {driver_id} (similarity={similarity:.3f})")
            return driver_id
        else:
            # New driver - create profile
            new_driver_id = self.face_recognition.generate_driver_id(embedding)
            self._create_new_profile(new_driver_id, embedding)
            self.active_profile_id = new_driver_id
            logger.info(f"New driver profile created: {new_driver_id}")
            return new_driver_id
    
    def _create_new_profile(self, driver_id: str, face_embedding: np.ndarray):
        """Create a new driver profile."""
        now = datetime.now().isoformat()
        
        profile = DriverProfile(
            driver_id=driver_id,
            face_embedding=face_embedding.tolist(),
            created_at=now,
            last_updated=now
        )
        
        self.profiles[driver_id] = profile
        
        if self.auto_save:
            self.save_profile(driver_id)
    
    def start_session(self, driver_id: str, timestamp: float):
        """
        Start a tracking session for a driver.
        
        Args:
            driver_id: Driver identifier
            timestamp: Session start timestamp
        """
        if driver_id not in self.profiles:
            logger.warning(f"Profile {driver_id} not found")
            return
        
        self.active_profile_id = driver_id
        self.metrics_tracker.start_session(timestamp)
        logger.info(f"Session started for driver {driver_id}")
    
    def end_session(self, timestamp: float):
        """
        End the current tracking session and update profile.
        
        Args:
            timestamp: Session end timestamp
        """
        if self.active_profile_id is None:
            logger.warning("No active profile to end session")
            return
        
        self.metrics_tracker.end_session(timestamp)
        
        # Update profile with session data
        self._update_profile_from_session()
        
        # Reset metrics tracker
        self.metrics_tracker.reset()
        
        logger.info(f"Session ended for driver {self.active_profile_id}")
        
        self.active_profile_id = None
    
    def _update_profile_from_session(self):
        """Update active profile with current session metrics."""
        if self.active_profile_id is None:
            return
        
        profile = self.profiles[self.active_profile_id]
        
        # Get session metrics
        metrics = self.metrics_tracker.get_summary()
        
        # Classify driving style
        driving_style = self.style_classifier.classify(metrics)
        
        # Generate report
        report = self.report_generator.generate_report(
            metrics, driving_style, self.active_profile_id
        )
        
        # Update profile with weighted average of old and new data
        session_weight = 0.3  # Weight for new session data
        
        # Update aggregated metrics
        if profile.session_count > 0:
            profile.avg_reaction_time = (
                profile.avg_reaction_time * (1 - session_weight) +
                metrics['reaction_time']['mean'] * session_weight
            )
            profile.avg_following_distance = (
                profile.avg_following_distance * (1 - session_weight) +
                metrics['following_distance']['mean'] * session_weight
            )
            profile.avg_lane_change_freq = (
                profile.avg_lane_change_freq * (1 - session_weight) +
                metrics['lane_change_frequency'] * session_weight
            )
            profile.avg_speed = (
                profile.avg_speed * (1 - session_weight) +
                metrics['speed_profile']['mean'] * session_weight
            )
            profile.risk_tolerance = (
                profile.risk_tolerance * (1 - session_weight) +
                metrics['risk_tolerance'] * session_weight
            )
        else:
            # First session - use session data directly
            profile.avg_reaction_time = metrics['reaction_time']['mean']
            profile.avg_following_distance = metrics['following_distance']['mean']
            profile.avg_lane_change_freq = metrics['lane_change_frequency']
            profile.avg_speed = metrics['speed_profile']['mean']
            profile.risk_tolerance = metrics['risk_tolerance']
        
        # Update scores
        profile.safety_score = report['scores']['safety']
        profile.attention_score = report['scores']['attention']
        profile.eco_score = report['scores']['eco_driving']
        
        # Update totals
        profile.total_distance += metrics['total_distance']
        profile.total_time += metrics['session_duration']
        profile.session_count += 1
        
        # Update driving style
        profile.driving_style = driving_style.value
        
        # Update timestamp
        profile.last_updated = datetime.now().isoformat()
        
        logger.info(f"Profile updated for {self.active_profile_id}: "
                   f"sessions={profile.session_count}, "
                   f"safety={profile.safety_score:.1f}, "
                   f"style={profile.driving_style}")
        
        # Auto-save if enabled
        if self.auto_save:
            self.save_profile(self.active_profile_id)
    
    def get_profile(self, driver_id: str) -> Optional[DriverProfile]:
        """
        Get a driver profile.
        
        Args:
            driver_id: Driver identifier
        
        Returns:
            DriverProfile or None if not found
        """
        return self.profiles.get(driver_id)
    
    def get_all_profiles(self) -> List[DriverProfile]:
        """Get all driver profiles."""
        return list(self.profiles.values())
    
    def delete_profile(self, driver_id: str):
        """
        Delete a driver profile.
        
        Args:
            driver_id: Driver identifier
        """
        if driver_id in self.profiles:
            del self.profiles[driver_id]
            
            # Delete file
            filepath = self.profiles_dir / f"{driver_id}.json"
            if filepath.exists():
                filepath.unlink()
            
            logger.info(f"Profile deleted: {driver_id}")
    
    def get_adapted_thresholds(self, driver_id: str) -> Dict[str, float]:
        """
        Get adapted thresholds for a driver.
        
        Args:
            driver_id: Driver identifier
        
        Returns:
            Dictionary with adapted thresholds
        """
        profile = self.get_profile(driver_id)
        if profile is None:
            return self.threshold_adapter.get_adapted_thresholds()
        
        # Create metrics dict from profile
        metrics = {
            'reaction_time': {'mean': profile.avg_reaction_time},
            'following_distance': {'mean': profile.avg_following_distance},
            'risk_tolerance': profile.risk_tolerance
        }
        
        # Get driving style
        try:
            driving_style = DrivingStyle(profile.driving_style)
        except ValueError:
            driving_style = DrivingStyle.UNKNOWN
        
        # Adapt thresholds
        return self.threshold_adapter.adapt_thresholds(metrics, driving_style)
    
    def get_metrics_tracker(self) -> MetricsTracker:
        """Get the metrics tracker instance."""
        return self.metrics_tracker
    
    def get_active_profile_id(self) -> Optional[str]:
        """Get the active profile ID."""
        return self.active_profile_id
