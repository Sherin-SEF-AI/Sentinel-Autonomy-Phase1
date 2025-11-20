"""
Driver Behavior Profiling Example

Demonstrates the driver behavior profiling system including:
- Driver identification
- Metrics tracking
- Driving style classification
- Threshold adaptation
- Report generation
- Profile persistence
"""

import numpy as np
import cv2
import time
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.profiling import ProfileManager
from src.core.logging import setup_logging

# Setup logging
logger = setup_logging('driver_profiling_example', 'logs/profiling_example.log')


def simulate_driving_session(profile_manager: ProfileManager, driver_id: str, session_duration: float = 60.0):
    """
    Simulate a driving session with metrics tracking.
    
    Args:
        profile_manager: ProfileManager instance
        driver_id: Driver identifier
        session_duration: Session duration in seconds
    """
    logger.info(f"Starting simulated session for {driver_id}")
    
    # Start session
    start_time = time.time()
    profile_manager.start_session(driver_id, start_time)
    
    # Get metrics tracker
    tracker = profile_manager.get_metrics_tracker()
    
    # Simulate driving for session_duration
    current_time = start_time
    lane_id = 2
    alert_counter = 0
    
    while current_time - start_time < session_duration:
        # Simulate time step
        time.sleep(0.1)
        current_time = time.time()
        
        # Simulate vehicle state
        speed = 20.0 + np.random.randn() * 3.0  # ~20 m/s with variance
        following_distance = 25.0 + np.random.randn() * 5.0  # ~25m with variance
        risk_score = np.random.beta(2, 5)  # Skewed toward lower risk
        
        # Occasionally change lanes
        if np.random.rand() < 0.01:  # 1% chance per update
            lane_id = (lane_id + np.random.choice([-1, 1])) % 4
        
        # Update metrics
        tracker.update(
            timestamp=current_time,
            speed=max(0, speed),
            following_distance=max(0, following_distance),
            lane_id=lane_id,
            risk_score=risk_score
        )
        
        # Simulate occasional alerts
        if np.random.rand() < 0.05:  # 5% chance per update
            alert_counter += 1
            alert_time = current_time
            tracker.record_alert(alert_counter, alert_time)
            
            # Simulate driver reaction after delay
            reaction_delay = 0.8 + np.random.randn() * 0.3  # ~0.8s with variance
            time.sleep(max(0.1, reaction_delay))
            reaction_time = time.time()
            tracker.record_driver_action(alert_counter, reaction_time, 'brake')
        
        # Simulate occasional near-miss
        if risk_score > 0.7 and np.random.rand() < 0.02:
            ttc = 1.0 + np.random.rand() * 1.0  # 1-2 seconds
            tracker.record_near_miss(current_time, ttc, risk_score)
    
    # End session
    end_time = time.time()
    profile_manager.end_session(end_time)
    
    logger.info(f"Session completed for {driver_id}")


def main():
    """Main example function."""
    logger.info("=" * 60)
    logger.info("Driver Behavior Profiling Example")
    logger.info("=" * 60)
    
    # Configuration
    config = {
        'profiles_dir': 'profiles_test',
        'auto_save': True,
        'face_recognition': {
            'recognition_threshold': 0.6,
            'embedding_size': 128
        },
        'threshold_adapter': {
            'base_ttc_threshold': 2.0,
            'base_following_distance': 25.0,
            'base_alert_sensitivity': 0.7
        }
    }
    
    # Initialize profile manager
    logger.info("Initializing ProfileManager...")
    profile_manager = ProfileManager(config)
    
    # Simulate driver identification
    logger.info("\n--- Driver Identification ---")
    
    # Create a dummy frame (in real system, this would be from interior camera)
    dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Identify driver (will create new profile if not found)
    driver_id = profile_manager.identify_driver(dummy_frame)
    logger.info(f"Driver identified: {driver_id}")
    
    # Run multiple sessions to build up profile
    logger.info("\n--- Running Driving Sessions ---")
    num_sessions = 3
    
    for session_num in range(1, num_sessions + 1):
        logger.info(f"\nSession {session_num}/{num_sessions}")
        simulate_driving_session(profile_manager, driver_id, session_duration=30.0)
        
        # Get profile after session
        profile = profile_manager.get_profile(driver_id)
        if profile:
            logger.info(f"Profile updated:")
            logger.info(f"  Sessions: {profile.session_count}")
            logger.info(f"  Total distance: {profile.total_distance/1000:.2f} km")
            logger.info(f"  Total time: {profile.total_time/3600:.2f} hours")
            logger.info(f"  Driving style: {profile.driving_style}")
            logger.info(f"  Safety score: {profile.safety_score:.1f}")
            logger.info(f"  Attention score: {profile.attention_score:.1f}")
            logger.info(f"  Eco score: {profile.eco_score:.1f}")
    
    # Get adapted thresholds
    logger.info("\n--- Adapted Thresholds ---")
    thresholds = profile_manager.get_adapted_thresholds(driver_id)
    logger.info(f"TTC threshold: {thresholds['ttc_threshold']:.2f} seconds")
    logger.info(f"Following distance: {thresholds['following_distance']:.1f} meters")
    logger.info(f"Alert sensitivity: {thresholds['alert_sensitivity']:.2f}")
    
    # Get final profile
    logger.info("\n--- Final Profile Summary ---")
    profile = profile_manager.get_profile(driver_id)
    if profile:
        logger.info(f"Driver ID: {profile.driver_id}")
        logger.info(f"Driving Style: {profile.driving_style.upper()}")
        logger.info(f"Total Sessions: {profile.session_count}")
        logger.info(f"Total Distance: {profile.total_distance/1000:.2f} km")
        logger.info(f"Total Time: {profile.total_time/3600:.2f} hours")
        logger.info(f"\nBehavior Metrics:")
        logger.info(f"  Avg Reaction Time: {profile.avg_reaction_time:.2f} s")
        logger.info(f"  Avg Following Distance: {profile.avg_following_distance:.1f} m")
        logger.info(f"  Lane Changes/Hour: {profile.avg_lane_change_freq:.1f}")
        logger.info(f"  Avg Speed: {profile.avg_speed*3.6:.1f} km/h")
        logger.info(f"  Risk Tolerance: {profile.risk_tolerance:.2f}")
        logger.info(f"\nPerformance Scores:")
        logger.info(f"  Safety: {profile.safety_score:.1f}/100")
        logger.info(f"  Attention: {profile.attention_score:.1f}/100")
        logger.info(f"  Eco-Driving: {profile.eco_score:.1f}/100")
        logger.info(f"  Overall: {(profile.safety_score + profile.attention_score + profile.eco_score)/3:.1f}/100")
    
    # Test profile persistence
    logger.info("\n--- Testing Profile Persistence ---")
    logger.info("Saving all profiles...")
    profile_manager.save_all_profiles()
    
    # Create new manager and load profiles
    logger.info("Creating new ProfileManager and loading profiles...")
    new_manager = ProfileManager(config)
    loaded_profile = new_manager.get_profile(driver_id)
    
    if loaded_profile:
        logger.info(f"Profile successfully loaded: {loaded_profile.driver_id}")
        logger.info(f"  Sessions: {loaded_profile.session_count}")
        logger.info(f"  Safety score: {loaded_profile.safety_score:.1f}")
    else:
        logger.error("Failed to load profile!")
    
    # Test multiple drivers
    logger.info("\n--- Testing Multiple Drivers ---")
    
    # Simulate second driver
    dummy_frame2 = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    driver_id2 = new_manager.identify_driver(dummy_frame2)
    logger.info(f"Second driver identified: {driver_id2}")
    
    # Run session for second driver
    simulate_driving_session(new_manager, driver_id2, session_duration=20.0)
    
    # List all profiles
    logger.info("\n--- All Driver Profiles ---")
    all_profiles = new_manager.get_all_profiles()
    logger.info(f"Total profiles: {len(all_profiles)}")
    for p in all_profiles:
        logger.info(f"  {p.driver_id}: {p.session_count} sessions, "
                   f"style={p.driving_style}, safety={p.safety_score:.1f}")
    
    logger.info("\n" + "=" * 60)
    logger.info("Example completed successfully!")
    logger.info("=" * 60)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\nExample interrupted by user")
    except Exception as e:
        logger.error(f"Example failed: {e}", exc_info=True)
