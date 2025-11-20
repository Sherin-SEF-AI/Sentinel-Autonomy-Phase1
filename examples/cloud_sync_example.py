"""
Example demonstrating cloud synchronization features.

This example shows how to:
1. Initialize cloud API client
2. Upload trip data
3. Upload scenarios
4. Download model updates
5. Sync driver profiles
6. View fleet statistics
7. Handle offline operation
"""

import sys
import os
import time
import logging

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.cloud import (
    CloudAPIClient,
    TripUploader,
    ScenarioUploader,
    ModelDownloader,
    ProfileSynchronizer,
    FleetManager,
    OfflineManager
)
from src.cloud.offline_manager import ConnectivityStatus

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def connectivity_callback(status: ConnectivityStatus):
    """Callback for connectivity changes"""
    logger.info(f"Connectivity changed: {status.value}")


def main():
    """Main example function"""
    
    # Configuration (in production, load from config file)
    config = {
        'api_url': 'https://api.sentinel-fleet.com',
        'api_key': 'your_api_key_here',
        'vehicle_id': 'vehicle_001',
        'fleet_id': 'fleet_alpha'
    }
    
    logger.info("=== Cloud Synchronization Example ===\n")
    
    # 1. Initialize API client
    logger.info("1. Initializing API client...")
    api_client = CloudAPIClient(
        api_url=config['api_url'],
        api_key=config['api_key'],
        vehicle_id=config['vehicle_id'],
        fleet_id=config['fleet_id']
    )
    
    # Check connectivity
    if api_client.check_connectivity():
        logger.info("✓ Connected to cloud backend")
    else:
        logger.warning("✗ Cannot connect to cloud backend (using mock mode)")
    
    # 2. Trip uploader
    logger.info("\n2. Setting up trip uploader...")
    trip_uploader = TripUploader(api_client)
    
    # Start a trip
    trip_id = trip_uploader.start_trip(driver_id="driver_123")
    logger.info(f"Started trip: {trip_id}")
    
    # Simulate trip updates
    for i in range(5):
        trip_uploader.update_trip(
            distance_delta=100.0,  # 100 meters
            speed=15.0,  # 15 m/s
            risk_score=0.2 + i * 0.1,
            alert_urgency='info' if i % 2 == 0 else None
        )
        time.sleep(0.1)
    
    # End trip
    completed_trip = trip_uploader.end_trip()
    logger.info(f"Ended trip: {completed_trip.distance:.1f}m, {completed_trip.alert_count} alerts")
    
    # Start background upload
    trip_uploader.start_background_upload()
    logger.info("Background trip upload started")
    
    # 3. Scenario uploader
    logger.info("\n3. Setting up scenario uploader...")
    scenario_uploader = ScenarioUploader(
        api_client,
        upload_consent=True  # User has consented
    )
    
    # Queue a scenario (if it exists)
    scenario_path = "scenarios/2024-11-17_10-30-45"
    if os.path.exists(scenario_path):
        scenario_uploader.queue_scenario(scenario_path, priority='high')
        logger.info(f"Queued scenario: {scenario_path}")
    else:
        logger.info(f"No scenario found at {scenario_path}")
    
    # Start background upload
    scenario_uploader.start_background_upload()
    logger.info("Background scenario upload started")
    
    # 4. Model downloader
    logger.info("\n4. Setting up model downloader...")
    model_downloader = ModelDownloader(
        api_client,
        auto_install=True
    )
    
    # Check for updates
    updates = model_downloader.check_for_updates()
    if updates:
        logger.info(f"Found {len(updates)} model updates:")
        for update in updates:
            logger.info(f"  - {update['name']}: {update['current_version']} -> {update['latest_version']}")
    else:
        logger.info("All models are up to date")
    
    # Start background checks
    model_downloader.start_background_checks()
    logger.info("Background model update checks started")
    
    # 5. Profile synchronizer
    logger.info("\n5. Setting up profile synchronizer...")
    profile_sync = ProfileSynchronizer(api_client)
    
    # Example profile data
    profile_data = {
        'driver_id': 'driver_123',
        'total_distance': 15000.0,
        'total_time': 3600.0,
        'driving_style': 'normal',
        'safety_score': 85.0,
        'attention_score': 90.0,
        'metrics': {
            'reaction_time': [0.8, 0.7, 0.9],
            'following_distance': [25.0, 28.0, 30.0]
        },
        'last_updated': time.time()
    }
    
    # Sync profile
    merged_profile = profile_sync.sync_profile('driver_123', profile_data)
    logger.info(f"Synced profile for driver_123: safety_score={merged_profile['safety_score']:.1f}")
    
    # 6. Fleet manager
    logger.info("\n6. Fetching fleet statistics...")
    fleet_manager = FleetManager(api_client)
    
    # Get fleet statistics
    stats = fleet_manager.get_fleet_statistics()
    if stats:
        logger.info("Fleet statistics:")
        logger.info(f"  Total vehicles: {stats.get('total_vehicles', 'N/A')}")
        logger.info(f"  Total distance: {stats.get('total_distance', 0) / 1000:.1f} km")
        logger.info(f"  Average safety score: {stats.get('average_safety_score', 0):.1f}")
    
    # Get vehicle rankings
    rankings = fleet_manager.get_vehicle_rankings(metric='safety_score', limit=5)
    if rankings:
        logger.info("\nTop 5 vehicles by safety score:")
        for i, vehicle in enumerate(rankings, 1):
            logger.info(f"  {i}. {vehicle['vehicle_id']}: {vehicle['safety_score']:.1f}")
    
    # Get driver leaderboard
    leaderboard = fleet_manager.get_driver_leaderboard(metric='safety_score', limit=5)
    if leaderboard:
        logger.info("\nTop 5 drivers by safety score:")
        for i, driver in enumerate(leaderboard, 1):
            logger.info(f"  {i}. {driver['driver_id']}: {driver['safety_score']:.1f}")
    
    # 7. Offline manager
    logger.info("\n7. Setting up offline manager...")
    offline_manager = OfflineManager(
        api_client,
        connectivity_callback=connectivity_callback
    )
    
    # Register sync callbacks
    offline_manager.register_sync_callback(trip_uploader._process_upload_queue)
    offline_manager.register_sync_callback(lambda: logger.info("Sync callback triggered"))
    
    # Start monitoring
    offline_manager.start_monitoring()
    logger.info("Background connectivity monitoring started")
    
    # Check status
    status = offline_manager.get_status()
    logger.info(f"Current connectivity: {status.value}")
    logger.info(f"Status message: {offline_manager.get_sync_status_message()}")
    
    # 8. Statistics
    logger.info("\n8. Current statistics:")
    
    trip_stats = trip_uploader.get_statistics()
    logger.info(f"Trip uploader: {trip_stats['queued_trips']} queued, "
               f"{trip_stats['total_distance'] / 1000:.1f} km total")
    
    scenario_stats = scenario_uploader.get_statistics()
    logger.info(f"Scenario uploader: {scenario_stats['queued_scenarios']} queued, "
               f"consent={scenario_stats['consent_enabled']}")
    
    installed_models = model_downloader.get_installed_models()
    logger.info(f"Installed models: {len(installed_models)}")
    for name, info in installed_models.items():
        logger.info(f"  - {name}: {info['version']}")
    
    # Keep running for a bit to see background operations
    logger.info("\n9. Running for 10 seconds to demonstrate background operations...")
    time.sleep(10)
    
    # Cleanup
    logger.info("\n10. Cleaning up...")
    trip_uploader.stop_background_upload()
    scenario_uploader.stop_background_upload()
    model_downloader.stop_background_checks()
    offline_manager.stop_monitoring()
    api_client.close()
    
    logger.info("\n=== Example Complete ===")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
