"""
Direct verification of cloud module without full package import.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def test_direct_imports():
    """Test direct imports of cloud modules"""
    print("Testing direct cloud module imports...")
    
    try:
        # Import cloud modules directly
        from src.cloud.api_client import CloudAPIClient
        from src.cloud.trip_uploader import TripUploader
        from src.cloud.scenario_uploader import ScenarioUploader
        from src.cloud.model_downloader import ModelDownloader
        from src.cloud.profile_sync import ProfileSynchronizer
        from src.cloud.fleet_manager import FleetManager
        from src.cloud.offline_manager import OfflineManager
        
        print("✓ All cloud modules imported successfully")
        
        # Test instantiation
        print("\nTesting module instantiation...")
        
        client = CloudAPIClient(
            api_url="https://api.test.com",
            api_key="test_key",
            vehicle_id="vehicle_001",
            fleet_id="fleet_alpha"
        )
        print("✓ CloudAPIClient instantiated")
        
        trip_uploader = TripUploader(client)
        print("✓ TripUploader instantiated")
        
        scenario_uploader = ScenarioUploader(client)
        print("✓ ScenarioUploader instantiated")
        
        model_downloader = ModelDownloader(client)
        print("✓ ModelDownloader instantiated")
        
        profile_sync = ProfileSynchronizer(client)
        print("✓ ProfileSynchronizer instantiated")
        
        fleet_manager = FleetManager(client)
        print("✓ FleetManager instantiated")
        
        offline_manager = OfflineManager(client)
        print("✓ OfflineManager instantiated")
        
        print("\n✓ All cloud components verified successfully!")
        return True
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = test_direct_imports()
    sys.exit(0 if success else 1)
