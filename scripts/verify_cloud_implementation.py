"""
Verification script for cloud synchronization implementation.

This script verifies that all cloud components are properly implemented
and can be imported without errors.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def verify_imports():
    """Verify all cloud module imports"""
    print("Verifying cloud module imports...")
    
    try:
        from src.cloud import (
            CloudAPIClient,
            TripUploader,
            ScenarioUploader,
            ModelDownloader,
            ProfileSynchronizer,
            FleetManager,
            OfflineManager
        )
        print("✓ All cloud modules imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False


def verify_api_client():
    """Verify CloudAPIClient implementation"""
    print("\nVerifying CloudAPIClient...")
    
    try:
        from src.cloud import CloudAPIClient
        
        # Create client
        client = CloudAPIClient(
            api_url="https://api.test.com",
            api_key="test_key",
            vehicle_id="vehicle_001",
            fleet_id="fleet_alpha"
        )
        
        # Check attributes
        assert hasattr(client, 'get')
        assert hasattr(client, 'post')
        assert hasattr(client, 'put')
        assert hasattr(client, 'delete')
        assert hasattr(client, 'check_connectivity')
        
        print("✓ CloudAPIClient verified")
        return True
    except Exception as e:
        print(f"✗ CloudAPIClient error: {e}")
        return False


def verify_trip_uploader():
    """Verify TripUploader implementation"""
    print("\nVerifying TripUploader...")
    
    try:
        from src.cloud import CloudAPIClient, TripUploader
        
        client = CloudAPIClient(
            api_url="https://api.test.com",
            api_key="test_key",
            vehicle_id="vehicle_001",
            fleet_id="fleet_alpha"
        )
        
        uploader = TripUploader(client)
        
        # Check methods
        assert hasattr(uploader, 'start_trip')
        assert hasattr(uploader, 'update_trip')
        assert hasattr(uploader, 'end_trip')
        assert hasattr(uploader, 'start_background_upload')
        assert hasattr(uploader, 'stop_background_upload')
        
        print("✓ TripUploader verified")
        return True
    except Exception as e:
        print(f"✗ TripUploader error: {e}")
        return False


def verify_scenario_uploader():
    """Verify ScenarioUploader implementation"""
    print("\nVerifying ScenarioUploader...")
    
    try:
        from src.cloud import CloudAPIClient, ScenarioUploader
        
        client = CloudAPIClient(
            api_url="https://api.test.com",
            api_key="test_key",
            vehicle_id="vehicle_001",
            fleet_id="fleet_alpha"
        )
        
        uploader = ScenarioUploader(client)
        
        # Check methods
        assert hasattr(uploader, 'set_consent')
        assert hasattr(uploader, 'queue_scenario')
        assert hasattr(uploader, 'start_background_upload')
        assert hasattr(uploader, 'get_upload_status')
        
        print("✓ ScenarioUploader verified")
        return True
    except Exception as e:
        print(f"✗ ScenarioUploader error: {e}")
        return False


def verify_model_downloader():
    """Verify ModelDownloader implementation"""
    print("\nVerifying ModelDownloader...")
    
    try:
        from src.cloud import CloudAPIClient, ModelDownloader
        
        client = CloudAPIClient(
            api_url="https://api.test.com",
            api_key="test_key",
            vehicle_id="vehicle_001",
            fleet_id="fleet_alpha"
        )
        
        downloader = ModelDownloader(client)
        
        # Check methods
        assert hasattr(downloader, 'check_for_updates')
        assert hasattr(downloader, 'download_and_install_model')
        assert hasattr(downloader, 'start_background_checks')
        assert hasattr(downloader, 'get_installed_models')
        
        print("✓ ModelDownloader verified")
        return True
    except Exception as e:
        print(f"✗ ModelDownloader error: {e}")
        return False


def verify_profile_sync():
    """Verify ProfileSynchronizer implementation"""
    print("\nVerifying ProfileSynchronizer...")
    
    try:
        from src.cloud import CloudAPIClient, ProfileSynchronizer
        
        client = CloudAPIClient(
            api_url="https://api.test.com",
            api_key="test_key",
            vehicle_id="vehicle_001",
            fleet_id="fleet_alpha"
        )
        
        sync = ProfileSynchronizer(client)
        
        # Check methods
        assert hasattr(sync, 'encrypt_profile')
        assert hasattr(sync, 'decrypt_profile')
        assert hasattr(sync, 'upload_profile')
        assert hasattr(sync, 'download_profile')
        assert hasattr(sync, 'sync_profile')
        
        print("✓ ProfileSynchronizer verified")
        return True
    except Exception as e:
        print(f"✗ ProfileSynchronizer error: {e}")
        return False


def verify_fleet_manager():
    """Verify FleetManager implementation"""
    print("\nVerifying FleetManager...")
    
    try:
        from src.cloud import CloudAPIClient, FleetManager
        
        client = CloudAPIClient(
            api_url="https://api.test.com",
            api_key="test_key",
            vehicle_id="vehicle_001",
            fleet_id="fleet_alpha"
        )
        
        manager = FleetManager(client)
        
        # Check methods
        assert hasattr(manager, 'get_fleet_statistics')
        assert hasattr(manager, 'get_vehicle_rankings')
        assert hasattr(manager, 'get_fleet_trends')
        assert hasattr(manager, 'get_aggregate_metrics')
        
        print("✓ FleetManager verified")
        return True
    except Exception as e:
        print(f"✗ FleetManager error: {e}")
        return False


def verify_offline_manager():
    """Verify OfflineManager implementation"""
    print("\nVerifying OfflineManager...")
    
    try:
        from src.cloud import CloudAPIClient, OfflineManager
        
        client = CloudAPIClient(
            api_url="https://api.test.com",
            api_key="test_key",
            vehicle_id="vehicle_001",
            fleet_id="fleet_alpha"
        )
        
        manager = OfflineManager(client)
        
        # Check methods
        assert hasattr(manager, 'start_monitoring')
        assert hasattr(manager, 'stop_monitoring')
        assert hasattr(manager, 'is_online')
        assert hasattr(manager, 'is_offline')
        assert hasattr(manager, 'register_sync_callback')
        
        print("✓ OfflineManager verified")
        return True
    except Exception as e:
        print(f"✗ OfflineManager error: {e}")
        return False


def verify_files():
    """Verify all required files exist"""
    print("\nVerifying file structure...")
    
    required_files = [
        'src/cloud/__init__.py',
        'src/cloud/api_client.py',
        'src/cloud/trip_uploader.py',
        'src/cloud/scenario_uploader.py',
        'src/cloud/model_downloader.py',
        'src/cloud/profile_sync.py',
        'src/cloud/fleet_manager.py',
        'src/cloud/offline_manager.py',
        'src/cloud/README.md',
        'examples/cloud_sync_example.py',
        'tests/unit/test_cloud.py'
    ]
    
    all_exist = True
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✓ {file_path}")
        else:
            print(f"✗ {file_path} - NOT FOUND")
            all_exist = False
    
    return all_exist


def main():
    """Run all verifications"""
    print("=" * 70)
    print("Cloud Synchronization Implementation Verification")
    print("=" * 70)
    
    results = []
    
    # Verify file structure
    results.append(("File Structure", verify_files()))
    
    # Verify imports
    results.append(("Module Imports", verify_imports()))
    
    # Verify individual components
    results.append(("CloudAPIClient", verify_api_client()))
    results.append(("TripUploader", verify_trip_uploader()))
    results.append(("ScenarioUploader", verify_scenario_uploader()))
    results.append(("ModelDownloader", verify_model_downloader()))
    results.append(("ProfileSynchronizer", verify_profile_sync()))
    results.append(("FleetManager", verify_fleet_manager()))
    results.append(("OfflineManager", verify_offline_manager()))
    
    # Summary
    print("\n" + "=" * 70)
    print("Verification Summary")
    print("=" * 70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{name:30s} {status}")
    
    print(f"\nTotal: {passed}/{total} checks passed")
    
    if passed == total:
        print("\n✓ All verifications passed!")
        return 0
    else:
        print(f"\n✗ {total - passed} verification(s) failed")
        return 1


if __name__ == '__main__':
    sys.exit(main())
