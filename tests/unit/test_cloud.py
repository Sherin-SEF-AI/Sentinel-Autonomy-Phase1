"""
Unit tests for cloud synchronization module.

Tests cover:
- API client with mock server (Requirement 24.1)
- Data upload and download (Requirements 24.1, 24.2)
- Offline queueing (Requirement 24.6)
- Encryption validation (Requirement 24.4)
"""

import pytest
import time
import json
import os
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock, mock_open
from datetime import datetime
from pathlib import Path

from src.cloud import (
    CloudAPIClient,
    TripUploader,
    ScenarioUploader,
    ModelDownloader,
    ProfileSynchronizer,
    FleetManager,
    OfflineManager
)
from src.cloud.api_client import APIResponse
from src.cloud.offline_manager import ConnectivityStatus


class TestCloudAPIClientWithMockServer:
    """Test CloudAPIClient with mock server (Requirement 24.1)"""
    
    def test_initialization(self):
        """Test API client initialization"""
        client = CloudAPIClient(
            api_url="https://api.test.com",
            api_key="test_key",
            vehicle_id="vehicle_001",
            fleet_id="fleet_alpha"
        )
        
        assert client.api_url == "https://api.test.com"
        assert client.api_key == "test_key"
        assert client.vehicle_id == "vehicle_001"
        assert client.fleet_id == "fleet_alpha"
        assert 'Authorization' in client.session.headers
        assert client.session.headers['Authorization'] == 'Bearer test_key'
    
    @patch('requests.Session.request')
    def test_successful_get_request(self, mock_request):
        """Test successful GET request with mock server"""
        # Mock response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'status': 'ok', 'data': 'test'}
        mock_response.content = b'{"status": "ok"}'
        mock_request.return_value = mock_response
        
        client = CloudAPIClient(
            api_url="https://api.test.com",
            api_key="test_key",
            vehicle_id="vehicle_001",
            fleet_id="fleet_alpha"
        )
        
        response = client.get('/test')
        
        assert response.success
        assert response.status_code == 200
        assert response.data == {'status': 'ok', 'data': 'test'}
        
        # Verify request was made with correct headers
        mock_request.assert_called_once()
        call_args = mock_request.call_args
        assert call_args[0][0] == 'GET'
        assert call_args[0][1] == 'https://api.test.com/test'
    
    @patch('requests.Session.request')
    def test_successful_post_request(self, mock_request):
        """Test successful POST request with data"""
        mock_response = Mock()
        mock_response.status_code = 201
        mock_response.json.return_value = {'id': '123', 'created': True}
        mock_response.content = b'{"id": "123"}'
        mock_request.return_value = mock_response
        
        client = CloudAPIClient(
            api_url="https://api.test.com",
            api_key="test_key",
            vehicle_id="vehicle_001",
            fleet_id="fleet_alpha"
        )
        
        test_data = {'name': 'test', 'value': 42}
        response = client.post('/create', data=test_data)
        
        assert response.success
        assert response.status_code == 201
        assert response.data['id'] == '123'
    
    @patch('requests.Session.request')
    def test_failed_request(self, mock_request):
        """Test failed request handling"""
        # Mock error response
        mock_response = Mock()
        mock_response.status_code = 404
        mock_response.text = "Not found"
        mock_request.return_value = mock_response
        
        client = CloudAPIClient(
            api_url="https://api.test.com",
            api_key="test_key",
            vehicle_id="vehicle_001",
            fleet_id="fleet_alpha"
        )
        
        response = client.get('/nonexistent')
        
        assert not response.success
        assert response.status_code == 404
        assert 'Not found' in response.error
    
    @patch('requests.Session.request')
    def test_timeout_handling(self, mock_request):
        """Test request timeout handling"""
        import requests
        mock_request.side_effect = requests.exceptions.Timeout()
        
        client = CloudAPIClient(
            api_url="https://api.test.com",
            api_key="test_key",
            vehicle_id="vehicle_001",
            fleet_id="fleet_alpha",
            timeout=1.0
        )
        
        response = client.get('/slow')
        
        assert not response.success
        assert response.status_code == 0
        assert 'timeout' in response.error.lower()
    
    @patch('requests.Session.request')
    def test_connection_error_handling(self, mock_request):
        """Test connection error handling"""
        import requests
        mock_request.side_effect = requests.exceptions.ConnectionError("Connection refused")
        
        client = CloudAPIClient(
            api_url="https://api.test.com",
            api_key="test_key",
            vehicle_id="vehicle_001",
            fleet_id="fleet_alpha"
        )
        
        response = client.get('/test')
        
        assert not response.success
        assert 'connection' in response.error.lower()
    
    def test_rate_limiting(self):
        """Test rate limiting configuration"""
        client = CloudAPIClient(
            api_url="https://api.test.com",
            api_key="test_key",
            vehicle_id="vehicle_001",
            fleet_id="fleet_alpha",
            rate_limit=5.0  # 5 requests per second
        )
        
        # Verify rate limiter is configured
        assert client.rate_limiter is not None
        assert client.rate_limiter.requests_per_second == 5.0
        assert client.rate_limiter.max_tokens == 5.0
        
        # Test token acquisition
        initial_tokens = client.rate_limiter.tokens
        client.rate_limiter.acquire()
        # Tokens should decrease after acquisition
        assert client.rate_limiter.tokens < initial_tokens
    
    @patch('requests.Session.request')
    def test_retry_configuration(self, mock_request):
        """Test retry configuration is set up correctly"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'status': 'ok'}
        mock_response.content = b'{"status": "ok"}'
        mock_request.return_value = mock_response
        
        client = CloudAPIClient(
            api_url="https://api.test.com",
            api_key="test_key",
            vehicle_id="vehicle_001",
            fleet_id="fleet_alpha",
            max_retries=3
        )
        
        # Verify retry adapter is configured
        adapter = client.session.get_adapter('https://api.test.com')
        assert adapter is not None
        assert adapter.max_retries.total == 3


class TestDataUploadDownload:
    """Test data upload and download (Requirements 24.1, 24.2)"""
    
    @patch('requests.Session.request')
    def test_trip_data_upload(self, mock_request):
        """Test trip data upload to cloud"""
        mock_response = Mock()
        mock_response.status_code = 201
        mock_response.json.return_value = {'trip_id': 'trip_123', 'status': 'uploaded'}
        mock_response.content = b'{"trip_id": "trip_123"}'
        mock_request.return_value = mock_response
        
        mock_client = CloudAPIClient(
            api_url="https://api.test.com",
            api_key="test_key",
            vehicle_id="vehicle_001",
            fleet_id="fleet_alpha"
        )
        
        trip_data = {
            'trip_id': 'trip_123',
            'distance': 5000.0,
            'duration': 600.0,
            'alert_count': 5
        }
        
        response = mock_client.post('/trips', data=trip_data)
        
        assert response.success
        assert response.data['trip_id'] == 'trip_123'
    
    @patch('requests.Session.request')
    def test_profile_data_download(self, mock_request):
        """Test profile data download from cloud"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'driver_id': 'driver_123',
            'safety_score': 85.0,
            'encrypted_data': 'encrypted_profile_data'
        }
        mock_response.content = b'{"driver_id": "driver_123"}'
        mock_request.return_value = mock_response
        
        mock_client = CloudAPIClient(
            api_url="https://api.test.com",
            api_key="test_key",
            vehicle_id="vehicle_001",
            fleet_id="fleet_alpha"
        )
        
        response = mock_client.get('/drivers/driver_123/profile')
        
        assert response.success
        assert response.data['driver_id'] == 'driver_123'
        assert 'encrypted_data' in response.data
    
    @patch('requests.Session.request')
    def test_model_download(self, mock_request):
        """Test model download from cloud (Requirement 24.3)"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'model_name': 'test_model',
            'version': 'v1.0',
            'download_url': 'https://cdn.test.com/model.pth',
            'checksum': 'abc123def456'
        }
        mock_response.content = b'{"model_name": "test_model"}'
        mock_request.return_value = mock_response
        
        mock_client = CloudAPIClient(
            api_url="https://api.test.com",
            api_key="test_key",
            vehicle_id="vehicle_001",
            fleet_id="fleet_alpha"
        )
        
        response = mock_client.get('/models/test_model/latest')
        
        assert response.success
        assert response.data['model_name'] == 'test_model'
        assert 'download_url' in response.data
        assert 'checksum' in response.data


class TestTripUploader:
    """Test TripUploader"""
    
    def test_start_trip(self):
        """Test starting a trip"""
        mock_client = Mock()
        uploader = TripUploader(mock_client, queue_file=tempfile.mktemp())
        
        trip_id = uploader.start_trip(driver_id="driver_123")
        
        assert trip_id.startswith("trip_")
        assert uploader.current_trip is not None
        assert uploader.current_trip.driver_id == "driver_123"
    
    def test_update_trip(self):
        """Test updating trip data"""
        mock_client = Mock()
        uploader = TripUploader(mock_client, queue_file=tempfile.mktemp())
        
        uploader.start_trip()
        uploader.update_trip(
            distance_delta=100.0,
            speed=15.0,
            risk_score=0.5,
            alert_urgency='warning'
        )
        
        assert uploader.current_trip.distance == 100.0
        assert uploader.current_trip.max_speed == 15.0
        assert uploader.current_trip.alert_count == 1
        assert uploader.current_trip.warning_alert_count == 1
    
    def test_end_trip(self):
        """Test ending a trip"""
        mock_client = Mock()
        uploader = TripUploader(mock_client, queue_file=tempfile.mktemp())
        
        uploader.start_trip()
        time.sleep(0.1)
        completed_trip = uploader.end_trip()
        
        assert completed_trip is not None
        assert completed_trip.end_time is not None
        assert completed_trip.duration > 0
        assert uploader.current_trip is None
    
    def test_gps_anonymization(self):
        """Test GPS coordinate anonymization"""
        mock_client = Mock()
        uploader = TripUploader(mock_client, queue_file=tempfile.mktemp())
        
        # Start trip with precise GPS
        location = {'lat': 37.7749295, 'lon': -122.4194155}
        uploader.start_trip(location=location)
        
        # Check that location is anonymized (rounded to ~1km precision)
        assert uploader.current_trip.start_location['lat'] == 37.77
        assert uploader.current_trip.start_location['lon'] == -122.42


class TestOfflineQueueing:
    """Test offline queueing (Requirement 24.6)"""
    
    def test_trip_queuing_when_offline(self):
        """Test that trips are queued when offline"""
        mock_client = Mock()
        mock_client.check_connectivity.return_value = False
        
        temp_queue_file = tempfile.mktemp()
        uploader = TripUploader(mock_client, queue_file=temp_queue_file)
        
        # Start and end trip
        uploader.start_trip()
        uploader.update_trip(distance_delta=1000.0)
        uploader.end_trip()
        
        # Trip should be queued
        assert uploader.upload_queue.qsize() == 1
        
        # Cleanup
        if os.path.exists(temp_queue_file):
            os.remove(temp_queue_file)
    
    def test_queue_persistence(self):
        """Test that offline queue persists to disk"""
        mock_client = Mock()
        temp_queue_file = tempfile.mktemp()
        
        # Create uploader and queue a trip
        uploader1 = TripUploader(mock_client, queue_file=temp_queue_file)
        uploader1.start_trip()
        uploader1.update_trip(distance_delta=500.0)
        uploader1.end_trip()
        
        # Queue should be saved
        assert os.path.exists(temp_queue_file)
        
        # Create new uploader - should load queue
        uploader2 = TripUploader(mock_client, queue_file=temp_queue_file)
        assert uploader2.upload_queue.qsize() == 1
        
        # Cleanup
        if os.path.exists(temp_queue_file):
            os.remove(temp_queue_file)
    
    @patch('requests.Session.request')
    def test_auto_sync_when_online(self, mock_request):
        """Test automatic sync when connectivity is restored"""
        mock_response = Mock()
        mock_response.status_code = 201
        mock_response.json.return_value = {'status': 'uploaded'}
        mock_response.content = b'{"status": "uploaded"}'
        mock_request.return_value = mock_response
        
        mock_client = CloudAPIClient(
            api_url="https://api.test.com",
            api_key="test_key",
            vehicle_id="vehicle_001",
            fleet_id="fleet_alpha"
        )
        
        temp_queue_file = tempfile.mktemp()
        uploader = TripUploader(mock_client, queue_file=temp_queue_file)
        
        # Queue a trip
        uploader.start_trip()
        uploader.end_trip()
        
        # Process upload queue (simulates coming online)
        uploader._process_upload_queue()
        
        # Queue should be empty after successful upload
        assert uploader.upload_queue.qsize() == 0
        
        # Cleanup
        if os.path.exists(temp_queue_file):
            os.remove(temp_queue_file)
    
    def test_offline_manager_status_tracking(self):
        """Test offline manager tracks connectivity status"""
        mock_client = Mock()
        mock_client.check_connectivity.return_value = False
        
        manager = OfflineManager(mock_client)
        
        # Check status
        status = manager._check_connectivity()
        assert status == ConnectivityStatus.OFFLINE
        
        # Simulate coming online
        mock_client.check_connectivity.return_value = True
        status = manager._check_connectivity()
        assert status == ConnectivityStatus.ONLINE


class TestScenarioUploader:
    """Test ScenarioUploader"""
    
    def test_consent_required(self):
        """Test that consent is required for upload"""
        mock_client = Mock()
        uploader = ScenarioUploader(mock_client, upload_consent=False)
        
        result = uploader.queue_scenario("test_scenario")
        
        assert not result
    
    def test_queue_scenario_with_consent(self):
        """Test queueing scenario with consent"""
        mock_client = Mock()
        temp_dir = tempfile.mkdtemp()
        
        try:
            uploader = ScenarioUploader(mock_client, scenarios_dir=temp_dir, upload_consent=True)
            
            # Create temporary scenario directory
            scenario_path = os.path.join(temp_dir, "test_scenario")
            os.makedirs(scenario_path, exist_ok=True)
            
            result = uploader.queue_scenario(scenario_path)
            assert result
            assert uploader.upload_queue.qsize() == 1
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    def test_consent_can_be_changed(self):
        """Test that consent can be enabled/disabled"""
        mock_client = Mock()
        uploader = ScenarioUploader(mock_client, upload_consent=False)
        
        assert not uploader.upload_consent
        
        uploader.set_consent(True)
        assert uploader.upload_consent
        
        uploader.set_consent(False)
        assert not uploader.upload_consent


class TestModelDownloader:
    """Test ModelDownloader"""
    
    def test_initialization(self):
        """Test model downloader initialization"""
        mock_client = Mock()
        downloader = ModelDownloader(mock_client)
        
        assert downloader.api_client == mock_client
        assert os.path.exists(downloader.models_dir)
    
    @patch('src.cloud.model_downloader.ModelDownloader._download_file')
    @patch('src.cloud.model_downloader.ModelDownloader._verify_checksum')
    def test_download_model(self, mock_verify, mock_download):
        """Test downloading a model"""
        mock_client = Mock()
        mock_client.get.return_value = APIResponse(
            success=True,
            status_code=200,
            data={'download_url': 'https://test.com/model.pth'}
        )
        
        mock_download.return_value = True
        mock_verify.return_value = True
        
        downloader = ModelDownloader(mock_client)
        
        # Mock install to avoid file operations
        with patch.object(downloader, '_install_model', return_value=True):
            result = downloader.download_and_install_model(
                "test_model",
                "v1.0",
                expected_checksum="abc123"
            )
        
        assert result


class TestEncryption:
    """Test encryption validation (Requirement 24.4)"""
    
    def test_encrypt_decrypt_profile(self):
        """Test profile encryption and decryption"""
        mock_client = Mock()
        mock_client.vehicle_id = "test_vehicle"
        
        temp_dir = tempfile.mkdtemp()
        try:
            sync = ProfileSynchronizer(mock_client, profiles_dir=temp_dir)
            
            profile_data = {
                'driver_id': 'driver_123',
                'safety_score': 85.0,
                'metrics': {'reaction_time': [0.8, 0.9]},
                'sensitive_data': 'confidential'
            }
            
            # Encrypt
            encrypted = sync.encrypt_profile(profile_data)
            assert isinstance(encrypted, str)
            assert len(encrypted) > 0
            
            # Encrypted data should not contain plaintext
            assert 'driver_123' not in encrypted
            assert 'confidential' not in encrypted
            
            # Decrypt
            decrypted = sync.decrypt_profile(encrypted)
            assert decrypted == profile_data
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    def test_encryption_with_different_keys(self):
        """Test that different keys produce different encrypted data"""
        mock_client1 = Mock()
        mock_client1.vehicle_id = "vehicle_001"
        
        mock_client2 = Mock()
        mock_client2.vehicle_id = "vehicle_002"
        
        temp_dir = tempfile.mkdtemp()
        try:
            sync1 = ProfileSynchronizer(mock_client1, profiles_dir=temp_dir)
            sync2 = ProfileSynchronizer(mock_client2, profiles_dir=temp_dir)
            
            profile_data = {'driver_id': 'driver_123', 'score': 85.0}
            
            encrypted1 = sync1.encrypt_profile(profile_data)
            encrypted2 = sync2.encrypt_profile(profile_data)
            
            # Different keys should produce different encrypted data
            assert encrypted1 != encrypted2
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    def test_encryption_integrity(self):
        """Test that tampered encrypted data cannot be decrypted"""
        mock_client = Mock()
        mock_client.vehicle_id = "test_vehicle"
        
        temp_dir = tempfile.mkdtemp()
        try:
            sync = ProfileSynchronizer(mock_client, profiles_dir=temp_dir)
            
            profile_data = {'driver_id': 'driver_123', 'score': 85.0}
            encrypted = sync.encrypt_profile(profile_data)
            
            # Tamper with encrypted data
            tampered = encrypted[:-10] + "TAMPERED=="
            
            # Should raise exception when trying to decrypt
            with pytest.raises(Exception):
                sync.decrypt_profile(tampered)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    def test_encryption_with_custom_key(self):
        """Test encryption with custom encryption key"""
        mock_client = Mock()
        mock_client.vehicle_id = "test_vehicle"
        
        temp_dir = tempfile.mkdtemp()
        try:
            # Use a properly formatted 32-byte key
            custom_key = "my_custom_encryption_key_1234567890ab"  # 36 chars, will be truncated to 32
            sync = ProfileSynchronizer(
                mock_client,
                profiles_dir=temp_dir,
                encryption_key=custom_key
            )
            
            profile_data = {'driver_id': 'driver_123', 'score': 85.0}
            
            encrypted = sync.encrypt_profile(profile_data)
            decrypted = sync.decrypt_profile(encrypted)
            
            assert decrypted == profile_data
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


class TestProfileSynchronizer:
    """Test ProfileSynchronizer"""
    
    def test_merge_profiles(self):
        """Test merging local and cloud profiles"""
        mock_client = Mock()
        mock_client.vehicle_id = "test_vehicle"
        
        temp_dir = tempfile.mkdtemp()
        try:
            sync = ProfileSynchronizer(mock_client, profiles_dir=temp_dir)
            
            local_profile = {
                'total_distance': 1000.0,
                'total_time': 100.0,
                'safety_score': 80.0,
                'last_updated': 100
            }
            
            cloud_profile = {
                'total_distance': 2000.0,
                'total_time': 200.0,
                'safety_score': 90.0,
                'last_updated': 200
            }
            
            merged = sync._merge_profiles(local_profile, cloud_profile)
            
            # Should aggregate distances
            assert merged['total_distance'] == 3000.0
            assert merged['total_time'] == 300.0
            
            # Should average scores
            assert merged['safety_score'] == 85.0
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    @patch('requests.Session.request')
    def test_upload_profile(self, mock_request):
        """Test uploading encrypted profile to cloud"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'status': 'uploaded'}
        mock_response.content = b'{"status": "uploaded"}'
        mock_request.return_value = mock_response
        
        mock_client = CloudAPIClient(
            api_url="https://api.test.com",
            api_key="test_key",
            vehicle_id="vehicle_001",
            fleet_id="fleet_alpha"
        )
        
        temp_dir = tempfile.mkdtemp()
        try:
            sync = ProfileSynchronizer(mock_client, profiles_dir=temp_dir)
            
            profile_data = {'driver_id': 'driver_123', 'safety_score': 85.0}
            result = sync.upload_profile('driver_123', profile_data)
            
            assert result
            # Verify encrypted data was sent
            assert mock_request.called
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    @patch('requests.Session.request')
    def test_download_profile(self, mock_request):
        """Test downloading and decrypting profile from cloud"""
        mock_client = CloudAPIClient(
            api_url="https://api.test.com",
            api_key="test_key",
            vehicle_id="vehicle_001",
            fleet_id="fleet_alpha"
        )
        
        temp_dir = tempfile.mkdtemp()
        try:
            sync = ProfileSynchronizer(mock_client, profiles_dir=temp_dir)
            
            # Encrypt profile data
            profile_data = {'driver_id': 'driver_123', 'safety_score': 85.0}
            encrypted_data = sync.encrypt_profile(profile_data)
            
            # Mock response with encrypted data
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {'encrypted_data': encrypted_data}
            mock_response.content = b'{"encrypted_data": "..."}'
            mock_request.return_value = mock_response
            
            # Download and decrypt
            downloaded = sync.download_profile('driver_123')
            
            assert downloaded is not None
            assert downloaded['driver_id'] == 'driver_123'
            assert downloaded['safety_score'] == 85.0
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


class TestFleetManager:
    """Test FleetManager"""
    
    def test_get_fleet_statistics(self):
        """Test fetching fleet statistics"""
        mock_client = Mock()
        mock_client.fleet_id = "fleet_alpha"
        mock_client.get.return_value = APIResponse(
            success=True,
            status_code=200,
            data={
                'total_vehicles': 10,
                'total_distance': 50000.0,
                'average_safety_score': 85.0
            }
        )
        
        manager = FleetManager(mock_client)
        stats = manager.get_fleet_statistics()
        
        assert stats is not None
        assert stats['total_vehicles'] == 10
        assert stats['total_distance'] == 50000.0
    
    def test_get_aggregate_metrics(self):
        """Test getting aggregate metrics"""
        mock_client = Mock()
        mock_client.fleet_id = "fleet_alpha"
        mock_client.get.return_value = APIResponse(
            success=True,
            status_code=200,
            data={
                'total_vehicles': 5,
                'total_distance': 10000.0,
                'average_safety_score': 80.0
            }
        )
        
        manager = FleetManager(mock_client)
        metrics = manager.get_aggregate_metrics()
        
        assert metrics['total_vehicles'] == 5
        assert metrics['total_distance'] == 10000.0


class TestOfflineManager:
    """Test OfflineManager"""
    
    def test_initialization(self):
        """Test offline manager initialization"""
        mock_client = Mock()
        manager = OfflineManager(mock_client)
        
        assert manager.api_client == mock_client
        assert manager.status == ConnectivityStatus.UNKNOWN
    
    @patch('socket.create_connection')
    def test_check_internet_online(self, mock_socket):
        """Test internet connectivity check when online"""
        mock_socket.return_value = Mock()
        
        mock_client = Mock()
        manager = OfflineManager(mock_client)
        
        result = manager._check_internet()
        assert result
    
    @patch('socket.create_connection')
    def test_check_internet_offline(self, mock_socket):
        """Test internet connectivity check when offline"""
        mock_socket.side_effect = OSError()
        
        mock_client = Mock()
        manager = OfflineManager(mock_client)
        
        result = manager._check_internet()
        assert not result
    
    def test_register_sync_callback(self):
        """Test registering sync callback"""
        mock_client = Mock()
        manager = OfflineManager(mock_client)
        
        callback = Mock()
        manager.register_sync_callback(callback)
        
        assert len(manager.sync_callbacks) == 1
        assert manager.sync_callbacks[0] == callback
    
    def test_sync_callback_triggered_on_reconnect(self):
        """Test that sync callbacks are triggered when coming online"""
        mock_client = Mock()
        manager = OfflineManager(mock_client)
        
        callback = Mock()
        manager.register_sync_callback(callback)
        
        # Simulate coming online
        manager._trigger_sync()
        
        callback.assert_called_once()
    
    def test_offline_duration_tracking(self):
        """Test tracking offline duration"""
        mock_client = Mock()
        manager = OfflineManager(mock_client)
        
        # Set as online
        manager.status = ConnectivityStatus.ONLINE
        manager.last_online_time = time.time()
        
        # Set as offline
        manager.status = ConnectivityStatus.OFFLINE
        time.sleep(0.1)
        
        duration = manager.get_offline_duration()
        assert duration >= 0.1
    
    def test_status_message(self):
        """Test human-readable status messages"""
        mock_client = Mock()
        manager = OfflineManager(mock_client)
        
        manager.status = ConnectivityStatus.ONLINE
        message = manager.get_sync_status_message()
        assert 'connected' in message.lower()
        
        manager.status = ConnectivityStatus.OFFLINE
        message = manager.get_sync_status_message()
        assert 'offline' in message.lower()


class TestIntegrationScenarios:
    """Integration tests for complete cloud sync workflows"""
    
    @patch('requests.Session.request')
    def test_complete_trip_upload_workflow(self, mock_request):
        """Test complete workflow: start trip, update, end, upload"""
        mock_response = Mock()
        mock_response.status_code = 201
        mock_response.json.return_value = {'status': 'uploaded'}
        mock_response.content = b'{"status": "uploaded"}'
        mock_request.return_value = mock_response
        
        mock_client = CloudAPIClient(
            api_url="https://api.test.com",
            api_key="test_key",
            vehicle_id="vehicle_001",
            fleet_id="fleet_alpha"
        )
        
        temp_queue_file = tempfile.mktemp()
        try:
            uploader = TripUploader(mock_client, queue_file=temp_queue_file)
            
            # Start trip
            trip_id = uploader.start_trip(driver_id="driver_123")
            assert trip_id is not None
            
            # Update trip multiple times
            for i in range(5):
                uploader.update_trip(
                    distance_delta=100.0,
                    speed=15.0,
                    risk_score=0.3 + i * 0.1
                )
            
            # End trip
            completed_trip = uploader.end_trip()
            assert completed_trip.distance == 500.0
            assert completed_trip.duration > 0
            
            # Upload
            uploader._process_upload_queue()
            
            # Verify upload was called
            assert mock_request.called
        finally:
            if os.path.exists(temp_queue_file):
                os.remove(temp_queue_file)
    
    @patch('requests.Session.request')
    def test_profile_sync_workflow(self, mock_request):
        """Test complete profile synchronization workflow"""
        mock_client = CloudAPIClient(
            api_url="https://api.test.com",
            api_key="test_key",
            vehicle_id="vehicle_001",
            fleet_id="fleet_alpha"
        )
        
        temp_dir = tempfile.mkdtemp()
        try:
            sync = ProfileSynchronizer(mock_client, profiles_dir=temp_dir)
            
            # Create local profile
            local_profile = {
                'driver_id': 'driver_123',
                'total_distance': 1000.0,
                'safety_score': 80.0,
                'last_updated': time.time()
            }
            
            # Mock cloud response (no existing profile)
            mock_response_404 = Mock()
            mock_response_404.status_code = 404
            mock_response_404.text = "Not found"
            
            # Mock upload success
            mock_response_200 = Mock()
            mock_response_200.status_code = 200
            mock_response_200.json.return_value = {'status': 'uploaded'}
            mock_response_200.content = b'{"status": "uploaded"}'
            
            mock_request.side_effect = [mock_response_404, mock_response_200]
            
            # Sync profile
            merged = sync.sync_profile('driver_123', local_profile)
            
            # Should return local profile since no cloud profile exists
            assert merged['driver_id'] == 'driver_123'
            assert merged['total_distance'] == 1000.0
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    def test_offline_to_online_transition(self):
        """Test transition from offline to online with queued data"""
        mock_client = Mock()
        mock_client.check_connectivity.return_value = False
        
        temp_queue_file = tempfile.mktemp()
        try:
            uploader = TripUploader(mock_client, queue_file=temp_queue_file)
            
            # Queue trips while offline
            for i in range(3):
                uploader.start_trip()
                uploader.update_trip(distance_delta=1000.0)
                uploader.end_trip()
            
            assert uploader.upload_queue.qsize() == 3
            
            # Come online
            mock_client.check_connectivity.return_value = True
            mock_client.post.return_value = APIResponse(success=True, status_code=201)
            
            # Process queue
            uploader._process_upload_queue()
            
            # Queue should be empty
            assert uploader.upload_queue.qsize() == 0
        finally:
            if os.path.exists(temp_queue_file):
                os.remove(temp_queue_file)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
