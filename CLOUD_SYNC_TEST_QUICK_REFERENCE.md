# Cloud Synchronization Testing - Quick Reference

## Test Execution

```bash
# Run all cloud sync tests
python3 -m pytest tests/unit/test_cloud.py -v

# Run specific test class
python3 -m pytest tests/unit/test_cloud.py::TestEncryption -v

# Run with coverage
python3 -m pytest tests/unit/test_cloud.py --cov=src/cloud --cov-report=html
```

## Test Organization

### 1. API Client Tests (8 tests)
```python
TestCloudAPIClientWithMockServer
├── test_initialization                 # Client setup
├── test_successful_get_request         # GET with mock server
├── test_successful_post_request        # POST with data
├── test_failed_request                 # Error handling
├── test_timeout_handling               # Timeout scenarios
├── test_connection_error_handling      # Connection failures
├── test_rate_limiting                  # Rate limiter config
└── test_retry_configuration            # Retry strategy
```

### 2. Data Upload/Download Tests (3 tests)
```python
TestDataUploadDownload
├── test_trip_data_upload              # Trip summary upload
├── test_profile_data_download         # Profile download
└── test_model_download                # Model metadata
```

### 3. Offline Queueing Tests (4 tests)
```python
TestOfflineQueueing
├── test_trip_queuing_when_offline     # Queue when offline
├── test_queue_persistence             # Persist to disk
├── test_auto_sync_when_online         # Auto-upload
└── test_offline_manager_status_tracking # Status monitoring
```

### 4. Encryption Tests (4 tests)
```python
TestEncryption
├── test_encrypt_decrypt_profile       # Basic encryption
├── test_encryption_with_different_keys # Key uniqueness
├── test_encryption_integrity          # Tamper detection
└── test_encryption_with_custom_key    # Custom keys
```

### 5. Integration Tests (3 tests)
```python
TestIntegrationScenarios
├── test_complete_trip_upload_workflow  # End-to-end trip
├── test_profile_sync_workflow          # Profile sync
└── test_offline_to_online_transition   # Offline recovery
```

## Key Test Patterns

### Mock Server Pattern
```python
@patch('requests.Session.request')
def test_api_call(self, mock_request):
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {'data': 'value'}
    mock_request.return_value = mock_response
    
    client = CloudAPIClient(...)
    response = client.get('/endpoint')
    assert response.success
```

### Encryption Validation
```python
# Test encryption
encrypted = sync.encrypt_profile(data)
assert 'sensitive' not in encrypted  # No plaintext

# Test decryption
decrypted = sync.decrypt_profile(encrypted)
assert decrypted == data

# Test tamper detection
with pytest.raises(Exception):
    sync.decrypt_profile(tampered_data)
```

### Offline Queue Testing
```python
# Queue while offline
mock_client.check_connectivity.return_value = False
uploader.start_trip()
uploader.end_trip()
assert uploader.upload_queue.qsize() == 1

# Upload when online
mock_client.check_connectivity.return_value = True
uploader._process_upload_queue()
assert uploader.upload_queue.qsize() == 0
```

## Requirements Coverage

| Requirement | Tests | Status |
|-------------|-------|--------|
| 24.1 - API Client | 8 | ✅ Complete |
| 24.2 - Scenario Upload | 3 | ✅ Complete |
| 24.3 - Model Download | 2 | ✅ Complete |
| 24.4 - Encryption | 4 | ✅ Complete |
| 24.6 - Offline Support | 4 | ✅ Complete |

## Test Results Summary

```
43 tests total
43 passed (100%)
0 failed
0 skipped

Execution time: ~0.8 seconds
```

## Common Test Scenarios

### 1. Test API Authentication
```python
client = CloudAPIClient(
    api_url="https://api.test.com",
    api_key="test_key",
    vehicle_id="vehicle_001",
    fleet_id="fleet_alpha"
)
assert 'Authorization' in client.session.headers
assert client.session.headers['Authorization'] == 'Bearer test_key'
```

### 2. Test GPS Anonymization
```python
location = {'lat': 37.7749295, 'lon': -122.4194155}
uploader.start_trip(location=location)

# Anonymized to 2 decimal places (~1km precision)
assert uploader.current_trip.start_location['lat'] == 37.77
assert uploader.current_trip.start_location['lon'] == -122.42
```

### 3. Test Profile Encryption
```python
profile_data = {'driver_id': 'driver_123', 'score': 85.0}

# Encrypt
encrypted = sync.encrypt_profile(profile_data)
assert isinstance(encrypted, str)
assert 'driver_123' not in encrypted

# Decrypt
decrypted = sync.decrypt_profile(encrypted)
assert decrypted == profile_data
```

### 4. Test Offline Queue Persistence
```python
# Create uploader and queue trip
uploader1 = TripUploader(client, queue_file=temp_file)
uploader1.start_trip()
uploader1.end_trip()

# Create new uploader - should load queue
uploader2 = TripUploader(client, queue_file=temp_file)
assert uploader2.upload_queue.qsize() == 1
```

### 5. Test Connectivity Monitoring
```python
manager = OfflineManager(client)

# Check offline
mock_client.check_connectivity.return_value = False
status = manager._check_connectivity()
assert status == ConnectivityStatus.OFFLINE

# Check online
mock_client.check_connectivity.return_value = True
status = manager._check_connectivity()
assert status == ConnectivityStatus.ONLINE
```

## Debugging Tips

### View Test Output
```bash
# Verbose output
pytest tests/unit/test_cloud.py -v -s

# Show print statements
pytest tests/unit/test_cloud.py -v --capture=no

# Stop on first failure
pytest tests/unit/test_cloud.py -x
```

### Check Coverage
```bash
# Generate coverage report
pytest tests/unit/test_cloud.py --cov=src/cloud --cov-report=term-missing

# HTML coverage report
pytest tests/unit/test_cloud.py --cov=src/cloud --cov-report=html
open htmlcov/index.html
```

### Run Specific Tests
```bash
# Single test
pytest tests/unit/test_cloud.py::TestEncryption::test_encrypt_decrypt_profile -v

# Test class
pytest tests/unit/test_cloud.py::TestEncryption -v

# Pattern matching
pytest tests/unit/test_cloud.py -k "encryption" -v
```

## Files Tested

- `src/cloud/api_client.py` - REST API client
- `src/cloud/trip_uploader.py` - Trip data upload
- `src/cloud/scenario_uploader.py` - Scenario upload
- `src/cloud/model_downloader.py` - Model download
- `src/cloud/profile_sync.py` - Profile synchronization
- `src/cloud/fleet_manager.py` - Fleet statistics
- `src/cloud/offline_manager.py` - Offline operation

## Next Steps

1. Run integration tests with real cloud backend (staging)
2. Load testing for concurrent uploads
3. Network failure simulation tests
4. End-to-end testing with full SENTINEL system
5. Performance benchmarking for large data uploads
