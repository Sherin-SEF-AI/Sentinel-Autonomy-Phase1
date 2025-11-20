# Cloud Synchronization Module

The cloud synchronization module provides connectivity to the SENTINEL fleet management backend for data upload, model updates, and fleet analytics.

## Components

### CloudAPIClient
REST API client with authentication, retry logic, and rate limiting.

**Features:**
- Bearer token authentication
- Automatic retry with exponential backoff
- Connection pooling
- Rate limiting (configurable requests per second)
- Timeout handling

**Usage:**
```python
from src.cloud import CloudAPIClient

client = CloudAPIClient(
    api_url="https://api.sentinel-fleet.com",
    api_key="your_api_key",
    vehicle_id="vehicle_001",
    fleet_id="fleet_alpha"
)

# Check connectivity
if client.check_connectivity():
    # Make API calls
    response = client.get('/fleet/fleet_alpha/statistics')
    if response.success:
        print(response.data)
```

### TripUploader
Uploads trip summaries to cloud backend with offline queueing.

**Features:**
- Periodic upload (every 5 minutes)
- GPS anonymization (rounded to ~1km precision)
- Offline queueing with persistence
- Background thread operation

**Usage:**
```python
from src.cloud import TripUploader

uploader = TripUploader(api_client)

# Start trip
trip_id = uploader.start_trip(driver_id="driver_123")

# Update trip
uploader.update_trip(
    distance_delta=100.0,  # meters
    speed=15.0,  # m/s
    risk_score=0.3,
    alert_urgency='warning'
)

# End trip
uploader.end_trip()

# Start background upload
uploader.start_background_upload()
```

### ScenarioUploader
Uploads recorded scenarios with video compression.

**Features:**
- User consent checking
- Video compression with ffmpeg
- Background upload
- Upload status tracking

**Usage:**
```python
from src.cloud import ScenarioUploader

uploader = ScenarioUploader(api_client, upload_consent=True)

# Queue scenario for upload
uploader.queue_scenario("scenarios/2024-11-17_10-30-45", priority='high')

# Start background upload
uploader.start_background_upload()

# Check status
status = uploader.get_upload_status("2024-11-17_10-30-45")
```

### ModelDownloader
Downloads and installs model updates with signature verification.

**Features:**
- Periodic update checks (every 24 hours)
- SHA256 checksum verification
- Digital signature verification
- Atomic installation with rollback
- Background check thread

**Usage:**
```python
from src.cloud import ModelDownloader

downloader = ModelDownloader(api_client, auto_install=True)

# Start background checks
downloader.start_background_checks()

# Manual check
updates = downloader.check_for_updates()

# Manual download
downloader.download_and_install_model(
    "yolov8m_automotive",
    "v2.1.0",
    expected_checksum="abc123...",
    signature="signature..."
)
```

### ProfileSynchronizer
Synchronizes driver profiles across vehicles with encryption.

**Features:**
- Profile encryption (Fernet symmetric encryption)
- Upload/download profiles
- Merge profiles across vehicles
- Conflict resolution

**Usage:**
```python
from src.cloud import ProfileSynchronizer

sync = ProfileSynchronizer(api_client)

# Upload profile
sync.upload_profile("driver_123", profile_data)

# Download profile
profile = sync.download_profile("driver_123")

# Sync (merge local and cloud)
merged = sync.sync_profile("driver_123", local_profile)

# Sync all profiles
sync.sync_all_profiles()
```

### FleetManager
Provides fleet-wide statistics and analytics.

**Features:**
- Fleet statistics
- Vehicle rankings
- Trend visualization
- Driver leaderboard
- Alert aggregation

**Usage:**
```python
from src.cloud import FleetManager

manager = FleetManager(api_client)

# Get fleet statistics
stats = manager.get_fleet_statistics()
print(f"Total vehicles: {stats['total_vehicles']}")

# Get vehicle rankings
rankings = manager.get_vehicle_rankings(metric='safety_score', limit=10)

# Get trends
trends = manager.get_fleet_trends(metric='safety_score', days=30)

# Get driver leaderboard
leaderboard = manager.get_driver_leaderboard(metric='safety_score')
```

### OfflineManager
Manages offline operation and automatic synchronization.

**Features:**
- Connectivity monitoring
- Offline detection
- Automatic sync when online
- Status notifications

**Usage:**
```python
from src.cloud import OfflineManager

def on_connectivity_change(status):
    print(f"Connectivity: {status.value}")

manager = OfflineManager(api_client, connectivity_callback=on_connectivity_change)

# Register sync callbacks
manager.register_sync_callback(trip_uploader._process_upload_queue)
manager.register_sync_callback(scenario_uploader._process_next_upload)

# Start monitoring
manager.start_monitoring()

# Check status
if manager.is_online():
    print("Connected to cloud")
else:
    print(f"Offline for {manager.get_offline_duration():.0f} seconds")
```

## Configuration

Cloud settings are configured in `configs/default.yaml`:

```yaml
cloud:
  enabled: false
  api_url: "https://api.sentinel-fleet.com"
  api_key: "your_api_key"
  vehicle_id: "vehicle_001"
  fleet_id: "fleet_alpha"
  sync_interval: 300  # seconds
  upload_scenarios: true
```

## API Endpoints

The cloud backend provides the following endpoints:

### Health Check
- `GET /health` - Check API availability

### Trips
- `POST /trips` - Upload trip summary

### Scenarios
- `POST /scenarios` - Upload scenario metadata
- `POST /scenarios/{id}/videos` - Upload scenario videos
- `POST /scenarios/{id}/annotations` - Upload annotations

### Models
- `GET /models` - List available models
- `GET /models/{name}/download` - Get model download URL

### Driver Profiles
- `GET /drivers/{id}/profile` - Download driver profile
- `PUT /drivers/{id}/profile` - Upload driver profile

### Fleet Statistics
- `GET /fleet/{id}/statistics` - Get fleet statistics
- `GET /fleet/{id}/rankings` - Get vehicle rankings
- `GET /fleet/{id}/trends` - Get trend data
- `GET /fleet/{id}/comparison` - Compare vehicles
- `GET /fleet/{id}/alerts` - Get fleet alerts
- `GET /fleet/{id}/drivers/leaderboard` - Get driver leaderboard

### Vehicle Status
- `GET /vehicles/{id}/status` - Get vehicle status

## Security

### Authentication
All API requests include Bearer token authentication:
```
Authorization: Bearer {api_key}
```

### Encryption
- Driver profiles are encrypted using Fernet symmetric encryption
- Encryption key derived from vehicle ID using PBKDF2
- In production, use secure key management system

### Data Privacy
- GPS coordinates anonymized (rounded to ~1km precision)
- Trip data contains no personally identifiable information
- Scenario upload requires explicit user consent
- All data transmitted over HTTPS

## Error Handling

All components implement robust error handling:
- Connection errors: Automatic retry with exponential backoff
- Timeout errors: Configurable timeout with fallback
- API errors: Logged with error details
- Offline operation: Queue operations for later sync

## Testing

Test cloud components with mock server:

```python
# tests/test_cloud.py
def test_api_client():
    client = CloudAPIClient(
        api_url="http://localhost:8000",
        api_key="test_key",
        vehicle_id="test_vehicle",
        fleet_id="test_fleet"
    )
    
    response = client.get('/health')
    assert response.success
```

## Dependencies

Required packages:
- `requests` - HTTP client
- `urllib3` - Connection pooling
- `cryptography` - Profile encryption

Optional:
- `ffmpeg` - Video compression (system package)
