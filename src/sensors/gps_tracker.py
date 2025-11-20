"""GPS tracking and speed limit lookup functionality."""

import logging
import time
from typing import Optional, Tuple, Dict
from dataclasses import dataclass
from pathlib import Path
import json

# For real GPS hardware integration, you would use:
# import serial  # For NMEA GPS devices
# import pynmea2  # For parsing NMEA sentences
# or gpsd for GPS daemon integration


@dataclass
class GPSData:
    """GPS data structure."""
    latitude: float
    longitude: float
    altitude: float  # meters
    speed: float  # m/s (from GPS)
    heading: float  # degrees (0-360, 0=North)
    timestamp: float
    satellites: int
    fix_quality: int  # 0=no fix, 1=GPS, 2=DGPS
    hdop: float  # Horizontal dilution of precision


@dataclass
class SpeedLimitInfo:
    """Speed limit information for current location."""
    speed_limit: Optional[float]  # km/h
    road_name: Optional[str]
    road_type: Optional[str]  # 'highway', 'arterial', 'residential', etc.
    confidence: float  # 0-1
    source: str  # 'osm', 'cached', 'default'


class GPSTracker:
    """
    GPS tracking system for location, speed, and heading.

    Supports:
    - Real GPS hardware (NMEA via serial)
    - GPS daemon (gpsd)
    - Simulation mode for testing
    """

    def __init__(self, config: Dict):
        """
        Initialize GPS tracker.

        Args:
            config: Configuration dictionary
        """
        self.logger = logging.getLogger(__name__)
        self.config = config

        self.enabled = config.get('enabled', False)
        self.device = config.get('device', '/dev/ttyUSB0')
        self.baudrate = config.get('baudrate', 9600)
        self.simulation_mode = config.get('simulation', True)

        # Current GPS data
        self.current_gps: Optional[GPSData] = None
        self.last_update_time = 0.0

        # Speed limit database (simplified - in production use OSM or commercial API)
        self.speed_limits_cache: Dict[Tuple[float, float], SpeedLimitInfo] = {}
        self.cache_file = Path(config.get('cache_file', 'data/speed_limits_cache.json'))

        # GPS hardware connection
        self.gps_device = None

        if self.enabled:
            self._initialize_gps()
            self._load_speed_limit_cache()

        self.logger.info(f"GPS tracker initialized (enabled={self.enabled}, simulation={self.simulation_mode})")

    def _initialize_gps(self):
        """Initialize GPS hardware connection."""
        if self.simulation_mode:
            self.logger.info("GPS in simulation mode")
            return

        try:
            # For real GPS hardware, you would do:
            # import serial
            # self.gps_device = serial.Serial(self.device, self.baudrate, timeout=1)
            # self.logger.info(f"GPS device opened: {self.device}")

            # Or for gpsd:
            # import gps
            # self.gps_session = gps.gps(mode=gps.WATCH_ENABLE)

            self.logger.warning("Real GPS hardware not implemented - using simulation")
            self.simulation_mode = True

        except Exception as e:
            self.logger.error(f"Failed to initialize GPS: {e}")
            self.simulation_mode = True

    def _load_speed_limit_cache(self):
        """Load cached speed limit data."""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r') as f:
                    cache_data = json.load(f)

                # Convert string keys back to tuples
                for key_str, value in cache_data.items():
                    lat, lon = map(float, key_str.split(','))
                    self.speed_limits_cache[(lat, lon)] = SpeedLimitInfo(**value)

                self.logger.info(f"Loaded {len(self.speed_limits_cache)} cached speed limits")
            except Exception as e:
                self.logger.error(f"Failed to load speed limit cache: {e}")

    def _save_speed_limit_cache(self):
        """Save speed limit cache to disk."""
        try:
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)

            # Convert tuple keys to strings for JSON
            cache_data = {}
            for (lat, lon), info in self.speed_limits_cache.items():
                key = f"{lat:.6f},{lon:.6f}"
                cache_data[key] = {
                    'speed_limit': info.speed_limit,
                    'road_name': info.road_name,
                    'road_type': info.road_type,
                    'confidence': info.confidence,
                    'source': info.source
                }

            with open(self.cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2)

        except Exception as e:
            self.logger.error(f"Failed to save speed limit cache: {e}")

    def update(self) -> Optional[GPSData]:
        """
        Update GPS data from hardware or simulation.

        Returns:
            Current GPS data
        """
        if not self.enabled:
            return None

        try:
            if self.simulation_mode:
                return self._simulate_gps()
            else:
                return self._read_gps_hardware()
        except Exception as e:
            self.logger.error(f"GPS update failed: {e}")
            return None

    def _read_gps_hardware(self) -> Optional[GPSData]:
        """Read GPS data from hardware device."""
        # For real GPS hardware:
        #
        # Example with NMEA serial:
        # line = self.gps_device.readline().decode('ascii', errors='ignore')
        # if line.startswith('$GPGGA') or line.startswith('$GPRMC'):
        #     msg = pynmea2.parse(line)
        #     gps_data = GPSData(
        #         latitude=msg.latitude,
        #         longitude=msg.longitude,
        #         altitude=msg.altitude if hasattr(msg, 'altitude') else 0.0,
        #         speed=msg.spd_over_grnd if hasattr(msg, 'spd_over_grnd') else 0.0,
        #         heading=msg.true_course if hasattr(msg, 'true_course') else 0.0,
        #         timestamp=time.time(),
        #         satellites=msg.num_sats if hasattr(msg, 'num_sats') else 0,
        #         fix_quality=msg.gps_qual if hasattr(msg, 'gps_qual') else 0,
        #         hdop=msg.horizontal_dil if hasattr(msg, 'horizontal_dil') else 99.9
        #     )
        #     self.current_gps = gps_data
        #     return gps_data

        # Example with gpsd:
        # report = self.gps_session.next()
        # if report['class'] == 'TPV':
        #     gps_data = GPSData(...)

        self.logger.warning("Real GPS hardware not implemented")
        return self._simulate_gps()

    def _simulate_gps(self) -> GPSData:
        """Simulate GPS data for testing."""
        # Simulate a vehicle moving along a route
        current_time = time.time()

        if self.current_gps is None:
            # Starting position (example: San Francisco)
            lat = 37.7749
            lon = -122.4194
            alt = 10.0
            speed = 0.0
            heading = 0.0
        else:
            # Simulate movement
            dt = current_time - self.current_gps.timestamp

            # Slowly increase speed to simulate acceleration
            speed = min(self.current_gps.speed + 0.5 * dt, 13.89)  # Max 50 km/h = 13.89 m/s

            # Move north-east
            heading = 45.0

            # Update position based on speed and heading
            # Rough approximation: 1 degree latitude ≈ 111 km
            # 1 degree longitude ≈ 111 km * cos(latitude)
            import math

            distance = speed * dt  # meters
            lat_change = (distance / 111000.0) * math.cos(math.radians(heading))
            lon_change = (distance / (111000.0 * math.cos(math.radians(self.current_gps.latitude)))) * math.sin(math.radians(heading))

            lat = self.current_gps.latitude + lat_change
            lon = self.current_gps.longitude + lon_change
            alt = self.current_gps.altitude + 0.01  # Slight climb

        gps_data = GPSData(
            latitude=lat,
            longitude=lon,
            altitude=alt,
            speed=speed,
            heading=heading,
            timestamp=current_time,
            satellites=8,
            fix_quality=1,
            hdop=1.2
        )

        self.current_gps = gps_data
        self.last_update_time = current_time

        return gps_data

    def get_speed_limit(self, lat: float, lon: float) -> SpeedLimitInfo:
        """
        Get speed limit for given coordinates.

        Args:
            lat: Latitude
            lon: Longitude

        Returns:
            Speed limit information
        """
        # Round coordinates to reduce cache size
        lat_rounded = round(lat, 4)
        lon_rounded = round(lon, 4)

        cache_key = (lat_rounded, lon_rounded)

        # Check cache first
        if cache_key in self.speed_limits_cache:
            cached = self.speed_limits_cache[cache_key]
            return cached

        # In production, query OpenStreetMap Overpass API or commercial service:
        #
        # Example with Overpass API:
        # query = f"""
        # [out:json];
        # way(around:50,{lat},{lon})["highway"];
        # out tags;
        # """
        # response = requests.post('https://overpass-api.de/api/interpreter', data={'data': query})
        # data = response.json()
        #
        # Extract speed limit from tags:
        # for element in data.get('elements', []):
        #     tags = element.get('tags', {})
        #     if 'maxspeed' in tags:
        #         speed_limit = float(tags['maxspeed'])
        #         road_type = tags.get('highway', 'unknown')
        #         ...

        # For now, use default speed limits based on road type heuristics
        speed_limit_info = self._get_default_speed_limit(lat, lon)

        # Cache the result
        self.speed_limits_cache[cache_key] = speed_limit_info

        return speed_limit_info

    def _get_default_speed_limit(self, lat: float, lon: float) -> SpeedLimitInfo:
        """Get default speed limit based on heuristics."""
        # In absence of real data, provide conservative defaults
        # This should be replaced with real OSM or commercial API data

        return SpeedLimitInfo(
            speed_limit=50.0,  # km/h - default urban speed limit
            road_name=None,
            road_type='urban',
            confidence=0.5,
            source='default'
        )

    def check_speed_violation(self, current_speed_kmh: float) -> Optional[Dict]:
        """
        Check if current speed exceeds speed limit.

        Args:
            current_speed_kmh: Current vehicle speed in km/h

        Returns:
            Violation info dict if speeding, None otherwise
        """
        if not self.enabled or self.current_gps is None:
            return None

        speed_limit_info = self.get_speed_limit(
            self.current_gps.latitude,
            self.current_gps.longitude
        )

        if speed_limit_info.speed_limit is None:
            return None

        excess = current_speed_kmh - speed_limit_info.speed_limit

        if excess > 0:
            # Speeding detected
            severity = 'low'
            if excess > 20:
                severity = 'critical'
            elif excess > 10:
                severity = 'high'
            elif excess > 5:
                severity = 'medium'

            return {
                'type': 'speed_violation',
                'current_speed': current_speed_kmh,
                'speed_limit': speed_limit_info.speed_limit,
                'excess': excess,
                'severity': severity,
                'road_name': speed_limit_info.road_name,
                'road_type': speed_limit_info.road_type
            }

        return None

    def get_location_info(self) -> Optional[Dict]:
        """
        Get current location information.

        Returns:
            Dictionary with location details
        """
        if not self.enabled or self.current_gps is None:
            return None

        speed_limit_info = self.get_speed_limit(
            self.current_gps.latitude,
            self.current_gps.longitude
        )

        return {
            'latitude': self.current_gps.latitude,
            'longitude': self.current_gps.longitude,
            'altitude': self.current_gps.altitude,
            'speed_gps': self.current_gps.speed * 3.6,  # m/s to km/h
            'heading': self.current_gps.heading,
            'satellites': self.current_gps.satellites,
            'fix_quality': self.current_gps.fix_quality,
            'hdop': self.current_gps.hdop,
            'speed_limit': speed_limit_info.speed_limit,
            'road_name': speed_limit_info.road_name,
            'road_type': speed_limit_info.road_type
        }

    def shutdown(self):
        """Shutdown GPS tracker and save cache."""
        if self.enabled:
            self._save_speed_limit_cache()

            if self.gps_device is not None:
                try:
                    self.gps_device.close()
                except Exception as e:
                    self.logger.error(f"Error closing GPS device: {e}")

        self.logger.info("GPS tracker shut down")
