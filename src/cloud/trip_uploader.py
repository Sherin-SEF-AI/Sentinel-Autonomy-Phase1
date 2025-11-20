"""
Trip data uploader for cloud synchronization.

Collects trip summaries and uploads them periodically to the cloud backend.
"""

import time
import logging
import threading
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from datetime import datetime
from queue import Queue
import json
import os

from .api_client import CloudAPIClient


@dataclass
class TripSummary:
    """Trip summary data"""
    trip_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    duration: float = 0.0  # seconds
    distance: float = 0.0  # meters
    average_speed: float = 0.0  # m/s
    max_speed: float = 0.0  # m/s
    alert_count: int = 0
    critical_alert_count: int = 0
    warning_alert_count: int = 0
    info_alert_count: int = 0
    average_risk_score: float = 0.0
    max_risk_score: float = 0.0
    driver_id: Optional[str] = None
    # GPS coordinates anonymized (rounded to ~1km precision)
    start_location: Optional[Dict[str, float]] = None
    end_location: Optional[Dict[str, float]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        data = asdict(self)
        # Convert datetime to ISO format
        if self.start_time:
            data['start_time'] = self.start_time.isoformat()
        if self.end_time:
            data['end_time'] = self.end_time.isoformat()
        return data


class TripUploader:
    """
    Uploads trip summaries to cloud backend.
    
    Features:
    - Periodic upload every 5 minutes
    - Offline queueing
    - GPS anonymization
    - Background thread operation
    """
    
    def __init__(
        self,
        api_client: CloudAPIClient,
        upload_interval: float = 300.0,  # 5 minutes
        queue_file: str = "data/trip_queue.json"
    ):
        """
        Initialize trip uploader.
        
        Args:
            api_client: Cloud API client
            upload_interval: Upload interval in seconds
            queue_file: File to persist offline queue
        """
        self.api_client = api_client
        self.upload_interval = upload_interval
        self.queue_file = queue_file
        
        self.logger = logging.getLogger(__name__)
        
        # Current trip being tracked
        self.current_trip: Optional[TripSummary] = None
        
        # Queue for offline trips
        self.upload_queue: Queue = Queue()
        
        # Load persisted queue
        self._load_queue()
        
        # Background upload thread
        self.running = False
        self.upload_thread: Optional[threading.Thread] = None
        
        # Statistics
        self.total_distance = 0.0
        self.total_alerts = 0
        self.risk_scores: List[float] = []
        
        self.logger.info("TripUploader initialized")
    
    def start_trip(self, driver_id: Optional[str] = None, location: Optional[Dict[str, float]] = None) -> str:
        """
        Start a new trip.
        
        Args:
            driver_id: Optional driver identifier
            location: Optional GPS location {'lat': float, 'lon': float}
        
        Returns:
            Trip ID
        """
        trip_id = f"trip_{int(time.time() * 1000)}"
        
        # Anonymize location (round to ~1km precision)
        anonymized_location = None
        if location:
            anonymized_location = {
                'lat': round(location['lat'], 2),
                'lon': round(location['lon'], 2)
            }
        
        self.current_trip = TripSummary(
            trip_id=trip_id,
            start_time=datetime.now(),
            driver_id=driver_id,
            start_location=anonymized_location
        )
        
        self.logger.info(f"Started trip {trip_id}")
        return trip_id
    
    def update_trip(
        self,
        distance_delta: float = 0.0,
        speed: float = 0.0,
        risk_score: float = 0.0,
        alert_urgency: Optional[str] = None
    ) -> None:
        """
        Update current trip with new data.
        
        Args:
            distance_delta: Distance traveled since last update (meters)
            speed: Current speed (m/s)
            risk_score: Current risk score (0-1)
            alert_urgency: Alert urgency if alert was generated ('info', 'warning', 'critical')
        """
        if not self.current_trip:
            return
        
        # Update distance
        self.current_trip.distance += distance_delta
        self.total_distance += distance_delta
        
        # Update speed
        if speed > self.current_trip.max_speed:
            self.current_trip.max_speed = speed
        
        # Update risk scores
        self.risk_scores.append(risk_score)
        if risk_score > self.current_trip.max_risk_score:
            self.current_trip.max_risk_score = risk_score
        
        # Update alerts
        if alert_urgency:
            self.current_trip.alert_count += 1
            self.total_alerts += 1
            
            if alert_urgency == 'critical':
                self.current_trip.critical_alert_count += 1
            elif alert_urgency == 'warning':
                self.current_trip.warning_alert_count += 1
            elif alert_urgency == 'info':
                self.current_trip.info_alert_count += 1
    
    def end_trip(self, location: Optional[Dict[str, float]] = None) -> Optional[TripSummary]:
        """
        End current trip and queue for upload.
        
        Args:
            location: Optional GPS location {'lat': float, 'lon': float}
        
        Returns:
            Completed trip summary
        """
        if not self.current_trip:
            return None
        
        # Finalize trip
        self.current_trip.end_time = datetime.now()
        self.current_trip.duration = (
            self.current_trip.end_time - self.current_trip.start_time
        ).total_seconds()
        
        # Calculate average speed
        if self.current_trip.duration > 0:
            self.current_trip.average_speed = self.current_trip.distance / self.current_trip.duration
        
        # Calculate average risk score
        if self.risk_scores:
            self.current_trip.average_risk_score = sum(self.risk_scores) / len(self.risk_scores)
            self.risk_scores.clear()
        
        # Anonymize end location
        if location:
            self.current_trip.end_location = {
                'lat': round(location['lat'], 2),
                'lon': round(location['lon'], 2)
            }
        
        # Queue for upload
        self.upload_queue.put(self.current_trip)
        self._save_queue()
        
        self.logger.info(f"Ended trip {self.current_trip.trip_id}: "
                        f"{self.current_trip.distance:.1f}m, "
                        f"{self.current_trip.duration:.1f}s, "
                        f"{self.current_trip.alert_count} alerts")
        
        completed_trip = self.current_trip
        self.current_trip = None
        
        return completed_trip
    
    def start_background_upload(self) -> None:
        """Start background upload thread"""
        if self.running:
            return
        
        self.running = True
        self.upload_thread = threading.Thread(target=self._upload_loop, daemon=True)
        self.upload_thread.start()
        self.logger.info("Background upload thread started")
    
    def stop_background_upload(self) -> None:
        """Stop background upload thread"""
        if not self.running:
            return
        
        self.running = False
        if self.upload_thread:
            self.upload_thread.join(timeout=5.0)
        self.logger.info("Background upload thread stopped")
    
    def _upload_loop(self) -> None:
        """Background upload loop"""
        last_upload = 0.0
        
        while self.running:
            try:
                now = time.time()
                
                # Check if it's time to upload
                if now - last_upload >= self.upload_interval:
                    self._process_upload_queue()
                    last_upload = now
                
                # Sleep for a short interval
                time.sleep(1.0)
            
            except Exception as e:
                self.logger.error(f"Error in upload loop: {e}", exc_info=True)
                time.sleep(5.0)
    
    def _process_upload_queue(self) -> None:
        """Process queued trips and upload to cloud"""
        if self.upload_queue.empty():
            return
        
        # Check connectivity
        if not self.api_client.check_connectivity():
            self.logger.warning("No connectivity, skipping upload")
            return
        
        uploaded_count = 0
        failed_trips: List[TripSummary] = []
        
        # Process all queued trips
        while not self.upload_queue.empty():
            try:
                trip = self.upload_queue.get_nowait()
                
                # Upload trip
                response = self.api_client.post('/trips', data=trip.to_dict())
                
                if response.success:
                    uploaded_count += 1
                    self.logger.debug(f"Uploaded trip {trip.trip_id}")
                else:
                    # Re-queue failed upload
                    failed_trips.append(trip)
                    self.logger.warning(f"Failed to upload trip {trip.trip_id}: {response.error}")
            
            except Exception as e:
                self.logger.error(f"Error uploading trip: {e}", exc_info=True)
        
        # Re-queue failed trips
        for trip in failed_trips:
            self.upload_queue.put(trip)
        
        if uploaded_count > 0:
            self.logger.info(f"Uploaded {uploaded_count} trips")
            self._save_queue()
    
    def _save_queue(self) -> None:
        """Persist upload queue to disk"""
        try:
            # Create directory if needed
            os.makedirs(os.path.dirname(self.queue_file), exist_ok=True)
            
            # Convert queue to list
            trips = []
            temp_queue = Queue()
            
            while not self.upload_queue.empty():
                trip = self.upload_queue.get()
                trips.append(trip.to_dict())
                temp_queue.put(trip)
            
            # Restore queue
            while not temp_queue.empty():
                self.upload_queue.put(temp_queue.get())
            
            # Save to file
            with open(self.queue_file, 'w') as f:
                json.dump(trips, f, indent=2)
        
        except Exception as e:
            self.logger.error(f"Error saving queue: {e}", exc_info=True)
    
    def _load_queue(self) -> None:
        """Load persisted upload queue from disk"""
        if not os.path.exists(self.queue_file):
            return
        
        try:
            with open(self.queue_file, 'r') as f:
                trips_data = json.load(f)
            
            for trip_data in trips_data:
                # Convert ISO strings back to datetime
                if 'start_time' in trip_data:
                    trip_data['start_time'] = datetime.fromisoformat(trip_data['start_time'])
                if 'end_time' in trip_data and trip_data['end_time']:
                    trip_data['end_time'] = datetime.fromisoformat(trip_data['end_time'])
                
                trip = TripSummary(**trip_data)
                self.upload_queue.put(trip)
            
            self.logger.info(f"Loaded {len(trips_data)} trips from queue")
        
        except Exception as e:
            self.logger.error(f"Error loading queue: {e}", exc_info=True)
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get trip statistics.
        
        Returns:
            Dictionary with statistics
        """
        return {
            'total_distance': self.total_distance,
            'total_alerts': self.total_alerts,
            'queued_trips': self.upload_queue.qsize(),
            'current_trip_active': self.current_trip is not None
        }
