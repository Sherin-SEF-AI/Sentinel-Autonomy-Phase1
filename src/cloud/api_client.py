"""
Cloud API client for SENTINEL fleet management.

Provides REST API communication with retry logic, rate limiting,
and authentication.
"""

import time
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


@dataclass
class APIResponse:
    """API response wrapper"""
    success: bool
    status_code: int
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class RateLimiter:
    """Simple token bucket rate limiter"""
    
    def __init__(self, requests_per_second: float = 10.0):
        self.requests_per_second = requests_per_second
        self.tokens = requests_per_second
        self.last_update = time.time()
        self.max_tokens = requests_per_second
    
    def acquire(self) -> None:
        """Wait until a token is available"""
        while True:
            now = time.time()
            elapsed = now - self.last_update
            self.last_update = now
            
            # Add tokens based on elapsed time
            self.tokens = min(
                self.max_tokens,
                self.tokens + elapsed * self.requests_per_second
            )
            
            if self.tokens >= 1.0:
                self.tokens -= 1.0
                break
            
            # Wait for next token
            time.sleep((1.0 - self.tokens) / self.requests_per_second)


class CloudAPIClient:
    """
    REST API client for SENTINEL cloud backend.
    
    Features:
    - Authentication with API key
    - Automatic retry with exponential backoff
    - Rate limiting
    - Connection pooling
    - Timeout handling
    """
    
    def __init__(
        self,
        api_url: str,
        api_key: str,
        vehicle_id: str,
        fleet_id: str,
        timeout: float = 30.0,
        max_retries: int = 3,
        rate_limit: float = 10.0
    ):
        """
        Initialize cloud API client.
        
        Args:
            api_url: Base URL for API (e.g., https://api.sentinel-fleet.com)
            api_key: Authentication API key
            vehicle_id: Unique vehicle identifier
            fleet_id: Fleet identifier
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
            rate_limit: Maximum requests per second
        """
        self.api_url = api_url.rstrip('/')
        self.api_key = api_key
        self.vehicle_id = vehicle_id
        self.fleet_id = fleet_id
        self.timeout = timeout
        
        self.logger = logging.getLogger(__name__)
        
        # Configure session with retry logic
        self.session = requests.Session()
        
        # Retry strategy: retry on connection errors and 5xx server errors
        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=1,  # Wait 1, 2, 4 seconds between retries
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "PUT", "POST", "DELETE", "OPTIONS", "TRACE"]
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy, pool_connections=10, pool_maxsize=20)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        # Set default headers
        self.session.headers.update({
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json',
            'User-Agent': f'SENTINEL-Vehicle/{self.vehicle_id}',
            'X-Fleet-ID': self.fleet_id,
            'X-Vehicle-ID': self.vehicle_id
        })
        
        # Rate limiter
        self.rate_limiter = RateLimiter(rate_limit)
        
        self.logger.info(f"CloudAPIClient initialized for vehicle {vehicle_id} in fleet {fleet_id}")
    
    def _make_request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        files: Optional[Dict[str, Any]] = None
    ) -> APIResponse:
        """
        Make HTTP request with rate limiting and error handling.
        
        Args:
            method: HTTP method (GET, POST, PUT, DELETE)
            endpoint: API endpoint (e.g., '/trips')
            data: JSON data for request body
            params: Query parameters
            files: Files to upload (multipart/form-data)
        
        Returns:
            APIResponse with success status and data or error
        """
        # Apply rate limiting
        self.rate_limiter.acquire()
        
        url = f"{self.api_url}{endpoint}"
        
        try:
            self.logger.debug(f"{method} {url}")
            
            # Prepare request kwargs
            kwargs = {
                'timeout': self.timeout,
                'params': params
            }
            
            if files:
                # Multipart upload - don't set Content-Type header
                kwargs['files'] = files
                if data:
                    kwargs['data'] = data
                # Remove Content-Type header for multipart
                headers = self.session.headers.copy()
                headers.pop('Content-Type', None)
                kwargs['headers'] = headers
            elif data:
                kwargs['json'] = data
            
            response = self.session.request(method, url, **kwargs)
            
            # Check for success
            if response.status_code >= 200 and response.status_code < 300:
                try:
                    response_data = response.json() if response.content else {}
                except ValueError:
                    response_data = {}
                
                self.logger.debug(f"Request successful: {response.status_code}")
                return APIResponse(
                    success=True,
                    status_code=response.status_code,
                    data=response_data
                )
            else:
                error_msg = f"HTTP {response.status_code}: {response.text}"
                self.logger.warning(f"Request failed: {error_msg}")
                return APIResponse(
                    success=False,
                    status_code=response.status_code,
                    error=error_msg
                )
        
        except requests.exceptions.Timeout:
            error_msg = f"Request timeout after {self.timeout}s"
            self.logger.error(error_msg)
            return APIResponse(success=False, status_code=0, error=error_msg)
        
        except requests.exceptions.ConnectionError as e:
            error_msg = f"Connection error: {str(e)}"
            self.logger.error(error_msg)
            return APIResponse(success=False, status_code=0, error=error_msg)
        
        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            return APIResponse(success=False, status_code=0, error=error_msg)
    
    def get(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> APIResponse:
        """Make GET request"""
        return self._make_request('GET', endpoint, params=params)
    
    def post(
        self,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        files: Optional[Dict[str, Any]] = None
    ) -> APIResponse:
        """Make POST request"""
        return self._make_request('POST', endpoint, data=data, files=files)
    
    def put(self, endpoint: str, data: Optional[Dict[str, Any]] = None) -> APIResponse:
        """Make PUT request"""
        return self._make_request('PUT', endpoint, data=data)
    
    def delete(self, endpoint: str) -> APIResponse:
        """Make DELETE request"""
        return self._make_request('DELETE', endpoint)
    
    def check_connectivity(self) -> bool:
        """
        Check if API is reachable.
        
        Returns:
            True if API responds to health check
        """
        try:
            response = self.get('/health')
            return response.success
        except Exception:
            return False
    
    def close(self) -> None:
        """Close session and cleanup resources"""
        self.session.close()
        self.logger.info("CloudAPIClient closed")
