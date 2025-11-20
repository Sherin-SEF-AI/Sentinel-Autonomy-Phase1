"""
FastAPI backend server for SENTINEL visualization dashboard.

Provides:
- WebSocket endpoint for real-time data streaming
- REST endpoints for configuration management
- REST endpoints for scenario playback
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from typing import Dict, List, Optional, Any
import asyncio
import json
import logging
from pathlib import Path
from datetime import datetime

from src.core.data_structures import (
    BEVOutput, SegmentationOutput, Detection3D, DriverState,
    RiskAssessment, Alert
)

logger = logging.getLogger("VisualizationBackend")


class ConnectionManager:
    """Manages WebSocket connections for real-time streaming."""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        """Accept and register a new WebSocket connection."""
        logger.debug(f"Accepting WebSocket connection from {websocket.client}")
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket client connected: client={websocket.client}, total_connections={len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket):
        """Remove a WebSocket connection."""
        logger.debug(f"Disconnecting WebSocket client: client={websocket.client}")
        self.active_connections.remove(websocket)
        logger.info(f"WebSocket client disconnected: client={websocket.client}, remaining_connections={len(self.active_connections)}")
    
    async def broadcast(self, message: Dict[str, Any]):
        """Broadcast message to all connected clients."""
        if not self.active_connections:
            logger.debug("Broadcast skipped: no active connections")
            return
        
        logger.debug(f"Broadcasting message to {len(self.active_connections)} clients: message_type={message.get('type', 'unknown')}")
        disconnected = []
        success_count = 0
        
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
                success_count += 1
            except Exception as e:
                logger.error(f"Broadcast failed to client: client={connection.client}, error={e}")
                disconnected.append(connection)
        
        # Clean up disconnected clients
        for connection in disconnected:
            if connection in self.active_connections:
                self.active_connections.remove(connection)
        
        if disconnected:
            logger.warning(f"Broadcast completed with failures: success={success_count}, failed={len(disconnected)}, remaining_connections={len(self.active_connections)}")
        else:
            logger.debug(f"Broadcast completed successfully: clients={success_count}")


class VisualizationServer:
    """FastAPI server for SENTINEL visualization."""
    
    def __init__(self, config: Dict[str, Any]):
        logger.debug("Initializing VisualizationServer")
        self.config = config
        self.app = FastAPI(title="SENTINEL Visualization API", version="1.0")
        self.connection_manager = ConnectionManager()
        
        # Configure CORS
        logger.debug("Configuring CORS middleware")
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # Configure appropriately for production
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Register routes
        logger.debug("Registering API routes")
        self._register_routes()
        
        # Mount frontend static files
        frontend_path = Path(__file__).parent.parent / "frontend"
        if frontend_path.exists():
            self.app.mount("/", StaticFiles(directory=str(frontend_path), html=True), name="frontend")
            logger.info(f"Frontend static files mounted: path={frontend_path}")
        else:
            logger.warning(f"Frontend directory not found: path={frontend_path}")
        
        # State
        self.latest_data: Optional[Dict[str, Any]] = None
        self.scenarios_path = Path(config.get('recording', {}).get('storage_path', 'scenarios/'))
        
        logger.info(f"VisualizationServer initialized: scenarios_path={self.scenarios_path}")
    
    def _register_routes(self):
        """Register all API routes."""
        
        @self.app.get("/")
        async def root():
            """Health check endpoint."""
            return {"status": "ok", "service": "SENTINEL Visualization API"}
        
        @self.app.get("/api/config")
        async def get_config():
            """Get current system configuration."""
            logger.debug("Configuration requested via API")
            return JSONResponse(content=self.config)
        
        @self.app.post("/api/config")
        async def update_config(config_update: Dict[str, Any]):
            """Update system configuration."""
            logger.debug(f"Configuration update requested: keys={list(config_update.keys())}")
            try:
                # Merge configuration updates
                self._merge_config(self.config, config_update)
                logger.info(f"Configuration updated successfully: updates={config_update}")
                return {"status": "success", "message": "Configuration updated"}
            except Exception as e:
                logger.error(f"Configuration update failed: error={e}", exc_info=True)
                raise HTTPException(status_code=400, detail=str(e))
        
        @self.app.get("/api/scenarios")
        async def list_scenarios():
            """List all recorded scenarios."""
            logger.debug(f"Listing scenarios from: path={self.scenarios_path}")
            try:
                scenarios = []
                if self.scenarios_path.exists():
                    for scenario_dir in sorted(self.scenarios_path.iterdir(), reverse=True):
                        if scenario_dir.is_dir():
                            metadata_file = scenario_dir / "metadata.json"
                            if metadata_file.exists():
                                with open(metadata_file, 'r') as f:
                                    metadata = json.load(f)
                                    metadata['id'] = scenario_dir.name
                                    scenarios.append(metadata)
                
                logger.debug(f"Scenarios listed: count={len(scenarios)}")
                return {"scenarios": scenarios}
            except Exception as e:
                logger.error(f"Scenario listing failed: error={e}", exc_info=True)
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/scenarios/{scenario_id}")
        async def get_scenario(scenario_id: str):
            """Get details of a specific scenario."""
            logger.debug(f"Scenario details requested: scenario_id={scenario_id}")
            try:
                scenario_dir = self.scenarios_path / scenario_id
                if not scenario_dir.exists():
                    logger.warning(f"Scenario not found: scenario_id={scenario_id}, path={scenario_dir}")
                    raise HTTPException(status_code=404, detail="Scenario not found")
                
                # Load metadata
                metadata_file = scenario_dir / "metadata.json"
                annotations_file = scenario_dir / "annotations.json"
                
                result = {"id": scenario_id}
                
                if metadata_file.exists():
                    with open(metadata_file, 'r') as f:
                        result['metadata'] = json.load(f)
                    logger.debug(f"Scenario metadata loaded: scenario_id={scenario_id}")
                
                if annotations_file.exists():
                    with open(annotations_file, 'r') as f:
                        annotations = json.load(f)
                        result['annotations'] = annotations
                    logger.debug(f"Scenario annotations loaded: scenario_id={scenario_id}, frame_count={len(annotations.get('frames', []))}")
                
                logger.debug(f"Scenario details retrieved: scenario_id={scenario_id}")
                return result
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Scenario retrieval failed: scenario_id={scenario_id}, error={e}", exc_info=True)
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.delete("/api/scenarios/{scenario_id}")
        async def delete_scenario(scenario_id: str):
            """Delete a recorded scenario."""
            logger.info(f"Scenario deletion requested: scenario_id={scenario_id}")
            try:
                scenario_dir = self.scenarios_path / scenario_id
                if not scenario_dir.exists():
                    logger.warning(f"Scenario not found for deletion: scenario_id={scenario_id}, path={scenario_dir}")
                    raise HTTPException(status_code=404, detail="Scenario not found")
                
                # Delete scenario directory
                import shutil
                shutil.rmtree(scenario_dir)
                logger.info(f"Scenario deleted successfully: scenario_id={scenario_id}, path={scenario_dir}")
                
                return {"status": "success", "message": f"Scenario {scenario_id} deleted"}
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Scenario deletion failed: scenario_id={scenario_id}, error={e}", exc_info=True)
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/status")
        async def get_status():
            """Get current system status."""
            logger.debug("System status requested via API")
            status = {
                "timestamp": datetime.now().isoformat(),
                "connected_clients": len(self.connection_manager.active_connections),
                "latest_data_available": self.latest_data is not None
            }
            return status
        
        @self.app.websocket("/ws/stream")
        async def websocket_endpoint(websocket: WebSocket):
            """WebSocket endpoint for real-time data streaming."""
            await self.connection_manager.connect(websocket)
            try:
                # Send initial data if available
                if self.latest_data:
                    logger.debug(f"Sending initial data to new client: client={websocket.client}")
                    await websocket.send_json(self.latest_data)
                
                # Keep connection alive and handle client messages
                while True:
                    try:
                        # Wait for client messages (e.g., control commands)
                        data = await asyncio.wait_for(websocket.receive_text(), timeout=1.0)
                        message = json.loads(data)
                        
                        # Handle client commands
                        if message.get('type') == 'ping':
                            logger.debug(f"Ping received from client: client={websocket.client}")
                            await websocket.send_json({'type': 'pong'})
                        else:
                            logger.debug(f"Client message received: client={websocket.client}, type={message.get('type', 'unknown')}")
                        
                    except asyncio.TimeoutError:
                        # No message received, continue
                        continue
                    
            except WebSocketDisconnect:
                logger.debug(f"WebSocket disconnected normally: client={websocket.client}")
                self.connection_manager.disconnect(websocket)
            except Exception as e:
                logger.error(f"WebSocket error occurred: client={websocket.client}, error={e}", exc_info=True)
                self.connection_manager.disconnect(websocket)
    
    def _merge_config(self, base: Dict, update: Dict):
        """Recursively merge configuration updates."""
        logger.debug(f"Merging configuration: update_keys={list(update.keys())}")
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._merge_config(base[key], value)
            else:
                base[key] = value
                logger.debug(f"Configuration key updated: key={key}")
    
    async def stream_data(self, data: Dict[str, Any]):
        """Stream data to all connected WebSocket clients."""
        import time
        start_time = time.time()
        
        self.latest_data = data
        await self.connection_manager.broadcast(data)
        
        duration_ms = (time.time() - start_time) * 1000
        if duration_ms > 5.0:  # Log if streaming takes > 5ms
            logger.warning(f"Data streaming slow: duration={duration_ms:.2f}ms, clients={len(self.connection_manager.active_connections)}")
        else:
            logger.debug(f"Data streamed: duration={duration_ms:.2f}ms, clients={len(self.connection_manager.active_connections)}")
    
    def run(self, host: str = "0.0.0.0", port: int = 8080):
        """Run the FastAPI server."""
        import uvicorn
        logger.info(f"Starting visualization server: host={host}, port={port}")
        try:
            uvicorn.run(self.app, host=host, port=port, log_level="info")
            logger.info("Visualization server stopped normally")
        except Exception as e:
            logger.error(f"Visualization server error: error={e}", exc_info=True)
            raise


def create_server(config: Dict[str, Any]) -> VisualizationServer:
    """Factory function to create visualization server."""
    logger.debug("Creating visualization server instance")
    server = VisualizationServer(config)
    logger.debug("Visualization server instance created")
    return server
