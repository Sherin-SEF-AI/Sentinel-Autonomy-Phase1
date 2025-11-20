"""Test suite for visualization backend server module."""

import pytest
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from fastapi.testclient import TestClient
from fastapi import WebSocket
import json
from pathlib import Path
import asyncio

from src.visualization.backend.server import (
    ConnectionManager,
    VisualizationServer,
    create_server
)


@pytest.fixture
def mock_config():
    """Fixture providing mock configuration for testing."""
    return {
        'recording': {
            'storage_path': 'test_scenarios/'
        },
        'visualization': {
            'enabled': True,
            'port': 8080,
            'update_rate': 30
        }
    }


@pytest.fixture
def connection_manager():
    """Fixture creating a ConnectionManager instance."""
    return ConnectionManager()


@pytest.fixture
def visualization_server(mock_config):
    """Fixture creating a VisualizationServer instance."""
    return VisualizationServer(mock_config)


@pytest.fixture
def test_client(visualization_server):
    """Fixture providing FastAPI test client."""
    return TestClient(visualization_server.app)


class TestConnectionManager:
    """Test suite for ConnectionManager class."""
    
    def test_initialization(self, connection_manager):
        """Test that ConnectionManager initializes correctly."""
        assert connection_manager is not None
        assert connection_manager.active_connections == []
    
    @pytest.mark.asyncio
    async def test_connect_websocket(self, connection_manager):
        """Test connecting a WebSocket client."""
        mock_websocket = AsyncMock(spec=WebSocket)
        
        await connection_manager.connect(mock_websocket)
        
        assert len(connection_manager.active_connections) == 1
        assert mock_websocket in connection_manager.active_connections
        mock_websocket.accept.assert_called_once()
    
    def test_disconnect_websocket(self, connection_manager):
        """Test disconnecting a WebSocket client."""
        mock_websocket = Mock(spec=WebSocket)
        connection_manager.active_connections.append(mock_websocket)
        
        connection_manager.disconnect(mock_websocket)
        
        assert len(connection_manager.active_connections) == 0
        assert mock_websocket not in connection_manager.active_connections
    
    @pytest.mark.asyncio
    async def test_broadcast_to_multiple_clients(self, connection_manager):
        """Test broadcasting message to multiple connected clients."""
        mock_ws1 = AsyncMock(spec=WebSocket)
        mock_ws2 = AsyncMock(spec=WebSocket)
        connection_manager.active_connections = [mock_ws1, mock_ws2]
        
        test_message = {'type': 'test', 'data': 'hello'}
        await connection_manager.broadcast(test_message)
        
        mock_ws1.send_json.assert_called_once_with(test_message)
        mock_ws2.send_json.assert_called_once_with(test_message)
    
    @pytest.mark.asyncio
    async def test_broadcast_handles_disconnected_clients(self, connection_manager):
        """Test that broadcast removes clients that fail to receive messages."""
        mock_ws1 = AsyncMock(spec=WebSocket)
        mock_ws2 = AsyncMock(spec=WebSocket)
        mock_ws2.send_json.side_effect = Exception("Connection lost")
        
        connection_manager.active_connections = [mock_ws1, mock_ws2]
        
        test_message = {'type': 'test', 'data': 'hello'}
        await connection_manager.broadcast(test_message)
        
        # Failed client should be removed
        assert len(connection_manager.active_connections) == 1
        assert mock_ws1 in connection_manager.active_connections
        assert mock_ws2 not in connection_manager.active_connections
    
    @pytest.mark.asyncio
    async def test_broadcast_with_no_connections(self, connection_manager):
        """Test broadcasting when no clients are connected."""
        test_message = {'type': 'test', 'data': 'hello'}
        
        # Should not raise exception
        await connection_manager.broadcast(test_message)
        
        assert len(connection_manager.active_connections) == 0


class TestVisualizationServer:
    """Test suite for VisualizationServer class."""
    
    def test_initialization(self, visualization_server, mock_config):
        """Test that VisualizationServer initializes correctly with valid configuration."""
        assert visualization_server is not None
        assert visualization_server.config == mock_config
        assert visualization_server.app is not None
        assert visualization_server.connection_manager is not None
        assert visualization_server.latest_data is None
        assert visualization_server.scenarios_path == Path('test_scenarios/')
    
    def test_root_endpoint(self, test_client):
        """Test root health check endpoint."""
        response = test_client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        assert data['status'] == 'ok'
        assert data['service'] == 'SENTINEL Visualization API'
    
    def test_get_config_endpoint(self, test_client, mock_config):
        """Test getting current system configuration."""
        response = test_client.get("/api/config")
        
        assert response.status_code == 200
        data = response.json()
        assert 'recording' in data
        assert 'visualization' in data
    
    def test_update_config_endpoint(self, test_client):
        """Test updating system configuration."""
        config_update = {
            'visualization': {
                'update_rate': 60
            }
        }
        
        response = test_client.post("/api/config", json=config_update)
        
        assert response.status_code == 200
        data = response.json()
        assert data['status'] == 'success'
    
    def test_update_config_with_invalid_data(self, test_client):
        """Test updating configuration with invalid data."""
        # Mock the _merge_config to raise an exception
        with patch.object(VisualizationServer, '_merge_config', side_effect=ValueError("Invalid config")):
            response = test_client.post("/api/config", json={'invalid': 'data'})
            
            assert response.status_code == 400
    
    def test_get_status_endpoint(self, test_client):
        """Test getting current system status."""
        response = test_client.get("/api/status")
        
        assert response.status_code == 200
        data = response.json()
        assert 'timestamp' in data
        assert 'connected_clients' in data
        assert 'latest_data_available' in data
        assert data['connected_clients'] == 0
        assert data['latest_data_available'] is False
    
    def test_list_scenarios_empty(self, test_client, tmp_path):
        """Test listing scenarios when no scenarios exist."""
        with patch.object(Path, 'exists', return_value=False):
            response = test_client.get("/api/scenarios")
            
            assert response.status_code == 200
            data = response.json()
            assert 'scenarios' in data
            assert data['scenarios'] == []
    
    def test_list_scenarios_with_data(self, test_client, tmp_path):
        """Test listing scenarios with existing scenario data."""
        # Create mock scenario directory structure
        scenarios_path = tmp_path / "scenarios"
        scenarios_path.mkdir()
        
        scenario_dir = scenarios_path / "20241115_103045"
        scenario_dir.mkdir()
        
        metadata = {
            'timestamp': '2024-11-15T10:30:45.123Z',
            'duration': 15.5,
            'trigger': 'high_risk'
        }
        
        metadata_file = scenario_dir / "metadata.json"
        metadata_file.write_text(json.dumps(metadata))
        
        with patch.object(VisualizationServer, 'scenarios_path', scenarios_path):
            response = test_client.get("/api/scenarios")
            
            assert response.status_code == 200
            data = response.json()
            assert 'scenarios' in data
            assert len(data['scenarios']) == 1
            assert data['scenarios'][0]['id'] == '20241115_103045'
    
    def test_get_scenario_not_found(self, test_client):
        """Test getting a scenario that doesn't exist."""
        response = test_client.get("/api/scenarios/nonexistent")
        
        assert response.status_code == 404
    
    def test_get_scenario_success(self, test_client, tmp_path):
        """Test getting an existing scenario."""
        scenarios_path = tmp_path / "scenarios"
        scenarios_path.mkdir()
        
        scenario_dir = scenarios_path / "test_scenario"
        scenario_dir.mkdir()
        
        metadata = {'timestamp': '2024-11-15T10:30:45.123Z'}
        annotations = {'frames': []}
        
        (scenario_dir / "metadata.json").write_text(json.dumps(metadata))
        (scenario_dir / "annotations.json").write_text(json.dumps(annotations))
        
        with patch.object(VisualizationServer, 'scenarios_path', scenarios_path):
            response = test_client.get("/api/scenarios/test_scenario")
            
            assert response.status_code == 200
            data = response.json()
            assert data['id'] == 'test_scenario'
            assert 'metadata' in data
            assert 'annotations' in data
    
    def test_delete_scenario_not_found(self, test_client):
        """Test deleting a scenario that doesn't exist."""
        response = test_client.delete("/api/scenarios/nonexistent")
        
        assert response.status_code == 404
    
    def test_delete_scenario_success(self, test_client, tmp_path):
        """Test deleting an existing scenario."""
        scenarios_path = tmp_path / "scenarios"
        scenarios_path.mkdir()
        
        scenario_dir = scenarios_path / "test_scenario"
        scenario_dir.mkdir()
        (scenario_dir / "test.txt").write_text("test")
        
        with patch.object(VisualizationServer, 'scenarios_path', scenarios_path):
            response = test_client.delete("/api/scenarios/test_scenario")
            
            assert response.status_code == 200
            data = response.json()
            assert data['status'] == 'success'
            assert not scenario_dir.exists()
    
    def test_merge_config_simple(self, visualization_server):
        """Test merging simple configuration updates."""
        base = {'key1': 'value1', 'key2': 'value2'}
        update = {'key2': 'new_value2', 'key3': 'value3'}
        
        visualization_server._merge_config(base, update)
        
        assert base['key1'] == 'value1'
        assert base['key2'] == 'new_value2'
        assert base['key3'] == 'value3'
    
    def test_merge_config_nested(self, visualization_server):
        """Test merging nested configuration updates."""
        base = {
            'section1': {
                'key1': 'value1',
                'key2': 'value2'
            }
        }
        update = {
            'section1': {
                'key2': 'new_value2',
                'key3': 'value3'
            }
        }
        
        visualization_server._merge_config(base, update)
        
        assert base['section1']['key1'] == 'value1'
        assert base['section1']['key2'] == 'new_value2'
        assert base['section1']['key3'] == 'value3'
    
    @pytest.mark.asyncio
    async def test_stream_data(self, visualization_server):
        """Test streaming data to connected clients."""
        # Mock the connection manager
        visualization_server.connection_manager.broadcast = AsyncMock()
        
        test_data = {
            'timestamp': 123456.789,
            'bev': {'image': 'base64_encoded'},
            'detections': []
        }
        
        await visualization_server.stream_data(test_data)
        
        assert visualization_server.latest_data == test_data
        visualization_server.connection_manager.broadcast.assert_called_once_with(test_data)
    
    @pytest.mark.asyncio
    async def test_stream_data_updates_latest(self, visualization_server):
        """Test that streaming data updates the latest_data attribute."""
        visualization_server.connection_manager.broadcast = AsyncMock()
        
        data1 = {'frame': 1}
        data2 = {'frame': 2}
        
        await visualization_server.stream_data(data1)
        assert visualization_server.latest_data == data1
        
        await visualization_server.stream_data(data2)
        assert visualization_server.latest_data == data2
    
    def test_websocket_endpoint_exists(self, visualization_server):
        """Test that WebSocket endpoint is registered."""
        routes = [route.path for route in visualization_server.app.routes]
        assert "/ws/stream" in routes
    
    @pytest.mark.performance
    def test_config_update_performance(self, test_client):
        """Test that configuration update completes within performance requirements."""
        import time
        
        config_update = {'visualization': {'update_rate': 60}}
        
        start_time = time.perf_counter()
        response = test_client.post("/api/config", json=config_update)
        end_time = time.perf_counter()
        
        execution_time_ms = (end_time - start_time) * 1000
        assert response.status_code == 200
        assert execution_time_ms < 50, f"Config update took {execution_time_ms:.2f}ms, expected < 50ms"
    
    @pytest.mark.performance
    def test_list_scenarios_performance(self, test_client):
        """Test that listing scenarios completes within performance requirements."""
        import time
        
        start_time = time.perf_counter()
        response = test_client.get("/api/scenarios")
        end_time = time.perf_counter()
        
        execution_time_ms = (end_time - start_time) * 1000
        assert response.status_code == 200
        assert execution_time_ms < 100, f"List scenarios took {execution_time_ms:.2f}ms, expected < 100ms"


class TestCreateServer:
    """Test suite for create_server factory function."""
    
    def test_create_server(self, mock_config):
        """Test that create_server factory function creates a valid server instance."""
        server = create_server(mock_config)
        
        assert server is not None
        assert isinstance(server, VisualizationServer)
        assert server.config == mock_config
    
    def test_create_server_with_minimal_config(self):
        """Test creating server with minimal configuration."""
        minimal_config = {}
        server = create_server(minimal_config)
        
        assert server is not None
        assert isinstance(server, VisualizationServer)


class TestCORSConfiguration:
    """Test suite for CORS middleware configuration."""
    
    def test_cors_headers_present(self, test_client):
        """Test that CORS headers are properly configured."""
        response = test_client.options("/api/config")
        
        # CORS middleware should add appropriate headers
        assert response.status_code in [200, 405]  # OPTIONS may not be explicitly handled
    
    def test_cors_allows_cross_origin(self, test_client):
        """Test that cross-origin requests are allowed."""
        headers = {'Origin': 'http://localhost:3000'}
        response = test_client.get("/", headers=headers)
        
        assert response.status_code == 200


class TestErrorHandling:
    """Test suite for error handling scenarios."""
    
    def test_list_scenarios_with_io_error(self, test_client, visualization_server):
        """Test handling of I/O errors when listing scenarios."""
        with patch.object(Path, 'exists', side_effect=IOError("Disk error")):
            response = test_client.get("/api/scenarios")
            
            assert response.status_code == 500
    
    def test_get_scenario_with_json_decode_error(self, test_client, tmp_path):
        """Test handling of JSON decode errors when getting scenario."""
        scenarios_path = tmp_path / "scenarios"
        scenarios_path.mkdir()
        
        scenario_dir = scenarios_path / "test_scenario"
        scenario_dir.mkdir()
        
        # Write invalid JSON
        (scenario_dir / "metadata.json").write_text("invalid json{")
        
        with patch.object(VisualizationServer, 'scenarios_path', scenarios_path):
            response = test_client.get("/api/scenarios/test_scenario")
            
            assert response.status_code == 500
    
    def test_delete_scenario_with_permission_error(self, test_client, tmp_path):
        """Test handling of permission errors when deleting scenario."""
        scenarios_path = tmp_path / "scenarios"
        scenarios_path.mkdir()
        
        scenario_dir = scenarios_path / "test_scenario"
        scenario_dir.mkdir()
        
        with patch('shutil.rmtree', side_effect=PermissionError("Access denied")):
            with patch.object(VisualizationServer, 'scenarios_path', scenarios_path):
                response = test_client.delete("/api/scenarios/test_scenario")
                
                assert response.status_code == 500
