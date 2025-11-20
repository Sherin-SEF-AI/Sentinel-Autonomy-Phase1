"""Test suite for perception.detection module initialization and exports."""

import pytest
from unittest.mock import Mock, patch, MagicMock
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))


class TestDetectionModuleImports:
    """Test suite for detection module imports and exports."""
    
    def test_detector_2d_import(self):
        """Test that Detector2D can be imported from the module."""
        from src.perception.detection import Detector2D
        assert Detector2D is not None
        assert hasattr(Detector2D, '__init__')
    
    def test_estimator_3d_import(self):
        """Test that Estimator3D can be imported from the module."""
        from src.perception.detection import Estimator3D
        assert Estimator3D is not None
        assert hasattr(Estimator3D, '__init__')
    
    def test_multi_view_fusion_import(self):
        """Test that MultiViewFusion can be imported from the module."""
        from src.perception.detection import MultiViewFusion
        assert MultiViewFusion is not None
        assert hasattr(MultiViewFusion, '__init__')
    
    def test_object_tracker_import(self):
        """Test that ObjectTracker can be imported from the module."""
        from src.perception.detection import ObjectTracker
        assert ObjectTracker is not None
        assert hasattr(ObjectTracker, '__init__')
    
    def test_track_import(self):
        """Test that Track can be imported from the module."""
        from src.perception.detection import Track
        assert Track is not None
        assert hasattr(Track, '__init__')
    
    def test_object_detector_import(self):
        """Test that ObjectDetector can be imported from the module."""
        from src.perception.detection import ObjectDetector
        assert ObjectDetector is not None
        assert hasattr(ObjectDetector, '__init__')
    
    def test_all_exports(self):
        """Test that __all__ contains all expected exports."""
        from src.perception import detection
        
        expected_exports = [
            'Detector2D',
            'Estimator3D',
            'MultiViewFusion',
            'ObjectDetector',
            'ObjectTracker',
            'Track',
        ]
        
        assert hasattr(detection, '__all__')
        assert set(detection.__all__) == set(expected_exports)
    
    def test_all_exports_are_accessible(self):
        """Test that all items in __all__ are actually accessible."""
        from src.perception import detection
        
        for export_name in detection.__all__:
            assert hasattr(detection, export_name), f"{export_name} not accessible"
            obj = getattr(detection, export_name)
            assert obj is not None, f"{export_name} is None"


class TestDetectionModuleStructure:
    """Test suite for detection module structure and organization."""
    
    def test_module_docstring(self):
        """Test that the module has a proper docstring."""
        from src.perception import detection
        assert detection.__doc__ is not None
        assert len(detection.__doc__) > 0
        assert "detection" in detection.__doc__.lower()
    
    def test_submodules_exist(self):
        """Test that all expected submodules exist."""
        import importlib
        
        submodules = [
            'src.perception.detection.detector_2d',
            'src.perception.detection.estimator_3d',
            'src.perception.detection.fusion',
            'src.perception.detection.tracker',
            'src.perception.detection.detector',
        ]
        
        for module_name in submodules:
            try:
                module = importlib.import_module(module_name)
                assert module is not None
            except ImportError as e:
                pytest.fail(f"Failed to import {module_name}: {e}")
    
    def test_no_circular_imports(self):
        """Test that importing the module doesn't cause circular import errors."""
        try:
            from src.perception import detection
            # Reimport to ensure no circular dependencies
            import importlib
            importlib.reload(detection)
        except ImportError as e:
            pytest.fail(f"Circular import detected: {e}")


class TestDetectionClassInterfaces:
    """Test suite for detection class interfaces and methods."""
    
    def test_detector_2d_has_detect_method(self):
        """Test that Detector2D has the required detect method."""
        from src.perception.detection import Detector2D
        assert hasattr(Detector2D, 'detect')
        assert callable(getattr(Detector2D, 'detect'))
    
    def test_estimator_3d_has_estimate_method(self):
        """Test that Estimator3D has the required estimate method."""
        from src.perception.detection import Estimator3D
        assert hasattr(Estimator3D, 'estimate')
        assert callable(getattr(Estimator3D, 'estimate'))
    
    def test_multi_view_fusion_has_fuse_method(self):
        """Test that MultiViewFusion has the required fuse method."""
        from src.perception.detection import MultiViewFusion
        assert hasattr(MultiViewFusion, 'fuse')
        assert callable(getattr(MultiViewFusion, 'fuse'))
    
    def test_object_tracker_has_update_method(self):
        """Test that ObjectTracker has the required update method."""
        from src.perception.detection import ObjectTracker
        assert hasattr(ObjectTracker, 'update')
        assert callable(getattr(ObjectTracker, 'update'))
    
    def test_object_detector_implements_interface(self):
        """Test that ObjectDetector implements IObjectDetector interface."""
        from src.perception.detection import ObjectDetector
        from src.core.interfaces import IObjectDetector
        
        # Check that ObjectDetector is a subclass of IObjectDetector
        assert issubclass(ObjectDetector, IObjectDetector)
        
        # Check that it has the required detect method
        assert hasattr(ObjectDetector, 'detect')
        assert callable(getattr(ObjectDetector, 'detect'))


class TestDetectionDataStructures:
    """Test suite for detection data structures compatibility."""
    
    def test_detection_2d_structure(self):
        """Test that Detection2D structure is compatible with module."""
        from src.core.data_structures import Detection2D
        
        # Create a sample Detection2D
        det = Detection2D(
            bbox=(10.0, 20.0, 100.0, 200.0),
            class_name='vehicle',
            confidence=0.95,
            camera_id=1
        )
        
        assert det.bbox == (10.0, 20.0, 100.0, 200.0)
        assert det.class_name == 'vehicle'
        assert det.confidence == 0.95
        assert det.camera_id == 1
    
    def test_detection_3d_structure(self):
        """Test that Detection3D structure is compatible with module."""
        from src.core.data_structures import Detection3D
        
        # Create a sample Detection3D
        det = Detection3D(
            bbox_3d=(5.0, 2.0, 0.0, 1.8, 1.5, 4.5, 0.0),
            class_name='vehicle',
            confidence=0.90,
            velocity=(10.0, 0.0, 0.0),
            track_id=42
        )
        
        assert len(det.bbox_3d) == 7
        assert det.class_name == 'vehicle'
        assert det.confidence == 0.90
        assert det.velocity == (10.0, 0.0, 0.0)
        assert det.track_id == 42


class TestDetectionModuleIntegration:
    """Test suite for detection module integration."""
    
    def test_full_pipeline_imports(self):
        """Test that all components can be imported and instantiated together."""
        from src.perception.detection import (
            Detector2D, Estimator3D, MultiViewFusion, 
            ObjectTracker, ObjectDetector
        )
        
        # Create mock configurations
        fusion_config = {
            'iou_threshold_3d': 0.3,
            'confidence_weighting': True
        }
        
        tracking_config = {
            'max_age': 30,
            'min_hits': 3,
            'iou_threshold': 0.3
        }
        
        calibration_data = {
            1: {
                'intrinsics': {
                    'fx': 800.0, 'fy': 800.0,
                    'cx': 640.0, 'cy': 360.0
                },
                'extrinsics': {
                    'translation': [2.0, 0.5, 1.0],
                    'rotation': [0.0, 0.0, -0.785]
                }
            }
        }
        
        # Test instantiation of components that don't require external dependencies
        try:
            estimator_3d = Estimator3D(calibration_data)
            fusion = MultiViewFusion(fusion_config)
            tracker = ObjectTracker(tracking_config)
            
            assert estimator_3d is not None
            assert fusion is not None
            assert tracker is not None
            
        except Exception as e:
            pytest.fail(f"Failed to instantiate detection components: {e}")
    
    def test_module_version_compatibility(self):
        """Test that the module is compatible with current Python version."""
        import sys
        assert sys.version_info >= (3, 10), "Python 3.10+ required"


class TestDetectionModuleDocumentation:
    """Test suite for detection module documentation."""
    
    def test_all_classes_have_docstrings(self):
        """Test that all exported classes have docstrings."""
        from src.perception import detection
        
        for export_name in detection.__all__:
            obj = getattr(detection, export_name)
            if isinstance(obj, type):  # It's a class
                assert obj.__doc__ is not None, f"{export_name} missing docstring"
                assert len(obj.__doc__) > 0, f"{export_name} has empty docstring"
    
    def test_detector_2d_docstring_content(self):
        """Test that Detector2D has meaningful docstring."""
        from src.perception.detection import Detector2D
        doc = Detector2D.__doc__
        assert 'detect' in doc.lower() or 'yolo' in doc.lower()
    
    def test_object_detector_docstring_content(self):
        """Test that ObjectDetector has meaningful docstring."""
        from src.perception.detection import ObjectDetector
        doc = ObjectDetector.__doc__
        assert 'detect' in doc.lower() or 'track' in doc.lower() or 'fusion' in doc.lower()


@pytest.mark.performance
class TestDetectionModulePerformance:
    """Test suite for detection module import performance."""
    
    def test_import_time(self):
        """Test that module imports within reasonable time (< 100ms)."""
        import time
        import importlib
        
        # Clear module from cache if present
        if 'src.perception.detection' in sys.modules:
            del sys.modules['src.perception.detection']
        
        start_time = time.perf_counter()
        import src.perception.detection
        end_time = time.perf_counter()
        
        import_time_ms = (end_time - start_time) * 1000
        assert import_time_ms < 100, f"Import took {import_time_ms:.2f}ms, expected < 100ms"
    
    def test_no_heavy_computation_on_import(self):
        """Test that importing the module doesn't trigger heavy computation."""
        import time
        import importlib
        
        # Import multiple times and check consistency
        times = []
        for _ in range(3):
            if 'src.perception.detection' in sys.modules:
                del sys.modules['src.perception.detection']
            
            start = time.perf_counter()
            import src.perception.detection
            end = time.perf_counter()
            times.append((end - start) * 1000)
        
        # Import times should be relatively consistent (no lazy loading issues)
        avg_time = sum(times) / len(times)
        for t in times:
            assert abs(t - avg_time) < 50, "Import time inconsistent, possible lazy loading"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
