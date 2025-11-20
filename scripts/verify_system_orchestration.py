#!/usr/bin/env python3
"""Verification script for SENTINEL system orchestration."""

import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from core.config import ConfigManager
from core.logging import LoggerSetup


def test_system_initialization():
    """Test that SentinelSystem can be initialized."""
    print("=" * 60)
    print("TEST: System Initialization")
    print("=" * 60)
    
    try:
        # Load configuration
        config = ConfigManager('configs/default.yaml')
        
        # Set up logging
        logger = LoggerSetup.setup(log_level='INFO')
        
        # Import SentinelSystem
        from main import SentinelSystem
        
        # Initialize system
        print("Initializing SENTINEL system...")
        system = SentinelSystem(config)
        
        # Verify all modules are initialized
        assert system.camera_manager is not None, "CameraManager not initialized"
        assert system.bev_generator is not None, "BEVGenerator not initialized"
        assert system.segmentor is not None, "SemanticSegmentor not initialized"
        assert system.detector is not None, "ObjectDetector not initialized"
        assert system.dms is not None, "DriverMonitor not initialized"
        assert system.intelligence is not None, "ContextualIntelligence not initialized"
        assert system.alert_system is not None, "AlertSystem not initialized"
        assert system.recorder is not None, "ScenarioRecorder not initialized"
        
        print("✓ All modules initialized successfully")
        
        # Verify performance monitoring structures
        assert hasattr(system, 'module_latencies'), "Performance monitoring not set up"
        assert hasattr(system, 'cpu_usage_history'), "CPU monitoring not set up"
        assert hasattr(system, 'memory_usage_history'), "Memory monitoring not set up"
        
        print("✓ Performance monitoring structures initialized")
        
        # Verify state persistence methods exist
        assert hasattr(system, '_save_system_state'), "State save method missing"
        assert hasattr(system, '_restore_system_state'), "State restore method missing"
        
        print("✓ State persistence methods available")
        
        print("\n✓ System initialization test PASSED")
        return True
        
    except Exception as e:
        print(f"\n✗ System initialization test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_graceful_shutdown():
    """Test graceful shutdown functionality."""
    print("\n" + "=" * 60)
    print("TEST: Graceful Shutdown")
    print("=" * 60)
    
    try:
        from main import SentinelSystem
        config = ConfigManager('configs/default.yaml')
        
        # Initialize system
        system = SentinelSystem(config)
        
        # Test shutdown without starting
        print("Testing shutdown without start...")
        system.stop()
        print("✓ Shutdown without start successful")
        
        # Test state save
        print("Testing state save...")
        system.frame_count = 100
        system.start_time = time.time()
        system._save_system_state()
        
        # Verify state file was created
        state_file = Path("state/system_state.pkl")
        assert state_file.exists(), "State file not created"
        print("✓ State file created successfully")
        
        # Test state restore
        print("Testing state restore...")
        system2 = SentinelSystem(config)
        # State should be restored in __init__
        print("✓ State restore completed")
        
        print("\n✓ Graceful shutdown test PASSED")
        return True
        
    except Exception as e:
        print(f"\n✗ Graceful shutdown test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_performance_monitoring():
    """Test performance monitoring functionality."""
    print("\n" + "=" * 60)
    print("TEST: Performance Monitoring")
    print("=" * 60)
    
    try:
        from main import SentinelSystem
        config = ConfigManager('configs/default.yaml')
        
        # Initialize system
        system = SentinelSystem(config)
        
        # Simulate some latency data
        system.module_latencies['camera'] = [0.005, 0.006, 0.005]
        system.module_latencies['bev'] = [0.015, 0.014, 0.016]
        system.module_latencies['detection'] = [0.020, 0.019, 0.021]
        
        # Test performance logging
        print("Testing performance metrics logging...")
        system.frame_count = 30
        system.start_time = time.time() - 1.0  # 1 second ago
        system._log_performance_metrics()
        print("✓ Performance metrics logged successfully")
        
        # Test final statistics
        print("Testing final statistics...")
        system._log_final_statistics()
        print("✓ Final statistics logged successfully")
        
        print("\n✓ Performance monitoring test PASSED")
        return True
        
    except Exception as e:
        print(f"\n✗ Performance monitoring test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all verification tests."""
    print("\n" + "=" * 60)
    print("SENTINEL SYSTEM ORCHESTRATION VERIFICATION")
    print("=" * 60 + "\n")
    
    results = []
    
    # Run tests
    results.append(("System Initialization", test_system_initialization()))
    results.append(("Graceful Shutdown", test_graceful_shutdown()))
    results.append(("Performance Monitoring", test_performance_monitoring()))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{test_name}: {status}")
    
    all_passed = all(result[1] for result in results)
    
    print("=" * 60)
    if all_passed:
        print("✓ ALL TESTS PASSED")
        return 0
    else:
        print("✗ SOME TESTS FAILED")
        return 1


if __name__ == '__main__':
    sys.exit(main())
