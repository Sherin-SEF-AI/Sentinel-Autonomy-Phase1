"""Simple verification script for advanced trajectory logging."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_module_import():
    """Test that the module can be imported."""
    print("\n" + "="*60)
    print("Testing Module Import")
    print("="*60)
    
    try:
        from src.intelligence import advanced_trajectory
        print("✓ Module imported successfully")
        
        # Check for logger
        if hasattr(advanced_trajectory, 'logger'):
            print("✓ Logger instance found")
        else:
            print("✗ Logger instance not found")
            return False
        
        # Check TORCH_AVAILABLE flag
        print(f"  PyTorch available: {advanced_trajectory.TORCH_AVAILABLE}")
        
        return True
        
    except Exception as e:
        print(f"✗ Module import failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_logging_configuration():
    """Test logging configuration."""
    print("\n" + "="*60)
    print("Testing Logging Configuration")
    print("="*60)
    
    try:
        import yaml
        
        config_file = Path("configs/logging.yaml")
        if not config_file.exists():
            print(f"✗ Logging config not found: {config_file}")
            return False
        
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        # Check for advanced trajectory logger
        loggers = config.get('loggers', {})
        
        checks = [
            ('src.intelligence.advanced_trajectory', 'Advanced trajectory logger'),
            ('src.intelligence.trajectory_performance', 'Trajectory performance logger'),
            ('src.intelligence.trajectory_visualization', 'Trajectory visualization logger'),
            ('src.intelligence.advanced_risk', 'Advanced risk logger'),
        ]
        
        all_found = True
        for logger_name, description in checks:
            if logger_name in loggers:
                level = loggers[logger_name].get('level', 'N/A')
                print(f"✓ {description}: level={level}")
            else:
                print(f"✗ {description}: not found")
                all_found = False
        
        return all_found
        
    except Exception as e:
        print(f"✗ Logging configuration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_code_structure():
    """Test code structure and logging statements."""
    print("\n" + "="*60)
    print("Testing Code Structure")
    print("="*60)
    
    try:
        file_path = Path("src/intelligence/advanced_trajectory.py")
        
        if not file_path.exists():
            print(f"✗ File not found: {file_path}")
            return False
        
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Check for key logging patterns
        checks = [
            ('logger = logging.getLogger(__name__)', 'Logger initialization'),
            ('logger.info', 'INFO level logging'),
            ('logger.debug', 'DEBUG level logging'),
            ('logger.error', 'ERROR level logging'),
            ('logger.warning', 'WARNING level logging'),
            ('duration=', 'Performance timing'),
            ('exc_info=True', 'Exception logging'),
        ]
        
        all_found = True
        for pattern, description in checks:
            if pattern in content:
                count = content.count(pattern)
                print(f"✓ {description}: found {count} occurrence(s)")
            else:
                print(f"✗ {description}: not found")
                all_found = False
        
        return all_found
        
    except Exception as e:
        print(f"✗ Code structure test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all verification tests."""
    print("="*60)
    print("Advanced Trajectory Logging Verification (Simple)")
    print("="*60)
    
    # Run tests
    results = []
    
    results.append(("Module Import", test_module_import()))
    results.append(("Logging Configuration", test_logging_configuration()))
    results.append(("Code Structure", test_code_structure()))
    
    # Summary
    print("\n" + "="*60)
    print("Verification Summary")
    print("="*60)
    
    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    all_passed = all(passed for _, passed in results)
    
    print("\n" + "="*60)
    if all_passed:
        print("✓ All verification tests passed!")
    else:
        print("✗ Some verification tests failed")
    print("="*60)
    
    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())
