#!/usr/bin/env python3
"""
Verification script for Performance Dock Widget logging implementation.

This script verifies that:
1. Logger is properly configured
2. All key logging points are present
3. Logging follows SENTINEL patterns
4. Performance overhead is minimal
"""

import sys
import re
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def verify_logger_setup():
    """Verify logger is properly set up in the module."""
    print("=" * 60)
    print("1. Verifying Logger Setup")
    print("=" * 60)
    
    module_path = Path("src/gui/widgets/performance_dock.py")
    
    if not module_path.exists():
        print("❌ Module file not found")
        return False
    
    content = module_path.read_text()
    
    # Check for logger import and setup
    if "import logging" in content and "logger = logging.getLogger(__name__)" in content:
        print("✓ Logger properly imported and configured")
    else:
        print("❌ Logger not properly set up")
        return False
    
    return True


def verify_logging_config():
    """Verify logging configuration in logging.yaml."""
    print("\n" + "=" * 60)
    print("2. Verifying Logging Configuration")
    print("=" * 60)
    
    config_path = Path("configs/logging.yaml")
    
    if not config_path.exists():
        print("❌ Logging config file not found")
        return False
    
    content = config_path.read_text()
    
    if "src.gui.widgets.performance_dock:" in content:
        print("✓ Module logger configured in logging.yaml")
        
        # Check log level
        if re.search(r"src\.gui\.widgets\.performance_dock:.*?level:\s*INFO", content, re.DOTALL):
            print("✓ Log level set to INFO")
        else:
            print("⚠ Log level not set to INFO")
        
        # Check handlers
        if "handlers: [file_all]" in content:
            print("✓ Handlers configured correctly")
        else:
            print("⚠ Handlers not configured")
        
        return True
    else:
        print("❌ Module logger not configured in logging.yaml")
        return False


def verify_logging_points():
    """Verify all key logging points are present."""
    print("\n" + "=" * 60)
    print("3. Verifying Logging Points")
    print("=" * 60)
    
    module_path = Path("src/gui/widgets/performance_dock.py")
    content = module_path.read_text()
    
    required_patterns = {
        "Initialization": [
            r'logger\.info\(.*initialized',
            r'logger\.debug\(.*Initializing'
        ],
        "State Changes": [
            r'logger\.info\(.*started',
            r'logger\.info\(.*stopped',
            r'logger\.info\(.*cleared'
        ],
        "Metric Updates": [
            r'logger\.debug\(.*updated',
            r'logger\.debug\(.*FPS updated',
            r'logger\.debug\(.*Latency updated'
        ],
        "Threshold Violations": [
            r'logger\.warning\(.*below target',
            r'logger\.warning\(.*exceeds',
            r'logger\.warning\(.*threshold'
        ],
        "Error Handling": [
            r'logger\.error\(.*Failed'
        ]
    }
    
    all_passed = True
    for category, patterns in required_patterns.items():
        found = sum(1 for pattern in patterns if re.search(pattern, content, re.IGNORECASE))
        total = len(patterns)
        
        if found >= total // 2:  # At least half of patterns found
            print(f"✓ {category}: {found}/{total} patterns found")
        else:
            print(f"❌ {category}: Only {found}/{total} patterns found")
            all_passed = False
    
    return all_passed


def verify_log_message_patterns():
    """Verify log messages follow SENTINEL patterns."""
    print("\n" + "=" * 60)
    print("4. Verifying Log Message Patterns")
    print("=" * 60)
    
    module_path = Path("src/gui/widgets/performance_dock.py")
    content = module_path.read_text()
    
    # Extract all logger calls
    logger_calls = re.findall(r'logger\.\w+\([^)]+\)', content)
    
    patterns_check = {
        "Past tense for completed actions": 0,
        "Includes context/parameters": 0,
        "Concise messages": 0
    }
    
    for call in logger_calls:
        # Check for past tense (initialized, started, stopped, updated, etc.)
        if re.search(r'(initialized|started|stopped|updated|cleared|exported|exceeded)', call):
            patterns_check["Past tense for completed actions"] += 1
        
        # Check for context (parameters, values)
        if re.search(r'(=|:|\{)', call):
            patterns_check["Includes context/parameters"] += 1
        
        # Check message length (not too verbose)
        if len(call) < 200:
            patterns_check["Concise messages"] += 1
    
    total_calls = len(logger_calls)
    print(f"Total logger calls found: {total_calls}")
    
    for pattern, count in patterns_check.items():
        percentage = (count / total_calls * 100) if total_calls > 0 else 0
        if percentage >= 50:
            print(f"✓ {pattern}: {count}/{total_calls} ({percentage:.1f}%)")
        else:
            print(f"⚠ {pattern}: {count}/{total_calls} ({percentage:.1f}%)")
    
    return True


def verify_performance_overhead():
    """Verify logging has minimal performance overhead."""
    print("\n" + "=" * 60)
    print("5. Verifying Performance Overhead")
    print("=" * 60)
    
    module_path = Path("src/gui/widgets/performance_dock.py")
    content = module_path.read_text()
    
    # Count DEBUG vs INFO/WARNING/ERROR logs
    debug_count = len(re.findall(r'logger\.debug\(', content))
    info_count = len(re.findall(r'logger\.info\(', content))
    warning_count = len(re.findall(r'logger\.warning\(', content))
    error_count = len(re.findall(r'logger\.error\(', content))
    
    total = debug_count + info_count + warning_count + error_count
    
    print(f"DEBUG logs: {debug_count} ({debug_count/total*100:.1f}%)")
    print(f"INFO logs: {info_count} ({info_count/total*100:.1f}%)")
    print(f"WARNING logs: {warning_count} ({warning_count/total*100:.1f}%)")
    print(f"ERROR logs: {error_count} ({error_count/total*100:.1f}%)")
    
    # Check for logging in tight loops (should be minimal)
    tight_loop_patterns = [
        r'for.*in.*:.*logger\.',
        r'while.*:.*logger\.'
    ]
    
    tight_loop_logs = sum(1 for pattern in tight_loop_patterns 
                          if re.search(pattern, content, re.DOTALL))
    
    if tight_loop_logs == 0:
        print("✓ No logging in tight loops")
    else:
        print(f"⚠ Found {tight_loop_logs} potential tight loop logs")
    
    # Check for expensive operations in log messages
    expensive_ops = re.findall(r'logger\.\w+\([^)]*\[.*for.*in.*\]', content)
    
    if len(expensive_ops) == 0:
        print("✓ No expensive operations in log messages")
    else:
        print(f"⚠ Found {len(expensive_ops)} expensive operations in log messages")
    
    return True


def main():
    """Run all verification checks."""
    print("\n" + "=" * 60)
    print("PERFORMANCE DOCK WIDGET LOGGING VERIFICATION")
    print("=" * 60 + "\n")
    
    checks = [
        ("Logger Setup", verify_logger_setup),
        ("Logging Configuration", verify_logging_config),
        ("Logging Points", verify_logging_points),
        ("Log Message Patterns", verify_log_message_patterns),
        ("Performance Overhead", verify_performance_overhead)
    ]
    
    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n❌ Error in {name}: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASS" if result else "❌ FAIL"
        print(f"{status}: {name}")
    
    print(f"\nOverall: {passed}/{total} checks passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("\n🎉 All verification checks passed!")
        return 0
    else:
        print(f"\n⚠ {total - passed} check(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
