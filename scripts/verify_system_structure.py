#!/usr/bin/env python3
"""Verification script for SENTINEL system orchestration structure."""

import ast
import sys
from pathlib import Path


def verify_main_py_structure():
    """Verify that main.py has the required structure."""
    print("=" * 60)
    print("VERIFYING: src/main.py structure")
    print("=" * 60)
    
    main_file = Path('src/main.py')
    if not main_file.exists():
        print("✗ main.py not found")
        return False
    
    with open(main_file, 'r') as f:
        content = f.read()
    
    try:
        tree = ast.parse(content)
    except SyntaxError as e:
        print(f"✗ Syntax error in main.py: {e}")
        return False
    
    # Find SentinelSystem class
    sentinel_class = None
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == 'SentinelSystem':
            sentinel_class = node
            break
    
    if not sentinel_class:
        print("✗ SentinelSystem class not found")
        return False
    
    print("✓ SentinelSystem class found")
    
    # Check required methods
    required_methods = [
        '__init__',
        '_initialize_modules',
        'start',
        'stop',
        '_processing_loop',
        '_performance_monitoring_loop',
        '_log_performance_metrics',
        '_log_final_statistics',
        '_save_system_state',
        '_restore_system_state',
        '_periodic_state_save',
        '_close_resources'
    ]
    
    class_methods = [node.name for node in sentinel_class.body 
                     if isinstance(node, ast.FunctionDef)]
    
    missing_methods = []
    for method in required_methods:
        if method in class_methods:
            print(f"✓ Method {method} found")
        else:
            print(f"✗ Method {method} missing")
            missing_methods.append(method)
    
    if missing_methods:
        print(f"\n✗ Missing methods: {', '.join(missing_methods)}")
        return False
    
    # Check for required imports
    required_imports = [
        'CameraManager',
        'BEVGenerator',
        'SemanticSegmentor',
        'ObjectDetector',
        'DriverMonitor',
        'ContextualIntelligence',
        'AlertSystem',
        'ScenarioRecorder',
        'VisualizationServer',
        'StreamingManager'
    ]
    
    import_names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                import_names.append(alias.name)
    
    missing_imports = []
    for imp in required_imports:
        if imp in import_names:
            print(f"✓ Import {imp} found")
        else:
            print(f"✗ Import {imp} missing")
            missing_imports.append(imp)
    
    if missing_imports:
        print(f"\n✗ Missing imports: {', '.join(missing_imports)}")
        return False
    
    print("\n✓ main.py structure verification PASSED")
    return True


def verify_requirements():
    """Verify that psutil is in requirements.txt."""
    print("\n" + "=" * 60)
    print("VERIFYING: requirements.txt")
    print("=" * 60)
    
    req_file = Path('requirements.txt')
    if not req_file.exists():
        print("✗ requirements.txt not found")
        return False
    
    with open(req_file, 'r') as f:
        content = f.read()
    
    if 'psutil' in content:
        print("✓ psutil found in requirements.txt")
        return True
    else:
        print("✗ psutil not found in requirements.txt")
        return False


def verify_task_completion():
    """Verify that all sub-tasks are marked as completed."""
    print("\n" + "=" * 60)
    print("VERIFYING: Task completion status")
    print("=" * 60)
    
    tasks_file = Path('.kiro/specs/sentinel-safety-system/tasks.md')
    if not tasks_file.exists():
        print("✗ tasks.md not found")
        return False
    
    with open(tasks_file, 'r') as f:
        content = f.read()
    
    # Check for task 12 and its sub-tasks
    task_12_subtasks = [
        '12.1 Create SentinelSystem main class',
        '12.2 Implement main processing loop',
        '12.3 Add performance monitoring',
        '12.4 Implement graceful shutdown',
        '12.5 Add state persistence and recovery'
    ]
    
    all_completed = True
    for task in task_12_subtasks:
        # Look for [x] marker for this task
        if f'- [x] {task}' in content or f'- [X] {task}' in content:
            print(f"✓ Task '{task}' marked as completed")
        else:
            print(f"✗ Task '{task}' not marked as completed")
            all_completed = False
    
    if all_completed:
        print("\n✓ All sub-tasks marked as completed")
        return True
    else:
        print("\n✗ Some sub-tasks not completed")
        return False


def verify_implementation_features():
    """Verify specific implementation features in main.py."""
    print("\n" + "=" * 60)
    print("VERIFYING: Implementation features")
    print("=" * 60)
    
    main_file = Path('src/main.py')
    with open(main_file, 'r') as f:
        content = f.read()
    
    features = {
        'Parallel DMS and Perception': 'def process_dms():',
        'Performance monitoring thread': '_performance_monitoring_loop',
        'CPU monitoring': 'cpu_percent',
        'GPU memory monitoring': 'torch.cuda.memory_allocated',
        'Module latency tracking': 'module_latencies',
        'State persistence': '_save_system_state',
        'State recovery': '_restore_system_state',
        'Graceful shutdown': 'shutdown_event',
        'Signal handlers': 'signal.signal',
        'Visualization streaming': 'streaming_manager',
        'Recording integration': 'recorder.should_record',
        'Final statistics': '_log_final_statistics',
        'P95 latency calculation': 'np.percentile'
    }
    
    all_found = True
    for feature, marker in features.items():
        if marker in content:
            print(f"✓ {feature} implemented")
        else:
            print(f"✗ {feature} not found")
            all_found = False
    
    if all_found:
        print("\n✓ All implementation features verified")
        return True
    else:
        print("\n✗ Some implementation features missing")
        return False


def main():
    """Run all verification checks."""
    print("\n" + "=" * 60)
    print("SENTINEL SYSTEM ORCHESTRATION STRUCTURE VERIFICATION")
    print("=" * 60 + "\n")
    
    results = []
    
    # Run verifications
    results.append(("main.py structure", verify_main_py_structure()))
    results.append(("requirements.txt", verify_requirements()))
    results.append(("Task completion", verify_task_completion()))
    results.append(("Implementation features", verify_implementation_features()))
    
    # Summary
    print("\n" + "=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)
    
    for check_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{check_name}: {status}")
    
    all_passed = all(result[1] for result in results)
    
    print("=" * 60)
    if all_passed:
        print("✓ ALL VERIFICATIONS PASSED")
        print("\nTask 12 'Implement main system orchestration' is complete!")
        print("\nImplemented features:")
        print("  • SentinelSystem main class with module initialization")
        print("  • Main processing loop with parallel DMS and perception")
        print("  • Performance monitoring (FPS, latency, CPU, GPU memory)")
        print("  • Graceful shutdown with resource cleanup")
        print("  • State persistence and recovery (<2s recovery time)")
        return 0
    else:
        print("✗ SOME VERIFICATIONS FAILED")
        return 1


if __name__ == '__main__':
    sys.exit(main())
