"""
Basic verification for visualization module structure.

Tests module structure without requiring FastAPI installation.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_module_structure():
    """Test that all required files exist."""
    print("Testing module structure...")
    
    base_path = Path(__file__).parent.parent / "src" / "visualization"
    
    required_files = [
        "README.md",
        "__init__.py",
        "backend/__init__.py",
        "backend/server.py",
        "backend/data_serializer.py",
        "backend/streaming.py",
        "backend/README.md",
        "frontend/__init__.py",
        "frontend/index.html",
        "frontend/app.js",
        "frontend/playback.html",
        "frontend/playback.js",
        "frontend/README.md",
    ]
    
    for file_path in required_files:
        full_path = base_path / file_path
        assert full_path.exists(), f"Missing file: {file_path}"
        print(f"  ✓ {file_path}")
    
    print("  ✓ All required files present")


def test_example_files():
    """Test that example files exist."""
    print("Testing example files...")
    
    base_path = Path(__file__).parent.parent / "examples"
    
    required_files = [
        "visualization_backend_example.py",
        "visualization_streaming_example.py",
        "visualization_complete_example.py",
    ]
    
    for file_path in required_files:
        full_path = base_path / file_path
        assert full_path.exists(), f"Missing file: {file_path}"
        print(f"  ✓ {file_path}")
    
    print("  ✓ All example files present")


def test_test_files():
    """Test that test files exist."""
    print("Testing test files...")
    
    base_path = Path(__file__).parent.parent / "tests"
    
    required_files = [
        "test_visualization_backend.py",
    ]
    
    for file_path in required_files:
        full_path = base_path / file_path
        assert full_path.exists(), f"Missing file: {file_path}"
        print(f"  ✓ {file_path}")
    
    print("  ✓ All test files present")


def test_file_syntax():
    """Test that Python files have valid syntax."""
    print("Testing Python file syntax...")
    
    base_path = Path(__file__).parent.parent
    
    python_files = [
        "src/visualization/__init__.py",
        "src/visualization/backend/__init__.py",
        "src/visualization/backend/server.py",
        "src/visualization/backend/data_serializer.py",
        "src/visualization/backend/streaming.py",
        "examples/visualization_backend_example.py",
        "examples/visualization_streaming_example.py",
        "examples/visualization_complete_example.py",
    ]
    
    for file_path in python_files:
        full_path = base_path / file_path
        with open(full_path, 'r') as f:
            code = f.read()
            try:
                compile(code, file_path, 'exec')
                print(f"  ✓ {file_path}")
            except SyntaxError as e:
                raise AssertionError(f"Syntax error in {file_path}: {e}")
    
    print("  ✓ All Python files have valid syntax")


def test_documentation():
    """Test that documentation files exist and are not empty."""
    print("Testing documentation...")
    
    base_path = Path(__file__).parent.parent
    
    doc_files = [
        "src/visualization/README.md",
        "src/visualization/backend/README.md",
        "src/visualization/frontend/README.md",
        ".kiro/specs/sentinel-safety-system/TASK_11_SUMMARY.md",
    ]
    
    for file_path in doc_files:
        full_path = base_path / file_path
        assert full_path.exists(), f"Missing documentation: {file_path}"
        
        with open(full_path, 'r') as f:
            content = f.read()
            assert len(content) > 100, f"Documentation too short: {file_path}"
        
        print(f"  ✓ {file_path}")
    
    print("  ✓ All documentation present and complete")


def main():
    """Run all verification tests."""
    print("=" * 70)
    print("SENTINEL Visualization Basic Verification")
    print("=" * 70)
    print()
    
    try:
        test_module_structure()
        print()
        test_example_files()
        print()
        test_test_files()
        print()
        test_file_syntax()
        print()
        test_documentation()
        
        print()
        print("=" * 70)
        print("✓ All basic verification tests passed!")
        print("=" * 70)
        print()
        print("Note: Full functionality requires FastAPI and dependencies.")
        print("Install with: pip install -r requirements.txt")
        
        return 0
        
    except Exception as e:
        print()
        print("=" * 70)
        print(f"✗ Verification failed: {e}")
        print("=" * 70)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
