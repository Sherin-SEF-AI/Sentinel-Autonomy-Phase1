# SENTINEL System Validation Suite

This directory contains comprehensive validation tests for the SENTINEL contextual safety intelligence platform.

## Overview

The validation suite is organized into four main categories:

1. **End-to-End Integration Testing** (`test_integration_e2e.py`)
2. **Performance Validation** (`test_performance_validation.py`)
3. **Reliability Validation** (`test_reliability_validation.py`)
4. **Accuracy Validation** (`test_accuracy_validation.py`)

## Test Categories

### 1. End-to-End Integration Testing

**File:** `test_integration_e2e.py`  
**Requirements:** 10.1, 10.2

Tests the complete pipeline with real camera feeds, verifies data flow between all modules, and checks timing constraints.

**Key Tests:**
- System initialization
- Camera to BEV pipeline
- BEV to segmentation pipeline
- Detection pipeline
- DMS pipeline
- Intelligence pipeline
- Alert generation pipeline
- Complete pipeline single frame processing
- Pipeline data flow consistency
- Module timing constraints
- Parallel processing
- Recording integration
- Visualization integration

### 2. Performance Validation

**File:** `test_performance_validation.py`  
**Requirements:** 10.1, 10.2, 10.3, 10.4

Measures system performance metrics against requirements.

**Key Tests:**
- **End-to-end latency:** Target <100ms at p95
- **Throughput:** Target ≥30 FPS
- **GPU memory usage:** Target ≤8GB
- **CPU usage:** Target ≤60% on 8-core processor

**Performance Targets:**
```
Latency:     <100ms (p95)
Throughput:  ≥30 FPS
GPU Memory:  ≤8GB
CPU Usage:   ≤60% (8-core)
```

### 3. Reliability Validation

**File:** `test_reliability_validation.py`  
**Requirements:** 11.1, 11.2, 11.3, 11.4

Tests system reliability and fault tolerance.

**Key Tests:**
- Camera disconnection detection (within 1 second)
- Camera reconnection
- Graceful degradation with single camera failure
- Inference error recovery
- Crash recovery and state persistence (within 2 seconds)
- System uptime (target: 99.9%)
- Concurrent error handling

### 4. Accuracy Validation

**File:** `test_accuracy_validation.py`  
**Requirements:** 3.1, 4.1, 5.2, 6.5

Validates accuracy of perception and intelligence modules.

**Key Tests:**
- **BEV segmentation mIoU:** Target ≥75%
- **Object detection mAP:** Target ≥80%
- **Gaze estimation error:** Target <5 degrees
- **Risk prediction accuracy:** Target ≥85%

**Note:** Real accuracy validation requires labeled datasets. The current tests validate metric calculation with synthetic data.

## Running the Tests

### Run All Validation Tests

```bash
# Run complete validation suite
python3 scripts/run_validation_suite.py

# Run quick validation (skip extended tests)
python3 scripts/run_validation_suite.py --quick

# Generate report file
python3 scripts/run_validation_suite.py --report validation_report.txt
```

### Run Individual Test Suites

```bash
# Integration tests
python3 -m pytest tests/test_integration_e2e.py -v

# Performance tests
python3 -m pytest tests/test_performance_validation.py -v

# Reliability tests
python3 -m pytest tests/test_reliability_validation.py -v

# Accuracy tests
python3 -m pytest tests/test_accuracy_validation.py -v
```

### Run Specific Tests

```bash
# Run a specific test
python3 -m pytest tests/test_integration_e2e.py::TestEndToEndIntegration::test_system_initialization -v

# Run tests matching a pattern
python3 -m pytest tests/test_performance_validation.py -k "latency" -v
```

## Test Output

### Console Output

Tests provide detailed console output including:
- Test progress and status
- Performance metrics
- Validation results
- Pass/fail status for each requirement

### Example Output

```
==============================================================
END-TO-END LATENCY VALIDATION
==============================================================
Mean latency:   85.23ms
Median latency: 82.15ms
P95 latency:    95.67ms
P99 latency:    98.34ms
Min latency:    75.12ms
Max latency:    102.45ms
==============================================================
✓ PASS: P95 latency 95.67ms <= 100ms
```

### Report Generation

The validation suite can generate a comprehensive report:

```bash
python3 scripts/run_validation_suite.py --report validation_report.txt
```

Report includes:
- Summary of all test suites
- Pass/fail status
- Execution time
- Detailed results for each validation category

## Requirements Mapping

| Test Suite | Requirements | Description |
|------------|--------------|-------------|
| Integration | 10.1, 10.2 | End-to-end pipeline testing |
| Performance | 10.1, 10.2, 10.3, 10.4 | Latency, throughput, resource usage |
| Reliability | 11.1, 11.2, 11.3, 11.4 | Fault tolerance, recovery |
| Accuracy | 3.1, 4.1, 5.2, 6.5 | Perception and intelligence accuracy |

## Mock Data

The validation tests use mock cameras and synthetic data to enable testing without physical hardware. This allows:
- Automated testing in CI/CD pipelines
- Reproducible test results
- Testing of error conditions and edge cases

For production validation, replace mock cameras with real hardware and use labeled datasets for accuracy validation.

## Continuous Integration

These tests are designed to run in CI/CD pipelines:

```yaml
# Example GitHub Actions workflow
- name: Run Validation Suite
  run: |
    python3 scripts/run_validation_suite.py --quick --report validation_report.txt
```

## Troubleshooting

### Tests Fail to Import Modules

Ensure all dependencies are installed:
```bash
pip install -r requirements.txt
```

### GPU Tests Fail

GPU tests require CUDA-capable hardware. Tests will skip if CUDA is not available.

### Camera Tests Fail

Camera tests use mock cameras by default. For real camera testing, modify the test fixtures.

### Performance Tests Show Poor Results

Performance depends on hardware. Adjust targets in test configuration if needed for your hardware.

## Adding New Tests

To add new validation tests:

1. Create test methods in the appropriate test class
2. Follow the naming convention: `test_<feature>_<aspect>`
3. Add docstrings with requirement references
4. Use appropriate assertions and logging
5. Update this README with new test descriptions

## Best Practices

- Run quick validation during development
- Run full validation before releases
- Generate reports for documentation
- Review performance trends over time
- Update tests when requirements change
- Keep mock data realistic

## Support

For issues or questions about the validation suite:
- Check test output for detailed error messages
- Review requirement specifications
- Consult system design documentation
- Check logs in `logs/` directory
