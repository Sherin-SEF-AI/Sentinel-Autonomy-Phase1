# SENTINEL System Validation Guide

## Overview

This guide provides comprehensive information about validating the SENTINEL contextual safety intelligence platform. The validation suite ensures the system meets all performance, reliability, and accuracy requirements.

## Quick Start

### Run Complete Validation

```bash
# Run all validation tests
python3 scripts/run_validation_suite.py

# Run quick validation (faster, skips extended tests)
python3 scripts/run_validation_suite.py --quick

# Generate detailed report
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

## Validation Categories

### 1. End-to-End Integration Testing

**Purpose:** Verify complete pipeline functionality and data flow between modules

**Test File:** `tests/test_integration_e2e.py`

**What It Tests:**
- System initialization and module setup
- Camera capture to BEV generation pipeline
- BEV to semantic segmentation pipeline
- Multi-view object detection and tracking
- Driver monitoring system (DMS)
- Contextual intelligence and risk assessment
- Alert generation and dispatch
- Complete single-frame processing
- Data flow consistency across frames
- Module timing constraints
- Parallel processing capabilities
- Recording and visualization integration

**Requirements Validated:** 10.1, 10.2

**Expected Results:**
- All modules initialize successfully
- Data flows correctly through entire pipeline
- Each module produces expected output types
- Timing constraints are met
- No data corruption or loss

### 2. Performance Validation

**Purpose:** Measure system performance against specified targets

**Test File:** `tests/test_performance_validation.py`

**What It Tests:**

#### End-to-End Latency
- **Target:** <100ms at 95th percentile
- **Measures:** Time from camera capture to alert generation
- **Metrics:** Mean, median, p95, p99, min, max

#### System Throughput
- **Target:** ≥30 FPS
- **Measures:** Frames processed per second
- **Metrics:** Actual FPS, mean FPS, median FPS

#### GPU Memory Usage
- **Target:** ≤8GB
- **Measures:** GPU memory allocation and reservation
- **Metrics:** Mean, max, p95 memory usage

#### CPU Usage
- **Target:** ≤60% on 8-core processor
- **Measures:** CPU utilization during operation
- **Metrics:** Mean, max, p95 CPU percentage

**Requirements Validated:** 10.1, 10.2, 10.3, 10.4

**Expected Results:**
```
Latency (p95):    <100ms
Throughput:       ≥30 FPS
GPU Memory:       ≤8GB
CPU Usage:        ≤60% (8-core)
```

### 3. Reliability Validation

**Purpose:** Verify system fault tolerance and recovery capabilities

**Test File:** `tests/test_reliability_validation.py`

**What It Tests:**

#### Camera Disconnection Detection
- **Target:** Detect within 1 second
- **Tests:** System detects camera failure quickly

#### Camera Reconnection
- **Target:** Automatic reconnection
- **Tests:** System reinitializes reconnected cameras

#### Graceful Degradation
- **Target:** Continue with reduced coverage
- **Tests:** System operates with one camera failed

#### Inference Error Recovery
- **Target:** Automatic recovery
- **Tests:** System recovers from model inference errors

#### Crash Recovery
- **Target:** Restore state within 2 seconds
- **Tests:** System saves and restores state after crash

#### System Uptime
- **Target:** 99.9% uptime
- **Tests:** Extended operation without failures

**Requirements Validated:** 11.1, 11.2, 11.3, 11.4

**Expected Results:**
- Camera failures detected within 1s
- Automatic reconnection works
- System continues with partial camera coverage
- Inference errors don't crash system
- State restored within 2s after crash
- Uptime ≥99.9%

### 4. Accuracy Validation

**Purpose:** Validate perception and intelligence accuracy

**Test File:** `tests/test_accuracy_validation.py`

**What It Tests:**

#### BEV Segmentation Accuracy
- **Target:** mIoU ≥75%
- **Metric:** Mean Intersection over Union
- **Classes:** 9 semantic classes

#### Object Detection Accuracy
- **Target:** mAP ≥80%
- **Metric:** Mean Average Precision
- **Classes:** Vehicle, pedestrian, cyclist, etc.

#### Gaze Estimation Accuracy
- **Target:** Error <5 degrees
- **Metric:** Angular error (pitch, yaw)
- **Output:** Gaze direction in vehicle frame

#### Risk Prediction Accuracy
- **Target:** Accuracy ≥85%
- **Metric:** Correct risk level prediction
- **Scenarios:** High-risk and low-risk situations

**Requirements Validated:** 3.1, 4.1, 5.2, 6.5

**Expected Results:**
```
Segmentation mIoU:    ≥75%
Detection mAP:        ≥80%
Gaze Error:           <5°
Risk Accuracy:        ≥85%
```

**Note:** Current tests use synthetic data. Real validation requires labeled datasets.

## Test Infrastructure

### Mock Cameras

Tests use mock cameras to enable:
- Testing without physical hardware
- Reproducible results
- Automated CI/CD integration
- Simulation of failure scenarios

### Performance Monitoring

The `PerformanceMonitor` class provides:
- Real-time metric collection
- Background CPU/memory monitoring
- Statistical analysis
- Comprehensive reporting

### Synthetic Data

Tests generate synthetic data for:
- Camera frames
- Ground truth labels
- Test scenarios
- Edge cases

## Interpreting Results

### Pass/Fail Criteria

Tests use the following criteria:

**PASS:** Metric meets or exceeds target
```
✓ PASS: P95 latency 95.67ms <= 100ms
```

**FAIL:** Metric significantly below target
```
✗ FAIL: Throughput 25.3 FPS < 30 FPS
```

**INFO:** Using synthetic data (not real validation)
```
✗ INFO: mIoU 45.2% (synthetic data)
```

### Performance Tolerance

Tests allow some tolerance for test environments:
- Latency: Up to 150% of target
- Throughput: Down to 80% of target
- Memory: Up to 120% of target
- CPU: Up to 150% of target

### Report Format

Validation reports include:

```
==============================================================
VALIDATION REPORT
==============================================================

Total Test Suites: 4
Passed: 3
Failed: 1

DETAILED RESULTS:
--------------------------------------------------------------

End-to-End Integration:
  Status: ✓ PASS
  Time: 45.23s

Performance Validation:
  Status: ✓ PASS
  Time: 120.45s

Reliability Validation:
  Status: ✓ PASS
  Time: 65.12s

Accuracy Validation:
  Status: ✗ FAIL
  Time: 30.67s
  Error: Accuracy below threshold

==============================================================
OVERALL STATUS: ✗ 1 VALIDATION(S) FAILED
==============================================================
```

## Production Validation

For production deployment, perform additional validation:

### 1. Hardware Validation

- Run on target hardware platform
- Verify GPU compatibility
- Test with real cameras
- Measure actual performance

### 2. Dataset Validation

- Collect labeled datasets
- Run accuracy validation with real data
- Measure actual mIoU, mAP, gaze error
- Validate risk prediction accuracy

### 3. Extended Testing

- Run for extended periods (hours/days)
- Test in various lighting conditions
- Test with different camera configurations
- Validate in real driving scenarios

### 4. Field Testing

- Test in real vehicles
- Validate with real drivers
- Collect edge case scenarios
- Measure real-world performance

## Continuous Integration

### GitHub Actions Example

```yaml
name: Validation

on: [push, pull_request]

jobs:
  validate:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v2
    
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.10'
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install -r requirements-dev.txt
    
    - name: Run validation suite
      run: |
        python3 scripts/run_validation_suite.py --quick --report validation_report.txt
    
    - name: Upload report
      uses: actions/upload-artifact@v2
      with:
        name: validation-report
        path: validation_report.txt
```

## Troubleshooting

### Common Issues

#### Import Errors

**Problem:** `ModuleNotFoundError: No module named 'cv2'`

**Solution:**
```bash
pip install -r requirements.txt
```

#### GPU Tests Fail

**Problem:** GPU tests skip or fail

**Solution:**
- Ensure CUDA is installed
- Verify GPU is available
- Check PyTorch CUDA support
- Tests will skip if GPU unavailable

#### Performance Below Target

**Problem:** Performance metrics below targets

**Solution:**
- Check hardware specifications
- Close other applications
- Verify GPU is being used
- Adjust targets for your hardware

#### Camera Tests Fail

**Problem:** Camera initialization fails

**Solution:**
- Tests use mock cameras by default
- Check mock camera implementation
- Verify test fixtures are correct

### Debug Mode

Run tests with verbose output:

```bash
python3 -m pytest tests/test_integration_e2e.py -v -s
```

Enable debug logging:

```bash
export LOG_LEVEL=DEBUG
python3 scripts/run_validation_suite.py
```

## Best Practices

### Development Workflow

1. **During Development:**
   - Run quick validation frequently
   - Focus on relevant test suites
   - Fix issues immediately

2. **Before Commits:**
   - Run full validation suite
   - Ensure all tests pass
   - Review performance metrics

3. **Before Releases:**
   - Run extended validation
   - Generate comprehensive reports
   - Document any deviations

### Performance Monitoring

- Track performance trends over time
- Set up automated alerts for regressions
- Review metrics regularly
- Optimize bottlenecks

### Test Maintenance

- Update tests when requirements change
- Keep mock data realistic
- Add tests for new features
- Remove obsolete tests

## Additional Resources

- **Test Documentation:** `tests/VALIDATION_README.md`
- **Task Summary:** `.kiro/specs/sentinel-safety-system/TASK_15_SUMMARY.md`
- **Requirements:** `.kiro/specs/sentinel-safety-system/requirements.md`
- **Design:** `.kiro/specs/sentinel-safety-system/design.md`

## Support

For validation issues:
1. Check test output for detailed errors
2. Review this guide
3. Consult requirement specifications
4. Check system logs in `logs/`
5. Review design documentation

## Summary

The SENTINEL validation suite provides comprehensive testing across:
- ✓ Integration (end-to-end pipeline)
- ✓ Performance (latency, throughput, resources)
- ✓ Reliability (fault tolerance, recovery)
- ✓ Accuracy (perception, intelligence)

Run validation regularly to ensure system quality and requirement compliance.
