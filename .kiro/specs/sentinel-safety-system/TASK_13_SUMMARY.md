# Task 13: Create Calibration Tooling - Implementation Summary

## Overview
Implemented comprehensive camera calibration tooling for the SENTINEL system, including scripts for camera calibration, validation, and detailed documentation.

## Implementation Details

### 13.1 Camera Calibration Script (`scripts/calibrate_camera.py`)

Created a full-featured calibration script that:

**Core Functionality:**
- Interactive checkerboard image capture with real-time corner detection
- Automatic computation of camera intrinsics (focal length, principal point, distortion)
- Extrinsics configuration (camera position and orientation relative to vehicle frame)
- Homography matrix computation for BEV transformation
- YAML file generation with all calibration parameters

**Key Features:**
- Supports all three cameras (interior, front_left, front_right)
- Real-time visual feedback during capture
- Minimum 10 images required, 20 recommended
- Corner refinement using sub-pixel accuracy
- Reprojection error calculation for quality assessment
- Interactive extrinsics input with sensible defaults
- Batch calibration mode (--all flag)

**Usage Examples:**
```bash
# Calibrate single camera
python3 scripts/calibrate_camera.py --camera interior --device 0

# Calibrate external camera with BEV
python3 scripts/calibrate_camera.py --camera front_left --device 1 --bev

# Calibrate all cameras
python3 scripts/calibrate_camera.py --all
```

**Calibration Process:**
1. Opens camera and displays live feed
2. Detects checkerboard pattern (9x6 default)
3. User captures 20 images at various positions/angles
4. Computes intrinsic parameters using OpenCV
5. Prompts for extrinsic parameters (external cameras)
6. Generates homography matrix (external cameras with --bev)
7. Saves complete calibration to YAML file

### 13.2 Calibration Validation Script (`scripts/validate_calibration.py`)

Created comprehensive validation script that:

**Quality Metrics:**
- Focal length reasonableness check (0.5-2.0x image width)
- Principal point location verification (near image center)
- Distortion coefficient magnitude check
- Overall quality rating (GOOD/ACCEPTABLE/POOR)

**Interactive Validation:**
- **Undistortion Visualization:**
  - Side-by-side comparison of original vs undistorted images
  - Grid overlay to verify straight lines
  - Real-time camera feed
  - Save comparison images (press 's')

- **BEV Transformation Visualization:**
  - Real-time BEV view from camera
  - Grid overlay showing 6.4m spacing
  - Vehicle position marker
  - Verify ground plane transformation
  - Save BEV images (press 's')

**Usage Examples:**
```bash
# Validate single camera
python3 scripts/validate_calibration.py --camera interior --device 0

# Validate with BEV visualization
python3 scripts/validate_calibration.py --camera front_left --device 1 --show-bev

# Validate all cameras
python3 scripts/validate_calibration.py --all
```

**Validation Output:**
- Quality metrics for each camera
- Interactive visual verification
- Recommendations for recalibration if needed
- Summary report for all cameras

### Supporting Files

**1. Calibration README (`configs/calibration/README.md`)**
- Complete calibration guide
- Step-by-step instructions
- Troubleshooting section
- Calibration file format documentation
- Coordinate system definitions
- Integration instructions

**2. Test Suite (`scripts/test_calibration_tools.py`)**
- Validates calibration file format
- Tests data structure requirements
- Verifies coordinate system definitions
- Checks quality assessment logic
- Tests existing calibration files
- All tests passing ✓

## File Structure

```
scripts/
├── calibrate_camera.py          # Main calibration script
├── validate_calibration.py      # Validation script
└── test_calibration_tools.py    # Test suite

configs/calibration/
├── README.md                     # Comprehensive guide
├── interior.yaml                 # Interior camera calibration
├── front_left.yaml              # Front-left camera calibration
└── front_right.yaml             # Front-right camera calibration
```

## Calibration File Format

### Interior Camera
```yaml
camera_name: interior
image_size:
  width: 640
  height: 480
intrinsics:
  fx: 500.0
  fy: 500.0
  cx: 320.0
  cy: 240.0
  distortion: [k1, k2, p1, p2, k3]
```

### External Camera (with BEV)
```yaml
camera_name: front_left
image_size:
  width: 1280
  height: 720
intrinsics:
  fx: 800.0
  fy: 800.0
  cx: 640.0
  cy: 360.0
  distortion: [k1, k2, p1, p2, k3]
extrinsics:
  translation: [1.5, 0.5, 1.2]    # [x, y, z] meters
  rotation: [0.0, -0.2, 0.785]    # [roll, pitch, yaw] radians
homography:
  matrix:
    - [h11, h12, h13]
    - [h21, h22, h23]
    - [h31, h32, h33]
```

## Coordinate Systems

### Vehicle Frame
- Origin: Center of rear axle
- X-axis: Forward
- Y-axis: Left
- Z-axis: Up

### BEV Frame
- Size: 640x640 pixels
- Scale: 0.1 meters/pixel (64m x 64m coverage)
- Vehicle position: Center-bottom of image

## Key Features

1. **User-Friendly Interface:**
   - Real-time visual feedback
   - Clear instructions
   - Progress indicators
   - Keyboard controls

2. **Robust Calibration:**
   - Sub-pixel corner refinement
   - Reprojection error calculation
   - Quality metrics validation
   - Minimum image requirements

3. **Comprehensive Validation:**
   - Multiple validation modes
   - Interactive visualization
   - Quality assessment
   - Save validation images

4. **Flexible Configuration:**
   - Single or batch calibration
   - Configurable checkerboard size
   - Custom output directory
   - Optional BEV computation

5. **Production Ready:**
   - Error handling
   - Input validation
   - Helpful error messages
   - Comprehensive documentation

## Testing Results

All tests passing:
```
✓ Calibration File Creation
✓ Calibration Data Structure
✓ Coordinate Systems
✓ Calibration Quality Checks
✓ Existing Calibration Files

Passed: 5/5
Failed: 0/5
```

## Integration with SENTINEL

The calibration system integrates seamlessly with SENTINEL:

1. **Camera Manager** loads calibration files at startup
2. **CalibrationLoader** class reads YAML files
3. **BEV Generator** uses homography matrices
4. **Undistortion** applied to all camera frames

Configuration in `configs/default.yaml`:
```yaml
cameras:
  interior:
    calibration: "configs/calibration/interior.yaml"
  front_left:
    calibration: "configs/calibration/front_left.yaml"
  front_right:
    calibration: "configs/calibration/front_right.yaml"
```

## Requirements Satisfied

✓ **Requirement 1.5:** Camera calibration data loading
- Intrinsic parameters (fx, fy, cx, cy, distortion)
- Extrinsic parameters (translation, rotation)
- Homography matrices for BEV transformation

✓ **Requirement 12.1:** Configuration management
- YAML-based calibration files
- Persistent storage
- Easy integration with system config

## Usage Workflow

1. **Calibrate Cameras:**
   ```bash
   python3 scripts/calibrate_camera.py --all
   ```

2. **Validate Calibration:**
   ```bash
   python3 scripts/validate_calibration.py --all
   ```

3. **Review Quality:**
   - Check validation output
   - Verify GOOD quality rating
   - Review visual validation

4. **Integrate with System:**
   - Calibration files already in correct location
   - System automatically loads on startup
   - No additional configuration needed

## Documentation

Comprehensive documentation provided:
- **README.md:** Complete calibration guide
- **Script help:** Built-in usage instructions
- **Code comments:** Detailed implementation notes
- **Examples:** Multiple usage scenarios

## Next Steps

The calibration tooling is complete and ready for use:

1. ✓ Camera calibration script implemented
2. ✓ Validation script implemented
3. ✓ Documentation created
4. ✓ Test suite passing
5. ✓ Integration verified

Users can now:
- Calibrate new camera installations
- Validate existing calibrations
- Recalibrate when needed
- Verify calibration quality

## Notes

- Scripts require OpenCV (cv2) for camera access and image processing
- Checkerboard pattern must be printed and mounted flat
- Good lighting conditions recommended for best results
- Minimum 10 images required, 20 recommended for optimal quality
- Validation provides real-time visual feedback
- All scripts include comprehensive help text
