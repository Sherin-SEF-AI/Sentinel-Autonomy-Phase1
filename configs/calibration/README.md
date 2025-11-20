# Camera Calibration Guide

This directory contains camera calibration files for the SENTINEL system. Each camera requires calibration to correct for lens distortion and enable accurate BEV transformation.

## Calibration Files

- `interior.yaml` - Interior (DMS) camera calibration
- `front_left.yaml` - Front-left external camera calibration (with BEV homography)
- `front_right.yaml` - Front-right external camera calibration (with BEV homography)

## Calibration Process

### Prerequisites

1. **Checkerboard Pattern**
   - Print a checkerboard pattern (9x6 squares recommended)
   - Use high-quality paper and ensure pattern is flat
   - Default square size: 25mm (configurable in script)
   - Mount on rigid board to keep flat

2. **Camera Setup**
   - Mount cameras in their final positions
   - Ensure stable mounting (no vibration)
   - Connect cameras via USB 3.0
   - Verify camera device IDs (use `ls /dev/video*`)

3. **Environment**
   - Good, even lighting
   - Minimal shadows on checkerboard
   - Stable environment (no camera movement)

### Step 1: Calibrate Cameras

#### Option A: Calibrate All Cameras

```bash
python scripts/calibrate_camera.py --all
```

This will calibrate all three cameras sequentially.

#### Option B: Calibrate Individual Cameras

**Interior Camera:**
```bash
python scripts/calibrate_camera.py --camera interior --device 0
```

**Front-Left Camera (with BEV):**
```bash
python scripts/calibrate_camera.py --camera front_left --device 1 --bev
```

**Front-Right Camera (with BEV):**
```bash
python scripts/calibrate_camera.py --camera front_right --device 2 --bev
```

### Step 2: Capture Calibration Images

During calibration:

1. **Position the checkerboard** in the camera view
2. **Press SPACE** when checkerboard is detected (corners highlighted)
3. **Move checkerboard** to different positions and angles:
   - Center, corners, and edges of frame
   - Various distances from camera
   - Different orientations (tilted, rotated)
4. **Capture 20 images** (minimum 10 for acceptable calibration)
5. **Press 'q'** to finish capturing

**Tips for good calibration:**
- Cover the entire field of view
- Include images with checkerboard at various angles
- Ensure checkerboard is fully visible in each image
- Avoid motion blur (hold steady when capturing)

### Step 3: Configure Extrinsics (External Cameras Only)

For external cameras, you'll be prompted to enter:

**Position (meters from vehicle origin - rear axle center):**
- X: Forward distance (e.g., 1.5m for front cameras)
- Y: Left/right offset (e.g., 0.5m left, -0.5m right)
- Z: Height above ground (e.g., 1.2m)

**Orientation (radians):**
- Roll: Rotation around forward axis (usually 0)
- Pitch: Tilt up/down (e.g., -0.2 for slight downward tilt)
- Yaw: Rotation left/right (e.g., 0.785 for 45° left, -0.785 for 45° right)

**Default values are provided** - press Enter to accept defaults.

### Step 4: Validate Calibration

```bash
# Validate all cameras
python scripts/validate_calibration.py --all

# Validate individual camera
python scripts/validate_calibration.py --camera front_left --device 1 --show-bev
```

Validation includes:

1. **Quality Metrics Check**
   - Focal length reasonableness
   - Principal point location
   - Distortion coefficient magnitude

2. **Undistortion Visualization**
   - Side-by-side comparison of original vs undistorted
   - Grid overlay to check for straight lines
   - Press 's' to save comparison images

3. **BEV Transformation Visualization** (external cameras)
   - Real-time BEV view
   - Grid overlay (6.4m spacing)
   - Vehicle position marker
   - Press 's' to save BEV images

### Step 5: Review Results

Check calibration quality:
- **GOOD**: Ready for production use
- **ACCEPTABLE**: Usable but may benefit from recalibration
- **POOR**: Recalibration recommended

If quality is POOR:
1. Ensure checkerboard pattern is accurate and flat
2. Improve lighting conditions
3. Capture more images with better coverage
4. Verify camera is in focus

## Calibration File Format

### Interior Camera (interior.yaml)

```yaml
camera_name: interior
image_size:
  width: 640
  height: 480
intrinsics:
  fx: 500.0          # Focal length X
  fy: 500.0          # Focal length Y
  cx: 320.0          # Principal point X
  cy: 240.0          # Principal point Y
  distortion: [k1, k2, p1, p2, k3]  # Distortion coefficients
```

### External Camera (front_left.yaml, front_right.yaml)

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
  translation: [1.5, 0.5, 1.2]      # [x, y, z] in meters
  rotation: [0.0, -0.2, 0.785]      # [roll, pitch, yaw] in radians
homography:
  matrix:
    - [h11, h12, h13]
    - [h21, h22, h23]
    - [h31, h32, h33]
```

## Coordinate Systems

### Vehicle Frame
- **Origin**: Center of rear axle
- **X-axis**: Forward (front of vehicle)
- **Y-axis**: Left
- **Z-axis**: Up

### BEV Frame
- **Size**: 640x640 pixels
- **Scale**: 0.1 meters per pixel (64m x 64m coverage)
- **Origin**: Center of image
- **Vehicle position**: Typically at (320, 480) - center-bottom

## Troubleshooting

### Checkerboard Not Detected
- Ensure good lighting
- Check checkerboard is flat and in focus
- Verify checkerboard size matches script parameters
- Try different distances from camera

### Poor Calibration Quality
- Capture more images (20+ recommended)
- Improve coverage of field of view
- Ensure checkerboard is at various angles
- Check for motion blur in captured images

### BEV Looks Wrong
- Verify extrinsic parameters (position and orientation)
- Check homography matrix computation
- Ensure ground plane is visible in camera view
- Consider manual homography adjustment

### Camera Not Opening
- Check device ID: `ls /dev/video*`
- Verify USB connection (use USB 3.0)
- Check camera permissions
- Try different device ID

## Recalibration

Recalibration is recommended when:
- Camera position or orientation changes
- Lens is adjusted or replaced
- Calibration quality degrades over time
- BEV transformation appears inaccurate

To recalibrate, simply run the calibration script again. Old calibration files will be overwritten.

## Advanced Configuration

### Custom Checkerboard Size

```bash
python scripts/calibrate_camera.py \
  --camera front_left \
  --device 1 \
  --checkerboard-size 10 7 \
  --square-size 0.03
```

### Manual Homography Adjustment

If automatic homography computation is inaccurate, you can manually adjust the matrix in the YAML file. Use the validation script to verify changes in real-time.

## Integration with SENTINEL

Once calibration is complete:

1. Verify calibration files exist in `configs/calibration/`
2. Update `configs/default.yaml` to reference calibration files:

```yaml
cameras:
  interior:
    calibration: "configs/calibration/interior.yaml"
  front_left:
    calibration: "configs/calibration/front_left.yaml"
  front_right:
    calibration: "configs/calibration/front_right.yaml"
```

3. Run SENTINEL system:

```bash
python src/main.py --config configs/default.yaml
```

The system will automatically load and apply calibration parameters.
