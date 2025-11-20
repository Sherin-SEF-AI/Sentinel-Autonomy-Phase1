# Semantic Segmentation Module

This module implements semantic segmentation for Bird's Eye View (BEV) images, classifying each pixel into one of 9 semantic classes.

## Components

### 1. BEVSegmentationModel (`model.py`)
- Wrapper for deep learning segmentation model
- Supports FP16 precision for performance
- GPU memory management
- Includes placeholder model for development/testing

### 2. TemporalSmoother (`smoother.py`)
- Exponential moving average for temporal stability
- Reduces segmentation flicker across frames
- Configurable alpha parameter (default: 0.7)

### 3. SemanticSegmentor (`segmentor.py`)
- Main interface implementing `ISemanticSegmentor`
- Integrates model and temporal smoother
- Error recovery and fallback mechanisms
- Performance tracking

## Semantic Classes

The segmentation model classifies pixels into 9 classes:

0. **road** - Drivable road surface
1. **lane_marking** - Lane markings and road paint
2. **vehicle** - Cars, trucks, buses
3. **pedestrian** - People
4. **cyclist** - Bicycles and riders
5. **obstacle** - Static obstacles
6. **parking_space** - Parking areas
7. **curb** - Road curbs and edges
8. **vegetation** - Trees, grass, plants

## Usage

### Basic Usage

```python
from src.perception.segmentation import SemanticSegmentor
import cv2

# Configuration
config = {
    'weights': 'models/bev_segmentation.pth',
    'device': 'cuda',
    'precision': 'fp16',
    'temporal_smoothing': True,
    'smoothing_alpha': 0.7
}

# Initialize segmentor
segmentor = SemanticSegmentor(config)

# Load BEV image
bev_image = cv2.imread('bev_frame.png')  # (640, 640, 3) BGR

# Run segmentation
output = segmentor.segment(bev_image)

# Access results
class_map = output.class_map      # (640, 640) int8, class indices
confidence = output.confidence    # (640, 640) float32, confidence scores
timestamp = output.timestamp      # float, timestamp
```

### With BEV Generator

```python
from src.perception.bev import BEVGenerator
from src.perception.segmentation import SemanticSegmentor

# Initialize modules
bev_gen = BEVGenerator(bev_config)
segmentor = SemanticSegmentor(seg_config)

# Process camera frames
frames = [front_left_frame, front_right_frame]
bev_output = bev_gen.generate(frames)

# Segment BEV
seg_output = segmentor.segment(bev_output.image)
```

### Performance Monitoring

```python
# Get performance statistics
stats = segmentor.get_performance_stats()

print(f"Mean inference time: {stats['mean_inference_time']*1000:.1f}ms")
print(f"P95 inference time: {stats['p95_inference_time']*1000:.1f}ms")
print(f"FPS: {stats['fps']:.1f}")
print(f"Target met: {stats['target_met']}")
```

## Configuration

Configuration is loaded from YAML file (e.g., `configs/default.yaml`):

```yaml
models:
  segmentation:
    architecture: "BEVFormer-Tiny"
    weights: "models/bev_segmentation.pth"
    device: "cuda"
    precision: "fp16"
    temporal_smoothing: true
    smoothing_alpha: 0.7
```

### Configuration Parameters

- **architecture**: Model architecture name (informational)
- **weights**: Path to pretrained model weights
- **device**: Device to run on ('cuda' or 'cpu')
- **precision**: Precision mode ('fp16' or 'fp32')
- **temporal_smoothing**: Enable temporal smoothing (true/false)
- **smoothing_alpha**: Smoothing factor (0-1), higher = more weight on current frame

## Performance Requirements

- **Target inference time**: 15ms per frame
- **Input size**: 640x640 pixels
- **Output**: Class map (640x640 int8) + confidence (640x640 float32)
- **Accuracy requirement**: mIoU ≥ 75% on validation set

## Validation

### Running Validation

To validate segmentation accuracy on a labeled dataset:

```bash
python scripts/validate_segmentation.py \
    --dataset /path/to/validation/dataset \
    --config configs/default.yaml \
    --output validation_results.json
```

### Dataset Format

The validation dataset should have the following structure:

```
dataset/
├── images/
│   ├── 000001.png
│   ├── 000002.png
│   └── ...
└── labels/
    ├── 000001.png
    ├── 000002.png
    └── ...
```

- **images/**: BEV images (640x640, RGB/BGR)
- **labels/**: Ground truth segmentation masks (640x640, grayscale)
  - Pixel values represent class indices (0-8)

### Validation Metrics

The validation script calculates:
- **Mean IoU (mIoU)**: Average Intersection over Union across all classes
- **Per-class IoU**: IoU for each semantic class
- **Pass/Fail**: Whether mIoU ≥ 75% requirement is met

### Example Output

```
============================================================
SEGMENTATION VALIDATION RESULTS
============================================================
Number of samples: 500
Mean IoU: 0.7823
Std IoU: 0.0456

Requirement: mIoU ≥ 0.75
Status: ✓ PASSED

Per-class IoU:
  road                : 0.9234
  lane_marking        : 0.7845
  vehicle             : 0.8123
  pedestrian          : 0.7234
  cyclist             : 0.6891
  obstacle            : 0.7456
  parking_space       : 0.8012
  curb                : 0.7567
  vegetation          : 0.8345
============================================================
```

## Testing

### Unit Tests

Run unit tests (no GPU required):

```bash
pytest tests/test_smoother_standalone.py -v
```

This tests:
- Temporal smoother functionality
- mIoU calculation
- Edge cases and error handling

### Integration Tests

Run full integration tests (requires GPU and PyTorch):

```bash
pytest tests/test_segmentation.py -v
```

This tests:
- Model loading and inference
- End-to-end segmentation pipeline
- Performance requirements
- Error recovery

## Error Handling

The segmentor includes robust error handling:

1. **Invalid input**: Returns fallback output (all road class)
2. **Inference failure**: Uses last valid output
3. **Repeated failures**: Automatically reloads model after 3 errors
4. **GPU memory issues**: Clears cache and retries

## Model Development

### Training a Custom Model

To train a custom segmentation model:

1. Prepare training dataset in the same format as validation
2. Train model using your preferred framework (PyTorch, etc.)
3. Export model to `.pth` format
4. Update `weights` path in configuration
5. Validate accuracy using validation script

### Model Requirements

- **Input**: (1, 3, 640, 640) tensor, RGB, normalized [0, 1]
- **Output**: (1, 9, 640, 640) tensor, logits for 9 classes
- **Architecture**: Any encoder-decoder architecture (U-Net, DeepLab, BEVFormer, etc.)

### Recommended Models

- **BEVFormer-Tiny**: Lightweight transformer-based model
- **DeepLabV3+**: Strong baseline with good accuracy
- **U-Net**: Simple and effective for segmentation
- **SegFormer**: Efficient transformer architecture

## Troubleshooting

### Slow Inference

If inference time exceeds 15ms:
- Enable FP16 precision
- Use TensorRT optimization
- Reduce model size
- Check GPU utilization

### Low Accuracy

If mIoU < 75%:
- Collect more training data
- Augment training data
- Fine-tune on domain-specific data
- Try different model architecture
- Adjust class weights for imbalanced classes

### Flickering Segmentation

If segmentation flickers between frames:
- Enable temporal smoothing
- Increase smoothing alpha (0.7 → 0.8)
- Apply post-processing (CRF, etc.)

## References

- [BEVFormer Paper](https://arxiv.org/abs/2203.17270)
- [Semantic Segmentation Overview](https://paperswithcode.com/task/semantic-segmentation)
- [IoU Metric Explanation](https://en.wikipedia.org/wiki/Jaccard_index)
