"""Tests for BEV generation module."""

import pytest
import numpy as np
import yaml
from pathlib import Path

from src.perception.bev import BEVGenerator, PerspectiveTransformer, ViewStitcher, MaskGenerator


@pytest.fixture
def sample_calibration():
    """Sample calibration data for testing."""
    return {
        'intrinsics': {
            'fx': 800.0,
            'fy': 800.0,
            'cx': 640.0,
            'cy': 360.0,
            'distortion': [0.0, 0.0, 0.0, 0.0, 0.0]
        },
        'extrinsics': {
            'translation': [2.0, 0.5, 1.0],
            'rotation': [0.0, 0.0, -0.785]
        },
        'homography': {
            'matrix': [[1.0, 0.0, 0.0],
                      [0.0, 1.0, 0.0],
                      [0.0, 0.0, 1.0]]
        }
    }


@pytest.fixture
def bev_config():
    """Sample BEV configuration."""
    return {
        'output_size': [640, 640],
        'scale': 0.1,
        'vehicle_position': [320, 480],
        'blend_width': 50
    }


@pytest.fixture
def sample_frame():
    """Generate a sample camera frame."""
    return np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)


def test_perspective_transformer_init(sample_calibration):
    """Test PerspectiveTransformer initialization."""
    transformer = PerspectiveTransformer(sample_calibration, (640, 640))
    assert transformer.output_size == (640, 640)
    assert transformer.fx == 800.0
    assert transformer.camera_matrix.shape == (3, 3)


def test_perspective_transformer_undistort(sample_calibration, sample_frame):
    """Test undistortion."""
    transformer = PerspectiveTransformer(sample_calibration, (640, 640))
    undistorted = transformer.undistort(sample_frame)
    assert undistorted.shape == sample_frame.shape


def test_perspective_transformer_warp(sample_calibration, sample_frame):
    """Test warping to BEV."""
    transformer = PerspectiveTransformer(sample_calibration, (640, 640))
    bev = transformer.warp_to_bev(sample_frame)
    assert bev.shape == (640, 640, 3)


def test_perspective_transformer_transform(sample_calibration, sample_frame):
    """Test complete transformation pipeline."""
    transformer = PerspectiveTransformer(sample_calibration, (640, 640))
    bev = transformer.transform(sample_frame)
    assert bev.shape == (640, 640, 3)


def test_view_stitcher_init():
    """Test ViewStitcher initialization."""
    stitcher = ViewStitcher((640, 640), blend_width=50)
    assert stitcher.output_size == (640, 640)
    assert stitcher.blend_width == 50


def test_view_stitcher_single_view():
    """Test stitching with single view."""
    stitcher = ViewStitcher((640, 640))
    view = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    result = stitcher.stitch([view])
    assert result.shape == (640, 640, 3)


def test_view_stitcher_multiple_views():
    """Test stitching with multiple views."""
    stitcher = ViewStitcher((640, 640))
    view1 = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    view2 = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    result = stitcher.stitch([view1, view2])
    assert result.shape == (640, 640, 3)


def test_mask_generator_init():
    """Test MaskGenerator initialization."""
    mask_gen = MaskGenerator((640, 640), (320, 480))
    assert mask_gen.output_size == (640, 640)
    assert mask_gen.vehicle_position == (320, 480)


def test_mask_generator_generate():
    """Test mask generation."""
    mask_gen = MaskGenerator((640, 640), (320, 480))
    bev_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    mask = mask_gen.generate(bev_image)
    assert mask.shape == (640, 640)
    assert mask.dtype == bool


def test_bev_generator_init(bev_config, sample_calibration):
    """Test BEVGenerator initialization."""
    calibrations = {
        'front_left': sample_calibration,
        'front_right': sample_calibration
    }
    generator = BEVGenerator(bev_config, calibrations)
    assert generator.output_size == (640, 640)
    assert len(generator.transformers) == 2


def test_bev_generator_generate(bev_config, sample_calibration, sample_frame):
    """Test BEV generation."""
    calibrations = {
        'front_left': sample_calibration,
        'front_right': sample_calibration
    }
    generator = BEVGenerator(bev_config, calibrations)
    
    frames = [sample_frame, sample_frame]
    output = generator.generate(frames)
    
    assert output.image.shape == (640, 640, 3)
    assert output.mask.shape == (640, 640)
    assert output.mask.dtype == bool
    assert output.timestamp > 0


def test_bev_generator_performance_tracking(bev_config, sample_calibration, sample_frame):
    """Test performance tracking."""
    calibrations = {
        'front_left': sample_calibration
    }
    generator = BEVGenerator(bev_config, calibrations)
    
    # Generate a few frames
    for _ in range(5):
        generator.generate([sample_frame])
    
    avg_time = generator.get_average_processing_time()
    assert avg_time > 0
    
    stats = generator.get_performance_stats()
    assert 'avg_ms' in stats
    assert 'min_ms' in stats
    assert 'max_ms' in stats
    assert 'p95_ms' in stats
