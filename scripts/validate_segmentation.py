#!/usr/bin/env python3
"""
Validation script for semantic segmentation module.

This script validates the segmentation accuracy on a labeled dataset
and verifies that mIoU ≥ 75% as required.

Usage:
    python scripts/validate_segmentation.py --dataset <path> --config <config_file>
"""

import argparse
import logging
import sys
from pathlib import Path
import numpy as np
import yaml
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from perception.segmentation import SemanticSegmentor
from core.data_structures import SegmentationOutput


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def calculate_miou(pred: np.ndarray, gt: np.ndarray, num_classes: int = 9) -> tuple:
    """
    Calculate mean Intersection over Union and per-class IoU.
    
    Args:
        pred: Predicted class map (H, W)
        gt: Ground truth class map (H, W)
        num_classes: Number of classes
        
    Returns:
        Tuple of (mean_iou, per_class_iou_dict)
    """
    ious = {}
    
    for cls in range(num_classes):
        pred_mask = pred == cls
        gt_mask = gt == cls
        
        intersection = np.logical_and(pred_mask, gt_mask).sum()
        union = np.logical_or(pred_mask, gt_mask).sum()
        
        if union == 0:
            # Class not present in ground truth or prediction
            continue
        
        iou = intersection / union
        ious[cls] = iou
    
    mean_iou = np.mean(list(ious.values())) if ious else 0.0
    return mean_iou, ious


def load_validation_dataset(dataset_path: Path):
    """
    Load validation dataset.
    
    Expected structure:
        dataset_path/
            images/
                000001.png
                000002.png
                ...
            labels/
                000001.png
                000002.png
                ...
    
    Args:
        dataset_path: Path to validation dataset
        
    Yields:
        Tuple of (image, ground_truth, image_id)
    """
    import cv2
    
    images_dir = dataset_path / 'images'
    labels_dir = dataset_path / 'labels'
    
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    
    if not labels_dir.exists():
        raise FileNotFoundError(f"Labels directory not found: {labels_dir}")
    
    image_files = sorted(images_dir.glob('*.png'))
    
    for image_file in image_files:
        label_file = labels_dir / image_file.name
        
        if not label_file.exists():
            logger.warning(f"Label not found for {image_file.name}, skipping")
            continue
        
        # Load image (BGR format)
        image = cv2.imread(str(image_file))
        
        # Load ground truth (grayscale, class indices)
        gt = cv2.imread(str(label_file), cv2.IMREAD_GRAYSCALE)
        
        if image is None or gt is None:
            logger.warning(f"Failed to load {image_file.name}, skipping")
            continue
        
        yield image, gt.astype(np.int8), image_file.stem


def validate_segmentation(
    segmentor: SemanticSegmentor,
    dataset_path: Path,
    num_classes: int = 9
) -> dict:
    """
    Validate segmentation model on dataset.
    
    Args:
        segmentor: SemanticSegmentor instance
        dataset_path: Path to validation dataset
        num_classes: Number of classes
        
    Returns:
        Dictionary with validation results
    """
    logger.info(f"Loading validation dataset from {dataset_path}")
    
    all_ious = []
    per_class_ious = {i: [] for i in range(num_classes)}
    
    try:
        dataset = list(load_validation_dataset(dataset_path))
        
        if not dataset:
            logger.error("No validation samples found")
            return {
                'mean_iou': 0.0,
                'num_samples': 0,
                'per_class_iou': {},
                'passed': False
            }
        
        logger.info(f"Found {len(dataset)} validation samples")
        
        for image, gt, image_id in tqdm(dataset, desc="Validating"):
            # Run segmentation
            output = segmentor.segment(image)
            
            # Calculate IoU
            miou, class_ious = calculate_miou(output.class_map, gt, num_classes)
            all_ious.append(miou)
            
            # Accumulate per-class IoUs
            for cls, iou in class_ious.items():
                per_class_ious[cls].append(iou)
        
        # Calculate statistics
        mean_iou = np.mean(all_ious)
        std_iou = np.std(all_ious)
        
        # Calculate per-class mean IoU
        per_class_mean = {}
        for cls, ious in per_class_ious.items():
            if ious:
                per_class_mean[cls] = np.mean(ious)
        
        # Check if requirement is met (mIoU ≥ 75%)
        passed = mean_iou >= 0.75
        
        results = {
            'mean_iou': mean_iou,
            'std_iou': std_iou,
            'num_samples': len(dataset),
            'per_class_iou': per_class_mean,
            'passed': passed,
            'requirement': 0.75
        }
        
        return results
        
    except FileNotFoundError as e:
        logger.error(f"Dataset not found: {e}")
        return {
            'mean_iou': 0.0,
            'num_samples': 0,
            'per_class_iou': {},
            'passed': False,
            'error': str(e)
        }


def main():
    parser = argparse.ArgumentParser(
        description='Validate semantic segmentation accuracy'
    )
    parser.add_argument(
        '--dataset',
        type=Path,
        required=True,
        help='Path to validation dataset'
    )
    parser.add_argument(
        '--config',
        type=Path,
        default='configs/default.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--output',
        type=Path,
        help='Path to save validation results (JSON)'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    logger.info(f"Loading configuration from {args.config}")
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Extract segmentation config
    seg_config = config.get('models', {}).get('segmentation', {})
    
    # Initialize segmentor
    logger.info("Initializing semantic segmentor")
    segmentor = SemanticSegmentor(seg_config)
    
    # Run validation
    logger.info("Starting validation")
    results = validate_segmentation(segmentor, args.dataset)
    
    # Print results
    print("\n" + "="*60)
    print("SEGMENTATION VALIDATION RESULTS")
    print("="*60)
    print(f"Number of samples: {results['num_samples']}")
    print(f"Mean IoU: {results['mean_iou']:.4f}")
    
    if 'std_iou' in results:
        print(f"Std IoU: {results['std_iou']:.4f}")
    
    print(f"\nRequirement: mIoU ≥ {results['requirement']:.2f}")
    print(f"Status: {'✓ PASSED' if results['passed'] else '✗ FAILED'}")
    
    if results['per_class_iou']:
        print("\nPer-class IoU:")
        class_names = [
            'road', 'lane_marking', 'vehicle', 'pedestrian',
            'cyclist', 'obstacle', 'parking_space', 'curb', 'vegetation'
        ]
        for cls, iou in sorted(results['per_class_iou'].items()):
            class_name = class_names[cls] if cls < len(class_names) else f'class_{cls}'
            print(f"  {class_name:20s}: {iou:.4f}")
    
    print("="*60)
    
    # Save results if output path provided
    if args.output:
        import json
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {args.output}")
    
    # Exit with appropriate code
    sys.exit(0 if results['passed'] else 1)


if __name__ == '__main__':
    main()
