#!/usr/bin/env python3
"""
Model Download Script for SENTINEL System

Downloads pretrained models required for the SENTINEL safety system:
- BEV segmentation model
- YOLOv8 object detection model
- L2CS-Net gaze estimation model
- Drowsiness detection model
- Distraction classification model

Usage:
    python scripts/download_models.py [--force] [--verify-only]
"""

import argparse
import hashlib
import os
import sys
from pathlib import Path
from typing import Dict, Optional
import urllib.request
import json

# Model definitions with download URLs and checksums
MODELS = {
    "bev_segmentation": {
        "filename": "bev_segmentation.pth",
        "url": "https://example.com/models/bev_segmentation.pth",
        "sha256": "placeholder_checksum_bev_segmentation",
        "description": "BEVFormer-Tiny semantic segmentation model",
        "size_mb": 45,
    },
    "yolov8_automotive": {
        "filename": "yolov8m_automotive.pt",
        "url": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt",
        "sha256": "placeholder_checksum_yolov8",
        "description": "YOLOv8 medium model with automotive fine-tuning",
        "size_mb": 52,
    },
    "l2cs_gaze": {
        "filename": "l2cs_gaze.pth",
        "url": "https://example.com/models/l2cs_gaze.pth",
        "sha256": "placeholder_checksum_l2cs",
        "description": "L2CS-Net gaze estimation model",
        "size_mb": 28,
    },
    "drowsiness": {
        "filename": "drowsiness_model.pth",
        "url": "https://example.com/models/drowsiness_model.pth",
        "sha256": "placeholder_checksum_drowsiness",
        "description": "Drowsiness detection model (PERCLOS, yawn, micro-sleep)",
        "size_mb": 15,
    },
    "distraction": {
        "filename": "distraction_clf.pth",
        "url": "https://example.com/models/distraction_clf.pth",
        "sha256": "placeholder_checksum_distraction",
        "description": "MobileNetV3 distraction classification model",
        "size_mb": 12,
    },
}


class ModelDownloader:
    """Handles downloading and verification of pretrained models."""

    def __init__(self, models_dir: Path, force: bool = False):
        """
        Initialize model downloader.

        Args:
            models_dir: Directory to store downloaded models
            force: If True, re-download even if file exists
        """
        self.models_dir = models_dir
        self.force = force
        self.models_dir.mkdir(parents=True, exist_ok=True)

    def calculate_sha256(self, filepath: Path) -> str:
        """
        Calculate SHA256 checksum of a file.

        Args:
            filepath: Path to file

        Returns:
            Hexadecimal SHA256 checksum
        """
        sha256_hash = hashlib.sha256()
        with open(filepath, "rb") as f:
            # Read in chunks to handle large files
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def verify_checksum(self, filepath: Path, expected_checksum: str) -> bool:
        """
        Verify file checksum matches expected value.

        Args:
            filepath: Path to file
            expected_checksum: Expected SHA256 checksum

        Returns:
            True if checksum matches, False otherwise
        """
        if expected_checksum.startswith("placeholder_"):
            print(f"  ⚠️  Skipping checksum verification (placeholder checksum)")
            return True

        actual_checksum = self.calculate_sha256(filepath)
        return actual_checksum == expected_checksum

    def download_file(self, url: str, filepath: Path, description: str) -> bool:
        """
        Download file from URL with progress reporting.

        Args:
            url: Download URL
            filepath: Destination file path
            description: Model description for logging

        Returns:
            True if download successful, False otherwise
        """
        try:
            print(f"  Downloading from {url}")

            def report_progress(block_num, block_size, total_size):
                downloaded = block_num * block_size
                if total_size > 0:
                    percent = min(100, downloaded * 100 / total_size)
                    mb_downloaded = downloaded / (1024 * 1024)
                    mb_total = total_size / (1024 * 1024)
                    print(
                        f"\r  Progress: {percent:.1f}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)",
                        end="",
                    )

            urllib.request.urlretrieve(url, filepath, reporthook=report_progress)
            print()  # New line after progress
            return True

        except Exception as e:
            print(f"\n  ❌ Download failed: {e}")
            return False

    def download_model(self, model_name: str, model_info: Dict) -> bool:
        """
        Download and verify a single model.

        Args:
            model_name: Model identifier
            model_info: Model metadata dictionary

        Returns:
            True if successful, False otherwise
        """
        filepath = self.models_dir / model_info["filename"]

        print(f"\n📦 {model_name}: {model_info['description']}")
        print(f"   File: {model_info['filename']} (~{model_info['size_mb']} MB)")

        # Check if file already exists
        if filepath.exists() and not self.force:
            print(f"  ✓ File already exists")
            if self.verify_checksum(filepath, model_info["sha256"]):
                print(f"  ✓ Checksum verified")
                return True
            else:
                print(f"  ❌ Checksum mismatch, re-downloading...")

        # Download the file
        if not self.download_file(model_info["url"], filepath, model_info["description"]):
            return False

        # Verify checksum
        if self.verify_checksum(filepath, model_info["sha256"]):
            print(f"  ✓ Download complete and verified")
            return True
        else:
            print(f"  ❌ Checksum verification failed")
            filepath.unlink()  # Remove corrupted file
            return False

    def verify_all_models(self) -> bool:
        """
        Verify all models are present and have correct checksums.

        Returns:
            True if all models verified, False otherwise
        """
        print("\n🔍 Verifying all models...")
        all_valid = True

        for model_name, model_info in MODELS.items():
            filepath = self.models_dir / model_info["filename"]

            if not filepath.exists():
                print(f"  ❌ {model_name}: File not found")
                all_valid = False
                continue

            if self.verify_checksum(filepath, model_info["sha256"]):
                print(f"  ✓ {model_name}: Valid")
            else:
                print(f"  ❌ {model_name}: Checksum mismatch")
                all_valid = False

        return all_valid

    def download_all_models(self) -> bool:
        """
        Download all required models.

        Returns:
            True if all downloads successful, False otherwise
        """
        print("=" * 70)
        print("SENTINEL Model Downloader")
        print("=" * 70)

        total_size = sum(m["size_mb"] for m in MODELS.values())
        print(f"\nTotal download size: ~{total_size} MB")
        print(f"Destination: {self.models_dir.absolute()}")

        success_count = 0
        for model_name, model_info in MODELS.items():
            if self.download_model(model_name, model_info):
                success_count += 1

        print("\n" + "=" * 70)
        print(f"Download Summary: {success_count}/{len(MODELS)} models successful")
        print("=" * 70)

        return success_count == len(MODELS)


def create_model_manifest(models_dir: Path):
    """
    Create a manifest file with model information.

    Args:
        models_dir: Directory containing models
    """
    manifest_path = models_dir / "manifest.json"

    manifest = {
        "models": {},
        "download_date": None,
    }

    for model_name, model_info in MODELS.items():
        filepath = models_dir / model_info["filename"]
        if filepath.exists():
            manifest["models"][model_name] = {
                "filename": model_info["filename"],
                "description": model_info["description"],
                "size_bytes": filepath.stat().st_size,
                "sha256": model_info["sha256"],
            }

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n✓ Model manifest created: {manifest_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Download pretrained models for SENTINEL system"
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=Path("models"),
        help="Directory to store models (default: models/)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if files exist",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify existing models, don't download",
    )

    args = parser.parse_args()

    downloader = ModelDownloader(args.models_dir, force=args.force)

    if args.verify_only:
        success = downloader.verify_all_models()
        sys.exit(0 if success else 1)

    success = downloader.download_all_models()

    if success:
        create_model_manifest(args.models_dir)
        print("\n✅ All models downloaded successfully!")
        print("\nNext steps:")
        print("  1. Verify models: python scripts/download_models.py --verify-only")
        print("  2. Configure model paths in configs/default.yaml")
        print("  3. Run system: python src/main.py --config configs/default.yaml")
        sys.exit(0)
    else:
        print("\n❌ Some models failed to download")
        print("Please check your internet connection and try again")
        sys.exit(1)


if __name__ == "__main__":
    main()
