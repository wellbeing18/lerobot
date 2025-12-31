#!/usr/bin/env python3
"""
Pi0.5 Training with OpenPI-style Augmentations

This wrapper applies OpenPI-style image augmentations before running LeRobot training.
It patches the Pi0.5 preprocessing to add augmentations during training.

Usage:
    python train_pi05_with_augmentations.py [lerobot training args...]

Example:
    python train_pi05_with_augmentations.py \
        --dataset.repo_id=pick_and_place \
        --policy.type=pi05 \
        --policy.use_lora=true \
        --steps=3000

Environment Variables:
    ENABLE_AUGMENTATIONS: Set to "false" to disable augmentations (default: "true")
    AUG_CROP_SCALE: Crop scale (default: 0.95)
    AUG_ROTATION_DEGREES: Max rotation degrees (default: 5.0)
    AUG_BRIGHTNESS_MIN: Min brightness factor (default: 0.7)
    AUG_BRIGHTNESS_MAX: Max brightness factor (default: 1.3)
    AUG_CONTRAST_MIN: Min contrast factor (default: 0.6)
    AUG_CONTRAST_MAX: Max contrast factor (default: 1.4)
    AUG_SATURATION_MIN: Min saturation factor (default: 0.5)
    AUG_SATURATION_MAX: Max saturation factor (default: 1.5)
"""

import os
import sys


def main():
    # Check if augmentations should be enabled
    enable_augmentations = os.environ.get("ENABLE_AUGMENTATIONS", "true").lower() == "true"

    if enable_augmentations:
        # Import and configure augmentor
        try:
            from pi05_augmentations import Pi05TrainingAugmentor, patch_pi05_preprocessing
        except ImportError:
            # Try adding script directory to path
            script_dir = os.path.dirname(os.path.abspath(__file__))
            sys.path.insert(0, script_dir)
            from pi05_augmentations import Pi05TrainingAugmentor, patch_pi05_preprocessing

        # Create augmentor with configurable parameters from environment
        augmentor = Pi05TrainingAugmentor(
            crop_scale=float(os.environ.get("AUG_CROP_SCALE", "0.95")),
            rotation_degrees=float(os.environ.get("AUG_ROTATION_DEGREES", "5.0")),
            brightness_range=(
                float(os.environ.get("AUG_BRIGHTNESS_MIN", "0.7")),
                float(os.environ.get("AUG_BRIGHTNESS_MAX", "1.3")),
            ),
            contrast_range=(
                float(os.environ.get("AUG_CONTRAST_MIN", "0.6")),
                float(os.environ.get("AUG_CONTRAST_MAX", "1.4")),
            ),
            saturation_range=(
                float(os.environ.get("AUG_SATURATION_MIN", "0.5")),
                float(os.environ.get("AUG_SATURATION_MAX", "1.5")),
            ),
        )

        # Apply patch
        print("=" * 60)
        print("Applying OpenPI-style augmentations to Pi0.5 preprocessing")
        print("=" * 60)
        success = patch_pi05_preprocessing(augmentor)
        if not success:
            print("WARNING: Failed to patch Pi0.5 preprocessing")
            print("Training will continue without augmentations")
        print()
    else:
        print("Augmentations disabled (ENABLE_AUGMENTATIONS=false)")

    # Import and run LeRobot training
    from lerobot.scripts.lerobot_train import main as lerobot_main

    # Pass through command line arguments
    sys.exit(lerobot_main())


if __name__ == "__main__":
    main()
