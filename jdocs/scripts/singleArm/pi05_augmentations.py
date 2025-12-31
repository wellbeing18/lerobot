"""
Pi0.5 Training Augmentations

This module provides image augmentation transforms matching OpenPI's preprocessing.
OpenPI applies these augmentations during training to improve generalization:

Geometric augmentations (non-wrist cameras only):
- Random crop: 95% of image, then resize back to original
- Random rotation: -5 to +5 degrees

Color augmentations (all cameras):
- Brightness: [0.7, 1.3] range
- Contrast: [0.6, 1.4] range
- Saturation: [0.5, 1.5] range

Usage:
    from pi05_augmentations import Pi05TrainingAugmentor, patch_pi05_preprocessing

    # Option 1: Use standalone augmentor
    augmentor = Pi05TrainingAugmentor()
    augmented_images = augmentor(images, camera_keys)

    # Option 2: Patch LeRobot's Pi0.5 preprocessing (before creating policy)
    patch_pi05_preprocessing()
"""

import math
from typing import Sequence

import torch
import torch.nn.functional as F


class Pi05TrainingAugmentor:
    """
    Applies OpenPI-style training augmentations to images.

    Matches the augmentation pipeline from:
    openpi/models_pytorch/preprocessing_pytorch.py

    Args:
        crop_scale: Fraction of image to keep during crop (default: 0.95)
        rotation_degrees: Max rotation in degrees (default: 5.0)
        brightness_range: (min, max) brightness factor (default: (0.7, 1.3))
        contrast_range: (min, max) contrast factor (default: (0.6, 1.4))
        saturation_range: (min, max) saturation factor (default: (0.5, 1.5))
        wrist_camera_keywords: Keywords to identify wrist cameras (no geometric aug)
    """

    def __init__(
        self,
        crop_scale: float = 0.95,
        rotation_degrees: float = 5.0,
        brightness_range: tuple[float, float] = (0.7, 1.3),
        contrast_range: tuple[float, float] = (0.6, 1.4),
        saturation_range: tuple[float, float] = (0.5, 1.5),
        wrist_camera_keywords: Sequence[str] = ("wrist",),
    ):
        self.crop_scale = crop_scale
        self.rotation_degrees = rotation_degrees
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.saturation_range = saturation_range
        self.wrist_camera_keywords = wrist_camera_keywords

    def is_wrist_camera(self, camera_key: str) -> bool:
        """Check if camera is a wrist camera (no geometric augmentation)."""
        camera_key_lower = camera_key.lower()
        return any(kw in camera_key_lower for kw in self.wrist_camera_keywords)

    def __call__(
        self,
        images: torch.Tensor,
        camera_key: str = "",
        train: bool = True,
    ) -> torch.Tensor:
        """
        Apply augmentations to images.

        Args:
            images: Tensor of shape [B, H, W, C] or [B, C, H, W] in range [-1, 1]
            camera_key: Camera identifier (used to determine if wrist camera)
            train: If False, skip all augmentations

        Returns:
            Augmented images in same format as input
        """
        if not train:
            return images

        device = images.device
        dtype = images.dtype

        # Detect format
        is_channels_first = images.ndim == 4 and images.shape[1] == 3

        if is_channels_first:
            images = images.permute(0, 2, 3, 1)  # [B, C, H, W] -> [B, H, W, C]

        # Convert [-1, 1] to [0, 1] for augmentation
        images = images / 2.0 + 0.5

        # Apply geometric augmentations (non-wrist cameras only)
        if not self.is_wrist_camera(camera_key):
            images = self._apply_random_crop(images)
            images = self._apply_random_rotation(images)

        # Apply color augmentations (all cameras)
        images = self._apply_color_augmentations(images)

        # Clamp and convert back to [-1, 1]
        images = torch.clamp(images, 0, 1)
        images = images * 2.0 - 1.0

        if is_channels_first:
            images = images.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]

        return images.to(dtype=dtype, device=device)

    def _apply_random_crop(self, images: torch.Tensor) -> torch.Tensor:
        """
        Apply random crop and resize back to original size.

        Crops 95% of image (5% removed from edges) at random position,
        then resizes back to original dimensions.
        """
        batch_size, height, width, channels = images.shape
        device = images.device

        crop_height = int(height * self.crop_scale)
        crop_width = int(width * self.crop_scale)

        max_h = height - crop_height
        max_w = width - crop_width

        if max_h <= 0 or max_w <= 0:
            return images

        # Random crop offset (same for entire batch for simplicity)
        start_h = torch.randint(0, max_h + 1, (1,), device=device).item()
        start_w = torch.randint(0, max_w + 1, (1,), device=device).item()

        # Crop
        cropped = images[:, start_h:start_h + crop_height, start_w:start_w + crop_width, :]

        # Resize back to original size
        cropped = cropped.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]
        resized = F.interpolate(
            cropped,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )
        resized = resized.permute(0, 2, 3, 1)  # [B, C, H, W] -> [B, H, W, C]

        return resized

    def _apply_random_rotation(self, images: torch.Tensor) -> torch.Tensor:
        """
        Apply random rotation between -rotation_degrees and +rotation_degrees.

        Uses grid sampling with bilinear interpolation and zero padding.
        """
        batch_size, height, width, channels = images.shape
        device = images.device
        dtype = images.dtype

        # Random angle in degrees
        angle_deg = (torch.rand(1, device=device) * 2 - 1) * self.rotation_degrees

        # Skip if rotation is negligible
        if torch.abs(angle_deg) < 0.1:
            return images

        # Convert to radians
        angle_rad = angle_deg * math.pi / 180.0

        # Create rotation matrix
        cos_a = torch.cos(angle_rad)
        sin_a = torch.sin(angle_rad)

        # Create normalized grid coordinates
        grid_y = torch.linspace(-1, 1, height, device=device, dtype=dtype)
        grid_x = torch.linspace(-1, 1, width, device=device, dtype=dtype)
        grid_y, grid_x = torch.meshgrid(grid_y, grid_x, indexing="ij")

        # Apply rotation transformation
        # x' = x*cos - y*sin
        # y' = x*sin + y*cos
        rotated_x = grid_x * cos_a - grid_y * sin_a
        rotated_y = grid_x * sin_a + grid_y * cos_a

        # Stack and expand for batch
        grid = torch.stack([rotated_x, rotated_y], dim=-1)  # [H, W, 2]
        grid = grid.unsqueeze(0).expand(batch_size, -1, -1, -1)  # [B, H, W, 2]

        # Apply grid sampling
        images_chw = images.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]
        rotated = F.grid_sample(
            images_chw,
            grid,
            mode="bilinear",
            padding_mode="zeros",  # Black padding for rotated areas
            align_corners=False,
        )
        rotated = rotated.permute(0, 2, 3, 1)  # [B, C, H, W] -> [B, H, W, C]

        return rotated

    def _apply_color_augmentations(self, images: torch.Tensor) -> torch.Tensor:
        """
        Apply random brightness, contrast, and saturation augmentations.
        """
        device = images.device

        # Random brightness: multiply by factor in [0.7, 1.3]
        brightness_factor = (
            self.brightness_range[0] +
            torch.rand(1, device=device) * (self.brightness_range[1] - self.brightness_range[0])
        )
        images = images * brightness_factor

        # Random contrast: (x - mean) * factor + mean, factor in [0.6, 1.4]
        contrast_factor = (
            self.contrast_range[0] +
            torch.rand(1, device=device) * (self.contrast_range[1] - self.contrast_range[0])
        )
        mean = images.mean(dim=[1, 2, 3], keepdim=True)
        images = (images - mean) * contrast_factor + mean

        # Random saturation: gray + (color - gray) * factor, factor in [0.5, 1.5]
        saturation_factor = (
            self.saturation_range[0] +
            torch.rand(1, device=device) * (self.saturation_range[1] - self.saturation_range[0])
        )
        # Approximate grayscale via channel averaging
        gray = images.mean(dim=-1, keepdim=True)
        images = gray + (images - gray) * saturation_factor

        return images


# Global augmentor instance for patching
_augmentor = None


def get_augmentor() -> Pi05TrainingAugmentor:
    """Get the global augmentor instance."""
    global _augmentor
    if _augmentor is None:
        _augmentor = Pi05TrainingAugmentor()
    return _augmentor


def patch_pi05_preprocessing(augmentor: Pi05TrainingAugmentor | None = None):
    """
    Monkey-patch LeRobot's Pi0.5 preprocessing to add OpenPI-style augmentations.

    Call this BEFORE creating the Pi05Policy to enable training augmentations.

    Args:
        augmentor: Custom augmentor instance (uses default if None)

    Example:
        from pi05_augmentations import patch_pi05_preprocessing

        # Patch before training
        patch_pi05_preprocessing()

        # Then create policy and train as usual
        policy = PI05Policy(config)
    """
    global _augmentor
    if augmentor is not None:
        _augmentor = augmentor
    else:
        _augmentor = Pi05TrainingAugmentor()

    try:
        from lerobot.policies.pi05 import modeling_pi05
    except ImportError:
        print("ERROR: Could not import lerobot.policies.pi05.modeling_pi05")
        print("Make sure LeRobot is installed and in PYTHONPATH")
        return False

    # Save original method
    if not hasattr(modeling_pi05.PI05Policy, "_original_preprocess_images"):
        modeling_pi05.PI05Policy._original_preprocess_images = modeling_pi05.PI05Policy._preprocess_images

    def _patched_preprocess_images(self, batch):
        """
        Patched preprocessing that adds OpenPI-style augmentations during training.
        """
        # Get original preprocessed images
        images, img_masks = self._original_preprocess_images(batch)

        # Apply augmentations if training
        if self.training:
            aug = get_augmentor()
            augmented_images = []

            # Get camera keys from batch
            camera_keys = [k for k in batch.keys() if k.startswith("observation.images.")]

            for i, img in enumerate(images):
                camera_key = camera_keys[i] if i < len(camera_keys) else ""
                augmented = aug(img, camera_key=camera_key, train=True)
                augmented_images.append(augmented)

            return augmented_images, img_masks

        return images, img_masks

    # Apply patch
    modeling_pi05.PI05Policy._preprocess_images = _patched_preprocess_images

    print("Pi0.5 preprocessing patched with OpenPI-style augmentations:")
    print(f"  - Crop scale: {_augmentor.crop_scale} (95% of image)")
    print(f"  - Rotation: +/- {_augmentor.rotation_degrees} degrees")
    print(f"  - Brightness: {_augmentor.brightness_range}")
    print(f"  - Contrast: {_augmentor.contrast_range}")
    print(f"  - Saturation: {_augmentor.saturation_range}")
    print(f"  - Wrist camera keywords (no geometric aug): {_augmentor.wrist_camera_keywords}")

    return True


def unpatch_pi05_preprocessing():
    """
    Remove the augmentation patch from Pi0.5 preprocessing.
    """
    try:
        from lerobot.policies.pi05 import modeling_pi05
    except ImportError:
        return False

    if hasattr(modeling_pi05.PI05Policy, "_original_preprocess_images"):
        modeling_pi05.PI05Policy._preprocess_images = modeling_pi05.PI05Policy._original_preprocess_images
        delattr(modeling_pi05.PI05Policy, "_original_preprocess_images")
        print("Pi0.5 preprocessing patch removed")
        return True

    return False


# Test function
def _test_augmentor():
    """Test augmentation transforms."""
    print("Testing Pi05TrainingAugmentor...")

    # Create test image batch [B, H, W, C] in [-1, 1]
    images = torch.rand(2, 224, 224, 3) * 2 - 1

    augmentor = Pi05TrainingAugmentor()

    # Test with non-wrist camera (full augmentation)
    augmented = augmentor(images, camera_key="head", train=True)
    print(f"  Non-wrist camera: input shape {images.shape}, output shape {augmented.shape}")
    print(f"  Value range: [{augmented.min():.3f}, {augmented.max():.3f}]")

    # Test with wrist camera (color only)
    augmented_wrist = augmentor(images, camera_key="left_wrist", train=True)
    print(f"  Wrist camera: input shape {images.shape}, output shape {augmented_wrist.shape}")

    # Test with train=False (no augmentation)
    no_aug = augmentor(images, camera_key="head", train=False)
    assert torch.allclose(images, no_aug), "train=False should not modify images"
    print("  train=False: images unchanged (correct)")

    print("All tests passed!")


if __name__ == "__main__":
    _test_augmentor()
