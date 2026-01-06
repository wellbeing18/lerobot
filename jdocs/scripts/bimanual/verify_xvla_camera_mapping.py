#!/usr/bin/env python3
"""
XVLA Camera Mapping Verification Script

This script verifies that the camera rename mapping for XVLA training is correct
by saving side-by-side comparison images from the dataset.

The script:
1. Loads images from the multitasks dataset with original camera names
2. Applies the XVLA rename mapping
3. Saves comparison images to outputs/xvla_camera_verification/
4. Shows which original camera maps to which XVLA key

Usage:
    python verify_xvla_camera_mapping.py
    python verify_xvla_camera_mapping.py --num-samples 5

Camera Mapping for XVLA:
    Original (dataset)              -> XVLA Expected
    observation.images.head         -> observation.images.image
    observation.images.left_wrist   -> observation.images.image2
    observation.images.right_wrist  -> observation.images.image3
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

# Add project src to path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Default paths
DEFAULT_DATASET_PATH = PROJECT_ROOT / "datasets_bimanuel" / "multitasks"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "xvla_camera_verification"

# Camera mapping configurations
ORIGINAL_CAMERAS = [
    "observation.images.head",
    "observation.images.left_wrist",
    "observation.images.right_wrist",
]

XVLA_RENAME_MAP = {
    "observation.images.head": "observation.images.image",
    "observation.images.left_wrist": "observation.images.image2",
    "observation.images.right_wrist": "observation.images.image3",
}

SMOLVLA_RENAME_MAP = {
    "observation.images.head": "observation.images.camera1",
    "observation.images.left_wrist": "observation.images.camera2",
    "observation.images.right_wrist": "observation.images.camera3",
}


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convert tensor (C, H, W) in [0, 1] to PIL Image."""
    if tensor.ndim == 4:
        tensor = tensor[0]  # Take first in batch
    if tensor.ndim == 3 and tensor.shape[0] in [1, 3]:
        # (C, H, W) -> (H, W, C)
        tensor = tensor.permute(1, 2, 0)

    # Convert to numpy
    arr = tensor.cpu().numpy()

    # Scale to 0-255 if needed
    if arr.max() <= 1.0:
        arr = (arr * 255).astype(np.uint8)
    else:
        arr = arr.astype(np.uint8)

    return Image.fromarray(arr)


def add_label_to_image(img: Image.Image, label: str, position: str = "top") -> Image.Image:
    """Add a text label to an image."""
    # Create a copy
    img = img.copy()
    draw = ImageDraw.Draw(img)

    # Try to use a font, fall back to default
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    except:
        font = ImageFont.load_default()

    # Get text size
    bbox = draw.textbbox((0, 0), label, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    # Position
    if position == "top":
        x = (img.width - text_width) // 2
        y = 5
    else:
        x = (img.width - text_width) // 2
        y = img.height - text_height - 5

    # Draw background rectangle
    padding = 3
    draw.rectangle(
        [x - padding, y - padding, x + text_width + padding, y + text_height + padding],
        fill="black"
    )

    # Draw text
    draw.text((x, y), label, fill="white", font=font)

    return img


def create_comparison_grid(images: dict, episode_idx: int, frame_idx: int) -> Image.Image:
    """Create a comparison grid showing original -> renamed mapping."""

    # Get individual images
    imgs = []
    labels = []

    for orig_key in ORIGINAL_CAMERAS:
        if orig_key in images:
            img = tensor_to_pil(images[orig_key])
            xvla_key = XVLA_RENAME_MAP[orig_key]
            smolvla_key = SMOLVLA_RENAME_MAP[orig_key]

            # Create label showing mapping
            orig_name = orig_key.split(".")[-1]  # e.g., "head"
            xvla_name = xvla_key.split(".")[-1]  # e.g., "image"

            label = f"{orig_name} -> XVLA:{xvla_name}"
            img = add_label_to_image(img, label, "top")

            imgs.append(img)
            labels.append(orig_name)

    if not imgs:
        return None

    # Create grid (1 row, 3 columns)
    img_width = imgs[0].width
    img_height = imgs[0].height

    grid_width = img_width * len(imgs) + 20 * (len(imgs) - 1)  # 20px spacing
    grid_height = img_height + 40  # Extra space for title

    grid = Image.new("RGB", (grid_width, grid_height), color="white")
    draw = ImageDraw.Draw(grid)

    # Add title
    title = f"Episode {episode_idx}, Frame {frame_idx} - Camera Mapping Verification"
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
    except:
        font = ImageFont.load_default()

    bbox = draw.textbbox((0, 0), title, font=font)
    title_width = bbox[2] - bbox[0]
    draw.text(((grid_width - title_width) // 2, 5), title, fill="black", font=font)

    # Paste images
    x_offset = 0
    for img in imgs:
        grid.paste(img, (x_offset, 30))
        x_offset += img_width + 20

    return grid


def verify_camera_mapping(dataset_path: Path, num_samples: int = 3):
    """Verify camera mapping by saving comparison images."""

    print("=" * 70)
    print("XVLA CAMERA MAPPING VERIFICATION")
    print("=" * 70)
    print(f"\nDataset: {dataset_path}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"\nCamera Mapping:")
    for orig, xvla in XVLA_RENAME_MAP.items():
        print(f"  {orig.split('.')[-1]:15} -> {xvla.split('.')[-1]}")
    print()

    # Load dataset
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    print("Loading dataset...")
    dataset = LeRobotDataset(
        repo_id="multitasks",
        root=str(dataset_path),
        video_backend="pyav",
    )

    print(f"  Total frames: {len(dataset)}")
    print(f"  Total episodes: {dataset.meta.total_episodes}")

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Get samples from different parts of the dataset
    total_frames = len(dataset)
    frame_indices = np.linspace(0, total_frames - 1, min(num_samples, total_frames), dtype=int)

    print(f"\nSaving {len(frame_indices)} verification images...")

    saved_files = []

    for i, frame_idx in enumerate(frame_indices):
        # Load the frame
        sample = dataset[frame_idx]

        # Get episode index and task string
        ep_idx = sample.get("episode_index", torch.tensor(0)).item()
        task = sample.get("task", "Unknown task")

        # Create comparison grid
        grid = create_comparison_grid(sample, ep_idx, frame_idx)

        if grid:
            # Save the grid
            filename = f"camera_mapping_ep{ep_idx:03d}_frame{frame_idx:05d}.png"
            filepath = OUTPUT_DIR / filename
            grid.save(filepath)
            saved_files.append(filepath)
            print(f"  Saved: {filename}")
            print(f"    Task: {task[:60]}...")

    # Also save individual camera images for detailed inspection
    print("\nSaving individual camera images from first sample...")
    sample = dataset[0]

    for orig_key in ORIGINAL_CAMERAS:
        if orig_key in sample:
            img = tensor_to_pil(sample[orig_key])
            orig_name = orig_key.split(".")[-1]
            xvla_name = XVLA_RENAME_MAP[orig_key].split(".")[-1]

            # Save with both names for clarity
            filename = f"camera_{orig_name}_to_xvla_{xvla_name}.png"
            filepath = OUTPUT_DIR / filename

            # Add detailed label
            label = f"Original: {orig_name}\nXVLA: {xvla_name}\nSmolVLA: {SMOLVLA_RENAME_MAP[orig_key].split('.')[-1]}"
            img_labeled = add_label_to_image(img, f"{orig_name} -> {xvla_name}", "top")
            img_labeled.save(filepath)
            print(f"  Saved: {filename}")

    print("\n" + "=" * 70)
    print("VERIFICATION COMPLETE")
    print("=" * 70)
    print(f"\nImages saved to: {OUTPUT_DIR}")
    print("\nPlease visually verify:")
    print("  1. 'head' camera shows the overhead/front view")
    print("  2. 'left_wrist' camera shows the left arm's gripper view")
    print("  3. 'right_wrist' camera shows the right arm's gripper view")
    print("\nIf mapping is correct, the XVLA training should work properly.")
    print(f"\nTo view images: ls {OUTPUT_DIR}/")

    return saved_files


def main():
    parser = argparse.ArgumentParser(
        description="Verify XVLA camera mapping",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=str(DEFAULT_DATASET_PATH),
        help=f"Path to dataset (default: {DEFAULT_DATASET_PATH})"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=3,
        help="Number of sample images to save (default: 3)"
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset_path)

    if not dataset_path.exists():
        print(f"ERROR: Dataset not found: {dataset_path}")
        sys.exit(1)

    verify_camera_mapping(dataset_path, args.num_samples)


if __name__ == "__main__":
    main()
