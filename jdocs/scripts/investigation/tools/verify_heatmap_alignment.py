#!/usr/bin/env python3
"""
Verify heatmap overlay alignment by creating test patterns.
This helps debug whether spatial attention maps are correctly aligned to images.
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

def create_attention_heatmap_overlay(
    image: np.ndarray,
    attention_map: np.ndarray,
    alpha: float = 0.5,
) -> np.ndarray:
    """Same function from cross_attention_capture.py"""
    h, w = image.shape[:2]
    attn_resized = cv2.resize(attention_map.astype(np.float32), (w, h))
    heatmap = plt.cm.jet(attn_resized)[:, :, :3]
    heatmap = (heatmap * 255).astype(np.uint8)
    overlay = cv2.addWeighted(image, 1 - alpha, heatmap, alpha, 0)
    return overlay

def verify_alignment():
    """Create test cases to verify heatmap alignment."""

    # Create a test image with clear quadrants
    h, w = 480, 640
    test_img = np.zeros((h, w, 3), dtype=np.uint8)

    # Mark quadrants with different colors
    test_img[:h//2, :w//2] = [255, 0, 0]      # Top-left: Blue
    test_img[:h//2, w//2:] = [0, 255, 0]      # Top-right: Green
    test_img[h//2:, :w//2] = [0, 0, 255]      # Bottom-left: Red
    test_img[h//2:, w//2:] = [255, 255, 0]    # Bottom-right: Cyan

    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(test_img, "TL", (50, 100), font, 2, (255, 255, 255), 3)
    cv2.putText(test_img, "TR", (400, 100), font, 2, (255, 255, 255), 3)
    cv2.putText(test_img, "BL", (50, 350), font, 2, (255, 255, 255), 3)
    cv2.putText(test_img, "BR", (400, 350), font, 2, (255, 255, 255), 3)

    # Test Case 1: Attention on TOP-LEFT (should overlay on blue quadrant)
    grid_size = 8
    attn_tl = np.zeros((grid_size, grid_size), dtype=np.float32)
    attn_tl[:grid_size//2, :grid_size//2] = 1.0  # Top-left quadrant of grid

    # Test Case 2: Attention on BOTTOM-RIGHT
    attn_br = np.zeros((grid_size, grid_size), dtype=np.float32)
    attn_br[grid_size//2:, grid_size//2:] = 1.0  # Bottom-right quadrant of grid

    # Test Case 3: Attention on CENTER
    attn_center = np.zeros((grid_size, grid_size), dtype=np.float32)
    attn_center[3:5, 3:5] = 1.0  # Center of grid

    # Test Case 4: Single patch in TOP-RIGHT corner (index [0, grid_size-1])
    attn_corner = np.zeros((grid_size, grid_size), dtype=np.float32)
    attn_corner[0, grid_size-1] = 1.0  # Should be TOP-RIGHT

    test_cases = [
        ("top_left", attn_tl, "Attention on TOP-LEFT quadrant"),
        ("bottom_right", attn_br, "Attention on BOTTOM-RIGHT quadrant"),
        ("center", attn_center, "Attention on CENTER"),
        ("top_right_corner", attn_corner, "Single patch at [0, 7] = TOP-RIGHT corner"),
    ]

    # Generate outputs
    output_dir = Path("logs/analysis/heatmap_verification")
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))

    for idx, (name, attn, desc) in enumerate(test_cases):
        ax1 = axes[0, idx]
        ax2 = axes[1, idx]

        # Show attention grid
        ax1.imshow(attn, cmap='hot', vmin=0, vmax=1)
        ax1.set_title(f"Attention Grid\n{desc}")
        ax1.set_xlabel("Grid column (0=left, 7=right)")
        ax1.set_ylabel("Grid row (0=top, 7=bottom)")

        # Show overlay
        overlay = create_attention_heatmap_overlay(test_img, attn, alpha=0.6)
        ax2.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        ax2.set_title(f"Overlay on Test Image")
        ax2.axis('off')

    plt.suptitle("Heatmap Alignment Verification\nGrid [0,0]=top-left should map to image top-left", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / "alignment_test.png", dpi=150)
    print(f"Saved to: {output_dir / 'alignment_test.png'}")

    # Also save the test image for reference
    cv2.imwrite(str(output_dir / "test_image.png"), test_img)
    print(f"Saved test image to: {output_dir / 'test_image.png'}")

    print("\n=== VERIFICATION GUIDE ===")
    print("Check the generated image:")
    print("- 'top_left' attention should highlight BLUE (TL) quadrant")
    print("- 'bottom_right' attention should highlight CYAN (BR) quadrant")
    print("- 'top_right_corner' attention should highlight GREEN (TR) quadrant")
    print("\nIf these don't match, there's a bug in the spatial mapping!")

if __name__ == "__main__":
    verify_alignment()
