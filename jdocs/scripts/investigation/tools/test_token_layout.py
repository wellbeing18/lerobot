#!/usr/bin/env python3
"""Quick test to verify token layout detection with 3 cameras."""

import numpy as np

# Token layout constants (copied from cross_attention_capture.py)
PATCHES_PER_CAMERA = 64  # 8x8 grid per camera (VERIFIED)
PATCH_GRID_SIZE = 8  # sqrt(64)
NUM_CAMERAS = 3  # head, left_wrist, right_wrist
NUM_ACTION_TOKENS = 50  # Chunk size

def infer_token_layout(prefix_len: int, debug: bool = False) -> dict:
    """
    Infer token layout from actual prefix length.

    IMPORTANT: The attention matrix key dimension may include action tokens for self-attention.
    If key_len > 250, it likely includes 50 action tokens that should be excluded.
    """
    # If key_len includes action tokens (50), remove them first
    actual_prefix_len = prefix_len
    if prefix_len > 250:
        actual_prefix_len = prefix_len - NUM_ACTION_TOKENS  # Remove 50 action tokens
        if debug:
            print(f"    Detected action tokens in key: {prefix_len} - 50 = {actual_prefix_len} prefix tokens")

    # Estimate language+state tokens at end (~49)
    estimated_lang_state = 49  # 48 language + 1 state

    total_image_tokens = actual_prefix_len - estimated_lang_state

    # Determine number of cameras based on actual prefix length
    # 3 cameras with 64 patches each = 192 image tokens → ~241 total prefix
    if 180 <= total_image_tokens <= 210:  # Around 192 = 3 cameras × 64
        num_cameras = 3
        patches_per_camera = total_image_tokens // 3  # ~64
    elif total_image_tokens > 210:  # More tokens = likely 1 camera with more patches
        num_cameras = 1
        patches_per_camera = total_image_tokens
    else:
        # Fallback: assume single camera
        num_cameras = 1
        patches_per_camera = max(total_image_tokens, 64)

    grid_size = int(np.sqrt(patches_per_camera))

    layout = {
        "num_cameras": num_cameras,
        "patches_per_camera": patches_per_camera,
        "grid_size": grid_size,
        "total_image_tokens": total_image_tokens,
        "lang_state_tokens": estimated_lang_state,
        "actual_prefix_len": actual_prefix_len,
        "raw_key_len": prefix_len,
    }

    if num_cameras == 3:
        layout["head_start"] = 0
        layout["head_end"] = patches_per_camera
        layout["left_start"] = patches_per_camera
        layout["left_end"] = 2 * patches_per_camera
        layout["right_start"] = 2 * patches_per_camera
        layout["right_end"] = 3 * patches_per_camera
        layout["lang_start"] = 3 * patches_per_camera
        layout["lang_end"] = actual_prefix_len - 1
        layout["state_idx"] = actual_prefix_len - 1

    if debug:
        print(f"    Key length: {prefix_len}, actual prefix: {actual_prefix_len}, inferred {num_cameras} cameras")
        print(f"    Image tokens: {total_image_tokens}, patches per camera: {patches_per_camera} ({grid_size}x{grid_size} grid)")
        if num_cameras == 3:
            print(f"    Head=[0:{layout['head_end']}], Left=[{layout['left_start']}:{layout['left_end']}], Right=[{layout['right_start']}:{layout['right_end']}]")
            print(f"    Lang=[{layout['lang_start']}:{layout['lang_end']}], State=[{layout['state_idx']}]")

    return layout


if __name__ == "__main__":
    print("=" * 60)
    print("Token Layout Detection Tests")
    print("=" * 60)

    test_cases = [
        (291, "key_len=291 (from actual inference: 241 prefix + 50 action)"),
        (241, "prefix_len=241 (3 cameras: 192 img + 48 lang + 1 state)"),
        (778, "prefix_len=778 (legacy single camera estimate)"),
        (200, "prefix_len=200 (small prefix)"),
    ]

    for key_len, desc in test_cases:
        print(f"\nTest: {desc}")
        print("-" * 50)
        layout = infer_token_layout(key_len, debug=True)
        print(f"  Result: {layout['num_cameras']} cameras detected")
        if layout['num_cameras'] == 3:
            print(f"  ✓ Per-camera breakdown will be computed")
        else:
            print(f"  ✗ Single camera mode (per_camera will be None)")
