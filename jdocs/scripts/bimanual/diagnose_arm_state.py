#!/usr/bin/env python3
"""
Diagnostic script to verify arm port assignments and calibration.

This script helps diagnose the issue where left and right arms might be swapped
or calibration might be incorrect.

Usage:
    python diagnose_arm_state.py

What it does:
1. Connects to both arms using the inference script's configuration
2. Reads current state from each arm
3. Compares with expected training starting positions
4. Helps identify if arms are swapped or calibration is wrong
"""

import sys
from pathlib import Path

# Add project src to path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import yaml


def load_hardware_config():
    """Load hardware configuration from YAML file."""
    config_path = SCRIPT_DIR / "bimanual_so101_hardware.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def main():
    print("=" * 70)
    print("BIMANUAL ARM DIAGNOSTIC")
    print("=" * 70)
    print()

    hw_config = load_hardware_config()
    robot_config = hw_config.get("robot", {})

    left_port = robot_config.get("left_arm", {}).get("port", "/dev/ttyACM3")
    right_port = robot_config.get("right_arm", {}).get("port", "/dev/ttyACM2")
    left_arm_id = robot_config.get("left_arm_id", "xlerobot_left_arm")
    right_arm_id = robot_config.get("right_arm_id", "xlerobot_right_arm")

    print(f"Configuration from YAML:")
    print(f"  Left arm port:  {left_port}")
    print(f"  Right arm port: {right_port}")
    print(f"  Left arm ID:    {left_arm_id}")
    print(f"  Right arm ID:   {right_arm_id}")
    print()

    # Expected starting positions from training data analysis
    print("Expected positions (from training data, FOLDED home position):")
    print("  Left arm:  shoulder_lift ~ -98°, elbow ~ 99°")
    print("  Right arm: shoulder_lift ~ -99°, elbow ~ 99°")
    print()

    # Connect to robot
    print("Connecting to robot...")
    try:
        from lerobot.robots.bi_so101_follower.config_bi_so101_follower import BiSO101FollowerConfig
        from lerobot.robots.bi_so101_follower.bi_so101_follower import BiSO101Follower

        config = BiSO101FollowerConfig(
            id="diagnostic_test",
            left_arm_port=left_port,
            right_arm_port=right_port,
            left_arm_id=left_arm_id,
            right_arm_id=right_arm_id,
            left_arm_use_degrees=True,
            right_arm_use_degrees=True,
        )

        robot = BiSO101Follower(config)
        robot.connect(calibrate=False)

        print("  Connected successfully!")
        print()

        # Read state
        obs = robot.get_observation()

        # Extract motor positions
        motor_names = [
            "left_shoulder_pan", "left_shoulder_lift", "left_elbow_flex",
            "left_wrist_flex", "left_wrist_roll", "left_gripper",
            "right_shoulder_pan", "right_shoulder_lift", "right_elbow_flex",
            "right_wrist_flex", "right_wrist_roll", "right_gripper",
        ]

        state = np.array([obs[f"{name}.pos"] for name in motor_names], dtype=np.float32)

        print("=" * 70)
        print("CURRENT STATE (degrees):")
        print("=" * 70)
        print()

        left_state = state[:6]
        right_state = state[6:]

        print(f"LEFT arm (port {left_port}):")
        for i, name in enumerate(motor_names[:6]):
            short_name = name.replace("left_", "")
            print(f"  {short_name:15s}: {left_state[i]:8.1f}°")
        print()

        print(f"RIGHT arm (port {right_port}):")
        for i, name in enumerate(motor_names[6:]):
            short_name = name.replace("right_", "")
            print(f"  {short_name:15s}: {right_state[i]:8.1f}°")
        print()

        # Analysis
        print("=" * 70)
        print("ANALYSIS:")
        print("=" * 70)
        print()

        # Check if left arm looks folded (shoulder_lift near -98, elbow near 99)
        left_folded = abs(left_state[1] - (-98)) < 20 and abs(left_state[2] - 99) < 20
        right_folded = abs(right_state[1] - (-99)) < 20 and abs(right_state[2] - 99) < 20

        # Check if values are swapped
        left_looks_like_right = abs(left_state[1] - (-99)) < 20 and abs(left_state[2] - 99) < 20
        right_looks_like_left = abs(right_state[1] - (-98)) < 20 and abs(right_state[2] - 99) < 20

        if left_folded and right_folded:
            print("✓ BOTH arms appear to be in FOLDED home position (matching training)")
            print("  This is the expected starting position!")
        elif not left_folded and right_folded:
            print("⚠ LEFT arm does NOT look folded, but RIGHT arm does")
            print()
            print("  Possible causes:")
            print("  1. Left arm is physically in a different position")
            print("  2. USB ports are SWAPPED (left arm hardware on right port)")
            print("  3. Calibration file for left arm is wrong/different")
            print()
            print("  To diagnose:")
            print("  - Visually check: Is the left arm PHYSICALLY folded?")
            print("  - If yes, the issue is software (ports or calibration)")
            print("  - If no, move the left arm to home position")
        elif left_folded and not right_folded:
            print("⚠ RIGHT arm does NOT look folded, but LEFT arm does")
        else:
            print("⚠ NEITHER arm appears to be in folded home position")
            print("  Expected: shoulder_lift ~ -98°, elbow ~ 99°")

        print()
        print("=" * 70)
        print("MANUAL VERIFICATION:")
        print("=" * 70)
        print()
        print("Please VISUALLY verify the following:")
        print()
        print("1. Is the LEFT arm PHYSICALLY in a folded/home position?")
        print("   - Upper arm pointing up, forearm folded back")
        print()
        print("2. Which physical arm moves when you run:")
        print("   - If you see unexpected arm moving, ports are likely swapped")
        print()

        # Disconnect
        robot.disconnect()

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
