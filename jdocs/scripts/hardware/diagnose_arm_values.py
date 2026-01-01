#!/usr/bin/env python3
"""
Diagnostic script to compare arm positions between:
1. Current hardware readings
2. Training dataset values

This helps diagnose if ports are swapped or calibration is wrong.
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import pyarrow.parquet as pq

from lerobot.robots.so101_follower import SO101Follower
from lerobot.robots.so101_follower.config_so101_follower import SO101FollowerConfig


def get_dataset_first_frame_values(dataset_path: str) -> dict:
    """Get observation.state values from the first frame of each episode."""
    dataset_path = Path(dataset_path)

    # Find all parquet files
    parquet_files = sorted(dataset_path.glob("data/**/*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {dataset_path}")

    # Read first parquet file
    table = pq.read_table(parquet_files[0])
    df = table.to_pandas()

    # Get first frames of first 5 episodes
    first_frames = []
    for ep_idx in range(min(5, df['episode_index'].max() + 1)):
        ep_data = df[df['episode_index'] == ep_idx]
        if len(ep_data) > 0:
            first_frame = ep_data.iloc[0]
            state = first_frame['observation.state']
            first_frames.append(state)

    return {
        "first_frames": first_frames,
        "mean": np.mean(first_frames, axis=0) if first_frames else None,
    }


def read_current_arm_position(port: str, arm_id: str) -> dict:
    """Read current position from a single arm."""
    import os
    # Disable calibration prompt by setting env var
    os.environ["LEROBOT_CALIBRATION_MODE"] = "skip"

    config = SO101FollowerConfig(
        id=arm_id,
        port=port,
        use_degrees=True,
    )

    arm = SO101Follower(config)
    try:
        # Connect with calibrate=False to avoid prompt, then check if calibration exists
        arm.connect(calibrate=False)
        if arm.calibration is None:
            # Try to load calibration manually
            from lerobot.motors.calibration import Calibration
            calib_path = arm.calibration_fpath
            if calib_path.exists():
                arm.calibration = Calibration.load(calib_path)
        obs = arm.get_observation()

        # Extract position values
        positions = []
        motor_names = []
        for key, value in sorted(obs.items()):
            if key.endswith('.pos'):
                positions.append(value)
                motor_names.append(key)

        return {
            "positions": positions,
            "motor_names": motor_names,
            "raw_obs": obs,
        }
    finally:
        arm.disconnect()


def main():
    import yaml

    # Load config
    config_path = PROJECT_ROOT / "jdocs/configs/hardware/xlerobot_bimanual.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    robot_cfg = config.get("robot", {})
    left_port = robot_cfg.get("left_arm", {}).get("port", "/dev/ttyACM3")
    right_port = robot_cfg.get("right_arm", {}).get("port", "/dev/ttyACM2")
    left_id = robot_cfg.get("left_arm", {}).get("id", "xlerobot_left_arm")
    right_id = robot_cfg.get("right_arm", {}).get("id", "xlerobot_right_arm")

    print("=" * 70)
    print("ARM POSITION DIAGNOSTIC")
    print("=" * 70)
    print(f"\nConfig: {config_path}")
    print(f"Left arm:  port={left_port}, id={left_id}")
    print(f"Right arm: port={right_port}, id={right_id}")

    # Read dataset first frame values
    print("\n" + "-" * 70)
    print("DATASET FIRST FRAME VALUES (expected home position)")
    print("-" * 70)

    dataset_path = PROJECT_ROOT / "datasets_bimanuel/bimanual/combined_pick_and_place"
    if dataset_path.exists():
        try:
            dataset_vals = get_dataset_first_frame_values(dataset_path)
            if dataset_vals["first_frames"]:
                print(f"\nFirst frame of episode 0:")
                state = dataset_vals["first_frames"][0]
                joint_names = [
                    "left_shoulder_pan", "left_shoulder_lift", "left_elbow_flex",
                    "left_wrist_flex", "left_wrist_roll", "left_gripper",
                    "right_shoulder_pan", "right_shoulder_lift", "right_elbow_flex",
                    "right_wrist_flex", "right_wrist_roll", "right_gripper"
                ]
                print("\n  LEFT ARM (indices 0-5):")
                for i in range(6):
                    print(f"    {joint_names[i]:25s}: {state[i]:8.2f}°")
                print("\n  RIGHT ARM (indices 6-11):")
                for i in range(6, 12):
                    print(f"    {joint_names[i]:25s}: {state[i]:8.2f}°")
        except Exception as e:
            print(f"  Error reading dataset: {e}")
    else:
        print(f"  Dataset not found: {dataset_path}")

    # Read current arm positions
    print("\n" + "-" * 70)
    print("CURRENT HARDWARE READINGS")
    print("-" * 70)

    print(f"\n  Reading LEFT arm from {left_port}...")
    try:
        left_result = read_current_arm_position(left_port, left_id)
        print(f"  LEFT ARM ({left_port}):")
        for name, pos in zip(left_result["motor_names"], left_result["positions"]):
            print(f"    {name:25s}: {pos:8.2f}°")
    except Exception as e:
        print(f"  Error reading left arm: {e}")
        left_result = None

    print(f"\n  Reading RIGHT arm from {right_port}...")
    try:
        right_result = read_current_arm_position(right_port, right_id)
        print(f"  RIGHT ARM ({right_port}):")
        for name, pos in zip(right_result["motor_names"], right_result["positions"]):
            print(f"    {name:25s}: {pos:8.2f}°")
    except Exception as e:
        print(f"  Error reading right arm: {e}")
        right_result = None

    # Compare hardware vs dataset
    print("\n" + "-" * 70)
    print("COMPARISON: Hardware vs Dataset Episode 0 First Frame")
    print("-" * 70)

    if dataset_vals.get("first_frames") and left_result and right_result:
        state = dataset_vals["first_frames"][0]

        # Dataset left arm values (indices 0-5)
        dataset_left = state[:6]
        # Dataset right arm values (indices 6-11)
        dataset_right = state[6:]

        # Hardware values
        hw_left = left_result["positions"]
        hw_right = right_result["positions"]

        print("\n  LEFT ARM comparison (hardware_left vs dataset_left):")
        for i, (hw, ds) in enumerate(zip(hw_left, dataset_left)):
            diff = hw - ds
            flag = " <-- MISMATCH!" if abs(diff) > 20 else ""
            print(f"    Joint {i}: hw={hw:8.2f}°, dataset={ds:8.2f}°, diff={diff:+8.2f}°{flag}")

        print("\n  RIGHT ARM comparison (hardware_right vs dataset_right):")
        for i, (hw, ds) in enumerate(zip(hw_right, dataset_right)):
            diff = hw - ds
            flag = " <-- MISMATCH!" if abs(diff) > 20 else ""
            print(f"    Joint {i}: hw={hw:8.2f}°, dataset={ds:8.2f}°, diff={diff:+8.2f}°{flag}")

        # Check for port swap
        print("\n  CHECKING FOR POSSIBLE PORT SWAP:")
        print("  (comparing hardware_left with dataset_RIGHT and vice versa)")

        # hardware_left vs dataset_right
        left_vs_right_diffs = [abs(hw - ds) for hw, ds in zip(hw_left, dataset_right)]
        # hardware_right vs dataset_left
        right_vs_left_diffs = [abs(hw - ds) for hw, ds in zip(hw_right, dataset_left)]

        swap_match_left_to_right = np.mean(left_vs_right_diffs) < 20
        swap_match_right_to_left = np.mean(right_vs_left_diffs) < 20

        if swap_match_left_to_right and swap_match_right_to_left:
            print("\n  ⚠️  PORTS LIKELY SWAPPED!")
            print("      Hardware LEFT matches Dataset RIGHT")
            print("      Hardware RIGHT matches Dataset LEFT")
            print("\n  RECOMMENDED FIX:")
            print(f"      Swap ports in config: left_arm.port={right_port}, right_arm.port={left_port}")
        else:
            normal_left_diff = np.mean([abs(hw - ds) for hw, ds in zip(hw_left, dataset_left)])
            normal_right_diff = np.mean([abs(hw - ds) for hw, ds in zip(hw_right, dataset_right)])

            if normal_left_diff < 20 and normal_right_diff < 20:
                print("\n  ✓ Ports appear to be configured CORRECTLY")
                print(f"      Left arm avg diff: {normal_left_diff:.1f}°")
                print(f"      Right arm avg diff: {normal_right_diff:.1f}°")
            else:
                print(f"\n  ? Position mismatch detected but not a simple swap")
                print(f"      Left arm avg diff (normal): {normal_left_diff:.1f}°")
                print(f"      Right arm avg diff (normal): {normal_right_diff:.1f}°")
                print(f"      Left arm avg diff (swapped): {np.mean(left_vs_right_diffs):.1f}°")
                print(f"      Right arm avg diff (swapped): {np.mean(right_vs_left_diffs):.1f}°")


if __name__ == "__main__":
    main()
