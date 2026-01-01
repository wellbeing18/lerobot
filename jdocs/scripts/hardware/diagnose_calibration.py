#!/usr/bin/env python3
"""
Comprehensive Calibration Diagnostic Script.

This script helps identify calibration issues by:
1. Reading RAW encoder values from both arms
2. Showing calibration file contents
3. Calculating degrees using BOTH calibration files on BOTH arms' raw values
4. Comparing with training data expected values

If ARM A's raw values produce correct degrees with ARM B's calibration file,
it means the calibration files are swapped or were created on wrong ports.
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import pyarrow.parquet as pq
import yaml

from lerobot.motors.feetech import FeetechMotorsBus
from lerobot.motors.motors_bus import Motor, MotorNormMode

MOTOR_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
MOTOR_IDS = [1, 2, 3, 4, 5, 6]
CALIB_DIR = Path.home() / ".cache/huggingface/lerobot/calibration/robots/so101_follower"


def load_calibration(arm_id: str) -> dict:
    """Load calibration file for an arm."""
    calib_path = CALIB_DIR / f"{arm_id}.json"
    if not calib_path.exists():
        raise FileNotFoundError(f"Calibration file not found: {calib_path}")
    with open(calib_path) as f:
        return json.load(f)


def read_raw_positions(port: str) -> dict[str, int]:
    """Read raw encoder positions from an arm."""
    motors = {
        name: Motor(id=id_, model="sts3215", norm_mode=MotorNormMode.DEGREES)
        for name, id_ in zip(MOTOR_NAMES, MOTOR_IDS)
    }

    bus = FeetechMotorsBus(port=port, motors=motors)
    bus.connect()

    raw_values = {}
    for name in MOTOR_NAMES:
        raw_values[name] = int(bus.read("Present_Position", name, normalize=False))

    bus.disconnect()
    return raw_values


def raw_to_degrees(raw: int, homing_offset: int) -> float:
    """Convert raw encoder value to degrees using calibration."""
    # Formula from SO101Follower: degrees = (raw - 2048 - homing_offset) / 11.375
    return (raw - 2048 - homing_offset) / 11.375


def get_training_first_frame() -> tuple[list[float], list[float]]:
    """Get first frame values from training dataset."""
    dataset_path = PROJECT_ROOT / "datasets_bimanuel/bimanual/combined_pick_and_place"
    parquet_files = sorted(dataset_path.glob("data/**/*.parquet"))

    if not parquet_files:
        return None, None

    table = pq.read_table(parquet_files[0])
    df = table.to_pandas()

    # Get first frame
    first_frame = df.iloc[0]['observation.state']

    # Split into left (0-5) and right (6-11)
    return list(first_frame[:6]), list(first_frame[6:])


def main():
    # Load config
    config_path = PROJECT_ROOT / "jdocs/configs/hardware/xlerobot_bimanual.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    robot_cfg = config.get("robot", {})
    left_port = robot_cfg["left_arm"]["port"]
    right_port = robot_cfg["right_arm"]["port"]
    left_id = robot_cfg["left_arm"]["id"]
    right_id = robot_cfg["right_arm"]["id"]

    print("=" * 80)
    print("COMPREHENSIVE CALIBRATION DIAGNOSTIC")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  LEFT arm:  port={left_port}, calibration_id={left_id}")
    print(f"  RIGHT arm: port={right_port}, calibration_id={right_id}")

    # Load calibration files
    print("\n" + "-" * 80)
    print("CALIBRATION FILES")
    print("-" * 80)

    left_calib = load_calibration(left_id)
    right_calib = load_calibration(right_id)

    print(f"\n{left_id}.json (used for LEFT arm):")
    for name in MOTOR_NAMES:
        offset = left_calib[name]["homing_offset"]
        print(f"  {name:20s}: homing_offset = {offset:+5d}")

    print(f"\n{right_id}.json (used for RIGHT arm):")
    for name in MOTOR_NAMES:
        offset = right_calib[name]["homing_offset"]
        print(f"  {name:20s}: homing_offset = {offset:+5d}")

    # Compare homing offsets
    print("\nHoming offset DIFFERENCE (left - right):")
    for name in MOTOR_NAMES:
        left_offset = left_calib[name]["homing_offset"]
        right_offset = right_calib[name]["homing_offset"]
        diff = left_offset - right_offset
        flag = " <-- SIGNIFICANT" if abs(diff) > 500 else ""
        print(f"  {name:20s}: {diff:+5d}{flag}")

    # Read raw positions
    print("\n" + "-" * 80)
    print("RAW ENCODER VALUES (no calibration applied)")
    print("-" * 80)

    print(f"\nReading from LEFT arm ({left_port})...")
    left_raw = read_raw_positions(left_port)

    print(f"Reading from RIGHT arm ({right_port})...")
    right_raw = read_raw_positions(right_port)

    print(f"\nLEFT arm raw values:")
    for name in MOTOR_NAMES:
        print(f"  {name:20s}: {left_raw[name]:5d}")

    print(f"\nRIGHT arm raw values:")
    for name in MOTOR_NAMES:
        print(f"  {name:20s}: {right_raw[name]:5d}")

    # Calculate degrees using CORRECT calibration
    print("\n" + "-" * 80)
    print("DEGREES WITH CORRECT CALIBRATION")
    print("-" * 80)

    print("\nLEFT arm raw → LEFT calibration → degrees:")
    left_degrees_correct = []
    for name in MOTOR_NAMES:
        deg = raw_to_degrees(left_raw[name], left_calib[name]["homing_offset"])
        left_degrees_correct.append(deg)
        print(f"  {name:20s}: {deg:+8.2f}°")

    print("\nRIGHT arm raw → RIGHT calibration → degrees:")
    right_degrees_correct = []
    for name in MOTOR_NAMES:
        deg = raw_to_degrees(right_raw[name], right_calib[name]["homing_offset"])
        right_degrees_correct.append(deg)
        print(f"  {name:20s}: {deg:+8.2f}°")

    # Calculate degrees using SWAPPED calibration
    print("\n" + "-" * 80)
    print("DEGREES WITH SWAPPED CALIBRATION (diagnostic)")
    print("-" * 80)

    print("\nLEFT arm raw → RIGHT calibration → degrees:")
    left_degrees_swapped = []
    for name in MOTOR_NAMES:
        deg = raw_to_degrees(left_raw[name], right_calib[name]["homing_offset"])
        left_degrees_swapped.append(deg)
        print(f"  {name:20s}: {deg:+8.2f}°")

    print("\nRIGHT arm raw → LEFT calibration → degrees:")
    right_degrees_swapped = []
    for name in MOTOR_NAMES:
        deg = raw_to_degrees(right_raw[name], left_calib[name]["homing_offset"])
        right_degrees_swapped.append(deg)
        print(f"  {name:20s}: {deg:+8.2f}°")

    # Compare with training data
    print("\n" + "-" * 80)
    print("COMPARISON WITH TRAINING DATA (first frame)")
    print("-" * 80)

    train_left, train_right = get_training_first_frame()

    if train_left and train_right:
        print("\nTraining data first frame (expected home position):")
        print("  LEFT arm:", [f"{v:.1f}" for v in train_left])
        print("  RIGHT arm:", [f"{v:.1f}" for v in train_right])

        # Calculate RMS errors for each combination
        def rms_error(a, b):
            return np.sqrt(np.mean([(x - y) ** 2 for x, y in zip(a, b)]))

        print("\n" + "-" * 80)
        print("RMS ERROR ANALYSIS (lower = better match)")
        print("-" * 80)

        # Current configuration
        err_left_correct = rms_error(left_degrees_correct, train_left)
        err_right_correct = rms_error(right_degrees_correct, train_right)

        # Swapped calibration
        err_left_swapped = rms_error(left_degrees_swapped, train_left)
        err_right_swapped = rms_error(right_degrees_swapped, train_right)

        # Cross comparison (if physical arms are swapped)
        err_left_to_train_right = rms_error(left_degrees_correct, train_right)
        err_right_to_train_left = rms_error(right_degrees_correct, train_left)

        print("\nCURRENT CONFIGURATION:")
        print(f"  LEFT arm (correct calib) vs train_LEFT:   {err_left_correct:6.1f}°")
        print(f"  RIGHT arm (correct calib) vs train_RIGHT: {err_right_correct:6.1f}°")

        print("\nIF CALIBRATION FILES WERE SWAPPED:")
        print(f"  LEFT arm (right calib) vs train_LEFT:     {err_left_swapped:6.1f}°")
        print(f"  RIGHT arm (left calib) vs train_RIGHT:    {err_right_swapped:6.1f}°")

        print("\nIF PHYSICAL ARMS WERE SWAPPED (different test):")
        print(f"  LEFT arm reading vs train_RIGHT:          {err_left_to_train_right:6.1f}°")
        print(f"  RIGHT arm reading vs train_LEFT:          {err_right_to_train_left:6.1f}°")

        print("\n" + "=" * 80)
        print("DIAGNOSIS")
        print("=" * 80)

        # Determine the issue
        if err_left_correct < 20 and err_right_correct < 20:
            print("\n✓ BOTH arms match training data well with current configuration.")
            print("  No calibration issue detected.")

        elif err_left_swapped < err_left_correct and err_right_swapped < err_right_correct:
            print("\n⚠️  CALIBRATION FILES ARE LIKELY SWAPPED!")
            print(f"  LEFT arm matches better with RIGHT calibration ({err_left_swapped:.1f}° vs {err_left_correct:.1f}°)")
            print(f"  RIGHT arm matches better with LEFT calibration ({err_right_swapped:.1f}° vs {err_right_correct:.1f}°)")
            print("\n  RECOMMENDED FIX:")
            print(f"    Swap calibration IDs in config:")
            print(f"      left_arm.id:  {right_id}  (was {left_id})")
            print(f"      right_arm.id: {left_id}  (was {right_id})")
            print(f"\n    OR rename calibration files:")
            print(f"      mv {left_id}.json {left_id}.json.bak")
            print(f"      mv {right_id}.json {left_id}.json")
            print(f"      mv {left_id}.json.bak {right_id}.json")

        elif err_left_to_train_right < err_left_correct and err_right_to_train_left < err_right_correct:
            print("\n⚠️  PHYSICAL ARMS MIGHT BE SWAPPED ON PORTS!")
            print(f"  LEFT port reading matches train_RIGHT ({err_left_to_train_right:.1f}° vs {err_left_correct:.1f}°)")
            print(f"  RIGHT port reading matches train_LEFT ({err_right_to_train_left:.1f}° vs {err_right_correct:.1f}°)")
            print("\n  RECOMMENDED FIX:")
            print(f"    Swap port assignments in config:")
            print(f"      left_arm.port:  {right_port}  (was {left_port})")
            print(f"      right_arm.port: {left_port}  (was {right_port})")

        elif err_left_correct > 50 and err_right_correct < 20:
            print("\n⚠️  LEFT ARM CALIBRATION ISSUE!")
            print(f"  LEFT arm error: {err_left_correct:.1f}° (too high)")
            print(f"  RIGHT arm error: {err_right_correct:.1f}° (OK)")
            print("\n  POSSIBLE CAUSES:")
            print("    1. LEFT arm calibration file was created at wrong position")
            print("    2. LEFT arm physical position changed since calibration")
            print("    3. LEFT arm needs recalibration")
            print("\n  RECOMMENDED FIX:")
            print("    Recalibrate the LEFT arm while ensuring it's in the same")
            print("    home position as during data collection.")

        elif err_right_correct > 50 and err_left_correct < 20:
            print("\n⚠️  RIGHT ARM CALIBRATION ISSUE!")
            print(f"  LEFT arm error: {err_left_correct:.1f}° (OK)")
            print(f"  RIGHT arm error: {err_right_correct:.1f}° (too high)")
            print("\n  POSSIBLE CAUSES:")
            print("    1. RIGHT arm calibration file was created at wrong position")
            print("    2. RIGHT arm physical position changed since calibration")
            print("    3. RIGHT arm needs recalibration")

        else:
            print("\n⚠️  COMPLEX ISSUE - Multiple problems possible")
            print(f"  LEFT arm error: {err_left_correct:.1f}°")
            print(f"  RIGHT arm error: {err_right_correct:.1f}°")
            print("\n  Please visually verify:")
            print("    1. Both arms are in the SAME folded home position")
            print("    2. Calibration was done with arms in correct home position")
            print("    3. Consider recalibrating both arms")

        # Show detailed joint-by-joint comparison
        print("\n" + "-" * 80)
        print("DETAILED JOINT-BY-JOINT COMPARISON")
        print("-" * 80)

        print("\nLEFT arm (current deg vs training):")
        for i, name in enumerate(MOTOR_NAMES):
            curr = left_degrees_correct[i]
            train = train_left[i]
            diff = curr - train
            flag = " <-- MISMATCH!" if abs(diff) > 20 else ""
            print(f"  {name:20s}: current={curr:+7.1f}°  train={train:+7.1f}°  diff={diff:+7.1f}°{flag}")

        print("\nRIGHT arm (current deg vs training):")
        for i, name in enumerate(MOTOR_NAMES):
            curr = right_degrees_correct[i]
            train = train_right[i]
            diff = curr - train
            flag = " <-- MISMATCH!" if abs(diff) > 20 else ""
            print(f"  {name:20s}: current={curr:+7.1f}°  train={train:+7.1f}°  diff={diff:+7.1f}°{flag}")


if __name__ == "__main__":
    main()
