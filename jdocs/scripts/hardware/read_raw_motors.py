#!/usr/bin/env python3
"""Read raw motor values without calibration to diagnose issues."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import time
from lerobot.motors.feetech import FeetechMotorsBus
from lerobot.motors.motors_bus import Motor, MotorNormMode

MOTOR_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
MOTOR_IDS = [1, 2, 3, 4, 5, 6]


def read_raw_positions(port: str, label: str):
    """Read raw encoder positions from motors."""
    print(f"\n{label} ({port}):")

    # Create motor configs
    motors = {
        name: Motor(id=id_, model="sts3215", norm_mode=MotorNormMode.DEGREES)
        for name, id_ in zip(MOTOR_NAMES, MOTOR_IDS)
    }

    bus = FeetechMotorsBus(
        port=port,
        motors=motors,
    )

    try:
        bus.connect()
        time.sleep(0.1)

        # Read raw positions for each motor
        for name in MOTOR_NAMES:
            raw_value = bus.read("Present_Position", name, normalize=False)
            print(f"  {name:20s}: raw={int(raw_value):5d}")

    finally:
        bus.disconnect()


def main():
    print("=" * 60)
    print("RAW MOTOR READINGS (no calibration)")
    print("=" * 60)

    # Read from both ports
    ports = [
        ("/dev/ttyACM3", "LEFT ARM config"),
        ("/dev/ttyACM2", "RIGHT ARM config"),
    ]

    for port, label in ports:
        try:
            read_raw_positions(port, label)
        except Exception as e:
            print(f"\n{label} ({port}): ERROR - {e}")

    print("\n" + "=" * 60)
    print("EXPECTED RAW VALUES FOR FOLDED HOME POSITION:")
    print("(Based on calibration files - raw = degrees * 11.375 + homing_offset + 2048)")
    print("=" * 60)

    # Load calibration and calculate expected raw values for home position
    import json

    calib_dir = Path.home() / ".cache/huggingface/lerobot/calibration/robots/so101_follower"

    # Home position in degrees (from dataset first frame)
    home_left = [-4.17, -99.24, 99.45, 51.22, -1.63, 0.50]  # Dataset left arm first frame
    home_right = [-1.96, -99.15, 99.27, 52.09, 3.26, 0.56]  # Dataset right arm first frame

    for arm_id, home_deg, arm_name in [
        ("xlerobot_left_arm", home_left, "LEFT"),
        ("xlerobot_right_arm", home_right, "RIGHT"),
    ]:
        calib_path = calib_dir / f"{arm_id}.json"
        if calib_path.exists():
            with open(calib_path) as f:
                calib = json.load(f)

            print(f"\n{arm_name} ARM expected raw for home position:")
            for i, name in enumerate(MOTOR_NAMES):
                motor_calib = calib.get(name, {})
                homing_offset = motor_calib.get("homing_offset", 0)
                # Formula: raw = degrees * 11.375 + homing_offset + 2048
                # Actually the formula is more complex, but let's approximate
                # From SO101Follower code: degrees = (raw - 2048 - homing_offset) / 11.375
                # So: raw = degrees * 11.375 + 2048 + homing_offset
                expected_raw = int(home_deg[i] * 11.375 + 2048 + homing_offset)
                print(f"  {name:20s}: expected_raw={expected_raw:5d} (homing_offset={homing_offset:5d}, target_deg={home_deg[i]:.1f}°)")


if __name__ == "__main__":
    main()
