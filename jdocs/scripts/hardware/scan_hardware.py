#!/usr/bin/env python3
"""
Hardware Scanner for XLeRobot Bimanual Setup.

This script scans connected hardware devices and updates the central
hardware configuration file. Run this whenever:
- System reboots
- USB devices are unplugged/replugged
- Hardware configuration needs to be verified

Usage:
    # Scan and show current hardware (no changes)
    python scan_hardware.py

    # Scan and update config file
    python scan_hardware.py --update

    # Interactive mode to identify which arm is on which port
    python scan_hardware.py --identify

    # Use a different config file
    python scan_hardware.py --config /path/to/config.yaml

Author: XLeRobot Project
"""

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import yaml

# ==============================================================================
# PATHS
# ==============================================================================
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "jdocs" / "configs" / "hardware" / "xlerobot_bimanual.yaml"


# ==============================================================================
# HARDWARE DETECTION
# ==============================================================================

def get_serial_devices() -> list[dict]:
    """Get list of serial devices (ttyACM*, ttyUSB*)."""
    devices = []

    for pattern in ["/dev/ttyACM*", "/dev/ttyUSB*"]:
        import glob
        for path in sorted(glob.glob(pattern)):
            device = {"path": path, "type": "serial"}

            # Get USB info using udevadm
            try:
                result = subprocess.run(
                    ["udevadm", "info", "-q", "property", path],
                    capture_output=True, text=True, timeout=5
                )
                if result.returncode == 0:
                    for line in result.stdout.split("\n"):
                        if "=" in line:
                            key, val = line.split("=", 1)
                            if key == "ID_SERIAL_SHORT":
                                device["serial"] = val
                            elif key == "ID_PATH":
                                device["usb_path"] = val
                            elif key == "ID_MODEL":
                                device["model"] = val
                            elif key == "ID_VENDOR":
                                device["vendor"] = val
            except Exception:
                pass

            devices.append(device)

    return devices


def get_video_devices() -> list[dict]:
    """Get list of video devices."""
    devices = []

    import glob
    for path in sorted(glob.glob("/dev/video*")):
        # Only include even-numbered video devices (actual cameras, not metadata)
        try:
            idx = int(path.replace("/dev/video", ""))
            if idx % 2 != 0:  # Skip odd numbers (metadata devices)
                continue
        except ValueError:
            continue

        device = {"path": path, "index": idx, "type": "camera"}

        # Get USB info
        try:
            result = subprocess.run(
                ["udevadm", "info", "-q", "property", path],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                for line in result.stdout.split("\n"):
                    if "=" in line:
                        key, val = line.split("=", 1)
                        if key == "ID_SERIAL_SHORT":
                            device["serial"] = val
                        elif key == "ID_PATH":
                            device["usb_path"] = val
                        elif key == "ID_MODEL":
                            device["model"] = val
                        elif key == "ID_V4L_PRODUCT":
                            device["product"] = val
        except Exception:
            pass

        devices.append(device)

    return devices


def test_serial_connection(port: str) -> dict:
    """Test if a serial port has a responsive robot motor bus."""
    result = {"port": port, "responsive": False, "motor_ids": []}

    try:
        # Try to import and connect
        sys.path.insert(0, str(PROJECT_ROOT / "src"))
        from lerobot.motors.feetech import FeetechMotorsBus
        from lerobot.motors.motors_bus import Motor, MotorNormMode

        # SO101 motor configuration
        motor_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
        motor_ids = [1, 2, 3, 4, 5, 6]
        motors = {
            name: Motor(id=id_, model="sts3215", norm_mode=MotorNormMode.DEGREES)
            for name, id_ in zip(motor_names, motor_ids)
        }

        bus = FeetechMotorsBus(port=port, motors=motors)
        bus.connect()

        # Check which motors respond
        for name in motor_names:
            try:
                # Try to read position
                pos = bus.read("Present_Position", name, normalize=False)
                if pos is not None:
                    motor_id = motors[name].id
                    result["motor_ids"].append(motor_id)
            except Exception:
                pass

        if result["motor_ids"]:
            result["responsive"] = True

        bus.disconnect()

    except Exception as e:
        result["error"] = str(e)

    return result


def wiggle_motor(port: str, motor_name: str = "elbow_flex", amplitude: int = 100, duration: float = 0.5) -> bool:
    """
    Wiggle a motor to help identify which arm is connected to a port.

    Args:
        port: Serial port path (e.g., /dev/ttyACM0)
        motor_name: Which motor to wiggle (default: elbow_flex for visibility)
        amplitude: Movement amplitude in raw units (default: 100, ~5 degrees)
        duration: Time for each direction in seconds

    Returns:
        True if wiggle was successful, False otherwise
    """
    import time

    try:
        sys.path.insert(0, str(PROJECT_ROOT / "src"))
        from lerobot.motors.feetech import FeetechMotorsBus
        from lerobot.motors.motors_bus import Motor, MotorNormMode

        # SO101 motor configuration
        motor_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
        motor_ids = [1, 2, 3, 4, 5, 6]
        motors = {
            name: Motor(id=id_, model="sts3215", norm_mode=MotorNormMode.DEGREES)
            for name, id_ in zip(motor_names, motor_ids)
        }

        bus = FeetechMotorsBus(port=port, motors=motors)
        bus.connect()

        # Read current position
        original_pos = bus.read("Present_Position", motor_name, normalize=False)
        if original_pos is None:
            bus.disconnect()
            return False

        original_pos = int(original_pos)

        # Enable torque for this motor
        bus.write("Torque_Enable", motor_name, 1, normalize=False)

        # Wiggle: move one direction, then back, then to original
        positions = [
            original_pos + amplitude,
            original_pos - amplitude,
            original_pos + amplitude,
            original_pos,
        ]

        for target_pos in positions:
            bus.write("Goal_Position", motor_name, target_pos, normalize=False)
            time.sleep(duration)

        # Disable torque
        bus.write("Torque_Enable", motor_name, 0, normalize=False)

        bus.disconnect()
        return True

    except Exception as e:
        print(f"    Wiggle error: {e}")
        return False


def detect_leader_movement(port: str, timeout: float = 5.0) -> bool:
    """
    Detect if a leader arm is being moved by the user.

    Args:
        port: Serial port path
        timeout: How long to wait for movement

    Returns:
        True if movement was detected, False otherwise
    """
    import time

    try:
        sys.path.insert(0, str(PROJECT_ROOT / "src"))
        from lerobot.motors.feetech import FeetechMotorsBus
        from lerobot.motors.motors_bus import Motor, MotorNormMode

        motor_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
        motor_ids = [1, 2, 3, 4, 5, 6]
        motors = {
            name: Motor(id=id_, model="sts3215", norm_mode=MotorNormMode.DEGREES)
            for name, id_ in zip(motor_names, motor_ids)
        }

        bus = FeetechMotorsBus(port=port, motors=motors)
        bus.connect()

        # Read initial positions
        initial_positions = {}
        for name in motor_names[:5]:  # Skip gripper
            pos = bus.read("Present_Position", name, normalize=False)
            if pos is not None:
                initial_positions[name] = int(pos)

        # Monitor for movement
        start_time = time.time()
        movement_threshold = 50  # Raw units, ~2.5 degrees

        while time.time() - start_time < timeout:
            for name, initial_pos in initial_positions.items():
                current_pos = bus.read("Present_Position", name, normalize=False)
                if current_pos is not None:
                    if abs(int(current_pos) - initial_pos) > movement_threshold:
                        bus.disconnect()
                        return True
            time.sleep(0.05)

        bus.disconnect()
        return False

    except Exception as e:
        print(f"    Detection error: {e}")
        return False


def test_camera(index: int) -> dict:
    """Test if a camera is accessible."""
    result = {"index": index, "accessible": False}

    try:
        import cv2
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                result["accessible"] = True
                result["width"] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                result["height"] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
    except Exception as e:
        result["error"] = str(e)

    return result


# ==============================================================================
# CONFIG MANAGEMENT
# ==============================================================================

def load_config(config_path: Path) -> dict:
    """Load hardware config from YAML."""
    if not config_path.exists():
        return {}
    with open(config_path) as f:
        return yaml.safe_load(f) or {}


def save_config(config: dict, config_path: Path) -> None:
    """Save hardware config to YAML."""
    config_path.parent.mkdir(parents=True, exist_ok=True)

    # Add update timestamp
    config["last_scan"] = datetime.now().isoformat()

    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def update_config_ports(config: dict, port_mapping: dict) -> dict:
    """Update config with new port mappings."""
    if "left_follower" in port_mapping:
        config["robot"]["left_arm"]["port"] = port_mapping["left_follower"]
    if "right_follower" in port_mapping:
        config["robot"]["right_arm"]["port"] = port_mapping["right_follower"]
    if "left_leader" in port_mapping:
        config["teleop"]["left_arm"]["port"] = port_mapping["left_leader"]
    if "right_leader" in port_mapping:
        config["teleop"]["right_arm"]["port"] = port_mapping["right_leader"]

    return config


# ==============================================================================
# INTERACTIVE IDENTIFICATION
# ==============================================================================

def identify_arms_interactive(serial_devices: list[dict]) -> dict:
    """Interactively identify which arm is on which port using wiggle test."""
    print("\n" + "=" * 70)
    print("INTERACTIVE ARM IDENTIFICATION")
    print("=" * 70)
    print()
    print("This will help identify which physical arm is on which USB port.")
    print()
    print("For FOLLOWERS: The script will wiggle the elbow - watch which arm moves.")
    print("For LEADERS:   You will manually move each arm when prompted.")
    print()

    # Filter to only responsive serial devices
    responsive = []
    for dev in serial_devices:
        test = test_serial_connection(dev["path"])
        if test["responsive"]:
            responsive.append(dev)
            print(f"Found responsive device: {dev['path']} (motors: {test['motor_ids']})")

    if len(responsive) < 2:
        print(f"\nError: Need at least 2 responsive serial devices, found {len(responsive)}")
        return {}

    print()
    print("=" * 50)
    print("IMPORTANT: Ensure robot is powered on!")
    print("=" * 50)
    input("Press Enter when ready to begin...")
    print()

    port_mapping = {}

    # Identify followers (usually ACM2 and ACM3)
    follower_ports = [d["path"] for d in responsive if "ACM2" in d["path"] or "ACM3" in d["path"]]
    if len(follower_ports) >= 2:
        print("-" * 50)
        print("FOLLOWER ARM IDENTIFICATION (Wiggle Test)")
        print("-" * 50)
        print(f"Testing ports: {follower_ports}")
        print()

        for port in follower_ports:
            print(f"Wiggling {port}... Watch which FOLLOWER arm moves!")
            success = wiggle_motor(port, motor_name="elbow_flex", amplitude=150, duration=0.4)

            if success:
                print(f"  Which PHYSICAL follower arm moved? (l=left, r=right, n=none/retry): ", end="")
                response = input().strip().lower()

                if response == "l":
                    port_mapping["left_follower"] = port
                    print(f"  ✓ {port} = LEFT follower")
                elif response == "r":
                    port_mapping["right_follower"] = port
                    print(f"  ✓ {port} = RIGHT follower")
                else:
                    print(f"  ? {port} = unknown (skipped)")
            else:
                print(f"  ✗ Failed to wiggle {port}")
            print()

    # Identify leaders (usually ACM0 and ACM1)
    leader_ports = [d["path"] for d in responsive if "ACM0" in d["path"] or "ACM1" in d["path"]]
    if len(leader_ports) >= 2:
        print("-" * 50)
        print("LEADER ARM IDENTIFICATION (Manual Movement)")
        print("-" * 50)
        print(f"Testing ports: {leader_ports}")
        print()
        print("For each port, you will be asked to move one leader arm.")
        print()

        for port in leader_ports:
            print(f"Testing {port}...")
            print(f"  Move the LEFT leader arm now (you have 5 seconds)...")
            left_moved = detect_leader_movement(port, timeout=5.0)

            if left_moved:
                port_mapping["left_leader"] = port
                print(f"  ✓ Movement detected! {port} = LEFT leader")
            else:
                print(f"  No movement on LEFT. Trying RIGHT...")
                print(f"  Move the RIGHT leader arm now (you have 5 seconds)...")
                right_moved = detect_leader_movement(port, timeout=5.0)

                if right_moved:
                    port_mapping["right_leader"] = port
                    print(f"  ✓ Movement detected! {port} = RIGHT leader")
                else:
                    # Fall back to manual input
                    print(f"  No movement detected. Manual identification:")
                    print(f"  Which PHYSICAL leader arm is on {port}? (l=left, r=right, n=none): ", end="")
                    response = input().strip().lower()

                    if response == "l":
                        port_mapping["left_leader"] = port
                        print(f"  ✓ {port} = LEFT leader")
                    elif response == "r":
                        port_mapping["right_leader"] = port
                        print(f"  ✓ {port} = RIGHT leader")
                    else:
                        print(f"  ? {port} = unknown (skipped)")
            print()

    return port_mapping


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Scan and configure XLeRobot hardware",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scan_hardware.py              # Show current hardware
    python scan_hardware.py --update     # Update config file
    python scan_hardware.py --identify   # Interactive arm identification
    python scan_hardware.py --test       # Test all devices
        """
    )
    parser.add_argument(
        "--config", "-c",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"Config file path (default: {DEFAULT_CONFIG})"
    )
    parser.add_argument(
        "--update", "-u",
        action="store_true",
        help="Update config file with detected hardware"
    )
    parser.add_argument(
        "--identify", "-i",
        action="store_true",
        help="Interactive mode to identify which arm is on which port"
    )
    parser.add_argument(
        "--test", "-t",
        action="store_true",
        help="Test connectivity of all devices"
    )
    args = parser.parse_args()

    print("=" * 70)
    print("XLEROBOT HARDWARE SCANNER")
    print("=" * 70)
    print(f"Config file: {args.config}")
    print()

    # Load current config
    config = load_config(args.config)

    # Scan serial devices
    print("Scanning serial devices...")
    serial_devices = get_serial_devices()
    print(f"Found {len(serial_devices)} serial device(s):")
    for dev in serial_devices:
        serial = dev.get("serial", "unknown")
        model = dev.get("model", "unknown")
        print(f"  {dev['path']}: serial={serial}, model={model}")
    print()

    # Scan video devices
    print("Scanning video devices...")
    video_devices = get_video_devices()
    print(f"Found {len(video_devices)} camera(s):")
    for dev in video_devices:
        product = dev.get("product", "unknown")
        print(f"  /dev/video{dev['index']}: {product}")
    print()

    # Show current config
    if config:
        print("Current configuration:")
        print(f"  Left follower:  {config.get('robot', {}).get('left_arm', {}).get('port', 'NOT SET')}")
        print(f"  Right follower: {config.get('robot', {}).get('right_arm', {}).get('port', 'NOT SET')}")
        print(f"  Left leader:    {config.get('teleop', {}).get('left_arm', {}).get('port', 'NOT SET')}")
        print(f"  Right leader:   {config.get('teleop', {}).get('right_arm', {}).get('port', 'NOT SET')}")
        print(f"  Head camera:    /dev/video{config.get('cameras', {}).get('head', {}).get('index_or_path', 'NOT SET')}")
        print(f"  Left wrist:     /dev/video{config.get('cameras', {}).get('left_wrist', {}).get('index_or_path', 'NOT SET')}")
        print(f"  Right wrist:    /dev/video{config.get('cameras', {}).get('right_wrist', {}).get('index_or_path', 'NOT SET')}")
        print()

    # Test mode
    if args.test:
        print("=" * 70)
        print("TESTING DEVICES")
        print("=" * 70)
        print()

        print("Testing serial ports...")
        for dev in serial_devices:
            result = test_serial_connection(dev["path"])
            status = "OK" if result["responsive"] else "FAIL"
            motors = result.get("motor_ids", [])
            error = result.get("error", "")
            print(f"  {dev['path']}: {status} (motors: {motors}) {error}")
        print()

        print("Testing cameras...")
        for dev in video_devices:
            result = test_camera(dev["index"])
            status = "OK" if result["accessible"] else "FAIL"
            res = f"{result.get('width', '?')}x{result.get('height', '?')}" if result["accessible"] else ""
            print(f"  /dev/video{dev['index']}: {status} {res}")
        print()

    # Interactive identification
    if args.identify:
        port_mapping = identify_arms_interactive(serial_devices)

        if port_mapping:
            print("\nIdentified port mapping:")
            for key, port in port_mapping.items():
                print(f"  {key}: {port}")

            if args.update:
                config = update_config_ports(config, port_mapping)
                save_config(config, args.config)
                print(f"\nConfig updated: {args.config}")
        else:
            print("\nNo port mapping identified.")

    # Update mode (without identification)
    elif args.update:
        # Auto-detect based on device order (less reliable)
        print("Auto-updating config with detected devices...")
        print("WARNING: This assumes standard USB enumeration order.")
        print("         Use --identify for reliable detection.")
        print()

        # Update serial numbers in config for future verification
        for dev in serial_devices:
            serial = dev.get("serial")
            usb_path = dev.get("usb_path")

            if "ACM3" in dev["path"] and config.get("robot", {}).get("left_arm"):
                config["robot"]["left_arm"]["usb_serial"] = serial
                config["robot"]["left_arm"]["usb_path"] = usb_path
            elif "ACM2" in dev["path"] and config.get("robot", {}).get("right_arm"):
                config["robot"]["right_arm"]["usb_serial"] = serial
                config["robot"]["right_arm"]["usb_path"] = usb_path
            elif "ACM0" in dev["path"] and config.get("teleop", {}).get("left_arm"):
                config["teleop"]["left_arm"]["usb_serial"] = serial
                config["teleop"]["left_arm"]["usb_path"] = usb_path
            elif "ACM1" in dev["path"] and config.get("teleop", {}).get("right_arm"):
                config["teleop"]["right_arm"]["usb_serial"] = serial
                config["teleop"]["right_arm"]["usb_path"] = usb_path

        save_config(config, args.config)
        print(f"Config updated: {args.config}")

    # Verification
    print()
    print("=" * 70)
    print("VERIFICATION CHECKLIST")
    print("=" * 70)
    print()

    # Check if config ports exist
    issues = []
    if config:
        for name, port in [
            ("Left follower", config.get("robot", {}).get("left_arm", {}).get("port")),
            ("Right follower", config.get("robot", {}).get("right_arm", {}).get("port")),
            ("Left leader", config.get("teleop", {}).get("left_arm", {}).get("port")),
            ("Right leader", config.get("teleop", {}).get("right_arm", {}).get("port")),
        ]:
            if port:
                exists = Path(port).exists()
                status = "[OK]" if exists else "[MISSING]"
                print(f"  {status} {name}: {port}")
                if not exists:
                    issues.append(f"{name} port {port} does not exist")
            else:
                print(f"  [NOT SET] {name}")
                issues.append(f"{name} not configured")

        for name, idx in [
            ("Head camera", config.get("cameras", {}).get("head", {}).get("index_or_path")),
            ("Left wrist", config.get("cameras", {}).get("left_wrist", {}).get("index_or_path")),
            ("Right wrist", config.get("cameras", {}).get("right_wrist", {}).get("index_or_path")),
        ]:
            if idx is not None:
                path = f"/dev/video{idx}"
                exists = Path(path).exists()
                status = "[OK]" if exists else "[MISSING]"
                print(f"  {status} {name}: {path}")
                if not exists:
                    issues.append(f"{name} {path} does not exist")
            else:
                print(f"  [NOT SET] {name}")

    print()
    if issues:
        print("ISSUES FOUND:")
        for issue in issues:
            print(f"  - {issue}")
        print()
        print("Run with --identify to fix port assignments.")
    else:
        print("All configured devices found!")

    return 0 if not issues else 1


if __name__ == "__main__":
    sys.exit(main())
