#!/usr/bin/env python3
"""
Hardware Configuration Loader for XLeRobot.

This module provides a single interface to load hardware configuration
from the central config file. All scripts should use this instead of
hardcoding hardware settings.

Usage:
    from hardware_config import get_hardware_config, get_robot_config, get_camera_config

    # Get full config
    config = get_hardware_config()

    # Get specific sections
    robot_cfg = get_robot_config()
    camera_cfg = get_camera_config()

    # Access specific values
    left_port = robot_cfg["left_arm"]["port"]
    head_camera_idx = camera_cfg["head"]["index_or_path"]
"""

import os
from pathlib import Path
from typing import Any, Optional

import yaml

# ==============================================================================
# CONFIG FILE PATHS
# ==============================================================================

# Default config file location (relative to this file)
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parents[2]
_DEFAULT_CONFIG_PATH = _PROJECT_ROOT / "jdocs" / "configs" / "hardware" / "xlerobot_bimanual.yaml"

# Environment variable to override config path
_CONFIG_ENV_VAR = "XLEROBOT_HARDWARE_CONFIG"


# ==============================================================================
# CONFIG LOADING
# ==============================================================================

_cached_config: Optional[dict] = None


def get_config_path() -> Path:
    """Get the hardware config file path.

    Priority:
    1. XLEROBOT_HARDWARE_CONFIG environment variable
    2. Default path: jdocs/configs/hardware/xlerobot_bimanual.yaml

    Returns:
        Path to config file
    """
    env_path = os.environ.get(_CONFIG_ENV_VAR)
    if env_path:
        return Path(env_path)
    return _DEFAULT_CONFIG_PATH


def get_hardware_config(reload: bool = False) -> dict:
    """Load hardware configuration from YAML file.

    Args:
        reload: If True, reload from disk even if cached

    Returns:
        Full hardware configuration dictionary

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config file is invalid
    """
    global _cached_config

    if _cached_config is not None and not reload:
        return _cached_config

    config_path = get_config_path()

    if not config_path.exists():
        raise FileNotFoundError(
            f"Hardware config not found: {config_path}\n"
            f"Run 'python jdocs/scripts/hardware/scan_hardware.py' to create it."
        )

    with open(config_path) as f:
        config = yaml.safe_load(f)

    if not config:
        raise ValueError(f"Empty or invalid config file: {config_path}")

    _cached_config = config
    return config


def get_robot_config() -> dict:
    """Get robot (follower) configuration.

    Returns:
        Robot config with left_arm and right_arm settings
    """
    config = get_hardware_config()
    return config.get("robot", {})


def get_teleop_config() -> dict:
    """Get teleoperator (leader) configuration.

    Returns:
        Teleop config with left_arm and right_arm settings
    """
    config = get_hardware_config()
    return config.get("teleop", {})


def get_camera_config() -> dict:
    """Get camera configuration.

    Returns:
        Camera config with head, left_wrist, right_wrist settings
    """
    config = get_hardware_config()
    return config.get("cameras", {})


def get_action_config() -> dict:
    """Get action/observation configuration.

    Returns:
        Action config with dim, fps, joint_names
    """
    config = get_hardware_config()
    return config.get("action", {})


def get_inference_config() -> dict:
    """Get inference default settings.

    Returns:
        Inference config with action_interval, duration, device
    """
    config = get_hardware_config()
    return config.get("inference", {})


# ==============================================================================
# CONVENIENCE FUNCTIONS
# ==============================================================================

def get_left_arm_port() -> str:
    """Get left follower arm port."""
    return get_robot_config().get("left_arm", {}).get("port", "/dev/ttyACM3")


def get_right_arm_port() -> str:
    """Get right follower arm port."""
    return get_robot_config().get("right_arm", {}).get("port", "/dev/ttyACM2")


def get_left_arm_id() -> str:
    """Get left arm calibration ID."""
    return get_robot_config().get("left_arm", {}).get("id", "xlerobot_left_arm")


def get_right_arm_id() -> str:
    """Get right arm calibration ID."""
    return get_robot_config().get("right_arm", {}).get("id", "xlerobot_right_arm")


def get_camera_index(camera_name: str) -> int:
    """Get camera video device index.

    Args:
        camera_name: One of "head", "left_wrist", "right_wrist"

    Returns:
        Video device index (e.g., 4 for /dev/video4)
    """
    cameras = get_camera_config()
    if camera_name not in cameras:
        raise ValueError(f"Unknown camera: {camera_name}. Available: {list(cameras.keys())}")
    return cameras[camera_name].get("index_or_path", 0)


def validate_hardware() -> tuple[bool, list[str]]:
    """Validate that all configured hardware exists.

    Returns:
        Tuple of (all_valid, list_of_issues)
    """
    issues = []

    try:
        config = get_hardware_config()
    except Exception as e:
        return False, [str(e)]

    # Check robot ports
    robot_cfg = config.get("robot", {})
    for arm in ["left_arm", "right_arm"]:
        port = robot_cfg.get(arm, {}).get("port")
        if port and not Path(port).exists():
            issues.append(f"Robot {arm} port not found: {port}")

    # Check teleop ports
    teleop_cfg = config.get("teleop", {})
    for arm in ["left_arm", "right_arm"]:
        port = teleop_cfg.get(arm, {}).get("port")
        if port and not Path(port).exists():
            issues.append(f"Teleop {arm} port not found: {port}")

    # Check cameras
    camera_cfg = config.get("cameras", {})
    for cam_name, cam_settings in camera_cfg.items():
        idx = cam_settings.get("index_or_path")
        if idx is not None:
            path = f"/dev/video{idx}"
            if not Path(path).exists():
                issues.append(f"Camera {cam_name} not found: {path}")

    return len(issues) == 0, issues


def print_config_summary() -> None:
    """Print a summary of current hardware configuration."""
    try:
        config = get_hardware_config()
    except Exception as e:
        print(f"Error loading config: {e}")
        return

    print("Hardware Configuration Summary")
    print("=" * 50)
    print(f"Config file: {get_config_path()}")
    print()

    robot_cfg = config.get("robot", {})
    print("Robot (Follower):")
    print(f"  Left arm:  {robot_cfg.get('left_arm', {}).get('port', 'NOT SET')}")
    print(f"  Right arm: {robot_cfg.get('right_arm', {}).get('port', 'NOT SET')}")
    print()

    teleop_cfg = config.get("teleop", {})
    print("Teleop (Leader):")
    print(f"  Left arm:  {teleop_cfg.get('left_arm', {}).get('port', 'NOT SET')}")
    print(f"  Right arm: {teleop_cfg.get('right_arm', {}).get('port', 'NOT SET')}")
    print()

    camera_cfg = config.get("cameras", {})
    print("Cameras:")
    for name, cam in camera_cfg.items():
        idx = cam.get("index_or_path", "NOT SET")
        print(f"  {name}: /dev/video{idx}")
    print()

    valid, issues = validate_hardware()
    if valid:
        print("Status: All hardware found")
    else:
        print("Status: ISSUES FOUND")
        for issue in issues:
            print(f"  - {issue}")


# ==============================================================================
# CLI
# ==============================================================================

if __name__ == "__main__":
    print_config_summary()
