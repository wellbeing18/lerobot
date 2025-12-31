#!/usr/bin/env python3
"""
XLeRobot Data Collection Script

Industrial-level data collection script for collecting multi-task datasets
to finetune Pi0.5 VLA models and enable VLM-VLA collaboration.

This script provides:
1. Interactive task selection from 6 core primitives
2. Automatic parameter calculation based on task complexity
3. Separate datasets per task type
4. Hardware validation before recording
5. Resume support for adding episodes to existing datasets
6. VLM-compatible task strings for seamless VLM-VLA integration

Usage:
    # Interactive mode (guided collection)
    python scripts/collect_xlerobot_data.py

    # Direct task selection
    python scripts/collect_xlerobot_data.py --task pick --num_episodes 15

    # Custom task with specific object/location
    python scripts/collect_xlerobot_data.py --task pick --object "red cube" --location "table" --num_episodes 15

    # Resume existing dataset
    python scripts/collect_xlerobot_data.py --task pick --resume

Author: XLeRobot Project
Date: 2025-11-29
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import yaml

# ==============================================================================
# CONFIGURATION CONSTANTS
# ==============================================================================

# Base paths
PROJECT_ROOT = Path("/home/jrobot/project/lerobot/jdocs/bimanual/jassy")
DATASETS_BASE = Path("/home/jrobot/project/lerobot/datasets_bimanuel")  # Bimanual datasets folder
DATASETS_BASE_LEGACY = Path("/home/jrobot/project/XLeRobot/datasets")  # Legacy single-arm datasets
CONFIGS_DIR = PROJECT_ROOT / "configs"


# ==============================================================================
# YAML CONFIGURATION LOADER
# ==============================================================================

def load_config(config_path: Optional[Path]) -> dict:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to YAML config file (optional)

    Returns:
        Configuration dictionary (empty if no config provided)
    """
    if config_path is None:
        return {}

    config_path = Path(config_path)
    if not config_path.exists():
        print(f"Warning: Config file not found: {config_path}")
        return {}

    with open(config_path) as f:
        config = yaml.safe_load(f)

    return config or {}


def save_config_to_output(config: dict, output_path: Path, source_config_path: Optional[Path] = None) -> None:
    """
    Save the effective configuration to the output directory for reproducibility.

    Args:
        config: Configuration dictionary to save
        output_path: Directory to save config to
        source_config_path: Original config file path (for reference)
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # Add metadata
    effective_config = config.copy()
    effective_config["_metadata"] = {
        "saved_at": datetime.now().isoformat(),
        "source_config": str(source_config_path) if source_config_path else None,
        "script": "collect_xlerobot_data.py"
    }

    # Save as YAML
    config_output = output_path / "config.yaml"
    with open(config_output, "w") as f:
        yaml.dump(effective_config, f, default_flow_style=False, sort_keys=False)

    print(f"Configuration saved to: {config_output}")


def merge_config_with_args(config: dict, args: argparse.Namespace) -> dict:
    """
    Merge YAML config with command-line arguments.
    CLI args take precedence over config file.

    Args:
        config: Configuration from YAML file
        args: Parsed command-line arguments

    Returns:
        Merged configuration dictionary
    """
    merged = config.copy()

    # CLI overrides
    if args.arm:
        merged.setdefault("hardware", {})["arm"] = args.arm

    if args.task:
        merged.setdefault("data_collection", {}).setdefault("task", {})["type"] = args.task

    if args.num_episodes:
        merged.setdefault("data_collection", {}).setdefault("task", {})["num_episodes"] = args.num_episodes

    if args.object:
        merged.setdefault("data_collection", {}).setdefault("task", {})["object"] = args.object

    if args.location:
        merged.setdefault("data_collection", {}).setdefault("task", {})["location"] = args.location

    if args.target:
        merged.setdefault("data_collection", {}).setdefault("task", {})["target"] = args.target

    if args.direction:
        merged.setdefault("data_collection", {}).setdefault("task", {})["direction"] = args.direction

    return merged

# Robot configuration - supports single arm and bimanual modes
# Default: bimanual (both arms), can use --arm left/right for single arm mode
ARM_CONFIGS = {
    "left": {
        "robot": {
            "type": "so101_follower",
            "port": "/dev/ttyACM3",  # left follower
            "id": "xlerobot_left_arm",
        },
        "teleop": {
            "type": "so101_leader",
            "port": "/dev/ttyACM0",  # left leader
            "id": "xlerobot_left_leader",
        },
        "cameras": {
            # Using MJPG (compressed) to reduce USB bandwidth: ~3 MB/s vs ~28 MB/s raw
            "left_wrist": {"type": "opencv", "index_or_path": 8, "width": 640, "height": 480, "fps": 30, "fourcc": "MJPG"},
            "head": {"type": "opencv", "index_or_path": 4, "width": 640, "height": 480, "fps": 30, "fourcc": "MJPG"}
        }
    },
    "right": {
        "robot": {
            "type": "so101_follower",
            "port": "/dev/ttyACM2",  # right follower
            "id": "xlerobot_right_arm",
        },
        "teleop": {
            "type": "so101_leader",
            "port": "/dev/ttyACM1",  # right leader
            "id": "xlerobot_right_leader",
        },
        "cameras": {
            # Using MJPG (compressed) to reduce USB bandwidth: ~3 MB/s vs ~28 MB/s raw
            "right_wrist": {"type": "opencv", "index_or_path": 6, "width": 640, "height": 480, "fps": 30, "fourcc": "MJPG"},
            "head": {"type": "opencv", "index_or_path": 4, "width": 640, "height": 480, "fps": 30, "fourcc": "MJPG"}
        }
    },
    # BIMANUAL configuration - controls both arms simultaneously
    "bimanual": {
        "robot": {
            "type": "bi_so101_follower",
            "left_arm_port": "/dev/ttyACM3",   # left follower
            "right_arm_port": "/dev/ttyACM2",  # right follower
            "id": "xlerobot_bimanual",
            # Use existing calibration IDs
            "left_arm_id": "xlerobot_left_arm",
            "right_arm_id": "xlerobot_right_arm",
        },
        "teleop": {
            "type": "bi_so101_leader",
            "left_arm_port": "/dev/ttyACM0",   # left leader
            "right_arm_port": "/dev/ttyACM1",  # right leader
            "id": "xlerobot_bimanual_leader",
            # Use existing calibration IDs
            "left_arm_id": "xlerobot_left_leader",
            "right_arm_id": "xlerobot_right_leader",
        },
        "cameras": {
            # All 3 cameras for bimanual operation
            "head": {"type": "opencv", "index_or_path": 4, "width": 640, "height": 480, "fps": 30, "fourcc": "MJPG"},
            "left_wrist": {"type": "opencv", "index_or_path": 8, "width": 640, "height": 480, "fps": 30, "fourcc": "MJPG"},
            "right_wrist": {"type": "opencv", "index_or_path": 6, "width": 640, "height": 480, "fps": 30, "fourcc": "MJPG"}
        },
        # Bimanual-specific metadata
        "action_dim": 12,  # 6 left + 6 right
        "observation_keys": {
            "left": ["left_shoulder_pan", "left_shoulder_lift", "left_elbow_flex",
                     "left_wrist_flex", "left_wrist_roll", "left_gripper"],
            "right": ["right_shoulder_pan", "right_shoulder_lift", "right_elbow_flex",
                      "right_wrist_flex", "right_wrist_roll", "right_gripper"]
        }
    }
}

# Default arm selection - bimanual for dual-arm operation
DEFAULT_ARM = "bimanual"

# Legacy compatibility (will be overridden by arm selection)
ROBOT_CONFIG = ARM_CONFIGS["left"]["robot"]
TELEOP_CONFIG = ARM_CONFIGS["left"]["teleop"]

# Camera configuration is now per-arm in ARM_CONFIGS above
# Default camera config (left arm) for backward compatibility
CAMERA_CONFIG = ARM_CONFIGS["left"]["cameras"]

# Action frequency - CRITICAL: Pi0.5 expects 5Hz
ACTION_FPS = 30

# ==============================================================================
# TASK PRESETS - Industry Best Practices
# ==============================================================================

TASK_PRESETS = {
    "pick": {
        "template": "pick the {object} from the {location}",
        "episode_time_s": 30,
        "reset_time_s": 30,
        "recommended_episodes": 25,
        "default_object": "cube",
        "default_location": "table",
        "description": "Approach -> grasp -> lift -> hold",
        "phases": ["Approach (0-10s)", "Descend (10-15s)", "Grasp (15-18s)",
                   "Lift (18-25s)", "Hold (25-30s)"]
    },
    "place": {
        "template": "place the {object} in the {target}",
        "episode_time_s": 30,
        "reset_time_s": 30,
        "recommended_episodes": 15,
        "default_object": "cube",
        "default_target": "box",
        "description": "Position -> lower -> release -> retract",
        "phases": ["Position (0-10s)", "Lower (10-18s)", "Release (18-22s)",
                   "Retract (22-30s)"]
    },
    "push": {
        "template": "push the {object} to the {direction}",
        "episode_time_s": 40,
        "reset_time_s": 35,
        "recommended_episodes": 10,
        "default_object": "cube",
        "default_direction": "target",
        "description": "Contact -> slide -> release contact",
        "phases": ["Approach (0-10s)", "Contact (10-15s)", "Push (15-30s)",
                   "Release (30-40s)"]
    },
    "reach": {
        "template": "reach the {location}",
        "episode_time_s": 20,
        "reset_time_s": 20,
        "recommended_episodes": 10,
        "default_location": "center",
        "description": "Move to position without manipulation",
        "phases": ["Move (0-15s)", "Hold (15-20s)"]
    },
    "grasp": {
        "template": "grasp the {object}",
        "episode_time_s": 25,
        "reset_time_s": 20,
        "recommended_episodes": 10,
        "default_object": "cube",
        "description": "Close gripper (already positioned above object)",
        "phases": ["Align (0-10s)", "Close (10-18s)", "Verify (18-25s)"]
    },
    "release": {
        "template": "release",
        "episode_time_s": 15,
        "reset_time_s": 25,
        "recommended_episodes": 10,
        "description": "Open gripper (already at release position)",
        "phases": ["Open (0-8s)", "Observe (8-15s)"]
    },
    "pick_and_place": {
        "template": "pick up the {object} and place it on the {target}",
        "episode_time_s": 100,
        "reset_time_s": 45,
        "recommended_episodes": 10,
        "default_object": "cube",
        "default_target": "plate",
        "description": "Complete pick-and-place: approach -> grasp -> lift -> transport -> place -> release",
        "phases": [
            "Approach (0-8s)", "Descend (8-12s)", "Grasp (12-15s)",
            "Lift (15-20s)", "Transport (20-35s)", "Lower (35-42s)",
            "Release (42-48s)", "Retract (48-55s)"
        ]
    },
    # =========================================================================
    # BIMANUAL TASK PRESETS - Require both arms
    # =========================================================================
    "handover": {
        "template": "hand the {object} from left arm to right arm",
        "episode_time_s": 45,
        "reset_time_s": 25,
        "recommended_episodes": 50,
        "default_object": "cube",
        "description": "BIMANUAL: Left picks -> position -> right receives -> transfer complete",
        "requires_bimanual": True,
        "phases": [
            "Left arm picks object (0-15s)",
            "Position for handover (15-25s)",
            "Right arm receives (25-35s)",
            "Complete transfer (35-45s)"
        ]
    },
    "bimanual_pick_and_place": {
        "template": "pick up the {object} and place it on the {target}",
        "episode_time_s": 60,
        "reset_time_s": 30,
        "recommended_episodes": 50,
        "default_object": "tissue packet",
        "default_target": "plate",
        "description": "BIMANUAL: Full pick-and-place with both arms coordinated",
        "requires_bimanual": True,
        "phases": [
            "Approach (0-10s)",
            "Grasp (10-20s)",
            "Lift (20-30s)",
            "Transport (30-45s)",
            "Place (45-55s)",
            "Release (55-60s)"
        ]
    },
    "left_arm_pick_and_place": {
        "template": "Left arm pick up the {object} and place it on the {target}",
        "episode_time_s": 60,
        "reset_time_s": 30,
        "recommended_episodes": 10,
        "default_object": "tissue packet",
        "default_target": "plate",
        "description": "BIMANUAL: Left arm only pick-and-place (right arm stays idle)",
        "requires_bimanual": True,
        "phases": [
            "Left arm approach (0-10s)",
            "Left arm grasp (10-20s)",
            "Left arm lift (20-30s)",
            "Left arm transport (30-45s)",
            "Left arm place (45-55s)",
            "Left arm release (55-60s)"
        ]
    },
    "right_arm_pick_and_place": {
        "template": "Right arm pick up the {object} and place it on the {target}",
        "episode_time_s": 60,
        "reset_time_s": 30,
        "recommended_episodes": 10,
        "default_object": "tissue packet",
        "default_target": "plate",
        "description": "BIMANUAL: Right arm only pick-and-place (left arm stays idle)",
        "requires_bimanual": True,
        "phases": [
            "Right arm approach (0-10s)",
            "Right arm grasp (10-20s)",
            "Right arm lift (20-30s)",
            "Right arm transport (30-45s)",
            "Right arm place (45-55s)",
            "Right arm release (55-60s)"
        ]
    }
}

# Object and location vocabulary for VLM compatibility
OBJECT_VOCABULARY = [
    "cube", "block", "object",
    "red cube", "blue cube", "green cube",
    "small cube", "large cube",
    "tissue packet", "tissue pack", "packet"
]

LOCATION_VOCABULARY = [
    "table", "center", "left", "right", "front", "back",
    "box", "target", "bin", "container", "plate", "white plate", "tray"
]


# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================

def print_header(title: str) -> None:
    """Print a formatted header."""
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)


def print_section(title: str) -> None:
    """Print a section title."""
    print(f"\n--- {title} ---")


def validate_hardware(arm: str = "bimanual") -> tuple[bool, list[str]]:
    """
    Validate hardware connections (robot ports and cameras) for specified arm.

    Args:
        arm: Which arm to validate ("left", "right", or "bimanual")

    Returns:
        Tuple of (success, list of issues)
    """
    issues = []
    arm_config = ARM_CONFIGS[arm]

    # Check robot ports - bimanual has different port structure
    if arm == "bimanual":
        # Bimanual mode - check both arm ports
        left_robot_port = Path(arm_config["robot"]["left_arm_port"])
        right_robot_port = Path(arm_config["robot"]["right_arm_port"])
        left_teleop_port = Path(arm_config["teleop"]["left_arm_port"])
        right_teleop_port = Path(arm_config["teleop"]["right_arm_port"])

        if not left_robot_port.exists():
            issues.append(f"Left robot port not found: {arm_config['robot']['left_arm_port']}")
        if not right_robot_port.exists():
            issues.append(f"Right robot port not found: {arm_config['robot']['right_arm_port']}")
        if not left_teleop_port.exists():
            issues.append(f"Left teleop port not found: {arm_config['teleop']['left_arm_port']}")
        if not right_teleop_port.exists():
            issues.append(f"Right teleop port not found: {arm_config['teleop']['right_arm_port']}")
    else:
        # Single arm mode
        robot_port = Path(arm_config["robot"]["port"])
        teleop_port = Path(arm_config["teleop"]["port"])

        if not robot_port.exists():
            issues.append(f"Robot port not found: {arm_config['robot']['port']}")
        if not teleop_port.exists():
            issues.append(f"Teleop port not found: {arm_config['teleop']['port']}")

    # Check camera devices
    for cam_name, cam_cfg in arm_config["cameras"].items():
        video_path = Path(f"/dev/video{cam_cfg['index_or_path']}")
        if not video_path.exists():
            issues.append(f"Camera '{cam_name}' not found: /dev/video{cam_cfg['index_or_path']}")

    return len(issues) == 0, issues


def get_dataset_path(task_type: str, arm: str = "bimanual") -> Path:
    """
    Get the dataset directory path for a task type and arm configuration.

    Dataset organization:
        datasets_bimanuel/
        ├── bimanual/                # Bimanual datasets (both arms)
        │   ├── pick/
        │   ├── handover/
        │   └── ...
        ├── left/                    # Left arm only datasets
        │   ├── pick/
        │   └── ...
        └── right/                   # Right arm only datasets
            ├── pick/
            └── ...

    Args:
        task_type: Task primitive type (pick, place, handover, etc.)
        arm: Which arm configuration ("left", "right", or "bimanual")

    Returns:
        Path to dataset directory
    """
    return DATASETS_BASE / arm / task_type


def check_existing_dataset(dataset_path: Path) -> tuple[bool, int]:
    """
    Check if dataset exists and get episode count.

    Returns:
        Tuple of (exists, episode_count)
    """
    info_path = dataset_path / "meta" / "info.json"
    if info_path.exists():
        with open(info_path) as f:
            info = json.load(f)
            return True, info.get("total_episodes", 0)
    return False, 0


def build_task_string(task_type: str, **kwargs) -> str:
    """
    Build the task description string from template.

    Args:
        task_type: One of the TASK_PRESETS keys
        **kwargs: Values to fill in the template (object, location, target, direction)

    Returns:
        Formatted task string
    """
    preset = TASK_PRESETS[task_type]
    template = preset["template"]

    # Use defaults if not provided
    if task_type == "pick":
        obj = kwargs.get("object", preset.get("default_object", "cube"))
        loc = kwargs.get("location", preset.get("default_location", "table"))
        return template.format(object=obj, location=loc)
    elif task_type == "place":
        obj = kwargs.get("object", preset.get("default_object", "cube"))
        target = kwargs.get("target", preset.get("default_target", "box"))
        return template.format(object=obj, target=target)
    elif task_type == "push":
        obj = kwargs.get("object", preset.get("default_object", "cube"))
        direction = kwargs.get("direction", preset.get("default_direction", "target"))
        return template.format(object=obj, direction=direction)
    elif task_type == "reach":
        loc = kwargs.get("location", preset.get("default_location", "center"))
        return template.format(location=loc)
    elif task_type == "grasp":
        obj = kwargs.get("object", preset.get("default_object", "cube"))
        return template.format(object=obj)
    elif task_type == "release":
        return template
    elif task_type == "pick_and_place":
        obj = kwargs.get("object", preset.get("default_object", "cube"))
        target = kwargs.get("target", preset.get("default_target", "plate"))
        return template.format(object=obj, target=target)
    # Bimanual task types
    elif task_type == "handover":
        obj = kwargs.get("object", preset.get("default_object", "cube"))
        return template.format(object=obj)
    elif task_type == "bimanual_pick_and_place":
        obj = kwargs.get("object", preset.get("default_object", "tissue packet"))
        target = kwargs.get("target", preset.get("default_target", "plate"))
        return template.format(object=obj, target=target)
    elif task_type == "left_arm_pick_and_place":
        obj = kwargs.get("object", preset.get("default_object", "tissue packet"))
        target = kwargs.get("target", preset.get("default_target", "plate"))
        return template.format(object=obj, target=target)
    elif task_type == "right_arm_pick_and_place":
        obj = kwargs.get("object", preset.get("default_object", "tissue packet"))
        target = kwargs.get("target", preset.get("default_target", "plate"))
        return template.format(object=obj, target=target)
    else:
        raise ValueError(f"Unknown task type: {task_type}")


def generate_lerobot_command(
    task_type: str,
    task_string: str,
    num_episodes: int,
    arm: str = "bimanual",
    resume: bool = False
) -> list[str]:
    """
    Generate the lerobot-record command.

    Args:
        task_type: Task type for dataset naming
        task_string: Full task description string
        num_episodes: Number of episodes to record
        arm: Which arm to use ("left", "right", or "bimanual")
        resume: Whether to resume existing dataset

    Returns:
        Command as list of strings
    """
    preset = TASK_PRESETS[task_type]
    dataset_path = get_dataset_path(task_type, arm)
    arm_config = ARM_CONFIGS[arm]

    # Build camera config JSON
    camera_json = json.dumps(arm_config["cameras"])

    # Build repo_id - LeRobot expects format "username/dataset_name"
    # Using "local" as username for local datasets (not pushed to HuggingFace)
    repo_id = f"local/xlerobot_{arm}_{task_type}"

    robot_cfg = arm_config["robot"]
    teleop_cfg = arm_config["teleop"]

    cmd = ["lerobot-record"]

    # Build command differently for bimanual vs single arm
    if arm == "bimanual":
        # Bimanual mode - uses left_arm_port and right_arm_port
        cmd.extend([
            f"--robot.type={robot_cfg['type']}",
            f"--robot.left_arm_port={robot_cfg['left_arm_port']}",
            f"--robot.right_arm_port={robot_cfg['right_arm_port']}",
            f"--robot.id={robot_cfg['id']}",
            f"--robot.left_arm_id={robot_cfg['left_arm_id']}",
            f"--robot.right_arm_id={robot_cfg['right_arm_id']}",
            f"--robot.cameras={camera_json}",
            f"--teleop.type={teleop_cfg['type']}",
            f"--teleop.left_arm_port={teleop_cfg['left_arm_port']}",
            f"--teleop.right_arm_port={teleop_cfg['right_arm_port']}",
            f"--teleop.id={teleop_cfg['id']}",
            f"--teleop.left_arm_id={teleop_cfg['left_arm_id']}",
            f"--teleop.right_arm_id={teleop_cfg['right_arm_id']}",
        ])
    else:
        # Single arm mode - uses single port
        cmd.extend([
            f"--robot.type={robot_cfg['type']}",
            f"--robot.port={robot_cfg['port']}",
            f"--robot.id={robot_cfg['id']}",
            f"--robot.cameras={camera_json}",
            f"--teleop.type={teleop_cfg['type']}",
            f"--teleop.port={teleop_cfg['port']}",
            f"--teleop.id={teleop_cfg['id']}",
        ])

    # Common dataset parameters
    cmd.extend([
        f"--dataset.repo_id={repo_id}",
        f"--dataset.root={dataset_path}",
        f"--dataset.fps={ACTION_FPS}",
        f"--dataset.episode_time_s={preset['episode_time_s']}",
        f"--dataset.reset_time_s={preset['reset_time_s']}",
        f"--dataset.num_episodes={num_episodes}",
        f"--dataset.single_task={task_string}",
        "--dataset.push_to_hub=false",
        "--display_data=false",
    ])

    if resume:
        cmd.append("--resume=true")

    return cmd


def interactive_task_selection() -> tuple[str, dict]:
    """
    Interactive CLI for task selection.

    Returns:
        Tuple of (task_type, kwargs for build_task_string)
    """
    print_header("XLeRobot Data Collection")

    print("\nTASK PRESETS:")
    print("-" * 60)
    print("  [BIMANUAL] = requires both arms, [SINGLE] = one arm only")
    print("-" * 60)

    task_list = list(TASK_PRESETS.keys())
    for i, task_type in enumerate(task_list, 1):
        preset = TASK_PRESETS[task_type]
        bimanual_tag = "[BIMANUAL]" if preset.get("requires_bimanual", False) else "[SINGLE]  "
        print(f"  {i}. {bimanual_tag} {task_type:15} - \"{preset['template']}\"")
        print(f"     {' ' * 28} ({preset['episode_time_s']}s, "
              f"{preset['recommended_episodes']} episodes recommended)")
        print(f"     {' ' * 28} {preset['description']}")
        print()

    # Select task type
    while True:
        try:
            choice = input(f"Select task type [1-{len(task_list)}]: ").strip()
            idx = int(choice) - 1
            if 0 <= idx < len(task_list):
                task_type = task_list[idx]
                break
            print(f"Please enter a number between 1 and {len(task_list)}")
        except ValueError:
            print("Please enter a valid number")

    # Get task-specific parameters
    kwargs = {}
    preset = TASK_PRESETS[task_type]

    print_section(f"Configure '{task_type}' task")

    if task_type in ["pick", "place", "push", "grasp", "pick_and_place", "handover", "bimanual_pick", "bimanual_place"]:
        default_obj = preset.get("default_object", "cube")
        obj = input(f"OBJECT (default: {default_obj}): ").strip()
        if obj:
            kwargs["object"] = obj

    if task_type == "pick":
        default_loc = preset.get("default_location", "table")
        loc = input(f"FROM LOCATION (default: {default_loc}): ").strip()
        if loc:
            kwargs["location"] = loc

    if task_type == "place":
        default_target = preset.get("default_target", "box")
        target = input(f"TARGET (default: {default_target}): ").strip()
        if target:
            kwargs["target"] = target

    if task_type in ["pick_and_place", "bimanual_place"]:
        default_target = preset.get("default_target", "plate")
        target = input(f"PLACE ON TARGET (default: {default_target}): ").strip()
        if target:
            kwargs["target"] = target

    if task_type == "push":
        default_dir = preset.get("default_direction", "target")
        direction = input(f"DIRECTION/TARGET (default: {default_dir}): ").strip()
        if direction:
            kwargs["direction"] = direction

    if task_type == "reach":
        default_loc = preset.get("default_location", "center")
        loc = input(f"LOCATION (default: {default_loc}): ").strip()
        if loc:
            kwargs["location"] = loc

    return task_type, kwargs


def show_recording_tips(task_type: str) -> None:
    """Show tips for recording the specific task type."""
    preset = TASK_PRESETS[task_type]
    episode_time = preset['episode_time_s']

    print_section("Recording Tips")

    print(f"Task: {task_type.upper()}")
    print(f"Duration: {episode_time} seconds per episode (max)")
    print(f"Description: {preset['description']}")
    print()

    print("Phases:")
    for phase in preset.get("phases", []):
        print(f"  - {phase}")
    print()

    # Keyboard controls - IMPORTANT!
    print("=" * 60)
    print("KEYBOARD CONTROLS (use during recording):")
    print("=" * 60)
    print("  RIGHT ARROW → : End episode EARLY (task complete)")
    print("  LEFT ARROW  ← : Re-record episode (made mistake)")
    print("  ESCAPE      ⎋ : Stop all recording")
    print("=" * 60)
    print()

    # TIMING GUIDE - Visual reference for pacing
    print("=" * 60)
    print("TIMING GUIDE - Target pacing for this task:")
    print("=" * 60)
    print()
    print(f"  Total episode time: {episode_time} seconds")
    print(f"  Sampling rate: {ACTION_FPS} Hz (one sample every {1000//ACTION_FPS}ms)")
    print(f"  Total frames: ~{episode_time * ACTION_FPS} frames per episode")
    print()

    # Show phase timing breakdown
    phases = preset.get("phases", [])
    if phases:
        print("  Recommended pace by phase:")
        print("  " + "-" * 50)
        for phase in phases:
            print(f"    {phase}")
        print("  " + "-" * 50)
        print()

    # Visual timing bar
    print("  TIMING REFERENCE (each █ = 5 seconds):")
    print()
    num_blocks = episode_time // 5
    remaining = episode_time % 5

    # Create visual timeline
    timeline = ""
    for i in range(num_blocks):
        timeline += "█"
    if remaining > 0:
        timeline += "▌" if remaining >= 3 else "▎"

    print(f"  0s ", end="")
    for i in range(num_blocks):
        sec = (i + 1) * 5
        print(f"    {sec:2d}s", end="")
    print()
    print(f"  |{timeline}|")
    print(f"  START" + " " * (len(timeline) - 3) + "END")
    print()

    # Speed guidance based on task type
    print("  SPEED GUIDANCE:")
    if task_type == "pick":
        print("    - Approach object: ~10 seconds (slow, deliberate)")
        print("    - Descend to grasp: ~5 seconds")
        print("    - Close gripper: ~3 seconds (pause to grip)")
        print("    - Lift object: ~7 seconds")
        print("    - Hold position: ~3 seconds → then press RIGHT ARROW")
    elif task_type == "place":
        print("    - Move to target: ~10 seconds")
        print("    - Lower object: ~8 seconds (slow descent)")
        print("    - Open gripper: ~4 seconds")
        print("    - Retract arm: ~5 seconds → then press RIGHT ARROW")
    elif task_type == "push":
        print("    - Approach object: ~10 seconds")
        print("    - Make contact: ~5 seconds (gentle)")
        print("    - Push motion: ~15 seconds (slow, steady)")
        print("    - Release: ~5 seconds → then press RIGHT ARROW")
    elif task_type == "reach":
        print("    - Move to target: ~15 seconds (smooth arc)")
        print("    - Hold position: ~3 seconds → then press RIGHT ARROW")
    elif task_type == "grasp":
        print("    - Align with object: ~10 seconds")
        print("    - Close gripper: ~8 seconds (slow close)")
        print("    - Verify grip: ~5 seconds → then press RIGHT ARROW")
    elif task_type == "release":
        print("    - Open gripper: ~8 seconds (slow open)")
        print("    - Hold open: ~5 seconds → then press RIGHT ARROW")
    elif task_type == "pick_and_place":
        print("    - Approach object: ~8 seconds")
        print("    - Descend to grasp: ~4 seconds")
        print("    - Close gripper: ~3 seconds")
        print("    - Lift object: ~5 seconds")
        print("    - Transport to target: ~15 seconds (smooth arc)")
        print("    - Lower to target: ~7 seconds")
        print("    - Open gripper: ~6 seconds")
        print("    - Retract arm: ~7 seconds → then press RIGHT ARROW")
    else:
        print("    - Move at 2-3x slower than natural speed")
        print("    - Press RIGHT ARROW when task completes")

    print()
    print("=" * 60)
    print()

    print("Best Practices:")
    print("  1. Move SLOWLY (2-3x slower than natural speed)")
    print("     - 5Hz sampling = 200ms between frames")
    print("     - Fast motion = missing intermediate positions")
    print()
    print("  2. When task is COMPLETE:")
    print("     → Press RIGHT ARROW immediately!")
    print("     - Don't wait for timeout (creates redundant 'holding' data)")
    print()
    print("  3. If you make a MISTAKE:")
    print("     → Press LEFT ARROW to discard and re-record")
    print()
    print("  4. Recording feels chunky - this is NORMAL")
    print("     - 5Hz control frequency makes follower move in steps")
    print("     - Model will still learn smooth trajectories")
    print()


def run_recording(cmd: list[str], task_type: str, num_episodes: int) -> bool:
    """
    Execute the recording command.

    Returns:
        True if successful, False otherwise
    """
    print_section("Starting Recording")

    print(f"Episodes to record: {num_episodes}")
    print(f"Estimated time: {TASK_PRESETS[task_type]['episode_time_s'] * num_episodes / 60:.1f} minutes")
    print()

    # Show command (for debugging)
    print("Command:")
    print(" \\\n  ".join(cmd[:5]) + " \\")
    print("  [... additional parameters ...]")
    print()

    input("Press ENTER to start recording (Ctrl+C to abort)...")
    print()

    try:
        # Run the lerobot-record command
        result = subprocess.run(cmd, check=True)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"\nRecording failed with error: {e}")
        return False
    except KeyboardInterrupt:
        print("\n\nRecording interrupted by user")
        return False


def post_recording_summary(task_type: str, dataset_path: Path) -> None:
    """Show summary after recording."""
    print_header("Recording Complete")

    exists, episode_count = check_existing_dataset(dataset_path)

    if exists:
        print(f"Dataset: {dataset_path}")
        print(f"Total episodes: {episode_count}")
        print()

        print("Dataset structure:")
        print(f"  {dataset_path}/")
        print("  ├── meta/")
        print("  │   ├── info.json")
        print("  │   ├── tasks.jsonl")
        print("  │   └── stats.json")
        print("  ├── data/chunk-000/")
        print("  └── videos/")
        print()

        print("Next Steps:")
        print("  1. Review recorded videos:")
        videos_path = dataset_path / "videos" / "observation.images.head" / "chunk-000"
        print(f"     ls -lh {videos_path}")
        print()
        print("  2. Train on this dataset:")
        print(f"     # Update train_pi05_mini_mvp.sh with:")
        print(f"     DATASET_PATH=\"{dataset_path}\"")
        print()
        print("  3. Add more episodes (resume):")
        print(f"     python scripts/collect_xlerobot_data.py --task {task_type} --resume")
        print()
    else:
        print("Warning: Dataset metadata not found. Recording may have failed.")


# ==============================================================================
# MAIN ENTRY POINT
# ==============================================================================

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="XLeRobot Data Collection Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                                      # Interactive mode (left arm)
  %(prog)s --task pick --num_episodes 15        # Pick task with left arm
  %(prog)s --arm right --task pick -n 15        # Pick task with right arm
  %(prog)s --task pick --object "red cube" --location "center"
  %(prog)s --task pick --resume
  %(prog)s --config configs/experiments/left_arm_pick_mvp.yaml  # Use YAML config
        """
    )

    parser.add_argument(
        "--config", "-c",
        type=Path,
        help="Path to YAML configuration file (CLI args override config)"
    )
    parser.add_argument(
        "--arm", "-a",
        choices=["left", "right", "bimanual"],
        default=None,  # Changed to None so we can detect if it was set
        help="Which arm configuration to use (default: bimanual)"
    )
    parser.add_argument(
        "--task", "-t",
        choices=list(TASK_PRESETS.keys()),
        help="Task type to record"
    )
    parser.add_argument(
        "--num_episodes", "-n",
        type=int,
        help="Number of episodes to record (default: recommended for task)"
    )
    parser.add_argument(
        "--object", "-o",
        help="Object name (e.g., 'cube', 'red cube')"
    )
    parser.add_argument(
        "--location", "-l",
        help="Location (e.g., 'table', 'center', 'left')"
    )
    parser.add_argument(
        "--target",
        help="Target location for place task (e.g., 'box')"
    )
    parser.add_argument(
        "--direction",
        help="Direction for push task (e.g., 'left', 'forward', 'target')"
    )
    parser.add_argument(
        "--resume", "-r",
        action="store_true",
        help="Resume adding episodes to existing dataset"
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip hardware validation (use with caution)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show command without executing"
    )

    args = parser.parse_args()

    # Load YAML config if provided
    config = load_config(args.config)

    # Merge config with CLI args (CLI takes precedence)
    if config:
        config = merge_config_with_args(config, args)
        print(f"Loaded configuration from: {args.config}")

    # Get arm from config or CLI, default to "bimanual"
    arm = (
        config.get("hardware", {}).get("arm")
        or args.arm
        or "bimanual"
    )
    arm_config = ARM_CONFIGS[arm]

    # Print header
    print_header("XLeRobot Data Collection Script")
    print("Industrial-level data collection for Pi0.5 VLA training")
    print(f"Arm: {arm.upper()}")
    if args.config:
        print(f"Config: {args.config}")
    print("=" * 70)

    # Validate hardware (unless skipped)
    if not args.skip_validation:
        print_section(f"Hardware Validation ({arm} configuration)")
        valid, issues = validate_hardware(arm)

        if valid:
            print("All hardware checks passed!")
            if arm == "bimanual":
                print(f"  Left robot port: {arm_config['robot']['left_arm_port']}")
                print(f"  Right robot port: {arm_config['robot']['right_arm_port']}")
                print(f"  Left teleop port: {arm_config['teleop']['left_arm_port']}")
                print(f"  Right teleop port: {arm_config['teleop']['right_arm_port']}")
            else:
                print(f"  Robot port: {arm_config['robot']['port']}")
                print(f"  Teleop port: {arm_config['teleop']['port']}")
            for cam_name, cam_cfg in arm_config['cameras'].items():
                print(f"  Camera '{cam_name}': /dev/video{cam_cfg['index_or_path']}")
        else:
            print("Hardware validation failed:")
            for issue in issues:
                print(f"  - {issue}")
            print()
            print("Tips:")
            print("  - Check USB connections")
            print("  - Run: ls -l /dev/ttyACM* /dev/video*")
            print("  - Use --skip-validation to bypass (not recommended)")
            if arm == "bimanual":
                print("  - For bimanual, ensure both arms are connected")
            sys.exit(1)

    # Determine task from config, CLI, or interactive mode
    task_config = config.get("data_collection", {}).get("task", {})
    task_type = args.task or task_config.get("type")

    # Validate task/arm compatibility
    if task_type:
        preset = TASK_PRESETS[task_type]
        requires_bimanual = preset.get("requires_bimanual", False)

        if requires_bimanual and arm != "bimanual":
            print(f"\nError: Task '{task_type}' requires bimanual mode!")
            print(f"  Current mode: {arm}")
            print(f"  Solution: Use --arm bimanual or remove --arm flag")
            sys.exit(1)

        if not requires_bimanual and arm == "bimanual":
            print(f"\nWarning: Task '{task_type}' is a single-arm task but bimanual mode selected.")
            print("  The task will use both arms but may collect redundant data.")
            print("  Consider using --arm left or --arm right for single-arm tasks.")
            response = input("Continue anyway? [y/N]: ").strip().lower()
            if response not in ["y", "yes"]:
                print("Aborting. Use --arm left or --arm right for single-arm tasks.")
                sys.exit(0)

    if task_type:
        # Task specified via CLI or config
        kwargs = {}
        # Check CLI args first, then config
        kwargs["object"] = args.object or task_config.get("object")
        kwargs["location"] = args.location or task_config.get("location")
        kwargs["target"] = args.target or task_config.get("target")
        kwargs["direction"] = args.direction or task_config.get("direction")
        # Remove None values
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
    else:
        # Interactive mode
        task_type, kwargs = interactive_task_selection()

    # Build task string
    task_string = build_task_string(task_type, **kwargs)

    # Get episode count (CLI > config > preset default)
    preset = TASK_PRESETS[task_type]
    num_episodes = (
        args.num_episodes
        or task_config.get("num_episodes")
        or preset["recommended_episodes"]
    )

    # Check existing dataset
    dataset_path = get_dataset_path(task_type, arm)
    exists, current_episodes = check_existing_dataset(dataset_path)

    if exists and not args.resume:
        print_section("Existing Dataset Found")
        print(f"Dataset: {dataset_path}")
        print(f"Current episodes: {current_episodes}")
        print()
        response = input("Resume adding episodes? [Y/n]: ").strip().lower()
        if response not in ["", "y", "yes"]:
            print("Aborting. Use a different task type or --resume flag.")
            sys.exit(0)
        args.resume = True

    # Show configuration
    print_section("Configuration")
    print(f"Arm: {arm.upper()}")
    print(f"Task Type: {task_type}")
    print(f"Task String: \"{task_string}\"")
    print(f"Episode Time: {preset['episode_time_s']} seconds")
    print(f"Reset Time: {preset['reset_time_s']} seconds")
    print(f"Episodes: {num_episodes}")
    print(f"Action FPS: {ACTION_FPS} Hz (Pi0.5 requirement)")
    print(f"Camera FPS: {list(arm_config['cameras'].values())[0]['fps']} fps")
    print(f"Dataset: {dataset_path}")
    if args.resume:
        print(f"Mode: RESUME (adding to {current_episodes} existing episodes)")
    else:
        print("Mode: NEW DATASET")

    # Generate command
    cmd = generate_lerobot_command(
        task_type=task_type,
        task_string=task_string,
        num_episodes=num_episodes,
        arm=arm,
        resume=args.resume
    )

    # Dry run - just show command
    if args.dry_run:
        print_section("Generated Command (Dry Run)")
        print(" \\\n  ".join(cmd))
        sys.exit(0)

    # Confirm and record
    print()
    response = input("Proceed with recording? [Y/n]: ").strip().lower()
    if response not in ["", "y", "yes"]:
        print("Aborting.")
        sys.exit(0)

    # Show tips
    show_recording_tips(task_type)

    # Run recording
    success = run_recording(cmd, task_type, num_episodes)

    # Post-recording summary
    if success:
        post_recording_summary(task_type, dataset_path)

        # Save effective configuration to dataset directory for reproducibility
        effective_config = {
            "experiment": {
                "name": f"xlerobot_{arm}_{task_type}",
                "date": datetime.now().isoformat(),
            },
            "hardware": {
                "arm": arm,
                "arm_config": arm_config,
            },
            "data_collection": {
                "task": {
                    "type": task_type,
                    "task_string": task_string,
                    **kwargs,
                },
                "action_fps": ACTION_FPS,
                "episode_time_s": preset["episode_time_s"],
                "reset_time_s": preset["reset_time_s"],
                "num_episodes": num_episodes,
            }
        }
        save_config_to_output(effective_config, dataset_path / "meta", args.config)
    else:
        print("\nRecording did not complete successfully.")
        print("Check the error messages above and try again.")
        sys.exit(1)


if __name__ == "__main__":
    main()
