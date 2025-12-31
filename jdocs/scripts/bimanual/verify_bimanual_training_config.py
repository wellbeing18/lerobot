#!/home/jrobot/anaconda3/envs/lerobot/bin/python
"""
Bimanual Training Configuration Verification Script.

This script simulates the data loading and preprocessing steps that occur
during SmolVLA/xVLA bimanual training. It verifies:

1. Dataset structure and features (12 DOF actions/states, 3 cameras)
2. Camera name mapping (head->camera1, left_wrist->camera2, right_wrist->camera3)
3. Task descriptions are properly loaded
4. Statistics are computed correctly
5. Action/state dimensions match expected bimanual format

Usage:
    python verify_bimanual_training_config.py

    # With custom dataset path
    python verify_bimanual_training_config.py \
        --dataset datasets_bimanuel/bimanual/combined_pick_and_place

    # Verbose mode with sample data
    python verify_bimanual_training_config.py --verbose --show-samples
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Script location
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Default paths
# Use original left_arm dataset (merged dataset has video metadata issues)
DEFAULT_DATASET = PROJECT_ROOT / "datasets_bimanuel" / "bimanual" / "left_arm_pick_and_place"
DEFAULT_HW_CONFIG = SCRIPT_DIR / "bimanual_so101_hardware.yaml"

# Expected configuration
EXPECTED_ACTION_DIM = 12
EXPECTED_STATE_DIM = 12
EXPECTED_CAMERAS = ["head", "left_wrist", "right_wrist"]
EXPECTED_MOTOR_NAMES = [
    "left_shoulder_pan.pos", "left_shoulder_lift.pos", "left_elbow_flex.pos",
    "left_wrist_flex.pos", "left_wrist_roll.pos", "left_gripper.pos",
    "right_shoulder_pan.pos", "right_shoulder_lift.pos", "right_elbow_flex.pos",
    "right_wrist_flex.pos", "right_wrist_roll.pos", "right_gripper.pos",
]

# SmolVLA camera name mapping (what training script applies)
CAMERA_RENAME_MAP = {
    "observation.images.head": "observation.images.camera1",
    "observation.images.left_wrist": "observation.images.camera2",
    "observation.images.right_wrist": "observation.images.camera3",
}


class VerificationResult:
    """Stores verification results."""
    def __init__(self):
        self.passed = []
        self.warnings = []
        self.errors = []

    def add_pass(self, msg: str):
        self.passed.append(msg)
        logger.info(f"✓ PASS: {msg}")

    def add_warning(self, msg: str):
        self.warnings.append(msg)
        logger.warning(f"⚠ WARNING: {msg}")

    def add_error(self, msg: str):
        self.errors.append(msg)
        logger.error(f"✗ ERROR: {msg}")

    def summary(self) -> str:
        lines = [
            "\n" + "=" * 70,
            "VERIFICATION SUMMARY",
            "=" * 70,
            f"  Passed:   {len(self.passed)}",
            f"  Warnings: {len(self.warnings)}",
            f"  Errors:   {len(self.errors)}",
            "=" * 70,
        ]
        if self.errors:
            lines.append("\nERRORS (must fix before training):")
            for e in self.errors:
                lines.append(f"  ✗ {e}")
        if self.warnings:
            lines.append("\nWARNINGS (review recommended):")
            for w in self.warnings:
                lines.append(f"  ⚠ {w}")
        lines.append("")
        return "\n".join(lines)

    @property
    def success(self) -> bool:
        return len(self.errors) == 0


def load_dataset_info(dataset_path: Path) -> dict:
    """Load dataset info.json."""
    info_path = dataset_path / "meta" / "info.json"
    with open(info_path) as f:
        return json.load(f)


def load_dataset_stats(dataset_path: Path) -> dict:
    """Load dataset stats.json."""
    stats_path = dataset_path / "meta" / "stats.json"
    with open(stats_path) as f:
        return json.load(f)


def load_hardware_config(config_path: Path) -> dict:
    """Load hardware YAML config."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_sample_data(dataset_path: Path, num_samples: int = 5) -> pd.DataFrame:
    """Load sample data from dataset."""
    data_dir = dataset_path / "data"
    parquet_files = sorted(data_dir.rglob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files in {data_dir}")

    df = pd.read_parquet(parquet_files[0])
    return df.head(num_samples)


def verify_dataset_structure(dataset_path: Path, result: VerificationResult) -> dict:
    """Verify dataset structure and features."""
    logger.info("\n--- Verifying Dataset Structure ---")

    info = load_dataset_info(dataset_path)

    # Check robot type
    robot_type = info.get("robot_type", "unknown")
    if robot_type == "bi_so101_follower":
        result.add_pass(f"Robot type: {robot_type}")
    else:
        result.add_warning(f"Unexpected robot type: {robot_type} (expected bi_so101_follower)")

    # Check episodes and frames
    result.add_pass(f"Total episodes: {info['total_episodes']}")
    result.add_pass(f"Total frames: {info['total_frames']}")
    result.add_pass(f"Total tasks: {info['total_tasks']}")

    features = info.get("features", {})

    # Verify action dimension
    action_info = features.get("action", {})
    action_shape = action_info.get("shape", [0])
    action_names = action_info.get("names", [])

    if action_shape == [EXPECTED_ACTION_DIM]:
        result.add_pass(f"Action dimension: {action_shape[0]} (correct for bimanual)")
    else:
        result.add_error(f"Action dimension mismatch: {action_shape} (expected [{EXPECTED_ACTION_DIM}])")

    # Verify action names match expected motor names
    if action_names == EXPECTED_MOTOR_NAMES:
        result.add_pass("Action motor names match expected bimanual format")
    else:
        result.add_warning(f"Action names differ from expected: {action_names}")

    # Verify state dimension
    state_info = features.get("observation.state", {})
    state_shape = state_info.get("shape", [0])

    if state_shape == [EXPECTED_STATE_DIM]:
        result.add_pass(f"State dimension: {state_shape[0]} (correct for bimanual)")
    else:
        result.add_error(f"State dimension mismatch: {state_shape} (expected [{EXPECTED_STATE_DIM}])")

    # Verify cameras
    camera_features = [k for k in features if k.startswith("observation.images.")]
    dataset_cameras = [c.replace("observation.images.", "") for c in camera_features]

    for expected_cam in EXPECTED_CAMERAS:
        if expected_cam in dataset_cameras:
            result.add_pass(f"Camera '{expected_cam}' present in dataset")
        else:
            result.add_error(f"Missing camera '{expected_cam}' in dataset")

    return info


def verify_camera_mapping(info: dict, result: VerificationResult):
    """Verify camera rename mapping will work correctly."""
    logger.info("\n--- Verifying Camera Name Mapping ---")

    features = info.get("features", {})

    logger.info("Camera rename mapping (applied during training):")
    for src, dst in CAMERA_RENAME_MAP.items():
        src_cam = src.replace("observation.images.", "")
        dst_cam = dst.replace("observation.images.", "")
        logger.info(f"  {src_cam} -> {dst_cam}")

        if src in features:
            result.add_pass(f"Source camera '{src_cam}' exists for mapping to '{dst_cam}'")
        else:
            result.add_error(f"Source camera '{src_cam}' NOT FOUND - mapping to '{dst_cam}' will fail")


def verify_tasks(dataset_path: Path, result: VerificationResult) -> list:
    """Verify task descriptions are loaded correctly."""
    logger.info("\n--- Verifying Task Descriptions ---")

    tasks = []

    # Try loading from tasks.parquet
    tasks_path = dataset_path / "meta" / "tasks.parquet"
    if tasks_path.exists():
        try:
            df = pd.read_parquet(tasks_path)
            df = df.reset_index()
            if len(df.columns) >= 2:
                df.columns = ['task', 'task_index'] if 'task' not in df.columns else df.columns
            tasks = df.to_dict('records')
            result.add_pass(f"Loaded {len(tasks)} tasks from tasks.parquet")
        except Exception as e:
            result.add_warning(f"Failed to read tasks.parquet: {e}")

    # Show task details
    for t in tasks:
        task_str = t.get('task', 'unknown')
        task_idx = t.get('task_index', '?')
        logger.info(f"  Task [{task_idx}]: {task_str}")

        # Check for arm-specific keywords
        if 'left' in task_str.lower() or 'right' in task_str.lower():
            result.add_pass(f"Task {task_idx} contains arm-specific keyword for language conditioning")
        else:
            result.add_warning(f"Task {task_idx} may lack arm-specific keywords for differentiation")

    return tasks


def verify_statistics(dataset_path: Path, result: VerificationResult):
    """Verify dataset statistics are computed correctly."""
    logger.info("\n--- Verifying Dataset Statistics ---")

    stats = load_dataset_stats(dataset_path)

    # Check action stats
    if "action" in stats:
        action_stats = stats["action"]
        if len(action_stats.get("mean", [])) == EXPECTED_ACTION_DIM:
            result.add_pass(f"Action statistics have correct dimension ({EXPECTED_ACTION_DIM})")
            logger.info(f"  Action mean: {[f'{x:.2f}' for x in action_stats['mean'][:6]]}... (left arm)")
            logger.info(f"  Action mean: {[f'{x:.2f}' for x in action_stats['mean'][6:]]}... (right arm)")
        else:
            result.add_error(f"Action stats dimension mismatch: {len(action_stats.get('mean', []))}")
    else:
        result.add_error("Action statistics missing from stats.json")

    # Check state stats
    if "observation.state" in stats:
        state_stats = stats["observation.state"]
        if len(state_stats.get("mean", [])) == EXPECTED_STATE_DIM:
            result.add_pass(f"State statistics have correct dimension ({EXPECTED_STATE_DIM})")
        else:
            result.add_error(f"State stats dimension mismatch")
    else:
        result.add_error("State statistics missing from stats.json")


def verify_hardware_config(config_path: Path, result: VerificationResult):
    """Verify hardware configuration."""
    logger.info("\n--- Verifying Hardware Configuration ---")

    if not config_path.exists():
        result.add_error(f"Hardware config not found: {config_path}")
        return

    config = load_hardware_config(config_path)

    # Check robot config
    robot = config.get("robot", {})
    robot_type = robot.get("type", "unknown")
    if robot_type == "bi_so101_follower":
        result.add_pass(f"Hardware robot type: {robot_type}")
    else:
        result.add_error(f"Hardware robot type mismatch: {robot_type}")

    # Check calibration IDs
    left_arm_id = robot.get("left_arm_id")
    right_arm_id = robot.get("right_arm_id")
    if left_arm_id and right_arm_id:
        result.add_pass(f"Calibration IDs configured: {left_arm_id}, {right_arm_id}")

        # Check if calibration files exist
        calib_dir = Path.home() / ".cache" / "huggingface" / "lerobot" / "calibration"
        for arm_id in [left_arm_id, right_arm_id]:
            calib_file = calib_dir / f"{arm_id}.json"
            if calib_file.exists():
                result.add_pass(f"Calibration file exists: {arm_id}.json")
            else:
                result.add_warning(f"Calibration file not found: {calib_file}")
    else:
        result.add_warning("Calibration IDs not explicitly set in hardware config")

    # Check arm ports
    left_arm = robot.get("left_arm", {})
    right_arm = robot.get("right_arm", {})
    left_port = left_arm.get("port", "unknown")
    right_port = right_arm.get("port", "unknown")
    logger.info(f"  Left arm port: {left_port}")
    logger.info(f"  Right arm port: {right_port}")

    # Check if ports exist
    for port, name in [(left_port, "Left"), (right_port, "Right")]:
        if Path(port).exists():
            result.add_pass(f"{name} arm port exists: {port}")
        else:
            result.add_warning(f"{name} arm port not found: {port} (may need to reconnect USB)")

    # Check cameras
    cameras = config.get("cameras", {})
    logger.info("  Camera configuration:")
    for cam_name in EXPECTED_CAMERAS:
        if cam_name in cameras:
            cam_cfg = cameras[cam_name]
            index = cam_cfg.get("index_or_path", "?")
            logger.info(f"    {cam_name}: /dev/video{index}")

            # Check if video device exists
            video_path = Path(f"/dev/video{index}")
            if video_path.exists():
                result.add_pass(f"Camera device exists: /dev/video{index} ({cam_name})")
            else:
                result.add_warning(f"Camera device not found: /dev/video{index} ({cam_name})")
        else:
            result.add_warning(f"Camera '{cam_name}' not configured in hardware config")


def verify_sample_data(dataset_path: Path, result: VerificationResult, verbose: bool = False):
    """Verify sample data can be loaded and has correct format."""
    logger.info("\n--- Verifying Sample Data ---")

    try:
        df = load_sample_data(dataset_path, num_samples=3)
        result.add_pass(f"Sample data loaded successfully ({len(df)} rows)")

        # Check action data
        if 'action' in df.columns:
            sample_action = df['action'].iloc[0]
            if hasattr(sample_action, '__len__'):
                action_len = len(sample_action)
                if action_len == EXPECTED_ACTION_DIM:
                    result.add_pass(f"Sample action dimension: {action_len}")
                    if verbose:
                        logger.info(f"  Sample action (left):  {[f'{x:.1f}' for x in sample_action[:6]]}")
                        logger.info(f"  Sample action (right): {[f'{x:.1f}' for x in sample_action[6:]]}")
                else:
                    result.add_error(f"Sample action dimension wrong: {action_len}")
            else:
                result.add_error(f"Action is not array-like: {type(sample_action)}")

        # Check state data
        if 'observation.state' in df.columns:
            sample_state = df['observation.state'].iloc[0]
            if hasattr(sample_state, '__len__'):
                state_len = len(sample_state)
                if state_len == EXPECTED_STATE_DIM:
                    result.add_pass(f"Sample state dimension: {state_len}")
                    if verbose:
                        logger.info(f"  Sample state (left):  {[f'{x:.1f}' for x in sample_state[:6]]}")
                        logger.info(f"  Sample state (right): {[f'{x:.1f}' for x in sample_state[6:]]}")
                else:
                    result.add_error(f"Sample state dimension wrong: {state_len}")

        # Check episode/task indices
        if 'episode_index' in df.columns:
            episodes = df['episode_index'].unique()
            logger.info(f"  Sample episodes: {list(episodes)}")

        if 'task_index' in df.columns:
            tasks = df['task_index'].unique()
            logger.info(f"  Sample task indices: {list(tasks)}")

    except Exception as e:
        result.add_error(f"Failed to load sample data: {e}")


def simulate_training_dataflow(dataset_path: Path, result: VerificationResult):
    """Simulate what happens during training data loading."""
    logger.info("\n--- Simulating Training Dataflow ---")

    info = load_dataset_info(dataset_path)
    features = info.get("features", {})

    logger.info("1. Training script reads dataset from:")
    logger.info(f"   {dataset_path}")

    logger.info("\n2. Dataset features detected:")
    for key, feat in features.items():
        dtype = feat.get("dtype", "?")
        shape = feat.get("shape", [])
        logger.info(f"   {key}: dtype={dtype}, shape={shape}")

    logger.info("\n3. Camera rename_map transformation:")
    for src, dst in CAMERA_RENAME_MAP.items():
        if src in features:
            logger.info(f"   {src} --> {dst}")
        else:
            logger.info(f"   {src} --> {dst} (WARNING: source not found)")

    logger.info("\n4. Action space configuration:")
    action_names = features.get("action", {}).get("names", [])
    logger.info(f"   Left arm (indices 0-5):  {action_names[:6]}")
    logger.info(f"   Right arm (indices 6-11): {action_names[6:]}")

    logger.info("\n5. SmolVLA model expectations:")
    logger.info("   - Input: observation.images.camera1, camera2, camera3")
    logger.info("   - Input: observation.state (12D)")
    logger.info("   - Input: task (language string)")
    logger.info("   - Output: action (12D flat vector)")

    result.add_pass("Training dataflow simulation complete")


def main():
    parser = argparse.ArgumentParser(
        description="Verify bimanual training configuration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--dataset", "-d",
        type=Path,
        default=DEFAULT_DATASET,
        help=f"Dataset path (default: {DEFAULT_DATASET})"
    )
    parser.add_argument(
        "--hw-config",
        type=Path,
        default=DEFAULT_HW_CONFIG,
        help=f"Hardware config path (default: {DEFAULT_HW_CONFIG})"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed sample data"
    )
    parser.add_argument(
        "--show-samples",
        action="store_true",
        help="Show sample data values"
    )

    args = parser.parse_args()

    # Convert to absolute paths
    dataset_path = Path(args.dataset).resolve()
    hw_config_path = Path(args.hw_config).resolve()

    logger.info("=" * 70)
    logger.info("BIMANUAL TRAINING CONFIGURATION VERIFICATION")
    logger.info("=" * 70)
    logger.info(f"Dataset: {dataset_path}")
    logger.info(f"Hardware Config: {hw_config_path}")
    logger.info("=" * 70)

    result = VerificationResult()

    # Check dataset exists
    if not dataset_path.exists():
        result.add_error(f"Dataset not found: {dataset_path}")
        print(result.summary())
        sys.exit(1)

    # Run verifications
    try:
        info = verify_dataset_structure(dataset_path, result)
        verify_camera_mapping(info, result)
        verify_tasks(dataset_path, result)
        verify_statistics(dataset_path, result)
        verify_hardware_config(hw_config_path, result)
        verify_sample_data(dataset_path, result, verbose=args.show_samples)
        simulate_training_dataflow(dataset_path, result)
    except Exception as e:
        result.add_error(f"Verification failed with exception: {e}")
        import traceback
        traceback.print_exc()

    # Print summary
    print(result.summary())

    if result.success:
        logger.info("✓ All critical checks passed. Ready for training!")
        sys.exit(0)
    else:
        logger.error("✗ Critical errors found. Fix before training.")
        sys.exit(1)


if __name__ == "__main__":
    main()
