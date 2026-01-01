#!/usr/bin/env python3
"""
xVLA Bimanual Robot Inference for SO-101.

This script runs a finetuned xVLA policy on the bimanual SO-101 robot
(two arms) using LeRobot's infrastructure.

Key differences from single arm:
- Uses BiSO101Follower robot class (two arms)
- 12 DOF state/action (6 per arm)
- Motor names: left_shoulder_pan, right_shoulder_pan, etc.
- Action trimming: 20D model output -> 12D robot action
- Optional 3-camera setup (head, left_wrist, right_wrist)

xVLA Bimanual Support:
- BimanualSO101ActionSpace handles 12D real <-> 20D model conversion
- Separate gripper handling for left (index 5) and right (index 11)
- Proper arm coordination

Execution Flow:
1. Load hardware config from YAML
2. Initialize BiSO101Follower robot (two arms)
3. Initialize cameras (head + wrist cameras)
4. Load XVLAPolicy with checkpoint
5. Main loop:
   a. Capture images from all cameras
   b. Read robot state (12 DOF)
   c. Format observation with task and domain_id
   d. Run policy.select_action(observation)
   e. Trim action to 12D and execute at 30Hz

Usage:
    # Basic inference
    python infer_xvla_bimanual.py \\
        --checkpoint outputs/xvla_bimanual_*/checkpoints/020000/pretrained_model

    # With custom task and domain
    python infer_xvla_bimanual.py \\
        -c outputs/xvla_bimanual \\
        --task "pick up the block with both hands" \\
        --domain-id 21 \\
        --duration 60

    # Dry run (no robot commands)
    python infer_xvla_bimanual.py -c outputs/xvla_bimanual --dry-run

References:
    - https://huggingface.co/docs/lerobot/xvla
    - jdocs/bimanual/bin/claude_bimanual_plans.md
"""

import argparse
import logging
import signal
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import yaml

# ============================================================================
# KEY CONFIGURATION
# ============================================================================
# Model
DEFAULT_CHECKPOINT = "outputs/xvla_bimanual/checkpoints/020000/pretrained_model"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Task description (language prompt for xVLA)
# Must match training task strings from tasks.parquet
DEFAULT_TASK_LEFT = "Left arm pick up the tissue packet and place it on the plate"
DEFAULT_TASK_RIGHT = "Right arm pick up the tissue packet and place it on the plate"
DEFAULT_TASK = DEFAULT_TASK_LEFT  # Default to left arm task

# Domain ID (must match training domain_id)
# Use 21 for bimanual (different from single arm's 20)
DEFAULT_DOMAIN_ID = 21

# Inference Settings
ACTION_INTERVAL = 0.033   # 30Hz execution rate (1/30 seconds)

# Diagnostic Settings
DIAGNOSTIC_MODE = True    # Enable detailed pipeline logging
SAFETY_MAX_DELTA = 30.0   # Maximum degrees change per step (safety limit)

# Hardware Config - Use CENTRAL config as single source of truth
# All scripts should read from this path to avoid port mismatch issues
HARDWARE_CONFIG_CENTRAL = "jdocs/configs/hardware/xlerobot_bimanual.yaml"
# Legacy path (for backwards compatibility)
HARDWARE_CONFIG_LEGACY = "jdocs/scripts/bimanual/bimanual_so101_hardware.yaml"

# Dataset (for loading stats) - relative to PROJECT_ROOT
DATASET_PATH = None  # Will be set after PROJECT_ROOT is defined

# Recording
RECORD_IMAGES = False     # Save images to eval_images/
MAX_DURATION = 60.0       # Maximum run duration in seconds

# State/Action dimensions for Bimanual SO-101
# Each arm: 5 joints + 1 gripper = 6 DOF
# Total: 12 DOF (6 per arm)
LEFT_ARM_DIM = 6
RIGHT_ARM_DIM = 6
TOTAL_DIM = LEFT_ARM_DIM + RIGHT_ARM_DIM  # 12

# Safety: Maximum action delta per step (degrees) to prevent dangerous movements
MAX_ACTION_DELTA = 5.0  # Maximum degrees change per action step

# Expected starting positions from training data analysis
# Format: [shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper]
# LEFT ARM TASK: Left arm starts folded, right arm in resting position
LEFT_TASK_START_LEFT_ARM = np.array([0.0, -98.0, 99.5, 51.0, 0.0, 0.5])
LEFT_TASK_START_RIGHT_ARM = np.array([-2.0, -99.0, 99.3, 52.0, 3.3, 0.5])

# RIGHT ARM TASK: Right arm starts folded, left arm in resting position
RIGHT_TASK_START_LEFT_ARM = np.array([-2.0, -99.0, 99.3, 52.0, 3.3, 0.5])
RIGHT_TASK_START_RIGHT_ARM = np.array([0.0, -98.0, 99.5, 51.0, 0.0, 0.5])

# ============================================================================

# Add project src to path
# Script is at: jdocs/scripts/bimanual/ -> parents[2] to reach project root
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Set dataset path (now that PROJECT_ROOT is defined)
# Use merged dataset (same as training) for stats
DATASET_PATH = str(PROJECT_ROOT / "datasets_bimanuel" / "bimanual" / "combined_pick_and_place")

# Log directory
LOG_DIR = PROJECT_ROOT / "jdocs" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)


def setup_logging(log_file: Path = None):
    """Set up logging to both terminal and file."""
    log_format = '%(asctime)s - %(levelname)s - %(message)s'

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter(log_format))
    logger.addHandler(console_handler)

    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter(log_format))
        logger.addHandler(file_handler)
        logger.info(f"Logging to: {log_file}")

    return logger


logger = logging.getLogger(__name__)
running = True


def signal_handler(sig, frame):
    """Handle Ctrl+C for clean shutdown."""
    global running
    logger.info("\nShutdown requested...")
    running = False


def load_hardware_config(config_path: str = None) -> dict:
    """Load hardware configuration from YAML file.

    Priority:
    1. Explicit config_path argument
    2. Central config (jdocs/configs/hardware/xlerobot_bimanual.yaml)
    3. Legacy config (jdocs/scripts/bimanual/bimanual_so101_hardware.yaml)

    This ensures all scripts use the same hardware settings.
    """
    # Try paths in priority order
    paths_to_try = []
    if config_path:
        paths_to_try.append(PROJECT_ROOT / config_path)
    paths_to_try.append(PROJECT_ROOT / HARDWARE_CONFIG_CENTRAL)
    paths_to_try.append(PROJECT_ROOT / HARDWARE_CONFIG_LEGACY)

    full_path = None
    for path in paths_to_try:
        if path.exists():
            full_path = path
            break

    if full_path is None:
        raise FileNotFoundError(
            f"Hardware config not found. Tried:\n"
            f"  - {paths_to_try[0] if config_path else 'N/A'}\n"
            f"  - {PROJECT_ROOT / HARDWARE_CONFIG_CENTRAL}\n"
            f"  - {PROJECT_ROOT / HARDWARE_CONFIG_LEGACY}\n"
            f"Run 'python jdocs/scripts/hardware/scan_hardware.py' to create config."
        )

    with open(full_path) as f:
        config = yaml.safe_load(f)

    logger.info(f"Loaded hardware config from: {full_path}")
    return config


class CameraManager:
    """Manage camera capture for bimanual setup."""

    def __init__(self, hw_config: dict):
        self.cameras = {}
        cam_config = hw_config.get("cameras", {})

        # Initialize head camera
        if "head" in cam_config:
            head_cfg = cam_config["head"]
            self.cameras["head"] = self._init_camera(
                head_cfg.get("index_or_path", 4),
                head_cfg.get("width", 640),
                head_cfg.get("height", 480),
                "head"
            )

        # Initialize left wrist camera
        if "left_wrist" in cam_config:
            left_cfg = cam_config["left_wrist"]
            self.cameras["left_wrist"] = self._init_camera(
                left_cfg.get("index_or_path", 6),
                left_cfg.get("width", 640),
                left_cfg.get("height", 480),
                "left_wrist"
            )

        # Initialize right wrist camera (optional)
        if "right_wrist" in cam_config:
            right_cfg = cam_config["right_wrist"]
            self.cameras["right_wrist"] = self._init_camera(
                right_cfg.get("index_or_path", 8),
                right_cfg.get("width", 640),
                right_cfg.get("height", 480),
                "right_wrist"
            )

    def _init_camera(self, device_index: int, width: int, height: int, name: str) -> cv2.VideoCapture:
        """Initialize a single camera."""
        cap = cv2.VideoCapture(device_index)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open {name} camera at index {device_index}")

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FPS, 30)

        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        logger.info(f"  {name} camera initialized: {actual_w}x{actual_h}")

        return cap

    def capture(self) -> dict:
        """Capture frames from all cameras."""
        frames = {}
        for name, cap in self.cameras.items():
            ret, frame = cap.read()
            if not ret:
                raise RuntimeError(f"Failed to capture from {name} camera")
            frames[name] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frames

    def release(self):
        """Release all cameras."""
        for cap in self.cameras.values():
            cap.release()


class BimanualRobotController:
    """Interface to BiSO101Follower bimanual robot."""

    def __init__(self, hw_config: dict):
        robot_config = hw_config.get("robot", {})
        self.robot_id = robot_config.get("id", "xlerobot_bimanual")

        # Get arm configs
        left_config = robot_config.get("left_arm", {})
        right_config = robot_config.get("right_arm", {})

        self.left_port = left_config.get("port", "/dev/ttyACM0")
        self.right_port = right_config.get("port", "/dev/ttyACM1")
        self.left_use_degrees = left_config.get("use_degrees", True)
        self.right_use_degrees = right_config.get("use_degrees", True)

        # Calibration IDs (required to load calibration files)
        # IDs are nested under left_arm.id and right_arm.id in the config
        self.left_arm_id = left_config.get("id")
        self.right_arm_id = right_config.get("id")

        # Validate required config - fail fast instead of silent None
        if not self.left_arm_id:
            raise ValueError(
                f"Missing 'id' in robot.left_arm config. "
                f"Expected 'robot.left_arm.id' in hardware config, got: {left_config}"
            )
        if not self.right_arm_id:
            raise ValueError(
                f"Missing 'id' in robot.right_arm config. "
                f"Expected 'robot.right_arm.id' in hardware config, got: {right_config}"
            )

        # Motor names for bimanual (order matters for action vector)
        self.motor_names = [
            # Left arm (indices 0-5)
            "left_shoulder_pan", "left_shoulder_lift", "left_elbow_flex",
            "left_wrist_flex", "left_wrist_roll", "left_gripper",
            # Right arm (indices 6-11)
            "right_shoulder_pan", "right_shoulder_lift", "right_elbow_flex",
            "right_wrist_flex", "right_wrist_roll", "right_gripper",
        ]

        self._init_robot()

    def _init_robot(self):
        """Initialize bimanual robot connection."""
        try:
            from lerobot.robots.bi_so101_follower.config_bi_so101_follower import BiSO101FollowerConfig
            from lerobot.robots.bi_so101_follower.bi_so101_follower import BiSO101Follower

            robot_config = BiSO101FollowerConfig(
                id=self.robot_id,
                left_arm_port=self.left_port,
                right_arm_port=self.right_port,
                left_arm_id=self.left_arm_id,
                right_arm_id=self.right_arm_id,
                left_arm_use_degrees=self.left_use_degrees,
                right_arm_use_degrees=self.right_use_degrees,
            )

            self.robot = BiSO101Follower(robot_config)
            self.robot.connect(calibrate=False)  # Use existing calibration files

            logger.info(f"  Bimanual robot connected:")
            logger.info(f"    Left arm: {self.left_port}")
            logger.info(f"    Right arm: {self.right_port}")

        except ImportError as e:
            logger.warning(f"lerobot import failed: {e}")
            logger.warning("Using mock robot")
            self.robot = None

        except Exception as e:
            logger.error(f"Failed to connect to bimanual robot: {e}")
            import traceback
            traceback.print_exc()
            logger.warning("Using mock robot for testing")
            self.robot = None

    def get_state(self) -> np.ndarray:
        """Get current robot state (12 DOF)."""
        if self.robot is None:
            return np.zeros(TOTAL_DIM, dtype=np.float32)

        obs = self.robot.get_observation()

        # Extract motor positions in order (BiSO101Follower uses prefixed names)
        state = np.array([obs[f"{name}.pos"] for name in self.motor_names], dtype=np.float32)
        return state

    def send_action(self, action: np.ndarray):
        """Send action to bimanual robot."""
        if self.robot is None:
            return

        # Convert numpy array to action dict
        action_dict = {f"{name}.pos": float(action[i]) for i, name in enumerate(self.motor_names)}
        self.robot.send_action(action_dict)

    def disconnect(self):
        """Disconnect from robot."""
        if self.robot is not None:
            self.robot.disconnect()


def move_to_start_position(
    robot: BimanualRobotController,
    target_position: np.ndarray,
    duration: float = 3.0,
    rate: float = 30.0,
):
    """Smoothly move robot to target starting position.

    Args:
        robot: BimanualRobotController instance
        target_position: 12D array of target joint positions
        duration: Time to reach target (seconds)
        rate: Control rate (Hz)
    """
    logger.info(f"\nMoving to starting position over {duration}s...")
    logger.info(f"  Target left arm:  [{', '.join([f'{v:.1f}' for v in target_position[:6]])}]")
    logger.info(f"  Target right arm: [{', '.join([f'{v:.1f}' for v in target_position[6:]])}]")

    current = robot.get_state()
    logger.info(f"  Current left arm:  [{', '.join([f'{v:.1f}' for v in current[:6]])}]")
    logger.info(f"  Current right arm: [{', '.join([f'{v:.1f}' for v in current[6:]])}]")

    num_steps = int(duration * rate)
    interval = 1.0 / rate

    for step in range(num_steps + 1):
        alpha = step / num_steps  # Linear interpolation factor
        interpolated = current + alpha * (target_position - current)
        robot.send_action(interpolated)
        time.sleep(interval)

        if step % int(rate) == 0:  # Log every second
            logger.info(f"  Move progress: {step}/{num_steps} ({alpha*100:.0f}%)")

    final_state = robot.get_state()
    logger.info(f"  Final left arm:  [{', '.join([f'{v:.1f}' for v in final_state[:6]])}]")
    logger.info(f"  Final right arm: [{', '.join([f'{v:.1f}' for v in final_state[6:]])}]")
    logger.info("  Move complete!")


def clip_action_delta(action: np.ndarray, current_state: np.ndarray, max_delta: float = MAX_ACTION_DELTA) -> np.ndarray:
    """Clip action to prevent dangerous large movements.

    Args:
        action: Target action (12D)
        current_state: Current robot state (12D)
        max_delta: Maximum allowed change per joint (degrees)

    Returns:
        Clipped action that limits movement to max_delta per joint
    """
    delta = action - current_state
    clipped_delta = np.clip(delta, -max_delta, max_delta)
    return current_state + clipped_delta


def log_diagnostic_info(
    step: int,
    state: np.ndarray,
    action: np.ndarray,
    dataset_stats: dict,
    preprocessor_output: dict = None,
    raw_policy_action: np.ndarray = None,
):
    """Log detailed diagnostic information for debugging inference issues."""
    if not DIAGNOSTIC_MODE:
        return

    state_stats = dataset_stats.get("observation.state", {})
    action_stats = dataset_stats.get("action", {})

    # Only log detailed diagnostics on first step
    if step == 0:
        logger.info("\n" + "=" * 70)
        logger.info("DIAGNOSTIC: FIRST STEP DETAILED ANALYSIS")
        logger.info("=" * 70)

        # Compare current state with training data
        logger.info("\n[State vs Training Data]")
        logger.info(f"{'Joint':<20} {'Current':>10} {'TrainMean':>10} {'TrainMin':>10} {'TrainMax':>10} {'Status':<15}")
        logger.info("-" * 75)

        state_mean = state_stats.get("mean", [0] * 12)
        state_min = state_stats.get("min", [-180] * 12)
        state_max = state_stats.get("max", [180] * 12)

        joint_names = ["L_pan", "L_lift", "L_elbow", "L_wflex", "L_wroll", "L_grip",
                       "R_pan", "R_lift", "R_elbow", "R_wflex", "R_wroll", "R_grip"]

        for i, name in enumerate(joint_names):
            curr = state[i]
            mean = state_mean[i]
            min_v = state_min[i]
            max_v = state_max[i]

            in_range = min_v <= curr <= max_v
            diff_from_mean = curr - mean
            status = "OK" if in_range else "OUT OF RANGE!"
            if abs(diff_from_mean) > 30:
                status = f"DIFF={diff_from_mean:+.0f}°"

            logger.info(f"{name:<20} {curr:>10.1f} {mean:>10.1f} {min_v:>10.1f} {max_v:>10.1f} {status:<15}")

        # Show action output
        logger.info("\n[Action Output vs Training Action Stats]")
        logger.info(f"{'Joint':<20} {'Action':>10} {'ActMean':>10} {'ActMin':>10} {'ActMax':>10} {'Delta':>10}")
        logger.info("-" * 70)

        action_mean = action_stats.get("mean", [0] * 12)
        action_min = action_stats.get("min", [-180] * 12)
        action_max = action_stats.get("max", [180] * 12)

        for i, name in enumerate(joint_names):
            act = action[i]
            mean = action_mean[i]
            min_v = action_min[i]
            max_v = action_max[i]
            delta = act - state[i]

            logger.info(f"{name:<20} {act:>10.1f} {mean:>10.1f} {min_v:>10.1f} {max_v:>10.1f} {delta:>+10.1f}")

        # Show normalized state if available
        if preprocessor_output is not None:
            obs_state = preprocessor_output.get("observation.state")
            if obs_state is not None:
                if isinstance(obs_state, torch.Tensor):
                    norm_state = obs_state.squeeze().cpu().numpy()
                    logger.info("\n[Normalized State (what policy sees)]")
                    logger.info(f"  Left arm:  [{', '.join([f'{v:+.3f}' for v in norm_state[:6]])}]")
                    logger.info(f"  Right arm: [{', '.join([f'{v:+.3f}' for v in norm_state[6:12]])}]")

        if raw_policy_action is not None:
            logger.info("\n[Raw Policy Output (before postprocessor)]")
            logger.info(f"  Left arm:  [{', '.join([f'{v:+.3f}' for v in raw_policy_action[:6]])}]")
            logger.info(f"  Right arm: [{', '.join([f'{v:+.3f}' for v in raw_policy_action[6:12]])}]")

        logger.info("\n" + "=" * 70)

    # Safety check: large action deltas
    action_delta = action - state
    max_delta = np.max(np.abs(action_delta))
    if max_delta > SAFETY_MAX_DELTA:
        logger.warning(f"Step {step}: LARGE ACTION DELTA detected! Max delta: {max_delta:.1f}°")
        logger.warning(f"  State:  [{', '.join([f'{v:.1f}' for v in state])}]")
        logger.warning(f"  Action: [{', '.join([f'{v:.1f}' for v in action])}]")
        logger.warning(f"  Delta:  [{', '.join([f'{v:+.1f}' for v in action_delta])}]")


def format_observation(
    images: dict,
    state: np.ndarray,
    task: str,
    domain_id: int,
    device: str = "cuda",
) -> dict:
    """Format observation for xVLA bimanual policy input.

    xVLA expects:
        observation.state: (B=1, D=12) float32 tensor (bimanual: 6 per arm)
        observation.images.camera1: (B=1, C=3, H, W) float32 tensor [0, 1]
        observation.images.camera2: (B=1, C=3, H, W) float32 tensor [0, 1]
        observation.images.camera3: (B=1, C=3, H, W) float32 tensor [0, 1] (optional)
        task: str (language description)
        domain_id: int (robot configuration identifier)
    """
    observation = {}

    # State: convert to tensor with batch dimension (12 DOF)
    observation["observation.state"] = torch.from_numpy(state).float().unsqueeze(0).to(device)

    # Images: map camera names to xVLA expected names
    key_mapping = {
        "head": "camera1",
        "left_wrist": "camera2",
        "right_wrist": "camera3",
    }

    for name, frame in images.items():
        policy_key_name = key_mapping.get(name, name)
        img_tensor = torch.from_numpy(frame).float() / 255.0
        img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)
        observation[f"observation.images.{policy_key_name}"] = img_tensor.to(device)

    observation["task"] = task
    observation["domain_id"] = domain_id

    return observation


def run_inference_loop(
    policy,
    preprocessor,
    postprocessor,
    cameras: CameraManager,
    robot: BimanualRobotController,
    task: str,
    domain_id: int,
    max_duration: float = 60.0,
    action_interval: float = 0.033,
    record_images: bool = False,
    device: str = "cuda",
    dataset_stats: dict = None,
    freeze_left: bool = False,
    freeze_right: bool = False,
    clip_actions: bool = False,
    max_delta: float = MAX_ACTION_DELTA,
):
    """Main bimanual inference loop."""
    global running

    logger.info(f"\nStarting BIMANUAL inference loop (max {max_duration}s)...")
    logger.info(f"Task: {task}")
    logger.info(f"Domain ID: {domain_id}")
    logger.info(f"Action dim: {TOTAL_DIM} (6 per arm)")
    logger.info(f"Action interval: {action_interval*1000:.1f}ms ({1/action_interval:.1f}Hz)")
    if freeze_left:
        logger.info(">>> LEFT ARM FROZEN - holding current position <<<")
    if freeze_right:
        logger.info(">>> RIGHT ARM FROZEN - holding current position <<<")
    if clip_actions:
        logger.info(f">>> ACTION CLIPPING ENABLED - max delta: {max_delta}° per step <<<")
    logger.info("Press Ctrl+C to stop\n")

    start_time = time.time()
    step_count = 0
    inference_times = []

    if record_images:
        record_dir = PROJECT_ROOT / "jdocs" / "eval_images" / f"bimanual_inference_{int(start_time)}"
        record_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Recording images to: {record_dir}")

    # Joint names for logging (bimanual)
    joint_names = robot.motor_names

    action_history = []
    state_history = []

    policy.reset()

    while running and (time.time() - start_time) < max_duration:
        loop_start = time.time()

        images = cameras.capture()
        state = robot.get_state()
        state_history.append(state.copy())

        observation = format_observation(images, state, task, domain_id, device)
        preprocessed_obs = preprocessor(observation)

        inf_start = time.time()
        with torch.inference_mode():
            raw_action = policy.select_action(preprocessed_obs)
        inf_time = time.time() - inf_start
        inference_times.append(inf_time)

        # Capture raw policy output for diagnostics
        raw_policy_action = None
        if DIAGNOSTIC_MODE and step_count == 0:
            if isinstance(raw_action, torch.Tensor):
                raw_policy_action = raw_action.squeeze(0).cpu().numpy()
            elif isinstance(raw_action, dict) and "action" in raw_action:
                act = raw_action["action"]
                if isinstance(act, torch.Tensor):
                    raw_policy_action = act.squeeze(0).cpu().numpy()

        action = postprocessor(raw_action)

        if isinstance(action, torch.Tensor):
            action = action.squeeze(0).cpu().numpy()
        elif isinstance(action, dict):
            action = action.get("action", action)
            if isinstance(action, torch.Tensor):
                action = action.squeeze(0).cpu().numpy()

        # xVLA with so101_bimanual outputs 20D, trim to 12D for robot
        if len(action) > TOTAL_DIM:
            action = action[:TOTAL_DIM]

        action_history.append(action.copy())

        # Run diagnostics
        if dataset_stats:
            log_diagnostic_info(
                step=step_count,
                state=state,
                action=action,
                dataset_stats=dataset_stats,
                preprocessor_output=preprocessed_obs,
                raw_policy_action=raw_policy_action,
            )

        # Log detailed info every 30 steps (~1 second at 30Hz)
        if step_count % 30 == 0:
            logger.info(f"Step {step_count}: inference={inf_time*1000:.1f}ms")

            # Log left arm state/action
            left_state = state[:LEFT_ARM_DIM]
            left_action = action[:LEFT_ARM_DIM]
            logger.info(f"  Left arm state:  [{', '.join([f'{v:.1f}' for v in left_state])}]")
            logger.info(f"  Left arm action: [{', '.join([f'{v:.1f}' for v in left_action])}]")

            # Log right arm state/action
            right_state = state[LEFT_ARM_DIM:]
            right_action = action[LEFT_ARM_DIM:]
            logger.info(f"  Right arm state:  [{', '.join([f'{v:.1f}' for v in right_state])}]")
            logger.info(f"  Right arm action: [{', '.join([f'{v:.1f}' for v in right_action])}]")

        if record_images:
            for name, frame in images.items():
                img_path = record_dir / f"step_{step_count:04d}_{name}.jpg"
                cv2.imwrite(str(img_path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        # Apply action clipping for safety (before freeze, so we clip model output)
        if clip_actions:
            action = clip_action_delta(action, state, max_delta)

        # Apply arm freezing - replace model action with current state to hold position
        if freeze_left:
            action[:LEFT_ARM_DIM] = state[:LEFT_ARM_DIM]
        if freeze_right:
            action[LEFT_ARM_DIM:] = state[LEFT_ARM_DIM:]

        robot.send_action(action)
        step_count += 1

        elapsed = time.time() - loop_start
        sleep_time = action_interval - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)

    # Summary
    total_time = time.time() - start_time
    logger.info(f"\n{'='*50}")
    logger.info(f"BIMANUAL Inference loop complete")
    logger.info(f"  Total time: {total_time:.1f}s")
    logger.info(f"  Total steps: {step_count}")
    logger.info(f"  Effective rate: {step_count/total_time:.1f}Hz")

    if inference_times:
        logger.info(f"  Avg inference time: {np.mean(inference_times)*1000:.1f}ms")
        logger.info(f"  Max inference time: {np.max(inference_times)*1000:.1f}ms")

    if action_history:
        action_arr = np.array(action_history)
        state_arr = np.array(state_history) if state_history else None

        logger.info(f"\n{'='*50}")
        logger.info("Action Statistics (bimanual):")

        # Left arm statistics
        logger.info("  Left Arm (indices 0-5):")
        for i in range(LEFT_ARM_DIM):
            name = joint_names[i].replace("left_", "")
            act_min, act_max = action_arr[:, i].min(), action_arr[:, i].max()
            act_mean = action_arr[:, i].mean()
            logger.info(f"    {name}: min={act_min:.1f}, max={act_max:.1f}, mean={act_mean:.1f}")

        # Right arm statistics
        logger.info("  Right Arm (indices 6-11):")
        for i in range(LEFT_ARM_DIM, TOTAL_DIM):
            name = joint_names[i].replace("right_", "")
            act_min, act_max = action_arr[:, i].min(), action_arr[:, i].max()
            act_mean = action_arr[:, i].mean()
            logger.info(f"    {name}: min={act_min:.1f}, max={act_max:.1f}, mean={act_mean:.1f}")

        # Gripper analysis (both arms)
        logger.info(f"\nGripper Analysis:")
        left_gripper = action_arr[:, 5]
        right_gripper = action_arr[:, 11]
        logger.info(f"  Left gripper range: [{left_gripper.min():.1f}, {left_gripper.max():.1f}]")
        logger.info(f"  Right gripper range: [{right_gripper.min():.1f}, {right_gripper.max():.1f}]")

    logger.info(f"{'='*50}")


def main():
    parser = argparse.ArgumentParser(
        description="xVLA bimanual robot inference for SO-101",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--checkpoint", "-c",
        type=str,
        default=DEFAULT_CHECKPOINT,
        help=f"Path to checkpoint (default: {DEFAULT_CHECKPOINT})"
    )
    parser.add_argument(
        "--task", "-t",
        type=str,
        default=None,
        help="Task description for language conditioning (overrides --left/--right)"
    )
    parser.add_argument(
        "--left",
        action="store_true",
        help="Use left arm task (default)"
    )
    parser.add_argument(
        "--right",
        action="store_true",
        help="Use right arm task"
    )
    parser.add_argument(
        "--freeze-left",
        action="store_true",
        help="Freeze left arm (hold current position, don't send model actions)"
    )
    parser.add_argument(
        "--freeze-right",
        action="store_true",
        help="Freeze right arm (hold current position, don't send model actions)"
    )
    parser.add_argument(
        "--init-position",
        action="store_true",
        help="Move robot to training starting position before inference (RECOMMENDED)"
    )
    parser.add_argument(
        "--init-duration",
        type=float,
        default=3.0,
        help="Duration (seconds) to move to starting position (default: 3.0)"
    )
    parser.add_argument(
        "--clip-actions",
        action="store_true",
        help="Clip action deltas to prevent dangerous large movements"
    )
    parser.add_argument(
        "--max-delta",
        type=float,
        default=MAX_ACTION_DELTA,
        help=f"Maximum action delta per step in degrees (default: {MAX_ACTION_DELTA})"
    )
    parser.add_argument(
        "--domain-id",
        type=int,
        default=DEFAULT_DOMAIN_ID,
        help=f"Domain ID for bimanual configuration (default: {DEFAULT_DOMAIN_ID})"
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=MAX_DURATION,
        help=f"Max duration in seconds (default: {MAX_DURATION})"
    )
    parser.add_argument(
        "--record",
        action="store_true",
        help="Record images during inference"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run without sending actions to robot (for testing)"
    )
    parser.add_argument(
        "--hw-config",
        type=str,
        default=None,  # None means use central config (see load_hardware_config)
        help=f"Hardware config path (default: central config at {HARDWARE_CONFIG_CENTRAL})"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=DATASET_PATH,
        help=f"Dataset path for loading stats (default: {DATASET_PATH})"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=DEVICE,
        help=f"Device for inference (default: {DEVICE})"
    )
    parser.add_argument(
        "--diagnostic",
        action="store_true",
        default=True,
        help="Enable detailed diagnostic logging (default: True)"
    )
    parser.add_argument(
        "--no-diagnostic",
        action="store_true",
        help="Disable diagnostic logging"
    )
    args = parser.parse_args()

    # Set diagnostic mode
    global DIAGNOSTIC_MODE
    if args.no_diagnostic:
        DIAGNOSTIC_MODE = False
    else:
        DIAGNOSTIC_MODE = args.diagnostic

    # Determine task based on flags
    if args.task:
        task = args.task
    elif args.right:
        task = DEFAULT_TASK_RIGHT
    else:
        task = DEFAULT_TASK_LEFT  # Default to left arm

    global logger
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = LOG_DIR / f"inference_xvla_bimanual_{timestamp}.log"
    logger = setup_logging(log_file)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    logger.info("=" * 70)
    logger.info("xVLA BIMANUAL Robot Inference for SO-101")
    logger.info("=" * 70)
    logger.info(f"Checkpoint:      {args.checkpoint}")
    logger.info(f"Task:            {task}")
    logger.info(f"Domain ID:       {args.domain_id}")
    logger.info(f"Action Dim:      {TOTAL_DIM} (6 per arm)")
    logger.info(f"Duration:        {args.duration}s")
    logger.info(f"Device:          {args.device}")
    logger.info(f"Dry run:         {args.dry_run}")
    logger.info(f"Dataset:         {args.dataset}")
    logger.info(f"Log file:        {log_file}")
    logger.info("=" * 70)

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        if not args.checkpoint.startswith("lerobot/"):
            logger.error(f"Checkpoint not found: {args.checkpoint}")
            sys.exit(1)

    cameras = None
    robot = None

    try:
        hw_config = load_hardware_config(args.hw_config)

        logger.info("\nLoading dataset metadata...")
        from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
        # repo_id should match the dataset directory name
        dataset_name = Path(args.dataset).name
        dataset_metadata = LeRobotDatasetMetadata(
            repo_id=dataset_name,
            root=args.dataset,
        )
        logger.info(f"  Dataset: {args.dataset}")
        logger.info(f"  Episodes: {dataset_metadata.total_episodes}")

        logger.info("\nLoading xVLA policy...")
        from lerobot.policies.xvla.modeling_xvla import XVLAPolicy
        from lerobot.policies.factory import make_pre_post_processors

        policy = XVLAPolicy.from_pretrained(str(checkpoint_path))
        policy.eval()
        policy.to(args.device)
        logger.info(f"  Policy loaded on {args.device}")
        logger.info(f"  Chunk size: {policy.config.chunk_size}")
        logger.info(f"  Action mode: {policy.config.action_mode}")
        logger.info(f"  Num denoising steps: {policy.config.num_denoising_steps}")

        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=policy.config,
            pretrained_path=str(checkpoint_path),
            dataset_stats=dataset_metadata.stats,
            preprocessor_overrides={
                "device_processor": {"device": args.device},
            },
        )
        logger.info("  Preprocessor and postprocessor created")

        logger.info("\nInitializing bimanual hardware...")
        cameras = CameraManager(hw_config)
        robot = BimanualRobotController(hw_config)

        if args.dry_run:
            logger.info("  DRY RUN mode - actions will not be sent to robot")
            robot.robot = None

        # Move to training starting position if requested
        if args.init_position and not args.dry_run:
            # Determine target position based on task
            if args.right:
                target_left = RIGHT_TASK_START_LEFT_ARM
                target_right = RIGHT_TASK_START_RIGHT_ARM
            else:
                target_left = LEFT_TASK_START_LEFT_ARM
                target_right = LEFT_TASK_START_RIGHT_ARM

            target_position = np.concatenate([target_left, target_right])
            move_to_start_position(robot, target_position, duration=args.init_duration)

            # Brief pause to let robot settle
            time.sleep(0.5)

        run_inference_loop(
            policy=policy,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            cameras=cameras,
            robot=robot,
            task=task,
            domain_id=args.domain_id,
            max_duration=args.duration,
            action_interval=ACTION_INTERVAL,
            record_images=args.record,
            device=args.device,
            dataset_stats=dataset_metadata.stats,
            freeze_left=args.freeze_left,
            freeze_right=args.freeze_right,
            clip_actions=args.clip_actions,
            max_delta=args.max_delta,
        )

    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")

    except Exception as e:
        logger.error(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    finally:
        logger.info("\nCleaning up...")
        if cameras is not None:
            cameras.release()
        if robot is not None:
            robot.disconnect()
        logger.info("Done")


if __name__ == "__main__":
    main()
