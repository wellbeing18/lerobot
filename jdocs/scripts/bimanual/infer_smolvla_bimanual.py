#!/usr/bin/env python3
"""
SmolVLA Bimanual Robot Inference for SO-101.

WARNING: SmolVLA has NO NATIVE BIMANUAL SUPPORT. This script is EXPERIMENTAL.

SmolVLA treats actions as flat vectors without arm-specific handling.
For better bimanual results, consider using xVLA with so101_bimanual action mode.

Key differences from single arm:
- Uses BiSO101Follower robot class (two arms)
- 12 DOF state/action (6 per arm) as flat vector
- Motor names: left_shoulder_pan, right_shoulder_pan, etc.
- Action indices 0-5: left arm, 6-11: right arm

Execution Flow:
1. Load hardware config from YAML
2. Initialize BiSO101Follower robot (two arms)
3. Initialize cameras (head + wrist cameras)
4. Load SmolVLAPolicy with checkpoint
5. Main loop:
   a. Capture images from all cameras
   b. Read robot state (12 DOF)
   c. Format observation with task description
   d. Run policy.select_action(observation)
   e. Execute 12D action at 30Hz

Usage:
    # Basic inference
    python infer_smolvla_bimanual.py \\
        --checkpoint outputs/smolvla_bimanual_*/checkpoints/020000/pretrained_model

    # With custom task
    python infer_smolvla_bimanual.py \\
        -c outputs/smolvla_bimanual \\
        --task "pick up the object using both hands" \\
        --duration 60

    # Dry run (no robot commands)
    python infer_smolvla_bimanual.py -c outputs/smolvla_bimanual --dry-run

References:
    - https://huggingface.co/docs/lerobot/smolvla
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
DEFAULT_CHECKPOINT = "outputs/smolvla_bimanual/checkpoints/020000/pretrained_model"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Task description (language prompt for SmolVLA)
# Must match training task strings from tasks.parquet
DEFAULT_TASK_LEFT = "Left arm pick up the tissue packet and place it on the plate"
DEFAULT_TASK_RIGHT = "Right arm pick up the tissue packet and place it on the plate"
DEFAULT_TASK = DEFAULT_TASK_LEFT  # Default to left arm task

# Inference Settings
ACTION_INTERVAL = 0.033   # 30Hz execution rate (1/30 seconds)

# Hardware Config (external file)
HARDWARE_CONFIG = "jdocs/scripts/bimanual/bimanual_so101_hardware.yaml"

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


def load_hardware_config(config_path: str) -> dict:
    """Load hardware configuration from YAML file."""
    full_path = PROJECT_ROOT / config_path
    if not full_path.exists():
        raise FileNotFoundError(f"Hardware config not found: {full_path}")

    with open(full_path) as f:
        config = yaml.safe_load(f)

    logger.info(f"Loaded hardware config from: {config_path}")
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
                left_arm_use_degrees=self.left_use_degrees,
                right_arm_use_degrees=self.right_use_degrees,
            )

            self.robot = BiSO101Follower(robot_config)
            self.robot.connect()

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
        state = np.array([obs[f"{name}.pos"] for name in self.motor_names], dtype=np.float32)
        return state

    def send_action(self, action: np.ndarray):
        """Send action to bimanual robot."""
        if self.robot is None:
            return

        action_dict = {f"{name}.pos": float(action[i]) for i, name in enumerate(self.motor_names)}
        self.robot.send_action(action_dict)

    def disconnect(self):
        """Disconnect from robot."""
        if self.robot is not None:
            self.robot.disconnect()


def format_observation(
    images: dict,
    state: np.ndarray,
    task: str,
    device: str = "cuda",
) -> dict:
    """Format observation for SmolVLA bimanual policy input.

    SmolVLA expects:
        observation.state: (B=1, D=12) float32 tensor (bimanual: 6 per arm)
        observation.images.camera1: (B=1, C=3, H, W) float32 tensor [0, 1]
        observation.images.camera2: (B=1, C=3, H, W) float32 tensor [0, 1]
        observation.images.camera3: (B=1, C=3, H, W) float32 tensor [0, 1] (optional)
        task: str (language description)
    """
    observation = {}

    # State: convert to tensor with batch dimension (12 DOF)
    observation["observation.state"] = torch.from_numpy(state).float().unsqueeze(0).to(device)

    # Images: map camera names to SmolVLA expected names
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

    # Task description for SmolVLA (language conditioning)
    observation["task"] = task

    return observation


def run_inference_loop(
    policy,
    preprocessor,
    postprocessor,
    cameras: CameraManager,
    robot: BimanualRobotController,
    task: str,
    max_duration: float = 60.0,
    action_interval: float = 0.033,
    record_images: bool = False,
    device: str = "cuda",
):
    """Main bimanual inference loop."""
    global running

    logger.info(f"\nStarting BIMANUAL inference loop (max {max_duration}s)...")
    logger.info(f"Task: {task}")
    logger.info(f"Action dim: {TOTAL_DIM} (6 per arm, flat vector)")
    logger.info(f"Action interval: {action_interval*1000:.1f}ms ({1/action_interval:.1f}Hz)")
    logger.info("")
    logger.info("WARNING: SmolVLA has NO native bimanual support!")
    logger.info("         Actions are treated as flat 12D vector.")
    logger.info("")
    logger.info("Press Ctrl+C to stop\n")

    start_time = time.time()
    step_count = 0
    inference_times = []

    if record_images:
        record_dir = PROJECT_ROOT / "jdocs" / "eval_images" / f"smolvla_bimanual_{int(start_time)}"
        record_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Recording images to: {record_dir}")

    joint_names = robot.motor_names
    action_history = []
    state_history = []

    policy.reset()

    while running and (time.time() - start_time) < max_duration:
        loop_start = time.time()

        images = cameras.capture()
        state = robot.get_state()
        state_history.append(state.copy())

        observation = format_observation(images, state, task, device)
        observation = preprocessor(observation)

        inf_start = time.time()
        with torch.inference_mode():
            action = policy.select_action(observation)
        inf_time = time.time() - inf_start
        inference_times.append(inf_time)

        action = postprocessor(action)

        if isinstance(action, torch.Tensor):
            action = action.squeeze(0).cpu().numpy()
        elif isinstance(action, dict):
            action = action.get("action", action)
            if isinstance(action, torch.Tensor):
                action = action.squeeze(0).cpu().numpy()

        # Ensure action is correct dimension
        # SmolVLA may output more dims if max_action_dim > 12
        if len(action) > TOTAL_DIM:
            action = action[:TOTAL_DIM]

        action_history.append(action.copy())

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

        robot.send_action(action)
        step_count += 1

        elapsed = time.time() - loop_start
        sleep_time = action_interval - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)

    # Summary
    total_time = time.time() - start_time
    logger.info(f"\n{'='*50}")
    logger.info(f"BIMANUAL (SmolVLA) Inference loop complete")
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
        logger.info("Action Statistics (bimanual, flat vector):")

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
        description="SmolVLA bimanual robot inference for SO-101 (EXPERIMENTAL)",
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
        default=HARDWARE_CONFIG,
        help=f"Hardware config path (default: {HARDWARE_CONFIG})"
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
    args = parser.parse_args()

    # Determine task based on flags
    if args.task:
        task = args.task
    elif args.right:
        task = DEFAULT_TASK_RIGHT
    else:
        task = DEFAULT_TASK_LEFT  # Default to left arm

    global logger
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = LOG_DIR / f"inference_smolvla_bimanual_{timestamp}.log"
    logger = setup_logging(log_file)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    logger.info("=" * 70)
    logger.info("SmolVLA BIMANUAL Robot Inference for SO-101 (EXPERIMENTAL)")
    logger.info("=" * 70)
    logger.info("")
    logger.info("WARNING: SmolVLA has NO native bimanual support!")
    logger.info("         For better results, consider using xVLA.")
    logger.info("")
    logger.info(f"Checkpoint:      {args.checkpoint}")
    logger.info(f"Task:            {task}")
    logger.info(f"Action Dim:      {TOTAL_DIM} (6 per arm, flat vector)")
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

        logger.info("\nLoading SmolVLA policy...")
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        from lerobot.policies.factory import make_pre_post_processors

        policy = SmolVLAPolicy.from_pretrained(str(checkpoint_path))
        policy.eval()
        policy.to(args.device)
        logger.info(f"  Policy loaded on {args.device}")
        logger.info(f"  Chunk size: {policy.config.chunk_size}")
        logger.info(f"  N action steps: {policy.config.n_action_steps}")
        logger.info(f"  Num denoising steps: {policy.config.num_steps}")

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

        run_inference_loop(
            policy=policy,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            cameras=cameras,
            robot=robot,
            task=task,
            max_duration=args.duration,
            action_interval=ACTION_INTERVAL,
            record_images=args.record,
            device=args.device,
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
