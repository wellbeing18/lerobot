#!/usr/bin/env python3
"""
SmolVLA Robot Inference for SO-101.

This script runs a finetuned SmolVLA (Small Vision-Language-Action) policy
on the real SO-101 robot arm using LeRobot's infrastructure.

SmolVLA differs from ACT:
- Uses language/task description for conditioning
- Flow matching for action generation (not VAE)
- Pretrained VLM backbone

Execution Flow:
1. Load hardware config from YAML
2. Initialize robot connection (serial port)
3. Initialize cameras (head + left_wrist)
4. Load SmolVLAPolicy with checkpoint and preprocessors
5. Main loop:
   a. Capture images from both cameras
   b. Read robot state (6 DOF)
   c. Format observation dict with task description
   d. Run policy.select_action(observation)
   e. Execute action at 30Hz
   f. Repeat until duration exceeded or user stops

Usage:
    python infer_smolvla_so101.py --checkpoint outputs/smolvla_pickplace_*/checkpoints/020000/pretrained_model
    python infer_smolvla_so101.py -c outputs/smolvla_pickplace --task "pick up the block" --dry-run

Reference: examples/tutorial/smolvla/using_smolvla_example.py
"""

import argparse
import logging
import signal
import sys
import time
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
DEFAULT_CHECKPOINT = "outputs/smolvla_pickplace/checkpoints/020000/pretrained_model"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Task description (language prompt for SmolVLA)
DEFAULT_TASK = "pick up the block and place it on the plate"

# Inference Settings
ACTION_INTERVAL = 0.033   # 30Hz execution rate (1/30 seconds)

# Hardware Config (external file)
HARDWARE_CONFIG = "jdocs/scripts/so101_hardware.yaml"

# Dataset (for loading stats) - relative to PROJECT_ROOT
DATASET_PATH = None  # Will be set after PROJECT_ROOT is defined

# Recording
RECORD_IMAGES = False     # Save images to eval_images/
MAX_DURATION = 60.0       # Maximum run duration in seconds

# State/Action dimensions for SO-101
ARM_DIM = 5               # shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll
GRIPPER_DIM = 1           # gripper position
# ============================================================================

# Add project src to path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Set dataset path (now that PROJECT_ROOT is defined)
DATASET_PATH = str(PROJECT_ROOT / "datasets" / "pick_and_place")

# Log directory
LOG_DIR = PROJECT_ROOT / "jdocs" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)


def setup_logging(log_file: Path = None):
    """Set up logging to both terminal and file."""
    log_format = '%(asctime)s - %(levelname)s - %(message)s'

    # Create logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # Clear existing handlers
    logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter(log_format))
    logger.addHandler(console_handler)

    # File handler (if log_file provided)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter(log_format))
        logger.addHandler(file_handler)
        logger.info(f"Logging to: {log_file}")

    return logger


# Initialize logger (will be reconfigured in main with file handler)
logger = logging.getLogger(__name__)

# Global flag for clean shutdown
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
    """Manage dual camera capture."""

    def __init__(self, hw_config: dict):
        self.cameras = {}
        cam_config = hw_config.get("cameras", {})

        # Initialize head camera
        head_cfg = cam_config.get("head", {})
        self.cameras["head"] = self._init_camera(
            head_cfg.get("index_or_path", 4),
            head_cfg.get("width", 640),
            head_cfg.get("height", 480),
            "head"
        )

        # Initialize wrist camera
        wrist_cfg = cam_config.get("left_wrist", {})
        self.cameras["left_wrist"] = self._init_camera(
            wrist_cfg.get("index_or_path", 6),
            wrist_cfg.get("width", 640),
            wrist_cfg.get("height", 480),
            "left_wrist"
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

        # Verify resolution
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
            # Convert BGR to RGB
            frames[name] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frames

    def release(self):
        """Release all cameras."""
        for cap in self.cameras.values():
            cap.release()


class RobotController:
    """Interface to SO-101 robot arm via serial."""

    def __init__(self, hw_config: dict):
        robot_config = hw_config.get("robot", {})
        self.port = robot_config.get("port", "/dev/ttyACM1")
        self.use_degrees = robot_config.get("use_degrees", True)
        self.robot_id = robot_config.get("id", "xlerobot_left_arm")

        # Initialize robot connection
        self._init_robot()

    def _init_robot(self):
        """Initialize robot connection."""
        try:
            # LeRobot v3 API
            from lerobot.robots.so101_follower.config_so101_follower import SO101FollowerConfig
            from lerobot.robots.so101_follower.so101_follower import SO101Follower

            # Create config
            robot_config = SO101FollowerConfig(
                port=self.port,
                id=self.robot_id,
            )

            self.robot = SO101Follower(robot_config)
            self.robot.connect()

            logger.info(f"  Robot connected on {self.port} (id={self.robot_id})")

        except ImportError as e:
            logger.warning(f"lerobot import failed: {e}")
            logger.warning("Using mock robot")
            self.robot = None

        except Exception as e:
            logger.error(f"Failed to connect to robot: {e}")
            import traceback
            traceback.print_exc()
            logger.warning("Using mock robot for testing")
            self.robot = None

    def get_state(self) -> np.ndarray:
        """Get current robot state (6 DOF)."""
        if self.robot is None:
            # Mock state for testing
            return np.zeros(ARM_DIM + GRIPPER_DIM, dtype=np.float32)

        # Read state from robot (LeRobot API)
        obs = self.robot.get_observation()

        # Extract motor positions in order
        # LeRobot format: {"shoulder_pan.pos": float, "shoulder_lift.pos": float, ...}
        motor_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
        state = np.array([obs[f"{name}.pos"] for name in motor_names], dtype=np.float32)

        return state

    def send_action(self, action: np.ndarray):
        """Send action to robot."""
        if self.robot is None:
            return  # Skip for mock robot

        # Convert numpy array to LeRobot action dict format
        motor_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
        action_dict = {f"{name}.pos": float(action[i]) for i, name in enumerate(motor_names)}

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
    """Format observation for SmolVLA policy input.

    SmolVLA expects:
        observation.state: (B=1, D=6) float32 tensor
        observation.images.head: (B=1, C=3, H=480, W=640) float32 tensor [0, 1]
        observation.images.left_wrist: (B=1, C=3, H=480, W=640) float32 tensor [0, 1]
        task: str (language description)
    """
    observation = {}

    # State: convert to tensor with batch dimension
    observation["observation.state"] = torch.from_numpy(state).float().unsqueeze(0).to(device)

    # Images: convert to tensor format (B, C, H, W) normalized to [0, 1]
    # NOTE: Training script maps 'head' -> 'camera1' and 'left_wrist' -> 'camera2'
    # We must match this mapping for the policy.
    key_mapping = {
        "head": "camera1",
        "left_wrist": "camera2"
    }

    for name, frame in images.items():
        # Map specific camera names to generic policy inputs if needed
        policy_key_name = key_mapping.get(name, name)
        
        # frame is (H, W, C) uint8 RGB, convert to (B=1, C, H, W) float32
        img_tensor = torch.from_numpy(frame).float() / 255.0
        img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)  # (1, C, H, W)
        observation[f"observation.images.{policy_key_name}"] = img_tensor.to(device)

    # Task description for SmolVLA (language conditioning)
    observation["task"] = task

    return observation


def run_inference_loop(
    policy,
    preprocessor,
    postprocessor,
    cameras: CameraManager,
    robot: RobotController,
    task: str,
    max_duration: float = 60.0,
    action_interval: float = 0.033,
    record_images: bool = False,
    device: str = "cuda",
):
    """Main inference loop."""
    global running

    logger.info(f"\nStarting inference loop (max {max_duration}s)...")
    logger.info(f"Task: {task}")
    logger.info(f"Action interval: {action_interval*1000:.1f}ms ({1/action_interval:.1f}Hz)")
    logger.info("Press Ctrl+C to stop\n")

    start_time = time.time()
    step_count = 0
    inference_times = []

    # Create record directory if needed
    if record_images:
        record_dir = PROJECT_ROOT / "jdocs" / "eval_images" / f"inference_{int(start_time)}"
        record_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Recording images to: {record_dir}")

    # Joint names for logging
    joint_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]

    # Track action history for analysis
    action_history = []
    state_history = []

    # Reset policy's internal action queue
    policy.reset()

    while running and (time.time() - start_time) < max_duration:
        loop_start = time.time()

        # Capture images
        images = cameras.capture()

        # Get robot state
        state = robot.get_state()
        state_history.append(state.copy())

        # Format observation with task description
        observation = format_observation(images, state, task, device)

        # Run preprocessing (normalization, tokenization)
        observation = preprocessor(observation)

        # Run inference
        inf_start = time.time()
        with torch.inference_mode():
            action = policy.select_action(observation)
        inf_time = time.time() - inf_start
        inference_times.append(inf_time)

        # Postprocess (unnormalize) action
        action = postprocessor(action)

        # Convert to numpy
        if isinstance(action, torch.Tensor):
            action = action.squeeze(0).cpu().numpy()  # Remove batch dim
        elif isinstance(action, dict):
            # Handle dict format from postprocessor
            action = action.get("action", action)
            if isinstance(action, torch.Tensor):
                action = action.squeeze(0).cpu().numpy()

        action_history.append(action.copy())

        # Log detailed info every 30 steps (~1 second at 30Hz)
        if step_count % 30 == 0:
            logger.info(f"Step {step_count}: inference={inf_time*1000:.1f}ms")
            # Log current state
            state_str = ", ".join([f"{joint_names[i]}={state[i]:.1f}" for i in range(len(state))])
            logger.info(f"  State: [{state_str}]")
            # Log action
            action_str = ", ".join([f"{joint_names[i]}={action[i]:.1f}" for i in range(len(action))])
            logger.info(f"  Action: [{action_str}]")
            # Log action delta (action - state)
            delta = action - state
            delta_str = ", ".join([f"{joint_names[i]}={delta[i]:+.1f}" for i in range(len(delta))])
            logger.info(f"  Delta: [{delta_str}]")

        # Record images if enabled
        if record_images:
            for name, frame in images.items():
                img_path = record_dir / f"step_{step_count:04d}_{name}.jpg"
                cv2.imwrite(str(img_path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        # Execute action
        robot.send_action(action)
        step_count += 1

        # Maintain action rate
        elapsed = time.time() - loop_start
        sleep_time = action_interval - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)

    # Summary
    total_time = time.time() - start_time
    logger.info(f"\n{'='*50}")
    logger.info(f"Inference loop complete")
    logger.info(f"  Total time: {total_time:.1f}s")
    logger.info(f"  Total steps: {step_count}")
    logger.info(f"  Effective rate: {step_count/total_time:.1f}Hz")

    if inference_times:
        logger.info(f"  Avg inference time: {np.mean(inference_times)*1000:.1f}ms")
        logger.info(f"  Max inference time: {np.max(inference_times)*1000:.1f}ms")

    # Action statistics
    if action_history:
        action_arr = np.array(action_history)
        state_arr = np.array(state_history) if state_history else None

        logger.info(f"\n{'='*50}")
        logger.info("Action Statistics (across all steps):")
        for i, name in enumerate(joint_names):
            act_min, act_max = action_arr[:, i].min(), action_arr[:, i].max()
            act_mean, act_std = action_arr[:, i].mean(), action_arr[:, i].std()
            logger.info(f"  {name}: min={act_min:.1f}, max={act_max:.1f}, mean={act_mean:.1f}, std={act_std:.1f}")

        if state_arr is not None and len(state_arr) > 0:
            logger.info(f"\nState Statistics:")
            for i, name in enumerate(joint_names):
                st_min, st_max = state_arr[:, i].min(), state_arr[:, i].max()
                st_mean, st_std = state_arr[:, i].mean(), state_arr[:, i].std()
                logger.info(f"  {name}: min={st_min:.1f}, max={st_max:.1f}, mean={st_mean:.1f}, std={st_std:.1f}")

            # Gripper specific analysis
            logger.info(f"\nGripper Analysis:")
            gripper_actions = action_arr[:, -1]
            gripper_states = state_arr[:, -1]
            logger.info(f"  Gripper action range: [{gripper_actions.min():.1f}, {gripper_actions.max():.1f}]")
            logger.info(f"  Gripper state range: [{gripper_states.min():.1f}, {gripper_states.max():.1f}]")

    logger.info(f"{'='*50}")


def main():
    parser = argparse.ArgumentParser(
        description="SmolVLA robot inference for SO-101"
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
        default=DEFAULT_TASK,
        help=f"Task description for language conditioning (default: {DEFAULT_TASK})"
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

    # Set up logging with timestamped log file
    global logger
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = LOG_DIR / f"inference_smolvla_{timestamp}.log"
    logger = setup_logging(log_file)

    # Set up signal handler
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    logger.info("=" * 70)
    logger.info("SmolVLA Robot Inference for SO-101")
    logger.info("=" * 70)
    logger.info(f"Checkpoint:      {args.checkpoint}")
    logger.info(f"Task:            {args.task}")
    logger.info(f"Duration:        {args.duration}s")
    logger.info(f"Device:          {args.device}")
    logger.info(f"Dry run:         {args.dry_run}")
    logger.info(f"Dataset:         {args.dataset}")
    logger.info(f"Log file:        {log_file}")
    logger.info("=" * 70)

    # Validate checkpoint
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        # Check if it's a HuggingFace model ID
        if not args.checkpoint.startswith("lerobot/"):
            logger.error(f"Checkpoint not found: {args.checkpoint}")
            sys.exit(1)

    cameras = None
    robot = None

    try:
        # Load hardware config
        hw_config = load_hardware_config(args.hw_config)

        # Load dataset metadata for stats
        logger.info("\nLoading dataset metadata...")
        from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
        dataset_metadata = LeRobotDatasetMetadata(
            repo_id="pick_and_place",
            root=args.dataset,
        )
        logger.info(f"  Dataset: {args.dataset}")
        logger.info(f"  Episodes: {dataset_metadata.total_episodes}")

        # Load policy
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

        # Create preprocessor and postprocessor
        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=policy.config,
            pretrained_path=str(checkpoint_path),
            dataset_stats=dataset_metadata.stats,
            preprocessor_overrides={"device_processor": {"device": args.device}},
        )
        logger.info("  Preprocessor and postprocessor created")

        # Initialize hardware
        logger.info("\nInitializing hardware...")
        cameras = CameraManager(hw_config)
        robot = RobotController(hw_config)

        # Override robot with dry-run mock if requested
        if args.dry_run:
            logger.info("  DRY RUN mode - actions will not be sent to robot")
            robot.robot = None

        # Run inference
        run_inference_loop(
            policy=policy,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            cameras=cameras,
            robot=robot,
            task=args.task,
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
        # Clean up
        logger.info("\nCleaning up...")
        if cameras is not None:
            cameras.release()
        if robot is not None:
            robot.disconnect()
        logger.info("Done")


if __name__ == "__main__":
    main()
