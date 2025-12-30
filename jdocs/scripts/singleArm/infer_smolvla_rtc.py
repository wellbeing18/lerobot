#!/usr/bin/env python3
"""
SmolVLA Real-Time Chunking (RTC) Inference for SO-101.

This script runs SmolVLA with Real-Time Chunking (RTC) for smoother, more
reactive robot control. RTC eliminates jerky transitions between action chunks
by asynchronously generating new chunks and blending them with ongoing actions.

Key Features:
- RTC: Smooth transitions between action chunks via guided denoising
- Multi-threaded: Action prediction and execution run in parallel
- Latency tracking: Automatically adapts to inference delays
- Action queue: Prevents idle frames during inference

How RTC Works:
1. While robot executes current action chunk, next chunk is computed in background
2. New chunks are "guided" to blend smoothly with already-executed actions
3. Result: No pauses, no sudden jumps, continuous smooth motion

Usage:
    # Basic RTC inference
    python infer_smolvla_rtc.py \\
        --checkpoint outputs/smolvla_pickplace_*/checkpoints/020000/pretrained_model \\
        --task "pick up the block and place it on the plate"

    # With custom RTC parameters
    python infer_smolvla_rtc.py \\
        --checkpoint outputs/smolvla_pickplace_*/checkpoints/020000/pretrained_model \\
        --task "pick up the block" \\
        --execution-horizon 15 \\
        --fps 30 \\
        --duration 120

    # Dry run (no robot commands)
    python infer_smolvla_rtc.py \\
        --checkpoint outputs/smolvla_pickplace_*/checkpoints/020000/pretrained_model \\
        --dry-run

References:
    - https://huggingface.co/docs/lerobot/rtc
    - https://huggingface.co/docs/lerobot/async
    - examples/rtc/eval_with_real_robot.py
"""

import argparse
import logging
import math
import signal
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from threading import Event, Lock, Thread
from typing import Any

import cv2
import numpy as np
import torch
import yaml
from torch import Tensor

# Import LeRobot's built-in RTC classes
from lerobot.policies.rtc.action_queue import ActionQueue
from lerobot.policies.rtc.latency_tracker import LatencyTracker
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.configs.types import RTCAttentionSchedule

# ============================================================================
# KEY CONFIGURATION
# ============================================================================
DEFAULT_CHECKPOINT = "outputs/smolvla_pickplace/checkpoints/020000/pretrained_model"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Task description (language prompt for SmolVLA)
DEFAULT_TASK = "pick up the block and place it on the plate"

# RTC Parameters
DEFAULT_FPS = 30  # Control loop frequency
DEFAULT_EXECUTION_HORIZON = 10  # Steps to blend with previous chunk
DEFAULT_MAX_GUIDANCE_WEIGHT = 10.0  # How strongly to enforce consistency
DEFAULT_ACTION_QUEUE_THRESHOLD = 30  # Request new chunk when queue <= this

# Hardware Config
HARDWARE_CONFIG = "jdocs/scripts/so101_hardware.yaml"

# Timing
MAX_DURATION = 60.0  # Maximum run duration in seconds
# ============================================================================

# Add project src to path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Dataset path for loading stats
DATASET_PATH = str(PROJECT_ROOT / "datasets" / "pick_and_place")

# Log directory
LOG_DIR = PROJECT_ROOT / "jdocs" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)


def setup_logging(log_file: Path = None) -> logging.Logger:
    """Set up logging to both terminal and file."""
    log_format = '%(asctime)s - %(levelname)s - %(name)s - %(message)s'

    logger = logging.getLogger("rtc_inference")
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

    return logger


# Global logger (reconfigured in main)
logger = logging.getLogger("rtc_inference")


def load_hardware_config(config_path: str) -> dict:
    """Load hardware configuration from YAML file."""
    full_path = PROJECT_ROOT / config_path
    if not full_path.exists():
        raise FileNotFoundError(f"Hardware config not found: {full_path}")

    with open(full_path) as f:
        config = yaml.safe_load(f)

    return config


class ThreadSafeRobot:
    """Thread-safe wrapper for robot operations."""

    def __init__(self, hw_config: dict, dry_run: bool = False):
        self.lock = Lock()
        self.dry_run = dry_run
        self.hw_config = hw_config
        self.robot = None
        self.motor_names = [
            "shoulder_pan", "shoulder_lift", "elbow_flex",
            "wrist_flex", "wrist_roll", "gripper"
        ]

        if not dry_run:
            self._init_robot()

    def _init_robot(self):
        """Initialize robot connection."""
        try:
            from lerobot.robots.so101_follower.config_so101_follower import SO101FollowerConfig
            from lerobot.robots.so101_follower.so101_follower import SO101Follower

            robot_config = self.hw_config.get("robot", {})
            config = SO101FollowerConfig(
                port=robot_config.get("port", "/dev/ttyACM1"),
                id=robot_config.get("id", "xlerobot_left_arm"),
            )

            self.robot = SO101Follower(config)
            self.robot.connect()
            logger.info(f"Robot connected on {config.port}")

        except Exception as e:
            logger.error(f"Failed to connect to robot: {e}")
            logger.warning("Running in mock mode")
            self.robot = None

    def get_state(self) -> np.ndarray:
        """Get current robot state (thread-safe)."""
        with self.lock:
            if self.robot is None:
                return np.zeros(6, dtype=np.float32)

            obs = self.robot.get_observation()
            state = np.array(
                [obs[f"{name}.pos"] for name in self.motor_names],
                dtype=np.float32
            )
            return state

    def send_action(self, action: np.ndarray):
        """Send action to robot (thread-safe)."""
        with self.lock:
            if self.robot is None or self.dry_run:
                return

            action_dict = {
                f"{name}.pos": float(action[i])
                for i, name in enumerate(self.motor_names)
            }
            self.robot.send_action(action_dict)

    def disconnect(self):
        """Disconnect from robot."""
        with self.lock:
            if self.robot is not None:
                self.robot.disconnect()


class ThreadSafeCameras:
    """Thread-safe camera manager."""

    def __init__(self, hw_config: dict):
        self.lock = Lock()
        self.cameras = {}

        cam_config = hw_config.get("cameras", {})

        # Initialize cameras
        for name, cfg in [("head", cam_config.get("head", {})),
                          ("left_wrist", cam_config.get("left_wrist", {}))]:
            device_index = cfg.get("index_or_path", 4 if name == "head" else 6)
            width = cfg.get("width", 640)
            height = cfg.get("height", 480)

            cap = cv2.VideoCapture(device_index)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
                cap.set(cv2.CAP_PROP_FPS, 30)
                self.cameras[name] = cap
                logger.info(f"Camera '{name}' initialized at index {device_index}")
            else:
                logger.warning(f"Failed to open camera '{name}' at index {device_index}")

    def capture(self) -> dict:
        """Capture frames from all cameras (thread-safe)."""
        with self.lock:
            frames = {}
            for name, cap in self.cameras.items():
                ret, frame = cap.read()
                if ret:
                    frames[name] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                else:
                    # Return black frame on failure
                    frames[name] = np.zeros((480, 640, 3), dtype=np.uint8)
            return frames

    def release(self):
        """Release all cameras."""
        with self.lock:
            for cap in self.cameras.values():
                cap.release()


def format_observation(
    images: dict,
    state: np.ndarray,
    task: str,
    device: str = "cuda",
) -> dict:
    """Format observation for SmolVLA policy input."""
    observation = {}

    # State
    observation["observation.state"] = torch.from_numpy(state).float().unsqueeze(0).to(device)

    # Images with camera name mapping (head->camera1, left_wrist->camera2)
    key_mapping = {"head": "camera1", "left_wrist": "camera2"}

    for name, frame in images.items():
        policy_key = key_mapping.get(name, name)
        img_tensor = torch.from_numpy(frame).float() / 255.0
        img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)
        observation[f"observation.images.{policy_key}"] = img_tensor.to(device)

    observation["task"] = [task]  # Must be a list for SmolVLA

    return observation


def get_actions_thread(
    policy,
    preprocessor,
    postprocessor,
    cameras: ThreadSafeCameras,
    robot: ThreadSafeRobot,
    action_queue: ActionQueue,
    shutdown_event: Event,
    task: str,
    fps: float,
    action_queue_threshold: int,
    device: str,
    rtc_enabled: bool = True,
):
    """Background thread for action prediction with RTC."""
    try:
        logger.info("[GET_ACTIONS] Starting action prediction thread")

        latency_tracker = LatencyTracker()
        time_per_step = 1.0 / fps

        while not shutdown_event.is_set():
            if action_queue.qsize() <= action_queue_threshold:
                start_time = time.perf_counter()
                action_index_before = action_queue.get_action_index()
                prev_actions = action_queue.get_left_over()

                # Calculate inference delay from latency
                inference_latency = latency_tracker.max() or 0.5  # Default 0.5s if no data
                inference_delay = math.ceil(inference_latency / time_per_step)

                # Capture observation
                images = cameras.capture()
                state = robot.get_state()

                # Format observation
                obs = format_observation(images, state, task, device)

                # Preprocess
                obs = preprocessor(obs)

                # Generate actions
                # NOTE: RTC requires gradients for guided denoising correction
                # When RTC is disabled, we can use inference_mode for efficiency
                if rtc_enabled:
                    # RTC mode: needs gradients for torch.autograd.grad()
                    actions = policy.predict_action_chunk(
                        obs,
                        inference_delay=inference_delay,
                        prev_chunk_left_over=prev_actions,
                    )
                else:
                    # Non-RTC mode: use inference_mode for efficiency
                    with torch.inference_mode():
                        actions = policy.predict_action_chunk(
                            obs,
                            inference_delay=0,  # No delay tracking without RTC
                            prev_chunk_left_over=None,  # No blending without RTC
                        )

                # Store original for RTC
                original_actions = actions.squeeze(0).clone()

                # Postprocess
                postprocessed = postprocessor(actions).squeeze(0)

                # Track latency
                new_latency = time.perf_counter() - start_time
                latency_tracker.add(new_latency)
                new_delay = math.ceil(new_latency / time_per_step)

                # Merge into queue
                action_queue.merge(
                    original_actions, postprocessed,
                    new_delay, action_index_before
                )

                logger.debug(
                    f"[GET_ACTIONS] Generated {len(postprocessed)} actions, "
                    f"latency={new_latency:.3f}s, delay={new_delay}, "
                    f"queue_size={action_queue.qsize()}"
                )
            else:
                time.sleep(0.05)

        logger.info("[GET_ACTIONS] Thread shutting down")

    except Exception as e:
        logger.error(f"[GET_ACTIONS] Fatal error: {e}")
        logger.error(traceback.format_exc())
        shutdown_event.set()


def execute_actions_thread(
    robot: ThreadSafeRobot,
    action_queue: ActionQueue,
    shutdown_event: Event,
    fps: float,
    stats: dict,
):
    """Background thread for action execution."""
    try:
        logger.info("[EXECUTE] Starting action execution thread")

        action_interval = 1.0 / fps
        action_count = 0
        empty_count = 0

        while not shutdown_event.is_set():
            start_time = time.perf_counter()

            action = action_queue.get()

            if action is not None:
                action_np = action.cpu().numpy()
                robot.send_action(action_np)
                action_count += 1
                empty_count = 0
            else:
                empty_count += 1
                if empty_count % 30 == 0:
                    logger.warning(f"[EXECUTE] Action queue empty for {empty_count} steps")

            # Maintain timing
            elapsed = time.perf_counter() - start_time
            sleep_time = action_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

        stats["action_count"] = action_count
        logger.info(f"[EXECUTE] Thread shutting down. Total actions: {action_count}")

    except Exception as e:
        logger.error(f"[EXECUTE] Fatal error: {e}")
        logger.error(traceback.format_exc())
        shutdown_event.set()


def main():
    parser = argparse.ArgumentParser(
        description="SmolVLA RTC inference for SO-101",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Model
    parser.add_argument("--checkpoint", "-c", type=str, default=DEFAULT_CHECKPOINT,
                        help="Path to checkpoint")
    parser.add_argument("--task", "-t", type=str, default=DEFAULT_TASK,
                        help="Task description for language conditioning")

    # RTC parameters
    parser.add_argument("--execution-horizon", type=int, default=DEFAULT_EXECUTION_HORIZON,
                        help="RTC execution horizon (steps to blend)")
    parser.add_argument("--max-guidance-weight", type=float, default=DEFAULT_MAX_GUIDANCE_WEIGHT,
                        help="RTC guidance weight for consistency")
    parser.add_argument("--action-queue-threshold", type=int, default=DEFAULT_ACTION_QUEUE_THRESHOLD,
                        help="Request new chunk when queue size <= this")

    # Timing
    parser.add_argument("--fps", type=float, default=DEFAULT_FPS,
                        help="Control loop frequency (Hz)")
    parser.add_argument("--duration", type=float, default=MAX_DURATION,
                        help="Max run duration (seconds)")

    # Hardware
    parser.add_argument("--hw-config", type=str, default=HARDWARE_CONFIG,
                        help="Hardware config YAML path")
    parser.add_argument("--device", type=str, default=DEVICE,
                        help="Inference device (cuda/cpu)")

    # Options
    parser.add_argument("--dry-run", action="store_true",
                        help="Run without sending robot commands")
    parser.add_argument("--no-rtc", action="store_true",
                        help="Disable RTC (use standard chunking)")

    args = parser.parse_args()

    # Setup logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = LOG_DIR / f"inference_smolvla_rtc_{timestamp}.log"
    global logger
    logger = setup_logging(log_file)

    logger.info("=" * 70)
    logger.info("SmolVLA RTC Inference for SO-101")
    logger.info("=" * 70)
    logger.info(f"Checkpoint:         {args.checkpoint}")
    logger.info(f"Task:               {args.task}")
    logger.info(f"RTC Enabled:        {not args.no_rtc}")
    logger.info(f"Execution Horizon:  {args.execution_horizon}")
    logger.info(f"FPS:                {args.fps}")
    logger.info(f"Duration:           {args.duration}s")
    logger.info(f"Device:             {args.device}")
    logger.info(f"Dry Run:            {args.dry_run}")
    logger.info("=" * 70)

    # Validate checkpoint
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists() and not args.checkpoint.startswith("lerobot/"):
        logger.error(f"Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    shutdown_event = Event()

    def signal_handler(sig, frame):
        logger.info("\nShutdown requested...")
        shutdown_event.set()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    robot = None
    cameras = None
    stats = {"action_count": 0}

    try:
        # Load hardware config
        hw_config = load_hardware_config(args.hw_config)

        # Load policy with RTC
        logger.info("\nLoading SmolVLA policy with RTC...")
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

        # Load policy
        policy = SmolVLAPolicy.from_pretrained(str(checkpoint_path))

        # Configure RTC
        if not args.no_rtc:
            rtc_config = RTCConfig(
                enabled=True,
                execution_horizon=args.execution_horizon,
                max_guidance_weight=args.max_guidance_weight,
                prefix_attention_schedule=RTCAttentionSchedule.EXP,
            )
            policy.config.rtc_config = rtc_config
            policy.init_rtc_processor()
            logger.info(f"  RTC enabled: horizon={args.execution_horizon}, weight={args.max_guidance_weight}")
        else:
            # Create disabled RTC config for ActionQueue
            rtc_config = RTCConfig(enabled=False)
            logger.info("  RTC disabled (standard chunking)")

        policy.eval()
        policy.to(args.device)
        logger.info(f"  Policy loaded on {args.device}")

        # Load dataset metadata for stats
        dataset_metadata = LeRobotDatasetMetadata(
            repo_id="pick_and_place",
            root=DATASET_PATH,
        )

        # Create preprocessor/postprocessor
        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=policy.config,
            pretrained_path=str(checkpoint_path),
            dataset_stats=dataset_metadata.stats,
            preprocessor_overrides={"device_processor": {"device": args.device}},
        )
        logger.info("  Preprocessor/postprocessor created")

        # Initialize hardware
        logger.info("\nInitializing hardware...")
        cameras = ThreadSafeCameras(hw_config)
        robot = ThreadSafeRobot(hw_config, dry_run=args.dry_run)

        if args.dry_run:
            logger.info("  DRY RUN mode - no robot commands")

        # Create action queue (uses LeRobot's built-in RTC ActionQueue)
        action_queue = ActionQueue(rtc_config)

        # Start threads
        logger.info("\nStarting RTC inference threads...")

        get_thread = Thread(
            target=get_actions_thread,
            args=(policy, preprocessor, postprocessor, cameras, robot,
                  action_queue, shutdown_event, args.task, args.fps,
                  args.action_queue_threshold, args.device, not args.no_rtc),
            daemon=True,
            name="GetActions"
        )
        get_thread.start()

        exec_thread = Thread(
            target=execute_actions_thread,
            args=(robot, action_queue, shutdown_event, args.fps, stats),
            daemon=True,
            name="Execute"
        )
        exec_thread.start()

        # Main loop - monitor and log
        logger.info(f"\nRunning for {args.duration} seconds...")
        logger.info("Press Ctrl+C to stop\n")

        start_time = time.time()
        last_log = start_time

        while not shutdown_event.is_set() and (time.time() - start_time) < args.duration:
            time.sleep(1.0)

            # Periodic status log
            now = time.time()
            if now - last_log >= 5.0:
                elapsed = now - start_time
                queue_size = action_queue.qsize()
                logger.info(
                    f"[STATUS] elapsed={elapsed:.1f}s, queue_size={queue_size}, "
                    f"actions_executed={stats.get('action_count', 0)}"
                )
                last_log = now

        # Shutdown
        logger.info("\nShutting down...")
        shutdown_event.set()

        get_thread.join(timeout=3.0)
        exec_thread.join(timeout=3.0)

        # Final summary
        total_time = time.time() - start_time
        logger.info("\n" + "=" * 50)
        logger.info("Inference Complete")
        logger.info(f"  Total time:     {total_time:.1f}s")
        logger.info(f"  Actions executed: {stats.get('action_count', 0)}")
        if stats.get("action_count", 0) > 0:
            logger.info(f"  Effective rate: {stats['action_count']/total_time:.1f} Hz")
        logger.info("=" * 50)

    except Exception as e:
        logger.error(f"\nError: {e}")
        traceback.print_exc()
        shutdown_event.set()
        sys.exit(1)

    finally:
        if cameras:
            cameras.release()
        if robot:
            robot.disconnect()
        logger.info("Cleanup complete")


if __name__ == "__main__":
    main()
