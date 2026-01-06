#!/usr/bin/env python3
"""
GR00T 1.6 Bimanual Robot Inference for SO-101.

This script runs a finetuned GR00T model on the real bimanual SO-101 robot.
It captures images from 3 cameras, reads 12D robot state, and executes
predicted actions at 30Hz.

Bimanual Configuration:
    - State: 12 DOF (6 per arm: 5 joints + 1 gripper)
    - Action: 12 DOF (same as state)
    - Cameras: head, left_wrist, right_wrist

Execution Flow:
1. Load modality config (registers NEW_EMBODIMENT)
2. Load hardware config from YAML
3. Initialize BiSO101Follower robot (two arms)
4. Initialize 3 cameras
5. Load Gr00tPolicy with checkpoint
6. Main loop:
   a. Capture images from all cameras
   b. Read robot state (12 DOF)
   c. Format observation dict
   d. Run policy.get_action(observation)
   e. Execute 12D action chunk at 30Hz

Observation Format (critical for correct inference):
    observation = {
        "video": {
            "head": np.array(...),        # (B=1, T=1, H=480, W=640, C=3) uint8
            "left_wrist": np.array(...),  # (B=1, T=1, H=480, W=640, C=3) uint8
            "right_wrist": np.array(...), # (B=1, T=1, H=480, W=640, C=3) uint8
        },
        "state": {
            "left_arm": np.array(...),      # (B=1, T=1, D=5) float32
            "left_gripper": np.array(...),  # (B=1, T=1, D=1) float32
            "right_arm": np.array(...),     # (B=1, T=1, D=5) float32
            "right_gripper": np.array(...), # (B=1, T=1, D=1) float32
        },
        "language": {
            "annotation.human.action.task_description": [["Use left arm to pick up the orange"]]
        }
    }

Usage:
    # Basic inference
    python infer_groot16_bimanual.py --checkpoint outputs/groot16_bimanual/checkpoint-10000

    # With custom task
    python infer_groot16_bimanual.py -c outputs/groot16_bimanual --task "Use left arm to pick up the corn"

    # Dry run (no robot commands)
    python infer_groot16_bimanual.py -c outputs/groot16_bimanual --dry-run

References:
    - Isaac-GR00T/custom/scripts/ver1_6/infer_groot_so101_1_6.py
    - jdocs/configs/bimanual/hardware.yaml
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
DEFAULT_CHECKPOINT = "outputs/groot16_bimanual/checkpoint-10000"
MODALITY_CONFIG_PATH = "jdocs/scripts/cloud/groot16/so101_bimanual_config.py"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# Task description
DEFAULT_TASK_LEFT = "Use left arm to pick up the orange and place it on the plate"
DEFAULT_TASK_RIGHT = "Use right arm to pick up the orange and place it on the plate"
DEFAULT_TASK = DEFAULT_TASK_LEFT

# Task examples from multitasks dataset
TASK_EXAMPLES = {
    # Plate tasks
    "left_orange_plate": "Use left arm to pick up the orange and place it on the plate",
    "right_orange_plate": "Use right arm to pick up the orange and place it on the plate",
    "left_bread_plate": "Use left arm to pick up the bread and place it on the plate",
    "right_bread_plate": "Use right arm to pick up the bread and place it on the plate",
    "left_corn_plate": "Use left arm to pick up the corn and place it on the plate",
    "right_corn_plate": "Use right arm to pick up the corn and place it on the plate",
    "left_banana_plate": "Use left arm to pick up the banana and place it on the plate",
    "right_banana_plate": "Use right arm to pick up the banana and place it on the plate",
    # Bin tasks
    "left_icecream_bin": "Use left arm to pick up the ice cream and place it in the bin",
    "right_icecream_bin": "Use right arm to pick up the ice cream and place it in the bin",
    "left_ketchup_bin": "Use left arm to pick up the ketchup bottle and place it in the bin",
    "right_ketchup_bin": "Use right arm to pick up the ketchup bottle and place it in the bin",
    "left_yogurt_bin": "Use left arm to pick up the yogurt bottle and place it in the bin",
    "right_yogurt_bin": "Use right arm to pick up the yogurt bottle and place it in the bin",
    "left_tissue_bin": "Use left arm to pick up the used tissue and place it in the bin",
    "right_tissue_bin": "Use right arm to pick up the used tissue and place it in the bin",
    # Generalization test tasks (not in training!)
    "left_icecream_plate": "Use left arm to pick up the ice cream and place it on the plate",
    "right_tissue_plate": "Use right arm to pick up the tissue and place it on the plate",
}

# Inference Settings
ACTION_HORIZON = 8        # Actions to execute before re-inference (reduced from 16)
ACTION_INTERVAL = 0.033   # 30Hz execution rate
NUM_DENOISING_STEPS = 4   # DiT denoising iterations

# Camera names (must match dataset and modality.json)
CAMERA_NAMES = ["head", "left_wrist", "right_wrist"]

# Robot configuration
HARDWARE_CONFIG = "jdocs/configs/bimanual/hardware.yaml"

# State/action dimensions for bimanual SO-101
LEFT_ARM_DIM = 5      # shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll
LEFT_GRIPPER_DIM = 1  # gripper
RIGHT_ARM_DIM = 5     # same as left
RIGHT_GRIPPER_DIM = 1 # gripper
TOTAL_DIM = 12        # 6 per arm * 2 arms
# ============================================================================

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Global flag for clean shutdown
running = True


def signal_handler(sig, frame):
    """Handle Ctrl+C for clean shutdown."""
    global running
    logger.info("\nShutdown requested...")
    running = False


def import_modality_config(config_path: str):
    """Import the modality config module to register NEW_EMBODIMENT."""
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Modality config not found: {config_file}")

    import importlib.util
    spec = importlib.util.spec_from_file_location("so101_bimanual_config", config_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    logger.info("Modality config loaded and registered")


def load_robot(config_path: str, dry_run: bool = False):
    """Load bimanual robot from hardware config."""
    if dry_run:
        logger.info("Dry run mode - no robot connection")
        return None

    from lerobot.robots.bi_so101 import BiSO101Follower

    logger.info(f"Loading robot config from: {config_path}")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    robot = BiSO101Follower(**config["robot"])
    robot.connect()

    logger.info("Robot connected")
    return robot


def load_cameras(config_path: str, dry_run: bool = False):
    """Load cameras from hardware config."""
    if dry_run:
        logger.info("Dry run mode - no camera connection")
        return None

    from lerobot.cameras.opencv import OpenCVCamera

    logger.info(f"Loading camera config from: {config_path}")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    cameras = {}
    for name, cam_config in config.get("cameras", {}).items():
        cameras[name] = OpenCVCamera(**cam_config)
        cameras[name].connect()
        logger.info(f"Camera '{name}' connected")

    return cameras


def capture_images(cameras: dict, dry_run: bool = False) -> dict:
    """Capture images from all cameras."""
    if dry_run or cameras is None:
        # Return dummy images for dry run
        return {name: np.zeros((480, 640, 3), dtype=np.uint8) for name in CAMERA_NAMES}

    images = {}
    for name, camera in cameras.items():
        frame = camera.read()
        if frame is not None:
            # Convert BGR to RGB
            images[name] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            logger.warning(f"Failed to read from camera '{name}'")
            images[name] = np.zeros((480, 640, 3), dtype=np.uint8)

    return images


def read_robot_state(robot, dry_run: bool = False) -> np.ndarray:
    """Read 12D robot state (6 per arm)."""
    if dry_run or robot is None:
        return np.zeros(TOTAL_DIM, dtype=np.float32)

    state = robot.get_state()
    return np.array(state, dtype=np.float32)


def format_observation(
    images: dict,
    state: np.ndarray,
    task: str,
) -> dict:
    """Format observation for GR00T policy input.

    Expected format:
        video: (B=1, T=1, H, W, C) uint8
        state: (B=1, T=1, D) float32
        language: [[str]]
    """
    observation = {
        "video": {},
        "state": {},
        "language": {},
    }

    # Video: add batch and temporal dimensions
    for name, frame in images.items():
        # frame is (H, W, C), we need (B=1, T=1, H, W, C)
        observation["video"][name] = frame[np.newaxis, np.newaxis, :, :, :]

    # State: split into left/right arm and gripper
    # State layout: [left_arm(5), left_gripper(1), right_arm(5), right_gripper(1)]
    left_arm = state[0:5]
    left_gripper = state[5:6]
    right_arm = state[6:11]
    right_gripper = state[11:12]

    # Add batch and temporal dimensions: (B=1, T=1, D)
    observation["state"]["left_arm"] = left_arm[np.newaxis, np.newaxis, :]
    observation["state"]["left_gripper"] = left_gripper[np.newaxis, np.newaxis, :]
    observation["state"]["right_arm"] = right_arm[np.newaxis, np.newaxis, :]
    observation["state"]["right_gripper"] = right_gripper[np.newaxis, np.newaxis, :]

    # Language: nested list format
    observation["language"]["annotation.human.action.task_description"] = [[task]]

    return observation


def execute_action(robot, action: np.ndarray, dry_run: bool = False):
    """Execute 12D action on bimanual robot."""
    if dry_run or robot is None:
        logger.debug(f"Dry run - would execute action: left={action[:6]}, right={action[6:]}")
        return

    robot.send_action(action)


def run_inference_loop(
    policy,
    cameras,
    robot,
    task: str,
    max_duration: float = 60.0,
    action_interval: float = 0.033,
    action_horizon: int = 8,
    dry_run: bool = False,
):
    """Main inference loop."""
    global running

    logger.info(f"\nStarting inference loop (max {max_duration}s)...")
    logger.info(f"Task: {task}")
    logger.info(f"Action interval: {action_interval*1000:.1f}ms ({1/action_interval:.1f}Hz)")
    logger.info(f"Action horizon: {action_horizon}")
    logger.info("Press Ctrl+C to stop\n")

    start_time = time.time()
    step_count = 0
    inference_times = []
    action_buffer = []
    action_idx = 0

    # Joint names for logging
    joint_names = [
        "L_shoulder_pan", "L_shoulder_lift", "L_elbow_flex", "L_wrist_flex", "L_wrist_roll", "L_gripper",
        "R_shoulder_pan", "R_shoulder_lift", "R_elbow_flex", "R_wrist_flex", "R_wrist_roll", "R_gripper"
    ]

    while running and (time.time() - start_time) < max_duration:
        loop_start = time.time()

        # Check if we need new actions
        if len(action_buffer) == 0 or action_idx >= len(action_buffer):
            # Capture images
            images = capture_images(cameras, dry_run)

            # Get robot state
            state = read_robot_state(robot, dry_run)

            # Format observation
            observation = format_observation(images, state, task)

            # Run inference
            inf_start = time.time()
            action_dict, info = policy.get_action(observation)
            inf_time = time.time() - inf_start
            inference_times.append(inf_time)

            # Extract action buffer and concatenate all parts
            # action_dict keys: left_arm, left_gripper, right_arm, right_gripper
            # Each has shape (1, horizon, dim) -> we need (horizon, total_dim)
            left_arm_actions = action_dict["left_arm"][0]          # (horizon, 5)
            left_gripper_actions = action_dict["left_gripper"][0]  # (horizon, 1)
            right_arm_actions = action_dict["right_arm"][0]        # (horizon, 5)
            right_gripper_actions = action_dict["right_gripper"][0] # (horizon, 1)

            # Concatenate to get (horizon, 12)
            full_buffer = np.concatenate([
                left_arm_actions, left_gripper_actions,
                right_arm_actions, right_gripper_actions
            ], axis=1)

            # Slice to action_horizon
            action_buffer = full_buffer[:action_horizon]
            action_idx = 0

            # Log detailed info periodically
            if step_count % 80 == 0:
                logger.info(f"Step {step_count}: inference={inf_time*1000:.1f}ms, buffer_size={len(action_buffer)}")
                # Log current state
                state_str = ", ".join([f"{joint_names[i][:8]}={state[i]:.1f}" for i in range(len(state))])
                logger.info(f"  State: [{state_str}]")
                # Log first action
                action_str = ", ".join([f"{joint_names[i][:8]}={action_buffer[0][i]:.1f}" for i in range(len(action_buffer[0]))])
                logger.info(f"  Action[0]: [{action_str}]")

        # Execute next action
        action = action_buffer[action_idx]
        execute_action(robot, action, dry_run)
        action_idx += 1
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

    logger.info(f"{'='*50}")


def main():
    parser = argparse.ArgumentParser(description="GR00T 1.6 Bimanual Robot Inference")

    parser.add_argument(
        "-c", "--checkpoint",
        type=str,
        default=DEFAULT_CHECKPOINT,
        help=f"Path to GR00T checkpoint (default: {DEFAULT_CHECKPOINT})"
    )
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help="Custom task description"
    )
    parser.add_argument(
        "--task-key",
        type=str,
        choices=list(TASK_EXAMPLES.keys()),
        default=None,
        help="Select task from examples"
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=60.0,
        help="Inference duration in seconds (default: 60)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=HARDWARE_CONFIG,
        help=f"Hardware config path (default: {HARDWARE_CONFIG})"
    )
    parser.add_argument(
        "--modality-config",
        type=str,
        default=MODALITY_CONFIG_PATH,
        help=f"Modality config path (default: {MODALITY_CONFIG_PATH})"
    )
    parser.add_argument(
        "--action-horizon",
        type=int,
        default=ACTION_HORIZON,
        help=f"Actions to execute before re-inference (default: {ACTION_HORIZON})"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run without robot/camera connection"
    )
    parser.add_argument(
        "--list-tasks",
        action="store_true",
        help="List available task examples and exit"
    )

    args = parser.parse_args()

    # List tasks
    if args.list_tasks:
        print("Available task examples:")
        for key, task in TASK_EXAMPLES.items():
            print(f"  {key}: {task}")
        return

    # Determine task
    if args.task:
        task = args.task
    elif args.task_key:
        task = TASK_EXAMPLES[args.task_key]
    else:
        task = DEFAULT_TASK

    # Set up signal handler
    signal.signal(signal.SIGINT, signal_handler)

    logger.info("=" * 70)
    logger.info("GR00T 1.6 Bimanual Robot Inference")
    logger.info("=" * 70)
    logger.info(f"Checkpoint:      {args.checkpoint}")
    logger.info(f"Task:            {task}")
    logger.info(f"Duration:        {args.duration}s")
    logger.info(f"Action Horizon:  {args.action_horizon}")
    logger.info(f"Device:          {DEVICE}")
    logger.info(f"Dry run:         {args.dry_run}")
    logger.info("=" * 70)

    # Validate checkpoint
    if not Path(args.checkpoint).exists():
        if not args.checkpoint.startswith("nvidia/"):
            logger.error(f"Checkpoint not found: {args.checkpoint}")
            sys.exit(1)

    cameras = None
    robot = None

    try:
        # Load modality config
        import_modality_config(args.modality_config)

        # Load policy
        logger.info("\nLoading policy...")
        from gr00t.policy.gr00t_policy import Gr00tPolicy
        from gr00t.data.embodiment_tags import EmbodimentTag

        policy = Gr00tPolicy(
            embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
            model_path=args.checkpoint,
            device=DEVICE,
        )
        logger.info(f"  Policy loaded on {DEVICE}")

        # Initialize hardware
        logger.info("\nInitializing hardware...")
        cameras = load_cameras(args.config, args.dry_run)
        robot = load_robot(args.config, args.dry_run)

        # Run inference
        run_inference_loop(
            policy=policy,
            cameras=cameras,
            robot=robot,
            task=task,
            max_duration=args.duration,
            action_interval=ACTION_INTERVAL,
            action_horizon=args.action_horizon,
            dry_run=args.dry_run,
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
            for cam in cameras.values():
                cam.disconnect()
        if robot is not None:
            robot.disconnect()
        logger.info("Done")


if __name__ == "__main__":
    main()
