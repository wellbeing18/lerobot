#!/usr/bin/env python3
"""
Pi0.5 Bimanual Robot Inference for SO-101.

Pi0.5 is a 2.5B parameter VLA model with strong language understanding.
When fine-tuned with LoRA, it provides good task generalization.

Key differences from SmolVLA:
- Uses Gemma-based language model (larger, better language understanding)
- Flow matching for action generation (not diffusion)
- Quantile normalization for state/action (not mean-std)
- Longer context for language prompts (200 tokens)

Execution Flow:
1. Load hardware config from YAML
2. Initialize BiSO101Follower robot (two arms)
3. Initialize cameras (head + wrist cameras)
4. Load Pi05Policy with checkpoint
5. Main loop:
   a. Capture images from all cameras
   b. Read robot state (12 DOF)
   c. Format observation with task description
   d. Run policy.select_action(observation)
   e. Execute 12D action at 30Hz

Usage:
    # Basic inference
    python infer_pi05_bimanual.py \\
        --checkpoint outputs/pi05_bimanual_*/checkpoint-120000

    # With custom task
    python infer_pi05_bimanual.py \\
        -c outputs/pi05_bimanual \\
        --task "pick up the tissue and place it on the plate" \\
        --duration 60

    # Dry run (no robot commands)
    python infer_pi05_bimanual.py -c outputs/pi05_bimanual --dry-run

References:
    - https://huggingface.co/docs/lerobot/pi05
    - /home/jrobot/project/refs/openpi/LEROBOT_VS_OPENPI_INVESTIGATION.md
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
DEFAULT_CHECKPOINT = "outputs/pi05_bimanual/checkpoint-120000"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Task description (language prompt for Pi0.5)
# Pi0.5 has strong language understanding, so task variations should work better
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
ACTION_INTERVAL = 0.033   # 30Hz execution rate
CHUNK_SIZE = 50           # Pi0.5 default
N_ACTION_STEPS = 50       # Execute all predicted actions

# Camera names (must match dataset)
CAMERA_NAMES = ["head", "left_wrist", "right_wrist"]

# Robot configuration
HARDWARE_CONFIG = "jdocs/configs/bimanual/hardware.yaml"

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_policy(checkpoint_path: str, device: str = DEVICE):
    """Load Pi05Policy from checkpoint."""
    from lerobot.policies.pi05 import PI05Policy

    logger.info(f"Loading Pi0.5 policy from: {checkpoint_path}")

    # Load policy
    policy = PI05Policy.from_pretrained(checkpoint_path)
    policy.to(device)
    policy.eval()

    logger.info(f"Policy loaded on {device}")
    logger.info(f"  - Chunk size: {policy.config.chunk_size}")
    logger.info(f"  - N action steps: {policy.config.n_action_steps}")
    logger.info(f"  - Max state dim: {policy.config.max_state_dim}")
    logger.info(f"  - Max action dim: {policy.config.max_action_dim}")

    return policy


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
            images[name] = frame
        else:
            logger.warning(f"Failed to read from camera '{name}'")
            images[name] = np.zeros((480, 640, 3), dtype=np.uint8)

    return images


def read_robot_state(robot, dry_run: bool = False) -> np.ndarray:
    """Read 12D robot state (6 per arm)."""
    if dry_run or robot is None:
        return np.zeros(12, dtype=np.float32)

    state = robot.get_state()
    return np.array(state, dtype=np.float32)


def format_observation(images: dict, state: np.ndarray, task: str, device: str) -> dict:
    """Format observation for Pi0.5 policy."""
    obs = {}

    # Images: [1, 3, H, W] format, normalized to [0, 1]
    for name, img in images.items():
        # Convert BGR to RGB if needed
        if img.shape[-1] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Resize to 224x224 (Pi0.5 default)
        img = cv2.resize(img, (224, 224))

        # Normalize to [0, 1] and convert to tensor
        img_tensor = torch.from_numpy(img).float() / 255.0
        img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]

        obs[f"observation.images.{name}"] = img_tensor.to(device)

    # State: [1, 12] format
    state_tensor = torch.from_numpy(state).float().unsqueeze(0)
    obs["observation.state"] = state_tensor.to(device)

    # Task: string
    obs["task"] = task

    return obs


def execute_action(robot, action: np.ndarray, dry_run: bool = False):
    """Execute 12D action on bimanual robot."""
    if dry_run or robot is None:
        logger.debug(f"Dry run - would execute action: {action[:6]} (left), {action[6:]} (right)")
        return

    robot.send_action(action)


def run_inference(
    policy,
    robot,
    cameras,
    task: str,
    duration: float,
    device: str,
    dry_run: bool = False,
):
    """Run inference loop."""
    logger.info(f"Starting inference with task: '{task}'")
    logger.info(f"Duration: {duration}s, Device: {device}")

    start_time = time.time()
    step = 0
    action_buffer = []

    while time.time() - start_time < duration:
        loop_start = time.time()

        # Need new actions?
        if len(action_buffer) == 0:
            # Capture observation
            images = capture_images(cameras, dry_run)
            state = read_robot_state(robot, dry_run)

            # Format for policy
            obs = format_observation(images, state, task, device)

            # Run policy
            with torch.no_grad():
                action_chunk = policy.select_action(obs)

            # Convert to numpy
            if isinstance(action_chunk, torch.Tensor):
                action_chunk = action_chunk.cpu().numpy()

            # action_chunk shape: [chunk_size, action_dim]
            # Take n_action_steps actions
            n_steps = min(N_ACTION_STEPS, len(action_chunk))
            action_buffer = list(action_chunk[:n_steps])

            logger.debug(f"Generated {len(action_buffer)} actions")

        # Execute next action
        action = action_buffer.pop(0)

        # Ensure action is 12D (bimanual)
        if len(action) > 12:
            action = action[:12]  # Trim padding

        execute_action(robot, action, dry_run)

        step += 1
        if step % 30 == 0:  # Log every 30 steps (~1 second)
            elapsed = time.time() - start_time
            logger.info(f"Step {step}, Time: {elapsed:.1f}s/{duration}s")

        # Maintain control rate
        elapsed = time.time() - loop_start
        if elapsed < ACTION_INTERVAL:
            time.sleep(ACTION_INTERVAL - elapsed)

    logger.info(f"Inference complete. Total steps: {step}")


def main():
    parser = argparse.ArgumentParser(description="Pi0.5 Bimanual Robot Inference")

    parser.add_argument(
        "-c", "--checkpoint",
        type=str,
        default=DEFAULT_CHECKPOINT,
        help=f"Path to Pi0.5 checkpoint (default: {DEFAULT_CHECKPOINT})"
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
        default=30.0,
        help="Inference duration in seconds (default: 30)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=HARDWARE_CONFIG,
        help=f"Hardware config path (default: {HARDWARE_CONFIG})"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=DEVICE,
        help=f"Device to use (default: {DEVICE})"
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

    # Signal handler for graceful shutdown
    def signal_handler(sig, frame):
        logger.info("Shutting down...")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    try:
        # Load components
        policy = load_policy(args.checkpoint, args.device)
        robot = load_robot(args.config, args.dry_run)
        cameras = load_cameras(args.config, args.dry_run)

        # Run inference
        run_inference(
            policy=policy,
            robot=robot,
            cameras=cameras,
            task=task,
            duration=args.duration,
            device=args.device,
            dry_run=args.dry_run,
        )

    except Exception as e:
        logger.error(f"Error: {e}")
        raise
    finally:
        # Cleanup
        if robot is not None:
            robot.disconnect()
        if cameras is not None:
            for cam in cameras.values():
                cam.disconnect()


if __name__ == "__main__":
    main()
